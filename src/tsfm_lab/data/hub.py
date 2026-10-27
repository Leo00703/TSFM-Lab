"""Hugging Face datasets in the fev layout (one parquet per name/frequency, one row per series, list columns).

Files are fetched on demand into the project cache at a pinned revision, so results are reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_lab.data.schema import Dataset, Series, seasonal_period_for

FEV_REPO = "autogluon/fev_datasets"
FEV_REVISION = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"  # pinned: commit verified on 2026-10-27


@dataclass(frozen=True)
class Covariate:
    key: str  # stable id used by the API/UI
    label: str  # plain-language name
    col: str  # parquet column
    transform: str = "float"  # "float" | "nonzero_str" (e.g. Rossmann StateHoliday: '0' = none, 'a'/'b'/'c' = holiday)
    binary: bool = False
    hint: str = ""


def fev_file(name: str, freq: str) -> str:
    return f"{name}/{freq}/train-00000-of-00001.parquet"


def cached_path(name: str, freq: str) -> Path | None:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(FEV_REPO, fev_file(name, freq), repo_type="dataset", revision=FEV_REVISION,
                                    local_files_only=True))
    except Exception:  # noqa: BLE001 - not downloaded yet
        return None


def download(name: str, freq: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(FEV_REPO, fev_file(name, freq), repo_type="dataset", revision=FEV_REVISION))


_FEV_UNITS = {"T": "min", "H": "h", "M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}


def alias(freq: str) -> str:
    """fev frequency names ('1H', '15T', '1M') -> current pandas aliases ('1h', '15min', '1ME'); others pass through."""
    match = re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z]{3})?", freq.strip())
    if not match:
        return freq
    count, unit, anchor = match.groups()
    return f"{count}{_FEV_UNITS.get(unit, unit)}{anchor or ''}"


def _values(cell, transform: str) -> np.ndarray:
    if transform == "nonzero_str":
        return np.array([0.0 if str(v) in ("0", "0.0", "None", "") else 1.0 for v in cell], dtype=np.float64)
    return np.asarray(cell, dtype=np.float64)


def read_fev(name: str, freq: str, target: str, covariates: tuple[Covariate, ...], ids: list[str] | None = None,
             title: str = "", license_text: str = "other (repository card)") -> Dataset:
    import pyarrow.parquet as pq

    path = cached_path(name, freq) or download(name, freq)
    cols = ["id", "timestamp", target, *[c.col for c in covariates]]
    table = pq.read_table(path, columns=cols)
    all_ids = table.column("id").to_pylist()
    wanted = ids if ids is not None else all_ids
    index = {sid: i for i, sid in enumerate(all_ids)}
    series = []
    for sid in wanted:
        i = index[sid]
        ts = np.array(table.column("timestamp")[i].as_py(), dtype="datetime64[ms]")
        tgt = np.asarray(table.column(target)[i].as_py(), dtype=np.float64)
        cov = np.stack([_values(table.column(c.col)[i].as_py(), c.transform) for c in covariates]) if covariates else None
        # make the time axis strictly regular (the model assumes equal spacing); gaps become NaN
        step = pd.tseries.frequencies.to_offset(alias(freq))
        full = pd.date_range(ts[0], ts[-1], freq=step)
        if len(full) != len(ts) or not np.array_equal(full.values.astype("datetime64[ms]"), ts):
            pos = full.get_indexer(pd.DatetimeIndex(ts))
            ok = pos >= 0
            t2 = np.full(len(full), np.nan)
            t2[pos[ok]] = tgt[ok]
            tgt, ts = t2, full.values.astype("datetime64[ms]")
            if cov is not None:
                c2 = np.full((cov.shape[0], len(full)), np.nan)
                c2[:, pos[ok]] = cov[:, ok]
                cov = c2
        series.append(Series(str(sid), tgt, ts, None, cov))
    return Dataset(id=f"hub-{name}-{freq}", name=title or name, series=series, freq=freq,
                   seasonal_period=seasonal_period_for(freq), target_names=[target],
                   known_names=[c.label for c in covariates], source="hub", license=license_text,
                   description=f"{FEV_REPO} / {name} / {freq} @ {FEV_REVISION[:10]}", contamination_risk="possible",
                   tags=["hub", "fev"], extra={"repo": FEV_REPO, "revision": FEV_REVISION})
