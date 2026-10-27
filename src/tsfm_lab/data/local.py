"""Load user tables (CSV / TSV / Parquet / Excel) into a :class:`Dataset`, with a validation report.

The report is meant to be shown to the user *before* running anything: gaps, duplicates, missing
values and inferred frequency decide how trustworthy a backtest on this data can be.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsfm_lab.data.schema import Dataset, Series, seasonal_period_for

READERS = {".csv": pd.read_csv, ".tsv": lambda p: pd.read_csv(p, sep="\t"), ".txt": pd.read_csv,
           ".parquet": pd.read_parquet, ".xlsx": pd.read_excel, ".xls": pd.read_excel}


@dataclass
class ColumnMapping:
    timestamp: str
    targets: list[str]
    id: str | None = None  # column identifying each series; None = a single series
    past_covariates: list[str] = field(default_factory=list)
    known_covariates: list[str] = field(default_factory=list)
    freq: str | None = None  # pandas alias; None = infer


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"unsupported file type '{path.suffix}'. Supported: {', '.join(READERS)}")
    frame = reader(path)
    return frame.head(nrows) if nrows else frame


def guess_mapping(frame: pd.DataFrame) -> ColumnMapping:
    """Heuristic starting point for the mapping UI: first datetime-like column, numeric columns as targets."""
    ts = next((c for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])), None)
    if ts is None:
        for c in frame.columns:
            if frame[c].dtype == object:
                try:
                    pd.to_datetime(frame[c].head(50))
                    ts = c
                    break
                except (ValueError, TypeError):
                    continue
    ts = ts or frame.columns[0]
    ids = next((c for c in frame.columns if c != ts and frame[c].dtype == object), None)
    numeric = [c for c in frame.columns if c not in (ts, ids) and pd.api.types.is_numeric_dtype(frame[c])]
    return ColumnMapping(timestamp=ts, targets=numeric[:1], id=ids)


def _infer_freq(index: pd.DatetimeIndex) -> str | None:
    if len(index) >= 3:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freq = pd.infer_freq(index)
            if freq:
                return freq if isinstance(freq, str) else freq.freqstr
        except ValueError:
            pass
        delta = pd.Series(index).diff().dropna().mode()
        if len(delta):
            return pd.tseries.frequencies.to_offset(delta.iloc[0]).freqstr
    return None


def load_dataframe(frame: pd.DataFrame, mapping: ColumnMapping, name: str = "local",
                   source_path: str | None = None) -> tuple[Dataset, dict[str, Any]]:
    cols = [mapping.timestamp, *mapping.targets, *mapping.past_covariates, *mapping.known_covariates]
    if mapping.id:
        cols.append(mapping.id)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}")
    if not mapping.targets:
        raise ValueError("at least one target column is required")

    frame = frame.copy()
    frame[mapping.timestamp] = pd.to_datetime(frame[mapping.timestamp], errors="coerce")
    bad_ts = int(frame[mapping.timestamp].isna().sum())
    frame = frame.dropna(subset=[mapping.timestamp])
    for c in [*mapping.targets, *mapping.past_covariates, *mapping.known_covariates]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    groups = [(str(k), g) for k, g in frame.groupby(mapping.id, sort=True)] if mapping.id else [("series", frame)]
    report: dict[str, Any] = {"rows": int(len(frame)), "unparseable_timestamps": bad_ts, "series": []}
    freq = mapping.freq
    series: list[Series] = []
    for sid, group in groups:
        group = group.sort_values(mapping.timestamp)
        dup = int(group[mapping.timestamp].duplicated().sum())
        group = group.drop_duplicates(mapping.timestamp, keep="last").set_index(mapping.timestamp)
        f = freq or _infer_freq(group.index)
        freq = freq or f
        gaps = 0
        if f:
            full = pd.date_range(group.index[0], group.index[-1], freq=f)
            gaps = int(len(full.difference(group.index)))
            group = group.reindex(full)
        tgt = group[mapping.targets].to_numpy(dtype=float).T
        past = group[mapping.past_covariates].to_numpy(dtype=float).T if mapping.past_covariates else None
        known = group[mapping.known_covariates].to_numpy(dtype=float).T if mapping.known_covariates else None
        series.append(Series(sid, tgt, group.index.to_numpy(), past, known))
        report["series"].append({
            "id": sid, "length": int(tgt.shape[1]), "start": str(group.index[0]), "end": str(group.index[-1]),
            "freq": f, "duplicate_timestamps": dup, "gaps_filled_with_nan": gaps,
            "target_missing_pct": round(float(np.isnan(tgt).mean() * 100), 3),
            "covariate_missing_pct": None if known is None and past is None else round(float(np.isnan(
                np.concatenate([a for a in (past, known) if a is not None])).mean() * 100), 3),
        })
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame[cols], index=False).to_numpy().tobytes()).hexdigest()[:16]
    report["data_hash"] = digest
    report["freq"] = freq
    report["warnings"] = _warnings(report, series, freq)
    ds = Dataset(
        id=f"local-{digest}", name=name, series=series, freq=freq, seasonal_period=seasonal_period_for(freq),
        target_names=list(mapping.targets), past_names=list(mapping.past_covariates),
        known_names=list(mapping.known_covariates), source="local", license="user data",
        description=f"Loaded from {source_path or 'a table'}", contamination_risk="unknown",
        tags=["local"], extra={"mapping": asdict(mapping), "data_hash": digest})
    return ds, report


def _warnings(report: dict, series: list[Series], freq: str | None) -> list[str]:
    out = []
    if report["unparseable_timestamps"]:
        out.append(f"{report['unparseable_timestamps']} rows had unparseable timestamps and were dropped.")
    if freq is None:
        out.append("Could not infer a regular frequency: seasonal period set to 1 (seasonal baselines degrade to naive).")
    if any(s["duplicate_timestamps"] for s in report["series"]):
        out.append("Duplicate timestamps found; the last value was kept.")
    if any(s["gaps_filled_with_nan"] for s in report["series"]):
        out.append("Missing timestamps were inserted as NaN so the series sit on a regular grid.")
    short = [s.id for s in series if s.length < 64]
    if short:
        out.append(f"{len(short)} series are shorter than 64 steps: backtests will have little context.")
    return out


def load_file(path: str | Path, mapping: ColumnMapping | None = None, name: str | None = None
              ) -> tuple[Dataset, dict[str, Any]]:
    frame = read_table(path)
    mapping = mapping or guess_mapping(frame)
    return load_dataframe(frame, mapping, name or Path(path).stem, str(path))
