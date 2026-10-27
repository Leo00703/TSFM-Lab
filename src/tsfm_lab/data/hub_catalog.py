"""The curated online-dataset catalog (``catalog_fev.json``, built and verified by ``scripts/build_catalog.py``).

Every entry is one fev-bench task over a file of ``autogluon/fev_datasets`` at a pinned revision. Files are downloaded only
when the user asks, checked against the SHA-256 published by Hugging Face, and kept in the project cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsfm_lab.data.hub import FEV_REPO, FEV_REVISION
from tsfm_lab.data.schema import Dataset, Series

CATALOG_FILE = Path(__file__).with_name("catalog_fev.json")
DEFAULT_MAX_SERIES = 200  # a larger file is sampled (seeded, sorted) so memory and time stay bounded
MAX_SERIES_LIMIT = 5000


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def source() -> dict[str, Any]:
    return _catalog()["source"]


def entries() -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in _catalog()["entries"]}


def entry(entry_id: str) -> dict[str, Any]:
    found = entries().get(entry_id)
    if found is None:
        raise KeyError(f"unknown catalog dataset '{entry_id}'")
    return found


# ---------------------------------------------------------------------------- labels
def freq_label(pandas_freq: str) -> str:
    """'15min' -> 'every 15 minutes', 'h' -> 'hourly', 'W-SUN' -> 'weekly', 'QS-JAN' -> 'quarterly'."""
    match = re.fullmatch(r"(\d*)([A-Za-z]+)(?:-[A-Za-z]{3})?", pandas_freq)
    if not match:
        return pandas_freq
    n, unit = int(match.group(1) or 1), match.group(2).lower()
    names = {"min": "minute", "t": "minute", "h": "hour", "d": "day", "w": "week", "m": "month", "me": "month",
             "ms": "month", "q": "quarter", "qe": "quarter", "qs": "quarter", "y": "year", "ye": "year", "ys": "year",
             "a": "year"}
    word = names.get(unit)
    if word is None:
        return pandas_freq
    adjective = {"minute": "every minute", "hour": "hourly", "day": "daily", "week": "weekly", "month": "monthly",
                 "quarter": "quarterly", "year": "yearly"}[word]
    return adjective if n == 1 and word != "minute" else (f"every {n} {word}s" if n > 1 else adjective)


def title(e: dict[str, Any]) -> str:
    """The dataset name without its frequency suffix ('ETT_15T' -> 'ETT'); 'boomlet_1062' keeps its number."""
    base = re.sub(r"_\d*[THDWMQYA]$", "", e["config"]).replace("_", " ")
    return f"{base} · {e['variant'].replace('_', ' ')}" if e.get("variant") else base


# ---------------------------------------------------------------------------- files
def _cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / f"datasets--{FEV_REPO.replace('/', '--')}" / "snapshots" / FEV_REVISION


def file_path(e: dict[str, Any]) -> Path | None:
    path = _cache_dir() / e["path"]
    return path if path.is_file() else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(e: dict[str, Any]) -> Path:
    """Fetch the file at the pinned revision and refuse it if its hash differs from the one Hugging Face publishes."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(FEV_REPO, e["path"], repo_type="dataset", revision=FEV_REVISION))
    if sha256(path) != e["sha256"]:
        remove(e)
        raise ValueError(f"{e['id']}: the downloaded file does not match the published SHA-256; it was discarded")
    return path


def remove(e: dict[str, Any]) -> bool:
    path = file_path(e)
    if path is None:
        return False
    target = path.resolve()
    path.unlink()
    if target != path and target.is_file():
        target.unlink()  # the blob behind the snapshot link (Linux/macOS); on Windows the file is the blob itself
    return True


def listing() -> list[dict[str, Any]]:
    """Everything the catalog page needs, with the download status of each file."""
    out = []
    for e in _catalog()["entries"]:
        known_text = [c for c in e["known"] if e["columns"].get(c, {}).get("kind") == "list-text"]
        out.append({
            "id": e["id"], "title": title(e), "family": e["family"], "freq": e["pandas_freq"],
            "freq_label": freq_label(e["pandas_freq"]), "n_series": e["n_series"], "length": e["length"],
            "targets": e["targets"], "known": e["known"], "past": e["past"], "static": e["static"],
            "text_flags": known_text, "task": e["task"], "size_bytes": e["size_bytes"], "checks": e["checks"],
            "downloaded": file_path(e) is not None, "path": e["path"], "sha256": e["sha256"],
            "large": e["size_bytes"] > 50_000_000, "sampled_above": DEFAULT_MAX_SERIES,
        })
    return out


# ---------------------------------------------------------------------------- reading
_NO_EVENT = {"", "0", "0.0", "None", "none", "nan"}


def _numbers(cell, kind: str) -> np.ndarray:
    if kind == "list-text":  # an event name or nothing: becomes an on/off flag
        return np.array([0.0 if v is None or str(v) in _NO_EVENT else 1.0 for v in cell], dtype=np.float64)
    return np.asarray(cell, dtype=np.float64)  # missing values (None) become NaN; booleans become 0/1


def read(e: dict[str, Any], max_series: int = DEFAULT_MAX_SERIES, seed: int = 0) -> Dataset:
    """Build a Dataset from the downloaded file, streaming it so a 30,000-series file never sits in memory."""
    import pyarrow.parquet as pq

    path = file_path(e)
    if path is None:
        raise FileNotFoundError(f"'{e['id']}' is not downloaded yet: download it from the Datasets page first")
    max_series = max(1, min(int(max_series), MAX_SERIES_LIMIT))
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    keep = (set(range(total)) if total <= max_series
            else set(np.sort(np.random.default_rng(seed).choice(total, max_series, replace=False)).tolist()))
    groups = [("targets", e["targets"]), ("known", e["known"]), ("past", e["past"])]
    kinds = {c: e["columns"][c]["kind"] for _, cols in groups for c in cols}
    offset = pd.tseries.frequencies.to_offset(e["pandas_freq"])
    series: list[Series] = []
    row = 0
    for batch in pf.iter_batches(batch_size=256, columns=["id", "timestamp", *kinds]):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            if row in keep:
                ts = np.array(data["timestamp"][i], dtype="datetime64[ms]")
                mats = {g: [_numbers(data[c][i], kinds[c]) for c in cols] for g, cols in groups}
                full = pd.date_range(ts[0], ts[-1], freq=offset)
                if len(full) != len(ts) or not np.array_equal(full.values.astype("datetime64[ms]"), ts):
                    pos = full.get_indexer(pd.DatetimeIndex(ts))  # gaps and unaligned stamps become NaN on a regular grid
                    ok = pos >= 0
                    for g in mats:
                        for j, arr in enumerate(mats[g]):
                            grid = np.full(len(full), np.nan)
                            grid[pos[ok]] = arr[ok]
                            mats[g][j] = grid
                    ts = full.values.astype("datetime64[ms]")
                stack = lambda g: np.stack(mats[g]) if mats[g] else None  # noqa: E731
                series.append(Series(str(data["id"][i]), stack("targets"), ts, stack("past"), stack("known")))
            row += 1
    sampled = total > len(series)
    flags = [c for c in e["known"] if kinds[c] == "list-text"]
    known_names = [f"{c} (event: on/off)" if c in flags else c for c in e["known"]]
    note = (f" A seeded random sample of {len(series)} of {total} series is used (raise 'max series' to use more)."
            if sampled else "")
    return Dataset(
        id=f"hub-{e['id']}", name=title(e), series=series, freq=e["pandas_freq"], seasonal_period=e["task"]["seasonality"],
        target_names=list(e["targets"]), past_names=list(e["past"]), known_names=known_names, source="hub",
        license="other (declared on the Hugging Face card; check the original source's terms)",
        description=f"{FEV_REPO} / {e['config']} @ {FEV_REVISION[:10]}, fev-bench task definition.{note}",
        contamination_risk="possible", tags=["online", e["family"]],
        extra={"repo": FEV_REPO, "revision": FEV_REVISION, "entry": e["id"], "n_series_total": total, "sampled": sampled,
               "recommended": e["task"], "static_columns_ignored": e["static"]})
