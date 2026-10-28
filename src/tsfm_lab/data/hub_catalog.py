"""The curated online-dataset catalog: fev-bench, GIFT-Eval and the Monash archive.

Each source has a JSON file built and verified by ``scripts/build_catalog.py`` (fev-bench) or ``scripts/build_catalog_ext.py``
(GIFT-Eval, Monash). Every entry is one dataset task over one or more files of a Hugging Face dataset repository at a pinned
revision. Files are downloaded only when the user asks, checked against the SHA-256 published by Hugging Face, and kept in the
project cache.
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

from tsfm_lab.data import tsf
from tsfm_lab.data.schema import Dataset, Series

SOURCE_FILES = {"fev": "catalog_fev.json", "gift": "catalog_gift.json", "monash": "catalog_monash.json"}
DEFAULT_MAX_SERIES = 200  # a larger file is sampled (seeded, sorted) so memory and time stay bounded
MAX_SERIES_LIMIT = 5000
LARGE_BYTES = 50_000_000

# What each source says about licence and about the chance that a model has already seen its data.
TEXT = {
    "fev": {
        "name": "fev-bench", "benchmark": "fev-bench", "format": "fev-parquet",
        "license": "The dataset card declares the licence as “other”: the data come from public sources whose own terms apply.",
        "contamination": "Public benchmark: Timestral's training data is not disclosed, so it may have seen this data. Scores here are "
                         "indications, not proof of out-of-sample skill.",
    },
    "gift": {
        "name": "GIFT-Eval", "benchmark": "GIFT-Eval", "format": "gift-arrow",
        "license": "The dataset card declares Apache-2.0 for the compilation; each dataset keeps the terms of its original source.",
        "contamination": "Built as a held-out benchmark, but it reuses well-known public series (M4, ETT, electricity, solar…) that also "
                         "sit in public pretraining corpora, and Timestral's training data is not disclosed. Scores here are "
                         "indications, not proof of out-of-sample skill.",
    },
    "monash": {
        "name": "Monash archive", "benchmark": "Monash archive", "format": "monash-tsf",
        "license": "The dataset card declares CC BY 4.0 for the archive; the series come from competitions and public agencies whose "
                   "own terms apply.",
        "contamination": "A classic public archive: its datasets appear in public pretraining corpora (the Chronos corpus "
                         "autogluon/chronos_datasets carries Monash datasets) and Timestral's training data is not disclosed. Scores "
                         "here are indications, not proof of out-of-sample skill.",
    },
}


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for key, name in SOURCE_FILES.items():
        path = Path(__file__).with_name(name)
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        src = {**raw["source"], **TEXT[key], "key": key}
        if key == "fev":
            src.update(rules_repo=src["tasks_repo"], rules_ref=src["tasks_ref"], rules_path=src["tasks_path"])
        sources[key] = src
        for e in raw["entries"]:
            entries.append({
                **e, "source": key, "repo": src["repo"], "revision": src["revision"], "format": src["format"],
                "files": e.get("files") or [{"path": e["path"], "size_bytes": e["size_bytes"], "sha256": e["sha256"]}],
                "notes": e.get("notes", []), "description": e.get("description", "")})
    # The same data under two names: identical series count and number of observations, in a different source.
    shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for e in entries:
        shape.setdefault((e["n_series"], e["length"]["total_points"]), []).append(e)
    for e in entries:
        e["overlaps"] = [o["id"] for o in shape[(e["n_series"], e["length"]["total_points"])] if o["source"] != e["source"]]
    return {"sources": sources, "entries": entries}


def sources() -> dict[str, dict[str, Any]]:
    return _catalog()["sources"]


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
             "a": "year", "s": "second"}
    word = names.get(unit)
    if word is None:
        return pandas_freq
    adjective = {"second": "every second", "minute": "every minute", "hour": "hourly", "day": "daily", "week": "weekly",
                 "month": "monthly", "quarter": "quarterly", "year": "yearly"}[word]
    return adjective if n == 1 and word not in ("minute", "second") else (f"every {n} {word}s" if n > 1 else adjective)


def title(e: dict[str, Any]) -> str:
    """The dataset name without its frequency suffix ('ETT_15T' -> 'ETT'); 'boomlet_1062' keeps its number."""
    if e.get("title"):
        return e["title"]
    base = re.sub(r"_\d*[THDWMQYA]$", "", e["config"]).replace("_", " ")
    return f"{base} · {e['variant'].replace('_', ' ')}" if e.get("variant") else base


# ---------------------------------------------------------------------------- files
def _cache_dir(e: dict[str, Any]) -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / f"datasets--{e['repo'].replace('/', '--')}" / "snapshots" / e["revision"]


def file_paths(e: dict[str, Any]) -> list[Path] | None:
    """Every file of the entry in the cache, or None when any is missing."""
    paths = [_cache_dir(e) / f["path"] for f in e["files"]]
    return paths if all(p.is_file() for p in paths) else None


def file_path(e: dict[str, Any]) -> Path | None:
    paths = file_paths(e)
    return paths[0] if paths else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(e: dict[str, Any]) -> list[Path]:
    """Fetch the files at the pinned revision and refuse them if a hash differs from the one Hugging Face publishes."""
    from huggingface_hub import hf_hub_download

    paths = []
    for f in e["files"]:
        path = Path(hf_hub_download(e["repo"], f["path"], repo_type="dataset", revision=e["revision"]))
        if sha256(path) != f["sha256"]:
            remove(e)
            raise ValueError(f"{e['id']}: a downloaded file does not match the published SHA-256; it was discarded")
        paths.append(path)
    return paths


def remove(e: dict[str, Any]) -> bool:
    removed = False
    for f in e["files"]:
        path = _cache_dir(e) / f["path"]
        if not path.is_file():
            continue
        target = path.resolve()
        path.unlink()
        if target != path and target.is_file():
            target.unlink()  # the blob behind the snapshot link (Linux/macOS); on Windows the file is the blob itself
        removed = True
    return removed


def listing() -> list[dict[str, Any]]:
    """Everything the catalog page needs, with the download status of each entry."""
    by_id = entries()
    out = []
    for e in _catalog()["entries"]:
        known_text = [c for c in e["known"] if e["columns"].get(c, {}).get("kind") == "list-text"]
        out.append({
            "id": e["id"], "source": e["source"], "title": title(e), "family": e["family"], "freq": e["pandas_freq"],
            "freq_label": freq_label(e["pandas_freq"]), "n_series": e["n_series"], "length": e["length"],
            "targets": e["targets"], "known": e["known"], "past": e["past"], "static": e["static"],
            "text_flags": known_text, "task": e["task"], "size_bytes": e["size_bytes"], "checks": e["checks"],
            "downloaded": file_paths(e) is not None, "files": e["files"], "path": e["files"][0]["path"],
            "sha256": e["files"][0]["sha256"], "large": e["size_bytes"] > LARGE_BYTES, "sampled_above": DEFAULT_MAX_SERIES,
            "notes": e["notes"], "description": e["description"],
            "also_in": [{"id": o, "source": by_id[o]["source"], "label": f"{title(by_id[o])} · {freq_label(by_id[o]['pandas_freq'])}"}
                        for o in e["overlaps"]],
        })
    return out


# ---------------------------------------------------------------------------- reading
_NO_EVENT = {"", "0", "0.0", "None", "none", "nan"}


def _numbers(cell, kind: str) -> np.ndarray:
    if kind == "list-text":  # an event name or nothing: becomes an on/off flag
        return np.array([0.0 if v is None or str(v) in _NO_EVENT else 1.0 for v in cell], dtype=np.float64)
    return np.asarray(cell, dtype=np.float64)  # missing values (None) become NaN; booleans become 0/1


def _sample(total: int, max_series: int, seed: int) -> set[int]:
    if total <= max_series:
        return set(range(total))
    return set(np.sort(np.random.default_rng(seed).choice(total, max_series, replace=False)).tolist())


def read(e: dict[str, Any], max_series: int = DEFAULT_MAX_SERIES, seed: int = 0) -> Dataset:
    """Build a Dataset from the downloaded files, streaming them so a 30,000-series file never sits in memory."""
    paths = file_paths(e)
    if paths is None:
        raise FileNotFoundError(f"'{e['id']}' is not downloaded yet: download it from the Datasets page first")
    max_series = max(1, min(int(max_series), MAX_SERIES_LIMIT))
    keep = _sample(e["n_series"], max_series, seed)
    reader = {"fev-parquet": _read_fev, "gift-arrow": _read_gift, "monash-tsf": _read_monash}[e["format"]]
    series, total, known_flags = reader(e, paths, keep)
    src = sources()[e["source"]]
    sampled = total > len(series)
    known_names = [f"{c} (event: on/off)" if c in known_flags else c for c in e["known"]]
    note = (f" A seeded random sample of {len(series)} of {total} series is used (raise 'max series' to use more)."
            if sampled else "")
    return Dataset(
        id=f"hub-{e['id']}", name=title(e), series=series, freq=e["pandas_freq"], seasonal_period=e["task"]["seasonality"],
        target_names=list(e["targets"]), past_names=list(e["past"]), known_names=known_names, source="hub",
        license=f"{src.get('card_license') or 'other'} (declared on the Hugging Face card; check the original source's terms)",
        description=f"{e['repo']} / {e['config']} @ {e['revision'][:10]}, {src['benchmark']} task definition.{note}",
        contamination_risk="possible", tags=["online", e["source"], e["family"]],
        extra={"repo": e["repo"], "revision": e["revision"], "entry": e["id"], "source": e["source"], "n_series_total": total,
               "sampled": sampled, "recommended": e["task"], "static_columns_ignored": e["static"]})


def _regular(ts: np.ndarray, arrays: dict[str, list[np.ndarray]], offset) -> np.ndarray:
    """Put gaps and unaligned stamps on a regular grid (they become NaN) and return the grid."""
    full = pd.date_range(ts[0], ts[-1], freq=offset)
    if len(full) == len(ts) and np.array_equal(full.values.astype("datetime64[ms]"), ts):
        return ts
    pos = full.get_indexer(pd.DatetimeIndex(ts))
    ok = pos >= 0
    for group in arrays.values():
        for j, arr in enumerate(group):
            grid = np.full(len(full), np.nan)
            grid[pos[ok]] = arr[ok]
            group[j] = grid
    return full.values.astype("datetime64[ms]")


def _read_fev(e, paths, keep):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(paths[0])
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
                ts = _regular(ts, mats, offset)
                stack = lambda g: np.stack(mats[g]) if mats[g] else None  # noqa: E731
                series.append(Series(str(data["id"][i]), stack("targets"), ts, stack("past"), stack("known")))
            row += 1
    return series, pf.metadata.num_rows, {c for c in e["known"] if kinds[c] == "list-text"}


def _lists(array):
    """Fixed-size lists (the multivariate datasets store their variables that way) become ordinary lists."""
    import pyarrow as pa

    def plain(t):
        return pa.list_(plain(t.value_type)) if (pa.types.is_list(t) or pa.types.is_fixed_size_list(t)) else t

    return array if "fixed_size" not in str(array.type) else array.cast(plain(array.type))


def _matrix(cell, multi: bool) -> np.ndarray:
    """One series of a GIFT-Eval row -> (variables, time), whether the row holds one variable or several."""
    values = cell.values
    if multi:
        return np.stack([np.asarray(values[d].values.to_numpy(zero_copy_only=False), dtype=np.float64) for d in range(len(values))])
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)[None, :]


def _read_gift(e, paths, keep):
    import pyarrow as pa

    offset = pd.tseries.frequencies.to_offset(e["pandas_freq"])
    series: list[Series] = []
    row = 0
    for path in sorted(paths):
        with pa.memory_map(str(path)) as source:
            for batch in pa.ipc.open_stream(source):
                wanted = [i for i in range(batch.num_rows) if row + i in keep]
                if wanted:
                    target = _lists(batch.column("target"))
                    past = _lists(batch.column("past_feat_dynamic_real")) if "past_feat_dynamic_real" in batch.schema.names else None
                    multi = pa.types.is_list(target.type.value_type)
                    past_multi = past is not None and pa.types.is_list(past.type.value_type)
                    ids, starts = batch.column("item_id").to_pylist(), batch.column("start").to_pylist()
                    for i in wanted:
                        start = starts[i][0] if isinstance(starts[i], list) else starts[i]
                        y = _matrix(target[i], multi)
                        p = _matrix(past[i], past_multi) if past is not None else None
                        ts = pd.date_range(pd.Timestamp(start), periods=y.shape[1], freq=offset).values.astype("datetime64[ms]")
                        series.append(Series(str(ids[i]), y, ts, p, None))
                row += batch.num_rows
    return series, e["n_series"], set()


def _read_monash(e, paths, keep):
    header, lines = tsf.read(paths[0], wanted=keep.__contains__)
    series: list[Series] = []
    for index, (attrs, values) in enumerate(lines):
        if values is not None:
            start = tsf.parse_start(attrs.get("start_timestamp") or tsf.DEFAULT_START)
            series.append(Series(str(attrs.get("series_name", index)), values[None, :],
                                 tsf.timestamps(start, header.frequency, len(values)), None, None))
            if len(series) == len(keep):
                break  # the rest of the file is not needed: the number of series in it is in the catalog
    return series, e["n_series"], set()
