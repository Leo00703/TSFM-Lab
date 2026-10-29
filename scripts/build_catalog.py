#!/usr/bin/env python3
"""Build the curated online-dataset catalog (src/tsfm_lab/data/catalog_fev.json) and verify every entry on a sample.

Sources, all pinned so the result is reproducible:
  * the dataset card of ``autogluon/fev_datasets`` at a fixed revision: configs, features, row counts, file sizes;
  * ``benchmarks/fev_bench/tasks.yaml`` of ``autogluon/fev`` at a fixed commit: for every task the target column(s),
    which covariates are known in advance / past-only, the seasonality and the recommended horizon and windows.

For every entry the script reads, straight from Hugging Face without keeping the file, the exact length of every series
and a sample of up to 30 series, and records what it found (missing values, irregular time axis, text columns...).
Nothing is downloaded into the project cache. Run it again only to move to a newer pinned revision:

    .venv/bin/python scripts/build_catalog.py [--only ETT_1H rohlik_sales_1D] [--out path.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _common import ROOT

sys.path.insert(0, str(ROOT / "src"))

from tsfm_lab import isolation

isolation.apply(ROOT)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from tsfm_lab.data.hub import FEV_REPO, FEV_REVISION, alias

TASKS_REPO = "autogluon/fev"
TASKS_REF = "4367c673ce9cc4c2077f78b3046954258fadb63a"  # last commit that changed benchmarks/fev_bench/tasks.yaml
TASKS_PATH = "benchmarks/fev_bench/tasks.yaml"
OUT = ROOT / "src" / "tsfm_lab" / "data" / "catalog_fev.json"
SAMPLE_SERIES = 30
warnings.filterwarnings("ignore")


def fetch_tasks() -> tuple[list[dict], str]:
    url = f"https://raw.githubusercontent.com/{TASKS_REPO}/{TASKS_REF}/{TASKS_PATH}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tsfm-lab-catalog"})) as response:
        raw = response.read()
    return yaml.safe_load(raw)["tasks"], hashlib.sha256(raw).hexdigest()


def kind_of(arrow_type: pa.DataType) -> str:
    inner = arrow_type.value_type if pa.types.is_list(arrow_type) else arrow_type
    if pa.types.is_boolean(inner):
        return "binary"
    if pa.types.is_floating(inner) or pa.types.is_integer(inner):
        return "numeric"
    if pa.types.is_string(inner) or pa.types.is_large_string(inner):
        return "text"
    return "other"


def inspect_file(fs: HfFileSystem, path: str, wanted: dict[str, list[str]], freq: str | None) -> dict:
    """Exact lengths plus checks on a sample of series, read remotely (nothing is stored)."""
    with fs.open(f"datasets/{FEV_REPO}@{FEV_REVISION}/{path}", "rb") as handle:
        pf = pq.ParquetFile(handle)
        schema = pf.schema_arrow
        columns = {f.name: {"kind": ("list-" if pa.types.is_list(f.type) else "") + kind_of(f.type)} for f in schema}
        lengths = pc.list_value_length(pf.read(columns=["timestamp"]).column("timestamp")).to_numpy(zero_copy_only=False)
        used = [c for group in ("targets", "known", "past") for c in wanted[group] if c in schema.names]
        batch = next(pf.iter_batches(batch_size=SAMPLE_SERIES, columns=["timestamp", *used])).to_pydict()
    n = len(batch["timestamp"])

    def gaps(rule: str) -> int:
        offset = pd.tseries.frequencies.to_offset(rule)
        return sum(int(len(pd.date_range(ts[0], ts[-1], freq=offset)) != len(ts)) for ts in batch["timestamp"])

    # The pandas alias that really matches the time axis: the declared one when the sample is regular under it, else the
    # one pandas infers (monthly data may sit on month starts or ends, a few datasets declare no frequency at all).
    declared = alias(freq) if freq else None
    pandas_freq = declared
    irregular = gaps(declared) if declared else n
    if irregular:
        for ts in batch["timestamp"]:
            guess = pd.infer_freq(pd.DatetimeIndex(ts)) if len(ts) >= 3 else None
            if guess:
                pandas_freq, irregular = guess, gaps(guess)
                break

    def nan_pct(cols: list[str]) -> float | None:
        vals, total = 0, 0
        for c in cols:
            if c not in batch or columns[c]["kind"] != "list-numeric":
                continue
            for cell in batch[c]:
                arr = np.asarray(cell, dtype=np.float64)
                vals += int(np.isnan(arr).sum())
                total += arr.size
        return None if total == 0 else round(100 * vals / total, 3)

    constant = 0
    target_cols = [c for c in wanted["targets"] if c in batch]
    for i in range(n):
        rows = [np.asarray(batch[c][i], dtype=np.float64) for c in target_cols]
        constant += int(all(np.nanmax(r) == np.nanmin(r) for r in rows if np.isfinite(r).any()))
    return {
        "pandas_freq": pandas_freq, "n_series": len(lengths),
        "length": {"min": int(lengths.min()), "median": int(np.median(lengths)), "max": int(lengths.max()),
                   "total_points": int(lengths.sum())},
        "columns": columns,
        "checks": {"sampled_series": n, "target_missing_pct": nan_pct(wanted["targets"]),
                   "known_missing_pct": nan_pct(wanted["known"]), "past_missing_pct": nan_pct(wanted["past"]),
                   "constant_series": constant, "series_with_time_gaps": irregular},
    }


def build(only: set[str] | None) -> dict:
    tasks, tasks_sha = fetch_tasks()
    card = yaml.safe_load(Path(hf_hub_download(FEV_REPO, "README.md", repo_type="dataset", revision=FEV_REVISION))
                          .read_text(encoding="utf-8").split("---")[1])
    info = {c["config_name"]: c for c in card["dataset_info"]}
    paths = {c["config_name"]: c["data_files"][0]["path"].replace("train-*", "train-00000-of-00001.parquet")
             for c in card["configs"]}
    files = {e.path: e for e in HfApi().list_repo_tree(FEV_REPO, repo_type="dataset", revision=FEV_REVISION, recursive=True)
             if getattr(e, "size", None) is not None and e.path.endswith(".parquet")}

    per_config: dict[str, list[dict]] = {}
    for task in tasks:
        per_config.setdefault(task["dataset_config"], []).append(task)

    jobs = []
    for config, group in sorted(per_config.items()):
        if only and config not in only:
            continue
        path = paths[config]
        file = files[path]
        parts = path.split("/")
        # The folder after the family is a frequency ('1H', '15T') for most datasets, an identifier ('1062') for a few,
        # and absent for the rest: when it is not a frequency, the data tell.
        family = parts[0]
        freq = parts[1] if len(parts) == 3 and re.fullmatch(r"\d*[THDWMQYA]", parts[1]) else None
        for task in group:
            targets = task.get("target", "target")
            targets = [targets] if isinstance(targets, str) else list(targets)
            variant = task["task_name"].split("/", 1)[1] if "/" in task.get("task_name", "") else None
            jobs.append({
                "id": config + (f"--{variant}" if variant else ""), "config": config, "variant": variant, "family": family,
                "freq": freq, "path": path, "size_bytes": int(file.size), "sha256": file.lfs.sha256,
                "card_rows": info[config]["splits"][0]["num_examples"],
                "task": {k: task[k] for k in ("horizon", "num_windows", "seasonality") if k in task},
                "targets": targets, "known": list(task.get("known_dynamic_columns") or []),
                "past": list(task.get("past_dynamic_columns") or []), "static": list(task.get("static_columns") or []),
            })

    fs = HfFileSystem()
    cache: dict[tuple, dict] = {}

    def work(job: dict) -> dict:
        key = (job["path"], tuple(job["targets"]), tuple(job["known"]), tuple(job["past"]))
        started = time.time()
        if key not in cache:
            cache[key] = inspect_file(fs, job["path"], {"targets": job["targets"], "known": job["known"], "past": job["past"]},
                                      job["freq"])
        found = cache[key]
        if found["n_series"] != job["card_rows"]:
            raise SystemExit(f"{job['id']}: the card says {job['card_rows']} series, the file has {found['n_series']}")
        print(f"  {job['id']:34s} {found['n_series']:6d} series  {job['size_bytes'] / 1e6:7.1f} MB  ({time.time() - started:4.1f}s)",
              flush=True)
        return {**{k: v for k, v in job.items() if k != "card_rows"}, **found, "freq": job["freq"] or found["pandas_freq"]}

    with ThreadPoolExecutor(max_workers=6) as pool:
        entries = list(pool.map(work, jobs))
    return {
        "generated": time.strftime("%Y-%m-%d"),
        "source": {"repo": FEV_REPO, "revision": FEV_REVISION, "card_license": card.get("license"),
                   "tasks_repo": TASKS_REPO, "tasks_ref": TASKS_REF, "tasks_path": TASKS_PATH, "tasks_sha256": tasks_sha},
        "entries": sorted(entries, key=lambda e: e["id"].lower()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="dataset configs to (re)inspect, for a quick test")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    catalog = build(set(args.only) if args.only else None)
    args.out.write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(catalog['entries'])} entries written to {args.out}")


if __name__ == "__main__":
    main()
