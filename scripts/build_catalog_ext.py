#!/usr/bin/env python3
"""Build the GIFT-Eval and Monash parts of the online catalog (catalog_gift.json, catalog_monash.json).

Unlike fev-bench, these two repositories cannot be inspected remotely (Arrow streams and zipped text files have no footer to
read lengths from), so each file is downloaded once into ``.cache/research/catalog_build`` (outside the user-facing cache),
read completely, and deleted. What is recorded is exact: series counts, lengths, missing values, constant series, the time
axis, plus the Hugging Face LFS SHA-256 the app later checks every download against. Everything is pinned, so a run is
reproducible; results of finished datasets are kept next to the downloads, so an interrupted run resumes.

    .venv/Scripts/python.exe scripts/build_catalog_ext.py gift   [--only m4_yearly electricity/H]
    .venv/Scripts/python.exe scripts/build_catalog_ext.py monash [--only tourism_yearly]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

from _common import ROOT

sys.path.insert(0, str(ROOT / "src"))

from tsfm_lab import isolation

isolation.apply(ROOT)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from huggingface_hub import HfApi, hf_hub_download

from tsfm_lab.data import tsf

warnings.filterwarnings("ignore")
DATA = ROOT / "src" / "tsfm_lab" / "data"
WORK = ROOT / ".cache" / "research" / "catalog_build"

GIFT_REPO, GIFT_REVISION = "Salesforce/GiftEval", "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
GIFT_RULES = ("SalesforceAIResearch/gift-eval", "38d068b76984a29360ab340a0c693f9cd1090a30", "src/gift_eval/data.py")
MONASH_REPO, MONASH_REVISION = "Monash-University/monash_tsf", "58aafbe2712ff481c014f562e42723f2820fd5d4"

# ---------------------------------------------------------------------------- GIFT-Eval's own task rules (data.py above)
TEST_SPLIT, MAX_WINDOW = 0.1, 20
M4_PRED = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
TERMS = {"short": 1, "medium": 10, "long": 15}  # the longer terms exist only for sub-daily data
DOMAINS = {  # the benchmark's own domain labels (its dataset_properties.json)
    **dict.fromkeys(["m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly"], "Econ/Fin"),
    **dict.fromkeys(["electricity", "ett1", "ett2", "solar"], "Energy"),
    **dict.fromkeys(["hospital", "covid_deaths", "us_births"], "Healthcare"),
    **dict.fromkeys(["saugeen", "temperature_rain", "kdd_cup_2018", "jena_weather"], "Nature"),
    **dict.fromkeys(["car_parts", "restaurant", "hierarchical_sales"], "Sales"),
    **dict.fromkeys(["loop_seattle", "sz_taxi", "m_dense"], "Transport"),
    **dict.fromkeys(["bitbrains_fast_storage", "bitbrains_rnd", "bizitobs_application", "bizitobs_service", "bizitobs_l2c"],
                    "Web/CloudOps"),
}
PRETTY = {"ett1": "ETT1", "ett2": "ETT2", "loop_seattle": "LOOP Seattle", "sz_taxi": "SZ taxi", "m_dense": "M-DENSE",
          "us_births": "US births", "covid_deaths": "COVID deaths", "kdd_cup_2018_with_missing": "KDD Cup 2018",
          "temperature_rain_with_missing": "Temperature and rain", "car_parts_with_missing": "Car parts",
          "saugeenday": "Saugeen river flow", "jena_weather": "Jena weather", "bizitobs_l2c": "BizITObs L2C",
          "bizitobs_application": "BizITObs application", "bizitobs_service": "BizITObs service",
          "bitbrains_fast_storage": "Bitbrains fast storage", "bitbrains_rnd": "Bitbrains RND",
          "hierarchical_sales": "Hierarchical sales"}


# Variable names of the two ETT datasets (the columns of the ETT csv files). GIFT-Eval stores the seven variables unnamed; the order
# was checked against fev-bench's ETT task, which names them: ett1/ett2 at 15 minutes match its two series value for value, and the
# hourly, daily and weekly files are the sums of the finer ones, variable by variable (checked on 2026-10-28).
ETT_VARIABLES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def month_name(number: int) -> str:
    return ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][(number - 1) % 12]


def gift_alias(freq: str) -> str:
    """GIFT-Eval's stored 'start' is the first instant of the period, so month/quarter/year rules are the start-anchored ones."""
    n, unit, anchor = re.fullmatch(r"(\d*)([A-Z]+)(?:-([A-Z]{3}))?", freq).groups()
    n = int(n or 1)
    months = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    if unit == "T":
        return "min" if n == 1 else f"{n}min"
    if unit == "H":
        return "h" if n == 1 else f"{n}h"
    if unit == "S":
        return f"{n}s"
    if unit == "D":
        return "D" if n == 1 else f"{n}D"
    if unit == "W":
        return f"W-{anchor}" if anchor else "W-SUN"
    if unit == "M":
        return "MS"
    if unit == "Q":
        return f"QS-{month_name(months[anchor or 'DEC'] + 1)}"
    if unit == "A":
        return f"YS-{month_name(months[anchor or 'DEC'] + 1)}"
    raise ValueError(f"unsupported frequency {freq}")


def seasonality(freq: str) -> int:
    """The natural cycle in steps (daily data: the week; sub-daily: the day), as the fev-bench entries use."""
    n, unit, _ = re.fullmatch(r"(\d*)([A-Z]+)(?:-([A-Z]{3}))?", freq).groups()
    n = int(n or 1)
    return {"A": 1, "Q": 4, "M": 12, "W": 1, "D": 7, "H": 24}.get(unit) or (1440 // n if unit == "T" else 3600 // n)


def gift_task(name: str, freq: str, min_length: int) -> dict:
    unit = re.sub(r"^\d+", "", freq).split("-")[0]
    base = (M4_PRED if name.startswith("m4") else PRED)[unit]
    windows = lambda h: 1 if name.startswith("m4") else min(max(1, int(np.ceil(TEST_SPLIT * min_length / h))), MAX_WINDOW)
    terms = ["short", "medium", "long"] if unit in ("T", "H", "S") and not name.startswith("m4") else ["short"]
    out = [{"term": t, "horizon": base * TERMS[t], "num_windows": windows(base * TERMS[t])} for t in terms]
    return {"horizon": out[0]["horizon"], "num_windows": out[0]["num_windows"], "seasonality": seasonality(freq), "terms": out}


# ---------------------------------------------------------------------------- arrow reading
def listify(t: pa.DataType) -> pa.DataType:
    if pa.types.is_fixed_size_list(t) or pa.types.is_list(t) or pa.types.is_large_list(t):
        return pa.list_(listify(t.value_type))
    return t


def as_lists(array: pa.Array) -> pa.Array:
    """Fixed-size lists (the multivariate datasets store 7 variables that way) become ordinary lists."""
    return array.cast(listify(array.type))


def flat_floats(array: pa.Array) -> pa.Array:
    while pa.types.is_list(array.type):
        array = array.flatten()
    return array


def count_missing(array: pa.Array) -> tuple[int, int]:
    flat = flat_floats(array)
    nans = pc.sum(pc.is_nan(flat.cast(pa.float64()))).as_py() or 0
    return int(flat.null_count + nans), len(flat)


def inspect_arrow(paths: list[Path], name: str, freq_dir: str | None) -> dict:
    lengths: list[int] = []
    ids: list[str] = []
    freqs: set[str] = set()
    stats = collections.Counter()
    dims = pdims = None
    alias = None
    off_grid = 0
    for path in paths:
        with pa.memory_map(str(path)) as src:
            for batch in pa.ipc.open_stream(src):
                target = as_lists(batch.column("target"))
                multi = pa.types.is_list(target.type.value_type)
                dims = dims or (len(target[0].values) if multi else 1)
                lengths += (pc.list_value_length(pc.list_slice(target, 0, 1).flatten()) if multi
                            else pc.list_value_length(target)).to_pylist()
                m, t = count_missing(target)
                stats["target_missing"], stats["target_total"] = stats["target_missing"] + m, stats["target_total"] + t
                if "past_feat_dynamic_real" in batch.schema.names:
                    past = as_lists(batch.column("past_feat_dynamic_real"))
                    pm = pa.types.is_list(past.type.value_type)
                    pdims = pdims or (len(past[0].values) if pm else 1)
                    m, t = count_missing(past)
                    stats["past_missing"], stats["past_total"] = stats["past_missing"] + m, stats["past_total"] + t
                    plen = (pc.list_value_length(pc.list_slice(past, 0, 1).flatten()) if pm else pc.list_value_length(past)).to_pylist()
                    stats["past_length_mismatch"] += sum(int(a != b) for a, b in zip(plen, lengths[-len(plen):]))
                for r in range(len(target)):
                    cell = target[r].values
                    series = [cell[d].values for d in range(dims)] if multi else [cell]
                    stats["constant"] += int(all(np.nanmax(a) == np.nanmin(a) for a in
                                                 (np.asarray(s.to_numpy(zero_copy_only=False), dtype=np.float64) for s in series)
                                                 if np.isfinite(a).any()))
                ids += batch.column("item_id").to_pylist()
                freqs.update(batch.column("freq").unique().to_pylist())
                starts = batch.column("start").to_pylist()
                starts = [s[0] if isinstance(s, list) else s for s in starts]
                if len(freqs) == 1 and alias is None:
                    alias = gift_alias(next(iter(freqs)))
                    if alias.startswith("W-"):  # stored starts are not always the day after the anchor: follow the data
                        alias = "W-" + pd.Timestamp(starts[0]).day_name()[:3].upper()
                off = pd.tseries.frequencies.to_offset(alias)
                off_grid += sum(int(not off.is_on_offset(pd.Timestamp(s))) for s in starts)
    if len(freqs) != 1:
        raise SystemExit(f"{name}: several frequencies {freqs}")
    freq = next(iter(freqs))
    return {"freq": freq, "pandas_freq": alias, "n_series": len(lengths), "unique_ids": len(set(ids)), "dims": dims, "past_dims": pdims or 0,
            "lengths": lengths, "stats": dict(stats), "starts_off_grid": off_grid}


def length_summary(lengths: list[int]) -> dict:
    a = np.asarray(lengths)
    return {"min": int(a.min()), "median": int(np.median(a)), "max": int(a.max()), "total_points": int(a.sum())}


def pct(part: int, total: int) -> float | None:
    return None if not total else round(100 * part / total, 3)


# ---------------------------------------------------------------------------- GIFT-Eval
def build_gift(only: set[str] | None) -> dict:
    api = HfApi()
    info = api.dataset_info(GIFT_REPO, revision=GIFT_REVISION)
    tree = [e for e in api.list_repo_tree(GIFT_REPO, repo_type="dataset", revision=GIFT_REVISION, recursive=True)
            if getattr(e, "size", None) is not None]
    WORK.mkdir(parents=True, exist_ok=True)
    by_path = {f.path: f for f in tree}
    dirs: dict[str, tuple[list, int]] = {}  # folder -> (data files, row count the dataset itself declares)
    for path in sorted(by_path):
        if path.endswith("/state.json"):
            # The folder may also hold cache-*.arrow copies left by `datasets.map`: only the files state.json lists are the data.
            directory = path.rsplit("/", 1)[0]
            state = json.loads(Path(hf_hub_download(GIFT_REPO, path, repo_type="dataset", revision=GIFT_REVISION, cache_dir=WORK / "meta")).read_text())
            files = [by_path[f"{directory}/{d['filename']}"] for d in state["_data_files"]]
            expected = json.loads(Path(hf_hub_download(GIFT_REPO, f"{directory}/dataset_info.json", repo_type="dataset", revision=GIFT_REVISION,
                                                       cache_dir=WORK / "meta")).read_text())["splits"]["train"]["num_examples"]
            dirs[directory] = (files, expected)
    entries = []
    seen: dict[tuple, str] = {}
    for directory, (shards, declared) in sorted(dirs.items()):
        if only and directory not in only:
            continue
        shards.sort(key=lambda f: f.path)
        content = tuple(s_.lfs.sha256 for s_ in shards)
        if content in seen:  # the same bytes under two folder names (jena_weather and jena_weather/10T)
            print(f"  {directory}: identical to {seen[content]}, listed once", flush=True)
            continue
        seen[content] = directory
        key = "v2|" + "|".join(f"{s.path}:{s.lfs.sha256}" for s in shards)
        memo = WORK / ("gift-" + directory.replace("/", "-") + ".json")
        found = json.loads(memo.read_text()) if memo.exists() and json.loads(memo.read_text())["key"] == key else None
        started = time.time()
        if found is None:
            folder = WORK / "dl-gift"
            paths = [Path(hf_hub_download(GIFT_REPO, s.path, repo_type="dataset", revision=GIFT_REVISION, cache_dir=folder)) for s in shards]
            found = {"key": key, "result": inspect_arrow(paths, directory, None)}
            memo.write_text(json.dumps(found))
            shutil.rmtree(folder, ignore_errors=True)
        r = found["result"]
        if r["n_series"] != declared:  # dataset_info.json can be stale (restaurant: 809 declared, 807 in the files and in the paper)
            print(f"     !! {directory}: dataset_info.json declares {declared} series, the files hold {r['n_series']} (the files are used)", flush=True)
        name, _, freq_dir = directory.partition("/")
        base = name.lower().replace("_with_missing", "").replace("saugeenday", "saugeen")
        if base not in DOMAINS and not base.startswith("m4"):
            raise SystemExit(f"no domain for {directory}")
        pretty = PRETTY.get(name.lower(), name.replace("_", " "))
        size = sum(s.size for s in shards)
        lengths = r["lengths"]
        task = gift_task(name.lower(), r["freq"], min(lengths))
        s = r["stats"]
        entries.append({
            "id": "gift." + name.lower() + (f".{freq_dir}" if freq_dir else ""), "config": directory, "variant": None, "title": pretty,
            "family": DOMAINS.get(base, "Econ/Fin"), "freq": r["freq"], "pandas_freq": r["pandas_freq"],
            "files": [{"path": s_.path, "size_bytes": int(s_.size), "sha256": s_.lfs.sha256} for s_ in shards],
            "size_bytes": int(size), "n_series": r["n_series"], "length": length_summary(lengths), "task": task,
            "targets": (["target"] if r["dims"] == 1 else ETT_VARIABLES if name.lower() in ("ett1", "ett2") and r["dims"] == 7
                        else [f"variable {i + 1}" for i in range(r["dims"])]),
            "notes": ([("Variable names are those of the ETT csv files; the order was checked against fev-bench's ETT task value by value.")]
                      if name.lower() in ("ett1", "ett2") else
                      ["GIFT-Eval stores the variables unnamed, so they are numbered in the order of the file."] if r["dims"] > 1 else []),
            "known": [], "past": [f"past variable {i + 1}" for i in range(r["past_dims"])], "static": [],
            "columns": {"target": {"kind": "list-numeric"}, **({"past_feat_dynamic_real": {"kind": "list-numeric"}} if r["past_dims"] else {})},
            "checks": {"sampled_series": r["n_series"], "target_missing_pct": pct(s["target_missing"], s["target_total"]),
                       "known_missing_pct": None, "past_missing_pct": pct(s.get("past_missing", 0), s.get("past_total", 0)) if r["past_dims"] else None,
                       "constant_series": s["constant"], "series_with_time_gaps": 0,
                       "starts_off_grid": r["starts_off_grid"], "past_length_mismatch": s.get("past_length_mismatch", 0),
                       "duplicate_ids": r["n_series"] - r["unique_ids"]}
                      | ({"declared_n_series": declared} if declared != r["n_series"] else {}),
        })
        e = entries[-1]
        print(f"  {e['id']:44s} {e['n_series']:6d} series x {e['length']['median']:7d} ({e['pandas_freq']:6s}) "
              f"dims={r['dims']} past={r['past_dims']} h={task['horizon']}x{task['num_windows']} {size / 1e6:7.1f} MB ({time.time() - started:4.0f}s)", flush=True)
        bad = {k: v for k, v in e["checks"].items() if k in ("starts_off_grid", "past_length_mismatch", "duplicate_ids") and v}
        if bad:
            print("     !! ", bad, flush=True)
    license_ = info.card_data.get("license") if info.card_data else None
    return {
        "generated": time.strftime("%Y-%m-%d"),
        "source": {
            "key": "gift", "name": "GIFT-Eval", "repo": GIFT_REPO, "revision": GIFT_REVISION, "format": "gift-arrow",
            "card_license": license_[0] if isinstance(license_, list) else license_, "benchmark": "GIFT-Eval",
            "rules_repo": GIFT_RULES[0], "rules_ref": GIFT_RULES[1], "rules_path": GIFT_RULES[2]},
        "entries": entries,
    }


# ---------------------------------------------------------------------------- Monash
def monash_configs(script: Path) -> dict[str, dict]:
    """The dataset configurations of the loader script, read from its syntax tree (the script itself is never executed)."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "MonashTSFBuilderConfig":
            kwargs = {}
            for kw in node.keywords:
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    pass
            if "name" in kwargs:
                out[kwargs["name"]] = kwargs
    return out


# The Monash paper (Godahewa et al., 2021, section on experiments): competition datasets keep the competition's horizon, "12 months
# ahead for monthly datasets, 8 weeks for weekly datasets (5 for solar weekly), 30 days for daily datasets" and "for half-hourly,
# hourly and other low frequency datasets the forecasting horizon is one week, e.g. 168 for hourly datasets". One week of the minute
# and seconds data is not practical (and the paper did not evaluate those four datasets), so they keep the loader's default.
PAPER_HORIZON = {"monthly": 12, "weekly": 8, "daily": 30, "hourly": 168, "half_hourly": 336, "10_minutes": 1008}
COMPETITION_HORIZON = {"kdd_cup_2018": 48}  # KDD Cup 2018 asked for the next 48 hours; the paper keeps competition horizons
DEFAULT_HORIZON = {"yearly": 4, "quarterly": 4, "monthly": 12, "weekly": 8, "daily": 30, "hourly": 48, "half_hourly": 48,
                   "10_minutes": 48, "minutely": 60, "4_seconds": 60}  # the loader's own map; used only when a file declares none
SEASON = {"yearly": 1, "quarterly": 4, "monthly": 12, "weekly": 1, "daily": 7, "hourly": 24, "half_hourly": 48,
          "10_minutes": 144, "minutely": 1440, "4_seconds": 900}


def inspect_tsf(path: Path) -> dict:
    header, series = tsf.read(path)
    lengths, starts, constant, missing, total = [], [], 0, 0, 0
    names: list[str] = []
    for attrs, values in series:
        lengths.append(len(values))
        starts.append(attrs.get("start_timestamp"))
        names.append(attrs.get("series_name", ""))
        nan = int(np.isnan(values).sum())
        missing, total = missing + nan, total + len(values)
        finite = values[np.isfinite(values)]
        constant += int(finite.size > 0 and finite.min() == finite.max())
    return {"header": {"relation": header.relation, "frequency": header.frequency, "horizon": header.horizon,
                       "missing": header.missing, "equal_length": header.equal_length,
                       "attributes": header.attributes}, "lengths": lengths, "starts": starts, "constant": constant,
            "missing": missing, "total": total, "unique_names": len(set(names)),
            "attribute_names": [n for n, _ in header.attributes]}


def build_monash(only: set[str] | None) -> dict:
    api = HfApi()
    info = api.dataset_info(MONASH_REPO, revision=MONASH_REVISION)
    files = {e.path: e for e in api.list_repo_tree(MONASH_REPO, repo_type="dataset", revision=MONASH_REVISION, recursive=True)
             if getattr(e, "size", None) is not None and e.path.endswith(".zip")}
    WORK.mkdir(parents=True, exist_ok=True)
    script = Path(hf_hub_download(MONASH_REPO, "monash_tsf.py", repo_type="dataset", revision=MONASH_REVISION, cache_dir=WORK / "script"))
    configs = monash_configs(script)
    by_file = {}
    for cfg in configs.values():
        by_file[Path(cfg.get("file_name", "") or "").stem.removesuffix(".tsf")] = cfg
    entries = []
    for path, f in sorted(files.items()):
        stem = Path(path).stem
        cfg = by_file.get(stem) or by_file.get(stem.removesuffix("_dataset")) or next(
            (c for c in configs.values() if c.get("url", "").endswith(Path(path).name)), None)
        if cfg is None:
            raise SystemExit(f"no loader configuration for {path}")
        name = cfg["name"]
        if only and name not in only:
            continue
        memo = WORK / f"monash-{name}.json"
        key = f"{path}:{f.lfs.sha256}"
        found = json.loads(memo.read_text()) if memo.exists() and json.loads(memo.read_text())["key"] == key else None
        started = time.time()
        if found is None:
            folder = WORK / "dl-monash"
            local = Path(hf_hub_download(MONASH_REPO, path, repo_type="dataset", revision=MONASH_REVISION, cache_dir=folder))
            found = {"key": key, "result": inspect_tsf(local)}
            memo.write_text(json.dumps(found))
            shutil.rmtree(folder, ignore_errors=True)
        r = found["result"]
        h = r["header"]
        frequency = h["frequency"]
        if frequency not in tsf.STEPS:
            raise SystemExit(f"{name}: unknown frequency {frequency}")
        starts = [tsf.parse_start(s) if s else tsf.parse_start(tsf.DEFAULT_START) for s in r["starts"]]
        if frequency == "weekly":
            days = collections.Counter(pd.Timestamp(s).day_name()[:3].upper() for s in starts)
            pandas_freq = f"W-{days.most_common(1)[0][0]}"
        else:
            pandas_freq = tsf.ALIASES[frequency]
        month_based = frequency in ("yearly", "quarterly", "monthly")
        late_days = sum(int(pd.Timestamp(s).day > 28) for s in starts) if month_based else 0
        if h["horizon"]:
            horizon, source = h["horizon"], "header"
        elif cfg.get("prediction_length"):
            horizon, source = cfg["prediction_length"], "loader"
        elif name in COMPETITION_HORIZON:
            horizon, source = COMPETITION_HORIZON[name], "competition"
        elif frequency in PAPER_HORIZON:
            horizon, source = PAPER_HORIZON[frequency], "paper"
        else:
            horizon, source = DEFAULT_HORIZON[frequency], "default"
        lengths = r["lengths"]
        static = [n for n in r["attribute_names"] if n not in ("series_name", "start_timestamp")]
        notes = []
        if cfg.get("multivariate") or cfg.get("item_id_column"):
            notes.append("The loader on Hugging Face groups some lines into multivariate series; here every line of the file is its "
                         "own univariate series, which is how the archive publishes it.")
        entries.append({
            "id": f"monash.{name}", "config": name, "variant": None, "title": name.replace("_", " "), "family": "Monash",
            "freq": frequency, "pandas_freq": pandas_freq, "files": [{"path": path, "size_bytes": int(f.size), "sha256": f.lfs.sha256}],
            "size_bytes": int(f.size), "n_series": len(lengths), "length": length_summary(lengths),
            "task": {"horizon": int(horizon), "num_windows": int(cfg.get("rolling_evaluations", 1)), "seasonality": SEASON[frequency],
                     "horizon_source": source},
            "targets": ["target"], "known": [], "past": [], "static": static,
            "columns": {"target": {"kind": "list-numeric"}},
            "description": cfg.get("description", ""), "notes": notes,
            "checks": {"sampled_series": len(lengths), "target_missing_pct": pct(r["missing"], r["total"]), "known_missing_pct": None,
                       "past_missing_pct": None, "constant_series": r["constant"], "series_with_time_gaps": 0,
                       "declares_missing": h["missing"], "equal_length": h["equal_length"], "starts_after_day_28": late_days,
                       "duplicate_names": len(lengths) - r["unique_names"]},
        })
        e = entries[-1]
        print(f"  {e['id']:40s} {e['n_series']:7d} series  {e['length']['min']:7d}-{e['length']['max']:8d} {pandas_freq:7s} "
              f"h={horizon}({source}) {f.size / 1e6:7.1f} MB ({time.time() - started:4.0f}s)"
              + (f"  !! {late_days} starts after day 28" if late_days else ""), flush=True)
    license_ = info.card_data.get("license") if info.card_data else None
    return {
        "generated": time.strftime("%Y-%m-%d"),
        "source": {"key": "monash", "name": "Monash archive", "repo": MONASH_REPO, "revision": MONASH_REVISION, "format": "monash-tsf",
                   "card_license": license_[0] if isinstance(license_, list) else license_, "benchmark": "Monash archive",
                   "rules_repo": MONASH_REPO, "rules_ref": MONASH_REVISION, "rules_path": "monash_tsf.py"},
        "entries": sorted(entries, key=lambda e: e["id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=["gift", "monash"])
    parser.add_argument("--only", nargs="*", help="datasets to (re)inspect, for a quick test")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    build = {"gift": build_gift, "monash": build_monash}[args.which]
    catalog = build(set(args.only) if args.only else None)
    out = args.out or DATA / f"catalog_{args.which}.json"
    out.write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(catalog['entries'])} entries written to {out}")


if __name__ == "__main__":
    main()
