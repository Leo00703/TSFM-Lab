"""Persistence of runs: ``runs/<id>/meta.json`` + ``arrays.npz`` and a SQLite index for listing.

Everything needed to inspect or reproduce a run is stored: configuration, dataset summary, device and
environment, per-model load info/timings/warnings/metrics, and the raw arrays (actuals, history,
quantile forecasts, per-item metrics) so charts can be redrawn without recomputation.
"""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tsfm_lab.eval.backtest import BacktestResult
from tsfm_lab.isolation import find_root


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types, tuples and NaN/inf (invalid JSON) into plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class RunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else find_root() / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "index.db"
        with sqlite3.connect(self.db) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, created TEXT, dataset_id TEXT, dataset_name TEXT, models TEXT,
                devices TEXT, horizon INTEGER, n_items INTEGER, headline TEXT, label TEXT)""")

    def save(self, result: BacktestResult, run_meta: dict[str, Any], label: str | None = None) -> str:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        folder = self.root / run_id
        folder.mkdir(parents=True)
        arrays: dict[str, np.ndarray] = {"y": result.y, "history": result.history, "scale": result.scale}
        if result.covariates is not None:
            arrays["covariates"] = result.covariates
        models_meta: dict[str, Any] = {}
        for mid, out in result.models.items():
            arrays[f"{mid}/quantiles"] = out.quantiles
            for k, v in out.items.items():
                arrays[f"{mid}/item/{k}"] = v
            models_meta[mid] = {
                "device": out.device, "levels": out.levels, "total_s": out.total_s, "batch_ms": out.batch_ms,
                "warmup_ms": out.warmup_ms,
                "warnings": out.warnings, "notes": out.notes, "peak_vram_mb": out.peak_vram_mb,
                "summary": out.summary, "skill": out.skill,
                "horizon": out.horizon, "calibration": out.calibration,
                **run_meta.get("models", {}).get(mid, {}),
            }
        meta = {
            "id": run_id, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label,
            "dataset": result.dataset, "config": result.config, "environment": run_meta.get("environment"),
            "request": run_meta.get("request"),
            "items": {"series": result.item_series, "window": result.item_window, "origin": result.item_origin,
                      "origin_time": result.item_origin_time},
            "models": models_meta,
        }
        (folder / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=1), encoding="utf-8")
        np.savez_compressed(folder / "arrays.npz", **arrays)
        headline = {mid: {"mase": m["summary"]["mase"]["mean"], "sql": m["summary"]["sql"]["mean"]}
                    for mid, m in models_meta.items()}
        with sqlite3.connect(self.db) as con:
            con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)", (
                run_id, meta["created"], result.dataset["id"], result.dataset["name"],
                json.dumps(list(models_meta)), json.dumps(sorted({m["device"] for m in models_meta.values()})),
                result.config["horizon"], len(result.item_series), json.dumps(to_jsonable(headline)), label))
        return run_id

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("models", "devices", "headline"):
                d[k] = json.loads(d[k])
            out.append(d)
        return out

    def meta(self, run_id: str) -> dict[str, Any]:
        path = self.root / run_id / "meta.json"
        if not path.is_file():
            raise KeyError(f"unknown run '{run_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    def arrays(self, run_id: str) -> dict[str, np.ndarray]:
        with np.load(self.root / run_id / "arrays.npz") as data:
            return {k: data[k] for k in data.files}

    def delete(self, run_id: str) -> None:
        import shutil

        shutil.rmtree(self.root / run_id, ignore_errors=True)
        with sqlite3.connect(self.db) as con:
            con.execute("DELETE FROM runs WHERE id = ?", (run_id,))


class BenchmarkStore:
    """Benchmark results as JSON files under ``runs/benchmarks/``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (Path(root) if root else find_root() / "runs") / "benchmarks"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, result: dict[str, Any]) -> str:
        bench_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + result["model"]
        result = {**result, "id": bench_id}
        (self.root / f"{bench_id}.json").write_text(json.dumps(to_jsonable(result), indent=1), encoding="utf-8")
        return bench_id

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            d = json.loads(path.read_text(encoding="utf-8"))
            out.append({"id": d["id"], "model": d["model"], "name": d["name"], "created": d["created"],
                        "devices": list(d["devices"]), "repeats": d["repeats"]})
        return out

    def get(self, bench_id: str) -> dict[str, Any]:
        path = self.root / f"{Path(bench_id).name}.json"
        if not path.is_file():
            raise KeyError(f"unknown benchmark '{bench_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, bench_id: str) -> None:
        (self.root / f"{Path(bench_id).name}.json").unlink(missing_ok=True)
