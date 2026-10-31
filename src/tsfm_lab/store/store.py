"""Persistence of runs: ``runs/<id>/meta.json`` + ``arrays.npz`` and a SQLite index for listing.

Everything needed to inspect or reproduce a run is stored: configuration, dataset summary, device and
environment, per-model load info/timings/warnings/metrics, and the raw arrays (actuals, history,
quantile forecasts, per-item metrics) so charts can be redrawn without recomputation.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import shutil
import sqlite3
from datetime import UTC, datetime
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


RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")


def write_json(path: Path, data: Any) -> None:
    """Write a result file so that a reader, or a server stopped half way, never sees a partial one."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


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
            "id": run_id, "created": datetime.now(UTC).isoformat(timespec="seconds"), "label": label,
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

    def _folder(self, run_id: str) -> Path:
        """The folder of a run. The id comes from a URL, so it must be exactly the shape ``save`` makes: nothing like '..' or a path gets through."""
        if not RUN_ID.fullmatch(run_id):
            raise KeyError(f"unknown run '{run_id[:40]}'")
        return self.root / run_id

    def meta(self, run_id: str) -> dict[str, Any]:
        path = self._folder(run_id) / "meta.json"
        if not path.is_file():
            raise KeyError(f"unknown run '{run_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    def arrays(self, run_id: str) -> dict[str, np.ndarray]:
        with np.load(self._folder(run_id) / "arrays.npz") as data:
            return {k: data[k] for k in data.files}

    def delete(self, run_id: str) -> None:
        folder = self._folder(run_id)
        shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():  # the run is still readable from its folder: keep it listed rather than hide it
            raise ValueError(f"could not delete run {run_id}: some of its files are in use. Close whatever has them open and try again")
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
        write_json(self.root / f"{bench_id}.json", to_jsonable(result))
        return bench_id

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                out.append({"id": d["id"], "model": d["model"], "name": d["name"], "created": d["created"],
                            "devices": list(d["devices"]), "repeats": d["repeats"]})
            except (ValueError, KeyError, TypeError, OSError):  # one damaged file must not hide the rest, nor break the models page
                continue
        return out

    def get(self, bench_id: str) -> dict[str, Any]:
        path = self.root / f"{Path(bench_id).name}.json"
        if not path.is_file():
            raise KeyError(f"unknown benchmark '{bench_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, bench_id: str) -> None:
        (self.root / f"{Path(bench_id).name}.json").unlink(missing_ok=True)


class SuiteStore:
    """Leaderboard runs as JSON files under ``runs/suites/``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (Path(root) if root else find_root() / "runs") / "suites"
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)

    def save(self, result: dict[str, Any], label: str | None = None, suite_id: str | None = None, status: str = "done") -> str:
        """Write (or rewrite) a run. A run in progress is saved after every dataset with status 'running'."""
        suite_id = suite_id or self.new_id()
        path = self.root / f"{Path(suite_id).name}.json"
        created = json.loads(path.read_text(encoding="utf-8"))["created"] if path.is_file() else datetime.now(UTC).isoformat(timespec="seconds")
        result = {**result, "id": suite_id, "label": label, "created": created, "status": status}
        write_json(path, to_jsonable(result))
        return suite_id

    def finish(self, suite_id: str, status: str) -> None:
        """A run that was left 'running' (stopped, crashed, server restarted) gets its real status; finished runs are untouched."""
        path = self.root / f"{Path(suite_id).name}.json"
        if not path.is_file():
            return
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("status") == "running":
            d["status"] = status
            write_json(path, d)

    @staticmethod
    def _rescored(d: dict[str, Any]) -> dict[str, Any]:
        """The ranking of a saved run, counted by today's rules (a run saved before a rule changed would otherwise keep the old count)."""
        from tsfm_lab.data.hub_catalog import caution
        from tsfm_lab.eval.suite import REFERENCE, aggregate

        rows, board = d.get("datasets"), d.get("leaderboard")
        if not rows or not board:
            return d
        scored = [r for r in rows if r["models"] and REFERENCE in r["models"]]
        out = {**d, "leaderboard": {**aggregate(scored, board["ranked"]), "total_datasets": board["total_datasets"]}}
        for r in rows:  # a dataset that a ranked model lists as its training data does not compare that model with the others
            r["caution"] = caution(r["id"], board["ranked"])
        flagged = {r["id"] for r in rows if r["caution"]}
        if flagged:  # the same ranking without them
            out["leaderboard_trusted"] = {**aggregate([r for r in scored if r["id"] not in flagged], board["ranked"]),
                                          "total_datasets": board["total_datasets"] - len(flagged)}
        return out

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                d = self._rescored(json.loads(path.read_text(encoding="utf-8")))
                board = d.get("leaderboard_trusted") or d["leaderboard"]
                best = max(board["ranked"], key=lambda m: (board["overall"][m]["sql_skill"] is not None, board["overall"][m]["sql_skill"] or 0),
                           default=None) if board["common_datasets"] else None
                out.append({"id": d["id"], "created": d["created"], "label": d.get("label"), "models": board["ranked"], "device": d["device"],
                            "status": d.get("status", "done"), "planned": d.get("planned", board["total_datasets"]),
                            "datasets": d["leaderboard"]["total_datasets"], "common": board["common_datasets"], "best": best,
                            "best_skill": None if best is None else board["overall"][best]["sql_skill"], "elapsed_s": d["elapsed_s"]})
            except (ValueError, KeyError, TypeError, OSError):  # one damaged file must not hide the other runs
                continue
        return out

    def get(self, suite_id: str) -> dict[str, Any]:
        path = self.root / f"{Path(suite_id).name}.json"
        if not path.is_file():
            raise KeyError(f"unknown leaderboard run '{suite_id}'")
        return self._rescored(json.loads(path.read_text(encoding="utf-8")))

    def delete(self, suite_id: str) -> None:
        (self.root / f"{Path(suite_id).name}.json").unlink(missing_ok=True)
