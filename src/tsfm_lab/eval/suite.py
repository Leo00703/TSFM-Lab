"""A leaderboard run: several models on a group of catalog datasets, scored the way fev-bench scores.

Every model is loaded once and backtested on each dataset with the windows of its own benchmark task (capped, so a run stays
affordable). A seasonal-naive reference always runs on the same windows. Per dataset and model the error is the mean over
series and windows; the ranking score is the geometric mean over datasets of ``model error / reference error`` (reported as a
skill score: 0 = as good as seasonal naive, positive = better). The ranking only counts datasets every ranked model completed,
so a model that cannot take multivariate targets is not compared on a different, easier set.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tsfm_lab.data.hub_catalog import freq_label
from tsfm_lab.data.schema import Dataset
from tsfm_lab.eval.backtest import BacktestConfig, run_backtest
from tsfm_lab.models.registry import Registry

REFERENCE = "seasonal-naive"
METRICS = ("sql", "mase", "wql")


@dataclass
class SuiteConfig:
    max_series: int = 20  # series per dataset (a seeded sample: the same ones for every model)
    max_windows: int = 3  # cap on the windows of a benchmark task
    max_context: int | None = 2048  # the same cap for every model, so long-context models are not favoured
    use_covariates: bool = True
    options: dict[str, Any] = field(default_factory=lambda: {"mask_flat_covariates": True})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clipped_ratio(model: float, reference: float) -> float | None:
    if not (math.isfinite(model) and math.isfinite(reference)) or reference <= 0:
        return None
    return float(np.clip(model / reference, 0.01, 100.0))


def skill(ratios: list[float]) -> float | None:
    """100 * (1 - geometric mean of the ratios): the fev-bench skill score, in percent."""
    return None if not ratios else float(100 * (1 - math.exp(sum(math.log(r) for r in ratios) / len(ratios))))


def scorable(row: dict[str, Any]) -> bool:
    """A dataset scores nothing when the reference has no usable error: a series that is constant over its context (all zeros at night)
    has an error scale of zero, so every ratio is undefined."""
    reference = row["models"].get(REFERENCE) or {}
    sql = reference.get("sql")
    return sql is not None and math.isfinite(sql) and sql > 0


def aggregate(results: list[dict[str, Any]], ranked: list[str]) -> dict[str, Any]:
    """Leaderboard rows from per-dataset results. ``results[i]['models'][model]`` holds the errors or is missing when skipped."""
    common = [r for r in results if all(m in r["models"] for m in ranked) and scorable(r)]

    def rows(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for m in ranked:
            row: dict[str, Any] = {"datasets": len(subset)}
            for metric in METRICS:
                ratios = [x for r in subset if (x := clipped_ratio(r["models"][m][metric], r["models"][REFERENCE][metric])) is not None]
                row[f"{metric}_skill"] = skill(ratios)
            row["total_s"] = round(sum(r["models"][m]["total_s"] for r in subset), 2)
            out[m] = row
        for r in subset:  # a win is the lowest SQL on a dataset
            winner = min(ranked, key=lambda m: r["models"][m]["sql"] if math.isfinite(r["models"][m]["sql"]) else math.inf)
            if math.isfinite(r["models"][winner]["sql"]):
                out[winner]["wins"] = out[winner].get("wins", 0) + 1
        for row in out.values():
            row.setdefault("wins", 0)
        return out

    groups = sorted({r["group"] for r in common})
    return {"ranked": ranked, "common_datasets": len(common), "total_datasets": len(results),
            "overall": rows(common), "by_group": {g: rows([r for r in common if r["group"] == g]) for g in groups}}


def run_suite(entries: list[dict[str, Any]], model_ids: list[str], device: str, cfg: SuiteConfig, registry: Registry,
              load: Callable[[dict[str, Any], int], Dataset], progress: Callable[[int, int, str], None] | None = None,
              checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """``entries`` are catalog entries; ``load(entry, max_series)`` returns the dataset (injected so tests need no downloads).

    ``checkpoint`` receives the full result so far after every dataset, so a run that is stopped or killed keeps what it finished."""
    wanted = list(dict.fromkeys([*model_ids, REFERENCE]))
    adapters: dict[str, Any] = {}
    load_info: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    started = time.time()
    ranked = [m for m in model_ids if m != REFERENCE]

    def package() -> dict[str, Any]:
        board = aggregate([r for r in results if r["models"] and REFERENCE in r["models"]], ranked)
        board["total_datasets"] = len(results)  # including the ones that could not be scored
        return {"config": cfg.to_dict(), "device": device, "models": {m: _model_meta(registry, m, load_info[m]) for m in wanted},
                "datasets": results, "leaderboard": board, "planned": len(entries), "elapsed_s": round(time.time() - started, 1)}

    try:
        for mid in wanted:
            adapter = registry.create(mid)
            load_info[mid] = adapter.load("cpu" if adapter.info.kind == "baseline" else device)
            adapters[mid] = adapter
        for n, e in enumerate(entries):
            if progress:
                progress(n, len(entries), e["id"])
            ds = load(e, cfg.max_series)
            task = e["task"]
            bt = BacktestConfig(horizon=task["horizon"], n_windows=min(task["num_windows"], cfg.max_windows), max_series=cfg.max_series,
                                max_context=cfg.max_context, use_covariates=cfg.use_covariates, options=dict(cfg.options))
            usable = [m for m in wanted if ds.n_targets == 1 or adapters[m].info.capabilities.multivariate]
            row: dict[str, Any] = {"id": e["id"], "title": e["title"], "group": e["source"], "freq": e["pandas_freq"], "freq_label": freq_label(e["pandas_freq"]), "horizon": bt.horizon,
                                   "windows": bt.n_windows, "series": len(ds.series), "targets": ds.n_targets,
                                   "covariates": bool(ds.n_known or ds.n_past), "models": {}, "skipped": {}, "error": None}
            for m in wanted:
                if m not in usable:
                    row["skipped"][m] = "does not take multivariate targets"
            try:
                res = run_backtest(ds, [adapters[m] for m in usable], bt)
            except ValueError as error:  # recorded, the suite goes on (a stop is not a ValueError, so it still stops the run)
                too_short = str(error).startswith("no valid backtest windows")
                row["error"] = str(error) if too_short else f"a model failed on this dataset ({type(error).__name__}): {error}"
                results.append(row)
                if checkpoint:
                    checkpoint(package())
                continue
            row["items"] = int(res.y.shape[0])
            for m in usable:
                out = res.models[m]
                row["models"][m] = {**{k: float(np.nanmean(out.items[k])) for k in METRICS}, "total_s": round(out.total_s, 3),
                                    "warnings": out.warnings[:3], "notes": out.notes[:2]}
            results.append(row)
            if checkpoint:
                checkpoint(package())
        if progress:
            progress(len(entries), len(entries), "done")
    finally:
        for adapter in adapters.values():
            adapter.unload()
    return package()


def _model_meta(registry: Registry, model_id: str, load: dict[str, Any]) -> dict[str, Any]:
    info = registry.info(model_id)
    return {"name": info.name, "family": info.family, "kind": info.kind, "params": info.params, "license": info.license, "load": load}
