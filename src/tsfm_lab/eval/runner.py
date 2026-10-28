"""High-level experiment: load models on the chosen device, backtest, unload, persist."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tsfm_lab.data.schema import Dataset
from tsfm_lab.eval.backtest import BacktestConfig, BacktestResult, run_backtest
from tsfm_lab.models.registry import Registry, default_registry
from tsfm_lab.store.store import RunStore
from tsfm_lab.system.probe import environment_info

DEFAULT_BASELINES = ("naive", "seasonal-naive")


def run_experiment(model_id: str, dataset: Dataset, cfg: BacktestConfig, device: str = "cpu",
                   baselines: tuple[str, ...] = DEFAULT_BASELINES, store: RunStore | None = None,
                   registry: Registry | None = None, label: str | None = None,
                   progress: Callable[[int, int, str], None] | None = None,
                   request: dict[str, Any] | None = None) -> tuple[BacktestResult, str | None]:
    """Run ``model_id`` (on ``device``) plus the baselines on the same windows. Returns (result, run_id)."""
    registry = registry or default_registry()
    wanted = [model_id, *[b for b in baselines if b != model_id]]
    adapters, load_meta = [], {}
    try:
        for mid in wanted:
            adapter = registry.create(mid)
            is_baseline = adapter.info.kind == "baseline"
            load_meta[mid] = adapter.load("cpu" if is_baseline else device)
            adapters.append(adapter)
        result = run_backtest(dataset, adapters, cfg, progress)
    finally:
        for adapter in adapters:
            adapter.unload()
    run_id = None
    if store is not None:
        meta: dict[str, Any] = {"environment": environment_info(), "models": {}, "request": request}
        for mid in wanted:
            info = registry.info(mid)
            meta["models"][mid] = {"load": load_meta[mid], "name": info.name, "kind": info.kind,
                                   "params": info.params, "details": info.details}
        run_id = store.save(result, meta, label)
    return result, run_id
