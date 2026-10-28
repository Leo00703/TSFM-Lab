"""Rolling-origin backtest: run one or more loaded models on identical windows of a dataset.

Every model sees exactly the same contexts, covariates and horizons, so differences are due to the
model. Series of different lengths are left-padded with NaN (missing) inside a batch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tsfm_lab.data.schema import Dataset
from tsfm_lab.eval import metrics as M
from tsfm_lab.models.base import ForecastRequest, ModelAdapter


@dataclass
class BacktestConfig:
    horizon: int = 24
    n_windows: int = 3
    stride: int | None = None  # steps between window origins; default = horizon
    max_context: int | None = None  # cap on context length given to every model; None = all history
    max_series: int | None = None
    batch_size: int = 32
    use_covariates: bool = True
    min_context: int = 16
    history_tail: int = 512  # steps of history kept per item for plots
    warmup: bool = True  # one untimed pass first, so reported latency excludes one-time start-up costs
    options: dict[str, Any] = field(default_factory=dict)  # per-model options (symmetric_averaging, ...)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelOutcome:
    model_id: str
    device: str
    quantiles: np.ndarray  # (N, V, H, Q)
    levels: tuple[float, ...]
    batch_ms: list[float]
    warmup_ms: float | None  # cold first call (CUDA init, kernel compilation...), reported separately
    total_s: float
    warnings: list[str]
    peak_vram_mb: float | None
    notes: list[str]
    items: dict[str, np.ndarray] = field(default_factory=dict)
    horizon: dict[str, np.ndarray] = field(default_factory=dict)
    calibration: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    skill: dict[str, dict[str, float]] = field(default_factory=dict)  # vs each other model


def _shared_levels(all_levels: list[tuple[float, ...]]) -> tuple[float, ...]:
    shared = [lv for lv in all_levels[0] if all(any(abs(lv - o) < 1e-9 for o in levels) for levels in all_levels[1:])]
    return tuple(shared)


def _score_on_shared_levels(outcomes: dict[str, ModelOutcome], y: np.ndarray, scale: np.ndarray) -> None:
    """SQL and WQL average over quantile levels, so models that predict different levels are not comparable on them.

    When the models of one run do not all produce the same levels (Timestral's 21 against TimesFM's nine deciles), every model
    is scored on the levels they all share, which makes the headline SQL/WQL (and the skill scores built on them) comparable.
    Each model's own-level values stay available as ``sql_own_levels`` / ``wql_own_levels``.
    """
    if len(outcomes) < 2:
        return
    shared = _shared_levels([o.levels for o in outcomes.values()])
    if len(shared) < 2 or 0.5 not in shared or all(tuple(o.levels) == shared for o in outcomes.values()):
        return
    for out in outcomes.values():
        idx = [next(i for i, lv in enumerate(out.levels) if abs(lv - s) < 1e-9) for s in shared]
        q = out.quantiles[..., idx]
        common = M.item_metrics(y, q, shared, scale)
        if tuple(out.levels) != shared:
            for key in ("sql", "wql"):
                out.items[f"{key}_own_levels"] = out.items[key]
        for key in ("sql", "wql"):
            out.items[key] = common[key]
        out.horizon["sql"] = M.horizon_metrics(y, q, shared, scale)["sql"]
        out.summary = M.summarize(out.items)
        out.notes.append(f"SQL and WQL use the {len(shared)} quantile levels shared by every model in this run "
                         f"({shared[0]:g} to {shared[-1]:g}), so models with different level sets are comparable"
                         + ("; this model's own-level values are kept as sql_own_levels / wql_own_levels." if tuple(out.levels) != shared else "."))


@dataclass
class BacktestResult:
    dataset: dict[str, Any]
    config: dict[str, Any]
    item_series: list[str]
    item_window: list[int]  # 0 = latest window
    item_origin: list[int]  # index of the first forecast step in the series
    item_origin_time: list[str | None]
    y: np.ndarray  # (N, V, H)
    history: np.ndarray  # (N, V, L) left-NaN-padded context tail
    scale: np.ndarray  # (N, V)
    covariates: np.ndarray | None  # (N, Ck, L + H) known covariates over tail + horizon (NaN padded)
    models: dict[str, ModelOutcome]


def make_items(dataset: Dataset, cfg: BacktestConfig) -> list[tuple[int, int, int]]:
    stride = cfg.stride or cfg.horizon
    chosen = dataset.series[: cfg.max_series] if cfg.max_series else dataset.series
    items = []
    for si, s in enumerate(chosen):
        for k in range(cfg.n_windows):
            origin = s.length - cfg.horizon - k * stride
            if origin < cfg.min_context:
                break
            if not np.isfinite(s.target[:, :origin]).any():
                continue
            items.append((si, k, origin))
    if not items:
        raise ValueError("no valid backtest windows: series are too short for the requested horizon/windows")
    return items


def _pad_left(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[-1] >= length:
        return arr[..., -length:]
    pad = np.full((*arr.shape[:-1], length - arr.shape[-1]), np.nan)
    return np.concatenate([pad, arr], axis=-1)


def run_backtest(dataset: Dataset, adapters: list[ModelAdapter], cfg: BacktestConfig,
                 progress: Callable[[int, int, str], None] | None = None) -> BacktestResult:
    """``adapters`` must already be loaded (the caller chooses and times the device)."""
    for a in adapters:
        if not a.is_loaded:
            raise RuntimeError(f"model {a.info.id} is not loaded")
    items = make_items(dataset, cfg)
    n, h, v = len(items), cfg.horizon, dataset.n_targets
    series = dataset.series

    # Shared, model-independent arrays
    ctx_len = [min(o, cfg.max_context) if cfg.max_context else o for _, _, o in items]
    L = max(ctx_len)
    context = np.stack([_pad_left(series[si].target[:, :o], L) for (si, _, o) in items])  # (N, V, L)
    y = np.stack([series[si].target[:, o: o + h] for (si, _, o) in items])
    known = past = None
    if cfg.use_covariates and dataset.n_known:
        known = np.stack([_pad_left(series[si].known_covariates[:, : o + h], L + h) for (si, _, o) in items])
    if cfg.use_covariates and dataset.n_past:
        past = np.stack([_pad_left(series[si].past_covariates[:, :o], L) for (si, _, o) in items])
    tail = min(L, cfg.history_tail)
    history = context[:, :, -tail:]
    scale = M.history_scale(context, dataset.seasonal_period)

    outcomes: dict[str, ModelOutcome] = {}
    total_steps = n * len(adapters)
    done = 0
    for adapter in adapters:
        caps = adapter.info.capabilities
        notes: list[str] = []
        if v > 1 and not caps.multivariate:
            raise ValueError(f"{adapter.info.name} does not support multivariate targets")
        m_known, m_past = known, past
        if known is not None and not caps.known_covariates:
            m_known = None
            notes.append("known covariates ignored (unsupported by this model)")
        if past is not None and not caps.past_covariates:
            m_past = None
            notes.append("past covariates ignored (unsupported by this model)")
        if caps.max_context and L > caps.max_context:
            notes.append(f"context longer than {caps.max_context}: the model keeps only the most recent steps")

        def request_for(lo: int, hi: int, m_past=m_past, m_known=m_known) -> ForecastRequest:
            return ForecastRequest(
                context=context[lo:hi], horizon=h,
                past_covariates=None if m_past is None else m_past[lo:hi],
                known_covariates=None if m_known is None else m_known[lo:hi],
                seasonal_period=dataset.seasonal_period, options=dict(cfg.options))

        warmup_ms = None
        if cfg.warmup:
            t0 = time.perf_counter()
            adapter.forecast(request_for(0, min(cfg.batch_size, n)))
            warmup_ms = round((time.perf_counter() - t0) * 1000, 3)
        pieces, batch_ms, warns, peaks = [], [], [], []
        started = time.perf_counter()
        for lo in range(0, n, cfg.batch_size):
            hi = min(lo + cfg.batch_size, n)
            res = adapter.forecast(request_for(lo, hi))
            pieces.append(res.quantiles)
            batch_ms.append(res.timings.get("forecast_ms", float("nan")))
            if "peak_vram_mb" in res.timings:
                peaks.append(res.timings["peak_vram_mb"])
            for w in res.warnings:
                if w not in warns:
                    warns.append(w)
            levels = res.quantile_levels
            done += hi - lo
            if progress:
                progress(done, total_steps, adapter.info.id)
        q = np.concatenate(pieces, axis=0)
        out = ModelOutcome(adapter.info.id, adapter.device or "?", q.astype(np.float32), tuple(levels), batch_ms,
                           warmup_ms, time.perf_counter() - started, warns, max(peaks) if peaks else None, notes)
        out.items = M.item_metrics(y, q, levels, scale)
        out.horizon = M.horizon_metrics(y, q, levels, scale)
        out.calibration = M.calibration_curve(y, q, levels)
        out.summary = M.summarize(out.items)
        outcomes[adapter.info.id] = out

    _score_on_shared_levels(outcomes, y, scale)

    for mid, out in outcomes.items():
        for other_id, other in outcomes.items():
            if other_id != mid:
                out.skill[other_id] = {k: M.skill_score(out.items[k], other.items[k])
                                       for k in ("mase", "sql")}

    def stamp(si: int, origin: int) -> str | None:
        ts = series[si].timestamps
        return None if ts is None else str(ts[origin])

    return BacktestResult(
        dataset=dataset.summary(), config=cfg.to_dict(),
        item_series=[series[si].id for si, _, _ in items], item_window=[k for _, k, _ in items],
        item_origin=[o for _, _, o in items], item_origin_time=[stamp(si, o) for si, _, o in items],
        y=y, history=history, scale=scale,
        covariates=None if known is None else known[:, :, -(tail + h):], models=outcomes)
