"""Showcase computations: one interactive forecast, and the measured effect of covariates over many windows.

The effect is never read off a single window: it is the pooled weighted quantile loss over the most recent
windows of the chosen series, with and without each covariate, so it can be checked and is not cherry-picked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tsfm_lab.data.hub import cached_path, read_fev
from tsfm_lab.data.schema import Dataset, Series
from tsfm_lab.eval import metrics as M
from tsfm_lab.eval.backtest import BacktestConfig, run_backtest
from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.showcase.scenarios import SCENARIOS, Scenario

BANDS = (0.1, 0.25, 0.3, 0.5, 0.7, 0.75, 0.9)  # 80% and 50% central intervals + median (40% when a model has only deciles)
_CACHE: dict[str, Dataset] = {}


def model_options(body: dict[str, Any]) -> dict[str, Any]:
    """Showcase default: set covariates that are flat in the context to missing (upstream issue #7 workaround).

    Without it, a covariate that is constant in the history but changes in the forecast (e.g. a discount that never
    ran before) can blow up the upper quantiles. The caller may override it explicitly.
    """
    return {"mask_flat_covariates": True, **(body.get("options") or {})}


def scenario(sid: str) -> Scenario:
    if sid not in SCENARIOS:
        raise KeyError(f"unknown scenario '{sid}'")
    return SCENARIOS[sid]


def is_ready(sc: Scenario) -> bool:
    return cached_path(sc.name, sc.freq) is not None


def dataset_for(sc: Scenario) -> Dataset:
    if sc.id not in _CACHE:
        ids = list(sc.series)
        _CACHE[sc.id] = read_fev(sc.name, sc.freq, sc.target, sc.covariates, ids, title=sc.title)
    return _CACHE[sc.id]


def _series(ds: Dataset, sid: str) -> Series:
    for s in ds.series:
        if s.id == sid:
            return s
    raise KeyError(f"unknown series '{sid}'")


def _x(s: Series, a: int, b: int) -> list[str]:
    return [str(t) for t in s.timestamps[a:b]]


def window_params(sc: Scenario) -> dict[str, int]:
    """The chart shows a FIXED window of the most recent data; the user drags the divider inside it.

    ``k`` = number of window steps on the left of the divider (what the model sees); everything on its right is
    forecast, so the horizon is whatever remains up to the window's end.
    """
    w = sc.view + sc.horizon
    return {"window": w, "k_default": sc.view, "k_min": max(8, round(w * 0.2)), "k_max": w - max(4, round(w * 0.06))}


def forecast(svc, body: dict[str, Any], progress=None) -> dict[str, Any]:
    """Interactive forecast from a divider position ``cut`` (steps from the window start) to the end of the window."""
    sc = scenario(body["scenario"])
    ds = dataset_for(sc)
    s = _series(ds, body["series"])
    keys = [c.key for c in sc.covariates]
    enabled = [k for k in body["covariates"] if k in keys]
    wp = window_params(sc)
    w = wp["window"]
    k = int(body.get("cut") or wp["k_default"])
    k = max(wp["k_min"], min(wp["k_max"], k))
    start = s.length - w
    if start < 64:
        raise ValueError("series too short for this scenario")
    origin, h = start + k, w - k
    lo = max(0, origin - sc.context)
    ctx = s.target[:, lo:origin][None]
    sel = [keys.index(c) for c in enabled]
    known = s.known_covariates[sel][:, lo:origin + h][None] if sel else None

    adapter, loaded = svc.pooled(body["model"], body["device"])
    res = adapter.forecast(ForecastRequest(ctx, h, known_covariates=known, seasonal_period=ds.seasonal_period,
                                           options=model_options(body)))
    q = res.quantiles[0, 0]  # (H, Q)
    bands = {str(lv): q[:, i] for i, lv in enumerate(res.quantile_levels) if any(abs(lv - b) < 1e-9 for b in BANDS)}
    real = s.target[0, start:]
    actual = real[k:]
    scaled = M.history_scale(s.target[None, :, :origin], ds.seasonal_period)
    metrics = {}
    if np.isfinite(actual).any():
        items = M.item_metrics(actual[None, None], res.quantiles, res.quantile_levels, scaled)
        metrics = {m: float(v[0, 0]) for m, v in items.items() if m in ("mase", "coverage_80") and np.isfinite(v[0, 0])}
    x = _x(s, start, s.length)
    return {
        "scenario": sc.id, "series": s.id, "window": w, "k": k, "k_min": wp["k_min"], "k_max": wp["k_max"], "horizon": h,
        "x": x, "real": real, "bands": bands,
        "covariates": [{"key": c.key, "label": c.label, "hint": c.hint, "binary": c.binary, "enabled": c.key in enabled,
                        "values": s.known_covariates[i, start:]} for i, c in enumerate(sc.covariates)],
        "timings": {**res.timings, "model_load_s": None if loaded is None else loaded.get("load_s")},
        "warnings": res.warnings, "device": res.device, "metrics": metrics,
        "start": x[k], "end": x[-1],
    }


# ---------------------------------------------------------------------------- measured effect
def _subset(ds: Dataset, ids: tuple[str, ...], idx: list[int]) -> Dataset:
    series = [Series(s.id, s.target, s.timestamps, None, s.known_covariates[idx] if idx else None)
              for s in ds.series if s.id in ids]
    return Dataset(ds.id, ds.name, series, ds.freq, ds.seasonal_period, ds.target_names, [], [], ds.source, ds.license,
                   ds.description, ds.contamination_risk, ds.tags)


def impact(svc, body: dict[str, Any], progress=None) -> dict[str, Any]:
    """Pooled quantile loss without covariates, with all, and with each one alone, on the latest windows."""
    sc = scenario(body["scenario"])
    options = model_options(body)
    key = hashlib.sha1(json.dumps([sc.id, body["model"], options, sc.impact_windows, sc.horizon], sort_keys=True).encode()).hexdigest()[:12]
    store = Path(svc.root) / "runs" / "showcase"
    cache_file = store / f"{sc.id}-{key}.json"
    if cache_file.is_file() and not body.get("refresh"):
        return {**json.loads(cache_file.read_text(encoding="utf-8")), "cached": True}
    ds = dataset_for(sc)
    keys = [c.key for c in sc.covariates]
    configs = [("none", []), ("all", list(range(len(keys)))), *[(k, [i]) for i, k in enumerate(keys)]]
    adapter = svc.registry.create(body["model"])
    adapter.load(body["device"])
    cfg = BacktestConfig(horizon=sc.horizon, n_windows=sc.impact_windows, stride=sc.impact_stride, max_context=sc.context,
                         use_covariates=True, batch_size=64, options=options)
    out: dict[str, Any] = {}
    try:
        for n, (name, idx) in enumerate(configs):
            if progress:
                progress(n, len(configs), f"measuring: {name}")
            res = run_backtest(_subset(ds, sc.impact_series, idx), [adapter], cfg)
            o = res.models[body["model"]]
            out[name] = {"wql": M.pooled_wql(res.y, o.quantiles, o.levels),
                         "coverage_80": float(np.nanmean(o.items["coverage_80"])), "forecasts": int(res.y.shape[0])}
    finally:
        adapter.unload()
    base = out["none"]["wql"]
    gain = lambda k: (1 - out[k]["wql"] / base) if base and np.isfinite(out[k]["wql"]) else None
    result = {"scenario": sc.id, "model": body["model"], "horizon": sc.horizon, "windows": sc.impact_windows,
              "series": len(sc.impact_series), "forecasts": out["none"]["forecasts"], "gain_all": gain("all"),
              "gain_each": {k: gain(k) for k in keys}, "detail": out, "cached": False}
    store.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
