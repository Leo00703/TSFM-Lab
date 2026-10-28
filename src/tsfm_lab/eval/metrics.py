"""Forecast metrics, all computed from quantile forecasts with explicit, documented definitions.

Shapes: y (N, V, H) actuals (NaN = unobserved); q (N, V, H, Q) quantiles at ``levels``;
scale (N, V) = in-sample mean absolute (seasonal) difference of each item's history.

Definitions (per item = one series/variate in one backtest window):
  pinball_tau(y, q)  = max(tau*(y-q), (tau-1)*(y-q))
  SQL   (scaled quantile loss) = 2 * mean_{h,tau} pinball / scale     (a discretised, scaled CRPS)
  WQL   (weighted quantile loss) = 2 * mean_tau [ sum_h pinball ] / sum_h |y|
  MASE  = mean_h |y - median| / scale
  coverage_c  = share of steps with y inside the central c-interval (c in 50/80/90/98%)
  interval score_c (Gneiting & Raftery) = (hi-lo) + (2/alpha)(lo-y)+ + (2/alpha)(y-hi)+, alpha = 1-c, / scale
  bias  = mean_h (median - y) / scale   (positive = over-forecast)
Items whose scale is 0 or undefined (e.g. constant history) have NaN scaled metrics and are counted, not hidden.
"""

from __future__ import annotations

import numpy as np

CENTRAL_INTERVALS = (0.5, 0.8, 0.9, 0.98)


def history_scale(history: np.ndarray, period: int) -> np.ndarray:
    """(N, V, L) histories (NaN-padded) -> (N, V) mean |x_t - x_{t-m}|; falls back to lag 1, then NaN."""
    n, v, _ = history.shape
    out = np.full((n, v), np.nan)
    for i in range(n):
        for j in range(v):
            row = history[i, j]
            row = row[np.isfinite(row)]
            for lag in (max(1, period), 1):
                if row.size > lag:
                    value = np.mean(np.abs(row[lag:] - row[:-lag]))
                    if value > 0:
                        out[i, j] = value
                        break
    return out


def pinball(y: np.ndarray, q: np.ndarray, levels) -> np.ndarray:
    diff = y[..., None] - q
    tau = np.asarray(levels, dtype=float)
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def _level_index(levels, target: float) -> int | None:
    for i, lv in enumerate(levels):
        if abs(lv - target) < 1e-9:
            return i
    return None


def _central(levels, c: float) -> tuple[int, int] | None:
    lo, hi = _level_index(levels, (1 - c) / 2), _level_index(levels, (1 + c) / 2)
    return (lo, hi) if lo is not None and hi is not None else None


def item_metrics(y: np.ndarray, q: np.ndarray, levels, scale: np.ndarray) -> dict[str, np.ndarray]:
    """Per-item metrics, each (N, V)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _item_metrics(y, q, levels, scale)


def _item_metrics(y, q, levels, scale):
    med_i = _level_index(levels, 0.5)
    if med_i is None:
        raise ValueError("the model must produce the 0.5 quantile")
    median = q[..., med_i]
    pin = pinball(y, q, levels)  # (N, V, H, Q)
    with np.errstate(all="ignore"):
        out = {
            "mase": np.nanmean(np.abs(y - median), axis=-1) / scale,
            "sql": 2 * np.nanmean(pin, axis=(-1, -2)) / scale,
            "wql": 2 * np.nanmean(np.nansum(pin, axis=-2), axis=-1) / np.nansum(np.abs(y), axis=-1),
            "bias": np.nanmean(median - y, axis=-1) / scale,
            "mae": np.nanmean(np.abs(y - median), axis=-1),
        }
        out["wql"] = np.where(np.isfinite(out["wql"]), out["wql"], np.nan)
        for c in CENTRAL_INTERVALS:
            idx = _central(levels, c)
            if idx is None:
                continue
            lo, hi = q[..., idx[0]], q[..., idx[1]]
            tag = f"{round(c * 100)}"
            observed = np.isfinite(y)
            inside = (y >= lo) & (y <= hi)
            out[f"coverage_{tag}"] = np.where(observed.any(-1), (inside & observed).sum(-1) / observed.sum(-1), np.nan)
            alpha = 1 - c
            score = (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
            out[f"interval_score_{tag}"] = np.nanmean(score, axis=-1) / scale
            out[f"width_{tag}"] = np.nanmean(hi - lo, axis=-1) / scale
    return out


def horizon_metrics(y: np.ndarray, q: np.ndarray, levels, scale: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics per forecast step, each (H,), averaged over items with a defined scale."""
    valid = np.isfinite(scale)[..., None]  # (N, V, 1)
    med_i = _level_index(levels, 0.5)
    median = q[..., med_i]
    with np.errstate(all="ignore"):
        sc = np.where(valid, scale[..., None], np.nan)
        abs_err = np.abs(y - median) / sc
        sql = 2 * np.nanmean(pinball(y, q, levels), axis=-1) / sc
        out = {"mase": np.nanmean(abs_err, axis=(0, 1)), "sql": np.nanmean(sql, axis=(0, 1)),
               "bias": np.nanmean((median - y) / sc, axis=(0, 1))}
        for c in CENTRAL_INTERVALS:
            idx = _central(levels, c)
            if idx is None:
                continue
            observed = np.isfinite(y)
            inside = ((y >= q[..., idx[0]]) & (y <= q[..., idx[1]])) & observed
            out[f"coverage_{round(c * 100)}"] = inside.sum(axis=(0, 1)) / observed.sum(axis=(0, 1))
    return out


def calibration_curve(y: np.ndarray, q: np.ndarray, levels) -> dict[str, np.ndarray]:
    """Empirical share of actuals below each predicted quantile, per level (ideal: equals the level)."""
    observed = np.isfinite(y)
    below = ((y[..., None] <= q) & observed[..., None]).sum(axis=(0, 1, 2))
    return {"levels": np.asarray(levels, dtype=float), "empirical": below / max(1, int(observed.sum()))}


def summarize(items: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Mean / median / std / valid-count of every per-item metric (NaN items excluded, counted)."""
    out = {}
    for name, values in items.items():
        flat = values[np.isfinite(values)]
        out[name] = {
            "mean": float(flat.mean()) if flat.size else float("nan"),
            "median": float(np.median(flat)) if flat.size else float("nan"),
            "std": float(flat.std()) if flat.size else float("nan"),
            "n": int(flat.size), "n_nan": int(values.size - flat.size),
        }
    return out


def skill_score(model: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    """1 - geometric mean of per-item (model / baseline) ratios, over items where both are finite and > 0.

    Positive = better than the baseline (0.2 = 20% lower error on a geometric-mean basis).
    """
    m, b = np.asarray(model, float).ravel(), np.asarray(baseline, float).ravel()
    ok = np.isfinite(m) & np.isfinite(b) & (m > 0) & (b > 0)
    if not ok.any():
        return {"skill": float("nan"), "n": 0}
    ratio = np.exp(np.mean(np.log(m[ok] / b[ok])))
    return {"skill": float(1 - ratio), "n": int(ok.sum())}


def pooled_wql(y: np.ndarray, q: np.ndarray, levels) -> float:
    """Weighted quantile loss pooled over all items: 2 * mean_tau [sum pinball] / sum |y| (NaN actuals ignored).

    Pooling (instead of averaging per-item ratios) stays well defined when some items are all zeros,
    e.g. a store that is closed or a solar panel at night.
    """
    pin = pinball(y, q, levels)
    denom = float(np.nansum(np.abs(y)))
    if denom == 0:
        return float("nan")
    return float(2 * np.nanmean(np.nansum(pin, axis=tuple(range(pin.ndim - 1)))) / denom)
