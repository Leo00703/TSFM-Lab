"""Reference baselines (no weights, CPU only). They give every result a yardstick.

Probabilistic outputs are built from the baseline's own in-sample errors: the quantiles of the
(seasonal-)naive residuals, widened with sqrt(number of steps/seasons ahead). This is a transparent
empirical construction, not a fitted statistical model.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from tsfm_lab.models.base import (Capabilities, DeviceInfo, ForecastRequest, ForecastResult, ModelAdapter,
                                  ModelInfo)
from tsfm_lab.system.devices import cpu_name

LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
          0.9, 0.95, 0.99)


def _history(row: np.ndarray) -> np.ndarray:
    """Observed values of one series with leading/trailing NaN removed and inner gaps forward-filled."""
    valid = np.flatnonzero(np.isfinite(row))
    if valid.size == 0:
        return np.zeros(1)
    seg = row[valid[0]: valid[-1] + 1].copy()
    nan = ~np.isfinite(seg)
    if nan.any():
        idx = np.where(~nan, np.arange(seg.size), 0)
        np.maximum.accumulate(idx, out=idx)
        seg = seg[idx]
    return seg


def _residual_quantiles(errors: np.ndarray) -> np.ndarray:
    errors = errors[np.isfinite(errors)]
    if errors.size < 4:
        return np.zeros(len(LEVELS))
    return np.quantile(errors, LEVELS) - np.median(errors)  # centred: the median stays the point forecast


def _point_naive(hist: np.ndarray, h: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.full(h, hist[-1]), hist[1:] - hist[:-1], np.arange(1, h + 1, dtype=float)


def _point_seasonal(hist: np.ndarray, h: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = max(1, m)
    if hist.size <= m:
        return _point_naive(hist, h, 1)
    season = hist[-m:]
    point = np.array([season[i % m] for i in range(h)])
    steps = np.floor(np.arange(h) / m) + 1  # seasons ahead
    return point, hist[m:] - hist[:-m], steps


def _point_drift(hist: np.ndarray, h: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = (hist[-1] - hist[0]) / max(1, hist.size - 1)
    point = hist[-1] + slope * np.arange(1, h + 1)
    fitted = hist[0] + slope * np.arange(hist.size)
    return point, hist - fitted, np.arange(1, h + 1, dtype=float)


class BaselineAdapter(ModelAdapter):
    def __init__(self, info: ModelInfo, method: Callable) -> None:
        self.info = info
        self._method = method
        self._device: str | None = None

    def available_devices(self) -> list[DeviceInfo]:
        import psutil

        return [DeviceInfo("cpu", "cpu", cpu_name(), round(psutil.virtual_memory().total / 2**30, 1))]

    @property
    def device(self) -> str | None:
        return self._device

    def load(self, device: str = "cpu") -> dict:
        if device != "cpu":
            raise ValueError("baselines run on CPU only")
        self._device = "cpu"
        return {"load_s": 0.0, "revision": "n/a", "device": "cpu"}

    def unload(self) -> None:
        self._device = None

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        start = time.perf_counter()
        b, v, _ = request.context.shape
        h = request.horizon
        out = np.empty((b, v, h, len(LEVELS)))
        for i in range(b):
            for j in range(v):
                hist = _history(request.context[i, j])
                point, errors, steps = self._method(hist, h, request.seasonal_period)
                rq = _residual_quantiles(errors)  # (Q,), centred on the point forecast
                out[i, j] = point[:, None] + rq[None, :] * np.sqrt(steps)[:, None]
        out.sort(axis=-1)
        ms = (time.perf_counter() - start) * 1000
        return ForecastResult(out, LEVELS, "cpu", {"forecast_ms": round(ms, 3), "chunks": 1})


def _info(id_: str, name: str, description: str) -> ModelInfo:
    caps = Capabilities(multivariate=True, past_covariates=False, known_covariates=False,
                        categorical_covariates=False, max_context=None, max_horizon_per_pass=None,
                        quantile_levels=LEVELS, device_kinds=("cpu",))
    return ModelInfo(id=id_, family="baseline", name=name, params=None, description=description,
                     license="n/a (part of tsfm-lab)", license_url=None, source_url=None, capabilities=caps,
                     kind="baseline",
                     notes=("Covariates are ignored. Quantiles come from empirical in-sample errors, "
                            "widened with sqrt(steps ahead).",))


BASELINES = {
    "naive": ("Naive", "Repeats the last observed value.", _point_naive),
    "seasonal-naive": ("Seasonal naive", "Repeats the last full seasonal cycle (period from the data frequency).",
                       _point_seasonal),
    "drift": ("Drift", "Extends the straight line between the first and last observation.", _point_drift),
}


def register(registry) -> None:
    for id_, (name, desc, method) in BASELINES.items():
        info = _info(id_, name, desc)
        registry.add(info, lambda i=info, m=method: BaselineAdapter(i, m))
