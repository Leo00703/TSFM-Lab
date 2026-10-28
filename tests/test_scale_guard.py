"""Series with values far outside what float32 can square (a Bitcoin hash rate reaches 2e20) must not spoil the forecast, or the others in the batch."""

from __future__ import annotations

import numpy as np
import pytest

from tsfm_lab.models.base import (
    ForecastRequest,
    ForecastResult,
    rescale_extreme,
)
from tsfm_lab.models.torch_adapter import TorchAdapter


class Overflowing(TorchAdapter):
    """A stand-in for a model with float32 statistics: values it cannot square make the whole batch non-finite. Otherwise it repeats the last value."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[ForecastRequest] = []
        self.info = type("Info", (), {"capabilities": type("C", (), {"device_kinds": ("cpu",)})()})()

    def _load_on(self, device: str):
        self._model = object()
        return {}

    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        self.seen.append(request)
        last = np.nan_to_num(request.context[..., -1:], nan=0.0)
        q = np.repeat(last[..., None], request.horizon, axis=2).repeat(3, axis=3).astype(np.float32)  # (B, V, H, 3)
        if (np.abs(request.context) > 1e15).any():
            q[:] = np.nan  # the whole batch is lost, as it was for one model
        return ForecastResult(q, (0.1, 0.5, 0.9), "cpu", {}, [])


def loaded() -> Overflowing:
    a = Overflowing()
    a.load("cpu")
    return a


def test_a_series_beyond_float32_is_rescaled_and_forecast_back_in_its_own_units() -> None:
    ctx = np.stack([np.linspace(1.0, 2.0, 40), np.linspace(4.6e5, 1.98e20, 40)])[:, None, :]
    a = loaded()
    res = a.forecast(ForecastRequest(ctx, 5))
    assert np.isfinite(res.quantiles).all(), "the normal series is not spoiled by its neighbour"
    np.testing.assert_allclose(res.quantiles[0, 0, :, 1], 2.0)
    np.testing.assert_allclose(res.quantiles[1, 0, :, 1], 1.98e20, rtol=1e-6)
    assert np.abs(a.seen[0].context[1]).max() <= 1.0 + 1e-9 and np.array_equal(a.seen[0].context[0], ctx[0]) and any("float32" in w for w in res.warnings)


def test_series_inside_the_safe_range_are_not_touched_at_all() -> None:
    ctx = np.stack([np.linspace(1.0, 2.0, 40), np.linspace(1e-9, 3e12, 40)])[:, None, :]
    request = ForecastRequest(ctx, 5)
    scaled, factors, note = rescale_extreme(request)
    assert scaled is request and factors is None and note is None
    a = loaded()
    res = a.forecast(request)
    assert a.seen[0].context is ctx and not any("float32" in w for w in res.warnings)


def test_tiny_values_and_missing_values_and_extreme_covariates_are_handled() -> None:
    ctx = np.full((1, 1, 30), 3e-15)
    ctx[0, 0, :4] = np.nan
    cov = np.linspace(1e17, 1e20, 30 + 4)[None, None, :]
    scaled, factors, note = rescale_extreme(ForecastRequest(ctx, 4, known_covariates=cov))
    assert factors is not None and factors[0, 0] == pytest.approx(3e-15) and np.nanmax(np.abs(scaled.context)) == pytest.approx(1.0)
    assert np.abs(scaled.known_covariates).max() == pytest.approx(1.0) and "float32" in note
    only_cov, f2, n2 = rescale_extreme(ForecastRequest(np.ones((1, 1, 30)), 4, known_covariates=cov))
    assert f2 is None and "covariates" in n2 and np.abs(only_cov.known_covariates).max() == pytest.approx(1.0)
    all_nan, f3, _ = rescale_extreme(ForecastRequest(np.full((1, 1, 30), np.nan), 4))
    assert f3 is None and np.isnan(all_nan.context).all()


@pytest.mark.model
def test_the_real_models_forecast_a_hash_rate_sized_series_and_its_neighbours() -> None:
    from tsfm_lab.models import cache
    from tsfm_lab.models.registry import build_default_registry

    reg = build_default_registry()
    t = np.arange(400)
    normal = 100 + 10 * np.sin(2 * np.pi * t / 7)
    huge = 1e5 * np.exp(t * 0.11)  # grows to ~1e24
    ctx = np.stack([normal, huge])[:, None, :]
    checked = 0
    for model_id in ("chronos-2-small", "timesfm-2-5", "t0-gamma", "tirex-2"):
        if not cache.status(reg.info(model_id).details["repo"])["downloaded"]:
            continue
        a = reg.create(model_id)
        a.load("cpu")
        res = a.forecast(ForecastRequest(ctx, 14, seasonal_period=7))
        assert np.isfinite(res.quantiles).all(), model_id
        assert abs(res.median[0, 0, :7].mean() - 100) < 25, f"{model_id}: the normal series must not be spoiled by the huge one"
        assert 0.3 < res.median[1, 0, 0] / huge[-1] < 3, model_id
        a.unload()
        checked += 1
    if not checked:
        pytest.skip("no model with weights here")
