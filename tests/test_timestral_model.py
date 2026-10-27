"""Tests that load real Timestral-5M weights (cached after the first 21 MB download)."""

import numpy as np
import pytest

from tsfm_lab.data import synthetic
from tsfm_lab.eval.backtest import BacktestConfig
from tsfm_lab.eval.runner import run_experiment
from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.models.registry import build_default_registry
from tsfm_lab.system.devices import list_devices

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def model():
    adapter = build_default_registry().create("timestral-5m")
    meta = adapter.load("cpu")
    assert meta["device"] == "cpu" and meta["revision"]
    yield adapter
    adapter.unload()


def _ctx(batch=3, t=512):
    ds = synthetic.build("multi_seasonality", n_series=batch, length=t)
    return np.stack([s.target for s in ds.series])


def test_forecast_shape_levels_and_ordering(model) -> None:
    res = model.forecast(ForecastRequest(_ctx(), 48))
    assert res.quantiles.shape == (3, 1, 48, 21) and len(res.quantile_levels) == 21
    assert np.isfinite(res.quantiles).all() and (np.diff(res.quantiles, axis=-1) >= -1e-4).all()
    assert res.timings["forecast_ms"] > 0 and res.device == "cpu"
    assert model.info.details["num_heads"] and model.info.details["d_model"] == 192


def test_chunking_does_not_change_the_forecast(model) -> None:
    ctx = _ctx(batch=6)
    whole = model.forecast(ForecastRequest(ctx, 24, options={"max_packed_rows": 1000}))
    chunked = model.forecast(ForecastRequest(ctx, 24, options={"max_packed_rows": 1}))
    assert whole.timings["chunks"] == 1 and chunked.timings["chunks"] == 6
    np.testing.assert_allclose(whole.quantiles, chunked.quantiles, rtol=1e-4, atol=1e-4)


def test_long_horizon_is_flagged_and_still_finite(model) -> None:
    res = model.forecast(ForecastRequest(_ctx(batch=1), 1100))
    assert res.quantiles.shape[2] == 1100 and np.isfinite(res.quantiles).all()
    assert any("fed back as context" in w for w in res.warnings)


def test_flat_covariate_mitigation_changes_the_input_and_is_reported(model) -> None:
    ds = synthetic.build("flat_price_covariate", n_series=4, length=300)
    cfg = dict(horizon=28, n_windows=1, history_tail=64)
    plain, _ = run_experiment("timestral-5m", ds, BacktestConfig(**cfg), baselines=())
    masked, _ = run_experiment("timestral-5m", ds, BacktestConfig(**cfg, options={"mask_flat_covariates": True}),
                               baselines=())
    p, m = plain.models["timestral-5m"], masked.models["timestral-5m"]
    assert any("issue #7" in w for w in p.warnings)
    assert any("mask_flat_covariates applied" in w for w in m.warnings)
    assert np.isfinite(p.quantiles).all() and np.isfinite(m.quantiles).all()


def test_timestral_beats_naive_on_seasonal_data() -> None:
    ds = synthetic.build("multi_seasonality", n_series=4, length=500)
    result, _ = run_experiment("timestral-5m", ds, BacktestConfig(horizon=48, n_windows=2), baselines=("naive",))
    assert result.models["timestral-5m"].skill["naive"]["mase"]["skill"] > 0.3


@pytest.mark.skipif(not any(d.kind == "cuda" and d.supported for d in list_devices()), reason="no CUDA device")
def test_cuda_matches_cpu_and_reports_vram(model) -> None:
    gpu = build_default_registry().create("timestral-5m")
    gpu.load("cuda")
    ctx = _ctx()
    cpu_res, gpu_res = model.forecast(ForecastRequest(ctx, 48)), gpu.forecast(ForecastRequest(ctx, 48))
    np.testing.assert_allclose(cpu_res.quantiles, gpu_res.quantiles, rtol=1e-3, atol=1e-3)
    assert gpu_res.device == "cuda:0" and gpu_res.timings["peak_vram_mb"] > 0
    gpu.unload()


def test_strict_mode_rejects_an_unavailable_device_with_a_reason() -> None:
    adapter = build_default_registry().create("timestral-5m")
    adapter.strict_device = True
    present = {d.id for d in list_devices() if d.supported}
    with pytest.raises(ValueError, match="unavailable|not supported"):
        adapter.load("mps" if "mps" not in present else "cuda:7")


def test_asking_for_an_unavailable_gpu_runs_on_the_cpu_and_says_so() -> None:
    adapter = build_default_registry().create("timestral-5m")
    try:
        gone = next(k for k in ("mps", "cuda:0") if k not in {d.id for d in list_devices() if d.supported})
        meta = adapter.load(gone)
        assert meta["device"] == "cpu" and meta["requested_device"] == gone and gone in meta["fallback"]
        res = adapter.forecast(ForecastRequest(_ctx(), 24))
        assert res.device == "cpu" and np.isfinite(res.quantiles).all() and gone in res.warnings[0]
        adapter.strict_device = True  # benchmarks must not do this
        with pytest.raises(ValueError):
            adapter.load(gone)
    finally:
        adapter.unload()


def test_a_gpu_that_fails_while_forecasting_continues_on_the_cpu(model, monkeypatch) -> None:
    adapter = build_default_registry().create("timestral-5m")
    adapter.load("cpu")
    adapter._device = "mps"  # pretend the model sits on a GPU whose kernels fail
    real = adapter._forecast_once
    calls = []

    def flaky(request):
        calls.append(adapter.device)
        if adapter.device != "cpu":
            raise RuntimeError("MPS backend out of memory")
        return real(request)

    monkeypatch.setattr(adapter, "_forecast_once", flaky)
    try:
        res = adapter.forecast(ForecastRequest(_ctx(), 24))
        assert calls == ["mps", "cpu"] and res.device == "cpu" and np.isfinite(res.quantiles).all()
        assert "mps failed while forecasting" in res.warnings[0] and "continued on the CPU" in res.warnings[0]
        again = adapter.forecast(ForecastRequest(_ctx(), 24))  # stays on the CPU, still explains why
        assert again.device == "cpu" and "continued on the CPU" in again.warnings[0]
    finally:
        adapter.unload()


def test_a_real_error_on_the_cpu_is_not_swallowed(model, monkeypatch) -> None:
    def broken(request):
        raise RuntimeError("genuine bug")

    monkeypatch.setattr(model, "_forecast_once", broken)
    with pytest.raises(RuntimeError, match="genuine bug"):
        model.forecast(ForecastRequest(_ctx(), 24))
