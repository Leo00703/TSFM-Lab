import numpy as np
import pytest

from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.models.covariates import flat_context_rows, inspect_known_covariates, mask_flat_rows
from tsfm_lab.models.registry import build_default_registry


def _req(context, horizon=6, **kw):
    return ForecastRequest(np.asarray(context, float).reshape(1, 1, -1), horizon, **kw)


def _run(model_id, req):
    a = build_default_registry().create(model_id)
    a.load("cpu")
    return a.forecast(req)


def test_naive_repeats_last_value_and_quantiles_are_sorted() -> None:
    res = _run("naive", _req(np.arange(30.0) + np.random.default_rng(0).normal(0, .3, 30)))
    assert res.quantiles.shape == (1, 1, 6, 21)
    assert (np.diff(res.quantiles, axis=-1) >= 0).all()
    np.testing.assert_allclose(res.median[0, 0], res.median[0, 0, 0], atol=1e-9)  # flat median


def test_seasonal_naive_repeats_the_cycle() -> None:
    cycle = np.array([1.0, 5.0, 3.0, 9.0])
    res = _run("seasonal-naive", _req(np.tile(cycle, 10), horizon=8, seasonal_period=4))
    np.testing.assert_allclose(res.median[0, 0], np.tile(cycle, 2), atol=1e-9)


def test_drift_extends_the_line() -> None:
    res = _run("drift", _req(np.arange(20.0), horizon=3))
    np.testing.assert_allclose(res.median[0, 0], [20.0, 21.0, 22.0], atol=1e-9)


def test_baselines_survive_nans_and_all_missing() -> None:
    ctx = np.arange(30.0)
    ctx[:5] = np.nan
    ctx[10:14] = np.nan
    assert np.isfinite(_run("seasonal-naive", _req(ctx, seasonal_period=7)).quantiles).all()
    assert np.isfinite(_run("naive", _req(np.full(10, np.nan))).quantiles).all()


def test_baselines_reject_gpu_and_unknown_models() -> None:
    with pytest.raises(ValueError):
        build_default_registry().create("naive").load("cuda")
    with pytest.raises(KeyError, match="unknown model"):
        build_default_registry().create("nope")


def test_request_validates_shapes() -> None:
    with pytest.raises(ValueError):
        ForecastRequest(np.zeros((5, 10)), 3)
    with pytest.raises(ValueError, match="known_covariates"):
        ForecastRequest(np.zeros((1, 1, 10)), 3, known_covariates=np.zeros((1, 1, 10)))  # must be T + horizon
    with pytest.raises(ValueError):
        ForecastRequest(np.zeros((1, 1, 10)), 0)


def test_flat_covariate_detection_and_masking() -> None:
    ctx_len, h = 50, 5
    price = np.full((1, 2, ctx_len + h), 2.5)
    price[0, 0, -2:] = 2.6  # flat in context, changes in horizon -> issue #7 trigger
    price[0, 1, :] = np.linspace(1, 2, ctx_len + h)  # varying: fine
    flat = flat_context_rows(price, ctx_len, None)
    assert flat.tolist() == [[True, False]]
    warns, flagged = inspect_known_covariates(price, ctx_len, None, symmetric_averaging=True)
    assert any("issue #7" in w for w in warns) and any("never change sign" in w for w in warns)
    masked = mask_flat_rows(price, flagged)
    assert np.isnan(masked[0, 0]).all() and np.isfinite(masked[0, 1]).all() and np.isfinite(price).all()


def test_binary_covariates_do_not_trigger_the_sign_warning() -> None:
    known = np.zeros((1, 1, 30))
    known[0, 0, ::3] = 1
    warns, _ = inspect_known_covariates(known, 25, None, symmetric_averaging=True)
    assert warns == []


# --- the extensibility promise: a new model needs only an adapter + registration -----------------
def test_new_model_plugs_in_without_touching_the_core() -> None:
    from tsfm_lab.data import synthetic
    from tsfm_lab.eval.backtest import BacktestConfig, run_backtest
    from tsfm_lab.models.base import Capabilities, DeviceInfo, ForecastResult, ModelAdapter, ModelInfo
    from tsfm_lab.models.registry import Registry

    class MeanModel(ModelAdapter):
        info = ModelInfo("mean-model", "toy", "Mean", None, "toy", "n/a", None, None,
                         Capabilities(True, False, False, False, None, None, (0.1, 0.5, 0.9), ("cpu",)))
        _dev = None

        def available_devices(self): return [DeviceInfo("cpu", "cpu", "cpu", None)]
        @property
        def device(self): return self._dev
        def load(self, device="cpu"): self._dev = device; return {"load_s": 0.0}
        def unload(self): self._dev = None

        def forecast(self, r):
            m = np.nanmean(r.context, axis=-1)[..., None, None]
            q = np.repeat(m, r.horizon, axis=2) + np.array([-1.0, 0.0, 1.0])
            return ForecastResult(q, (0.1, 0.5, 0.9), "cpu", {"forecast_ms": 1.0})

    registry = Registry()
    registry.add(MeanModel.info, MeanModel)
    assert registry.ids() == ["mean-model"]
    ds = synthetic.build("trend", n_series=3, length=200)
    with registry.create("mean-model") as model:
        model.load()
        res = run_backtest(ds, [model], BacktestConfig(horizon=10, n_windows=2))
    assert res.models["mean-model"].summary["mase"]["n"] == 6
    with pytest.raises(ValueError, match="already registered"):
        registry.add(MeanModel.info, MeanModel)


def test_mps_is_offered_only_when_timestral_can_run_its_float64_statistics_on_the_cpu(monkeypatch) -> None:
    import sys
    import types

    import torch

    from tsfm_lab.system import devices

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    plain = types.ModuleType("timestral.scaler")  # the official package: no cpu_for_float64
    patched = types.ModuleType("timestral.scaler")
    patched.cpu_for_float64 = lambda *t: t
    pkg = types.ModuleType("timestral")

    def listed(scaler):
        monkeypatch.setitem(sys.modules, "timestral", pkg)
        monkeypatch.setitem(sys.modules, "timestral.scaler", scaler)
        pkg.scaler = scaler
        return {d.id: d for d in devices.list_devices()}["mps"]

    off = listed(plain)
    assert not off.supported and "float64" in off.reason
    assert listed(patched).supported


def test_a_missing_gpu_falls_back_to_the_cpu_unless_the_caller_is_strict(monkeypatch) -> None:
    from tsfm_lab.models.base import DeviceInfo
    from tsfm_lab.system import devices

    monkeypatch.setattr(devices, "list_devices", lambda: [DeviceInfo("cpu", "cpu", "CPU", 8.0)])
    assert devices.pick_device("cpu", ("cpu", "cuda", "mps")) == ("cpu", None)
    for gpu in ("cuda", "cuda:0", "mps"):
        device, note = devices.pick_device(gpu, ("cpu", "cuda", "mps"))
        assert device == "cpu" and "CPU" in note and "unavailable" in note
        with pytest.raises(ValueError):
            devices.pick_device(gpu, ("cpu", "cuda", "mps"), strict=True)
    device, note = devices.pick_device("mps", ("cpu", "cuda"))  # a model that cannot use that kind of device at all
    assert device == "cpu" and "not supported" in note


def test_a_broken_cuda_driver_never_hides_the_cpu(monkeypatch) -> None:
    import torch

    from tsfm_lab.system import devices

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: (_ for _ in ()).throw(RuntimeError("driver exploded")))
    listed = {d.id: d for d in devices.list_devices()}
    assert listed["cpu"].supported and not listed["cuda:0"].supported and "RuntimeError" in listed["cuda:0"].reason
