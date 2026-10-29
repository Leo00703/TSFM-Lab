"""TimesFM 2.5 and 3.0: manifests and licences, the translation of a ForecastRequest for each package, scoring models that
predict different quantile levels, and (when the checkpoints are on this machine) real forecasts."""

from __future__ import annotations

import numpy as np
import pytest

from tsfm_lab.data import synthetic
from tsfm_lab.eval import metrics as M
from tsfm_lab.eval.backtest import BacktestConfig, run_backtest
from tsfm_lab.models import cache, timesfm
from tsfm_lab.models.base import (
    Capabilities,
    ForecastRequest,
    ForecastResult,
    ModelAdapter,
    ModelInfo,
)
from tsfm_lab.models.registry import build_default_registry

DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_manifests_state_the_licences_and_capabilities_honestly() -> None:
    manifests = timesfm.load_manifests()
    assert [m["id"] for m in manifests] == ["timesfm-3", "timesfm-2-5"]  # largest first
    v3, v25 = (timesfm.info_from_manifest(m) for m in manifests)
    for info in (v3, v25):
        assert info.family == "timesfm" and info.kind == "foundation" and list(info.capabilities.quantile_levels) == DECILES
    assert v3.capabilities.known_covariates and v3.capabilities.past_covariates and v3.capabilities.max_context == 15360
    assert not v25.capabilities.known_covariates and not v25.capabilities.past_covariates and v25.capabilities.max_context == 16384
    # TimesFM 3.0's weights are non-commercial: that must be impossible to miss, and 2.5 must not claim it
    assert v3.details["restriction"] == "non-commercial" and "not permitted" in v3.details["restriction_note"]
    assert "Non-Commercial" in v3.license and v3.notes[0] == v3.details["restriction_note"]
    assert "restriction" not in v25.details and v25.license == "Apache-2.0"
    assert any("Covariates are ignored" in n for n in v25.notes)
    registry = build_default_registry()
    assert {"timesfm-3", "timesfm-2-5"} <= set(registry.ids())


class FakeV3:
    """Stands in for TimesFM3Evaluator: records the call and answers with deciles."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.config = type("C", (), {"per_core_batch_size": 16})()

    def predict_batch(self, **kw):
        self.calls.append(kw)
        for ctx in kw["contexts"]:
            v = ctx.shape[0]
            q = np.broadcast_to(np.linspace(0, 1, 9, dtype=np.float32)[None, None, :], (v, kw["horizon"], 9)).copy()
            yield type("O", (), {"quantiles": q, "forecast": q[..., 4]})()


def adapter_v3(fake: FakeV3):
    adapter = build_default_registry().create("timesfm-3")
    adapter._model, adapter._device = fake, "cpu"
    return adapter


def test_version_3_gets_the_past_and_future_parts_of_the_covariates() -> None:
    fake = FakeV3()
    ctx = np.random.default_rng(1).normal(size=(2, 2, 100))
    past = np.ones((2, 1, 100))
    known = np.arange(2 * 3 * 124, dtype=float).reshape(2, 3, 124)
    res = adapter_v3(fake).forecast(ForecastRequest(ctx, 24, past_covariates=past, known_covariates=known,
                                                    options={"symmetric_averaging": False, "clip_nonnegative": False}))
    call = fake.calls[0]
    assert call["horizon"] == 24 and call["padding_mode"] == "edge" and call["return_quantiles"] is True
    assert call["contexts"][0].shape == (2, 100)  # variables stay together
    assert call["past_only_covariates"][1].shape == (1, 100)
    assert call["past_future_covariates"][1].shape == (3, 124)  # context part plus the horizon part
    assert call["use_symmetric_averaging"] is False and call["make_positive"] is False  # Timestral's options mapped over
    assert res.quantiles.shape == (2, 2, 24, 9) and res.device == "cpu"
    benchmark_defaults = adapter_v3(FakeV3())
    benchmark_defaults.forecast(ForecastRequest(ctx, 8))
    assert (benchmark_defaults._model.calls[0]["use_symmetric_averaging"], benchmark_defaults._model.calls[0]["make_positive"]) == (True, True)


def test_version_3_keeps_only_the_most_recent_context_and_cuts_covariates_the_same_way() -> None:
    fake = FakeV3()
    length = 16000
    known = np.arange(length + 4, dtype=float)[None, None, :]
    adapter_v3(fake).forecast(ForecastRequest(np.zeros((1, 1, length)), 4, known_covariates=known))
    call = fake.calls[0]
    assert call["contexts"][0].shape[1] == 15360 and call["past_future_covariates"][0].shape[1] == 15360 + 4
    assert call["past_future_covariates"][0][0, -1] == length + 3  # the end of the series, not the start


class FakeV25:
    def __init__(self) -> None:
        self.compiled: list = []
        self.global_batch_size = 8

    def compile(self, config):
        self.compiled.append(config)

    def forecast(self, horizon, inputs):
        n = len(inputs)
        quantiles = np.zeros((n, horizon, 10), dtype=np.float32)
        quantiles[..., 0] = -999.0  # the mean column, which the adapter must drop
        quantiles[..., 1:] = np.linspace(0, 1, 9, dtype=np.float32)
        return np.zeros((n, horizon), dtype=np.float32), quantiles


def test_version_2_5_drops_the_mean_column_and_says_what_it_ignores() -> None:
    fake = FakeV25()
    adapter = build_default_registry().create("timesfm-2-5")
    adapter._model, adapter._device = fake, "cpu"
    ctx = np.random.default_rng(0).normal(size=(3, 2, 200))
    res = adapter.forecast(ForecastRequest(ctx, 48, known_covariates=np.zeros((3, 1, 248))))
    assert res.quantiles.shape == (3, 2, 48, 9) and (res.quantiles > -1).all()  # no -999 from the mean column
    assert any("covariates ignored" in w for w in res.warnings) and any("2 variables" in w for w in res.warnings)
    config = fake.compiled[0]
    assert config.max_horizon == 128 and config.max_context == 224 and config.normalize_inputs and config.use_continuous_quantile_head
    adapter.forecast(ForecastRequest(ctx, 48))  # same shape: no recompilation
    assert len(fake.compiled) == 1
    adapter.forecast(ForecastRequest(ctx, 300))  # a longer horizon needs a new decode function
    assert len(fake.compiled) == 2 and fake.compiled[1].max_horizon == 384


def test_version_2_5_survives_a_context_with_nothing_observed() -> None:
    seen = []

    class Recording(FakeV25):
        def forecast(self, horizon, inputs):
            seen.extend(np.array(x) for x in inputs)
            return super().forecast(horizon, inputs)

    adapter = build_default_registry().create("timesfm-2-5")
    adapter._model, adapter._device = Recording(), "cpu"
    ctx = np.random.default_rng(1).normal(size=(3, 1, 64))
    ctx[1] = np.nan  # a series that is all missing in this window (the package would crash on it)
    res = adapter.forecast(ForecastRequest(ctx, 8))
    assert res.quantiles.shape == (3, 1, 8, 9) and (seen[1] == 0).all() and np.isfinite(seen[0]).all()
    assert any("1 context(s) had no observed value" in w for w in res.warnings)


def test_loading_without_weights_says_where_to_get_them(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "hub_dir", lambda: tmp_path)
    for model in ("timesfm-3", "timesfm-2-5"):
        adapter = build_default_registry().create(model)
        adapter.strict_device = True
        with pytest.raises(FileNotFoundError, match="download its weights on the Models page"):
            adapter.load("cpu")


class LevelsModel(ModelAdapter):
    """A model with chosen quantile levels: the median is the last value, the spread grows with the distance from 0.5."""

    def __init__(self, name: str, levels: list[float]) -> None:
        caps = Capabilities(multivariate=True, past_covariates=False, known_covariates=False, categorical_covariates=False,
                            max_context=None, max_horizon_per_pass=None, quantile_levels=tuple(levels), device_kinds=("cpu",))
        self.info = ModelInfo(id=name, family="fake", name=name, params=0, description="", license="", license_url="",
                              source_url="", capabilities=caps, details={}, notes=(), kind="baseline")
        self._levels = levels

    def available_devices(self):
        return []

    def load(self, device="cpu"):
        return {"load_s": 0.0, "device": "cpu"}

    def unload(self):
        pass

    @property
    def device(self):
        return "cpu"

    def forecast(self, request):
        last = request.context[:, :, -1][..., None, None]  # (B, V, 1, 1)
        z = np.asarray([lv - 0.5 for lv in self._levels], dtype=float) * 10
        q = np.broadcast_to(last + z, (request.batch, request.context.shape[1], request.horizon, len(self._levels))).copy()
        return ForecastResult(q, tuple(self._levels), "cpu", {"forecast_ms": 1.0}, [])


def test_models_with_different_levels_are_scored_on_the_levels_they_share() -> None:
    wide = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
    ds = synthetic.build("trend", n_series=3, length=200)
    res = run_backtest(ds, [LevelsModel("wide", wide), LevelsModel("deciles", DECILES)], BacktestConfig(horizon=12, n_windows=2))
    shared = [0.1, 0.4, 0.5, 0.6, 0.9]  # the levels in both lists
    a, b = res.models["wide"], res.models["deciles"]
    assert (a.items["sql"].shape == b.items["sql"].shape) and "sql_own_levels" in a.items and "sql_own_levels" in b.items
    expected = M.item_metrics(res.y, a.quantiles[..., [wide.index(lv) for lv in shared]], tuple(shared), res.scale)["sql"]
    np.testing.assert_allclose(a.items["sql"], expected)
    assert not np.allclose(a.items["sql_own_levels"], a.items["sql"], equal_nan=True)  # the grids really differ
    assert all("shared by every model" in n for out in (a, b) for n in out.notes[-1:])
    assert a.skill["deciles"]["sql"] is not None
    same = run_backtest(ds, [LevelsModel("one", DECILES), LevelsModel("two", DECILES)], BacktestConfig(horizon=12, n_windows=2))
    assert all("sql_own_levels" not in out.items and not out.notes for out in same.models.values())  # nothing changes when levels agree


@pytest.mark.model
@pytest.mark.parametrize("model", ["timesfm-3", "timesfm-2-5"])
def test_the_real_checkpoints_continue_a_clean_cycle_when_they_are_present(model) -> None:
    adapter = build_default_registry().create(model)
    if not cache.status(adapter.info.details["repo"])["downloaded"]:
        pytest.skip(f"{adapter.info.name} is not downloaded here")
    adapter.load("cpu")
    t = np.arange(512)
    ctx = (10 + np.sin(2 * np.pi * t / 24))[None, None, :].repeat(2, 0)
    res = adapter.forecast(ForecastRequest(ctx, 48))
    assert res.quantiles.shape == (2, 1, 48, 9) and np.isfinite(res.quantiles).all()
    assert (np.diff(res.quantiles, axis=-1) >= -1e-6).all()
    assert np.abs(res.quantiles[0, 0, :, 4] - (10 + np.sin(2 * np.pi * np.arange(512, 560) / 24))).max() < 0.5
    adapter.unload()


def test_only_timesfm_3_offers_the_apple_gpu_and_2_5_falls_back_to_the_cpu() -> None:
    """TimesFM 2.5's package moves float64 arrays to the GPU, which Metal lacks (found on a MacBook Air M5): the Mac runs it on the CPU."""
    from tsfm_lab.system.devices import pick_device

    infos = {m["id"]: timesfm.info_from_manifest(m) for m in timesfm.load_manifests()}
    assert "mps" in infos["timesfm-3"].capabilities.device_kinds
    assert "mps" not in infos["timesfm-2-5"].capabilities.device_kinds and "cuda" in infos["timesfm-2-5"].capabilities.device_kinds
    device, note = pick_device("mps", infos["timesfm-2-5"].capabilities.device_kinds)
    assert device == "cpu" and "not supported" in note
    assert any("MPS" in n for n in infos["timesfm-2-5"].notes) and any("MPS" in n for n in infos["timesfm-3"].notes)
