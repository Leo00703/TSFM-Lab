"""Chronos-2: manifests, the translation of a ForecastRequest into the package's input format, and (when the small
checkpoint is on this machine) a real forecast. The package itself is not mocked away: its import is part of the check."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsfm_lab.models import cache, chronos
from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.models.registry import build_default_registry

LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def test_the_manifests_describe_two_apache_licensed_models_on_the_same_quantile_grid() -> None:
    manifests = chronos.load_manifests()
    assert [m["id"] for m in manifests] == ["chronos-2", "chronos-2-small"]  # largest first, like everywhere else
    for m in manifests:
        info = chronos.info_from_manifest(m)
        assert info.family == "chronos" and info.license == "Apache-2.0" and info.kind == "foundation"
        assert list(info.capabilities.quantile_levels) == LEVELS  # the grid Timestral uses, so metrics are comparable
        c = info.capabilities
        assert c.multivariate and c.past_covariates and c.known_covariates and not c.categorical_covariates
        assert c.max_context == 8192 and c.max_horizon_per_pass == 1024 and "cpu" in c.device_kinds
    registry = build_default_registry()
    assert {"chronos-2", "chronos-2-small", "timestral-5m"} <= set(registry.ids())
    foundation = [i.id for i in registry.infos() if i.kind == "foundation"]
    assert foundation.index("timestral-5m") < foundation.index("chronos-2")  # family order, each largest first


class FakePipeline:
    """Records what the adapter hands to Chronos-2 and answers like predict_quantiles does."""

    def __init__(self, trained=tuple(LEVELS), cross=False) -> None:
        self.quantiles = list(trained)
        self.calls: list[dict] = []
        self.cross = cross

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, batch_size, context_length):
        self.calls.append({"inputs": inputs, "h": prediction_length, "levels": quantile_levels, "batch": batch_size, "ctx": context_length})
        out = []
        for item in inputs:
            v = item["target"].shape[0]
            base = np.linspace(0, 1, len(quantile_levels), dtype=np.float32)
            q = np.broadcast_to(base[None, None, :], (v, prediction_length, len(quantile_levels))).copy()
            if self.cross:
                q[..., 3], q[..., 4] = q[..., 4].copy(), q[..., 3].copy()  # two levels cross
            out.append(torch.from_numpy(q))
        return out, [None] * len(inputs)


def adapter_with(pipe: FakePipeline, model: str = "chronos-2"):
    adapter = build_default_registry().create(model)
    adapter._model, adapter._device = pipe, "cpu"
    return adapter


def test_covariates_are_split_into_a_past_part_and_a_future_part() -> None:
    pipe = FakePipeline()
    ctx = np.random.default_rng(0).normal(size=(3, 2, 100))
    past = np.ones((3, 1, 100))
    known = np.arange(3 * 2 * 124, dtype=float).reshape(3, 2, 124)
    res = adapter_with(pipe).forecast(ForecastRequest(ctx, 24, past_covariates=past, known_covariates=known))
    call = pipe.calls[0]
    item = call["inputs"][1]
    assert call["h"] == 24 and call["levels"] == LEVELS and call["ctx"] == 100 and len(call["inputs"]) == 3
    assert item["target"].shape == (2, 100)  # a multivariate target keeps its variables together
    assert set(item["past_covariates"]) == {"past_0", "known_0", "known_1"} and set(item["future_covariates"]) == {"known_0", "known_1"}
    assert item["past_covariates"]["known_1"].shape == (100,) and item["future_covariates"]["known_1"].shape == (24,)
    np.testing.assert_array_equal(item["past_covariates"]["known_1"], known[1, 1, :100])  # context part
    np.testing.assert_array_equal(item["future_covariates"]["known_1"], known[1, 1, 100:])  # the part after the origin
    assert "past_0" not in item["future_covariates"]  # a past-only covariate has no future values
    assert res.quantiles.shape == (3, 2, 24, 21) and res.device == "cpu" and res.warnings == []


def test_without_covariates_only_the_target_is_sent_and_a_long_context_is_capped() -> None:
    pipe = FakePipeline()
    res = adapter_with(pipe).forecast(ForecastRequest(np.zeros((1, 1, 9000)), 12))
    item = pipe.calls[0]["inputs"][0]
    assert set(item) == {"target"} and pipe.calls[0]["ctx"] == 8192 and res.quantiles.shape == (1, 1, 12, 21)


def test_crossing_quantiles_are_sorted_and_the_small_model_says_it_interpolates() -> None:
    small_levels = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    res = adapter_with(FakePipeline(trained=small_levels, cross=True), "chronos-2-small").forecast(
        ForecastRequest(np.zeros((2, 1, 50)), 8))
    assert (np.diff(res.quantiles, axis=-1) >= 0).all() and res.timings["crossings_fixed"] > 0
    assert any("13 quantile levels" in w and "interpolated" in w for w in res.warnings)
    long = adapter_with(FakePipeline()).forecast(ForecastRequest(np.zeros((1, 1, 50)), 2000))
    assert any("unrolled autoregressively" in w for w in long.warnings)


def test_loading_without_downloaded_weights_says_where_to_get_them(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "hub_dir", lambda: tmp_path)
    adapter = build_default_registry().create("chronos-2-small")
    adapter.strict_device = True
    with pytest.raises(FileNotFoundError, match="download its weights on the Models page"):
        adapter.load("cpu")


@pytest.mark.model
def test_the_real_small_checkpoint_forecasts_on_the_cpu_when_it_is_present() -> None:
    adapter = build_default_registry().create("chronos-2-small")
    if not cache.status(adapter.info.details["repo"])["downloaded"]:
        pytest.skip("Chronos-2 Small (112 MB) is not downloaded here")
    meta = adapter.load("cpu")
    t = np.arange(400)
    ctx = (10 + np.sin(2 * np.pi * t / 24))[None, None, :].repeat(2, 0)
    known = np.sin(2 * np.pi * np.arange(448) / 24)[None, None, :].repeat(2, 0)
    res = adapter.forecast(ForecastRequest(ctx, 48, known_covariates=known))
    assert meta["revision"] and res.quantiles.shape == (2, 1, 48, 21) and np.isfinite(res.quantiles).all()
    assert (np.diff(res.quantiles, axis=-1) >= -1e-6).all()
    median = res.quantiles[0, 0, :, 10]  # a clean daily cycle must be continued
    assert np.abs(median - (10 + np.sin(2 * np.pi * np.arange(400, 448) / 24))).max() < 0.5
    adapter.unload()
