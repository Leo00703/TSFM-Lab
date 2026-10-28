"""TiRex-2: manifest and cautions, the translation of a request into the package's native input, and a real forecast when the
checkpoint is on this machine."""

from __future__ import annotations

import numpy as np
import pytest

from tsfm_lab.models import cache, tirex
from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.models.registry import build_default_registry

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_the_manifest_states_the_licence_situation_and_the_ranking_honestly() -> None:
    (manifest,) = tirex.load_manifests()
    info = tirex.info_from_manifest(manifest)
    assert info.id == "tirex-2" and info.family == "tirex" and info.license == "Apache-2.0" and info.kind == "foundation"
    c = info.capabilities
    assert list(c.quantile_levels) == LEVELS and c.known_covariates and c.past_covariates and c.multivariate
    assert not c.categorical_covariates and c.max_context == 2048 and c.max_horizon_per_pass == 320
    joined = " ".join(info.notes)
    assert "NXAI Community Licence" in joined and "EUR 100 million" in joined  # a dependency has a different licence
    assert "ranks below" in joined and "pure-PyTorch kernels" in joined
    assert "tirex-2" in build_default_registry().ids()


class FakeForecast:
    """Records the native inputs and answers with a monotone fan of nine quantiles per variable."""

    def __init__(self, crossing: bool = False) -> None:
        self.calls: list[dict] = []
        self.crossing = crossing

    def forecast(self, series, prediction_length, output_type, batch_size):
        self.calls.append({"series": series, "h": prediction_length, "type": output_type, "batch": batch_size})
        out = []
        for s in series:
            q = np.linspace(0, 1, 9, dtype=np.float32)[None, :, None].repeat(s.target.shape[0], 0).repeat(prediction_length, 2)
            out.append(q[:, ::-1].copy() if self.crossing else q)
        return out


def adapter_with(fake: FakeForecast):
    adapter = build_default_registry().create("tirex-2")
    adapter._model, adapter._device = fake, "cpu"
    return adapter


def test_each_item_becomes_one_group_of_variables_with_its_covariates_in_the_native_layout() -> None:
    fake = FakeForecast()
    ctx = np.random.default_rng(0).normal(size=(3, 2, 100))
    past = np.ones((3, 1, 100))
    known = np.arange(3 * 2 * 124, dtype=float).reshape(3, 2, 124)
    res = adapter_with(fake).forecast(ForecastRequest(ctx, 24, past_covariates=past, known_covariates=known))
    call = fake.calls[0]
    assert call["h"] == 24 and call["type"] == "numpy" and len(call["series"]) == 3  # one native series per item
    s = call["series"][0]
    assert tuple(s.target.shape) == (2, 100) and tuple(s.past_covariates.shape) == (1, 100) and tuple(s.future_covariates.shape) == (2, 124)
    assert res.quantiles.shape == (3, 2, 24, 9) and tuple(res.quantile_levels) == tuple(LEVELS)
    assert any("2 variables" in w for w in res.warnings)
    assert (np.diff(res.quantiles, axis=-1) >= 0).all()


def test_long_context_is_capped_gaps_are_kept_and_crossing_quantiles_are_put_in_order() -> None:
    fake = FakeForecast(crossing=True)
    ctx = np.random.default_rng(1).normal(size=(1, 1, 3000))
    ctx[0, 0, 2500] = np.nan  # a gap stays a gap: the model masks it itself
    known = np.zeros((1, 1, 3012))
    known[..., 3005:] = np.nan
    res = adapter_with(fake).forecast(ForecastRequest(ctx, 12, known_covariates=known))
    s = fake.calls[0]["series"][0]
    assert s.target.shape[-1] == 2048 and bool(np.isnan(s.target.numpy()).any()) and s.future_covariates.shape[-1] == 2048 + 12
    assert (np.diff(res.quantiles, axis=-1) >= 0).all()
    assert any("context longer than 2048" in w for w in res.warnings) and any("crossing quantiles" in w for w in res.warnings)
    assert any("missing over the forecast region" in w for w in res.warnings)


class CuttingForecast(FakeForecast):
    """Like the package: a request beyond 320 steps is silently cut to 320. The median of each answer is the step number it started at."""

    def forecast(self, series, prediction_length, output_type, batch_size):
        self.calls.append({"series": series, "h": prediction_length})
        h = min(prediction_length, 320)
        base = float(len(self.calls))
        return [np.full((s.target.shape[0], 9, h), base, dtype=np.float32) + np.linspace(0, 0.8, 9, dtype=np.float32)[None, :, None] for s in series]


def test_a_horizon_beyond_one_pass_is_forecast_in_passes_and_not_silently_cut() -> None:
    fake = CuttingForecast()
    ctx = np.random.default_rng(2).normal(size=(2, 1, 100))
    past = np.ones((2, 1, 100))
    known = np.zeros((2, 1, 100 + 700))
    res = adapter_with(fake).forecast(ForecastRequest(ctx, 700, past_covariates=past, known_covariates=known))
    assert res.quantiles.shape == (2, 1, 700, 9), "the whole horizon comes back, not the package's 320"
    assert [c["h"] for c in fake.calls] == [320, 320, 60]
    lengths = [(c["series"][0].target.shape[-1], c["series"][0].future_covariates.shape[-1]) for c in fake.calls]
    assert lengths == [(100, 420), (420, 740), (740, 800)], "each pass sees the earlier forecast as history and the known covariates that cover it"
    second, third = fake.calls[1]["series"][0], fake.calls[2]["series"][0]
    assert np.allclose(second.target.numpy()[0, 100:], 1.4)  # the median (index 4) of the first pass, fed back
    assert np.isnan(second.past_covariates.numpy()[0, 100:]).all() and np.isfinite(second.past_covariates.numpy()[0, :100]).all()
    assert third.target.shape[-1] == 740 and any("several passes" in w for w in res.warnings)


def test_the_history_handed_to_a_later_pass_stays_within_the_context_limit() -> None:
    fake = CuttingForecast()
    ctx = np.random.default_rng(3).normal(size=(1, 1, 2000))
    adapter_with(fake).forecast(ForecastRequest(ctx, 700))
    assert [c["series"][0].target.shape[-1] for c in fake.calls] == [2000, 2048, 2048]


def test_loading_without_weights_says_where_to_get_them(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "hub_dir", lambda: tmp_path)
    adapter = build_default_registry().create("tirex-2")
    adapter.strict_device = True
    with pytest.raises(FileNotFoundError, match="download its weights on the Models page"):
        adapter.load("cpu")


@pytest.mark.model
def test_the_real_checkpoint_continues_a_clean_cycle_on_the_cpu_kernels_when_it_is_present() -> None:
    adapter = build_default_registry().create("tirex-2")
    if not cache.status(adapter.info.details["repo"])["downloaded"]:
        pytest.skip("TiRex-2 (330 MB) is not downloaded here")
    adapter.load("cpu")
    t = np.arange(512)
    ctx = (10 + np.sin(2 * np.pi * t / 24))[None, None, :].repeat(2, 0)
    known = np.sin(2 * np.pi * np.arange(560) / 24)[None, None, :].repeat(2, 0)
    res = adapter.forecast(ForecastRequest(ctx, 48, known_covariates=known))
    assert res.quantiles.shape == (2, 1, 48, 9) and np.isfinite(res.quantiles).all()
    assert (np.diff(res.quantiles, axis=-1) >= -1e-6).all()
    assert np.abs(res.quantiles[0, 0, :, 4] - (10 + np.sin(2 * np.pi * np.arange(512, 560) / 24))).max() < 0.5
    adapter.unload()
