"""t0-gamma: manifest and cautions, the translation of a request into the package's native input (the authors' construction),
and a real forecast when the checkpoint is on this machine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsfm_lab.models import cache, t0
from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.models.registry import build_default_registry

LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def test_the_manifest_is_open_licensed_on_the_shared_grid_and_carries_its_cautions() -> None:
    (manifest,) = t0.load_manifests()
    info = t0.info_from_manifest(manifest)
    assert info.id == "t0-gamma" and info.family == "t0" and info.license == "Apache-2.0" and info.kind == "foundation"
    c = info.capabilities
    assert list(c.quantile_levels) == LEVELS and c.known_covariates and c.past_covariates and c.multivariate
    assert not c.categorical_covariates and c.max_context == 8192
    joined = " ".join(info.notes)
    assert "young model" in joined and "no independent evaluation" in joined  # a 2-week-old model must say so
    assert "64 unrelated series jointly" in joined  # the published benchmark used a mode this app does not
    assert "t0-gamma" in build_default_registry().ids()


class FakeT0(torch.nn.Module):
    """Records what the adapter hands to the model and answers with a monotone quantile fan."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1)  # gives the module a device, like the real model
        self.calls: list[dict] = []

    def predict(self, model_input, horizon, quantile_levels, **kwargs):
        self.calls.append({"input": model_input, "h": horizon, "levels": quantile_levels, **kwargs})
        from t0 import VariateType

        targets = [i for i in range(model_input.variates.shape[0]) if int(model_input.variate_type[i, -1]) == int(VariateType.TARGET)]
        q = torch.linspace(0, 1, len(quantile_levels)).repeat(len(targets), horizon, 1)
        return type("F", (), {"quantiles": q})()


def adapter_with(fake: FakeT0):
    adapter = build_default_registry().create("t0-gamma")
    adapter._model, adapter._device = fake, "cpu"
    return adapter


def test_each_target_row_becomes_one_series_with_its_covariates_in_the_authors_layout() -> None:
    from t0 import VariateType

    fake = FakeT0()
    ctx = np.random.default_rng(0).normal(size=(2, 2, 100))
    past = np.ones((2, 1, 100))
    known = np.arange(2 * 3 * 124, dtype=float).reshape(2, 3, 124)
    res = adapter_with(fake).forecast(ForecastRequest(ctx, 24, past_covariates=past, known_covariates=known,
                                                      options={"symmetric_averaging": False, "clip_nonnegative": False}))
    call = fake.calls[0]
    batch = call["input"]
    assert call["levels"] == LEVELS and call["h"] == 24 and call["context_length"] == batch.seq_len - 24 == 100
    # 2 series x 2 variables = 4 independent target rows, each followed by 3 known-future and 1 historical row
    kinds = [int(k) for k in batch.variate_type[:, -1]]
    assert kinds.count(int(VariateType.TARGET)) == 4 and kinds.count(int(VariateType.HISTORICAL)) == 4 and len(kinds) == 4 * 5
    assert call["symmetric_averaging"] is False and call["clip_nonnegative"] is False  # Timestral's options carried over
    assert call["trim_leading_missing"] is True and call["snap_integers"] is True  # the authors' recommended adjustments
    assert res.quantiles.shape == (2, 2, 24, 21) and any("2 variables" in w for w in res.warnings)
    default = adapter_with(FakeT0())
    default.forecast(ForecastRequest(ctx[:, :1], 8))
    assert (default._model.calls[0]["symmetric_averaging"], default._model.calls[0]["clip_nonnegative"]) == (True, True)


def test_long_context_is_capped_and_series_are_sent_in_chunks() -> None:
    fake = FakeT0()
    res = adapter_with(fake).forecast(ForecastRequest(np.zeros((40, 1, 9000)), 12))
    assert all(c["input"].seq_len == 8192 + 12 for c in fake.calls)  # most recent 8192 steps plus the forecast region
    assert len(fake.calls) == 2 and sum(int((c["input"].variate_type[:, -1] == 0).sum()) for c in fake.calls) == 40  # 32 rows per call
    assert res.quantiles.shape == (40, 1, 12, 21)


def test_missing_known_covariates_over_the_horizon_are_flagged() -> None:
    known = np.zeros((1, 1, 60))
    known[..., 50:] = np.nan
    res = adapter_with(FakeT0()).forecast(ForecastRequest(np.zeros((1, 1, 50)), 10, known_covariates=known))
    assert any("read them as 0" in w or "reads them as 0" in w for w in res.warnings)


def test_loading_without_weights_says_where_to_get_them(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "hub_dir", lambda: tmp_path)
    adapter = build_default_registry().create("t0-gamma")
    adapter.strict_device = True
    with pytest.raises(FileNotFoundError, match="download its weights on the Models page"):
        adapter.load("cpu")


@pytest.mark.model
def test_the_real_checkpoint_continues_a_clean_cycle_when_it_is_present() -> None:
    adapter = build_default_registry().create("t0-gamma")
    if not cache.status(adapter.info.details["repo"])["downloaded"]:
        pytest.skip("t0-gamma (1 GB) is not downloaded here")
    adapter.load("cpu")
    t = np.arange(512)
    ctx = (10 + np.sin(2 * np.pi * t / 24))[None, None, :].repeat(2, 0)
    known = np.sin(2 * np.pi * np.arange(560) / 24)[None, None, :].repeat(2, 0)
    res = adapter.forecast(ForecastRequest(ctx, 48, known_covariates=known))
    assert res.quantiles.shape == (2, 1, 48, 21) and np.isfinite(res.quantiles).all()
    assert (np.diff(res.quantiles, axis=-1) >= -1e-6).all()
    assert np.abs(res.quantiles[0, 0, :, 10] - (10 + np.sin(2 * np.pi * np.arange(512, 560) / 24))).max() < 0.5
    adapter.unload()
