import json

import numpy as np
import pytest

from tsfm_lab.data import synthetic
from tsfm_lab.eval.backtest import BacktestConfig, make_items
from tsfm_lab.eval.runner import run_experiment
from tsfm_lab.store.store import RunStore, to_jsonable


def test_windows_are_rolling_from_the_end() -> None:
    ds = synthetic.build("trend", n_series=2, length=100)
    items = make_items(ds, BacktestConfig(horizon=10, n_windows=3))
    assert [(si, k, o) for si, k, o in items if si == 0] == [(0, 0, 90), (0, 1, 80), (0, 2, 70)]
    assert make_items(ds, BacktestConfig(horizon=10, n_windows=3, stride=5))[1][2] == 85


def test_too_short_series_raise_a_clear_error() -> None:
    ds = synthetic.build("trend", n_series=2, length=30)
    with pytest.raises(ValueError, match="no valid backtest windows"):
        make_items(ds, BacktestConfig(horizon=24, n_windows=2))


def test_experiment_uses_identical_windows_and_ranks_models_sensibly(tmp_path) -> None:
    ds = synthetic.build("multi_seasonality", n_series=4, length=400)
    store = RunStore(tmp_path)
    result, run_id = run_experiment("drift", ds, BacktestConfig(horizon=24, n_windows=2),
                                    baselines=("naive", "seasonal-naive"), store=store, label="t")
    assert set(result.models) == {"drift", "naive", "seasonal-naive"}
    assert result.y.shape == (8, 1, 24)
    q = {m: o.quantiles for m, o in result.models.items()}
    assert all(v.shape == (8, 1, 24, 21) for v in q.values())
    # a strongly seasonal series: seasonal naive must beat naive
    assert result.models["seasonal-naive"].summary["mase"]["mean"] < result.models["naive"].summary["mase"]["mean"]
    assert result.models["seasonal-naive"].skill["naive"]["mase"]["skill"] > 0.3

    meta = store.meta(run_id)
    assert meta["dataset"]["id"] == "multi_seasonality" and meta["config"]["horizon"] == 24
    assert meta["environment"]["python"] and "torch" in meta["environment"]["packages"]
    json.dumps(meta, allow_nan=False)  # must be strict JSON (browsers reject NaN)
    arrays = store.arrays(run_id)
    np.testing.assert_allclose(arrays["naive/quantiles"], q["naive"])
    assert "seasonal-naive/item/mase" in arrays
    assert [r["id"] for r in store.list_runs()] == [run_id]
    store.delete(run_id)
    assert store.list_runs() == []
    with pytest.raises(KeyError):
        store.meta(run_id)


def test_series_of_different_lengths_are_padded_not_dropped(tmp_path) -> None:
    ds = synthetic.build("trend", n_series=3, length=200)
    ds.series[1].target = ds.series[1].target[:, :120]
    ds.series[1].timestamps = ds.series[1].timestamps[:120]
    result, _ = run_experiment("naive", ds, BacktestConfig(horizon=12, n_windows=2), baselines=())
    assert result.y.shape[0] == 6 and np.isfinite(result.y).all()
    assert np.isnan(result.history[2:4]).any() or result.history.shape[2] >= 188  # shorter context is NaN-padded


def test_covariates_are_ignored_with_a_note_for_models_without_support() -> None:
    ds = synthetic.build("covariate_response", n_series=2, length=200)
    result, _ = run_experiment("naive", ds, BacktestConfig(horizon=14, n_windows=1), baselines=())
    assert any("known covariates ignored" in n for n in result.models["naive"].notes)
    assert result.covariates is not None


def test_to_jsonable_makes_nan_and_numpy_strict_json() -> None:
    data = {"a": np.array([1.0, np.nan, np.inf]), "b": np.float32(2.5), "c": (1, np.int64(3)), "d": float("nan")}
    out = to_jsonable(data)
    assert out == {"a": [1.0, None, None], "b": 2.5, "c": [1, 3], "d": None}
    json.dumps(out, allow_nan=False)


def test_warmup_is_reported_separately_and_can_be_disabled() -> None:
    ds = synthetic.build("trend", n_series=2, length=120)
    on, _ = run_experiment("naive", ds, BacktestConfig(horizon=10, n_windows=1), baselines=())
    off, _ = run_experiment("naive", ds, BacktestConfig(horizon=10, n_windows=1, warmup=False), baselines=())
    assert on.models["naive"].warmup_ms is not None and off.models["naive"].warmup_ms is None
