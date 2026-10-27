import numpy as np
import pytest

from tsfm_lab.eval import metrics as M

LEVELS = (0.1, 0.5, 0.9)


def test_pinball_matches_hand_computation() -> None:
    y = np.array([[[10.0]]])  # (N=1, V=1, H=1)
    q = np.array([[[[8.0, 9.0, 12.0]]]])  # under-forecast at 0.1 and 0.5, over at 0.9
    p = M.pinball(y, q, LEVELS)[0, 0, 0]
    assert p == pytest.approx([0.1 * 2, 0.5 * 1, (0.9 - 1) * (10 - 12)])  # 0.2, 0.5, 0.2


def test_item_metrics_perfect_forecast_has_zero_error_and_full_coverage() -> None:
    y = np.array([[[1.0, 2.0, 3.0]]])
    q = np.repeat(y[..., None], 3, axis=-1)
    out = M.item_metrics(y, q, LEVELS, scale=np.array([[1.0]]))
    assert out["mase"][0, 0] == 0 and out["sql"][0, 0] == 0 and out["bias"][0, 0] == 0
    assert M.summarize(out)["mase"]["n"] == 1


def test_mase_and_bias_are_scaled_by_history() -> None:
    y = np.array([[[10.0, 10.0]]])
    q = np.stack([np.full((1, 1, 2), v) for v in (11.0, 12.0, 13.0)], axis=-1)  # median 12 -> |err| 2, bias +2
    out = M.item_metrics(y, q, LEVELS, scale=np.array([[4.0]]))
    assert out["mase"][0, 0] == pytest.approx(0.5)
    assert out["bias"][0, 0] == pytest.approx(0.5)


def test_history_scale_uses_seasonal_lag_then_falls_back() -> None:
    ramp = np.array([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]])
    assert M.history_scale(ramp, 2)[0, 0] == pytest.approx(2.0)  # mean |x_t - x_{t-2}|
    assert M.history_scale(ramp, 1)[0, 0] == pytest.approx(1.0)
    cycle = np.array([[[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]])
    assert M.history_scale(cycle, 2)[0, 0] == pytest.approx(1.0)  # seasonal diff is 0 -> falls back to lag 1
    flat = np.full((1, 1, 10), 5.0)
    assert np.isnan(M.history_scale(flat, 3)[0, 0])  # constant history: undefined, not hidden


def test_nan_actuals_are_ignored() -> None:
    y = np.array([[[1.0, np.nan, 3.0]]])
    q = np.repeat(np.array([[[1.0, 2.0, 3.0]]])[..., None], 3, axis=-1)
    out = M.item_metrics(y, q, LEVELS, scale=np.array([[1.0]]))
    assert out["mase"][0, 0] == 0
    assert out["coverage_80"][0, 0] == 1.0


def test_coverage_counts_inside_interval() -> None:
    levels = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    q = np.zeros((1, 1, 4, 7)) + np.array([-2, -1.5, -1, 0, 1, 1.5, 2.0])
    y = np.array([[[0.0, 1.2, 1.7, 5.0]]])  # inside 80% (|x|<=1.5): first two only
    out = M.item_metrics(y, q, levels, scale=np.array([[1.0]]))
    assert out["coverage_80"][0, 0] == pytest.approx(0.5)
    assert out["coverage_90"][0, 0] == pytest.approx(0.75)


def test_calibration_curve_of_a_calibrated_forecast() -> None:
    rng = np.random.default_rng(0)
    levels = np.linspace(0.05, 0.95, 19)
    y = rng.normal(size=(200, 1, 50))
    from statistics import NormalDist

    z = np.array([NormalDist().inv_cdf(p) for p in levels])
    q = np.broadcast_to(z, (200, 1, 50, 19))
    cal = M.calibration_curve(y, q, levels)
    assert np.abs(cal["empirical"] - levels).max() < 0.02


def test_skill_score() -> None:
    assert M.skill_score(np.array([1.0, 1.0]), np.array([2.0, 2.0]))["skill"] == pytest.approx(0.5)
    assert M.skill_score(np.array([2.0]), np.array([1.0]))["skill"] == pytest.approx(-1.0)
    nan = M.skill_score(np.array([np.nan, 0.0]), np.array([1.0, 1.0]))
    assert nan["n"] == 0 and np.isnan(nan["skill"])
