import numpy as np
import pandas as pd
import pytest

from tsfm_lab.data import synthetic
from tsfm_lab.data.local import ColumnMapping, load_dataframe, load_file
from tsfm_lab.data.schema import Dataset, Series, seasonal_period_for


@pytest.mark.parametrize("freq,expected", [("h", 24), ("D", 7), ("15min", 96), ("4h", 6), ("W-SUN", 52),
                                           ("MS", 12), ("QE-DEC", 4), (None, 1), ("weird", 1), ("7min", 1)])
def test_seasonal_period(freq, expected) -> None:
    assert seasonal_period_for(freq) == expected


def test_every_synthetic_scenario_builds_and_is_deterministic() -> None:
    assert len(synthetic.SCENARIOS) >= 16
    for sid in synthetic.SCENARIOS:
        a, b = synthetic.build(sid, **({"length": 600} if sid != "long_context" else {"length": 400})), \
               synthetic.build(sid, **({"length": 600} if sid != "long_context" else {"length": 400}))
        for sa, sb in zip(a.series, b.series):
            np.testing.assert_array_equal(sa.target, sb.target)
        assert a.series[0].timestamps is not None and a.seasonal_period >= 1


def test_synthetic_seed_changes_data() -> None:
    a, b = synthetic.build("trend", seed=1), synthetic.build("trend", seed=2)
    assert not np.array_equal(a.series[0].target, b.series[0].target)


def test_scenario_features() -> None:
    assert synthetic.build("cross_series_dependence").n_targets == 2
    cov = synthetic.build("covariate_response")
    assert cov.n_known == 2 and cov.known_names == ["promo", "price"]
    flat = synthetic.build("flat_price_covariate")
    price = flat.series[0].known_covariates[0]
    assert (price[:800] == 2.5).all() and price[-1] == pytest.approx(2.56)
    assert np.isnan(synthetic.build("missing_blocks").series[0].target).any()
    assert (synthetic.build("negative_values").series[0].target < 0).all()
    assert (synthetic.build("intermittent").series[0].target >= 0).all()
    assert np.ptp(synthetic.build("constant_series").series[0].target) == 0


def test_dataset_rejects_inconsistent_series() -> None:
    with pytest.raises(ValueError):
        Dataset("x", "x", [Series("a", np.zeros((1, 5))), Series("b", np.zeros((2, 5)))])
    with pytest.raises(ValueError):
        Series("a", np.zeros((1, 5)), known_covariates=np.zeros((1, 4)))


def _frame() -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=48, freq="h")
    frame = pd.DataFrame({"ts": t, "id": ["a"] * 24 + ["b"] * 24, "y": np.arange(48.0), "price": 1.0})
    frame["ts"] = list(t[:24]) + list(t[:24])
    return frame


def test_local_loader_groups_series_and_infers_frequency() -> None:
    ds, report = load_dataframe(_frame(), ColumnMapping("ts", ["y"], id="id", known_covariates=["price"]))
    assert [s.id for s in ds.series] == ["a", "b"] and ds.freq == "h" and ds.seasonal_period == 24
    assert ds.n_known == 1 and report["rows"] == 48 and ds.contamination_risk == "unknown"


def test_local_loader_reports_gaps_duplicates_and_bad_timestamps() -> None:
    frame = _frame().iloc[:24].copy()
    frame = frame.drop(index=[5, 6])  # two missing hours
    frame = pd.concat([frame, frame.iloc[[0]]])  # one duplicate timestamp
    frame["ts"] = frame["ts"].astype(object)
    frame.loc[frame.index[3], "ts"] = "not a date"
    ds, report = load_dataframe(frame, ColumnMapping("ts", ["y"], freq="h"))
    s = report["series"][0]
    assert s["duplicate_timestamps"] == 1 and s["gaps_filled_with_nan"] >= 2
    assert report["unparseable_timestamps"] == 1
    assert np.isnan(ds.series[0].target).any()
    assert len(report["warnings"]) >= 3


def test_local_loader_validates_columns_and_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="columns not found"):
        load_dataframe(_frame(), ColumnMapping("ts", ["nope"]))
    path = tmp_path / "data.csv"
    _frame().to_csv(path, index=False)
    ds, _ = load_file(path)
    assert ds.source == "local" and len(ds.series) >= 1
    with pytest.raises(ValueError, match="unsupported"):
        load_file(tmp_path / "x.json")
