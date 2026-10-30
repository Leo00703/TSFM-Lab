"""The command line must not fall over on a model that gives fewer metrics than the others."""

from __future__ import annotations

from tsfm_lab.cli import _mean


def test_a_metric_the_model_does_not_give_is_shown_as_a_dash() -> None:
    summary = {"mase": {"mean": 0.8123}, "coverage_80": {"mean": 0.79}}  # decile-only models have no 90% interval
    assert _mean(summary, "mase", 3) == "0.812"
    assert _mean(summary, "coverage_80", 2) == "0.79"
    assert _mean(summary, "coverage_90", 2) == "-"
