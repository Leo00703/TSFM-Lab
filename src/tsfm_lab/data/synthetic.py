"""Bundled synthetic test datasets (seeded, deterministic, no downloads).

Scenario families follow the eight structural features used by capability-focused evaluation
(arXiv 2610.26251: trend, multi-seasonality, time-varying seasonality, regime switching,
intermittency, common factor, cross-series dependence, covariate response), plus edge cases
that stress known weak spots (missing blocks, outliers, level shifts, constant series, extreme
scales, negative values, long context, and a reproduction of upstream issue #7).

Synthetic data is never part of any model's training set by construction of the *seed*, but a
foundation model trained on synthetic processes may still find these "easy": read results with that in mind.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tsfm_lab.data.schema import Dataset, Series, seasonal_period_for

START = np.datetime64("2024-01-01T00:00")


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    description: str
    tags: tuple[str, ...]
    build: Callable[..., Dataset]
    defaults: dict


def _stamps(length: int, freq: str) -> np.ndarray:
    step = {"h": np.timedelta64(1, "h"), "D": np.timedelta64(1, "D"), "15min": np.timedelta64(15, "m")}[freq]
    return START + np.arange(length) * step


def _make(sid: str, name: str, rows: list[np.ndarray], freq: str, description: str, tags: list[str],
          known: list[np.ndarray] | None = None, past: list[np.ndarray] | None = None,
          known_names: list[str] | None = None, past_names: list[str] | None = None,
          target_names: list[str] | None = None) -> Dataset:
    length = rows[0].shape[-1]
    ts = _stamps(length, freq)
    series = []
    for i, tgt in enumerate(rows):
        series.append(Series(f"s{i:02d}", tgt, ts, None if past is None else past[i],
                             None if known is None else known[i]))
    return Dataset(id=sid, name=name, series=series, freq=freq, seasonal_period=seasonal_period_for(freq),
                   source="synthetic", license="generated (CC0)", description=description, tags=tags,
                   known_names=known_names or [], past_names=past_names or [], target_names=target_names or [])


def trend(n_series=8, length=720, seed=0, freq="h") -> Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    rows = [rng.uniform(-0.05, 0.1) * t + rng.uniform(50, 150) + 0.0004 * t**2 * rng.uniform(-1, 1)
            + rng.normal(0, 1.5, length) for _ in range(n_series)]
    return _make("trend", "Trend", rows, freq, "Linear and mildly curved trends with noise.", ["trend"])


def multi_seasonality(n_series=8, length=1344, seed=0, freq="h") -> Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    rows = []
    for _ in range(n_series):
        daily, weekly = rng.uniform(5, 15), rng.uniform(3, 10)
        rows.append(100 + daily * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6.28))
                    + weekly * np.sin(2 * np.pi * t / 168 + rng.uniform(0, 6.28)) + rng.normal(0, 1, length))
    return _make("multi_seasonality", "Multiple seasonalities", rows, freq,
                 "Daily (24) + weekly (168) cycles with noise.", ["seasonality"])


def time_varying_seasonality(n_series=8, length=1344, seed=0, freq="h") -> Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    rows = []
    for _ in range(n_series):
        amp = 10 * (1 + 0.8 * np.sin(2 * np.pi * t / (length * rng.uniform(0.6, 1.2))))
        period = 24 + 4 * np.sin(2 * np.pi * t / length)
        phase = np.cumsum(2 * np.pi / period)
        rows.append(100 + amp * np.sin(phase) + rng.normal(0, 1, length))
    return _make("time_varying_seasonality", "Time-varying seasonality", rows, freq,
                 "Seasonal amplitude and period drift over time.", ["seasonality", "non-stationary"])


def regime_switching(n_series=8, length=720, seed=0, freq="h") -> Dataset:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_series):
        x = np.empty(length)
        level, regime = 100.0, 0
        for i in range(length):
            if rng.random() < 0.01:
                regime = 1 - regime
            level += (0.15 if regime == 0 else -0.2) + rng.normal(0, 0.5 if regime == 0 else 1.5)
            x[i] = level
        rows.append(x)
    return _make("regime_switching", "Regime switching", rows, freq,
                 "Two regimes with different drift and volatility; switch probability 1%/step.", ["regime"])


def intermittent(n_series=8, length=720, seed=0, freq="D") -> Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    rows = []
    for _ in range(n_series):
        rate = rng.uniform(0.2, 1.2) * (1 + 0.3 * np.sin(2 * np.pi * t / 7))
        rows.append(rng.poisson(rate).astype(float))
    return _make("intermittent", "Intermittent demand", rows, freq,
                 "Poisson counts with many zeros and weekly seasonality (non-negative).", ["intermittency"])


def common_factor(n_series=8, length=720, seed=0, freq="h") -> Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    factor = np.cumsum(rng.normal(0, 1, length)) + 12 * np.sin(2 * np.pi * t / 24)
    rows = [rng.uniform(0.5, 2) * factor + rng.uniform(50, 100) + rng.normal(0, 1.5, length) for _ in range(n_series)]
    return _make("common_factor", "Common factor", rows, freq,
                 "All series load on one shared latent factor.", ["relational", "cross-series"])


def cross_series_dependence(n_series=4, length=720, seed=0, freq="h") -> Dataset:
    """Multivariate series: variate 1 follows variate 0 with a lag (a model using variate attention can exploit it)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    rows = []
    for _ in range(n_series):
        lead = np.cumsum(rng.normal(0, 1, length)) + 8 * np.sin(2 * np.pi * t / 24)
        follow = np.roll(lead, 6) * 0.8 + rng.normal(0, 1, length) + 30
        rows.append(np.stack([lead + 100, follow]))
    return _make("cross_series_dependence", "Cross-variate dependence", rows, freq,
                 "Two variates; the second follows the first with a 6-step lag.", ["relational", "multivariate"],
                 target_names=["leader", "follower"])


def covariate_response(n_series=8, length=720, seed=0, freq="D", horizon_extra=0) -> Dataset:
    """Demand driven by a known-future promo flag and price; includes the promo effect in the test horizon."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    rows, known = [], []
    for _ in range(n_series):
        promo = (rng.random(length) < 0.12).astype(float)
        price = 10 + 1.5 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 0.1, length)
        base = 50 + 8 * np.sin(2 * np.pi * t / 7)
        demand = base * (1 + 0.6 * promo) * (10 / price) ** 1.2 + rng.normal(0, 2, length)
        rows.append(demand)
        known.append(np.stack([promo, price]))
    return _make("covariate_response", "Covariate response", rows, freq,
                 "Demand reacts to a known-future promo flag (+60%) and price elasticity. Compare with and "
                 "without covariates.", ["relational", "covariates"], known=known, known_names=["promo", "price"])


# ---------------------------------------------------------------- edge cases
def missing_blocks(n_series=8, length=720, seed=0, freq="h") -> Dataset:
    ds = multi_seasonality(n_series, length, seed, freq)
    rng = np.random.default_rng(seed + 1)
    for s in ds.series:
        for _ in range(3):
            start = rng.integers(10, length - 60)
            s.target[:, start: start + rng.integers(5, 40)] = np.nan
    ds.id, ds.name = "missing_blocks", "Missing blocks"
    ds.description = "Seasonal series with three random NaN gaps of 5-40 steps each in the history."
    ds.tags = ["edge-case", "missing"]
    return ds


def outliers(n_series=8, length=720, seed=0, freq="h") -> Dataset:
    ds = multi_seasonality(n_series, length, seed, freq)
    rng = np.random.default_rng(seed + 2)
    for s in ds.series:
        idx = rng.choice(np.arange(20, length - 60), size=6, replace=False)
        s.target[:, idx] += rng.choice([-1, 1], size=6) * rng.uniform(60, 120, size=6)
    ds.id, ds.name = "outliers", "Outliers"
    ds.description = "Seasonal series with six large spikes (5-10 sigma) in the history."
    ds.tags = ["edge-case", "outliers"]
    return ds


def level_shift(n_series=8, length=720, seed=0, freq="h") -> Dataset:
    ds = multi_seasonality(n_series, length, seed, freq)
    for s in ds.series:
        s.target[:, length - 100:] += 40
    ds.id, ds.name = "level_shift", "Level shift"
    ds.description = "Seasonal series whose level jumps by +40 about 100 steps before the end."
    ds.tags = ["edge-case", "non-stationary"]
    return ds


def constant_series(n_series=4, length=300, seed=0, freq="h") -> Dataset:
    rows = [np.full(length, float(v)) for v in (0.0, 1.0, 42.0, -7.5)[:n_series]]
    return _make("constant_series", "Constant series", rows, freq,
                 "Perfectly flat series (zero variance): a degenerate-input check.", ["edge-case"])


def extreme_scales(n_series=4, length=500, seed=0, freq="h") -> Dataset:
    ds = multi_seasonality(n_series, length, seed, freq)
    for s, scale in zip(ds.series, (1e-6, 1e-2, 1e6, 1e9)):
        s.target = s.target * scale
    ds.id, ds.name = "extreme_scales", "Extreme scales"
    ds.description = "Same seasonal shape multiplied by 1e-6, 1e-2, 1e6 and 1e9."
    ds.tags = ["edge-case", "scale"]
    return ds


def negative_values(n_series=8, length=720, seed=0, freq="h") -> Dataset:
    ds = multi_seasonality(n_series, length, seed, freq)
    for s in ds.series:
        s.target = s.target - 200
    ds.id, ds.name = "negative_values", "Negative values"
    ds.description = "Seasonal series centred on -100 (disables the non-negative clipping path)."
    ds.tags = ["edge-case"]
    return ds


def long_context(n_series=1, length=33000, seed=0, freq="15min") -> Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    rows = [100 + 10 * np.sin(2 * np.pi * t / 96) + 4 * np.sin(2 * np.pi * t / 672)
            + 0.0005 * t + rng.normal(0, 1, length) for _ in range(n_series)]
    return _make("long_context", "Long context (33k steps)", rows, freq,
                 "One 33,000-step 15-minute series: exceeds Timestral's 32,768 context (oldest steps are dropped).",
                 ["edge-case", "long-context"])


def flat_price_covariate(n_series=16, length=1052, seed=0, freq="D", horizon=28) -> Dataset:
    """Reproduces upstream issue #7: Poisson demand with a price that is constant, then changes once in the
    last 200 steps (inside the history or the test horizon), plus a 0/1 indicator."""
    rng = np.random.default_rng(seed)
    rows, known = [], []
    for i in range(n_series):
        demand = rng.poisson(2.0, length).astype(float)
        price = np.full(length, 2.50)
        # even series: the change lands inside the final `horizon` steps (flat context, changing horizon: the
        # issue-#7 trigger); odd series: anywhere in the last 200 steps (often inside the context: harmless)
        change = rng.integers(length - horizon, length) if i % 2 == 0 else rng.integers(length - 200, length)
        price[change:] = 2.56
        snap = (rng.random(length) < 0.3).astype(float)
        rows.append(demand)
        known.append(np.stack([price, snap]))
    return _make("flat_price_covariate", "Flat price covariate (issue #7)", rows, freq,
                 "Poisson(2) demand; price is constant 2.50 then changes once (even series: within the last 28 steps, "
                 "the trigger; odd series: within the last 200). Reproduces the upper-quantile blow-up reported upstream; compare mask_flat_covariates and "
                 "symmetric_averaging settings.", ["edge-case", "covariates", "issue-7"],
                 known=known, known_names=["price", "snap"])


SCENARIOS: dict[str, Scenario] = {}
for _fn, _tags in (
    (trend, ("trend",)), (multi_seasonality, ("seasonality",)),
    (time_varying_seasonality, ("seasonality", "non-stationary")), (regime_switching, ("regime",)),
    (intermittent, ("intermittency",)), (common_factor, ("relational",)),
    (cross_series_dependence, ("relational", "multivariate")), (covariate_response, ("covariates",)),
    (missing_blocks, ("edge-case",)), (outliers, ("edge-case",)), (level_shift, ("edge-case",)),
    (constant_series, ("edge-case",)), (extreme_scales, ("edge-case",)), (negative_values, ("edge-case",)),
    (long_context, ("edge-case",)), (flat_price_covariate, ("edge-case", "covariates")),
):
    _ds = _fn(n_series=2, length=300)  # tiny build, only to read name/description/tags
    SCENARIOS[_fn.__name__] = Scenario(_fn.__name__, _ds.name, _ds.description, tuple(_ds.tags), _fn,
                                       {k: v.default for k, v in inspect.signature(_fn).parameters.items()})


def build(scenario_id: str, **overrides) -> Dataset:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown synthetic scenario '{scenario_id}'. Available: {', '.join(SCENARIOS)}")
    return SCENARIOS[scenario_id].build(**overrides)
