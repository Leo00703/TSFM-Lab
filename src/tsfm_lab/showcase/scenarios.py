"""Showcase scenarios: real datasets told as plain-language stories.

Each scenario says what is forecast, which known-in-advance factors the model may be given, and the
honest caveats. Data comes from ``autogluon/fev_datasets`` (pinned revision); nothing here is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tsfm_lab.data.hub import Covariate


@dataclass(frozen=True)
class Scenario:
    id: str
    tab: str
    title: str
    subtitle: str
    story: str
    name: str  # fev dataset name
    freq: str
    size_mb: float
    target: str
    unit: str
    horizon: int
    view: int  # steps of history shown next to the forecast
    context: int  # history given to the model
    horizon_label: str
    series_label: str
    series: dict[str, str]  # id -> label, in display order
    default_series: str
    impact_series: tuple[str, ...]
    impact_windows: int
    impact_stride: int
    covariates: tuple[Covariate, ...]
    caveats: tuple[str, ...] = ()
    max_offset_steps: int = 0  # how far back the forecast start can be moved
    extra: dict = field(default_factory=dict)


def _stores() -> dict[str, str]:
    ids = [1, 3, 7, 11, 18, 25, 33, 40, 55, 62, 74, 85, 99, 110, 122, 150, 181, 207, 250, 291, 322, 388, 413, 466, 512,
           560, 612, 667, 700, 733, 777, 801, 849, 872, 910, 939, 975, 1002, 1040, 1099]
    return {str(i): f"Store {i}" for i in ids}


SCENARIOS: dict[str, Scenario] = {s.id: s for s in (
    Scenario(
        id="electricity", tab="Electricity", title="Electricity demand", subtitle="Hourly demand of a country",
        unit="",
        story="How much electricity will a country use over the next week? Demand follows a daily and weekly rhythm and reacts "
              "to sunshine and temperature. Switch those factors on and see how the forecast changes.",
        name="entsoe", freq="1H", size_mb=5.8, target="target", horizon=168, view=336, context=2048,
        horizon_label="next 7 days", series_label="Country",
        series={"DE": "Germany", "AT": "Austria", "BE": "Belgium", "HU": "Hungary", "LU": "Luxembourg", "NL": "Netherlands"},
        default_series="DE", impact_series=("DE", "AT", "BE", "HU", "LU", "NL"), impact_windows=30, impact_stride=24,
        covariates=(
            Covariate("solar", "Solar generation", "solar_generation_actual", hint="How much solar power is being produced"),
            Covariate("wind", "Wind generation", "wind_onshore_generation_actual", hint="How much onshore wind power is being produced"),
            Covariate("temperature", "Temperature", "temperature", hint="Air temperature"),
            Covariate("sun", "Sunshine", "radiation_direct_horizontal", hint="Direct sunlight reaching the ground"),
        ),
        caveats=("The factors are the values that were actually measured, not forecasts. In real life you would only have forecasts "
                 "of solar, wind and temperature, so the benefit shown here is an upper bound.",
                 "The dataset does not state the unit of demand, so the values are shown as they are."),
        max_offset_steps=24 * 60),
    Scenario(
        id="sales", tab="Sales", title="Store sales", subtitle="Daily sales per store", unit="€",
        story="How much will a store sell over the next four weeks? Opening days, promotions and holidays are known in advance. "
              "Switch them on and watch the forecast react.",
        name="rossmann", freq="1D", size_mb=3.7, target="Sales", horizon=28, view=90, context=730,
        horizon_label="next 4 weeks", series_label="Store", series=_stores(), default_series="1",
        impact_series=tuple(_stores())[:40], impact_windows=10, impact_stride=28,
        covariates=(
            Covariate("open", "Store open", "Open", binary=True, hint="Whether the store is open that day"),
            Covariate("promo", "Promotion", "Promo", binary=True, hint="Whether a promotion runs that day"),
            Covariate("school", "School holiday", "SchoolHoliday", binary=True, hint="Whether schools are on holiday"),
            Covariate("public", "Public holiday", "StateHoliday", transform="nonzero_str", binary=True, hint="Whether it is a public holiday"),
        ),
        caveats=("Opening days, promotions and holidays are genuinely known ahead of time, so this scenario is a fair test of "
                 "what covariates can add.",),
        max_offset_steps=28 * 6),
    Scenario(
        id="solar", tab="Solar power", title="Solar power output", subtitle="Hourly output of a solar installation", unit="power units of the dataset",
        story="How much power will a solar installation produce over the next three days? It depends on sunlight, clouds and "
              "temperature. Switch the weather on and compare.",
        name="solar_with_weather", freq="1H", size_mb=0.8, target="target", horizon=72, view=168, context=2048,
        horizon_label="next 3 days", series_label="Installation", series={"0": "Installation 0"}, default_series="0",
        impact_series=("0",), impact_windows=40, impact_stride=24,
        covariates=(
            Covariate("sunlight", "Sunlight intensity", "global_horizontal_irradiance", hint="Solar radiation reaching the panels"),
            Covariate("clouds", "Cloud cover", "clouds_all", hint="How cloudy the sky is"),
            Covariate("temp", "Temperature", "temp", hint="Air temperature"),
            Covariate("humidity", "Humidity", "humidity", hint="Air humidity"),
            Covariate("wind", "Wind speed", "wind_speed", hint="Wind speed"),
        ),
        caveats=("The weather values are the ones measured at the time, not forecasts: with real forecasts the gain would be smaller.",
                 "The dataset does not state the unit of the power output."),
        max_offset_steps=24 * 60),
)}
