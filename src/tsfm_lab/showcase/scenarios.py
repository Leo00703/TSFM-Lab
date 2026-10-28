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


def _grocery() -> dict[str, str]:
    """The 40 best-selling products with a complete, recent history: a rule that never looks at the covariates."""
    rows = ("4720 Roll_51 Brno_1;4885 Roll_57 Prague_1;4884 Roll_57 Prague_2;4883 Roll_57 Prague_3;652 Cucumber_10 Prague_1;"
            "139 Pastry_4 Budapest_1;4879 Roll_56 Prague_1;525 Roll_8 Prague_1;2561 Banana_10 Brno_1;3144 Bread_122 Prague_1;"
            "655 Cucumber_10 Prague_2;654 Cucumber_10 Prague_3;2562 Banana_10 Prague_1;3184 Roll_30 Brno_1;1032 Roll_11 Brno_1;"
            "4877 Roll_56 Prague_3;524 Roll_8 Prague_3;2159 Roll_20 Prague_1;526 Roll_8 Prague_2;2935 Bread_113 Brno_1;"
            "2937 Bread_113 Prague_1;3616 Bread_143 Prague_1;2216 Croissant_16 Prague_1;4965 Kohlrabi_10 Prague_1;"
            "393 Avocado_1 Prague_1;1408 Lime_4 Prague_1;5293 Baguette_40 Prague_1;4878 Roll_56 Prague_2;2563 Banana_10 Prague_3;"
            "2160 Roll_20 Prague_2;2560 Banana_10 Prague_2;2694 Roll_24 Brno_1;3871 Avocado_7 Prague_1;2161 Roll_20 Prague_3;"
            "3456 Roll_34 Prague_1;1391 Roll_12 Prague_1;3686 Kiwi_17 Prague_1;1927 Pastry_81 Brno_1;2938 Bread_113 Prague_2;"
            "2936 Bread_113 Prague_3")
    out = {}
    for row in rows.split(";"):
        sid, name, wh = row.split()
        city, _, n = wh.partition("_")
        out[sid] = f"{name.replace('_', ' ')} · {city if city != 'Prague' else 'Prague ' + n}"
    return out


def _retail() -> dict[str, str]:
    """The 30 best-selling items of the California stores that have the full 280 weeks."""
    ids = ["FOODS_3_090_CA_3", "FOODS_3_586_CA_3", "FOODS_3_090_CA_1", "FOODS_3_586_CA_1", "FOODS_3_252_CA_3", "FOODS_3_541_CA_3", "FOODS_3_635_CA_3", "FOODS_3_252_CA_1", "FOODS_3_808_CA_3", "FOODS_3_586_CA_2", "FOODS_3_555_CA_3", "FOODS_3_252_CA_2", "FOODS_3_090_CA_2", "FOODS_3_681_CA_3", "FOODS_3_714_CA_1", "FOODS_3_090_CA_4", "FOODS_3_714_CA_3", "FOODS_3_587_CA_1", "FOODS_3_202_CA_3", "FOODS_3_808_CA_1", "FOODS_3_723_CA_3", "FOODS_3_318_CA_3", "FOODS_3_752_CA_3", "HOUSEHOLD_1_118_CA_3", "FOODS_2_197_CA_3", "FOODS_3_694_CA_3", "FOODS_3_234_CA_3", "FOODS_2_197_CA_1", "FOODS_3_501_CA_3", "FOODS_2_360_CA_3"]
    out = {}
    for sid in ids:
        cat, dept, item, _, store = sid.split("_")
        out[sid] = f"{cat.title()} item {dept}-{item} · store {store}"
    return out


def _supermarkets() -> dict[str, str]:
    """All 51 stores of the dataset, labelled with their city."""
    rows = ("1 Quito;2 Quito;3 Quito;4 Quito;5 Santo_Domingo;6 Quito;7 Quito;8 Quito;9 Quito;10 Quito;11 Cayambe;12 Latacunga;13 Latacunga;14 Riobamba;16 Santo_Domingo;17 Quito;18 Quito;19 Guaranda;20 Quito;21 Santo_Domingo;22 Puyo;23 Ambato;24 Guayaquil;25 Salinas;26 Guayaquil;27 Daule;28 Guayaquil;29 Guayaquil;30 Guayaquil;31 Babahoyo;32 Guayaquil;33 Quevedo;34 Guayaquil;35 Playas;36 Libertad;38 Loja;39 Cuenca;40 Machala;41 Machala;42 Cuenca;43 Esmeraldas;44 Quito;45 Quito;46 Quito;47 Quito;49 Quito;50 Ambato;51 Guayaquil;52 Manta;53 Manta;54 El_Carmen")
    return {sid: f"Store {sid} · {city.replace('_', ' ')}" for sid, city in (r.split() for r in rows.split(";"))}


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
        caveats=(("The factors are the values that were actually measured, not forecasts. In real life you would only have forecasts "
                 "of solar, wind and temperature, so the benefit shown here is an upper bound."),
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
        caveats=(("Opening days, promotions and holidays are genuinely known ahead of time, so this scenario is a fair test of "
                 "what covariates can add."),),
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
    Scenario(
        id="grocery", tab="Groceries", title="Online grocery sales", subtitle="Daily units sold per product and warehouse", unit="units",
        story="How many rolls, bananas or cucumbers will an online grocer sell over the next four weeks? Prices, discounts and "
              "holidays are set or known in advance. Switch them on and see whether the forecast improves.",
        name="rohlik_sales", freq="1D", size_mb=30.5, target="sales", horizon=28, view=90, context=730,
        horizon_label="next 4 weeks", series_label="Product", series=_grocery(), default_series="4720",
        impact_series=tuple(_grocery()), impact_windows=10, impact_stride=28,
        covariates=(
            Covariate("price", "Price", "sell_price_main", hint="The selling price that day"),
            Covariate("discount", "Discount", "type_0_discount", hint="The size of the discount, if there is one"),
            Covariate("holiday", "Public holiday", "holiday", binary=True, hint="Whether it is a public holiday"),
            Covariate("winter", "Winter school break", "winter_school_holidays", binary=True, hint="Whether schools are on winter break"),
        ),
        caveats=(("Product names are anonymised in the dataset. The 40 products shown are simply the best sellers with a complete recent "
                 "history; the choice does not look at the factors."),
                 ("Discounts are rare for most of these products, so the gain from \"Discount\" comes from the few that have them. A discount "
                 "that never appeared in the history is hidden from the model, which avoids a known flaw of the model."),
                 "Stock availability is left out on purpose: it is not known in advance."),
        ),
    Scenario(
        id="stores", tab="Store traffic", title="Supermarket traffic", subtitle="Daily purchases per supermarket in Ecuador", unit="purchases",
        story="How many customers will a supermarket serve over the next four weeks? Holidays are known in advance. The oil price "
              "is shown on purpose: it is a factor that looks relevant but is not, so you can see what a useless factor does.",
        name="favorita_transactions", freq="1D", size_mb=0.2, target="transactions", horizon=28, view=90, context=730,
        horizon_label="next 4 weeks", series_label="Store", series=_supermarkets(),
        default_series="1", impact_series=tuple(_supermarkets()), impact_windows=10, impact_stride=28,
        covariates=(
            Covariate("holiday", "Holiday", "holiday", transform="nonzero_str", binary=True, hint="Whether it is a holiday near the store"),
            Covariate("oil", "Oil price", "oil_price", hint="The world market price of oil"),
        ),
        caveats=("Holidays are matched to the store's own region, as published with the data.",
                 ("The oil price is not known in advance: the value used is the one that was recorded. It is included on purpose, "
                 "as an example of a factor that adds nothing. Not every factor helps, and the measured effect shows it."),
                 ("How much the holidays help depends on the model: the larger ones use them clearly better. With the smaller ones, "
                 "all factors together can even make the forecast slightly worse. Switch the model above to see it."),
                 "The data are the daily purchase counts of 51 stores of one supermarket chain, from 2013 to 2017."),
        ),
    Scenario(
        id="retail", tab="US retail", title="US retail items", subtitle="Weekly units sold per item and store (California)", unit="units",
        story="How many units of a popular item will a California store sell over the next twelve weeks? The price the store sets, "
              "food-aid payment days and public events are known in advance. Switch them on and compare.",
        name="m5", freq="1W", size_mb=9.8, target="target", horizon=12, view=52, context=256,
        horizon_label="next 12 weeks", series_label="Item and store", series=_retail(), default_series="FOODS_3_090_CA_3",
        impact_series=tuple(_retail()), impact_windows=10, impact_stride=12,
        covariates=(
            Covariate("price", "Price", "sell_price", hint="The selling price that week"),
            Covariate("snap", "Food-aid payment days", "snap_CA", hint="Days in the week when SNAP food-aid benefits can be spent"),
            Covariate("national", "National holidays", "event_National", hint="National holidays in the week"),
            Covariate("religious", "Religious holidays", "event_Religious", hint="Religious holidays in the week"),
            Covariate("cultural", "Cultural events", "event_Cultural", hint="Cultural events in the week"),
            Covariate("sporting", "Sporting events", "event_Sporting", hint="Major sporting events in the week"),
        ),
        caveats=(("The price is the one the store really set. Prices are decided in advance, so this is a fair test of what a planned price "
                 "can add."),
                 ("More factors is not always better: with everything switched on the gain can be smaller than with the price alone, "
                 "because the other factors add little and the model has more to weigh."),
                 "The 30 items shown are the best sellers of the California stores; the choice does not look at the factors."),
        ),
)}
