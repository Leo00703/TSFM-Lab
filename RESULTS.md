# Results: ten time-series foundation models on 187 public datasets

Measured with the Leaderboard page of this app on 28 October 2026. This is an independent comparison, not affiliated with any of the
model makers. The full working log (plans, research notes, every run, in Italian) is [`RESOCONTO.md`](RESOCONTO.md); this file is
the part of it that matters to a reader, in English.

**Read it as an indication.** The datasets are public and most makers do not list the real data their model was trained on (see
"Training data" below), so a model may have seen some of them. The uncertainty bars only cover *which datasets were picked*, not the
series sampled inside each dataset. Use the results to compare models with each other on the same footing, not as proof of
out-of-sample skill.

## What was measured

| | |
|---|---|
| Models (10) | Timestral 209M, 123M, 28M, 5M (Mistral AI), Chronos-2 and Chronos-2 small (Amazon), TimesFM 3.0 and 2.5 (Google), t0-gamma (The Forecasting Company), TiRex-2 (NXAI), plus seasonal naive as the reference |
| Datasets (187) | the whole catalog: fev-bench (100), GIFT-Eval (55), Monash archive (32), each with the horizon and number of windows of its own benchmark |
| Sample | 100 series per dataset (a seeded sample, the same series for every model), at most 3 windows per series, context capped at 2048 steps for all models |
| Covariates | used where the dataset has them and the model can take them; ignored automatically otherwise |
| Score | **skill** = 100 × (1 − geometric mean over datasets of the model's error ÷ seasonal naive's error), on SQL (scaled quantile loss; MASE is reported too). 0% = as good as seasonal naive, +40% = 40% lower error. Each ratio is clipped to [0.01, 100] so one dataset cannot dominate |
| Counted | only datasets that every ranked model completed and on which seasonal naive has an error to score against. In the app's Leaderboard, datasets that a ranked model's authors list as its training data are also left out by default (see "Training data"); the tables below give both |
| Hardware | one RTX 5060 Ti 16 GB, Windows 11, PyTorch 2.11 (CUDA 12.8); times are a single pass on an otherwise idle GPU |
| Uncertainty | bootstrap over datasets, 2000 resamples; differences are given as median and 90% interval |

Files above 50 MB were measured in a separate run (third table). The ranked numbers below come from three runs, because the datasets
fall into three groups that cannot be ranked on the same footing: with covariates, without, and very large.

## 1. 127 datasets without covariates (the main ranking)

57 from fev-bench, 41 from GIFT-Eval, 29 from Monash; 126 count (see "Not counted" below). 43 of them appear in more than one
benchmark, so a "de-duplicated" score is given too (107 distinct, a shared dataset weighs once); the bootstrap is on the 107.
One counted dataset (`fred_md_2025--macro`) is listed as TimesFM 2.5 training data; without it (125) nothing changes: the same order,
Timestral-209M +39.4 and t0-gamma +39.1, first in 67% and 32% of the resamples.

| Model | Skill SQL | De-duplicated | Skill MASE | fev-bench | GIFT-Eval | Monash | Difference from Timestral-209M (90%) | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Timestral-209M | +39.3% | +40.1% | +35.0% | 39.2 | 31.4 | 49.3 | — | 92 |
| t0-gamma | +39.0% | +39.6% | +35.6% | 38.4 | 32.3 | 48.5 | −0.4 [−1.8; +0.8] | 106 |
| Timestral-123M | +38.8% | +39.6% | +34.8% | 38.9 | 31.0 | 48.5 | −0.5 [−0.8; −0.2] | 66 |
| Timestral-28M | +38.2% | +39.1% | +34.2% | 38.0 | 30.6 | 48.4 | −1.0 [−1.4; −0.7] | 34 |
| TimesFM 3.0 | +38.1% | +38.8% | +33.9% | 38.5 | 30.5 | 46.9 | −1.3 [−2.8; 0.0] | 474 |
| Timestral-5M | +36.8% | +37.5% | +32.2% | 36.8 | 29.6 | 46.1 | −2.6 [−3.6; −1.7] | 25 |
| Chronos-2 | +35.9% | +36.6% | +31.2% | 36.1 | 28.8 | 44.8 | −3.5 [−4.6; −2.4] | 48 |
| TimesFM 2.5 | +35.0% | +35.8% | +30.7% | 34.5 | 26.8 | 46.3 | −4.3 [−6.8; −2.1] | 282 |
| TiRex-2 | +34.8% | +35.5% | +31.1% | 35.8 | 28.1 | 41.7 | −4.5 [−6.4; −3.1] | 200 |
| Chronos-2 small | +34.4% | +34.8% | +30.6% | 34.9 | 26.8 | 43.1 | −5.3 [−6.5; −4.1] | 23 |

- **At the top**, Timestral-209M is first in 72% of the resamples and t0-gamma in 25%; the two **cannot be told apart**
  (−0.4 [−1.8; +0.8]). Timestral-123M is 0.5 points behind in a small but consistent way, TimesFM 3.0 is at the edge.
- **Small models hold up**: Timestral-28M (34 s) is ahead of Chronos-2, TimesFM 2.5 and TiRex-2; Timestral-5M (25 s) beats
  Chronos-2 small by 2.7 points [+1.7; +3.8] at the same compute time.
- **TimesFM 3.0 beats TimesFM 2.5** by 3.0 points [+1.7; +4.4].
- By source the order changes: GIFT-Eval is the hardest benchmark (all scores below 33), Monash the easiest (up to 49).
- No model has a dataset where its error is more than 5 times seasonal naive's.

## 2. 48 datasets with covariates

40 from fev-bench and 8 from GIFT-Eval (1,849 series in all). TimesFM 2.5 and seasonal naive ignore covariates. **Seven of the 48
are listed by TimesFM 2.5's authors as its training data** (`favorita_transactions_1D/1W`, `fred_md_2025--cee`, `m5_1W`,
`proenfo_gfc12/14/17`); the table gives all 48, then the 41 that are not.

| Model | All 48: skill SQL | MASE | First in bootstrap | Difference from Timestral-209M (90%) | **Without the 7 (41): skill SQL** | First in bootstrap | Difference from t0-gamma (90%) | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Timestral-209M | +42.3% | +36.3% | 47% | — | +40.8% (2nd) | 28% | −0.3 [−2.1; +1.4] | 804 |
| Timestral-123M | +42.0% | +36.1% | 19% | −0.3 [−1.1; +0.7] | +40.7% (3rd) | 14% | −0.5 [−2.2; +1.3] | 501 |
| t0-gamma | +41.9% | +35.4% | 25% | −0.4 [−2.0; +1.3] | **+41.1% (1st)** | 43% | — | 372 |
| TimesFM 3.0 | +40.9% | +34.7% | 6% | −1.4 [−3.2; +0.6] | +39.6% (5th) | 5% | −1.5 [−3.6; +0.5] | 719 |
| Chronos-2 | +40.1% | +33.3% | 1% | −2.2 [−4.2; 0.0] | +39.6% (4th) | 4% | −1.5 [−3.3; +0.4] | 252 |
| Timestral-28M | +39.9% | +33.9% | 0% | −2.4 [−3.7; −1.0] | +39.0% (8th) | 0% | −2.1 [−4.1; −0.3] | 173 |
| Chronos-2 small | +38.6% | +32.3% | 0% | −3.7 [−6.2; −0.8] | +39.0% (7th) | 2% | −2.2 [−4.1; 0.0] | 61 |
| TiRex-2 | +37.7% | +31.0% | 1% | −4.5 [−7.9; −0.9] | +39.0% (6th) | 5% | −2.2 [−4.3; +0.4] | 415 |
| Timestral-5M | +37.2% | +31.4% | 0% | −5.0 [−7.6; −2.2] | +37.3% (9th) | 0% | −3.8 [−6.2; −1.7] | 72 |
| TimesFM 2.5 | +37.0% | +31.1% | 2% | −5.2 [−10.3; −0.7] | +32.2% (10th) | 0% | −8.8 [−13.4; −4.8] | 193 |

**On the 41 datasets that TimesFM 2.5's authors do not list, t0-gamma is first and Timestral-209M second, and the top three still
cannot be told apart.** The headline "Timestral-209M first" of the 48-dataset table therefore depends on the choice; what holds in
both is that Timestral-209M, Timestral-123M and t0-gamma cannot be told apart. Timestral-123M is 0.3 points behind the 209M with 62% of its
compute time, so on these data the largest size brings no demonstrable gain. Timestral-5M, in 72 s, is level with TiRex-2 and
TimesFM 2.5. The six models of an earlier run reproduced bit for bit (to 2e-6 for t0-gamma) when the ten were run together, also
with the GPU shared with other programs: sharing changed neither scores nor, visibly, times.

## 3. The 12 largest datasets (50 MB and above)

8 without covariates, 4 with. One run, 19 minutes. With so few datasets the intervals are wide; this run confirms the two main
rankings and does not replace them.

| Model | All 12 | Without the 2 declared (10) | Difference from Timestral-209M (90%) | Only the 8 without covariates |
|---|---|---|---|---|
| Timestral-209M | +45.8% | **+46.7% (1st)** | — | **48.5** |
| t0-gamma | +45.1% | +46.3% | −0.3 [−3.2; +1.8] | 48.1 |
| Chronos-2 | +44.9% | +46.1% | −0.6 [−2.5; +1.6] | 48.4 |
| Timestral-123M | +45.0% | +45.9% | −0.8 [−1.5; −0.3] | 47.7 |
| TiRex-2 | +44.6% | +45.8% | −1.0 [−3.0; +1.8] | 48.1 |
| Timestral-28M | +44.8% | +45.7% | −0.9 [−1.6; −0.4] | 47.8 |
| TimesFM 3.0 | +44.6% | +45.7% | −0.9 [−3.4; +1.1] | 47.8 |
| TimesFM 2.5 | **+49.9% (1st)** | +44.9% | −1.8 [−4.0; +0.1] | 47.6 |
| Timestral-5M | +43.6% | +44.3% | −2.4 [−4.9; −0.5] | 46.3 |
| Chronos-2 small | +41.7% | +42.0% | −4.6 [−8.6; −1.8] | 43.5 |

**Why TimesFM 2.5 is first with all 12.** On `kdd_cup_2022_10T` (wind-turbine power, 48 hours ahead) TimesFM 2.5's error is 0.15
times seasonal naive's, against 0.55–0.61 for the other nine, about four times lower than any of them, in all three windows and
from the first hours of the forecast. It is not the procedure (every model goes through the same code, and the context ends before
the window) nor covariates (TimesFM 2.5 ignores them). It is the training data: TimesFM 2.5's authors list this dataset, and nine
others, as part of its training data in the fev-bench results they submitted (see "Training data"). The error profile along the
horizon did not show it by itself (TimesFM 2.5's error grows with the horizon like the others', only from three times lower), so
the declaration, not the shape of the error, is the evidence. The two datasets of this run that it lists (`kdd_cup_2022_10T`,
`m5_1D`) are marked in the Catalog and left out of the ranking by default; the "Without" column is that ranking (10 datasets).

## Training data

What the makers state, checked on 28 October 2026:

- **Timestral** (Mistral AI): "the majority of the training data was synthetic"; real data only in a second stage, with a
  context four times longer; no real source is named; Mistral says nothing it backtests was seen in pre-training, which is a
  statement about its process and not an overlap analysis. The model card and the repository add nothing; a technical report is
  announced. The fev-bench result files it submitted declare no overlap on any of the 100 tasks.
- **TimesFM 2.5** (Google): the result files it submitted list ten fev-bench datasets as training data: `favorita_transactions_1D`,
  `favorita_transactions_1W`, `fred_md_2025--cee`, `fred_md_2025--macro`, `kdd_cup_2022_10T`, `m5_1D`, `m5_1W`, `proenfo_gfc12`,
  `proenfo_gfc14`, `proenfo_gfc17`. Declared overlap did not always give an advantage: only the `kdd_cup_2022_10T` and `proenfo_gfc*`
  datasets show a large one; on `m5_*`, `favorita_*` and `fred_md_*` it ranks with the others or behind.
- **Chronos-2, TimesFM 3.0, t0-gamma, TiRex-2**: their result files declare no overlap (Chronos-2 small has no file of its own, so nothing is declared for it). That is a self-declaration and not proof of
  absence; t0-gamma's authors describe the training data only as curated public datasets plus synthetic series, and TiRex-2's
  as two public corpora (LOTSA and the Chronos datasets) that overlap many benchmark datasets.
- **GIFT-Eval and Monash datasets**: nobody declares anything for them. Two results beyond the others remain unexplained and are
  not excluded: `monash.oikolab_weather` (TimesFM 2.5 at 0.34 times the median error of the other nine) and `gift.solar.H`
  (t0-gamma at 0.49). Excluding a model's best datasets on a suspicion alone would penalise it without evidence, and the
  conclusions do not change without them (Timestral-209M stays first in the 127-dataset ranking).

## Defects found by measuring, and fixed

- **Extreme scales (float32).** `monash.bitcoin` holds a hash rate that reaches 2e20; squaring it overflows float32, and Chronos-2,
  TiRex-2, TimesFM 2.5 and t0-gamma returned garbage or NaN on that series (t0-gamma, which batches series together, also spoiled the
  other 17). Not dirty data: dividing the series by its mean made every model normal. The base adapter now divides any series (or
  covariate) outside 1e-12…1e15 by its largest absolute value before the model and scales the forecast back, with a warning;
  series inside the range are not touched (bit-identical). Of the 175 datasets then downloaded only this one was outside it. A first
  version of the 127-dataset table, made before the fix, put Chronos-2 ninth and t0-gamma fifth; it is superseded.
- **TiRex-2 horizon.** The package silently truncated horizons above 320 steps (it only printed a warning). The adapter now makes the
  passes itself, each continuing from the median of the previous one; later intervals tend to be too narrow (measured on a
  sinusoid: 80% interval width 0.17 → 0.10 from the first to the last pass). The other nine models return the full horizon.
- **Not counted.** `monash.solar_4_seconds` is all zeros (night) over its context and window, so seasonal naive's error scale is
  zero and no score is defined for any model: 127 datasets, 126 count. It is listed under "Not counted" with the reason.

## Reproducing and checking

- Everything runs from the app: Leaderboard → pick the benchmark and the models, 100 series, 3 windows, context 2048, covariates on.
  Datasets are downloaded from Hugging Face at pinned revisions and checked against their SHA-256; weights at pinned commits.
- Runs are stored under `runs/suites/` (git-ignored); this file is the record. Reproducibility was checked on this one machine
  only (identical errors between two runs); other hardware may differ in the last digits.
- Not covered: datasets under 50 MB were measured on up to 100 series (not all), a single pass for times, and no confidence on
  the series sampled inside each dataset.
