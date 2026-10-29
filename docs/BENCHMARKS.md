# Online benchmarks

> Moved out of the [README](../README.md#web-app) to keep it short; nothing was changed. The measured results are in [`RESULTS.md`](../RESULTS.md).

The Datasets page lists **187 curated datasets** from three public benchmarks, none of them downloaded until you ask. The first is
[fev-bench](https://github.com/autogluon/fev): 100 tasks over 96 files of
[`autogluon/fev_datasets`](https://huggingface.co/datasets/autogluon/fev_datasets) at a fixed revision (about 650 MB in all).
It is a finite, verified list, not a search box: `scripts/build_catalog.py` read every file's schema and the exact length of every
series straight from Hugging Face, checked a sample of each (missing values, time axis, constant series, text columns) and took
from fev-bench which columns are the target, known in advance, past-only or static, plus the recommended horizon and seasonality.
The result is committed as `src/tsfm_lab/data/catalog_fev.json`.

The other two are listed the same way: [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval)
(55 datasets, about 1.2 GB, Arrow files; horizons and windows follow the benchmark's own rules, including the medium and long terms of
sub-daily data) and the [Monash archive](https://huggingface.co/datasets/Monash-University/monash_tsf) (32 datasets, about 0.6 GB, `.tsf`
files read by a parser that never executes anything). `scripts/build_catalog_ext.py gift|monash` read every file completely once and
recorded exact lengths, missing values and the SHA-256 each download is checked against (`catalog_gift.json`, `catalog_monash.json`).
Data that two benchmarks share are marked "also in", every entry carries its licence and a contamination note, and nothing is
downloaded until you press Download.

The **Leaderboard** page runs several models on a whole benchmark (fev-bench, GIFT-Eval or the Monash archive, optionally limited by file
size) and ranks them. Every model is loaded once and forecasts the same windows of every dataset, with the horizon and windows of the
benchmark's own task (windows capped, a seeded sample of series, the same context cap for all). The score is the fev-bench skill: the
geometric mean over datasets of the model's error divided by seasonal naive's, on SQL and MASE; only datasets that every ranked model
completed, and on which seasonal naive has an error to score against, count, and models with different quantile levels are scored on the levels they share. Missing datasets are downloaded first
(after confirmation), results are stored under `runs/suites/` and can be reopened. The run is saved after every dataset: if you press Stop, or the
computation fails, the datasets finished so far are kept and the ranking says how many it counts.
The Leaderboard also ranks the models on **your own datasets** (the "Your datasets" tab, files added on the Datasets page): one horizon for
all of them, the last windows of each series, the same scoring. Data recorded after the models' training data were collected cannot have been
seen by them, which public benchmarks cannot promise.
A dataset that a ranked model's own authors list as training data (the `trained_on_this_dataset` column of the fev-bench results they submitted: ten
datasets for TimesFM 2.5, none for the other models) carries a warning in the Catalog and is left out of the ranking by default; the Leaderboard says so
and a button counts it anyway. On one of them, `kdd_cup_2022_10T`, TimesFM 2.5 makes a quarter of every other model's error.

Nothing is downloaded until you press **Download** on an entry, which shows its size and the licence note first. The file is
checked against the SHA-256 published by Hugging Face and discarded if it differs. Once downloaded, an entry can be previewed,
forecast in the Playground and backtested in Experiments like any other dataset; a very large file (30,000 series) is sampled
with a fixed seed (200 series by default, adjustable) and says so. Text columns such as holiday names become on/off flags;
static per-series attributes are not used. The cards declare the licence as "other" and the data come from public sources:
check their terms before reusing the data. Because the real training data of most models is not listed, these datasets are marked as
possibly seen: Mistral says Timestral's was mostly synthetic and that nothing it backtests was seen in pre-training, but names no real source and publishes no overlap analysis. Synthetic scenarios are the only guaranteed-unseen ones.
