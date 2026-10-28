# TSFM-Lab

A local sandbox to **run, compare and inspect time-series foundation models**. Ten models from five families run on the same data
and are compared side by side: Mistral's [Timestral](https://github.com/mistralai/timestral) (5M / 28M / 123M / 209M), Amazon's
[Chronos-2](https://github.com/amazon-science/chronos-forecasting) (120M and 28M), Google's
[TimesFM](https://github.com/google-research/timesfm) 3.0 and 2.5, The Forecasting Company's
[t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma) and NXAI's [TiRex-2](https://github.com/NX-AI/tirex-2).
Any other model plugs in as one adapter file.
Models run entirely on your machine, on the CPU or a GPU (NVIDIA CUDA, Apple Metal), and every result is shown with the
measurements and configuration needed to judge it honestly.

> **Unofficial.** This is an independent tool, not affiliated with or endorsed by Mistral AI.

**Status:** working local web app and command line. The home page is a **Showcase** of real-data forecasts explained in plain
language (six scenarios, with the measured effect of the factors known in advance); the Playground, Experiments, Leaderboard, Runs and
System pages hold the technical detail. The full plan, research notes, run results and decisions are in [`RESOCONTO.md`](RESOCONTO.md)
(Italian).

## Quick start

Requires **Python 3.12 or 3.13** (3.14 is not supported yet) and `git`. [`uv`](https://docs.astral.sh/uv/) is used if installed,
otherwise a private copy is installed inside the project. The first run downloads PyTorch (about 2.5 GB for the NVIDIA CUDA
build, a few hundred MB otherwise) and a few small tools; model weights are downloaded later, from the app, only when you ask.

**Windows**

```bash
python scripts/bootstrap.py
.venv\Scripts\python -m tsfm_lab serve
```

**macOS** (install a supported Python first if needed: `brew install python@3.13`; the system `python` command does not exist, use `python3.13`)

```bash
python3.13 scripts/bootstrap.py
.venv/bin/python -m tsfm_lab serve
```

**Linux**

```bash
python3 scripts/bootstrap.py          # python3.12 or python3.13
.venv/bin/python -m tsfm_lab serve
```

`serve` opens `http://127.0.0.1:8765` in your browser (add `--no-browser` to skip that, `--port N` to change the port).
`bootstrap.py` picks the PyTorch build for you: the CUDA build when an NVIDIA GPU is found, the small CPU build otherwise, the
standard macOS build on a Mac. Force one with `--device cpu`, `--device cuda` or `--device pypi`.

In every command below, `.venv\Scripts\python` is the Windows interpreter and `.venv/bin/python` the Linux/macOS one.

To check the installation:

```bash
.venv/bin/python scripts/doctor.py        # Windows: .venv\Scripts\python scripts/doctor.py
.venv/bin/python scripts/smoke_test.py    # downloads Timestral-5M (21 MB), checks shapes and quantile order,
                                          # then benchmarks every available device with warm-up and synchronisation
```

## Nothing is left behind

Everything the project installs or caches lives **inside this folder**: the virtual environment
(`.venv/`), model weights, datasets and every package/compiler cache (`.cache/`). Deleting the
folder removes it all. `scripts/leakcheck.py` proves it by comparing global cache locations
before and after an install.

| Script | Purpose |
|---|---|
| `scripts/bootstrap.py` | Create the environment, install PyTorch, `timestral` (pinned commit) and the Tailwind CLI (pinned checksum) |
| `scripts/doctor.py` | Versions, GPUs, disk usage, tool status |
| `scripts/smoke_test.py` | Run Timestral on every available device |
| `scripts/check_device_parity.py` | Compare a GPU with the CPU: same forecasts? how much faster? (`--device mps`, `--model timestral-28m`) |
| `scripts/leakcheck.py` | `--mark` before installing, `--report` after: verifies nothing was written elsewhere |
| `scripts/clean.py` | Remove `.venv` and/or `.cache` |
| `scripts/build_catalog.py` | Rebuild and verify the online-dataset catalog from the pinned sources (maintainers only; reads from Hugging Face, stores nothing) |
| `scripts/download_models.py` | Download Timestral weights without opening the app |
| `scripts/fetch_brand_assets.py` | Optional: fetch Mistral brand assets for local private use (see below) |
| `scripts/fetch_vendor.py` | Re-fetch the two vendored JavaScript libraries (integrity-checked) |

## Models

| Family | Models | Licence | Covariates |
|---|---|---|---|
| Timestral (Mistral AI) | 209M, 123M, 28M, 5M | package Apache-2.0; weights "Modified MIT" with a US$20M monthly revenue limit | past, known-future, categorical |
| Chronos-2 (Amazon) | 120M, 28M ("small") | Apache-2.0 | past, known-future (numeric) |
| TimesFM 3.0 (Google) | 331M | code Apache-2.0; **weights non-commercial** (testing, evaluation, research only) | past, known-future (numeric) |
| TimesFM 2.5 (Google) | 231M | Apache-2.0 | none (univariate; covariates are ignored and the result says so) |
| t0-gamma (The Forecasting Company) | 256M | Apache-2.0 (a very young model: read its notes) | past, known-future (numeric) |
| TiRex-2 (NXAI) | 82M | Apache-2.0; one of its runtime libraries (`mlstm_kernels`) is under the NXAI Community Licence (free below EUR 100 million revenue, see its notes) | past, known-future (numeric), several variables together |

**Your own checkpoints.** The Models page can add a public Hugging Face repository (for example a fine-tune of a supported model) and run it
with the adapter of the model family you pick. This is deliberately narrow, not "run anything": the repository is inspected first (public,
a single `model.safetensors` plus `config.json`, at most 4 GB, licence shown), **only those two files are ever downloaded, at the exact commit
that was inspected**, so nothing in the repository can execute code (no pickle, no remote code) and a later push cannot change what runs.
Gated and private repositories are refused (the app holds no credentials). Context length, quantile levels and covariate support are taken
from the base model, which is right for a fine-tune of it and wrong for another architecture; you confirm that you understand these are
unverified third-party weights. Custom models appear on the Models page with a "custom checkpoint" tag and work everywhere a model does
(Playground, Experiments, Leaderboard); the list is kept in `data/user/custom_models.json` (git-ignored).

Weights are downloaded only when you ask, from the Models page (Chronos-2 is 478 MB, its small variant 112 MB, TimesFM 2.5 925 MB,
TimesFM 3.0 1.3 GB, t0-gamma 1 GB, TiRex-2 330 MB). **TimesFM 3.0's weights are under Google's non-commercial licence**: the card says so and asks you to
acknowledge it before downloading; do not use them for commercial or production purposes, and do not redistribute them. In the Playground
and in Experiments, choose a model and tick the others under "Compare with": they run on the same windows, with the same metrics.
Chronos-2 and t0-gamma use the same 21 quantile levels as Timestral; both TimesFM models and TiRex-2 produce nine deciles (0.1 to 0.9). When models with
different levels run in one backtest, **SQL and WQL are computed for every model on the levels they all share**, so those scores
(and the skill scores built on them) stay comparable; each model's own-level values are kept alongside, and the 90% and 98%
interval coverage is only reported for models that predict those tails. Timestral's options map to the closest equivalent
(symmetric averaging, keeping non-negative inputs non-negative) for TimesFM and are ignored by Chronos-2.

Adding a model means writing one `ModelAdapter` (see `src/tsfm_lab/models/chronos.py` for a short example built on the shared
`TorchAdapter`, which already handles device choice, the CPU fallback and memory release) and one manifest per checkpoint.

## Platforms and compute devices

| Platform | Compute | Notes |
|---|---|---|
| Windows | CPU, NVIDIA CUDA | Developed and tested here, including the GPU |
| Linux | CPU, NVIDIA CUDA | Installation and tests pass on a hosted Linux runner (CPU); CUDA on Linux is not tested by the author |
| macOS (Apple Silicon) | CPU, Apple GPU (MPS) | Installation and tests pass on a hosted macOS runner; the Apple GPU was verified on a MacBook Air M5 (see below) |

You choose the device from the selector at the top right of the app, at any time, even when a GPU is present. The default is
the CPU on every system. **If the chosen GPU is missing, fails to load, or fails while forecasting, the app continues on the
CPU without an error and says so in the result's warnings.** The CPU-vs-GPU benchmark on the System page is the one exception:
it fails instead of labelling CPU numbers as GPU numbers.

**Apple GPU.** The official `timestral` package cannot use Apple's GPU because it computes in float64, which Metal lacks
([upstream issue #6](https://github.com/mistralai/timestral/issues/6)). On macOS only, `pyproject.toml` therefore pins a
one-commit fork of the very same upstream commit that runs those float64 statistics on the CPU
([`paschembri/timestral@c49d40f`](https://github.com/paschembri/timestral/commit/c49d40f08dc75c05d7489a3ef26fb3fd565584c7)).
The change was reviewed before being pinned (3 files, no new dependency, no network, process or file access) and the pin is
by commit hash, so it cannot change underneath you. Windows and Linux use the official repository. Measured on a MacBook Air M5:
the GPU gives the same forecasts as the CPU (relative difference below 5e-5) and is about 1.3 times faster on the 209M model,
so it is an option, not the default. Check your own machine with
`.venv/bin/python scripts/check_device_parity.py --device mps --model timestral-28m --series 64`.

## Web app

Pages: **Showcase** (home), Models (download weights, specs, your own checkpoints), Datasets (your own files, online benchmarks, bundled
scenarios), Playground (one forecast, all quantiles), Experiments (rolling-window backtests against baselines), Leaderboard (several models on
a whole benchmark), Runs (every metric, calibration, per-step error, reproducibility, comparison of runs), System (CPU vs GPU benchmark,
disk use), About (licences, definitions). Long jobs (experiments, benchmarks, downloads,
measurements) run in the background, survive navigation and reloads, and can be stopped from the Activity panel.

Dark by default with a light theme switch, responsive down to phone width, English only. Inference runs in a separate worker
process, so a PyTorch/CUDA/Metal crash cannot take the server down. The UI is built with Tailwind (standalone CLI installed by
`bootstrap.py`) and plain JavaScript, with Preact and ECharts vendored.

**From your other devices (Tailscale):** `python -m tsfm_lab serve --tailscale` listens on loopback and your tailnet address
only. There is no login, so anyone on your tailnet can use it.

### Online benchmarks

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
A dataset where one model's result is far beyond what the others reach and nothing explains it (today only `kdd_cup_2022_10T`, where TimesFM 2.5 makes a quarter of every other
model's error) carries a warning in the Catalog and is left out of the ranking by default; the Leaderboard says so and a button counts it anyway.

Nothing is downloaded until you press **Download** on an entry, which shows its size and the licence note first. The file is
checked against the SHA-256 published by Hugging Face and discarded if it differs. Once downloaded, an entry can be previewed,
forecast in the Playground and backtested in Experiments like any other dataset; a very large file (30,000 series) is sampled
with a fixed seed (200 series by default, adjustable) and says so. Text columns such as holiday names become on/off flags;
static per-series attributes are not used. The cards declare the licence as "other" and the data come from public sources:
check their terms before reusing the data. Because the models' training data is not disclosed, these datasets are marked as
possibly seen; synthetic scenarios are the only guaranteed-unseen ones.

### Showcase data

The Showcase downloads small public datasets from Hugging Face (`autogluon/fev_datasets`, pinned revision) the first time you
open a scenario, after asking: electricity demand, store sales, solar power, online grocery sales, supermarket traffic and US
retail items. The repository card declares the licence as "other" and the data come from public sources: check their terms
before reusing the data. Nothing from your own files or runs is ever uploaded.

## Command line

```bash
.venv/bin/python -m tsfm_lab models        # registered models          (Windows: .venv\Scripts\python -m tsfm_lab ...)
.venv/bin/python -m tsfm_lab devices       # CPU / GPU devices and whether each is usable
.venv/bin/python -m tsfm_lab datasets      # bundled synthetic scenarios
.venv/bin/python -m tsfm_lab run --model timestral-5m --dataset multi_seasonality --device cuda --horizon 48
.venv/bin/python -m tsfm_lab run --model timestral-28m --dataset file:my_data.csv --targets sales
.venv/bin/python -m tsfm_lab runs          # stored runs
```

`run` backtests the model and the baselines on identical rolling windows and stores the run under `runs/`.
To add a model, write one `ModelAdapter` and register it (see `src/tsfm_lab/models/`).

## Security

The app is built to be safe to leave running on your own machine, and to say plainly where it is not.

- **Local only by default.** The server listens on `127.0.0.1` and refuses any request whose `Host` is not a local name (which also blocks
  DNS-rebinding), and any state-changing request from another origin. With `--tailscale` it also listens on your tailnet address; **there
  is no login**, so anyone on your tailnet can use the app and see your runs and datasets.
- **Nothing from outside is ever executed.** Weights are loaded from `safetensors` files only (never pickle, never `trust_remote_code`);
  the Monash `.tsf` files are read by a parser, and your uploads by `pandas` readers for CSV, TSV, Parquet and Excel. Catalog dataset downloads
  are pinned to a commit and checked against a SHA-256 before use, and a custom checkpoint is fetched at the exact commit that was inspected.
- **Uploads** are limited to 100 MB (refused as soon as the limit is passed, before the body is held in memory), must be one of the
  supported types, and are stored under `data/user/` with a name stripped of anything but letters, digits and `._-`.
- **The page** is served with a Content-Security-Policy (only its own scripts, no inline script, no framing), `X-Frame-Options: DENY` and
  `nosniff`; it loads nothing from another site. Run ids that come from a URL are checked against the exact shape the app generates, so
  no URL can reach a file outside `runs/`. CSV exports quote cells that a spreadsheet would run as a formula.
- **Inference runs in a separate process**, so a crash of PyTorch or the GPU driver cannot take the server down, and a dataset or a leaderboard
  run cannot be deleted while a job is using it.
- **Not covered:** a malicious file that exploits a bug in `pandas`, `pyarrow` or `openpyxl` while it is being read (keep them updated with
  `uv sync`; the lock file pins what was tested), and anything a person with access to your tailnet or to this folder can do.

## Development

```bash
.venv/bin/python -m pytest        # Windows: .venv\Scripts\python -m pytest   (no model weights needed; about a minute on a laptop CPU)
```

Set `CUDA_VISIBLE_DEVICES=""` to keep the tests off the GPU. The styles are built by the app itself when it starts (`web/src/app.css` into
`web/dist/app.css`, with the pinned Tailwind CLI); the JavaScript is plain modules, so there is no build step.
`src/tsfm_lab/` holds `models/` (adapters and manifests), `data/` (loaders, catalogs), `eval/` (backtests, metrics, the leaderboard),
`api/` (FastAPI app, jobs, the worker process), `store/` (runs on disk), `showcase/` and `system/`; `web/` is the interface and `tests/` the tests.

## Continuous integration

`.github/workflows/ci.yml` ("Cross-platform check") runs Linux, Windows and macOS through the reusable `check.yml` (one system). It
**never runs by itself**: start it from the Actions tab (Run workflow) or with `gh workflow run ci.yml -f systems=all` (or `linux`,
`windows`, `macos`). Each system runs the real
installation (`scripts/bootstrap.py`), the environment report, the tests without model weights, and the tests with the real
Timestral-5M checkpoint on the CPU. Hosted runners have no GPU, so the GPU paths are checked by hand. On private repositories
the minutes are multiplied (Linux x1, Windows x2, macOS x10): a full run costs roughly 15 billable minutes.

## Licences and third-party material

- The `timestral` Python package is Apache-2.0. The **model weights** are under Mistral's
  "Modified MIT" licence, which withholds all rights if your company's (or employer's) global
  consolidated **monthly** revenue exceeds US$20 million. Read it before use:
  <https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE>. Weights are downloaded
  at run time and are never part of this repository.
- Mistral's brand assets (the ALTMistral font and the Timestral logos) belong to Mistral AI.
  They are **not** in this repository; `scripts/fetch_brand_assets.py` downloads them to the
  git-ignored `web/assets/mistral/` for local private use only (`bootstrap.py` runs it unless you pass
  `--skip-brand-assets`). Without them the UI falls back to Inter.
- Inter and Space Mono (`web/fonts/`) are licensed under the SIL Open Font License.
- Licence for this project's own code: not yet chosen.
