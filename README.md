# TSFM-Lab

A local sandbox to **run, compare and inspect time-series foundation models**, starting with
Mistral's [Timestral](https://github.com/mistralai/timestral) family (5M / 28M / 123M / 209M).
Models run entirely on your machine, on CPU or GPU (CUDA), and every result is shown with the
measurements and configuration needed to judge it honestly.

> **Unofficial.** This is an independent tool, not affiliated with or endorsed by Mistral AI.

**Status:** Phases 0-2 done (isolated environment, model/dataset/metric core, CLI, local web app).
The full plan, research notes and design decisions are in [`RESOCONTO.md`](RESOCONTO.md) (Italian).

## Quick start

Requires Python 3.12 or 3.13 and `git`. [`uv`](https://docs.astral.sh/uv/) is used if installed,
otherwise a private copy is installed inside the project.

```bash
python scripts/bootstrap.py            # .venv + PyTorch (CUDA build if an NVIDIA GPU is found) + timestral
python scripts/bootstrap.py --device cpu   # force the small CPU-only PyTorch instead
```

```bash
# Windows                                   # Linux / macOS
.venv\Scripts\python scripts/doctor.py      .venv/bin/python scripts/doctor.py
.venv\Scripts\python scripts/smoke_test.py  .venv/bin/python scripts/smoke_test.py
```

`smoke_test.py` downloads Timestral-5M (21 MB), checks the forecast shapes and quantile ordering,
and benchmarks CPU vs GPU with warm-up and proper synchronisation.

## Nothing is left behind

Everything the project installs or caches lives **inside this folder**: the virtual environment
(`.venv/`), model weights, datasets and every package/compiler cache (`.cache/`). Deleting the
folder removes it all. `scripts/leakcheck.py` proves it by comparing global cache locations
before and after an install.

| Script | Purpose |
|---|---|
| `scripts/bootstrap.py` | Create the environment, install PyTorch, `timestral` (pinned commit) and the Tailwind CLI |
| `scripts/doctor.py` | Versions, GPUs, disk usage, tool status |
| `scripts/smoke_test.py` | Run Timestral on every available device |
| `scripts/leakcheck.py` | `--mark` before installing, `--report` after: verifies nothing was written elsewhere |
| `scripts/clean.py` | Remove `.venv` and/or `.cache` |
| `scripts/fetch_brand_assets.py` | Optional: fetch Mistral brand assets for local private use (see below) |

## Platforms

| Platform | Compute | Notes |
|---|---|---|
| Windows | CPU, CUDA | Developed and tested here |
| Linux | CPU, CUDA | Same code path; not yet tested |
| macOS | CPU | Apple GPU (MPS) is not enabled: the upstream package uses float64 on the device ([upstream issue #6](https://github.com/mistralai/timestral/issues/6)); not yet tested |

## Licences and third-party material

- The `timestral` Python package is Apache-2.0. The **model weights** are under Mistral's
  "Modified MIT" licence, which withholds all rights if your company's (or employer's) global
  consolidated **monthly** revenue exceeds US$20 million. Read it before use:
  <https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE>. Weights are downloaded
  at run time and are never part of this repository.
- Mistral's brand assets (the ALTMistral font and the Timestral logos) belong to Mistral AI.
  They are **not** in this repository; `scripts/fetch_brand_assets.py` downloads them to the
  git-ignored `web/assets/mistral/` for local private use only. Without them the UI falls back to Inter.
- Inter and Space Mono (`web/fonts/`) are licensed under the SIL Open Font License.
- Licence for this project's own code: not yet chosen.

## Web app

```bash
.venv\Scripts\python -m tsfm_lab serve          # http://127.0.0.1:8765 (local only; .venv/bin/python on Linux/macOS)
```

Dark by default with a light theme switch, responsive down to phone width. Pages: Overview, Models (download weights,
specs), Datasets (bundled scenarios, your own files), Playground (single forecast), Experiments (backtests),
Runs (every metric, calibration, per-step error, reproducibility), System (CPU vs GPU benchmark). The compute device
(CPU or CUDA) is selectable at any time, even when a GPU is present. The UI is built with Tailwind (standalone CLI
installed by `bootstrap.py`) and plain JavaScript; `scripts/fetch_vendor.py` re-fetches the two vendored libraries.

Inference runs in a separate worker process, so a PyTorch/CUDA crash cannot take the server down. To reach the app from
your other devices over Tailscale: `python -m tsfm_lab serve --tailscale` (listens on loopback and your tailnet address only;
there is no login, so anyone on your tailnet can use it).

## Command line

```bash
.venv\Scripts\python -m tsfm_lab models        # registered models (use .venv/bin/python on Linux/macOS)
.venv\Scripts\python -m tsfm_lab devices       # CPU / CUDA devices
.venv\Scripts\python -m tsfm_lab datasets      # bundled synthetic scenarios
.venv\Scripts\python -m tsfm_lab run --model timestral-5m --dataset multi_seasonality --device cuda --horizon 48
.venv\Scripts\python -m tsfm_lab run --model timestral-28m --dataset file:my_data.csv --targets sales
```

`run` backtests the model and the baselines on identical rolling windows and stores the run under `runs/`.
To add a model, write one `ModelAdapter` and register it (see `src/tsfm_lab/models/`).
