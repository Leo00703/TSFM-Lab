# Contributing to TSFM-Lab

Thanks for looking. This is a one-person project, so the process is light: open an issue or a pull request, and say what you saw or want.

## Before you start

- **A bug or a question:** open an issue with the form (it asks for the version, the system and what you ran). A security problem does **not**
  go in a public issue: see [`SECURITY.md`](SECURITY.md).
- **A new feature or a new model:** open an issue first, so we can agree that it fits before you spend the time. A model needs a licence that
  allows it, weights in `safetensors` (never pickle), and a look at what its authors say about their training data, because the Leaderboard
  flags datasets a model was trained on.
- **Small fixes** (typos, wording, a clear bug with a test) can go straight to a pull request.

## Set up and check

```bash
python scripts/bootstrap.py                 # creates .venv, installs everything inside the project folder (python3 on Linux and macOS)
.venv/bin/python -m pytest                  # Windows: .venv\Scripts\python -m pytest   (no model weights needed)
.venv/bin/ruff check .                      # Windows: .venv\Scripts\ruff check .
```

Set `CUDA_VISIBLE_DEVICES=""` to keep the tests off the GPU. Do not run `ruff format`: the code is not auto-formatted, and a reformat would
bury your change in noise. The README's [Development](README.md#development) section describes the layout of `src/tsfm_lab/`.

## What a good pull request looks like

- **One change, with a test.** A bug fix comes with a test that fails without it. A change to the interface is covered by the browser smoke
  test (`tests/test_ui_smoke.py`) if it adds a page or a control.
- **Match the surrounding code**: names, comment density, idiom. English only, in the code and in the interface.
- **Keep the isolation.** Nothing the project installs, downloads or caches may leave its folder (`.venv`, `.cache`); `tests/test_isolation.py`
  checks it.
- **No proprietary material.** Mistral's font and logos are not in the repository and must not be added; screenshots in the README are taken
  from a copy without them.
- **Say what you measured.** A claim about model quality (a ranking, a number in `RESULTS.md`) needs the run that produced it and its caveats.

The cross-platform check (Linux, Windows, macOS) runs on every pull request that touches code; it needs to be green. It runs on CPU: GPU paths
are checked by hand, so say in the pull request if you tested one.

## Licence

By contributing you agree that your contribution is licensed under the [Apache License 2.0](LICENSE), like the rest of the project.
