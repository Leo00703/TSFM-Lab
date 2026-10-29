# Security model

> Moved out of the [README](../README.md#security) to keep it short; nothing was changed. To report a vulnerability, see [`SECURITY.md`](../SECURITY.md).

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
