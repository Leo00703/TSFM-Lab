#!/usr/bin/env python3
r"""Download Timestral checkpoints into the project cache (.cache/huggingface).

    .venv\Scripts\python scripts/download_models.py 5m 28m 123m 209m
"""
import sys
import time

from timestral.loading import resolve_checkpoint

import tsfm_lab  # noqa: F401  (cache isolation)

SIZES = {"5m": "5M", "28m": "28M", "123m": "123M", "209m": "209M"}

for key in sys.argv[1:] or SIZES:
    repo = f"mistralai/Timestral-{SIZES[key.lower()]}"
    start = time.time()
    path = resolve_checkpoint(repo)
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 2**20
    print(f"{repo}: {size:.0f} MB in {time.time() - start:.0f} s -> revision {path.name}", flush=True)
