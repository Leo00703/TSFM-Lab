"""Where the project's disk space goes (everything lives inside the project folder)."""

from __future__ import annotations

import os
from pathlib import Path


def _stats(path: Path) -> tuple[int, int]:
    files = size = 0
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        else:
                            files += 1
                            size += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return files, size


def usage(root: Path) -> list[dict]:
    rows = [(".venv", root / ".venv"), ("model weights (.cache/huggingface)", root / ".cache" / "huggingface"),
            ("package caches (.cache/uv, pip, torch...)", None), ("tools (.cache/tools)", root / ".cache" / "tools"),
            ("your datasets (data/user)", root / "data" / "user"), ("runs and benchmarks (runs/)", root / "runs")]
    out = []
    cache = root / ".cache"
    other = [p for p in cache.iterdir() if p.is_dir() and p.name not in ("huggingface", "tools")] if cache.exists() else []
    for label, path in rows:
        if path is None:
            files = size = 0
            for p in other:
                f, s = _stats(p)
                files, size = files + f, size + s
        else:
            files, size = _stats(path)
        out.append({"label": label, "files": files, "bytes": size})
    return out
