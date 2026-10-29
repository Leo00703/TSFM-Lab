"""Compile web/src/app.css with the standalone Tailwind CLI (no Node) when the output is missing or stale."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path

from tsfm_lab.isolation import find_root


def tailwind_binary():
    tools = find_root() / ".cache" / "tools"
    found = sorted(tools.glob("tailwindcss-*")) if tools.exists() else []
    return found[0] if found else None


def _native_leftovers() -> set[Path]:
    """The standalone CLI unpacks a ~9 MB native module into the temp folder on every run and never removes it."""
    return set(Path(tempfile.gettempdir()).glob(".*.node"))


def ensure_css(force: bool = False) -> bool:
    root = find_root()
    src, out = root / "web" / "src" / "app.css", root / "web" / "dist" / "app.css"
    inputs = [src, *(root / "web" / "js").rglob("*.js"), root / "web" / "index.html"]
    if not force and out.exists() and out.stat().st_mtime >= max(p.stat().st_mtime for p in inputs if p.exists()):
        return True
    binary = tailwind_binary()
    if binary is None:
        print("Tailwind CLI not found: run scripts/bootstrap.py", file=sys.stderr)
        return out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    before = _native_leftovers()
    result = subprocess.run([str(binary), "-i", str(src), "-o", str(out), "--minify"], cwd=root / "web",
                            capture_output=True, text=True, check=False)
    for leftover in _native_leftovers() - before:  # only what this run left; one still in use by another run stays
        with contextlib.suppress(OSError):
            leftover.unlink()
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False
    return True
