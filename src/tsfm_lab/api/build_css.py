"""Compile web/src/app.css with the standalone Tailwind CLI (no Node) when the output is missing or stale."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from tsfm_lab.isolation import find_root


def tailwind_binary():
    tools = find_root() / ".cache" / "tools"
    found = sorted(tools.glob("tailwindcss-*")) if tools.exists() else []
    return found[0] if found else None


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
    scratch_root = root / ".cache" / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    # The standalone CLI unpacks a ~9 MB native module into its temp folder on every run and never removes it: it gets a folder of its own,
    # named in its environment (not guessed from tempfile, which pytest fixes before the isolation applies), that goes away with the run.
    with tempfile.TemporaryDirectory(prefix="tailwind-", dir=scratch_root, ignore_cleanup_errors=True) as scratch:
        env = {**os.environ, "TEMP": scratch, "TMP": scratch, "TMPDIR": scratch}
        result = subprocess.run([str(binary), "-i", str(src), "-o", str(out), "--minify"], cwd=root / "web",
                                capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False
    return True
