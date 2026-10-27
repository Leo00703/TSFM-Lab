"""Shared helpers for the project scripts (standard library only)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
TOOLS = CACHE / "tools"
IS_WINDOWS = os.name == "nt"


def _load_isolation():
    """Load src/tsfm_lab/isolation.py by path so nothing needs to be installed yet."""
    path = ROOT / "src" / "tsfm_lab" / "isolation.py"
    spec = importlib.util.spec_from_file_location("tsfm_lab_isolation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


isolation = _load_isolation()


def project_env() -> dict[str, str]:
    """A copy of os.environ with every cache redirected into the project."""
    env = dict(os.environ)
    isolation.apply(root=ROOT, environ=env)
    return env


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd: list, env: dict[str, str] | None = None, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", flush=True)
    return subprocess.run([str(c) for c in cmd], env=env or project_env(), check=check, cwd=ROOT, **kwargs)


def dir_stats(path: Path) -> tuple[int, int, float]:
    """(file_count, total_bytes, newest_mtime) of a directory tree; zeros if missing."""
    count = total = 0
    newest = 0.0
    if not path.exists():
        return 0, 0, 0.0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        else:
                            st = entry.stat(follow_symlinks=False)
                            count += 1
                            total += st.st_size
                            newest = max(newest, st.st_mtime)
                    except OSError:
                        continue
        except OSError:
            continue
    return count, total, newest


def human(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"
