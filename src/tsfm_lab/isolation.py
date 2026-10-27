"""Keep every cache, temp file and tool download inside the project folder.

Single source of truth for the mapping: it is imported by the package (so the app
isolates itself even when started without the helper scripts) and loaded by file
path from ``scripts/`` (so the scripts need nothing installed). Deleting the project
folder therefore removes everything this project ever wrote.

Set ``TSFM_LAB_GLOBAL_CACHES=1`` to opt out (not recommended).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

OPT_OUT_VAR = "TSFM_LAB_GLOBAL_CACHES"


def find_root(start: Path | None = None) -> Path:
    """Project root = nearest parent holding pyproject.toml (src layout: parents[2])."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "tsfm_lab").is_dir():
            return parent
    return here.parents[2]


def cache_env(root: Path) -> dict[str, str]:
    """Environment variables that redirect third-party caches into ``<root>/.cache``."""
    c = root / ".cache"
    tmp = c / "tmp"
    return {
        "TSFM_LAB_ROOT": str(root),
        # Hugging Face (weights, datasets) and telemetry
        "HF_HOME": str(c / "huggingface"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        # PyTorch / CUDA / compilers
        "TORCH_HOME": str(c / "torch"),
        "TORCHINDUCTOR_CACHE_DIR": str(c / "torch" / "inductor"),
        "TRITON_CACHE_DIR": str(c / "triton"),
        "CUDA_CACHE_PATH": str(c / "cuda"),
        # Package managers and the Python they use
        "UV_CACHE_DIR": str(c / "uv"),
        "UV_PYTHON_INSTALL_DIR": str(c / "python"),
        "UV_PYTHON_PREFERENCE": "system",
        "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
        "PIP_CACHE_DIR": str(c / "pip"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        # Generic locations used by many libraries
        "XDG_CACHE_HOME": str(c / "xdg"),
        "MPLCONFIGDIR": str(c / "matplotlib"),
        "NUMBA_CACHE_DIR": str(c / "numba"),
        "PYTHONPYCACHEPREFIX": str(c / "pycache"),
        "TMPDIR": str(tmp),
        "TEMP": str(tmp),
        "TMP": str(tmp),
    }


def apply(root: Path | None = None, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Force the isolation variables into ``environ`` (default ``os.environ``).

    Returns the mapping that was applied (empty if the user opted out).
    """
    env = os.environ if environ is None else environ
    if env.get(OPT_OUT_VAR) == "1":
        return {}
    mapping = cache_env(root or find_root())
    if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in env:
        # Not a cache: an operation the Apple GPU lacks runs on the CPU instead of raising. torch reads it at import.
        mapping["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    for key, value in mapping.items():
        env[key] = value
    for key in ("HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME",
                "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "PYTHONPYCACHEPREFIX", "TEMP"):
        Path(mapping[key]).mkdir(parents=True, exist_ok=True)
    if environ is None:
        # Only read by the interpreter at startup, so set it explicitly for this process too.
        sys.pycache_prefix = mapping["PYTHONPYCACHEPREFIX"]
    return mapping
