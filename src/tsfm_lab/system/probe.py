"""Describe the machine and software stack a result was produced on (stored with every run)."""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata
from typing import Any

from tsfm_lab.system.devices import cpu_name, list_devices


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _timestral_commit() -> str | None:
    try:
        import json

        raw = metadata.distribution("timestral").read_text("direct_url.json")
        return json.loads(raw).get("vcs_info", {}).get("commit_id") if raw else None
    except Exception:  # noqa: BLE001
        return None


def environment_info() -> dict[str, Any]:
    import psutil

    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in ("torch", "numpy", "pandas", "timestral", "tsfm-lab")},
        "timestral_commit": _timestral_commit(),
        "devices": [d.__dict__ for d in list_devices()],
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available() and torch.cuda.device_count() > 0  # CUDA_VISIBLE_DEVICES="" can leave the first True with no device
        if info["cuda_available"]:
            info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - a broken CUDA driver must not take the System page down: it is only a report
        info["cuda_available"] = False
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
        info["tsfm_lab_commit"] = sha.stdout.strip() or None
    except Exception:  # noqa: BLE001
        info["tsfm_lab_commit"] = None
    return info
