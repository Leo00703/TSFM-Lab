#!/usr/bin/env python3
"""Environment report: versions, GPUs, where the disk space goes, and brand-asset status."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from _common import CACHE, ROOT, TOOLS, dir_stats, human, project_env, venv_python

PROBE = r"""
import json
info = {}
try:
    import tsfm_lab
    info["tsfm_lab"] = tsfm_lab.__version__
except Exception as e:
    info["tsfm_lab"] = "ERROR: %s" % e
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    info["threads"] = torch.get_num_threads()
    info["gpus"] = [
        {"name": torch.cuda.get_device_name(i),
         "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
         "capability": list(torch.cuda.get_device_capability(i))}
        for i in range(torch.cuda.device_count())
    ]
    info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
except Exception as e:
    info["torch"] = "ERROR: %s" % e
try:
    import importlib.metadata as md, json as _j
    info["timestral"] = md.version("timestral")
    direct = md.distribution("timestral").read_text("direct_url.json")
    if direct:
        info["timestral_commit"] = _j.loads(direct).get("vcs_info", {}).get("commit_id")
except Exception as e:
    info["timestral"] = "ERROR: %s" % e
import os
info["isolation"] = {k: os.environ.get(k) for k in ("HF_HOME", "TORCH_HOME", "TEMP", "UV_CACHE_DIR")}
print("@@" + json.dumps(info))
"""


def line(label: str, value: object) -> None:
    print(f"  {label:<22} {value}")


def main() -> None:
    print("== System")
    line("OS", f"{platform.system()} {platform.release()} ({platform.machine()})")
    line("Python (this script)", sys.version.split()[0])
    line("Project root", ROOT)

    py = venv_python()
    print("\n== Virtual environment")
    if not py.exists():
        line(".venv", "MISSING: run python scripts/bootstrap.py")
    else:
        result = subprocess.run([str(py), "-c", PROBE], env=project_env(), capture_output=True, text=True, cwd=ROOT)
        payload = next((ln[2:] for ln in result.stdout.splitlines() if ln.startswith("@@")), None)
        if payload is None:
            line("probe", "failed: " + (result.stderr.strip().splitlines() or ["no output"])[-1])
        else:
            info = json.loads(payload)
            line("tsfm_lab", info.get("tsfm_lab"))
            line("torch", f"{info.get('torch')} (CUDA build: {info.get('cuda_build')})")
            line("CUDA available", info.get("cuda_available"))
            for gpu in info.get("gpus", []):
                line("GPU", f"{gpu['name']}, {gpu['vram_gb']} GB, compute capability {gpu['capability']}")
            line("MPS available", info.get("mps_available"))
            line("CPU threads", info.get("threads"))
            line("timestral", f"{info.get('timestral')} @ {info.get('timestral_commit')}")
            print("  isolation env (as seen by the project interpreter):")
            for key, value in info["isolation"].items():
                print(f"    {key}={value}")

    print("\n== Disk usage (everything below lives inside the project folder)")
    rows = [("[.venv]", ROOT / ".venv"), ("[tools]", TOOLS)]
    if CACHE.exists():
        rows += [(f"[.cache/{p.name}]", p) for p in sorted(CACHE.iterdir()) if p.is_dir() and p != TOOLS]
    rows.append(("[web/assets/mistral]", ROOT / "web" / "assets" / "mistral"))
    # uv hardlinks .venv files to its cache when both are on the same volume: count that space once.
    probe = ROOT / ".venv" / "Lib" / "site-packages" / "numpy" / "__init__.py"
    if not probe.exists():
        probe = ROOT / ".venv" / "lib" / next(iter((ROOT / ".venv" / "lib").glob("python*")), Path("x")).name / "site-packages" / "numpy" / "__init__.py"
    shared = probe.exists() and probe.stat().st_nlink > 1
    total = 0
    for name, path in rows:
        files, size, _ = dir_stats(path)
        if not files:
            continue
        is_uv_cache = path == CACHE / "uv"
        note = "  (hardlinked with .venv: no extra disk)" if (is_uv_cache and shared) else ""
        if not (is_uv_cache and shared):
            total += size
        line(name, f"{human(size)} in {files} files{note}")
    line("TOTAL on disk", human(total))

    print("\n== Tools and assets")
    tailwind = sorted(TOOLS.glob("tailwindcss-*")) if TOOLS.exists() else []
    line("Tailwind CLI", tailwind[0].name if tailwind else "missing")
    fonts = list((ROOT / "web" / "assets" / "mistral" / "fonts").glob("*.woff2"))
    logos = list((ROOT / "web" / "assets" / "mistral" / "logos").glob("*"))
    line("ALTMistral font", f"{len(fonts)} files" if fonts else "missing (UI falls back to Inter)")
    line("Timestral logos", f"{len(logos)} files" if logos else "missing (UI uses a neutral icon)")


if __name__ == "__main__":
    main()
