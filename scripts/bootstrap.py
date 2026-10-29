#!/usr/bin/env python3
"""Create the isolated environment: .venv, PyTorch (CUDA or CPU), timestral, Tailwind CLI.

Everything is installed inside the project folder; the script can be re-run safely.

    python scripts/bootstrap.py                  # auto: CUDA build if an NVIDIA GPU is found
    python scripts/bootstrap.py --device cpu     # force the small CPU-only PyTorch
    python scripts/bootstrap.py --skip-tailwind
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request

from _common import IS_WINDOWS, ROOT, TOOLS, human, project_env, run

TAILWIND_VERSION = "4.3.3"
# SHA-256 of the official standalone binaries for v4.3.3 (from the release's sha256sums.txt).
TAILWIND_SHA256 = {
    "tailwindcss-windows-x64.exe": "e0e260ce048014e9268f6237ff18f8ccf02cef521cbd0ae04e82c2cdf7aa3955",
    "tailwindcss-linux-x64": "dc61b3ac6b8c9ca874c0cc4c57b2409791a64c5540404ca5f5367360babc313a",
    "tailwindcss-linux-arm64": "55fd0b241214eff3de1e8ee4f22796662f2d2e7a49bcfca7477cfd0bac398195",
    "tailwindcss-macos-x64": "7922e0953f2110c05976e3bf58f14e643d90427575e766b7d433f5f80cbee7e1",
    "tailwindcss-macos-arm64": "cdf646702987a743464dff4d9c60fd4480d1c1e73dd819a9a67f1078815dce9d",
}
EXTRA_FOR_DEVICE = {"cuda": "cu128", "cpu": "cpu", "pypi": "pypi"}


def detect_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if platform.system() == "Darwin":
        return "pypi"  # standard wheel (CPU; MPS is not enabled by default, see README)
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if probe.returncode == 0 and "GPU" in probe.stdout:
            return "cuda"
    return "cpu"


def find_uv(env: dict[str, str]) -> list[str]:
    """Use uv from PATH, or install a private copy inside .cache/tools."""
    found = shutil.which("uv")
    if found:
        return [found]
    venv_dir = TOOLS / "uv-venv"
    py = venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if not py.exists():
        TOOLS.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", venv_dir], env=env)
    run([py, "-m", "pip", "install", "uv"], env=env)
    return [str(py), "-m", "uv"]


def tailwind_asset() -> str:
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        return "tailwindcss-windows-x64.exe"
    if system == "Darwin":
        return "tailwindcss-macos-arm64" if arm else "tailwindcss-macos-x64"
    return "tailwindcss-linux-arm64" if arm else "tailwindcss-linux-x64"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_tailwind() -> str:
    asset = tailwind_asset()
    expected = TAILWIND_SHA256[asset]
    dest = TOOLS / f"tailwindcss-{TAILWIND_VERSION}{'.exe' if IS_WINDOWS else ''}"
    if dest.exists() and sha256(dest) == expected:
        print(f"Tailwind CLI {TAILWIND_VERSION}: already present and verified ({dest.name})")
        return str(dest)
    TOOLS.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/tailwindlabs/tailwindcss/releases/download/v{TAILWIND_VERSION}/{asset}"
    part = dest.with_name(dest.name + ".part")
    print(f"Downloading Tailwind CLI {TAILWIND_VERSION} ({asset}) from the official release...")
    request = urllib.request.Request(url, headers={"User-Agent": "tsfm-lab-bootstrap"})
    with urllib.request.urlopen(request) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
        print(f"  downloaded {human(done)}" + (f" of {human(total)}" if total else ""))
    if sha256(part) != expected:
        part.unlink(missing_ok=True)
        sys.exit("Tailwind CLI checksum mismatch: file discarded. Nothing was installed.")
    part.replace(dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
    print(f"Tailwind CLI verified against the pinned SHA-256: {dest.name}")
    return str(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "pypi"], default="auto",
                        help="PyTorch build to install (default: auto-detect)")
    parser.add_argument("--skip-tailwind", action="store_true")
    parser.add_argument("--brand-assets", action="store_true",
                        help="also fetch Mistral's font and logos for local private use (they belong to Mistral AI; off by default)")
    parser.add_argument("--skip-brand-assets", action="store_true", help=argparse.SUPPRESS)  # the old default, still accepted
    args = parser.parse_args()

    env = project_env()
    device = detect_device(args.device)
    extra = EXTRA_FOR_DEVICE[device]
    print(f"Project root : {ROOT}")
    print(f"PyTorch build: {device} (extra '{extra}')")
    print("All caches   : " + str(ROOT / ".cache"))

    uv = find_uv(env)
    sync = [*uv, "sync", "--extra", extra]
    if (ROOT / "uv.lock").exists():
        sync.append("--locked")
    run(sync, env=env)

    if not args.skip_tailwind:
        ensure_tailwind()
    if args.brand_assets and not args.skip_brand_assets:
        run([sys.executable, ROOT / "scripts" / "fetch_brand_assets.py"], env=env)

    print("\nDone. Next:")
    py = ".venv\\Scripts\\python" if IS_WINDOWS else ".venv/bin/python"
    print(f"  {py} scripts/doctor.py        # environment report")
    print(f"  {py} scripts/smoke_test.py    # Timestral-5M on CPU and GPU")


if __name__ == "__main__":
    main()
