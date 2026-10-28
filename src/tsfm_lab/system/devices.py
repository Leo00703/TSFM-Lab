"""Compute-device discovery shared by every adapter (CPU always; CUDA when present)."""

from __future__ import annotations

import os
import platform
import subprocess

from tsfm_lab.models.base import DeviceInfo


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except Exception:  # noqa: BLE001, S110 - purely informational
        pass
    return platform.processor() or platform.machine()


def mps_usable() -> bool:
    """True when the installed timestral routes its float64 statistics through the CPU (the Apple GPU has no float64).

    Detected by feature, not by platform: the official package lacks ``cpu_for_float64`` (upstream issue #6).
    """
    try:
        from timestral import scaler
    except ImportError:
        return False
    return hasattr(scaler, "cpu_for_float64")


def list_devices() -> list[DeviceInfo]:
    """CPU is always listed; CUDA GPUs when torch sees them; MPS when the installed timestral can use it."""
    import psutil

    devices = [DeviceInfo("cpu", "cpu", cpu_name(), round(psutil.virtual_memory().total / 2**30, 1))]
    try:
        import torch
    except ImportError:
        return devices
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(DeviceInfo(f"cuda:{i}", "cuda", torch.cuda.get_device_name(i),
                                          round(props.total_memory / 2**30, 1)))
    except Exception as exc:  # noqa: BLE001 - a broken driver must never take the CPU path down with it
        devices.append(DeviceInfo("cuda:0", "cuda", "CUDA GPU", None, supported=False,
                                  reason=f"the CUDA driver could not be queried ({type(exc).__name__})"))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        if mps_usable():
            devices.append(DeviceInfo("mps", "mps", f"{devices[0].name} GPU (Metal)", devices[0].memory_gb))  # unified memory
        else:
            devices.append(DeviceInfo(
                "mps", "mps", f"{devices[0].name} GPU (Metal)", None, supported=False,
                reason="this timestral build computes in float64 on the device, which MPS lacks "
                       "(upstream issue #6); use CPU, or the macOS build of the package"))
    elif platform.system() != "Darwin" and not torch.cuda.is_available() and os.environ.get("TSFM_LAB_HINT_CUDA", "1") == "1":
        devices.append(DeviceInfo("cuda:0", "cuda", "CUDA GPU", None, supported=False,
                                  reason="no CUDA device visible to this PyTorch build"))
    return devices


def pick_device(requested: str, allowed_kinds: tuple[str, ...], strict: bool = False) -> tuple[str, str | None]:
    """Like ``resolve_device``, but a GPU that is missing or unsupported falls back to the CPU (unless ``strict``).

    Returns ``(device, note)``; ``note`` explains the fallback and is None when the request was honoured.
    """
    try:
        return resolve_device(requested, allowed_kinds), None
    except ValueError as exc:
        if strict or requested.split(":")[0] == "cpu":
            raise
        return "cpu", f"{exc}; the forecast ran on the CPU instead."


def resolve_device(requested: str, allowed_kinds: tuple[str, ...]) -> str:
    """Validate a device id against what is present and what the adapter supports."""
    kind = requested.split(":")[0]
    if kind not in allowed_kinds:
        raise ValueError(f"device '{requested}' is not supported by this model (supports: {', '.join(allowed_kinds)})")
    present = {d.id: d for d in list_devices()}
    if kind == "cuda" and requested == "cuda":
        requested = "cuda:0"
    info = present.get(requested)
    if info is None or not info.supported:
        reason = info.reason if info else "not found"
        raise ValueError(f"device '{requested}' is unavailable: {reason}")
    return requested
