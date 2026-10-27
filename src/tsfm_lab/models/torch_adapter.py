"""Shared lifecycle of adapters whose model runs on a torch device (CPU, CUDA or Apple's Metal).

An adapter only says how to *load* its model on a device and how to run *one* forecast; this base takes care of what must
behave the same for every model: picking the device, continuing on the CPU (with a visible note) when a GPU is missing,
fails to load or fails while forecasting, releasing GPU memory, and timing with proper synchronisation. A benchmark sets
``strict_device`` so that a failure is an error instead of a quiet switch to the CPU.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

from tsfm_lab.models.base import DeviceInfo, ForecastRequest, ForecastResult, ModelAdapter
from tsfm_lab.system.devices import list_devices, pick_device


class TorchAdapter(ModelAdapter):
    def __init__(self) -> None:
        self._model: Any = None
        self._device: str | None = None
        self._load_meta: dict[str, Any] = {}
        self._fallback_note: str | None = None  # why the model runs on the CPU although a GPU was requested

    # ------------------------------------------------------------------ what a model provides
    @abstractmethod
    def _load_on(self, device: str) -> dict[str, Any]:
        """Load the weights onto ``device``, store the model in ``self._model``, return extra load metadata."""

    @abstractmethod
    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        """One forecast on ``self._device``; may raise on a GPU problem (the caller then retries on the CPU)."""

    # ------------------------------------------------------------------ lifecycle
    def available_devices(self) -> list[DeviceInfo]:
        return [d for d in list_devices() if d.kind in self.info.capabilities.device_kinds or not d.supported]

    @property
    def device(self) -> str | None:
        return self._device

    def load(self, device: str = "cpu") -> dict[str, Any]:
        requested = device
        device, note = pick_device(device, self.info.capabilities.device_kinds, self.strict_device)
        if self._model is not None:
            self.unload()
        start = time.perf_counter()
        try:
            meta = self._load_on(device)
            self._synchronize(device)
        except Exception as exc:  # noqa: BLE001 - driver, memory or an unsupported operation on the GPU
            if device == "cpu" or self.strict_device:
                raise
            self._model = None
            self._release(device)
            note = f"{device} could not load the model ({type(exc).__name__}: {str(exc)[:160]}); it runs on the CPU instead."
            device = "cpu"
            meta = self._load_on(device)
        self._fallback_note = note
        self._device = device
        self._load_meta = {
            "load_s": round(time.perf_counter() - start, 3), "device": device, **meta,
            **({"requested_device": requested, "fallback": note} if note else {}),
        }
        return dict(self._load_meta)

    def unload(self) -> None:
        self._model = None
        if self._device:
            self._release(self._device)
        self._device = None

    @staticmethod
    def _release(device: str) -> None:
        """Hand cached GPU memory back; a failing driver must not turn a clean unload into an error."""
        try:
            import torch

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _synchronize(device: str) -> None:
        import torch

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # ------------------------------------------------------------------ inference
    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if self._model is None or self._device is None:
            raise RuntimeError("model not loaded: call load(device) first")
        try:
            result = self._forecast_once(request)
        except (RuntimeError, NotImplementedError) as exc:  # includes out-of-memory and unsupported GPU operations
            if self._device == "cpu" or self.strict_device:
                raise
            failed = self._device
            self.load("cpu")
            self._fallback_note = f"{failed} failed while forecasting ({type(exc).__name__}: {str(exc)[:160]}); continued on the CPU."
            result = self._forecast_once(request)
        if self._fallback_note:
            result.warnings.insert(0, self._fallback_note)
        return result

    def _timed(self, work) -> tuple[Any, dict[str, Any]]:
        """Run ``work()`` between device synchronisations; returns its result and ``forecast_ms`` (+ peak VRAM on CUDA)."""
        import torch

        device = self._device or "cpu"
        cuda = device.startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        else:
            self._synchronize(device)
        start = time.perf_counter()
        out = work()
        self._synchronize(device)
        timings: dict[str, Any] = {"forecast_ms": round((time.perf_counter() - start) * 1000, 3)}
        if cuda:
            timings["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        return out, timings
