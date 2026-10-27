"""Adapter for Mistral's Timestral models (package ``timestral``, pinned in pyproject)."""

from __future__ import annotations

import json
import time
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from tsfm_lab.models.base import (Capabilities, DeviceInfo, ForecastRequest, ForecastResult, ModelAdapter,
                                  ModelInfo)
from tsfm_lab.models.covariates import inspect_known_covariates, mask_flat_rows
from tsfm_lab.system.devices import list_devices, pick_device

MANIFEST_PACKAGE = "tsfm_lab.models.manifests"
DEFAULT_MAX_PACKED_ROWS = 256  # target + covariate rows per forecast() call (memory grows ~quadratically)
MPS_MAX_PACKED_ROWS = 64  # unified memory is shared with the system: an unchunked MPS call tried to allocate 62 GiB (issue #6)

LICENSE_NOTE = ("Weights license (Modified MIT): all rights are withheld if your company's or employer's global "
                "consolidated MONTHLY revenue exceeds US$20 million; the restriction extends to derivatives. "
                "Read it before use.")
KNOWN_ISSUES = (
    "Apple GPU (MPS) needs a timestral build that runs its float64 statistics on the CPU (issue #6); the official "
    "package does not, so the device is listed as unavailable there.",
    "Flat-in-context known covariates (e.g. constant prices) can blow up upper quantiles (issue #7).",
    "Memory grows roughly quadratically with series x (1 + covariates) per call: batches are chunked automatically.",
)


def load_manifests() -> list[dict[str, Any]]:
    out = []
    for entry in sorted(resources.files(MANIFEST_PACKAGE).iterdir(), key=lambda p: p.name):
        if entry.name.startswith("timestral-") and entry.name.endswith(".toml"):
            out.append(tomllib.loads(entry.read_text(encoding="utf-8")))
    return sorted(out, key=lambda m: -m["params"])  # 209M, 123M, 28M, 5M (file names sort alphabetically: 123m < 209m < 28m < 5m)


def info_from_manifest(manifest: dict[str, Any]) -> ModelInfo:
    caps = Capabilities(
        multivariate=True, past_covariates=True, known_covariates=True, categorical_covariates=True,
        max_context=manifest["max_context"], max_horizon_per_pass=manifest["max_horizon_per_pass"],
        quantile_levels=tuple(manifest["quantile_levels"]), device_kinds=("cpu", "cuda", "mps"))
    details = dict(manifest.get("details", {}))
    details["repo"] = manifest["repo"]
    details["vendor_reported"] = manifest.get("reported", {})
    return ModelInfo(
        id=manifest["id"], family=manifest["family"], name=manifest["name"], params=manifest["params"],
        description=manifest["description"], license=manifest["license"], license_url=manifest["license_url"],
        source_url=manifest["source_url"], capabilities=caps, details=details,
        notes=(LICENSE_NOTE, *KNOWN_ISSUES))


class TimestralAdapter(ModelAdapter):
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.info = info_from_manifest(manifest)
        self._forecaster = None
        self._device: str | None = None
        self._load_meta: dict[str, Any] = {}
        self._fallback_note: str | None = None  # why the model runs on the CPU although a GPU was requested

    # ------------------------------------------------------------------ lifecycle
    def available_devices(self) -> list[DeviceInfo]:
        return [d for d in list_devices() if d.kind in self.info.capabilities.device_kinds or not d.supported]

    @property
    def device(self) -> str | None:
        return self._device

    def load(self, device: str = "cpu") -> dict[str, Any]:
        import torch
        from timestral import TimestralForecaster
        from timestral.loading import resolve_checkpoint

        requested = device
        device, note = pick_device(device, self.info.capabilities.device_kinds, self.strict_device)
        if self._forecaster is not None:
            self.unload()
        start = time.perf_counter()
        checkpoint = resolve_checkpoint(self.manifest["repo"])
        try:
            self._forecaster = TimestralForecaster.from_pretrained(self.manifest["repo"], device=device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
        except Exception as exc:  # noqa: BLE001 - driver, memory or an unsupported operation on the GPU
            if device == "cpu" or self.strict_device:
                raise
            self._forecaster = None
            self._release(device)
            note = f"{device} could not load the model ({type(exc).__name__}: {str(exc)[:160]}); it runs on the CPU instead."
            device = "cpu"
            self._forecaster = TimestralForecaster.from_pretrained(self.manifest["repo"], device=device)
        self._fallback_note = note
        self._device = device
        config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        self.info.details.update({k: config[k] for k in ("num_heads", "mlp_hidden_dim", "d_model", "patch_size")
                                  if k in config})
        self._load_meta = {
            "load_s": round(time.perf_counter() - start, 3),
            "revision": Path(checkpoint).name,  # Hugging Face snapshot = commit sha
            "device": device,
            **({"requested_device": requested, "fallback": note} if note else {}),
        }
        return dict(self._load_meta)

    def unload(self) -> None:
        self._forecaster = None
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

    # ------------------------------------------------------------------ inference
    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if self._forecaster is None or self._device is None:
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

    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        import torch

        opts = request.options
        symmetric = bool(opts.get("symmetric_averaging", True))
        clip_nonneg = bool(opts.get("clip_nonnegative", True))
        cap_k = opts.get("quantile_real_cap_k")
        max_rows = int(opts.get("max_packed_rows", MPS_MAX_PACKED_ROWS if self._device == "mps" else DEFAULT_MAX_PACKED_ROWS))
        if cap_k is not None:
            self._forecaster.quantile_real_cap_k = float(cap_k)

        warnings: list[str] = []
        known = request.known_covariates
        if known is not None:
            found, flat = inspect_known_covariates(
                known, request.context.shape[2], self.info.capabilities.max_context, symmetric)
            warnings.extend(found)
            if opts.get("mask_flat_covariates") and flat.any():
                known = mask_flat_rows(known, flat)
                warnings.append("mask_flat_covariates applied: flagged rows were set to missing.")
        if request.horizon > self.info.capabilities.max_horizon_per_pass:
            warnings.append(
                f"horizon {request.horizon} exceeds {self.info.capabilities.max_horizon_per_pass}: forecast in blocks "
                "with each block's median fed back as context (uncertainty from earlier blocks is not propagated).")

        rows_per_series = request.context.shape[1] + request.n_covariate_rows
        chunk = max(1, max_rows // rows_per_series)
        is_cuda = self._device.startswith("cuda")
        is_mps = self._device == "mps"
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif is_mps:
            torch.mps.synchronize()
        start = time.perf_counter()
        pieces = []
        for lo in range(0, request.batch, chunk):
            hi = min(lo + chunk, request.batch)

            def part(arr):
                return None if arr is None else arr[lo:hi]

            kwargs: dict[str, Any] = dict(prediction_length=request.horizon, symmetric_averaging=symmetric,
                                          clip_nonnegative=clip_nonneg)
            for name, arr in (("past_covariates", part(request.past_covariates)),
                              ("known_covariates", part(known)),
                              ("categorical_past_covariates", part(request.categorical_past_covariates)),
                              ("categorical_known_covariates", part(request.categorical_known_covariates))):
                if arr is not None:
                    kwargs[name] = arr
            out = self._forecaster.forecast(request.context[lo:hi], **kwargs)
            pieces.append(out.quantiles.detach().float().cpu().numpy())
            levels = tuple(float(x) for x in out.quantile_levels)
        if is_cuda:
            torch.cuda.synchronize()
        elif is_mps:
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings: dict[str, Any] = {"forecast_ms": round(elapsed_ms, 3), "chunks": len(pieces),
                                   "series_per_chunk": chunk}
        if is_cuda:
            timings["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        return ForecastResult(np.concatenate(pieces, axis=0), levels, self._device, timings, warnings)


def register(registry) -> None:
    for manifest in load_manifests():
        registry.add(info_from_manifest(manifest), lambda m=manifest: TimestralAdapter(m))
