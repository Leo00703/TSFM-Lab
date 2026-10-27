"""Adapter for Mistral's Timestral models (package ``timestral``, pinned in pyproject)."""

from __future__ import annotations

import json
import time
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from tsfm_lab.models.base import Capabilities, ForecastRequest, ForecastResult, ModelInfo
from tsfm_lab.models.covariates import inspect_known_covariates, mask_flat_rows
from tsfm_lab.models.torch_adapter import TorchAdapter

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


class TimestralAdapter(TorchAdapter):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__()
        self.manifest = manifest
        self.info = info_from_manifest(manifest)

    def _load_on(self, device: str) -> dict[str, Any]:
        from timestral import TimestralForecaster
        from timestral.loading import resolve_checkpoint

        checkpoint = resolve_checkpoint(self.manifest["repo"])
        self._model = TimestralForecaster.from_pretrained(self.manifest["repo"], device=device)
        config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        self.info.details.update({k: config[k] for k in ("num_heads", "mlp_hidden_dim", "d_model", "patch_size")
                                  if k in config})
        return {"revision": Path(checkpoint).name}  # Hugging Face snapshot = commit sha

    # ------------------------------------------------------------------ inference
    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        import torch

        opts = request.options
        symmetric = bool(opts.get("symmetric_averaging", True))
        clip_nonneg = bool(opts.get("clip_nonnegative", True))
        cap_k = opts.get("quantile_real_cap_k")
        max_rows = int(opts.get("max_packed_rows", MPS_MAX_PACKED_ROWS if self._device == "mps" else DEFAULT_MAX_PACKED_ROWS))
        if cap_k is not None:
            self._model.quantile_real_cap_k = float(cap_k)

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
            out = self._model.forecast(request.context[lo:hi], **kwargs)
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
