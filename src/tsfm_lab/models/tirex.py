"""Adapter for TiRex-2 (NXAI; package ``tirex-2``, Apache-2.0, pure Python).

The model reads one *group* of target variables at a time (its variate-mixing attention lets them inform each other) with
past-only and known-future covariates, so a request item maps to one native ``TimeseriesType``: target ``[V, T]``,
past covariates ``[Cp, T]``, future covariates ``[Cf, T + H]``. Missing values are passed as NaN, which the model masks itself.

Kernels: the package's fused CUDA recurrent kernels (FlashRNN, Triton) need a compiler toolchain that Windows machines rarely
have, so the model is built with its pure-PyTorch kernels and moved to the chosen device; they give the same numbers
(CPU against GPU agree to about 1e-6) and run in eager mode on every platform. ``torch.compile`` is switched off for the same
reason (it needs a C++ compiler on the CPU and Triton on the GPU).
"""

from __future__ import annotations

import tomllib
from importlib import resources
from typing import Any

import numpy as np

from tsfm_lab.models import cache
from tsfm_lab.models.base import Capabilities, ForecastRequest, ForecastResult, ModelInfo
from tsfm_lab.models.torch_adapter import TorchAdapter

MANIFEST_PACKAGE = "tsfm_lab.models.manifests"
SERIES_PER_CALL = {"cuda": 64, "cpu": 16, "mps": 16}

KNOWN_ISSUES = (
    "Licence: the model and its package are Apache-2.0, but one of the libraries it runs on (mlstm_kernels) is under the NXAI "
    "Community Licence: free for research and for organisations below EUR 100 million annual revenue, otherwise a commercial "
    "licence from NXAI is needed to use it in a commercial product or service, and 'Built with technology from NXAI' must be "
    "shown when redistributing. Check it before commercial use.",
    "On the public fev-bench it ranks below Timestral, t0-gamma, TimesFM 3.0 and Chronos-2 (45.5 against 47-50 skill); the "
    "result was submitted by its own authors. Its training data are two public corpora (LOTSA and the Chronos datasets), which "
    "overlap many benchmark datasets: treat results on public data as possibly optimistic.",
    "Runs here with its pure-PyTorch kernels (the fused CUDA kernels need a compiler): same numbers, but slower than NXAI's "
    "optimised path. The Apple GPU (MPS) has not been verified: compare it with the CPU first (scripts/check_device_parity.py). "
    "A GPU that fails continues on the CPU.",
    "Context is limited to the most recent 2048 steps; horizons beyond 320 are produced in several passes by the model itself.",
    "Several variables of one series are forecast together, so each one can use the others as context. Categorical "
    "covariates are not used.",
    "Quantiles are the nine deciles; where the model returns crossing quantiles they are put back in order and a warning says so.",
)


def load_manifests() -> list[dict[str, Any]]:
    out = []
    for entry in sorted(resources.files(MANIFEST_PACKAGE).iterdir(), key=lambda p: p.name):
        if entry.name.startswith("tirex-") and entry.name.endswith(".toml"):
            out.append(tomllib.loads(entry.read_text(encoding="utf-8")))
    return sorted(out, key=lambda m: -m["params"])


def info_from_manifest(manifest: dict[str, Any]) -> ModelInfo:
    caps = Capabilities(
        multivariate=True, past_covariates=True, known_covariates=True, categorical_covariates=False,
        max_context=manifest["max_context"], max_horizon_per_pass=manifest["max_horizon_per_pass"],
        quantile_levels=tuple(manifest["quantile_levels"]), device_kinds=("cpu", "cuda", "mps"))
    details = dict(manifest.get("details", {}))
    details["repo"] = manifest["repo"]
    details["vendor_reported"] = manifest.get("reported", {})
    return ModelInfo(
        id=manifest["id"], family=manifest["family"], name=manifest["name"], params=manifest["params"],
        description=manifest["description"], license=manifest["license"], license_url=manifest["license_url"],
        source_url=manifest["source_url"], capabilities=caps, details=details, notes=KNOWN_ISSUES)


class TiRexAdapter(TorchAdapter):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__()
        self.manifest = manifest
        self.info = info_from_manifest(manifest)

    def _load_on(self, device: str) -> dict[str, Any]:
        import torch

        torch._dynamo.config.disable = True  # the package decorates a function with torch.compile: run it eagerly (no compiler needed)
        from tirex2 import load_model

        repo = self.manifest["repo"]
        status = cache.status(repo)
        if not status["downloaded"]:
            raise FileNotFoundError(f"{self.info.name} is not downloaded: download its weights on the Models page first")
        snapshot = cache.repo_dir(repo) / "snapshots" / status["revision"]
        kind = device.split(":")[0]
        # pure-PyTorch kernels for every device (the fused CUDA ones need nvcc/Triton), then the weights go to the device
        model = load_model(str(snapshot), device="mps" if kind == "mps" else "cpu", compile=False)
        model.model.to(torch.device(device))
        self._model = model
        return {"revision": status["revision"]}

    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        import torch
        from tirex2 import TimeseriesType

        caps = self.info.capabilities
        batch, variables, length = request.context.shape
        horizon = request.horizon
        warnings: list[str] = []
        if length > caps.max_context:
            warnings.append(f"context longer than {caps.max_context}: the model keeps only the most recent steps.")
        if horizon > caps.max_horizon_per_pass:
            # The package itself silently cuts a longer request to its maximum, so the passes are made here: each one feeds the median of the
            # previous back as history (past-only covariates are unknown there and are passed as missing).
            warnings.append(f"horizon {horizon} exceeds {caps.max_horizon_per_pass}: forecast in several passes, each continuing from the median of the "
                            "previous one, so the later steps are conditional on it and their intervals tend to be too narrow.")
        known, past = request.known_covariates, request.past_covariates
        if known is not None and np.isnan(known[..., length:length + horizon]).any():
            warnings.append("known covariates are missing over the forecast region: the model treats them as unobserved.")
        if variables > 1:
            warnings.append(f"{variables} variables per series are forecast together.")

        def tensor(a: np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

        per_call = SERIES_PER_CALL.get((self._device or "cpu").split(":")[0], 16)
        median = list(caps.quantile_levels).index(0.5)

        def run() -> np.ndarray:
            full = np.ascontiguousarray(request.context, dtype=np.float32)  # (B, V, length + steps already forecast)
            full_past = None if past is None else np.ascontiguousarray(past[..., :length], dtype=np.float32)
            pieces, done = [], 0
            while done < horizon:
                step = min(caps.max_horizon_per_pass, horizon - done)
                end = full.shape[2]
                first = max(0, end - caps.max_context)
                series = [TimeseriesType(
                    target=tensor(full[b, :, first:end]),
                    past_covariates=None if full_past is None else tensor(full_past[b, :, first:end]),
                    future_covariates=None if known is None else tensor(known[b, :, first:length + done + step])) for b in range(batch)]
                with torch.inference_mode():
                    out = self._model.forecast(series, prediction_length=step, output_type="numpy", batch_size=per_call)
                piece = np.transpose(np.stack([np.asarray(o, dtype=np.float32) for o in out]), (0, 1, 3, 2))  # (B, V, Q, step) -> (B, V, step, Q)
                pieces.append(piece)
                done += step
                if done < horizon:
                    full = np.concatenate([full, piece[..., median]], axis=2)
                    if full_past is not None:
                        full_past = np.concatenate([full_past, np.full((*full_past.shape[:2], step), np.nan, dtype=np.float32)], axis=2)
            return np.concatenate(pieces, axis=2)

        q, timings = self._timed(run)
        if (np.diff(q, axis=-1) < 0).any():
            q = np.sort(q, axis=-1)
            warnings.append("the model returned crossing quantiles for some steps: they were put back in order.")
        timings["series_per_call"] = per_call
        return ForecastResult(q, tuple(caps.quantile_levels), self._device, timings, warnings)


def register(registry) -> None:
    for manifest in load_manifests():
        registry.add(info_from_manifest(manifest), lambda m=manifest: TiRexAdapter(m))
