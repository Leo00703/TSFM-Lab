"""Adapter for Amazon's Chronos-2 models (package ``chronos-forecasting``, Apache-2.0).

Chronos-2 takes the same kind of input as Timestral (history, optionally several target variables, covariates that are only
known in the past or also known in the future) and returns quantiles, so both can be run on identical windows and compared.
Differences worth knowing: Chronos-2 keeps up to 8192 steps of context (Timestral 32768), has no categorical covariates here,
and learns across the related series of one call only when ``cross_learning`` is requested (off, so every series is
independent, as in Timestral).
"""

from __future__ import annotations

import json
import tomllib
from importlib import resources
from typing import Any

import numpy as np

from tsfm_lab.models import cache
from tsfm_lab.models.base import (
    Capabilities,
    ForecastRequest,
    ForecastResult,
    ModelInfo,
)
from tsfm_lab.models.torch_adapter import TorchAdapter

MANIFEST_PACKAGE = "tsfm_lab.models.manifests"
DEFAULT_MAX_PACKED_ROWS = 256  # target + covariate rows per model call (the pipeline's own batch size)
MPS_MAX_PACKED_ROWS = 64  # unified memory is shared with the system

KNOWN_ISSUES = (
    ("The Apple GPU (MPS) has not been verified for Chronos-2: if you use it, compare it with the CPU first "
    "(scripts/check_device_parity.py). A GPU that fails continues on the CPU."),
    "Categorical covariates are not passed to Chronos-2 here: only numeric (and 0/1) covariates are used.",
    "Options that belong to Timestral (symmetric averaging, clipping, masking of flat covariates) are ignored.",
    "Horizons longer than 1024 steps are produced by unrolling the model autoregressively.",
)


def load_manifests() -> list[dict[str, Any]]:
    out = []
    for entry in sorted(resources.files(MANIFEST_PACKAGE).iterdir(), key=lambda p: p.name):
        if entry.name.startswith("chronos-") and entry.name.endswith(".toml"):
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
        source_url=manifest["source_url"], capabilities=caps, details=details, notes=(*KNOWN_ISSUES,))


class Chronos2Adapter(TorchAdapter):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__()
        self.manifest = manifest
        self.info = info_from_manifest(manifest)

    def _load_on(self, device: str) -> dict[str, Any]:
        from chronos import Chronos2Pipeline

        repo = self.manifest["repo"]
        status = cache.status(repo)
        if not status["downloaded"]:
            raise FileNotFoundError(f"{self.info.name} is not downloaded: download its weights on the Models page first")
        snapshot = cache.repo_dir(repo) / "snapshots" / status["revision"]
        self._model = Chronos2Pipeline.from_pretrained(str(snapshot), device_map=device)
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        self.info.details.update({k: config[k] for k in ("num_heads", "d_model") if k in config})
        return {"revision": status["revision"]}

    # ------------------------------------------------------------------ inference
    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        pipe = self._model
        levels = self.info.capabilities.quantile_levels
        opts = request.options
        batch_size = int(opts.get("max_packed_rows", MPS_MAX_PACKED_ROWS if self._device == "mps" else DEFAULT_MAX_PACKED_ROWS))
        batch, _, length = request.context.shape
        horizon = request.horizon

        inputs = []
        for b in range(batch):
            item: dict[str, Any] = {"target": np.asarray(request.context[b], dtype=np.float32)}
            past: dict[str, np.ndarray] = {}
            future: dict[str, np.ndarray] = {}
            if request.past_covariates is not None:
                for j in range(request.past_covariates.shape[1]):
                    past[f"past_{j}"] = np.asarray(request.past_covariates[b, j, :length], dtype=np.float32)
            if request.known_covariates is not None:
                for j in range(request.known_covariates.shape[1]):  # a known covariate has a past part and a future part
                    row = np.asarray(request.known_covariates[b, j], dtype=np.float32)
                    past[f"known_{j}"], future[f"known_{j}"] = row[:length], row[length:length + horizon]
            if past:
                item["past_covariates"] = past
            if future:
                item["future_covariates"] = future
            inputs.append(item)

        warnings: list[str] = []
        if horizon > self.info.capabilities.max_horizon_per_pass:
            warnings.append(f"horizon {horizon} exceeds {self.info.capabilities.max_horizon_per_pass}: the model is unrolled "
                            "autoregressively (uncertainty from earlier blocks is not propagated).")
        trained = tuple(pipe.quantiles)
        if not set(levels) <= set(trained):
            warnings.append(f"this checkpoint was trained on {len(trained)} quantile levels; the other levels are interpolated.")

        (quantiles, _), timings = self._timed(lambda: pipe.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=list(levels), batch_size=batch_size,
            context_length=min(length, self.info.capabilities.max_context)))
        out = np.stack([q.detach().float().cpu().numpy() for q in quantiles])  # (B, V, H, Q)
        crossings = int((np.diff(out, axis=-1) < -1e-9).sum())
        if crossings:  # quantiles must not cross: sort each forecast's levels (the usual monotone rearrangement)
            out = np.sort(out, axis=-1)
            timings["crossings_fixed"] = crossings
        timings["series_per_call"] = batch_size
        return ForecastResult(out, tuple(float(x) for x in levels), self._device, timings, warnings)


def register(registry) -> None:
    for manifest in load_manifests():
        registry.add(info_from_manifest(manifest), lambda m=manifest: Chronos2Adapter(m))
