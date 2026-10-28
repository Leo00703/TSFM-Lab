"""Adapters for Google's TimesFM 2.5 and 3.0 (package ``timesfm``; the code is Apache-2.0).

* **TimesFM 3.0** (``timesfm3``): univariate and multivariate targets, past-only and known-future covariates, a point forecast
  (the median) and nine deciles. The *weights* are under Google's non-commercial licence: the manifest says so and the
  Models page asks for an explicit acknowledgement before downloading them.
* **TimesFM 2.5**: univariate zero-shot forecasts, up to 16k steps of context, nine deciles. It has no native covariates (the
  package offers them only through an external regression that needs JAX), so covariates are ignored and the result says so.

Both produce deciles (0.1 ... 0.9), not the 21 levels of Timestral and Chronos-2: when models are compared in one backtest the
scores that depend on the level grid are computed on the levels they all share.
"""

from __future__ import annotations

import math
import tomllib
from importlib import resources
from typing import Any

import numpy as np

from tsfm_lab.models import cache
from tsfm_lab.models.base import Capabilities, ForecastRequest, ForecastResult, ModelInfo
from tsfm_lab.models.torch_adapter import TorchAdapter

MANIFEST_PACKAGE = "tsfm_lab.models.manifests"
BATCH_BY_DEVICE = {"cuda": 16, "cpu": 8, "mps": 4}  # series per model call (the checkpoints are large)

COMMON_ISSUES = (
    "Produces nine quantile levels (0.1 ... 0.9): there is no 5%/95% or 1%/99% tail, so coverage of the 90% and 98% "
    "intervals is not available and a multi-model backtest scores every model on the levels they share.",
    "Timestral's options map as follows: symmetric averaging -> flip invariance / symmetric averaging, clip non-negative -> "
    "keep non-negative inputs non-negative. Masking of flat covariates is ignored.",
    "The Apple GPU (MPS) has not been verified for TimesFM: compare it with the CPU first (scripts/check_device_parity.py). "
    "A GPU that fails continues on the CPU.",
)


def load_manifests() -> list[dict[str, Any]]:
    out = []
    for entry in sorted(resources.files(MANIFEST_PACKAGE).iterdir(), key=lambda p: p.name):
        if entry.name.startswith("timesfm-") and entry.name.endswith(".toml"):
            out.append(tomllib.loads(entry.read_text(encoding="utf-8")))
    return sorted(out, key=lambda m: -m["params"])  # TimesFM 3.0 (331M), then 2.5 (231M)


def info_from_manifest(manifest: dict[str, Any]) -> ModelInfo:
    v3 = manifest["id"] == "timesfm-3"
    caps = Capabilities(
        multivariate=True, past_covariates=v3, known_covariates=v3, categorical_covariates=False,
        max_context=manifest["max_context"], max_horizon_per_pass=manifest["max_horizon_per_pass"],
        quantile_levels=tuple(manifest["quantile_levels"]), device_kinds=("cpu", "cuda", "mps"))
    details = dict(manifest.get("details", {}))
    details["repo"] = manifest["repo"]
    details["vendor_reported"] = manifest.get("reported", {})
    if manifest.get("restriction"):
        details["restriction"] = manifest["restriction"]
        details["restriction_note"] = manifest["restriction_note"]
    own: list[str] = []
    if manifest.get("restriction"):
        own.append(manifest["restriction_note"])
    if not v3:
        own += ["Covariates are ignored: TimesFM 2.5 has them only through an external regression (XReg) that needs JAX, "
                "which this project does not install.",
                "Several variables of one series are forecast one at a time (no joint modelling)."]
    return ModelInfo(
        id=manifest["id"], family=manifest["family"], name=manifest["name"], params=manifest["params"],
        description=manifest["description"], license=manifest["license"], license_url=manifest["license_url"],
        source_url=manifest["source_url"], capabilities=caps, details=details, notes=(*own, *COMMON_ISSUES))


def _snapshot(adapter: TorchAdapter, repo: str, name: str):
    status = cache.status(repo)
    if not status["downloaded"]:
        raise FileNotFoundError(f"{name} is not downloaded: download its weights on the Models page first")
    return cache.repo_dir(repo) / "snapshots" / status["revision"], status["revision"]


class TimesFM3Adapter(TorchAdapter):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__()
        self.manifest = manifest
        self.info = info_from_manifest(manifest)

    def _load_on(self, device: str) -> dict[str, Any]:
        from timesfm3 import TimesFM3Evaluator

        snapshot, revision = _snapshot(self, self.manifest["repo"], self.info.name)
        kind = device.split(":")[0]
        self._model = TimesFM3Evaluator.from_pretrained(
            str(snapshot), device=device, local_files_only=True, per_core_batch_size=BATCH_BY_DEVICE.get(kind, 8))
        return {"revision": revision}

    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        caps = self.info.capabilities
        batch, _, length = request.context.shape
        horizon = request.horizon
        keep = min(length, caps.max_context)
        start = length - keep  # only the most recent steps are used, covariates cut the same way
        contexts = [np.asarray(request.context[b, :, start:], dtype=np.float32) for b in range(batch)]
        past = future = None
        if request.past_covariates is not None:
            past = [np.asarray(request.past_covariates[b, :, start:length], dtype=np.float32) for b in range(batch)]
        if request.known_covariates is not None:  # known covariates cover the context and the horizon
            future = [np.asarray(request.known_covariates[b, :, start:length + horizon], dtype=np.float32) for b in range(batch)]
        opts = request.options
        warnings: list[str] = []
        if horizon > caps.max_horizon_per_pass:
            warnings.append(f"horizon {horizon} exceeds {caps.max_horizon_per_pass}: the model may stitch several blocks.")

        outputs, timings = self._timed(lambda: list(self._model.predict_batch(
            contexts=contexts, horizon=horizon, past_only_covariates=past, past_future_covariates=future,
            return_quantiles=True, use_symmetric_averaging=bool(opts.get("symmetric_averaging", True)),
            make_positive=bool(opts.get("clip_nonnegative", True)), sort_quantiles=True, padding_mode="edge")))
        q = np.stack([np.asarray(o.quantiles, dtype=np.float32) for o in outputs])  # (B, V, H, 9)
        timings["series_per_call"] = int(self._model.config.per_core_batch_size)
        return ForecastResult(q, tuple(caps.quantile_levels), self._device, timings, warnings)


class TimesFM25Adapter(TorchAdapter):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__()
        self.manifest = manifest
        self.info = info_from_manifest(manifest)
        self._compiled: tuple | None = None  # (max_context, max_horizon, flags) the model was last compiled for

    def _load_on(self, device: str) -> dict[str, Any]:
        import torch
        from timesfm import TimesFM_2p5_200M_torch

        snapshot, revision = _snapshot(self, self.manifest["repo"], self.info.name)
        # torch.compile needs a C++ toolchain that Windows machines rarely have: run eagerly.
        model = TimesFM_2p5_200M_torch.from_pretrained(str(snapshot), torch_compile=False)
        target = torch.device(device)
        model.model.device, model.model.device_count = target, 1  # the package picks CUDA by itself; the user chose the device
        model.model.to(target)
        self._model, self._compiled = model, None
        return {"revision": revision}

    def _compile(self, context: int, horizon: int, opts: dict[str, Any]) -> None:
        from timesfm import ForecastConfig

        flip = bool(opts.get("symmetric_averaging", True))
        positive = bool(opts.get("clip_nonnegative", True))
        kind = (self._device or "cpu").split(":")[0]
        padded_h = math.ceil(horizon / 128) * 128
        padded_c = math.ceil(max(32, min(context, self.info.capabilities.max_context - padded_h)) / 32) * 32
        key = (padded_c, padded_h, flip, positive, kind)
        if self._compiled == key:
            return
        self._model.compile(ForecastConfig(
            max_context=padded_c, max_horizon=padded_h, normalize_inputs=True, per_core_batch_size=BATCH_BY_DEVICE.get(kind, 8),
            use_continuous_quantile_head=padded_h <= 1024, force_flip_invariance=flip, infer_is_positive=positive,
            fix_quantile_crossing=True))
        self._compiled = key

    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        batch, variables, length = request.context.shape
        horizon = request.horizon
        warnings: list[str] = []
        if request.past_covariates is not None or request.known_covariates is not None:
            warnings.append("covariates ignored: TimesFM 2.5 does not use them here.")
        if variables > 1:
            warnings.append(f"{variables} variables per series are forecast one at a time.")
        self._compile(length, horizon, request.options)
        series = [np.asarray(request.context[b, v], dtype=np.float32) for b in range(batch) for v in range(variables)]
        empty = [i for i, x in enumerate(series) if not np.isfinite(x).any()]
        for i in empty:  # the package's own fallback for a context with nothing observed crashes: it meant to read it as zeros
            series[i] = np.zeros_like(series[i])
        if empty:
            warnings.append(f"{len(empty)} context(s) had no observed value at all; TimesFM 2.5 read them as zeros.")
        (_, quantiles), timings = self._timed(lambda: self._model.forecast(horizon=horizon, inputs=series))
        q = np.asarray(quantiles, dtype=np.float32)[:, :horizon, 1:10]  # column 0 is the mean; 1..9 are the deciles
        q = q.reshape(batch, variables, horizon, 9)
        timings["series_per_call"] = int(self._model.global_batch_size)
        return ForecastResult(q, tuple(self.info.capabilities.quantile_levels), self._device, timings, warnings)


def register(registry) -> None:
    for manifest in load_manifests():
        factory = TimesFM3Adapter if manifest["id"] == "timesfm-3" else TimesFM25Adapter
        registry.add(info_from_manifest(manifest), lambda m=manifest, f=factory: f(m))
