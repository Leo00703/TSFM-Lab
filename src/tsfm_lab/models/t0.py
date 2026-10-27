"""Adapter for t0-gamma (The Forecasting Company; package ``tfc-t0``, Apache-2.0, pure Python).

How a request is turned into the model's input follows the evaluation wrapper that the model's authors published in the
fev repository (``models/t0-gamma/model.py``, Apache-2.0): every target variable is forecast as its own series, known-future
covariates go into the forecast region, past-only covariates are historical rows that are missing over the horizon, and the
four recommended adjustments (trimming leading gaps, symmetric averaging, non-negative clipping, integer snapping) are on.

One deliberate difference: the authors' benchmark run forecast groups of 64 *unrelated* series jointly (cross-inference) on
the tasks without covariates. This app forecasts every series on its own: a backtest places different windows of the same
series next to each other, and joint inference across them would let a later window's history inform an earlier window's
forecast. The published score therefore describes a slightly different mode than the one measured here.
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
ROWS_PER_CALL = {"cuda": 64, "cpu": 32, "mps": 32}  # target + covariate rows per model call

KNOWN_ISSUES = (
    "A young model: released on 2026-10-12, with no independent evaluation yet. Its fev-bench result was submitted by its own "
    "authors, and the training data are described only as curated public datasets plus synthetic series, without a list: "
    "treat results on public data as possibly optimistic and judge it on your own backtests.",
    "The authors' published benchmark forecast 64 unrelated series jointly on the tasks without covariates; this app "
    "forecasts every series on its own, so measured scores can differ from the published ones.",
    "Several variables of one series are forecast one at a time, each with the same covariates.",
    "Known covariates that are missing over the forecast region are read by the model as 0.",
    "Categorical covariates are not used here. The Apple GPU (MPS) has not been verified: compare it with the CPU first "
    "(scripts/check_device_parity.py). A GPU that fails continues on the CPU.",
)


def load_manifests() -> list[dict[str, Any]]:
    out = []
    for entry in sorted(resources.files(MANIFEST_PACKAGE).iterdir(), key=lambda p: p.name):
        if entry.name.startswith("t0-") and entry.name.endswith(".toml"):
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


class T0Adapter(TorchAdapter):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__()
        self.manifest = manifest
        self.info = info_from_manifest(manifest)

    def _load_on(self, device: str) -> dict[str, Any]:
        from t0 import T0Forecaster

        repo = self.manifest["repo"]
        status = cache.status(repo)
        if not status["downloaded"]:
            raise FileNotFoundError(f"{self.info.name} is not downloaded: download its weights on the Models page first")
        snapshot = cache.repo_dir(repo) / "snapshots" / status["revision"]
        self._model = T0Forecaster.from_pretrained(str(snapshot)).to(device).eval()
        return {"revision": status["revision"]}

    @staticmethod
    def _series(target: np.ndarray, known: np.ndarray | None, past: np.ndarray | None, horizon: int):
        """One target row with its covariates as the model's native ``TimeSeries`` (the authors' construction)."""
        import torch
        from t0 import MaskType, TimeSeries, VariateType

        future = None if known is None else torch.from_numpy(np.asarray(known, dtype=np.float32))[None]
        series = TimeSeries.from_array(torch.from_numpy(np.asarray(target, dtype=np.float32))[None],
                                       future_covariates=future, horizon=horizon)
        if past is None:
            return series
        length = target.shape[-1]
        width = series.seq_len
        values = np.asarray(past, dtype=np.float32)
        hist = torch.zeros((values.shape[0], width), dtype=torch.float32)
        hist[:, :length] = torch.from_numpy(np.nan_to_num(values, nan=0.0))
        mask = torch.full((values.shape[0], width), MaskType.MISSING, dtype=torch.int8)
        mask[:, :length] = MaskType.VALID
        # a gap in a covariate is an absent observation, not a zero
        mask[:, :length] = torch.where(torch.from_numpy(np.isnan(values)), torch.tensor(MaskType.MISSING, dtype=torch.int8), mask[:, :length])
        return TimeSeries(
            variates=torch.cat([series.variates, hist]), mask=torch.cat([series.mask, mask]),
            group_ids=torch.cat([series.group_ids, torch.zeros_like(mask, dtype=torch.long)]),
            variate_type=torch.cat([series.variate_type, torch.full_like(mask, VariateType.HISTORICAL, dtype=torch.long)]))

    def _forecast_once(self, request: ForecastRequest) -> ForecastResult:
        import torch
        from t0 import TimeSeries

        caps = self.info.capabilities
        batch, variables, length = request.context.shape
        horizon = request.horizon
        keep = min(length, caps.max_context)
        start = length - keep
        opts = request.options
        adjustments = {"trim_leading_missing": True, "snap_integers": True,
                       "symmetric_averaging": bool(opts.get("symmetric_averaging", True)),
                       "clip_nonnegative": bool(opts.get("clip_nonnegative", True))}
        warnings: list[str] = []
        known, past = request.known_covariates, request.past_covariates
        if known is not None and np.isnan(known[..., length:length + horizon]).any():
            warnings.append("known covariates are missing over the forecast region: the model reads them as 0.")
        if horizon > caps.max_horizon_per_pass:
            warnings.append(f"horizon {horizon} exceeds {caps.max_horizon_per_pass}: the model continues autoregressively.")
        if variables > 1:
            warnings.append(f"{variables} variables per series are forecast one at a time.")

        series = []
        for b in range(batch):
            k = None if known is None else known[b, :, start:length + horizon]
            p = None if past is None else past[b, :, start:length]
            for v in range(variables):
                series.append(self._series(request.context[b, v, start:], k, p, horizon))
        rows = 1 + (0 if known is None else known.shape[1]) + (0 if past is None else past.shape[1])
        per_call = max(1, ROWS_PER_CALL.get((self._device or "cpu").split(":")[0], 32) // rows)
        device = next(self._model.parameters()).device
        levels = list(caps.quantile_levels)

        def run() -> np.ndarray:
            pieces = []
            for lo in range(0, len(series), per_call):
                chunk = TimeSeries.batch(series[lo:lo + per_call]).to(device)
                with torch.inference_mode():
                    out = self._model.predict(chunk, horizon=horizon, quantile_levels=levels,
                                              context_length=chunk.seq_len - horizon, **adjustments)
                pieces.append(out.quantiles.float().cpu().numpy())
            return np.concatenate(pieces, axis=0)  # (series, H, Q)

        out, timings = self._timed(run)
        timings["series_per_call"] = per_call
        return ForecastResult(out.reshape(batch, variables, horizon, len(levels)), tuple(levels), self._device, timings, warnings)


def register(registry) -> None:
    for manifest in load_manifests():
        registry.add(info_from_manifest(manifest), lambda m=manifest: T0Adapter(m))
