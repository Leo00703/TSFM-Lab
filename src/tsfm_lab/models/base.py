"""Model-agnostic contract: every forecasting model (Timestral, baselines, future models)
implements :class:`ModelAdapter` and exchanges :class:`ForecastRequest` / :class:`ForecastResult`.

Adding a model = one adapter module + one registration; nothing else in the app changes.
Adapters declare their own quantile levels; downstream code never assumes a fixed grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np


@dataclass(frozen=True)
class Capabilities:
    multivariate: bool
    past_covariates: bool
    known_covariates: bool
    categorical_covariates: bool
    max_context: int | None  # None = unlimited
    max_horizon_per_pass: int | None  # None = no limit
    quantile_levels: tuple[float, ...]
    device_kinds: tuple[str, ...]  # subset of ("cpu", "cuda", "mps") the adapter can use


@dataclass(frozen=True)
class ModelInfo:
    id: str  # unique, lowercase, e.g. "timestral-209m"
    family: str  # e.g. "timestral", "baseline"
    name: str
    params: int | None  # parameter count, None for non-parametric baselines
    description: str
    license: str
    license_url: str | None
    source_url: str | None
    capabilities: Capabilities
    details: dict[str, Any] = field(default_factory=dict)  # layers, d_model, weights size...
    notes: tuple[str, ...] = ()  # caveats shown to the user (known issues, license warnings)
    kind: str = "foundation"  # "foundation" | "baseline"


@dataclass(frozen=True)
class DeviceInfo:
    id: str  # "cpu", "cuda:0", "mps"
    kind: str  # "cpu" | "cuda" | "mps"
    name: str
    memory_gb: float | None
    supported: bool = True
    reason: str | None = None  # why it is unavailable/unsupported


@dataclass
class ForecastRequest:
    """Inputs for one forecast call. All arrays are float, NaN = missing.

    context: (B, V, T) target history; past_covariates: (B, Cp, T);
    known_covariates: (B, Ck, T + horizon); categorical_* hold numeric category codes.
    """

    context: np.ndarray
    horizon: int
    past_covariates: np.ndarray | None = None
    known_covariates: np.ndarray | None = None
    categorical_past_covariates: np.ndarray | None = None
    categorical_known_covariates: np.ndarray | None = None
    seasonal_period: int = 1
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context.ndim != 3:
            raise ValueError(f"context must be (B, V, T), got shape {self.context.shape}")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        b, _, t = self.context.shape
        for name, arr, expected in (
            ("past_covariates", self.past_covariates, t),
            ("known_covariates", self.known_covariates, t + self.horizon),
            ("categorical_past_covariates", self.categorical_past_covariates, t),
            ("categorical_known_covariates", self.categorical_known_covariates, t + self.horizon),
        ):
            if arr is not None and (arr.ndim != 3 or arr.shape[0] != b or arr.shape[2] != expected):
                raise ValueError(f"{name} must have shape ({b}, C, {expected}), got {arr.shape}")

    @property
    def batch(self) -> int:
        return self.context.shape[0]

    @property
    def n_covariate_rows(self) -> int:
        return sum(a.shape[1] for a in (
            self.past_covariates, self.known_covariates,
            self.categorical_past_covariates, self.categorical_known_covariates) if a is not None)


@dataclass
class ForecastResult:
    quantiles: np.ndarray  # (B, V, H, Q), non-decreasing along Q
    quantile_levels: tuple[float, ...]
    device: str
    timings: dict[str, Any] = field(default_factory=dict)  # forecast_ms, chunks, peak_vram_mb...
    warnings: list[str] = field(default_factory=list)

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.5)

    def quantile(self, level: float) -> np.ndarray:
        for i, lv in enumerate(self.quantile_levels):
            if abs(lv - level) < 1e-9:
                return self.quantiles[..., i]
        raise ValueError(f"quantile level {level} not produced by this model: {self.quantile_levels}")


# Values outside this range make the float32 arithmetic inside several models overflow or underflow (a square of 1e20 is already past the
# largest float32): they came back as NaN or garbage, and in one model a single such series spoiled every other series of its batch.
SAFE_ABS_RANGE = (1e-12, 1e15)


def rescale_extreme(request: ForecastRequest) -> tuple[ForecastRequest, np.ndarray | None, str | None]:
    """Divide every target (and covariate) series whose values are outside what float32 handles safely by its largest absolute value.

    The models are scale-equivariant by design, so this changes nothing for them except removing the overflow; the forecast of a
    rescaled target is multiplied back by the returned factors, shape (B, V). Series inside the range are not touched at all (no
    arithmetic, so a normal run is bit-for-bit what it was). Returns ``(request, None, None)`` when nothing was outside.
    """
    lo, hi = SAFE_ABS_RANGE

    def factors(a: np.ndarray) -> np.ndarray:
        top = np.where(np.isfinite(a), np.abs(a), 0.0).max(axis=-1, initial=0.0)  # missing and infinite values do not count
        return np.where((top > hi) | ((top > 0) & (top < lo)), top, 1.0)

    f_target = factors(request.context)
    if not (f_target != 1.0).any() and all(a is None or not (factors(a) != 1.0).any() for a in (request.past_covariates, request.known_covariates)):
        return request, None, None

    def divide(a: np.ndarray | None) -> np.ndarray | None:
        if a is None:
            return None
        f = factors(a)
        return a if not (f != 1.0).any() else a / f[..., None]

    scaled = ForecastRequest(
        context=request.context / f_target[..., None], horizon=request.horizon, past_covariates=divide(request.past_covariates),
        known_covariates=divide(request.known_covariates), categorical_past_covariates=request.categorical_past_covariates,
        categorical_known_covariates=request.categorical_known_covariates, seasonal_period=request.seasonal_period, options=request.options)
    n = int((f_target != 1.0).sum())
    peak = float(np.nanmax(np.abs(request.context[np.isfinite(request.context)]))) if np.isfinite(request.context).any() else 0.0
    note = (f"{n} target series had values outside the range float32 handles safely ({lo:g} to {hi:g}; the largest is {peak:.3g}): each was divided "
            "by its largest absolute value before the model and its forecast multiplied back." if n else
            f"some covariates had values outside the range float32 handles safely ({lo:g} to {hi:g}): they were divided by their largest absolute value.")
    return scaled, f_target if n else None, note


class ModelAdapter(ABC):
    """One loaded-or-loadable model. Instances hold weights only while loaded."""

    info: ModelInfo
    # False (default): asking for a GPU that is missing or fails continues on the CPU and says so in the warnings.
    # True: raise instead (benchmarks must never label CPU numbers as GPU numbers).
    strict_device: bool = False

    @abstractmethod
    def available_devices(self) -> list[DeviceInfo]:
        """Devices this adapter could run on right now (with reasons when not usable)."""

    @abstractmethod
    def load(self, device: str = "cpu") -> dict[str, Any]:
        """Load weights onto ``device`` and return load metadata (``load_s``, ``revision``...)."""

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def device(self) -> str | None:
        """Currently loaded device id, or None when not loaded."""

    @abstractmethod
    def forecast(self, request: ForecastRequest) -> ForecastResult: ...

    @property
    def is_loaded(self) -> bool:
        return self.device is not None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.unload()
