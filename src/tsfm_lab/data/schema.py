"""Internal dataset representation shared by synthetic, local and online sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import re

# Steps per natural seasonal cycle for one step of each pandas frequency unit.
_CYCLE = {"min": 1440, "t": 1440, "h": 24, "d": 7, "w": 52, "m": 12, "ms": 12, "me": 12, "q": 4, "qs": 4,
          "qe": 4, "y": 1, "ys": 1, "ye": 1, "a": 1}


def seasonal_period_for(freq: str | None) -> int:
    """Seasonal period for a pandas frequency string: 'h' -> 24, '15min' -> 96, '4h' -> 6, 'D' -> 7."""
    if not freq:
        return 1
    match = re.fullmatch(r"(\d*)([a-zA-Z]+)(?:-[A-Za-z]+)?", freq.strip())
    if not match:
        return 1
    n = int(match.group(1) or 1)
    cycle = _CYCLE.get(match.group(2).lower(), 1)
    return cycle // n if n and cycle % n == 0 and cycle // n >= 1 else 1


@dataclass
class Series:
    """One series. target: (V, T); past: (Cp, T); known: (Ck, T) covering the whole span (incl. test horizon)."""

    id: str
    target: np.ndarray
    timestamps: np.ndarray | None = None  # datetime64, length T
    past_covariates: np.ndarray | None = None
    known_covariates: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.target = np.atleast_2d(np.asarray(self.target, dtype=np.float64))
        t = self.target.shape[1]
        for name in ("past_covariates", "known_covariates"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                if arr.shape[1] != t:
                    raise ValueError(f"series {self.id}: {name} length {arr.shape[1]} != target length {t}")
                setattr(self, name, arr)
        if self.timestamps is not None and len(self.timestamps) != t:
            raise ValueError(f"series {self.id}: timestamps length != target length")

    @property
    def length(self) -> int:
        return self.target.shape[1]


@dataclass
class Dataset:
    id: str
    name: str
    series: list[Series]
    freq: str | None = None
    seasonal_period: int = 1
    target_names: list[str] = field(default_factory=list)
    past_names: list[str] = field(default_factory=list)
    known_names: list[str] = field(default_factory=list)
    source: str = "synthetic"  # synthetic | local | hub
    license: str = "n/a"
    description: str = ""
    contamination_risk: str = "none"  # none | possible | unknown
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError("dataset has no series")
        v = {s.target.shape[0] for s in self.series}
        cp = {0 if s.past_covariates is None else s.past_covariates.shape[0] for s in self.series}
        ck = {0 if s.known_covariates is None else s.known_covariates.shape[0] for s in self.series}
        if len(v) > 1 or len(cp) > 1 or len(ck) > 1:
            raise ValueError("all series must share the same number of targets and covariates")
        if not self.target_names:
            self.target_names = [f"target_{i}" for i in range(self.n_targets)]
        if not self.past_names and self.n_past:
            self.past_names = [f"past_{i}" for i in range(self.n_past)]
        if not self.known_names and self.n_known:
            self.known_names = [f"known_{i}" for i in range(self.n_known)]

    @property
    def n_targets(self) -> int:
        return self.series[0].target.shape[0]

    @property
    def n_past(self) -> int:
        c = self.series[0].past_covariates
        return 0 if c is None else c.shape[0]

    @property
    def n_known(self) -> int:
        c = self.series[0].known_covariates
        return 0 if c is None else c.shape[0]

    def summary(self) -> dict[str, Any]:
        lengths = [s.length for s in self.series]
        return {
            "id": self.id, "name": self.name, "source": self.source, "license": self.license,
            "n_series": len(self.series), "length_min": min(lengths), "length_max": max(lengths),
            "n_targets": self.n_targets, "n_past_covariates": self.n_past, "n_known_covariates": self.n_known,
            "freq": self.freq, "seasonal_period": self.seasonal_period,
            "contamination_risk": self.contamination_risk, "tags": self.tags, "description": self.description,
        }
