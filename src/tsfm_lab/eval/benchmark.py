"""Local performance benchmark: one model, one or more devices, explicit cold/warm separation.

Protocol per device: load (timed), first forecast (cold: CUDA init, kernel compilation...), then for each
scenario cell one untimed warm-up call followed by ``repeats`` timed calls (CUDA synchronised inside the
adapter). Reported: median / min / p90 latency, throughput, peak VRAM, process memory growth after load.
The first scenario reproduces Mistral's announced setup (1 series, 2016 steps of 15-minute history, 96 ahead)
so numbers can be compared with their 17 ms (5M) / 113 ms (209M) on a server CPU, keeping hardware in mind.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.models.registry import Registry, default_registry
from tsfm_lab.system.probe import environment_info

SCENARIOS = ("mistral", "batch", "context", "horizon")


def _series(batch: int, length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    base = 100 + 0.01 * t + 10 * np.sin(2 * np.pi * t / 96) + 4 * np.sin(2 * np.pi * t / 672)
    return (base[None, None, :] + rng.normal(0, 1, (batch, 1, length))).astype(np.float32)


def cells(max_context: int | None, scenarios: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "mistral" in scenarios:
        out.append({"scenario": "mistral", "label": "1 series · 2016 → 96 (Mistral setup)", "batch": 1, "context": 2016, "horizon": 96})
    if "batch" in scenarios:
        out += [{"scenario": "batch", "label": f"batch {b}", "batch": b, "context": 2016, "horizon": 96}
                for b in (1, 8, 32, 128)]
    if "context" in scenarios:
        out += [{"scenario": "context", "label": f"context {c}", "batch": 1, "context": c, "horizon": 96}
                for c in (256, 1024, 4096, 16384) if not max_context or c <= max_context]
    if "horizon" in scenarios:
        out += [{"scenario": "horizon", "label": f"horizon {h}", "batch": 1, "context": 2016, "horizon": h}
                for h in (24, 96, 384, 1024)]
    return out


def run_benchmark(model_id: str, devices: list[str], repeats: int = 15, scenarios: tuple[str, ...] = SCENARIOS,
                  registry: Registry | None = None,
                  progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    import psutil

    registry = registry or default_registry()
    info = registry.info(model_id)
    grid = cells(info.capabilities.max_context, scenarios)
    total = len(devices) * len(grid)
    done = 0
    result: dict[str, Any] = {"model": model_id, "name": info.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "repeats": repeats, "environment": environment_info(), "devices": {}, "cpu_vs_gpu": None}
    reference: dict[str, np.ndarray] = {}
    proc = psutil.Process()
    for device in devices:
        adapter = registry.create(model_id)
        adapter.strict_device = True  # a benchmark must fail, not quietly measure the CPU under a GPU's name
        rss_before = proc.memory_info().rss
        meta = adapter.load(device)
        rss_after = proc.memory_info().rss
        entry: dict[str, Any] = {"device": adapter.device, "load_s": meta["load_s"], "revision": meta.get("revision"),
                                 "process_memory_growth_mb": round((rss_after - rss_before) / 2**20, 1), "cells": []}
        first = ForecastRequest(_series(1, 2016), 96, seasonal_period=96)
        cold = adapter.forecast(first)
        entry["cold_first_call_ms"] = cold.timings["forecast_ms"]
        reference[adapter.device] = cold.quantiles
        for cell in grid:
            if progress:
                progress(done, total, f"{device}: measuring {cell['label']}…")
            ctx = _series(cell["batch"], cell["context"])
            req = ForecastRequest(ctx, cell["horizon"], seasonal_period=96)
            adapter.forecast(req)  # warm-up for this shape
            times, peaks = [], []
            for _ in range(repeats):
                res = adapter.forecast(req)
                times.append(res.timings["forecast_ms"])
                if "peak_vram_mb" in res.timings:
                    peaks.append(res.timings["peak_vram_mb"])
            times.sort()
            med = statistics.median(times)
            entry["cells"].append({**cell, "median_ms": round(med, 2), "min_ms": round(times[0], 2),
                                   "p90_ms": round(times[int(0.9 * (len(times) - 1))], 2),
                                   "series_per_s": round(cell["batch"] / (med / 1000), 1),
                                   "peak_vram_mb": max(peaks) if peaks else None, "chunks": res.timings.get("chunks")})
            done += 1
            if progress:
                progress(done, total, f"{device}: {cell['label']}")
        adapter.unload()
        result["devices"][device] = entry
    gpu = next((d for d in reference if d != "cpu"), None)
    if "cpu" in reference and gpu:
        diff = np.abs(reference["cpu"] - reference[gpu])
        rng = float(np.ptp(_series(1, 2016))) or 1.0
        result["cpu_vs_gpu"] = {"gpu": gpu, "max_abs_diff": float(diff.max()), "relative_to_series_range": float(diff.max() / rng)}
    return result
