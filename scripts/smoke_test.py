#!/usr/bin/env python3
"""Phase-0 smoke test: run a Timestral checkpoint on CPU and, if present, CUDA.

Checks output shape/ordering, measures latency with proper warm-up and GPU synchronisation,
and compares CPU and GPU forecasts. Run with the project interpreter:

    .venv\\Scripts\\python scripts/smoke_test.py            # Windows
    .venv/bin/python scripts/smoke_test.py                  # Linux / macOS
    ... --model mistralai/Timestral-28M --repeats 30 --devices cpu,cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import tsfm_lab  # noqa: F401  (applies cache isolation before torch / huggingface_hub are used)

import numpy as np
import torch
from timestral import TimestralForecaster

from tsfm_lab.isolation import find_root


def make_context(batch: int, length: int, seed: int = 0) -> np.ndarray:
    """Synthetic (B, 1, T) series: daily-like seasonality (period 96) + trend + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    series = [
        100 + 0.02 * t + 10 * np.sin(2 * np.pi * t / 96 + rng.uniform(0, 6.28)) + rng.normal(0, 1, length)
        for _ in range(batch)
    ]
    return np.stack(series)[:, None, :].astype(np.float32)


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench(forecaster, context: np.ndarray, horizon: int, device: str, repeats: int) -> dict:
    for _ in range(2):  # warm-up (kernel selection, allocator, caches)
        forecaster.forecast(context, prediction_length=horizon)
    sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        forecaster.forecast(context, prediction_length=horizon)
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    out = {
        "median_ms": round(statistics.median(times), 2),
        "min_ms": round(times[0], 2),
        "p90_ms": round(times[int(0.9 * (len(times) - 1))], 2),
        "series_per_s": round(context.shape[0] / (statistics.median(times) / 1000), 1),
    }
    if device.startswith("cuda"):
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


def check_forecast(forecast, batch: int, horizon: int) -> list[str]:
    problems = []
    q = forecast.quantiles.detach().cpu().numpy()
    if q.shape != (batch, 1, horizon, 21):
        problems.append(f"unexpected quantile shape {q.shape}")
    if not np.isfinite(q).all():
        problems.append("non-finite values in forecast")
    if (np.diff(q, axis=-1) < -1e-4).any():
        problems.append("quantiles cross (not monotone)")
    if len(forecast.quantile_levels) != 21 or abs(forecast.quantile_levels[0] - 0.01) > 1e-9:
        problems.append(f"unexpected quantile levels {forecast.quantile_levels}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mistralai/Timestral-5M")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--devices", default="auto", help="comma list of cpu,cuda or 'auto'")
    args = parser.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) if args.devices == "auto" \
        else [d.strip() for d in args.devices.split(",")]
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda available: {torch.cuda.is_available()}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print(f"model {args.model} | CPU threads: {torch.get_num_threads()}\n")

    # Scenario of Mistral's announcement: one series, 3 weeks of 15-min history, 1 day ahead.
    single = make_context(1, 2016)
    batch = make_context(64, 2016, seed=1)
    horizon = 96
    results: dict = {"model": args.model, "torch": torch.__version__, "devices": {}}
    reference: dict[str, np.ndarray] = {}
    failures: list[str] = []

    for device in devices:
        print(f"--- {device}")
        start = time.perf_counter()
        forecaster = TimestralForecaster.from_pretrained(args.model, device=device)
        sync(device)
        load_s = time.perf_counter() - start
        print(f"loaded in {load_s:.2f} s on {forecaster.device}")
        forecast = forecaster.forecast(single, prediction_length=horizon)
        for problem in check_forecast(forecast, 1, horizon):
            failures.append(f"{device}: {problem}")
        reference[device] = forecast.quantiles.detach().cpu().numpy()
        one = bench(forecaster, single, horizon, device, args.repeats)
        many = bench(forecaster, batch, horizon, device, max(3, args.repeats // 4))
        print(f"1 series x 2016 -> 96  : {one}")
        print(f"64 series x 2016 -> 96 : {many}")
        results["devices"][device] = {"load_s": round(load_s, 2), "single": one, "batch64": many}
        del forecaster
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "cpu" in reference and "cuda" in reference:
        scale = float(np.ptp(single)) or 1.0
        diff = np.abs(reference["cpu"] - reference["cuda"])
        results["cpu_vs_cuda"] = {"max_abs_diff": float(diff.max()), "max_abs_diff_over_series_range": float(diff.max() / scale)}
        print(f"\nCPU vs CUDA forecasts: max |diff| = {diff.max():.3e} ({diff.max() / scale:.2e} of the series range)")

    out_dir = find_root() / "runs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "phase0-smoke.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nOK: shapes, ordering and finiteness verified on every device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
