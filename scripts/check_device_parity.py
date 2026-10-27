#!/usr/bin/env python3
"""Does a GPU give the same forecasts as the CPU, and how much faster is it? Run it on a new machine (e.g. a Mac).

    .venv/bin/python scripts/check_device_parity.py --device mps --model timestral-5m

Uses a synthetic series, so it needs only the model weights. Exit code 1 if the devices disagree beyond float32 noise.
Each case is timed twice: the FIRST call (a GPU compiles its kernels for every new data shape, so it is slow) and the
STEADY state (median of --repeats further calls), which is what you feel when using the app.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from _common import ROOT

sys.path.insert(0, str(ROOT / "src"))

from tsfm_lab import isolation  # noqa: E402

isolation.apply(ROOT)

import numpy as np  # noqa: E402

from tsfm_lab.models.base import ForecastRequest  # noqa: E402
from tsfm_lab.models.registry import default_registry  # noqa: E402
from tsfm_lab.system.devices import list_devices  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", help="device to compare with the CPU (mps, cuda:0)")
    parser.add_argument("--model", default="timestral-5m")
    parser.add_argument("--series", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7, help="calls timed per case after the first one")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="max relative difference accepted")
    args = parser.parse_args()

    known = {d.id: d for d in list_devices()}
    dev = known.get(args.device)
    if dev is None or not dev.supported:
        sys.exit(f"{args.device}: {'not found' if dev is None else dev.reason}")

    t = np.arange(512)
    rng = np.random.default_rng(0)
    ctx = (50 + 10 * np.sin(2 * np.pi * t / 24)[None, None, :] + rng.normal(0, 1, (args.series, 1, 512))).astype(np.float32)
    covariate = np.sin(2 * np.pi * np.arange(512 + 48) / 24).astype(np.float32)[None, None, :].repeat(args.series, axis=0)
    cases = {"no covariates, 48 steps": ForecastRequest(ctx, 48, seasonal_period=24),
             "known covariate, 48 steps": ForecastRequest(ctx, 48, known_covariates=covariate, seasonal_period=24),
             "1200 steps (decoded in blocks)": ForecastRequest(ctx[:4], 1200, seasonal_period=24)}

    results: dict[str, dict[str, tuple[np.ndarray, float, float]]] = {}
    for device in ("cpu", args.device):
        adapter = default_registry().create(args.model)
        adapter.load(device)
        adapter.forecast(ForecastRequest(ctx[:2], 8, seasonal_period=24))  # warm-up: first call pays one-off costs
        for name, request in cases.items():
            start = time.perf_counter()
            out = adapter.forecast(request)
            first = time.perf_counter() - start
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                adapter.forecast(request)
                times.append(time.perf_counter() - start)
            results.setdefault(name, {})[device] = (out.quantiles, first, statistics.median(times))
        adapter.unload()

    worst = 0.0
    print(f"{args.model}: CPU vs {args.device}")
    for name, per_device in results.items():
        (a, first_a, ta), (b, first_b, tb) = per_device["cpu"], per_device[args.device]
        rel = float(np.abs(a - b).max() / max(np.abs(a).mean(), 1e-9))
        worst = max(worst, rel)
        print(f"  {name}")
        print(f"      same forecasts: max relative difference {rel:.2e}")
        print(f"      CPU    first call {first_a:6.3f} s   steady {ta:6.3f} s")
        print(f"      {args.device:6s} first call {first_b:6.3f} s   steady {tb:6.3f} s   -> {args.device} is {ta / tb:.2f}x the speed of the CPU")
    ok = worst <= args.tolerance
    print("OK: same forecasts up to float32 noise." if ok else f"MISMATCH: worst difference {worst:.2e} > {args.tolerance}.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
