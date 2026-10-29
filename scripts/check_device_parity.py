#!/usr/bin/env python3
"""Does a GPU give the same forecasts as the CPU, and how much faster is it? Run it on a new machine (e.g. a Mac).

    .venv/bin/python scripts/check_device_parity.py --device mps --model timestral-5m
    .venv/bin/python scripts/check_device_parity.py --device mps --all --download      # every model, fetching missing weights first

Uses a synthetic series, so it needs only the model weights. Exit code 1 if the devices disagree beyond float32 noise (or a model fails).
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

from tsfm_lab import isolation

isolation.apply(ROOT)

import numpy as np

from tsfm_lab.models import cache
from tsfm_lab.models.base import ForecastRequest
from tsfm_lab.models.registry import default_registry
from tsfm_lab.system.devices import list_devices


def compare(model: str, device: str, series: int, repeats: int, tolerance: float) -> dict:
    """Forecast the same input on the CPU and on ``device``; print the details and return the worst difference and the speed ratios."""
    registry = default_registry()
    info = registry.info(model)
    t = np.arange(512)
    rng = np.random.default_rng(0)
    ctx = (50 + 10 * np.sin(2 * np.pi * t / 24)[None, None, :] + rng.normal(0, 1, (series, 1, 512))).astype(np.float32)
    covariate = np.sin(2 * np.pi * np.arange(512 + 48) / 24).astype(np.float32)[None, None, :].repeat(series, axis=0)
    cases = {"no covariates, 48 steps": ForecastRequest(ctx, 48, seasonal_period=24)}
    if info.capabilities.known_covariates:
        cases["known covariate, 48 steps"] = ForecastRequest(ctx, 48, known_covariates=covariate, seasonal_period=24)
    cases["1200 steps (decoded in blocks)"] = ForecastRequest(ctx[:4], 1200, seasonal_period=24)

    results: dict[str, dict[str, tuple[np.ndarray, float, float]]] = {}
    for dev in ("cpu", device):
        adapter = registry.create(model)
        adapter.load(dev)
        adapter.forecast(ForecastRequest(ctx[:2], 8, seasonal_period=24))  # warm-up: first call pays one-off costs
        for name, request in cases.items():
            start = time.perf_counter()
            out = adapter.forecast(request)
            first = time.perf_counter() - start
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                adapter.forecast(request)
                times.append(time.perf_counter() - start)
            results.setdefault(name, {})[dev] = (out.quantiles, first, statistics.median(times))
        adapter.unload()

    worst, speeds = 0.0, []
    print(f"{model}: CPU vs {device}")
    for name, per_device in results.items():
        (a, first_a, ta), (b, first_b, tb) = per_device["cpu"], per_device[device]
        rel = float(np.abs(a - b).max() / max(np.abs(a).mean(), 1e-9))
        worst = max(worst, rel)
        speeds.append(ta / tb)
        print(f"  {name}")
        print(f"      same forecasts: max relative difference {rel:.2e}")
        print(f"      CPU    first call {first_a:6.3f} s   steady {ta:6.3f} s")
        print(f"      {device:6s} first call {first_b:6.3f} s   steady {tb:6.3f} s   -> {device} is {ta / tb:.2f}x the speed of the CPU")
    ok = worst <= tolerance
    print("OK: same forecasts up to float32 noise." if ok else f"MISMATCH: worst difference {worst:.2e} > {tolerance}.")
    return {"worst": worst, "ok": ok, "speed_min": min(speeds), "speed_max": max(speeds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", help="device to compare with the CPU (mps, cuda:0)")
    parser.add_argument("--model", action="append", help="a model id (repeat for several); default timestral-5m")
    parser.add_argument("--all", action="store_true", help="every foundation model of the registry")
    parser.add_argument("--download", action="store_true", help="fetch the weights of a model that is not downloaded yet (the same download the app does)")
    parser.add_argument("--series", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7, help="calls timed per case after the first one")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="max relative difference accepted")
    args = parser.parse_args()

    known = {d.id: d for d in list_devices()}
    dev = known.get(args.device)
    if dev is None or not dev.supported:
        sys.exit(f"{args.device}: {'not found' if dev is None else dev.reason}")

    registry = default_registry()
    if args.all:
        models = [i.id for i in registry.infos() if i.kind == "foundation" and not i.details.get("custom_revision")]
    else:
        models = args.model or ["timestral-5m"]

    rows: list[tuple[str, str]] = []
    failed = False
    for model in models:
        info = registry.info(model)
        if dev.kind not in info.capabilities.device_kinds:
            rows.append((model, f"skipped: no {dev.kind} support"))
            continue
        repo = info.details.get("repo")
        if repo and not cache.status(repo)["downloaded"]:
            if not args.download:
                rows.append((model, "skipped: weights not downloaded (add --download)"))
                continue
            print(f"{model}: downloading {repo} ...", flush=True)
            cache.download(repo)
        try:
            res = compare(model, args.device, args.series, args.repeats, args.tolerance)
        except Exception as error:  # noqa: BLE001 - one failing model must not hide the others
            print(f"{model}: FAILED {type(error).__name__}: {error}")
            rows.append((model, f"FAILED: {type(error).__name__}"))
            failed = True
            continue
        failed = failed or not res["ok"]
        rows.append((model, f"{'OK' if res['ok'] else 'MISMATCH'}: worst difference {res['worst']:.1e}, {args.device} {res['speed_min']:.2f}-{res['speed_max']:.2f}x the CPU"))
        print()

    if len(models) > 1:
        print(f"== Summary, CPU vs {args.device}")
        for model, text in rows:
            print(f"  {model:16s} {text}")
    elif rows and rows[0][1].startswith("skipped"):
        sys.exit(f"{rows[0][0]}: {rows[0][1]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
