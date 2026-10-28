"""Inference runs in a separate worker process.

A crash of PyTorch/CUDA (out-of-memory, driver fault, segfault) kills only the worker: the web server keeps
answering, reports the failure, and starts a fresh worker on the next request. The worker is started lazily
(spawn, so no CUDA state is inherited) and handles one operation at a time; progress and cancellation travel
over a queue and an event.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tsfm_lab.api.jobs import JobCancelled


class WorkerCrashed(RuntimeError):
    pass


def axis(series, start: int, stop: int) -> list[Any]:
    """Labels for steps [start, stop) of a series; timestamps are extended past the end using the last step."""
    ts = series.timestamps
    if ts is None:
        return list(range(start, stop))
    step = (ts[-1] - ts[-2]) if len(ts) > 1 else np.timedelta64(1, "s")
    return [str(ts[i] if i < len(ts) else ts[-1] + step * (i - len(ts) + 1)) for i in range(start, stop)]


# ====================================================================== worker side
class Services:
    """Everything the worker needs, built inside the worker process."""

    def __init__(self, root: Path) -> None:
        from tsfm_lab.data.catalog import DatasetCatalog
        from tsfm_lab.models.registry import default_registry
        from tsfm_lab.store.store import BenchmarkStore, RunStore

        self.root = root
        self.registry = default_registry()
        self.catalog = DatasetCatalog(root / "data" / "user")
        self.store, self.bench = RunStore(root / "runs"), BenchmarkStore(root / "runs")
        self._pool: OrderedDict[tuple[str, str], Any] = OrderedDict()

    def pooled(self, model_id: str, device: str):
        """Keep the last two loaded models so interactive forecasts do not pay the load time every click."""
        key = (model_id, device)
        if key in self._pool:
            self._pool.move_to_end(key)
            return self._pool[key], None
        adapter = self.registry.create(model_id)
        meta = adapter.load(device)
        self._pool[key] = adapter
        while len(self._pool) > 2:
            self._pool.popitem(last=False)[1].unload()
        return adapter, meta

    def drop(self, model_id: str) -> None:
        for key in [k for k in self._pool if k[0] == model_id]:
            self._pool.pop(key).unload()


def do_forecast(svc: Services, body: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.eval import metrics as M
    from tsfm_lab.models.base import ForecastRequest

    ds = svc.catalog.resolve(body["dataset"])
    if body["series"] >= len(ds.series):
        raise KeyError("series index out of range")
    s = ds.series[body["series"]]
    horizon = body["horizon"]
    origin = body["origin"] if body["origin"] is not None else max(16, s.length - horizon)
    origin = min(origin, s.length)
    if origin < 8:
        raise ValueError("need at least 8 steps of history before the forecast origin")
    end = origin + horizon
    use_cov = body["use_covariates"]
    if ds.n_known and use_cov and end > s.length:
        raise ValueError("known covariates do not cover the requested horizon: move the origin back")
    lo = max(0, origin - body["context_length"]) if body["context_length"] else 0
    ctx = s.target[:, lo:origin][None]
    past = s.past_covariates[:, lo:origin][None] if (use_cov and ds.n_past) else None
    known = s.known_covariates[:, lo:end][None] if (use_cov and ds.n_known) else None
    actual = s.target[:, origin:end]
    actual = np.concatenate([actual, np.full((actual.shape[0], horizon - actual.shape[1]), np.nan)], axis=1)
    scale = M.history_scale(s.target[None, :, :origin], ds.seasonal_period)
    outputs: dict[str, Any] = {}
    for mid in [body["model"], *[b for b in body["baselines"] if b != body["model"]]]:
        info = svc.registry.info(mid)
        if info.kind == "baseline":
            adapter, loaded = svc.registry.create(mid), None
            adapter.load("cpu")
        else:
            adapter, loaded = svc.pooled(mid, body["device"])
        res = adapter.forecast(ForecastRequest(ctx, horizon, past, known, seasonal_period=ds.seasonal_period,
                                               options=dict(body["options"])))
        entry = {"device": res.device, "levels": res.quantile_levels, "quantiles": res.quantiles[0],
                 "timings": {**res.timings, "model_load_s": None if loaded is None else loaded.get("load_s"),
                             "first_call_after_load": loaded is not None},
                 "warnings": res.warnings, "name": info.name, "kind": info.kind}
        if np.isfinite(actual).any():
            items = M.item_metrics(actual[None], res.quantiles, res.quantile_levels, scale)
            entry["metrics"] = {k: float(v[0, 0]) for k, v in items.items() if np.isfinite(v[0, 0])}
        outputs[mid] = entry
    tail = min(origin - lo, 1024)
    return {"series_id": s.id, "origin": origin, "horizon": horizon, "target_names": ds.target_names,
            "x_history": axis(s, origin - tail, origin), "x_future": axis(s, origin, end),
            "history": ctx[0][:, -tail:], "actual": actual, "models": outputs,
            "known": None if known is None else known[0][:, -(tail + horizon):] if known.shape[2] >= tail + horizon else known[0],
            "known_names": ds.known_names, "freq": ds.freq, "seasonal_period": ds.seasonal_period,
            "context_used": int(ctx.shape[2])}


def do_experiment(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.eval.backtest import BacktestConfig
    from tsfm_lab.eval.runner import run_experiment

    ds = svc.catalog.resolve(a["dataset"])
    cfg = BacktestConfig(horizon=a["horizon"], n_windows=a["windows"], stride=a["stride"], max_context=a["max_context"],
                         max_series=a["max_series"], use_covariates=a["use_covariates"], options=a["options"])
    svc.drop(a["model"])  # free the pooled copy: the experiment loads its own and times it cleanly
    _, run_id = run_experiment(a["model"], ds, cfg, a["device"], tuple(a["baselines"]), svc.store, svc.registry,
                               a["label"], progress=progress, request=a)
    return {"run_id": run_id}


def do_benchmark(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.eval.benchmark import run_benchmark

    svc.drop(a["model"])
    result = run_benchmark(a["model"], a["devices"], a["repeats"], tuple(a["scenarios"]), svc.registry, progress=progress)
    return {"benchmark_id": svc.bench.save(result)}


def do_download(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.models import cache

    progress(0, 0, "downloading from Hugging Face…")
    return {"cache": cache.download(a["repo"])}


def do_remove(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.models import cache

    svc.drop(a["model"])
    return {"removed": cache.remove(a["repo"])}


def do_selftest_crash(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    os._exit(13)  # simulates a native crash; only reachable when TSFM_LAB_TEST_OPS=1


def do_selftest_sleep(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    time.sleep(a.get("seconds", 60))  # a long, uninterruptible computation; only when TSFM_LAB_TEST_OPS=1
    return {"slept": True}


def do_showcase_prepare(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.data.hub import download
    from tsfm_lab.showcase import engine

    sc = engine.scenario(a["scenario"])
    progress(0, 0, f"downloading {sc.name} ({sc.size_mb} MB)…")
    download(sc.name, sc.freq)
    engine._CACHE.pop(sc.id, None)
    return {"ready": True}


def do_catalog_download(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.data import hub_catalog

    e = hub_catalog.entry(a["entry"])
    what = e["files"][0]["path"] if len(e["files"]) == 1 else f"{len(e['files'])} files"
    progress(0, 0, f"downloading {what} ({e['size_bytes'] / 1e6:.1f} MB) and checking the SHA-256…")
    hub_catalog.download(e)
    svc.catalog._cache.clear()  # a dataset resolved earlier must not outlive a re-download
    return {"downloaded": True}


def do_showcase_forecast(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.showcase import engine

    return engine.forecast(svc, a, progress)


def do_showcase_impact(svc: Services, a: dict[str, Any], progress) -> dict[str, Any]:
    from tsfm_lab.showcase import engine

    svc.drop(a["model"])
    return engine.impact(svc, a, progress)


OPS: dict[str, Callable] = {"catalog_download": do_catalog_download, "showcase_prepare": do_showcase_prepare, "showcase_forecast": do_showcase_forecast,
                            "showcase_impact": do_showcase_impact,"forecast": do_forecast, "experiment": do_experiment, "benchmark": do_benchmark,
                            "download": do_download, "remove": do_remove}
if os.environ.get("TSFM_LAB_TEST_OPS") == "1":
    OPS["crash"] = do_selftest_crash
    OPS["sleep"] = do_selftest_sleep


def worker_main(root: str, requests, responses, cancel) -> None:
    import tsfm_lab  # noqa: F401  (cache isolation before torch / huggingface_hub)

    svc = Services(Path(root))
    responses.put(("ready", None, None, None))
    while True:
        call_id, op, payload = requests.get()
        if op == "stop":
            return
        cancel.clear()

        def progress(done: int, total: int, message: str) -> None:
            if cancel.is_set():
                raise JobCancelled()
            responses.put(("progress", call_id, (done, total, message), None))

        try:
            result = OPS[op](svc, payload, progress)
            responses.put(("done", call_id, result, None))
        except JobCancelled:
            responses.put(("error", call_id, "JobCancelled", "cancelled"))
        except BaseException as error:  # noqa: BLE001 - always report, then keep serving
            responses.put(("error", call_id, type(error).__name__, str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)))


# ====================================================================== server side
class WorkerClient:
    """Talks to the worker process; restarts it if it dies. One call at a time (the caller holds a lock)."""

    STOP_GRACE_S = 4.0  # time a computation gets to stop on its own after a cancel request

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._guard = threading.Lock()
        self._next = 0

    def _start(self) -> None:
        self._req, self._res = self._ctx.Queue(), self._ctx.Queue()
        self._cancel = self._ctx.Event()
        self._proc = self._ctx.Process(target=worker_main, args=(str(self.root), self._req, self._res, self._cancel),
                                       daemon=True, name="tsfm-lab-worker")
        self._proc.start()
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                if self._res.get(timeout=0.5)[0] == "ready":
                    return
            except queue.Empty:
                if not self._proc.is_alive():
                    break
        self.stop()
        raise WorkerCrashed("the inference worker could not start")

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def call(self, op: str, payload: dict[str, Any], on_progress: Callable[[int, int, str], None] | None = None,
             cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        with self._guard:
            if not self.alive:
                self._start()
            self._next += 1
            call_id = self._next
            self._req.put((call_id, op, payload))
            asked_at: float | None = None
            while True:
                if cancelled and cancelled():
                    self._cancel.set()
                    asked_at = asked_at or time.time()
                    if time.time() - asked_at > self.STOP_GRACE_S:  # a long step does not look at the flag: stop the process
                        self.stop()
                        raise JobCancelled("stopped")
                try:
                    kind, rid, data, extra = self._res.get(timeout=0.3)
                except queue.Empty:
                    if not self._proc.is_alive():
                        code = self._proc.exitcode
                        self._proc = None
                        raise WorkerCrashed(f"the inference worker stopped unexpectedly (exit code {code}); "
                                            "it will restart on the next request") from None
                    continue
                if rid != call_id:
                    continue
                if kind == "progress":
                    if on_progress:
                        on_progress(*data)
                elif kind == "done":
                    return data
                elif kind == "error":
                    exc = {"KeyError": KeyError, "ValueError": ValueError, "JobCancelled": JobCancelled}.get(data, RuntimeError)
                    raise exc(extra)

    def stop(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._req.put((0, "stop", None))
                    self._proc.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None
