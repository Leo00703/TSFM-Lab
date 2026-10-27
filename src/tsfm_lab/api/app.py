"""Local web API + static UI. Bound to 127.0.0.1 only; requests from other hosts/origins are rejected."""

from __future__ import annotations

import csv
import io
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tsfm_lab.api.jobs import JobManager
from tsfm_lab.data.catalog import DatasetCatalog
from tsfm_lab.eval import metrics as M
from tsfm_lab.eval.backtest import BacktestConfig
from tsfm_lab.eval.benchmark import SCENARIOS, run_benchmark
from tsfm_lab.eval.runner import DEFAULT_BASELINES, run_experiment
from tsfm_lab.isolation import find_root
from tsfm_lab.models import cache as weights
from tsfm_lab.models.base import ForecastRequest, ModelAdapter
from tsfm_lab.models.registry import Registry, default_registry
from tsfm_lab.store.store import BenchmarkStore, RunStore, to_jsonable
from tsfm_lab.system.devices import list_devices
from tsfm_lab.system.probe import environment_info

ALLOWED_OPTIONS = {"symmetric_averaging": bool, "clip_nonnegative": bool, "mask_flat_covariates": bool,
                   "quantile_real_cap_k": float, "max_packed_rows": int}


# ---------------------------------------------------------------------------- request bodies
class DatasetSpec(BaseModel):
    source: Literal["synthetic", "local"]
    id: str
    params: dict[str, Any] = Field(default_factory=dict)


class ForecastBody(BaseModel):
    dataset: DatasetSpec
    series: int = Field(0, ge=0)
    origin: int | None = Field(None, ge=1)
    horizon: int = Field(24, ge=1, le=4096)
    model: str
    device: str = "cpu"
    baselines: list[str] = Field(default_factory=lambda: ["seasonal-naive"])
    context_length: int | None = Field(None, ge=8, le=200_000)
    use_covariates: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class ExperimentBody(BaseModel):
    model: str
    dataset: DatasetSpec
    device: str = "cpu"
    horizon: int = Field(24, ge=1, le=4096)
    windows: int = Field(3, ge=1, le=50)
    stride: int | None = Field(None, ge=1)
    baselines: list[str] = Field(default_factory=lambda: list(DEFAULT_BASELINES))
    max_series: int | None = Field(None, ge=1, le=2000)
    max_context: int | None = Field(None, ge=8, le=200_000)
    use_covariates: bool = True
    options: dict[str, Any] = Field(default_factory=dict)
    label: str | None = Field(None, max_length=80)


class BenchmarkBody(BaseModel):
    model: str
    devices: list[str] = Field(default_factory=lambda: ["cpu"])
    repeats: int = Field(15, ge=1, le=200)
    scenarios: list[str] = Field(default_factory=lambda: list(SCENARIOS))


class RegisterBody(BaseModel):
    file_id: str
    name: str | None = Field(None, max_length=80)
    mapping: dict[str, Any]


def clean_options(options: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in options.items():
        if key not in ALLOWED_OPTIONS:
            raise HTTPException(422, f"unknown option '{key}'")
        out[key] = ALLOWED_OPTIONS[key](value)
    return out


# ---------------------------------------------------------------------------- helpers
class ModelPool:
    """Keeps the last few loaded models so interactive forecasts do not pay the load time every click."""

    def __init__(self, registry: Registry, size: int = 2) -> None:
        self.registry, self.size = registry, size
        self._items: OrderedDict[tuple[str, str], ModelAdapter] = OrderedDict()

    def get(self, model_id: str, device: str) -> tuple[ModelAdapter, dict[str, Any] | None]:
        key = (model_id, device)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key], None
        adapter = self.registry.create(model_id)
        meta = adapter.load(device)
        self._items[key] = adapter
        while len(self._items) > self.size:
            self._items.popitem(last=False)[1].unload()
        return adapter, meta

    def drop_model(self, model_id: str) -> None:
        for key in [k for k in self._items if k[0] == model_id]:
            self._items.pop(key).unload()


def axis(series, start: int, stop: int) -> list[Any]:
    """Labels for steps [start, stop) of a series; timestamps are extended past the end using the last step."""
    ts = series.timestamps
    if ts is None:
        return list(range(start, stop))
    step = (ts[-1] - ts[-2]) if len(ts) > 1 else np.timedelta64(1, "s")
    return [str(ts[i] if i < len(ts) else ts[-1] + step * (i - len(ts) + 1)) for i in range(start, stop)]


def downsample_indices(n: int, limit: int) -> np.ndarray:
    return np.arange(n) if n <= limit else np.unique(np.linspace(0, n - 1, limit).astype(int))


def create_app(registry: Registry | None = None, root: Path | None = None,
               allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]")) -> FastAPI:
    root = Path(root) if root else find_root()
    registry = registry or default_registry()
    catalog = DatasetCatalog(root / "data" / "user")
    store, bench_store = RunStore(root / "runs"), BenchmarkStore(root / "runs")
    compute_lock = threading.Lock()
    jobs = JobManager(compute_lock)
    pool = ModelPool(registry)
    app = FastAPI(title="tsfm-lab", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def ok(payload: Any, status: int = 200) -> JSONResponse:
        return JSONResponse(to_jsonable(payload), status_code=status)

    # ------------------------------------------------------------------ security
    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0] if not request.headers.get("host", "").startswith("[") \
            else request.headers["host"].split("]")[0] + "]"
        if host not in allowed_hosts:
            return JSONResponse({"detail": "this server only answers to localhost"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            if origin.split("://", 1)[-1] != request.headers.get("host"):
                return JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(KeyError)
    async def key_error(_: Request, error: KeyError):
        return JSONResponse({"detail": str(error.args[0])}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, error: ValueError):
        return JSONResponse({"detail": str(error)}, status_code=422)

    # ------------------------------------------------------------------ system / brand
    @app.get("/api/health")
    def health():
        return ok({"ok": True, "time": time.time()})

    @app.get("/api/system")
    def system():
        return ok({"environment": environment_info(), "devices": [d.__dict__ for d in list_devices()],
                   "busy": compute_lock.locked()})

    @app.get("/api/system/disk")
    def system_disk():
        from tsfm_lab.system.disk import usage  # walks tens of thousands of files: only on demand

        return ok(usage(root))

    @app.get("/api/brand")
    def brand():
        base = root / "web" / "assets" / "mistral"
        return ok({"fonts": len(list((base / "fonts").glob("*.woff2"))) if (base / "fonts").exists() else 0,
                   "logo": (base / "logos" / "timestral-icon-model.svg").exists(),
                   "hourglass": (base / "logos" / "timestral-hourglass.png").exists()})

    # ------------------------------------------------------------------ models
    @app.get("/api/models")
    def models():
        latest: dict[str, dict] = {}
        for b in bench_store.list():
            if b["model"] not in latest:
                full = bench_store.get(b["id"])
                latest[b["model"]] = {
                    "id": b["id"], "created": b["created"],
                    "devices": {dev: {"load_s": d["load_s"], "cold_first_call_ms": d["cold_first_call_ms"],
                                      "mistral_median_ms": next((c["median_ms"] for c in d["cells"] if c["scenario"] == "mistral"), None),
                                      "peak_vram_mb": next((c["peak_vram_mb"] for c in d["cells"] if c["scenario"] == "mistral"), None)}
                                for dev, d in full["devices"].items()}}
        out = []
        for info in registry.infos():
            entry = to_jsonable(info.__dict__ | {"capabilities": info.capabilities.__dict__})
            repo = info.details.get("repo")
            if repo:
                entry["cache"] = weights.status(repo)
                cfg = weights.read_config(repo) if entry["cache"]["downloaded"] else {}
                entry["details"] = {**entry["details"], **{k: cfg[k] for k in ("num_heads", "mlp_hidden_dim") if k in cfg}}
            entry["local_benchmark"] = latest.get(info.id)
            out.append(entry)
        return ok(out)

    @app.post("/api/models/{model_id}/download")
    def download(model_id: str):
        repo = registry.info(model_id).details.get("repo")
        if not repo:
            raise HTTPException(400, "this model has no weights to download")
        job = jobs.submit("download", f"Download {registry.info(model_id).name}", lambda j: (
            j.update(message="downloading from Hugging Face…"), {"cache": weights.download(repo)})[1])
        return ok(job.to_dict(), 202)

    @app.delete("/api/models/{model_id}/cache")
    def remove_cache(model_id: str):
        repo = registry.info(model_id).details.get("repo")
        if not repo:
            raise HTTPException(400, "nothing to remove")
        if not compute_lock.acquire(blocking=False):
            raise HTTPException(409, "a job is running; try again when it finishes")
        try:
            pool.drop_model(model_id)
            return ok({"removed": weights.remove(repo)})
        finally:
            compute_lock.release()

    # ------------------------------------------------------------------ datasets
    @app.get("/api/datasets")
    def datasets():
        return ok(catalog.list())

    @app.post("/api/datasets/upload")
    async def upload(request: Request):
        name = request.headers.get("x-filename", "upload")
        data = await request.body()
        return ok(catalog.upload(name, data))

    @app.post("/api/datasets/register")
    def register(body: RegisterBody):
        return ok(catalog.register_local(body.file_id, body.mapping, body.name))

    @app.delete("/api/datasets/local/{dataset_id}")
    def delete_local(dataset_id: str):
        catalog.delete_local(dataset_id)
        return ok({"deleted": dataset_id})

    @app.post("/api/datasets/inspect")
    def inspect(spec: DatasetSpec):
        ds = catalog.resolve(spec.model_dump())
        return ok({**ds.summary(), "target_names": ds.target_names, "past_names": ds.past_names,
                   "known_names": ds.known_names,
                   "series": [{"id": s.id, "length": s.length} for s in ds.series[:500]]})

    @app.post("/api/datasets/series")
    def dataset_series(spec: DatasetSpec, index: int = 0, max_points: int = 4000):
        ds = catalog.resolve(spec.model_dump())
        if not 0 <= index < len(ds.series):
            raise HTTPException(404, "series index out of range")
        s = ds.series[index]
        keep = downsample_indices(s.length, max(100, min(max_points, 50_000)))
        x = axis(s, 0, s.length)
        return ok({"id": s.id, "length": s.length, "x": [x[i] for i in keep], "target": s.target[:, keep],
                   "target_names": ds.target_names,
                   "known": None if s.known_covariates is None else s.known_covariates[:, keep],
                   "past": None if s.past_covariates is None else s.past_covariates[:, keep],
                   "known_names": ds.known_names, "past_names": ds.past_names, "downsampled": len(keep) < s.length})

    # ------------------------------------------------------------------ interactive forecast
    @app.post("/api/forecast")
    def forecast(body: ForecastBody):
        options = clean_options(body.options)
        ds = catalog.resolve(body.dataset.model_dump())
        if body.series >= len(ds.series):
            raise HTTPException(404, "series index out of range")
        s = ds.series[body.series]
        origin = body.origin if body.origin is not None else max(16, s.length - body.horizon)
        origin = min(origin, s.length)
        if origin < 8:
            raise HTTPException(422, "need at least 8 steps of history before the forecast origin")
        end = origin + body.horizon
        if ds.n_known and body.use_covariates and end > s.length:
            raise HTTPException(422, "known covariates do not cover the requested horizon: move the origin back")
        lo = max(0, origin - body.context_length) if body.context_length else 0
        ctx = s.target[:, lo:origin][None]
        horizon = body.horizon
        past = s.past_covariates[:, lo:origin][None] if (body.use_covariates and ds.n_past) else None
        known = s.known_covariates[:, lo:end][None] if (body.use_covariates and ds.n_known) else None
        actual = s.target[:, origin:end]
        actual = np.concatenate([actual, np.full((actual.shape[0], horizon - actual.shape[1]), np.nan)], axis=1)
        if not compute_lock.acquire(blocking=False):
            raise HTTPException(409, "a job is running on this machine; wait for it to finish")
        try:
            outputs: dict[str, Any] = {}
            wanted = [body.model, *[b for b in body.baselines if b != body.model]]
            scale = M.history_scale(s.target[None, :, :origin], ds.seasonal_period)
            for mid in wanted:
                info = registry.info(mid)
                if info.kind == "baseline":
                    adapter, loaded = registry.create(mid), None
                    adapter.load("cpu")
                else:
                    adapter, loaded = pool.get(mid, body.device)
                req = ForecastRequest(ctx, horizon, past, known, seasonal_period=ds.seasonal_period, options=dict(options))
                res = adapter.forecast(req)
                entry = {"device": res.device, "levels": res.quantile_levels, "quantiles": res.quantiles[0],
                         "timings": {**res.timings, "model_load_s": None if loaded is None else loaded.get("load_s"),
                                     "first_call_after_load": loaded is not None},
                         "warnings": res.warnings, "name": info.name, "kind": info.kind}
                if np.isfinite(actual).any():
                    items = M.item_metrics(actual[None], res.quantiles, res.quantile_levels, scale)
                    entry["metrics"] = {k: float(v[0, 0]) for k, v in items.items() if np.isfinite(v[0, 0])}
                outputs[mid] = entry
        finally:
            compute_lock.release()
        tail = min(origin - lo, 1024)
        return ok({"series_id": s.id, "origin": origin, "horizon": horizon, "target_names": ds.target_names,
                   "x_history": axis(s, origin - tail, origin), "x_future": axis(s, origin, end),
                   "history": ctx[0][:, -tail:], "actual": actual, "models": outputs,
                   "known": None if known is None else known[0][:, -(tail + horizon):] if known.shape[2] >= tail + horizon else known[0],
                   "known_names": ds.known_names, "freq": ds.freq, "seasonal_period": ds.seasonal_period,
                   "context_used": int(ctx.shape[2])})

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs/experiment")
    def start_experiment(body: ExperimentBody):
        options = clean_options(body.options)
        ds = catalog.resolve(body.dataset.model_dump())
        registry.info(body.model)
        for b in body.baselines:
            registry.info(b)
        cfg = BacktestConfig(horizon=body.horizon, n_windows=body.windows, stride=body.stride,
                             max_context=body.max_context, max_series=body.max_series,
                             use_covariates=body.use_covariates, options=options)

        def work(job):
            _, run_id = run_experiment(body.model, ds, cfg, body.device, tuple(body.baselines), store, registry,
                                       body.label, progress=lambda d, t, m: job.update(d, t, m))
            return {"run_id": run_id}

        label = f"{registry.info(body.model).name} on {ds.name} · {body.device}"
        return ok(jobs.submit("experiment", label, work).to_dict(), 202)

    @app.post("/api/jobs/benchmark")
    def start_benchmark(body: BenchmarkBody):
        registry.info(body.model)
        if not set(body.scenarios) <= set(SCENARIOS):
            raise HTTPException(422, f"scenarios must be among {SCENARIOS}")
        valid = {d.id for d in list_devices() if d.supported}
        devices = ["cuda:0" if d == "cuda" else d for d in body.devices]
        if not set(devices) <= valid:
            raise HTTPException(422, f"unavailable device(s): {sorted(set(devices) - valid)}")

        def work(job):
            result = run_benchmark(body.model, devices, body.repeats, tuple(body.scenarios), registry,
                                   progress=lambda d, t, m: job.update(d, t, m))
            return {"benchmark_id": bench_store.save(result)}

        return ok(jobs.submit("benchmark", f"Benchmark {registry.info(body.model).name} · {', '.join(devices)}",
                              work).to_dict(), 202)

    @app.get("/api/jobs")
    def list_jobs():
        return ok([j.to_dict() for j in jobs.list()])

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return ok(jobs.get(job_id).to_dict())

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return ok(jobs.cancel(job_id).to_dict())

    # ------------------------------------------------------------------ benchmarks
    @app.get("/api/benchmarks")
    def benchmarks():
        return ok(bench_store.list())

    @app.get("/api/benchmarks/{bench_id}")
    def benchmark(bench_id: str):
        return ok(bench_store.get(bench_id))

    @app.delete("/api/benchmarks/{bench_id}")
    def delete_benchmark(bench_id: str):
        bench_store.delete(bench_id)
        return ok({"deleted": bench_id})

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def runs():
        return ok(store.list_runs())

    @app.get("/api/runs/{run_id}")
    def run_meta(run_id: str):
        return ok(store.meta(run_id))

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        store.meta(run_id)
        store.delete(run_id)
        return ok({"deleted": run_id})

    @app.get("/api/runs/{run_id}/items")
    def run_items(run_id: str):
        meta, arrays = store.meta(run_id), store.arrays(run_id)
        names = ("mase", "sql", "coverage_90", "bias")
        rows = []
        for i, sid in enumerate(meta["items"]["series"]):
            row = {"i": i, "series": sid, "window": meta["items"]["window"][i], "origin": meta["items"]["origin"][i],
                   "origin_time": meta["items"]["origin_time"][i], "metrics": {}}
            for mid in meta["models"]:
                row["metrics"][mid] = {n: arrays[f"{mid}/item/{n}"][i, :].mean() for n in names if f"{mid}/item/{n}" in arrays}
            rows.append(row)
        return ok(rows)

    @app.get("/api/runs/{run_id}/item/{i}")
    def run_item(run_id: str, i: int):
        meta, arrays = store.meta(run_id), store.arrays(run_id)
        n = arrays["y"].shape[0]
        if not 0 <= i < n:
            raise HTTPException(404, "item out of range")
        h = arrays["y"].shape[2]
        hist = arrays["history"][i]
        return ok({"i": i, "series": meta["items"]["series"][i], "window": meta["items"]["window"][i],
                   "origin": meta["items"]["origin"][i], "origin_time": meta["items"]["origin_time"][i],
                   "history": hist, "actual": arrays["y"][i], "scale": arrays["scale"][i],
                   "covariates": arrays["covariates"][i] if "covariates" in arrays else None,
                   "models": {mid: {"quantiles": arrays[f"{mid}/quantiles"][i], "levels": m["levels"], "device": m["device"],
                                    "name": m.get("name", mid), "kind": m.get("kind")}
                              for mid, m in meta["models"].items()}, "horizon": h})

    @app.get("/api/runs/{run_id}/analysis")
    def run_analysis(run_id: str, model: str):
        meta, arrays = store.meta(run_id), store.arrays(run_id)
        if model not in meta["models"]:
            raise HTTPException(404, "model not in this run")
        levels = meta["models"][model]["levels"]
        q, y, scale = arrays[f"{model}/quantiles"], arrays["y"], arrays["scale"]
        med = q[..., levels.index(0.5)]
        ok_mask = np.isfinite(y) & np.isfinite(med)
        actual, pred = y[ok_mask], med[ok_mask]
        scaled = ((med - y) / scale[..., None])[ok_mask & np.isfinite(scale)[..., None]]
        sel = downsample_indices(actual.size, 3000)
        hist = {"edges": [], "counts": []}
        if scaled.size:
            lim = float(np.percentile(np.abs(scaled), 99)) or 1.0
            counts, edges = np.histogram(np.clip(scaled, -lim, lim), bins=41, range=(-lim, lim))
            hist = {"edges": edges, "counts": counts}
        dist = {mid: {k: arrays[f"{mid}/item/{k}"].mean(axis=1) for k in ("mase", "sql", "coverage_90") if f"{mid}/item/{k}" in arrays}
                for mid in meta["models"]}
        return ok({"scatter": {"actual": actual[sel], "predicted": pred[sel], "n": int(actual.size)},
                   "residual_hist": hist, "distributions": dist})

    @app.get("/api/runs/{run_id}/export.csv")
    def export_csv(run_id: str):
        meta, arrays = store.meta(run_id), store.arrays(run_id)
        out = io.StringIO()
        w = csv.writer(out)
        metric_names = sorted({k.split("/item/")[1] for k in arrays if "/item/" in k})
        w.writerow(["model", "series", "window", "origin", "origin_time", "variate", *metric_names])
        for mid in meta["models"]:
            for i, sid in enumerate(meta["items"]["series"]):
                for v in range(arrays["y"].shape[1]):
                    w.writerow([mid, sid, meta["items"]["window"][i], meta["items"]["origin"][i],
                                meta["items"]["origin_time"][i], v,
                                *[f"{arrays[f'{mid}/item/{k}'][i, v]:.6g}" for k in metric_names]])
        return Response(out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'})

    # ------------------------------------------------------------------ static UI (last: API routes win)
    web = root / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
