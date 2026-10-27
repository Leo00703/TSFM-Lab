"""Local web API + static UI. Bound to 127.0.0.1 only; requests from other hosts/origins are rejected."""

from __future__ import annotations

import atexit
import csv
import io
import threading
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tsfm_lab.api.jobs import JobManager
from tsfm_lab.eval import metrics as M
from tsfm_lab.api.worker import WorkerClient, WorkerCrashed, axis
from tsfm_lab.data.catalog import DatasetCatalog
from tsfm_lab.eval.benchmark import SCENARIOS
from tsfm_lab.eval.runner import DEFAULT_BASELINES
from tsfm_lab.isolation import find_root
from tsfm_lab.models import cache as weights
from tsfm_lab.models.registry import Registry, default_registry
from tsfm_lab.store.store import BenchmarkStore, RunStore, to_jsonable
from tsfm_lab.system.devices import list_devices
from tsfm_lab.system.probe import environment_info

ALLOWED_OPTIONS = {"symmetric_averaging": bool, "clip_nonnegative": bool, "mask_flat_covariates": bool,
                   "quantile_real_cap_k": float, "max_packed_rows": int}


# ---------------------------------------------------------------------------- request bodies
class DatasetSpec(BaseModel):
    source: Literal["synthetic", "local", "hub"]
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


class ShowcaseBody(BaseModel):
    scenario: str
    series: str
    cut: int | None = Field(None, ge=1)
    model: str
    device: str = "cpu"
    covariates: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class ShowcaseImpactBody(BaseModel):
    scenario: str
    model: str
    device: str = "cpu"
    options: dict[str, Any] = Field(default_factory=dict)
    refresh: bool = False


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
    worker = WorkerClient(root)
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

    @app.exception_handler(WorkerCrashed)
    async def worker_crashed(_: Request, error: WorkerCrashed):
        return JSONResponse({"detail": str(error)}, status_code=503)

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
                   "busy": compute_lock.locked(), "worker_alive": worker.alive})

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
        job = jobs.submit("download", f"Download {registry.info(model_id).name}", lambda j: worker.call(
            "download", {"repo": repo}, lambda d, t, m: j.update(d, t, m), j.cancel.is_set), link=f"models?focus={model_id}")
        return ok(job.to_dict(), 202)

    @app.delete("/api/models/{model_id}/cache")
    def remove_cache(model_id: str):
        repo = registry.info(model_id).details.get("repo")
        if not repo:
            raise HTTPException(400, "nothing to remove")
        if not compute_lock.acquire(blocking=False):
            raise HTTPException(409, "a job is running; try again when it finishes")
        try:
            return ok(worker.call("remove", {"repo": repo, "model": model_id}))
        finally:
            compute_lock.release()

    # ------------------------------------------------------------------ datasets
    @app.get("/api/datasets")
    def datasets():
        return ok(catalog.list())

    # ------------------------------------------------------------------ online catalog
    @app.get("/api/catalog")
    def online_catalog():
        from tsfm_lab.data import hub_catalog

        src = hub_catalog.source()
        return ok({"source": {**src, "card_url": f"https://huggingface.co/datasets/{src['repo']}/tree/{src['revision']}",
                              "tasks_url": f"https://github.com/{src['tasks_repo']}/blob/{src['tasks_ref']}/{src['tasks_path']}"},
                   "entries": hub_catalog.listing(), "default_max_series": hub_catalog.DEFAULT_MAX_SERIES})

    @app.post("/api/catalog/{entry_id}/download")
    def catalog_download(entry_id: str):
        from tsfm_lab.data import hub_catalog

        try:
            e = hub_catalog.entry(entry_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        mb = e["size_bytes"] / 1e6
        name = f"{hub_catalog.title(e)} · {hub_catalog.freq_label(e['pandas_freq'])}"
        return ok(jobs.submit("download", f"Download {name} ({mb:.1f} MB)", lambda j: worker.call(
            "catalog_download", {"entry": entry_id}, lambda d, t, m: j.update(d, t, m), j.cancel.is_set), link=f"datasets?focus={entry_id}").to_dict(), 202)

    @app.delete("/api/catalog/{entry_id}")
    def catalog_remove(entry_id: str):
        from tsfm_lab.data import hub_catalog

        try:
            e = hub_catalog.entry(entry_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        removed = hub_catalog.remove(e)
        catalog.forget_hub(entry_id)
        return ok({"removed": removed})

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
        payload = body.model_dump()
        payload["options"] = clean_options(body.options)
        payload["dataset"] = body.dataset.model_dump()
        catalog.resolve(payload["dataset"])  # fail fast (404/422) without waking the worker
        registry.info(body.model)
        if not compute_lock.acquire(blocking=False):
            raise HTTPException(409, "a job is running on this machine; wait for it to finish")
        try:
            return ok(worker.call("forecast", payload))
        finally:
            compute_lock.release()

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs/experiment")
    def start_experiment(body: ExperimentBody):
        payload = body.model_dump()
        payload["options"] = clean_options(body.options)
        payload["dataset"] = body.dataset.model_dump()
        ds = catalog.resolve(payload["dataset"])
        registry.info(body.model)
        for b in body.baselines:
            registry.info(b)
        label = f"{registry.info(body.model).name} on {ds.name} · {body.device}"
        return ok(jobs.submit("experiment", label, lambda j: worker.call(
            "experiment", payload, lambda d, t, m: j.update(d, t, m), j.cancel.is_set)).to_dict(), 202)

    @app.post("/api/jobs/benchmark")
    def start_benchmark(body: BenchmarkBody):
        registry.info(body.model)
        if not set(body.scenarios) <= set(SCENARIOS):
            raise HTTPException(422, f"scenarios must be among {SCENARIOS}")
        valid = {d.id for d in list_devices() if d.supported}
        devices = ["cuda:0" if d == "cuda" else d for d in body.devices]
        if not set(devices) <= valid:
            raise HTTPException(422, f"unavailable device(s): {sorted(set(devices) - valid)}")
        payload = {"model": body.model, "devices": devices, "repeats": body.repeats, "scenarios": body.scenarios}
        return ok(jobs.submit("benchmark", f"Benchmark {registry.info(body.model).name} · {', '.join(devices)}",
                              lambda j: worker.call("benchmark", payload, lambda d, t, m: j.update(d, t, m),
                                                    j.cancel.is_set), link="system?focus=benchmark").to_dict(), 202)

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

    # ------------------------------------------------------------------ showcase
    @app.get("/api/showcase")
    def showcase_list():
        from tsfm_lab.data.hub import FEV_REPO, FEV_REVISION
        from tsfm_lab.showcase import engine
        from tsfm_lab.showcase.scenarios import SCENARIOS

        return ok([{
            "id": sc.id, "tab": sc.tab, "title": sc.title, "subtitle": sc.subtitle, "story": sc.story, "unit": sc.unit,
            "freq": sc.freq, "horizon": sc.horizon, "horizon_label": sc.horizon_label, "view": sc.view,
            "series_label": sc.series_label, "series": [{"id": k, "label": v} for k, v in sc.series.items()],
            "default_series": sc.default_series, **engine.window_params(sc),
            "covariates": [{"key": c.key, "label": c.label, "hint": c.hint, "binary": c.binary} for c in sc.covariates],
            "caveats": list(sc.caveats), "size_mb": sc.size_mb, "ready": engine.is_ready(sc),
            "source": {"repo": FEV_REPO, "revision": FEV_REVISION, "name": sc.name, "freq": sc.freq,
                       "license": "other (declared on the repository card; the original data come from public sources, check their terms)"},
            "impact_windows": sc.impact_windows, "impact_series": len(sc.impact_series),
        } for sc in SCENARIOS.values()])

    @app.post("/api/showcase/{scenario_id}/prepare")
    def showcase_prepare(scenario_id: str):
        from tsfm_lab.showcase import engine

        sc = engine.scenario(scenario_id)
        return ok(jobs.submit("download", f"Download {sc.title} data ({sc.size_mb} MB)", lambda j: worker.call(
            "showcase_prepare", {"scenario": scenario_id}, lambda d, t, m: j.update(d, t, m), j.cancel.is_set), link=f"showcase?tab={scenario_id}").to_dict(), 202)

    @app.post("/api/showcase/forecast")
    def showcase_forecast(body: ShowcaseBody):
        from tsfm_lab.showcase import engine

        engine.scenario(body.scenario)
        registry.info(body.model)
        payload = body.model_dump()
        payload["options"] = clean_options(body.options)
        if not compute_lock.acquire(blocking=False):
            raise HTTPException(409, "a job is running on this machine; wait for it to finish")
        try:
            return ok(worker.call("showcase_forecast", payload))
        finally:
            compute_lock.release()

    @app.post("/api/showcase/impact")
    def showcase_impact(body: ShowcaseImpactBody):
        from tsfm_lab.showcase import engine

        sc = engine.scenario(body.scenario)
        registry.info(body.model)
        payload = body.model_dump()
        payload["options"] = clean_options(body.options)
        return ok(jobs.submit("showcase", f"Measuring the effect of covariates · {sc.title}", lambda j: worker.call(
            "showcase_impact", payload, lambda d, t, m: j.update(d, t, m), j.cancel.is_set), link=f"showcase?tab={sc.id}").to_dict(), 202)

    @app.get("/api/compare")
    def compare(runs: str):
        ids = [r for r in runs.split(",") if r][:6]
        if len(ids) < 2:
            raise HTTPException(422, "choose at least two runs")
        metas = [store.meta(r) for r in ids]
        out = []
        for m in metas:
            out.append({"id": m["id"], "label": m.get("label"), "created": m["created"], "dataset": m["dataset"],
                        "config": m["config"], "request": m.get("request"), "items": len(m["items"]["series"]),
                        "models": {mid: {"name": mm.get("name", mid), "kind": mm.get("kind"), "device": mm["device"], "summary": mm["summary"],
                                         "warmup_ms": mm.get("warmup_ms"), "compute_ms": sum(mm["batch_ms"]),
                                         "peak_vram_mb": mm.get("peak_vram_mb"), "load": mm.get("load"), "warnings": len(mm["warnings"])}
                                   for mid, mm in m["models"].items()}})
        first = metas[0]
        same_items = all(m["items"]["series"] == first["items"]["series"] and m["items"]["origin"] == first["items"]["origin"]
                         for m in metas)
        return ok({"runs": out, "comparable": {"same_dataset": all(m["dataset"]["id"] == first["dataset"]["id"] for m in metas),
                                                "same_horizon": all(m["config"]["horizon"] == first["config"]["horizon"] for m in metas),
                                                "same_items": same_items}})

    @app.get("/api/compare/paired")
    def compare_paired(a: str, b: str, metric: str = "mase"):
        """Per-item comparison of two (run:model) entries over identical series/windows."""
        def load(ref: str):
            run_id, _, model = ref.partition(":")
            meta, arrays = store.meta(run_id), store.arrays(run_id)
            key = f"{model}/item/{metric}"
            if key not in arrays:
                raise HTTPException(404, f"{ref} has no metric '{metric}'")
            return meta, arrays[key].mean(axis=1)
        ma, xa = load(a)
        mb, xb = load(b)
        if ma["items"]["series"] != mb["items"]["series"] or ma["items"]["origin"] != mb["items"]["origin"]:
            raise HTTPException(422, "these runs did not forecast the same series and windows: a paired comparison needs identical items")
        ok_mask = np.isfinite(xa) & np.isfinite(xb)
        wins_b = float((xb[ok_mask] < xa[ok_mask]).mean()) if ok_mask.any() else None
        return ok({"metric": metric, "a": xa, "b": xb, "series": ma["items"]["series"], "window": ma["items"]["window"],
                   "n": int(ok_mask.sum()), "b_better_share": wins_b, "skill_b_vs_a": M.skill_score(xb, xa)})

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
    atexit.register(worker.stop)  # the worker is a daemon process, this just makes the exit clean
    app.state.worker = worker
    web = root / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
