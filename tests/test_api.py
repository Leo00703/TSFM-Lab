import datetime as dt
import time

import pytest
from fastapi.testclient import TestClient

from tsfm_lab.api.app import create_app

SPEC = {"source": "synthetic", "id": "multi_seasonality", "params": {"n_series": 3, "length": 300}}


@pytest.fixture()
def client(tmp_path):
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    yield TestClient(app)
    app.state.worker.stop()


def wait(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_server_only_answers_to_localhost(tmp_path) -> None:
    assert TestClient(create_app(root=tmp_path)).get("/api/health").status_code == 403  # host "testserver"
    local = TestClient(create_app(root=tmp_path), base_url="http://127.0.0.1:8765")
    assert local.get("/api/health").status_code == 200
    assert TestClient(create_app(root=tmp_path), base_url="http://localhost:1").get("/api/health").status_code == 200


def test_cross_origin_writes_are_rejected(client) -> None:
    assert client.post("/api/forecast", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/forecast", json={}, headers={"Origin": "http://testserver"}).status_code == 422


def test_models_datasets_and_system(client) -> None:
    models = {m["id"]: m for m in client.get("/api/models").json()}
    assert {"timestral-5m", "timestral-209m", "naive", "seasonal-naive", "drift"} <= set(models)
    t = models["timestral-5m"]
    assert t["capabilities"]["max_context"] == 32768 and t["cache"]["downloaded"] and t["details"]["num_heads"]
    assert len(t["capabilities"]["quantile_levels"]) == 21 and t["notes"]
    datasets = client.get("/api/datasets").json()
    assert any(d["id"] == "flat_price_covariate" for d in datasets) and len(datasets) >= 16
    system = client.get("/api/system").json()
    assert system["environment"]["python"] and system["devices"][0]["id"] == "cpu"
    disk = client.get("/api/system/disk").json()
    assert {d["label"] for d in disk} >= {".venv", "runs and benchmarks (runs/)"}


def test_invalid_inputs_are_rejected_cleanly(client) -> None:
    assert client.post("/api/datasets/inspect", json={"source": "synthetic", "id": "nope"}).status_code == 404
    bad = {"source": "synthetic", "id": "trend", "params": {"length": 5}}
    assert client.post("/api/datasets/inspect", json=bad).status_code == 422
    bad = {"source": "synthetic", "id": "trend", "params": {"__import__": 1}}
    assert client.post("/api/datasets/inspect", json=bad).status_code == 422
    assert client.post("/api/jobs/experiment", json={"model": "x", "dataset": SPEC}).status_code == 404
    assert client.post("/api/forecast", json={"dataset": SPEC, "model": "naive", "options": {"evil": 1}}).status_code == 422
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.post("/api/jobs/benchmark", json={"model": "naive", "devices": ["cuda:7"]}).status_code == 422


def test_dataset_series_and_interactive_forecast(client) -> None:
    info = client.post("/api/datasets/inspect", json=SPEC).json()
    assert info["n_series"] == 3 and info["series"][0]["length"] == 300
    s = client.post("/api/datasets/series?index=1&max_points=100", json=SPEC).json()
    assert len(s["x"]) <= 100 and s["downsampled"] and len(s["target"][0]) == len(s["x"])
    body = {"dataset": SPEC, "model": "seasonal-naive", "baselines": ["naive"], "horizon": 24, "origin": 250}
    out = client.post("/api/forecast", json=body).json()
    assert set(out["models"]) == {"seasonal-naive", "naive"} and out["origin"] == 250
    m = out["models"]["seasonal-naive"]
    assert len(m["quantiles"][0]) == 24 and len(m["levels"]) == 21 and "mase" in m["metrics"]
    assert len(out["x_future"]) == 24 and out["x_future"][0] > out["x_history"][-1]


def test_forecast_past_the_end_has_no_actuals_and_covariates_must_cover_horizon(client) -> None:
    body = {"dataset": SPEC, "model": "naive", "baselines": [], "horizon": 24, "origin": 300}
    out = client.post("/api/forecast", json=body).json()
    assert all(v is None for v in out["actual"][0]) and "metrics" not in out["models"]["naive"]
    cov = {"source": "synthetic", "id": "covariate_response", "params": {"n_series": 2, "length": 200}}
    r = client.post("/api/forecast", json={"dataset": cov, "model": "naive", "origin": 200, "horizon": 10})
    assert r.status_code == 422 and "known covariates" in r.json()["detail"]


def test_experiment_job_to_run_analysis_and_export(client) -> None:
    body = {"model": "drift", "dataset": SPEC, "horizon": 24, "windows": 2, "baselines": ["naive", "seasonal-naive"]}
    done = wait(client, client.post("/api/jobs/experiment", json=body).json()["id"])
    assert done["status"] == "done", done
    run_id = done["result"]["run_id"]
    assert client.get("/api/runs").json()[0]["id"] == run_id
    meta = client.get(f"/api/runs/{run_id}").json()
    assert set(meta["models"]) == {"drift", "naive", "seasonal-naive"}
    items = client.get(f"/api/runs/{run_id}/items").json()
    assert len(items) == 6 and "mase" in items[0]["metrics"]["drift"]
    item = client.get(f"/api/runs/{run_id}/item/0").json()
    assert len(item["models"]["naive"]["quantiles"][0]) == 24 and item["actual"]
    assert client.get(f"/api/runs/{run_id}/item/999").status_code == 404
    an = client.get(f"/api/runs/{run_id}/analysis?model=naive").json()
    assert an["scatter"]["n"] > 0 and len(an["residual_hist"]["counts"]) == 41 and "naive" in an["distributions"]
    csv = client.get(f"/api/runs/{run_id}/export.csv")
    assert csv.headers["content-type"].startswith("text/csv") and csv.text.count("\n") == 1 + 3 * 6
    assert client.delete(f"/api/runs/{run_id}").status_code == 200 and client.get("/api/runs").json() == []


def test_job_cancellation_and_unknown_job(client) -> None:
    assert client.get("/api/jobs/nope").status_code == 404
    big = {**SPEC, "params": {"n_series": 200, "length": 400}}
    job = client.post("/api/jobs/experiment", json={"model": "naive", "dataset": big, "horizon": 24, "windows": 5}).json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert wait(client, job["id"])["status"] in ("cancelled", "done")


def test_upload_register_and_use_a_local_file(client) -> None:
    rows = ["date,store,sales,price"]
    for store in ("a", "b"):
        for i in range(120):
            rows.append(f"{dt.date(2024, 1, 1) + dt.timedelta(days=i)},{store},{10 + i % 7},2.5")
    up = client.post("/api/datasets/upload", content="\n".join(rows).encode(),
                     headers={"X-Filename": "../../evil name.csv"}).json()
    assert up["filename"] == "evil_name.csv" and up["rows"] == 240 and up["guessed_mapping"]["timestamp"] == "date"
    mapping = {"timestamp": "date", "id": "store", "targets": ["sales"], "known_covariates": ["price"]}
    rec = client.post("/api/datasets/register", json={"file_id": up["file_id"], "mapping": mapping, "name": "Shops"}).json()
    assert rec["summary"]["n_series"] == 2 and rec["summary"]["freq"] == "D"
    spec = {"source": "local", "id": rec["id"]}
    out = client.post("/api/forecast", json={"dataset": spec, "model": "seasonal-naive", "origin": 100, "horizon": 14}).json()
    assert out["models"]["seasonal-naive"]["metrics"]["mase"] >= 0
    assert any(d["id"] == rec["id"] for d in client.get("/api/datasets").json())
    assert client.delete(f"/api/datasets/local/{rec['id']}").status_code == 200


def test_upload_rejects_unsupported_files_and_bad_ids(client) -> None:
    assert client.post("/api/datasets/upload", content=b"a,b\n1,2", headers={"X-Filename": "x.exe"}).status_code == 422
    assert client.post("/api/datasets/register", json={"file_id": "../../etc", "mapping": {}}).status_code == 422
    assert client.post("/api/datasets/inspect", json={"source": "local", "id": "local-../x"}).status_code == 404


def test_benchmark_job_on_a_baseline(client) -> None:
    body = {"model": "naive", "devices": ["cpu"], "repeats": 2, "scenarios": ["mistral", "horizon"]}
    done = wait(client, client.post("/api/jobs/benchmark", json=body).json()["id"])
    assert done["status"] == "done", done
    bench = client.get(f"/api/benchmarks/{done['result']['benchmark_id']}").json()
    cells = bench["devices"]["cpu"]["cells"]
    assert cells[0]["scenario"] == "mistral" and cells[0]["median_ms"] > 0 and len(cells) == 5
    assert client.get("/api/benchmarks").json()[0]["model"] == "naive"


def test_the_web_ui_is_served_with_its_vendored_libraries() -> None:
    from tsfm_lab.isolation import find_root

    client = TestClient(create_app(root=find_root(), allowed_hosts=("testserver",)))
    page = client.get("/")
    assert page.status_code == 200 and 'name="viewport"' in page.text and "tsfm-theme" in page.text
    for path in ("/js/main.js", "/js/pages/runs.js", "/vendor/htm-preact.js", "/vendor/echarts.min.js", "/fonts/Inter-VariableFont.woff2",
                 "/api/health"):
        assert client.get(path).status_code == 200, path
    assert client.get("/api/brand").json().keys() >= {"fonts", "logo"}


def test_worker_crash_does_not_take_the_server_down(tmp_path, monkeypatch) -> None:
    from tsfm_lab.api.worker import WorkerCrashed

    monkeypatch.setenv("TSFM_LAB_TEST_OPS", "1")
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    client = TestClient(app)
    try:
        body = {"dataset": SPEC, "model": "naive", "baselines": [], "horizon": 12, "origin": 200}
        assert client.post("/api/forecast", json=body).status_code == 200
        first_pid = app.state.worker._proc.pid
        with pytest.raises(WorkerCrashed, match="exit code 13"):
            app.state.worker.call("crash", {})
        assert client.get("/api/health").status_code == 200  # the server itself is fine
        assert client.get("/api/system").json()["worker_alive"] is False
        again = client.post("/api/forecast", json=body)  # a fresh worker starts on demand
        assert again.status_code == 200 and app.state.worker._proc.pid != first_pid
    finally:
        app.state.worker.stop()
