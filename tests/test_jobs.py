"""The job queue: what a user sees when stopping, and what the interface can rely on."""

import threading
import time

from tsfm_lab.api.jobs import JobManager


def test_stopping_a_job_that_is_still_queued_removes_it_at_once() -> None:
    manager = JobManager(threading.Lock())
    release, ran = threading.Event(), []
    first = manager.submit("benchmark", "long one", lambda j: (release.wait(10), {})[1])
    queued = manager.submit("download", "waiting one", lambda j: ran.append(True) or {}, link="datasets")
    time.sleep(0.2)
    assert first.status == "running" and queued.status == "queued" and queued.to_dict()["link"] == "datasets"
    manager.cancel(queued.id)
    assert queued.status == "cancelled" and queued.finished is not None  # no waiting for the long job to end
    release.set()
    time.sleep(0.3)
    assert first.status == "done" and queued.status == "cancelled" and ran == []  # and it never runs


def test_stopping_the_running_job_is_left_to_the_job() -> None:
    manager = JobManager(threading.Lock())
    started = threading.Event()

    def work(job):
        started.set()
        job.cancel.wait(5)
        return {}

    job = manager.submit("experiment", "x", work)
    assert started.wait(2)
    manager.cancel(job.id)
    assert job.cancel.is_set() and job.status in ("running", "done")  # the work function decides when it stops


def test_every_job_says_which_page_and_item_shows_its_outcome(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from tsfm_lab.api.app import create_app

    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    monkeypatch.setattr(app.state.worker, "call", lambda *a, **k: {})  # the work itself is not under test
    client = TestClient(app)
    try:
        def link(response):
            assert response.status_code == 202, response.text
            return response.json()["link"]

        assert link(client.post("/api/models/timestral-5m/download")) == "models?focus=timestral-5m"
        assert link(client.post("/api/catalog/ETT_1H/download")) == "datasets?focus=ETT_1H"
        assert link(client.post("/api/showcase/retail/prepare")) == "showcase?tab=retail"
        assert link(client.post("/api/showcase/impact", json={"scenario": "retail", "model": "naive", "device": "cpu"})) == "showcase?tab=retail"
        assert link(client.post("/api/jobs/benchmark", json={"model": "naive", "devices": ["cpu"], "repeats": 1})) == "system?focus=benchmark"
        spec = {"source": "synthetic", "id": "trend", "params": {"n_series": 2, "length": 200}}
        assert link(client.post("/api/jobs/experiment", json={"model": "naive", "dataset": spec, "horizon": 12, "windows": 1, "baselines": []})) is None  # the UI picks the run
    finally:
        app.state.worker.stop()
