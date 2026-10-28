"""The leaderboard: scoring and aggregation, a full run with baselines only, the store and the API guards."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tsfm_lab.api.app import create_app
from tsfm_lab.data.schema import Dataset, Series
from tsfm_lab.eval import suite
from tsfm_lab.models.registry import build_default_registry
from tsfm_lab.store.store import SuiteStore


def test_skill_is_the_geometric_mean_of_clipped_error_ratios() -> None:
    assert suite.skill([0.5, 0.5]) == pytest.approx(50.0) and suite.skill([2.0, 0.5]) == pytest.approx(0.0)
    assert suite.skill([]) is None
    assert suite.clipped_ratio(1.0, 0.0) is None and suite.clipped_ratio(float("nan"), 1.0) is None
    assert suite.clipped_ratio(None, 1.0) is None and suite.clipped_ratio(1.0, None) is None  # what a saved run holds for an undefined error
    assert suite.clipped_ratio(1e-9, 1.0) == 0.01 and suite.clipped_ratio(1e9, 1.0) == 100.0  # one dataset cannot dominate


def _row(id_, group, **models):
    return {"id": id_, "group": group, "models": {m: {"sql": v, "mase": v, "wql": v, "total_s": 1.0} for m, v in models.items()}}


def test_only_datasets_every_ranked_model_completed_are_counted() -> None:
    rows = [_row("a", "fev", **{"seasonal-naive": 1.0, "x": 0.5, "y": 0.8}), _row("b", "gift", **{"seasonal-naive": 1.0, "x": 0.5, "y": 0.8}),
            _row("c", "gift", **{"seasonal-naive": 1.0, "x": 0.1})]  # y could not run on c (multivariate): c is left out of the ranking
    board = suite.aggregate(rows, ["x", "y"])
    assert board["common_datasets"] == 2 and board["total_datasets"] == 3
    assert board["overall"]["x"]["sql_skill"] == pytest.approx(50.0) and board["overall"]["y"]["sql_skill"] == pytest.approx(20.0)
    assert board["overall"]["x"]["wins"] == 2 and board["overall"]["y"]["wins"] == 0 and set(board["by_group"]) == {"fev", "gift"}
    assert board["by_group"]["gift"]["x"]["datasets"] == 1



def test_a_dataset_the_reference_cannot_score_is_not_counted_and_old_runs_are_recounted(tmp_path) -> None:
    flat = _row("night", "monash", **{"seasonal-naive": float("nan"), "x": float("nan")})  # constant series: no error scale, nothing to score
    empty = _row("none", "monash", **{"seasonal-naive": None, "x": None})
    rows = [_row("a", "fev", **{"seasonal-naive": 1.0, "x": 0.5}), flat, empty]
    board = suite.aggregate(rows, ["x"])
    assert board["common_datasets"] == 1 and board["overall"]["x"]["sql_skill"] == pytest.approx(50.0) and "monash" not in board["by_group"]
    assert suite.scorable(rows[0]) and not suite.scorable(flat) and not suite.scorable(empty)
    # a run saved when every dataset with all the models counted still shows today's count when it is opened or listed
    old = {**suite.aggregate(rows[:1], ["x"]), "common_datasets": 3, "total_datasets": 3}
    store = SuiteStore(tmp_path)
    sid = store.save({"config": {}, "device": "cpu", "models": {}, "datasets": rows, "elapsed_s": 1.0, "leaderboard": old}, "old", status="done")
    assert store.get(sid)["leaderboard"]["common_datasets"] == 1 and store.get(sid)["leaderboard"]["total_datasets"] == 3
    assert store.list()[0]["common"] == 1 and store.list()[0]["datasets"] == 3


def test_a_saved_run_with_an_undefined_metric_on_a_scorable_dataset_still_opens(tmp_path) -> None:
    row = _row("a", "fev", **{"seasonal-naive": 1.0, "x": 0.5})
    row["models"]["x"]["mase"] = None  # saved as null: the sql error is defined, the mase one is not
    row["models"]["seasonal-naive"]["wql"] = None
    row["models"]["y"] = {"sql": None, "mase": None, "wql": None, "total_s": 1.0}  # a model that failed on this dataset
    store = SuiteStore(tmp_path)
    sid = store.save({"config": {}, "device": "cpu", "models": {}, "datasets": [row], "elapsed_s": 1.0, "leaderboard": suite.aggregate([_row("a", "fev", **{"seasonal-naive": 1.0, "x": 0.5, "y": 0.9})], ["x", "y"])}, "nulls")
    board = store.get(sid)["leaderboard"]
    assert board["common_datasets"] == 1 and board["overall"]["x"]["sql_skill"] == pytest.approx(50.0) and board["overall"]["x"]["mase_skill"] is None
    assert board["overall"]["x"]["wins"] == 1 and board["overall"]["y"]["wins"] == 0 and board["overall"]["y"]["sql_skill"] is None
    assert store.list()[0]["common"] == 1


def _dataset(n_series: int, length: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    ts = np.arange(np.datetime64("2024-01-01", "ms"), np.datetime64("2024-01-01", "ms") + length * np.timedelta64(3_600_000, "ms"),
                   np.timedelta64(3_600_000, "ms"))
    series = [Series(f"s{i}", (10 + np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=length))[None, :], ts, None, None) for i in range(n_series)]
    return Dataset(id=f"toy{seed}", name=f"toy{seed}", series=series, freq="h", seasonal_period=24, target_names=["y"], source="hub",
                   license="test", description="", tags=[], extra={})


def _entry(i: str, source: str = "fev") -> dict:
    return {"id": i, "title": i, "source": source, "pandas_freq": "h", "task": {"horizon": 12, "num_windows": 5, "seasonality": 24}}


def test_a_run_with_baselines_only_ranks_them_on_identical_windows_and_survives_a_short_dataset() -> None:
    data = {"fev1": _dataset(4, 200, 1), "gift1": _dataset(3, 300, 2), "tiny": _dataset(2, 10, 3)}
    seen = []

    def load(e, n):
        seen.append((e["id"], n))
        return data[e["id"]]

    entries = [_entry("fev1"), _entry("gift1", "gift"), _entry("tiny")]
    out = suite.run_suite(entries, ["drift", "naive"], "cpu", suite.SuiteConfig(max_series=3, max_windows=2, max_context=64), build_default_registry(), load)
    assert seen == [("fev1", 3), ("gift1", 3), ("tiny", 3)]
    a, _, tiny = out["datasets"]
    assert a["windows"] == 2 and a["horizon"] == 12 and a["items"] == 3 * 2 and set(a["models"]) == {"drift", "naive", "seasonal-naive"}
    assert tiny["error"] and not tiny["models"]  # too short for the task: recorded, the run goes on
    board = out["leaderboard"]
    assert board["common_datasets"] == 2 and board["total_datasets"] == 3 and board["ranked"] == ["drift", "naive"]
    assert math.isfinite(board["overall"]["naive"]["sql_skill"]) and set(board["by_group"]) == {"fev", "gift"}
    assert out["models"]["seasonal-naive"]["kind"] == "baseline" and out["config"]["max_context"] == 64


def test_the_store_lists_saves_and_deletes(tmp_path) -> None:
    store = SuiteStore(tmp_path)
    rows = [_row("a", "fev", **{"seasonal-naive": 1.0, "x": 0.5})]
    result = {"config": {}, "device": "cpu", "models": {}, "datasets": rows, "elapsed_s": 1.0, "leaderboard": suite.aggregate(rows, ["x"])}
    sid = store.save(result, "first")
    (item,) = store.list()
    assert item["id"] == sid and item["label"] == "first" and item["best"] == "x" and item["common"] == 1
    assert store.get(sid)["leaderboard"]["overall"]["x"]["sql_skill"] == pytest.approx(50.0)
    store.delete(sid)
    assert store.list() == []
    with pytest.raises(KeyError):
        store.get(sid)


def test_the_api_refuses_a_run_over_datasets_that_are_unknown_or_not_downloaded(tmp_path, monkeypatch) -> None:
    from tsfm_lab.data import hub_catalog as H

    monkeypatch.setattr(H, "_cache_dir", lambda e: tmp_path / "empty")
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    client = TestClient(app)
    try:
        post = lambda body: client.post("/api/jobs/suite", json={"device": "cpu", **body})
        assert post({"datasets": ["nope"], "models": ["timestral-5m"]}).status_code == 404
        every_model = [m["id"] for m in client.get("/api/models").json() if m["kind"] == "foundation"]
        assert len(every_model) > 8 and post({"datasets": ["nope"], "models": every_model}).status_code == 404  # a run may take every model
        r = post({"datasets": ["hospital"], "models": ["timestral-5m"]})
        assert r.status_code == 422 and "not downloaded" in r.text
        r = post({"datasets": ["hospital"], "models": ["naive"]})
        assert r.status_code in (404, 422) and ("not downloaded" in r.text or "at least one model" in r.text)
        assert client.get("/api/suites").json() == [] or client.get("/api/suites").json().get("data") == []
        assert client.get("/api/suites/missing").status_code == 404
    finally:
        app.state.worker.stop()


def test_the_users_own_datasets_run_through_the_leaderboard_with_one_horizon(tmp_path) -> None:
    from tsfm_lab.data.catalog import DatasetCatalog

    catalog = DatasetCatalog(tmp_path)
    rows = ["date,store,sales"]
    for store in ("a", "b", "c"):
        for i in range(200):
            rows.append(f"{dt.date(2024, 1, 1) + dt.timedelta(days=i)},{store},{10 + i % 7 + (i * i) % 5}")  # not exactly periodic: seasonal naive must make an error to be scored against
    up = catalog.upload("shops.csv", "\n".join(rows).encode())
    rec = catalog.register_local(up["file_id"], {"timestamp": "date", "id": "store", "targets": ["sales"]}, "Shops")
    entries = catalog.suite_entries([rec["id"], rec["id"]], 14, 2)  # the same dataset twice is one
    assert len(entries) == 1 and entries[0]["source"] == "mine" and entries[0]["title"] == "Shops"
    assert entries[0]["task"] == {"horizon": 14, "num_windows": 2} and entries[0]["pandas_freq"] == "D"
    load = lambda e, n: catalog.resolve({"source": "local", "id": e["id"]})
    out = suite.run_suite(entries, ["drift", "naive"], "cpu", suite.SuiteConfig(max_series=3, max_windows=2, max_context=64), build_default_registry(), load)
    row = out["datasets"][0]
    assert row["id"] == rec["id"] and row["group"] == "mine" and row["horizon"] == 14 and row["windows"] == 2 and row["error"] is None
    assert set(out["leaderboard"]["by_group"]) == {"mine"} and out["leaderboard"]["common_datasets"] == 1
    for bad in ("local-0000000000000000", "../x", "hospital"):
        with pytest.raises(KeyError):
            catalog.suite_entries([bad], 14, 2)


def test_the_api_asks_for_a_horizon_and_a_known_dataset_when_the_run_is_on_the_users_own_data(tmp_path) -> None:
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    client = TestClient(app)
    try:
        post = lambda body: client.post("/api/jobs/suite", json={"device": "cpu", "models": ["timestral-5m"], **body})
        assert post({}).status_code == 422 and "at least one dataset" in post({}).text
        r = post({"local_datasets": ["local-0123456789abcdef"]})
        assert r.status_code == 422 and "horizon" in r.text
        r = post({"local_datasets": ["local-0123456789abcdef"], "local_horizon": 14})
        assert r.status_code == 404 and "unknown local dataset" in r.text
        assert post({"local_datasets": ["local-0123456789abcdef"], "local_horizon": 0}).status_code == 422
    finally:
        app.state.worker.stop()


def test_a_run_is_checkpointed_after_every_dataset_and_can_be_cut_short(tmp_path) -> None:
    data = {"a": _dataset(3, 200, 1), "b": _dataset(3, 200, 2), "c": _dataset(3, 200, 3)}
    store = SuiteStore(tmp_path)
    sid, snapshots = store.new_id(), []

    def checkpoint(result):
        store.save(result, "cut", sid, "running")
        snapshots.append(len(store.get(sid)["datasets"]))

    def progress(done, total, message):
        if message == "c":  # the user presses Stop while the third dataset is about to start
            raise RuntimeError("stopped")

    entries = [_entry(i) for i in "abc"]
    with pytest.raises(RuntimeError):
        suite.run_suite(entries, ["naive"], "cpu", suite.SuiteConfig(max_series=3, max_windows=1), build_default_registry(), lambda e, n: data[e["id"]],
                        progress, checkpoint)
    assert snapshots == [1, 2]  # each finished dataset was written as it finished
    saved = store.get(sid)
    assert saved["status"] == "running" and saved["planned"] == 3 and saved["leaderboard"]["common_datasets"] == 2
    store.finish(sid, "stopped")
    assert store.get(sid)["status"] == "stopped" and store.list()[0]["status"] == "stopped" and store.list()[0]["planned"] == 3
    store.save(saved, "cut", sid, "done")
    store.finish(sid, "interrupted")  # a finished run is never downgraded
    assert store.get(sid)["status"] == "done"


def test_the_api_reports_a_run_nobody_is_running_as_interrupted(tmp_path) -> None:
    rows = [_row("a", "fev", **{"seasonal-naive": 1.0, "x": 0.5})]
    result = {"config": {}, "device": "cpu", "models": {}, "datasets": rows, "elapsed_s": 1.0, "planned": 5, "leaderboard": suite.aggregate(rows, ["x"])}
    (tmp_path / "runs").mkdir()
    sid = SuiteStore(tmp_path / "runs").save(result, "orphan", status="running")  # left behind by a server that died
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    client = TestClient(app)
    try:
        listed = client.get("/api/suites").json()
        assert listed[0]["status"] == "interrupted" and listed[0]["planned"] == 5
        assert client.get(f"/api/suites/{sid}").json()["status"] == "interrupted"
    finally:
        app.state.worker.stop()


def test_every_catalog_entry_of_every_benchmark_can_be_run() -> None:
    from tsfm_lab.data import hub_catalog as H

    entries = H.run_entries(list(H.entries()))
    assert len(entries) == 187 and all(isinstance(e["title"], str) and e["title"] and e["task"]["horizon"] and e["source"] for e in entries)
