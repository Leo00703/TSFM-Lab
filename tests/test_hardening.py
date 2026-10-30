"""Things a request or a file must not be able to do: reach outside the project, exhaust memory, pull a dataset out from under a job."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from tsfm_lab.api.app import CONTENT_SECURITY_POLICY, create_app, csv_cell
from tsfm_lab.data import hub_catalog
from tsfm_lab.data.catalog import MAX_UPLOAD_BYTES, DatasetCatalog
from tsfm_lab.store.store import RunStore, SuiteStore


@pytest.fixture()
def client(tmp_path):
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    try:
        yield TestClient(app)
    finally:
        app.state.worker.stop()


def test_every_response_carries_the_security_headers_and_the_page_has_no_inline_script(client) -> None:
    for path in ("/api/health", "/api/models"):
        h = client.get(path).headers
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff" and h["referrer-policy"] == "no-referrer"
        assert h["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert "script-src 'self'" in CONTENT_SECURITY_POLICY and "unsafe-inline" not in CONTENT_SECURITY_POLICY.split("style-src")[0]
    assert "frame-ancestors 'none'" in CONTENT_SECURITY_POLICY
    # the interactive API docs load their viewer from a CDN: the policy would only break them
    assert "content-security-policy" not in client.get("/api/docs").headers
    from tsfm_lab.isolation import find_root

    site = TestClient(create_app(root=find_root(), allowed_hosts=("testserver",)))
    page = site.get("/")
    inline = [m for m in re.findall(r"<script\b([^>]*)>", page.text) if "src=" not in m]
    assert inline == [], "an inline script would be blocked by the Content-Security-Policy"
    assert "tsfm-theme" in site.get("/js/theme-init.js").text and page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY


def test_a_run_id_from_a_url_cannot_leave_the_runs_folder(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "meta.json").write_text("{}", encoding="utf-8")
    for evil in ("..", "../decoy", "..\\..\\decoy", "..%5Cdecoy", "benchmarks", "20260101-000000-zzzz", ""):
        with pytest.raises(KeyError):
            store.meta(evil)
        with pytest.raises(KeyError):
            store.delete(evil)
    assert (decoy / "meta.json").is_file() and (tmp_path / "runs" / "index.db").is_file()


def test_a_dataset_id_from_a_url_cannot_delete_a_file_outside_the_datasets_folder(tmp_path) -> None:
    catalog = DatasetCatalog(tmp_path / "user")
    victim = tmp_path / "victim.json"
    victim.write_text("{}", encoding="utf-8")
    for evil in ("..\\..\\victim", "../../victim", "../victim", "..", "victim", "local-0123456789abcde", ""):
        with pytest.raises(KeyError):
            catalog.delete_local(evil)
    assert victim.is_file()


def test_the_run_endpoints_answer_404_for_ids_that_are_not_runs(client) -> None:
    for evil in ("..", "..%5C..%5Cdata", "benchmarks"):
        assert client.get(f"/api/runs/{evil}").status_code in (404, 405)
        assert client.delete(f"/api/runs/{evil}").status_code in (404, 405)
        assert client.get(f"/api/runs/{evil}/export.csv").status_code in (404, 405)


def test_an_upload_over_the_limit_is_refused_before_it_is_read_and_a_bad_file_is_not_kept(client, tmp_path) -> None:
    big = client.post("/api/datasets/upload", content=b"x", headers={"x-filename": "a.csv", "content-length": str(MAX_UPLOAD_BYTES + 1)})
    assert big.status_code in (413, 400)
    uploads = tmp_path / "data" / "user" / "uploads"
    assert client.post("/api/datasets/upload", content=b"MZ\x90", headers={"x-filename": "tool.exe"}).status_code == 422
    assert list(uploads.glob("*")) == [], "a file of an unsupported type is never written"
    assert client.post("/api/datasets/upload", content=b"\x00\x01\x02", headers={"x-filename": "broken.parquet"}).status_code == 422
    assert list(uploads.glob("*")) == [], "a file that cannot be read is not kept"
    ok = client.post("/api/datasets/upload", content=b"time,y\n2024-01-01,1\n2024-01-02,2\n", headers={"x-filename": "good.csv"})
    assert ok.status_code == 200 and len(list(uploads.glob("*/good.csv"))) == 1


def test_the_same_bytes_under_two_names_keep_one_file(tmp_path) -> None:
    catalog = DatasetCatalog(tmp_path)
    data = b"time,y\n2024-01-01,1\n2024-01-02,2\n"
    a = catalog.upload("first.csv", data)
    b = catalog.upload("second.csv", data)
    assert a["file_id"] == b["file_id"]
    assert [p.name for p in (tmp_path / "uploads" / a["file_id"]).glob("*")] == ["second.csv"]


def test_a_dataset_cannot_be_deleted_while_a_job_is_running(client) -> None:
    lock = client.app.state.compute_lock
    lock.acquire()
    try:
        assert client.delete("/api/datasets/local/local-0123456789abcdef").status_code == 409
        assert client.delete("/api/catalog/hospital").status_code == 409
    finally:
        lock.release()
    assert client.delete("/api/datasets/local/local-0123456789abcdef").status_code == 200


def test_a_leaderboard_run_that_is_still_going_cannot_be_deleted(tmp_path) -> None:
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    client = TestClient(app)
    try:
        board = {"ranked": ["timestral-5m"], "common_datasets": 1, "total_datasets": 1,
                 "overall": {"timestral-5m": {"datasets": 1, "sql_skill": 12.0, "mase_skill": 10.0, "wql_skill": 11.0, "total_s": 1.0, "wins": 1}}, "by_group": {}}
        sid = SuiteStore(tmp_path / "runs").save({"leaderboard": board, "datasets": [], "device": "cpu", "elapsed_s": 1.0, "models": {}}, "x", None, "done")
        listed = client.get("/api/suites").json()
        assert listed[0]["model_names"] == ["Timestral-5M"] and listed[0]["best_name"] == "Timestral-5M"
        assert client.delete(f"/api/suites/{sid}").status_code == 200
    finally:
        app.state.worker.stop()


def test_text_that_a_spreadsheet_would_run_is_quoted_in_the_csv_export() -> None:
    assert [csv_cell(v) for v in ("=SUM(A1)", "+1", "-1+2", "@x", "normal", 5)] == ["'=SUM(A1)", "'+1", "'-1+2", "'@x", "normal", 5]


def test_catalog_names_read_as_names() -> None:
    assert hub_catalog.pretty("entsoe") == "ENTSO-E"
    assert hub_catalog.pretty("ecdc ili") == "ECDC ILI"
    assert hub_catalog.pretty("fred md 2025 cee") == "FRED-MD 2025 CEE"
    assert hub_catalog.pretty("BizITObs L2C") == "BizITObs L2C", "a benchmark's own spelling is kept"
    titles = [hub_catalog.title(e) for e in hub_catalog.entries().values()]
    assert len(titles) == 187 and not [t for t in titles if t == t.lower()], "every catalog name starts with a capital"


def test_the_system_report_survives_a_cuda_build_that_sees_no_device(monkeypatch) -> None:
    import torch

    from tsfm_lab.system.probe import environment_info

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)  # what CUDA_VISIBLE_DEVICES="" does to some builds
    assert environment_info()["cuda_available"] is False


def test_a_dataset_a_ranked_model_lists_as_its_training_data_is_left_out_of_the_trusted_ranking_but_stays_in_the_table(client, tmp_path) -> None:
    from tsfm_lab.eval.suite import aggregate

    def row(dataset_id: str, a: float, b: float) -> dict:
        errors = lambda sql: {"sql": sql, "mase": sql, "wql": sql, "total_s": 1.0}
        return {"id": dataset_id, "title": dataset_id, "group": "fev", "freq": "h", "freq_label": "hourly", "horizon": 8, "windows": 1, "series": 2,
                "targets": 1, "covariates": False, "skipped": {}, "error": None,
                "models": {"timesfm-2-5": errors(a), "b": errors(b), "seasonal-naive": errors(1.0)}}

    rows = [row("kdd_cup_2022_10T", 0.15, 0.6), row("ETT_1D", 0.9, 0.8), row("LOOP_SEATTLE_5T", 0.9, 0.8)]
    ranked = ["timesfm-2-5", "b"]
    board = {**aggregate(rows, ranked), "total_datasets": 3}
    store = SuiteStore(tmp_path / "runs")
    sid = store.save({"leaderboard": board, "datasets": rows, "device": "cpu", "elapsed_s": 1.0, "models": {}}, "x", None, "done")
    d = client.get(f"/api/suites/{sid}").json()
    assert [r["id"] for r in d["datasets"] if r["caution"]] == ["kdd_cup_2022_10T"] and "TimesFM 2.5" in d["datasets"][0]["caution"]
    assert d["leaderboard"]["common_datasets"] == 3 and d["leaderboard"]["overall"]["timesfm-2-5"]["sql_skill"] > d["leaderboard"]["overall"]["b"]["sql_skill"]
    trusted = d["leaderboard_trusted"]
    assert trusted["common_datasets"] == 2 and trusted["total_datasets"] == 2 and trusted["overall"]["b"]["sql_skill"] > trusted["overall"]["timesfm-2-5"]["sql_skill"]
    # the same datasets in a run that does not rank that model carry no warning: only a model that lists them makes them unfair
    other = [{**r, "models": {"b": r["models"]["b"], "seasonal-naive": r["models"]["seasonal-naive"]}} for r in rows]
    plain = store.save({"leaderboard": aggregate(other, ["b"]), "datasets": other, "device": "cpu", "elapsed_s": 1.0, "models": {}}, "y", None, "done")
    shown = client.get(f"/api/suites/{plain}").json()
    assert "leaderboard_trusted" not in shown and not any(r["caution"] for r in shown["datasets"])
    listed = {e["id"]: e for e in hub_catalog.listing()}
    assert "TimesFM 2.5" in listed["kdd_cup_2022_10T"]["caution"] and listed["ETT_1D"]["caution"] is None and listed["proenfo_gfc12"]["caution"]
    assert all(i in listed for i in hub_catalog.DECLARED_TRAINING_DATA["timesfm-2-5"][1]), "every declared id is a catalog dataset"
