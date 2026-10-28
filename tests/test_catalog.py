"""The curated online catalog: its integrity, the reader, the safety checks, and one real (tiny) download."""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tsfm_lab.data import hub_catalog as H
from fastapi.testclient import TestClient

from tsfm_lab.api.app import create_app
from tsfm_lab.data.catalog import DatasetCatalog


def test_every_entry_is_complete_and_consistent() -> None:
    entries = list(H.entries().values())
    assert collections.Counter(e["source"] for e in entries) == {"fev": 100, "gift": 55, "monash": 32}
    assert len(H._catalog()["entries"]) == len(entries)  # ids are unique
    for key, src in H.sources().items():
        assert len(src["revision"]) == 40 and len(src["rules_ref"]) == 40 and src["card_license"] and src["name"] and src["contamination"]
    assert {k: v["card_license"] for k, v in H.sources().items()} == {"fev": "other", "gift": "apache-2.0", "monash": "cc-by-4.0"}
    labels = [(e["source"], H.title(e), H.freq_label(e["pandas_freq"])) for e in entries]
    assert [k for k, n in collections.Counter(labels).items() if n > 1] == []  # the page needs distinct names within a benchmark
    suffix = {"fev-parquet": ".parquet", "gift-arrow": ".arrow", "monash-tsf": ".zip"}
    for e in entries:
        assert e["files"] and all(len(f["sha256"]) == 64 and f["size_bytes"] > 0 and f["path"].endswith(suffix[e["format"]]) for f in e["files"])
        assert e["size_bytes"] == sum(f["size_bytes"] for f in e["files"])
        assert e["n_series"] >= 1 and e["length"]["min"] >= 1 and e["length"]["min"] <= e["length"]["max"]
        assert e["task"]["horizon"] >= 1 and e["task"]["seasonality"] >= 1 and e["task"]["num_windows"] >= 1
        assert e["targets"] and all(c in e["columns"] for c in e["targets"] + e["known"] + e["past"] + e["static"] if e["source"] == "fev")
        assert all(e["columns"][c]["kind"] in ("list-numeric", "list-binary") for c in e["targets"] + e["past"] if e["source"] == "fev")
        assert all(e["columns"][c]["kind"] != "other" for c in e["known"] if e["source"] == "fev")
        pd.tseries.frequencies.to_offset(e["pandas_freq"])  # a rule pandas understands
        assert e["checks"]["series_with_time_gaps"] == 0  # the sample was regular on the recorded frequency
    for e in (x for x in entries if x["source"] == "gift"):
        assert e["checks"]["starts_off_grid"] == e["checks"]["duplicate_ids"] == e["checks"]["past_length_mismatch"] == 0
        assert e["checks"]["sampled_series"] == e["n_series"] and not e["known"] and not e["static"]
    for e in (x for x in entries if x["source"] == "monash"):
        assert e["checks"]["starts_after_day_28"] == e["checks"]["duplicate_names"] == 0 and e["task"]["horizon_source"]


def test_gift_eval_tasks_follow_the_benchmarks_own_rules() -> None:
    def task(i):
        t = H.entry(i)["task"]
        return t["horizon"], t["num_windows"], [(x["term"], x["horizon"], x["num_windows"]) for x in t["terms"]]

    assert task("gift.m4_hourly")[:2] == (48, 1) and task("gift.m4_yearly")[:2] == (6, 1)  # M4: its own horizons, one window
    assert task("gift.hospital")[:2] == (12, 1) and task("gift.ett1.W")[:2] == (8, 2) and task("gift.us_births.M")[:2] == (12, 2)
    # sub-daily data also have a medium and a long term, 10x and 15x the short horizon; daily and slower ones only the short term
    assert task("gift.electricity.H")[2] == [("short", 48, 20), ("medium", 480, 8), ("long", 720, 5)]
    assert [t[0] for t in task("gift.electricity.D")[2]] == ["short"] and task("gift.bizitobs_service")[0] == 60
    assert H.entry("gift.bitbrains_rnd.H")["targets"] == ["variable 1", "variable 2"] and len(H.entry("gift.bitbrains_rnd.H")["past"]) == 5


def test_the_same_data_in_two_benchmarks_is_recognised_but_never_inside_one() -> None:
    assert set(H.entry("hospital")["overlaps"]) == {"gift.hospital", "monash.hospital"}
    assert set(H.entry("gift.solar.W")["overlaps"]) == {"solar_1W", "monash.solar_weekly"}
    assert H.entry("gift.electricity.H")["overlaps"] == []
    for e in H.entries().values():
        assert all(H.entry(o)["source"] != e["source"] and e["id"] in H.entry(o)["overlaps"] for o in e["overlaps"])  # symmetric
    listed = {e["id"]: e for e in H.listing()}["gift.hospital"]
    assert {o["source"] for o in listed["also_in"]} == {"fev", "monash"}


def test_frequency_labels_are_plain_language() -> None:
    expect = {"h": "hourly", "1h": "hourly", "15min": "every 15 minutes", "5min": "every 5 minutes", "min": "every minute",
              "D": "daily", "1D": "daily", "W-SUN": "weekly", "MS": "monthly", "1ME": "monthly", "QS-JAN": "quarterly",
              "QE-DEC": "quarterly", "YE-DEC": "yearly", "30min": "every 30 minutes"}
    assert {k: H.freq_label(k) for k in expect} == expect


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch):
    """A one-file Hugging Face cache laid out like the real one, so nothing needs the network."""
    cache = tmp_path / "snapshot"
    monkeypatch.setattr(H, "_cache_dir", lambda e: cache)
    ts = pd.date_range("2024-01-01", periods=40, freq="D")
    gap = ts.delete([10, 11])  # a series with two missing days
    rows = [("a", ts, np.arange(40.0), np.array([True, False] * 20), [None, "Navidad"] * 20),
            ("b", gap, np.arange(38.0) * 2, np.array([False] * 38), ["None"] * 38),
            ("c", ts, np.ones(40), np.array([True] * 40), ["0"] * 40),
            ("d", ts, np.arange(40.0) + 5, np.array([False, True] * 20), ["Dia"] * 40)]
    table = pa.table({
        "id": [r[0] for r in rows], "timestamp": pa.array([list(r[1].to_pydatetime()) for r in rows], pa.list_(pa.timestamp("ms"))),
        "y": pa.array([list(r[2]) for r in rows], pa.list_(pa.float32())),
        "promo": pa.array([[bool(v) for v in r[3]] for r in rows], pa.list_(pa.bool_())),
        "holiday": pa.array([list(r[4]) for r in rows], pa.list_(pa.string())),
        "shop": ["s1", "s2", "s3", "s4"]})
    path = cache / "toy" / "1D" / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table, path)
    e = {"id": "toy_1D", "config": "toy_1D", "variant": None, "family": "toy", "freq": "1D", "pandas_freq": "D", "path": "toy/1D/train-00000-of-00001.parquet",
         "source": "fev", "repo": "toy/repo", "revision": "0" * 40, "format": "fev-parquet", "n_series": 4, "notes": [], "description": "",
         "files": [{"path": "toy/1D/train-00000-of-00001.parquet", "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
         "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "task": {"horizon": 7, "num_windows": 2, "seasonality": 7},
         "targets": ["y"], "known": ["promo", "holiday"], "past": [], "static": ["shop"],
         "columns": {"id": {"kind": "string"}, "y": {"kind": "list-numeric"}, "promo": {"kind": "list-binary"},
                     "holiday": {"kind": "list-text"}, "shop": {"kind": "string"}}}
    return e


def test_reader_regularises_gaps_turns_text_into_flags_and_samples_reproducibly(fake_hub) -> None:
    e = fake_hub
    ds = H.read(e)
    assert [s.id for s in ds.series] == ["a", "b", "c", "d"] and ds.n_known == 2 and ds.n_past == 0 and ds.freq == "D"
    assert ds.source == "hub" and ds.contamination_risk == "possible" and ds.seasonal_period == 7
    assert ds.known_names == ["promo", "holiday (event: on/off)"]
    a, b = ds.series[0], ds.series[1]
    assert a.known_covariates[0, :4].tolist() == [1, 0, 1, 0]            # booleans become 0/1
    assert a.known_covariates[1, :4].tolist() == [0, 1, 0, 1]            # None -> no event, a name -> event
    assert ds.series[2].known_covariates[1].sum() == 0                   # "0" means no event
    assert b.length == 40 and np.isnan(b.target[0, 10:12]).all() and not np.isnan(b.target[0, 12])  # the gap became NaN
    assert not e["size_bytes"] == 0 and ds.extra["sampled"] is False and ds.extra["static_columns_ignored"] == ["shop"]
    few = H.read(e, max_series=2)
    again = H.read(e, max_series=2)
    assert len(few.series) == 2 and few.extra["sampled"] and [s.id for s in few.series] == [s.id for s in again.series]
    assert "sample of 2 of 4" in few.description


def test_reading_a_dataset_that_is_not_downloaded_says_what_to_do(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(H, "_cache_dir", lambda e: tmp_path / "empty")
    e = H.entry("australian_tourism")
    with pytest.raises(FileNotFoundError, match="download it from the Datasets page"):
        H.read(e)


def test_a_download_with_the_wrong_hash_is_discarded(fake_hub, monkeypatch) -> None:
    e = {**fake_hub, "files": [{**fake_hub["files"][0], "sha256": "0" * 64}]}
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(H.file_path(e)))
    with pytest.raises(ValueError, match="does not match the published SHA-256"):
        H.download(e)
    assert H.file_path(e) is None  # nothing untrusted is left behind


def test_remove_deletes_only_that_file(fake_hub) -> None:
    assert H.file_path(fake_hub) is not None
    assert H.remove(fake_hub) is True and H.file_path(fake_hub) is None and H.remove(fake_hub) is False


def test_the_catalog_layer_resolves_hub_specs_and_rejects_bad_parameters(fake_hub, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(H, "entry", lambda i: fake_hub if i == "toy_1D" else (_ for _ in ()).throw(KeyError(i)))
    cat = DatasetCatalog(tmp_path / "user")
    ds = cat.resolve({"source": "hub", "id": "toy_1D", "params": {"max_series": 3}})
    assert len(ds.series) == 3 and cat.resolve({"source": "hub", "id": "toy_1D", "params": {"max_series": 3}}) is ds  # cached
    for bad in ({"max_series": 0}, {"max_series": 10**6}, {"seed": 1}):
        with pytest.raises(ValueError):
            cat.resolve({"source": "hub", "id": "toy_1D", "params": bad})
    cat.forget_hub("toy_1D")
    assert cat.resolve({"source": "hub", "id": "toy_1D", "params": {"max_series": 3}}) is not ds
    with pytest.raises(KeyError):
        cat.resolve({"source": "hub", "id": "nope"})


def test_the_api_lists_the_catalog_and_refuses_unknown_entries(tmp_path) -> None:
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    client = TestClient(app)
    try:
        _check_api(client)
    finally:
        app.state.worker.stop()


def _check_api(client) -> None:
    data = client.get("/api/catalog").json()
    assert len(data["entries"]) == 187 and set(data["sources"]) == {"fev", "gift", "monash"}
    assert all(s["card_url"].startswith("https://huggingface.co/datasets/") for s in data["sources"].values())
    assert data["sources"]["fev"]["rules_url"].endswith("benchmarks/fev_bench/tasks.yaml")
    assert data["sources"]["gift"]["rules_url"].startswith("https://github.com/SalesforceAIResearch/gift-eval/blob/")
    assert data["sources"]["monash"]["rules_url"].endswith("/monash_tsf.py")
    first = data["entries"][0]
    assert {"id", "source", "title", "freq_label", "n_series", "length", "size_bytes", "downloaded", "checks", "sha256", "files", "also_in"} <= set(first)
    assert client.post("/api/catalog/does-not-exist/download").status_code == 404
    assert client.delete("/api/catalog/does-not-exist").status_code == 404


@pytest.mark.online
def test_a_real_download_is_verified_and_readable() -> None:
    e = H.entry("us_consumption_1M")  # 133 KB
    path = H.download(e)[0]
    assert H.sha256(path) == e["files"][0]["sha256"]
    ds = H.read(e, max_series=5)
    assert len(ds.series) == 5 and ds.freq == "MS" and ds.series[0].length == e["length"]["min"]
    assert str(ds.series[0].timestamps[0])[:7] == "1959-01"


# ---------------------------------------------------------------------------- the other two formats, on tiny stand-in files
def _entry(source: str, fmt: str, files: list[dict], **fields) -> dict:
    return {"id": f"{source}.toy", "config": "toy", "variant": None, "title": "toy", "family": "toy", "source": source, "repo": "toy/repo",
            "revision": "1" * 40, "format": fmt, "files": files, "notes": [], "description": "", "size_bytes": sum(f["size_bytes"] for f in files),
            "known": [], "static": [], "overlaps": [], "columns": {}, **fields}


def _register(path: Path, rel: str) -> dict:
    return {"path": rel, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def gift_hub(tmp_path: Path, monkeypatch):
    """Two Arrow stream shards like GIFT-Eval's: univariate targets with a missing value, past covariates, hourly data."""
    cache = tmp_path / "gift"
    monkeypatch.setattr(H, "_cache_dir", lambda e: cache)
    start = pd.Timestamp("2020-03-01 06:00")
    schema = pa.schema([("item_id", pa.string()), ("start", pa.timestamp("ms")), ("freq", pa.string()),
                        ("target", pa.list_(pa.float32())), ("past_feat_dynamic_real", pa.list_(pa.list_(pa.float32())))])

    def shard(rel: str, rows: list[tuple]) -> dict:
        path = cache / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        batch = pa.RecordBatch.from_pylist([{"item_id": i, "start": start, "freq": "h", "target": y, "past_feat_dynamic_real": p}
                                            for i, y, p in rows], schema=schema)
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
        return _register(path, rel)

    files = [shard("toy/H/data-00000-of-00002.arrow", [("a", [1.0, 2.0, None, 4.0], [[1, 1, 1, 1], [2, 2, 2, 2]]),
                                                       ("b", [5.0, 6.0, 7.0], [[0, 0, 0], [9, 9, 9]]), ("c", [8.0, 8.0], [[3, 3], [4, 4]])]),
             shard("toy/H/data-00001-of-00002.arrow", [("d", [1.0, 3.0, 5.0], [[7, 7, 7], [8, 8, 8]]), ("e", [2.0], [[5], [6]])])]
    return _entry("gift", "gift-arrow", files, pandas_freq="h", n_series=5, task={"horizon": 2, "num_windows": 1, "seasonality": 24},
                  targets=["target"], past=["past variable 1", "past variable 2"], length={"min": 1, "max": 4, "median": 3, "total_points": 13})


def test_gift_eval_shards_are_read_in_order_with_their_time_axis_and_past_covariates(gift_hub) -> None:
    ds = H.read(gift_hub)
    assert [s.id for s in ds.series] == list("abcde") and ds.freq == "h" and ds.seasonal_period == 24 and ds.n_past == 2
    a = ds.series[0]
    assert a.target.shape == (1, 4) and np.isnan(a.target[0, 2]) and a.past_covariates.shape == (2, 4)  # a null became NaN
    assert [str(t)[:13] for t in a.timestamps] == ["2020-03-01T06", "2020-03-01T07", "2020-03-01T08", "2020-03-01T09"]
    assert ds.series[3].past_covariates[1].tolist() == [8, 8, 8] and ds.source == "hub" and ds.contamination_risk == "possible"
    assert ds.extra["source"] == "gift" and "toy/repo / toy @ 1111111111" in ds.description
    few = H.read(gift_hub, max_series=3)
    assert len(few.series) == 3 and few.extra["sampled"] and [s.id for s in few.series] == [s.id for s in H.read(gift_hub, max_series=3).series]


def test_gift_eval_multivariate_rows_stored_as_fixed_size_lists_become_one_series_with_several_targets(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "gift"
    monkeypatch.setattr(H, "_cache_dir", lambda e: cache)
    schema = pa.schema([("item_id", pa.string()), ("start", pa.list_(pa.timestamp("ms"))), ("freq", pa.string()),
                        ("target", pa.list_(pa.list_(pa.float32()), 2))])
    path = cache / "toy" / "data-00000-of-00001.arrow"
    path.parent.mkdir(parents=True)
    row = {"item_id": "x", "start": [pd.Timestamp("2016-07-01")], "freq": "D", "target": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]}
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(pa.RecordBatch.from_pylist([row], schema=schema))
    e = _entry("gift", "gift-arrow", [_register(path, "toy/data-00000-of-00001.arrow")], pandas_freq="D", n_series=1,
               task={"horizon": 1, "num_windows": 1, "seasonality": 7}, targets=["variable 1", "variable 2"], past=[],
               length={"min": 3, "max": 3, "median": 3, "total_points": 3})
    (s,) = H.read(e).series
    assert s.target.tolist() == [[1, 2, 3], [10, 20, 30]] and s.past_covariates is None and str(s.timestamps[-1])[:10] == "2016-07-03"


TSF = """# Dataset of two tiny things
@relation toy
@attribute series_name string
@attribute start_timestamp date
@frequency monthly
@horizon 3
@missing true
@equallength false
@data
T1:1979-01-01 00-00-00:1,2,?,4
T2:1980-11-01 00-00-00:5,6
T3:1981-01-01 00-00-00:7,8,9
"""


@pytest.fixture
def monash_hub(tmp_path: Path, monkeypatch):
    import zipfile

    cache = tmp_path / "monash"
    monkeypatch.setattr(H, "_cache_dir", lambda e: cache)
    path = cache / "data" / "toy_dataset.zip"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("toy_dataset.tsf", TSF)
    return _entry("monash", "monash-tsf", [_register(path, "data/toy_dataset.zip")], pandas_freq="MS", n_series=3,
                  task={"horizon": 3, "num_windows": 1, "seasonality": 12}, targets=["target"], past=[],
                  length={"min": 2, "max": 4, "median": 3, "total_points": 9})


def test_monash_tsf_files_are_parsed_without_executing_anything(monash_hub) -> None:
    from tsfm_lab.data import tsf

    header, lines = tsf.read(H.file_path(monash_hub))
    assert (header.frequency, header.horizon, header.missing, header.equal_length) == ("monthly", 3, True, False)
    assert [n for n, _ in header.attributes] == ["series_name", "start_timestamp"] and "tiny things" in header.description
    rows = list(lines)
    assert [a["series_name"] for a, _ in rows] == ["T1", "T2", "T3"] and np.isnan(rows[0][1][2]) and rows[1][1].tolist() == [5, 6]
    skipped = [v for _, v in tsf.read(H.file_path(monash_hub), wanted=lambda i: i == 1)[1]]
    assert skipped[0] is None and skipped[1].tolist() == [5, 6] and skipped[2] is None  # unwanted lines are not even parsed


def test_monash_series_get_their_own_start_and_calendar_steps(monash_hub) -> None:
    ds = H.read(monash_hub)
    assert [s.id for s in ds.series] == ["T1", "T2", "T3"] and ds.freq == "MS" and ds.seasonal_period == 12
    t1, t2 = ds.series[0], ds.series[1]
    assert t1.target.shape == (1, 4) and np.isnan(t1.target[0, 2]) and [str(t)[:7] for t in t1.timestamps] == ["1979-01", "1979-02", "1979-03", "1979-04"]
    assert [str(t)[:7] for t in t2.timestamps] == ["1980-11", "1980-12"]  # across a year boundary
    assert len(H.read(monash_hub, max_series=1).series) == 1 and H.read(monash_hub, max_series=1).extra["sampled"]


def test_a_catalog_download_of_several_files_is_all_or_nothing(gift_hub, monkeypatch) -> None:
    import huggingface_hub

    e = {**gift_hub, "files": [gift_hub["files"][0], {**gift_hub["files"][1], "sha256": "f" * 64}]}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, **k: str(H._cache_dir(e) / path))
    with pytest.raises(ValueError, match="does not match the published SHA-256"):
        H.download(e)
    assert H.file_paths(e) is None and not any(p.is_file() for p in H._cache_dir(e).rglob("*.arrow"))  # the good shard went too


def test_calendar_and_fixed_steps_of_the_tsf_format_stay_exact() -> None:
    from tsfm_lab.data import tsf

    start = tsf.parse_start("2013-05-04 10-30-00")
    fast = tsf.timestamps(start, "4_seconds", 3)
    assert [str(t)[11:19] for t in fast] == ["10:30:00", "10:30:04", "10:30:08"]
    assert str(tsf.timestamps(start, "yearly", 3)[2])[:10] == "2015-05-04" and str(tsf.timestamps(start, "weekly", 2)[1])[:10] == "2013-05-11"
    assert len(tsf.timestamps(start, "4_seconds", 7_400_000)) == 7_400_000  # a series of millions of observations is instant
    with pytest.raises(ValueError, match="unknown frequency"):
        tsf.timestamps(start, "fortnightly", 2)
