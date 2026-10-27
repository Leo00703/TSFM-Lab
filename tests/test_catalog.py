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
    assert len(entries) == 100 and len(H._catalog()["entries"]) == len(entries)  # ids are unique
    src = H.source()
    assert len(src["revision"]) == 40 and len(src["tasks_ref"]) == 40 and src["card_license"] == "other"
    labels = [f"{H.title(e)} · {H.freq_label(e['pandas_freq'])}" for e in entries]
    assert [k for k, n in collections.Counter(labels).items() if n > 1] == []  # the page needs distinct names
    for e in entries:
        assert len(e["sha256"]) == 64 and e["size_bytes"] > 0 and e["path"].endswith("train-00000-of-00001.parquet")
        assert e["n_series"] >= 1 and e["length"]["min"] >= 1 and e["length"]["min"] <= e["length"]["max"]
        assert e["task"]["horizon"] >= 1 and e["task"]["seasonality"] >= 1 and e["task"]["num_windows"] >= 1
        assert e["targets"] and all(c in e["columns"] for c in e["targets"] + e["known"] + e["past"] + e["static"])
        assert all(e["columns"][c]["kind"] in ("list-numeric", "list-binary") for c in e["targets"] + e["past"])
        assert all(e["columns"][c]["kind"] != "other" for c in e["known"])
        pd.tseries.frequencies.to_offset(e["pandas_freq"])  # a rule pandas understands
        assert e["checks"]["series_with_time_gaps"] == 0  # the sample was regular on the recorded frequency


def test_frequency_labels_are_plain_language() -> None:
    expect = {"h": "hourly", "1h": "hourly", "15min": "every 15 minutes", "5min": "every 5 minutes", "min": "every minute",
              "D": "daily", "1D": "daily", "W-SUN": "weekly", "MS": "monthly", "1ME": "monthly", "QS-JAN": "quarterly",
              "QE-DEC": "quarterly", "YE-DEC": "yearly", "30min": "every 30 minutes"}
    assert {k: H.freq_label(k) for k in expect} == expect


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch):
    """A one-file Hugging Face cache laid out like the real one, so nothing needs the network."""
    cache = tmp_path / "snapshot"
    monkeypatch.setattr(H, "_cache_dir", lambda: cache)
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
    monkeypatch.setattr(H, "_cache_dir", lambda: tmp_path / "empty")
    e = H.entry("australian_tourism")
    with pytest.raises(FileNotFoundError, match="download it from the Datasets page"):
        H.read(e)


def test_a_download_with_the_wrong_hash_is_discarded(fake_hub, monkeypatch) -> None:
    e = {**fake_hub, "sha256": "0" * 64}
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
    assert len(data["entries"]) == 100 and data["source"]["card_url"].startswith("https://huggingface.co/datasets/")
    assert data["source"]["tasks_url"].endswith("benchmarks/fev_bench/tasks.yaml")
    first = data["entries"][0]
    assert {"id", "title", "freq_label", "n_series", "length", "size_bytes", "downloaded", "checks", "sha256"} <= set(first)
    assert client.post("/api/catalog/does-not-exist/download").status_code == 404
    assert client.delete("/api/catalog/does-not-exist").status_code == 404


@pytest.mark.online
def test_a_real_download_is_verified_and_readable() -> None:
    e = H.entry("us_consumption_1M")  # 133 KB
    path = H.download(e)
    assert H.sha256(path) == e["sha256"]
    ds = H.read(e, max_series=5)
    assert len(ds.series) == 5 and ds.freq == "MS" and ds.series[0].length == e["length"]["min"]
    assert str(ds.series[0].timestamps[0])[:7] == "1959-01"
