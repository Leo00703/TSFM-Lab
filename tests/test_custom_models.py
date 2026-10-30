"""Custom Hugging Face checkpoints: what is checked before one is added, how it is pinned, and that it behaves like a model."""

from __future__ import annotations

import datetime as dt
import threading
import time
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from tsfm_lab.api.app import create_app
from tsfm_lab.models import cache, custom
from tsfm_lab.models.registry import build_default_registry

SHA = "a" * 40


def fake_api(files=("config.json", "model.safetensors"), size=300_000_000, gated=False, private=False, license_="apache-2.0", params=82_000_000):
    siblings = [NS(rfilename=f, size=size if f == "model.safetensors" else 100) for f in files]
    info = NS(sha=SHA, siblings=siblings, private=private, gated=gated, author="someone", downloads=12, likes=3, tags=[f"license:{license_}"] if license_ else [],
              card_data={"license": license_} if license_ else None, last_modified=dt.datetime(2026, 10, 1), safetensors=NS(total=params))
    return NS(model_info=lambda repo, files_metadata, timeout: info)


def test_a_well_formed_public_repository_can_be_added_and_is_pinned_to_the_inspected_commit(tmp_path) -> None:
    registry = build_default_registry()
    found = custom.inspect("someone/my-tirex", "tirex-2", registry, fake_api(files=("config.json", "model.safetensors", "model.ckpt", "x.py")))
    assert found["blocking"] == [] and found["sha"] == SHA and found["params"] == 82_000_000 and found["license"] == "apache-2.0"
    assert found["ignored"] == ["model.ckpt", "x.py"] and any("never downloaded" in w for w in found["warnings"])
    entry = custom.add(tmp_path, registry, "someone/my-tirex", "tirex-2", "My TiRex", fake_api())
    info = registry.info(entry["id"])
    assert entry["id"] == "custom-someone-my-tirex" and info.name == "My TiRex" and info.family == "tirex" and info.kind == "foundation"
    assert info.details["repo"] == "someone/my-tirex" and info.details["custom_revision"] == SHA and info.details["custom"] is True
    assert info.capabilities == registry.info("tirex-2").capabilities and "third party" in " ".join(info.notes)
    adapter = registry.create(entry["id"])
    assert adapter.info.id == entry["id"] and adapter.manifest["repo"] == "someone/my-tirex"
    assert registry.create("tirex-2").manifest["repo"] == "NX-AI/TiRex-2"  # the built-in model is untouched
    # a second registry (the worker process) sees it from the file; removal takes it away again
    other = build_default_registry()
    custom.sync(other, tmp_path)
    assert entry["id"] in other.ids()
    custom.remove(tmp_path, registry, entry["id"])
    custom.sync(other, tmp_path)
    assert entry["id"] not in registry.ids() and entry["id"] not in other.ids()


@pytest.mark.parametrize("kwargs, expect", [
    ({"private": True}, "private or gated"), ({"gated": True}, "private or gated"),
    ({"files": ("config.json", "model-00001-of-00002.safetensors", "model.safetensors.index.json")}, "config.json and a single model.safetensors"),
    ({"files": ("model.safetensors",)}, "config.json and a single model.safetensors"),
    ({"size": 5_000_000_000}, "limit is 4 GB"),
])
def test_repositories_that_cannot_be_run_safely_are_refused(kwargs, expect) -> None:
    found = custom.inspect("someone/model", "tirex-2", build_default_registry(), fake_api(**kwargs))
    assert any(expect in b for b in found["blocking"])


def test_bad_input_and_built_in_repositories_are_refused_before_any_network_call() -> None:
    registry = build_default_registry()
    boom = NS(model_info=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network call expected")))
    assert custom.inspect("not a repo", "tirex-2", registry, boom)["blocking"]
    assert custom.inspect("../../etc/passwd", "tirex-2", registry, boom)["blocking"]
    assert custom.inspect("a/b", "no-such-model", registry, boom)["blocking"]
    assert custom.inspect("a/b", "naive", registry, boom)["blocking"]  # a baseline has no weights to swap
    builtin = custom.inspect("NX-AI/TiRex-2", "tirex-2", registry, fake_api())
    assert any("already a built-in model" in b for b in builtin["blocking"])


def test_a_licence_that_is_missing_or_restricted_is_flagged_and_unreachable_repositories_say_so() -> None:
    registry = build_default_registry()
    assert any("declares no licence" in w for w in custom.inspect("a/b", "tirex-2", registry, fake_api(license_=None))["warnings"])
    assert any("other" in w for w in custom.inspect("a/b", "tirex-2", registry, fake_api(license_="other"))["warnings"])
    failing = NS(model_info=lambda *a, **k: (_ for _ in ()).throw(type("RepositoryNotFoundError", (Exception,), {})("x")))
    assert any("not found" in b for b in custom.inspect("a/b", "tirex-2", registry, failing)["blocking"])


def test_adding_twice_is_refused(tmp_path) -> None:
    registry = build_default_registry()
    custom.add(tmp_path, registry, "someone/model", "tirex-2", api=fake_api())
    with pytest.raises(ValueError, match="already added"):
        custom.add(tmp_path, registry, "someone/model", "tirex-2", api=fake_api())
    with pytest.raises(ValueError, match="private or gated"):
        custom.add(tmp_path, registry, "someone/other", "tirex-2", api=fake_api(private=True))


def test_a_custom_download_fetches_only_the_two_files_at_the_pinned_commit(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    monkeypatch.setattr(cache, "hub_dir", lambda: tmp_path / "hub")
    calls = []

    def fake_snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        snap = cache.repo_dir(repo) / "snapshots" / kwargs["revision"]
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"0")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    status = cache.download("someone/model", SHA)
    assert calls == [("someone/model", {"revision": SHA, "allow_patterns": ["config.json", "model.safetensors"]})]
    assert status["downloaded"] and status["revision"] == SHA  # the pinned commit is the one the adapters will load


def test_the_api_adds_lists_and_removes_a_custom_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(custom, "_hub_api", lambda: fake_api())
    app = create_app(root=tmp_path, allowed_hosts=("testserver",))
    client = TestClient(app)
    try:
        body = {"repo": "someone/my-tirex", "base": "tirex-2", "name": "Mine"}
        found = client.post("/api/models/custom/inspect", json=body).json()
        assert found["blocking"] == [] and found["sha"] == SHA
        assert client.post("/api/models/custom", json=body).status_code == 422  # the acknowledgement is required
        assert client.post("/api/models/custom", json={**body, "accept": True}).status_code == 201
        assert any("already added" in b for b in client.post("/api/models/custom/inspect", json=body).json()["blocking"])
        listed = {m["id"]: m for m in client.get("/api/models").json()}
        mine = listed["custom-someone-my-tirex"]
        assert mine["details"]["custom"] and mine["name"] == "Mine" and mine["cache"]["downloaded"] is False
        assert client.delete("/api/models/custom/tirex-2").status_code == 400  # built-in models cannot be removed
        assert client.delete("/api/models/custom/custom-someone-my-tirex").status_code == 200
        assert "custom-someone-my-tirex" not in {m["id"] for m in client.get("/api/models").json()}
    finally:
        app.state.worker.stop()


def test_requests_that_sync_the_registry_at_the_same_time_do_not_register_a_model_twice(tmp_path) -> None:
    registry = build_default_registry()
    custom.add(tmp_path, registry, "someone/my-tirex", "tirex-2", "My TiRex", fake_api())
    other = build_default_registry()
    ids = other.ids

    def slow_ids():
        names = ids()
        time.sleep(0.02)  # widens the gap between "is it there?" and "add it", which two requests can hit at once
        return names

    other.ids = slow_ids
    errors: list[BaseException] = []
    start = threading.Barrier(6)

    def run() -> None:
        start.wait()
        try:
            custom.sync(other, tmp_path)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and "custom-someone-my-tirex" in ids()
