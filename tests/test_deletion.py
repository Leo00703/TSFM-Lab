"""A delete that leaves files behind (a model still open on Windows, say) must say so instead of reporting success."""

from __future__ import annotations

import os

import pytest

from tsfm_lab.data import synthetic
from tsfm_lab.eval.backtest import BacktestConfig
from tsfm_lab.eval.runner import run_experiment
from tsfm_lab.models import cache
from tsfm_lab.store.store import RunStore


def _refuse_to_unlink(monkeypatch, name: str) -> None:
    real = os.unlink

    def unlink(path, *args, **kwargs):
        if os.path.basename(path) == name:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)


def test_removing_weights_that_are_in_use_is_reported_instead_of_claiming_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshot = cache.repo_dir("owner/model") / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    _refuse_to_unlink(monkeypatch, "model.safetensors")
    with pytest.raises(ValueError, match="in use"):
        cache.remove("owner/model")
    assert (snapshot / "model.safetensors").is_file()
    monkeypatch.undo()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert cache.remove("owner/model") is True and cache.remove("owner/model") is False


def test_a_run_whose_files_cannot_be_deleted_stays_in_the_list(tmp_path, monkeypatch) -> None:
    store = RunStore(tmp_path)
    ds = synthetic.build("trend", n_series=2, length=120)
    _, run_id = run_experiment("naive", ds, BacktestConfig(horizon=8, n_windows=1), baselines=(), store=store)
    _refuse_to_unlink(monkeypatch, "arrays.npz")
    with pytest.raises(ValueError, match="in use"):
        store.delete(run_id)
    assert [r["id"] for r in store.list_runs()] == [run_id], "the index must not forget a run whose folder is still there"
    monkeypatch.undo()
    store.delete(run_id)
    assert store.list_runs() == [] and not (tmp_path / run_id).exists()
