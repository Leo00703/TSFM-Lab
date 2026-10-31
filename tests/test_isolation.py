import os
import subprocess
import sys
from pathlib import Path

import pytest

from tsfm_lab import isolation

PATH_VARS = (
    "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
    "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PROJECT_ENVIRONMENT", "PIP_CACHE_DIR",
    "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "PYTHONPYCACHEPREFIX",
    "TMPDIR", "TEMP", "TMP",
)


def test_every_cache_points_inside_the_project(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    mapping = isolation.apply(root=tmp_path, environ=env)
    assert env == mapping
    for name in PATH_VARS:
        assert Path(mapping[name]).is_relative_to(tmp_path), name


def test_telemetry_is_disabled(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    isolation.apply(root=tmp_path, environ=env)
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_opt_out_leaves_environment_untouched(tmp_path: Path) -> None:
    env = {isolation.OPT_OUT_VAR: "1", "HF_HOME": "/somewhere/else"}
    assert isolation.apply(root=tmp_path, environ=env) == {}
    assert env["HF_HOME"] == "/somewhere/else"


def test_find_root_locates_the_project() -> None:
    root = isolation.find_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "tsfm_lab").is_dir()


def test_on_macos_unsupported_gpu_operations_fall_back_to_the_cpu_unless_the_user_chose_otherwise(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(isolation.sys, "platform", "darwin")
    env: dict[str, str] = {}
    mapping = isolation.apply(root=tmp_path, environ=env)
    assert env == mapping and env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    mine: dict[str, str] = {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    isolation.apply(root=tmp_path, environ=mine)
    assert mine["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


@pytest.mark.parametrize("script", ["download_models", "smoke_test"])
def test_a_script_points_huggingface_at_the_project_cache(script: str) -> None:
    """The scripts import timestral, which imports huggingface_hub and reads its cache path once: tsfm_lab has to be imported first."""
    root = isolation.find_root()
    code = (f"import sys; sys.path.insert(0, {str(root / 'scripts')!r}); import {script}; "
            "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True, timeout=120).stdout.strip()
    assert Path(out).is_relative_to(root / ".cache"), out


def test_a_hub_cache_set_outside_the_project_is_overridden(tmp_path: Path) -> None:
    env = {"HF_HUB_CACHE": str(tmp_path / "elsewhere")}
    isolation.apply(root=tmp_path / "project", environ=env)
    assert Path(env["HF_HUB_CACHE"]).is_relative_to(tmp_path / "project")


def test_downloads_and_the_download_status_look_in_the_same_folder_even_when_the_user_set_a_hub_cache() -> None:
    """HF_HUB_CACHE beats HF_HOME inside huggingface_hub: left alone it sent the weights outside the project, where the status never found them."""
    code = ("import tsfm_lab; from huggingface_hub import constants; from tsfm_lab.models import cache; "
            "from tsfm_lab.data import hub_catalog as H; "
            "e = {'repo': 'a/b', 'revision': 'r'}; "
            "print(constants.HF_HUB_CACHE); print(cache.hub_dir()); print(H._cache_dir(e).parents[2])")
    env = {**os.environ, "HF_HUB_CACHE": str(Path(isolation.find_root().anchor) / "elsewhere" / "hub")}
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True, timeout=120, env=env).stdout.splitlines()
    downloads, status, datasets = (Path(x) for x in out[:3])
    assert downloads.is_relative_to(isolation.find_root() / ".cache"), downloads
    assert downloads == status == datasets


def test_without_the_isolation_the_status_follows_the_hub_cache_the_user_chose(tmp_path: Path, monkeypatch) -> None:
    from tsfm_lab.models import cache

    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert cache.hub_dir() == tmp_path / "hub"
    monkeypatch.delenv("HF_HUB_CACHE")
    assert cache.hub_dir() == tmp_path / "home" / "hub"
