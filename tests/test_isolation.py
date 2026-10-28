from pathlib import Path

from tsfm_lab import isolation

PATH_VARS = (
    "HF_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
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
