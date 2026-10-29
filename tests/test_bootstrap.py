import importlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def bootstrap(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("bootstrap")


@pytest.mark.parametrize(("driver", "usable"), [("621.01", True), ("580.65.06", True), ("580", True), ("575.57.08", False), ("570.86.10", False), ("535.183", False)])
def test_the_cuda_13_build_needs_a_driver_of_the_580_series_or_newer(bootstrap, driver, usable) -> None:
    assert bootstrap.cuda_build_usable(driver) is usable


@pytest.mark.parametrize("unreadable", [None, "", "N/A", "[Not Supported]"])
def test_an_unreadable_driver_version_does_not_block_the_gpu_build(bootstrap, unreadable) -> None:
    assert bootstrap.cuda_build_usable(unreadable) is True


def test_an_old_driver_gets_the_cpu_build_and_a_message(bootstrap, monkeypatch, capsys) -> None:
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class Probe:
        returncode, stdout = 0, "GPU 0: NVIDIA GeForce GTX 1660 (UUID: x)\n"

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: Probe())
    monkeypatch.setattr(bootstrap, "nvidia_driver_version", lambda: "570.86.10")
    assert bootstrap.detect_device("auto") == "cpu"
    assert "older than 580" in capsys.readouterr().out
    monkeypatch.setattr(bootstrap, "nvidia_driver_version", lambda: "621.01")
    assert bootstrap.detect_device("auto") == "cuda"
    assert bootstrap.detect_device("cuda") == "cuda"  # forcing it is always allowed
