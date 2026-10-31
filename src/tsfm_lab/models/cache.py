"""Weights cache (Hugging Face layout inside the project): status, download, removal."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from tsfm_lab.isolation import find_root


def hub_dir() -> Path:
    """Where huggingface_hub keeps the downloads: HF_HUB_CACHE if set, else HF_HOME/hub (the isolation sets both inside the project)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", find_root() / ".cache" / "huggingface")) / "hub"


def repo_dir(repo: str) -> Path:
    return hub_dir() / ("models--" + repo.replace("/", "--"))


def status(repo: str) -> dict[str, Any]:
    """Whether the checkpoint (config.json + model.safetensors) is fully present locally."""
    root = repo_dir(repo)
    ref = root / "refs" / "main"
    out: dict[str, Any] = {"downloaded": False, "size_bytes": 0, "revision": None, "path": str(root)}
    if not ref.is_file():
        return out
    revision = ref.read_text().strip()
    snap = root / "snapshots" / revision
    if (snap / "config.json").exists() and (snap / "model.safetensors").exists():
        out.update(downloaded=True, revision=revision)
        # Symlinked blobs (Linux/macOS) or plain copies (Windows without symlinks): stat() follows both.
        out["size_bytes"] = sum(f.stat().st_size for f in snap.iterdir() if f.is_file() or f.is_symlink())
    return out


def download(repo: str, revision: str | None = None) -> dict[str, Any]:
    """Fetch config.json and model.safetensors (nothing else). With a ``revision`` (a commit sha, for custom checkpoints) exactly that
    commit is fetched and recorded as the one ``status`` reports, so a later push to the repository cannot change what runs."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo, revision=revision, allow_patterns=["config.json", "model.safetensors"])
    if revision:
        ref = repo_dir(repo) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision)
    return status(repo)


def remove(repo: str) -> bool:
    root = repo_dir(repo)
    if not root.exists():
        return False
    shutil.rmtree(root, ignore_errors=True)
    if root.exists():  # on Windows a file that is open cannot be deleted: say so rather than report weights as gone
        raise ValueError(f"could not remove all the files of {repo}: some are in use. Close whatever has them open and try again")
    return True


def read_config(repo: str) -> dict[str, Any]:
    """The checkpoint's config.json (heads, widths...) when downloaded; empty otherwise."""
    import json

    st = status(repo)
    if not st["downloaded"]:
        return {}
    return json.loads((repo_dir(repo) / "snapshots" / st["revision"] / "config.json").read_text(encoding="utf-8"))
