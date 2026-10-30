"""Custom checkpoints: a Hugging Face repository the user names, run by the adapter of a model family they pick.

This is deliberately not "run any model": only the two files an adapter reads (``config.json`` and ``model.safetensors``) are ever
downloaded, so nothing in the repository can execute code (no pickle, no ``trust_remote_code``, no Python files), and the commit
that was inspected is the commit that is downloaded and run, so a later push to the repository cannot change what runs.
Gated and private repositories are refused (the app holds no credentials), as are sharded or oversized checkpoints.

What a custom checkpoint cannot declare is its own capabilities: context length, quantile levels and covariate support are
taken from the base model, which is right for a fine-tune of it and wrong for a different architecture (which then fails to load
or gives meaningless output). The page says so before anything is added.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tsfm_lab.models.base import ModelInfo
from tsfm_lab.models.registry import Registry

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
MAX_BYTES = 4_000_000_000
FILE = ("data", "user", "custom_models.json")
_LOCK = threading.RLock()  # requests run on several threads: the file and the registry change together, and sync() is called from add() and remove()
RISKY_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".py", ".onnx")

CUSTOM_NOTES = (
    ("A custom checkpoint from a third party, not reviewed by this app: its weights were inspected only for format (one "
    "safetensors file, no code is ever downloaded or run), not for quality, safety of the output or training-data provenance."),
    ("Its context length, quantile levels and covariate support are those of the base model it runs on: right for a fine-tune of "
    "that model, wrong for a different architecture."),
)


def path_of(root: Path) -> Path:
    return Path(root).joinpath(*FILE)


def read_entries(root: Path) -> list[dict[str, Any]]:
    path = path_of(root)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _write(root: Path, entries: list[dict[str, Any]]) -> None:
    path = path_of(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    tmp.replace(path)


def slug(repo: str) -> str:
    return "custom-" + re.sub(r"[^a-z0-9]+", "-", repo.lower()).strip("-")[:60]


def _hub_api():
    from huggingface_hub import HfApi

    return HfApi()


def inspect(repo: str, base_id: str, registry: Registry, api: Any = None) -> dict[str, Any]:
    """What the repository is and whether it can be added. ``blocking`` lists reasons it cannot; ``warnings`` are for the user."""
    repo = (repo or "").strip()
    out: dict[str, Any] = {"repo": repo, "base": base_id, "blocking": [], "warnings": [], "ignored": []}
    if not REPO_ID.match(repo):
        out["blocking"].append("Enter a repository id like organisation/name.")
        return out
    base = next((i for i in registry.infos() if i.id == base_id), None)
    if base is None or base.kind != "foundation" or base.details.get("custom"):
        out["blocking"].append("Pick one of the built-in models as the base: its adapter runs the checkpoint.")
        return out
    builtin = {str(i.details.get("repo", "")).lower() for i in registry.infos() if not i.details.get("custom")}
    if repo.lower() in builtin:
        out["blocking"].append("This repository is already a built-in model.")
    try:
        info = (api or _hub_api()).model_info(repo, files_metadata=True, timeout=20)
    except Exception as error:  # noqa: BLE001  (not found, offline, rate limit...)
        name = type(error).__name__
        out["blocking"].append("Repository not found, or Hugging Face could not be reached." if "NotFound" in name else
                               f"Could not read the repository ({name}).")
        return out
    siblings = {s.rfilename: s for s in (info.siblings or [])}
    card = info.card_data
    license_ = (card.get("license") if card else None) or next((t.split(":", 1)[1] for t in info.tags or [] if t.startswith("license:")), None)
    weights = siblings.get("model.safetensors")
    size = int(getattr(weights, "size", 0) or 0)
    params = getattr(getattr(info, "safetensors", None), "total", None)
    out.update(sha=info.sha, author=info.author or repo.split("/")[0], license=license_, downloads=info.downloads, likes=info.likes,
               last_modified=info.last_modified.isoformat() if info.last_modified else None, size_bytes=size, params=params,
               url=f"https://huggingface.co/{repo}/tree/{info.sha}")
    if info.private or info.gated:
        out["blocking"].append("This repository is private or gated: the app holds no Hugging Face credentials, so it cannot download it.")
    if "config.json" not in siblings or weights is None:
        out["blocking"].append("It must hold config.json and a single model.safetensors at its top level.")
    if any(n.endswith(".safetensors.index.json") for n in siblings):
        out["blocking"].append("Sharded checkpoints (several safetensors files) are not supported.")
    if size > MAX_BYTES:
        out["blocking"].append(f"The weights are {size / 1e9:.1f} GB; the limit is {MAX_BYTES / 1e9:.0f} GB.")
    out["ignored"] = sorted(n for n in siblings if n.endswith(RISKY_SUFFIXES))
    if out["ignored"]:
        out["warnings"].append(f"{len(out['ignored'])} other file(s) in the repository (pickle, ONNX or Python) are never downloaded: "
                               + ", ".join(out["ignored"][:4]) + (", …" if len(out["ignored"]) > 4 else "") + ".")
    if not license_:
        out["warnings"].append("The repository declares no licence: check its terms before relying on the weights.")
    elif license_.lower() in ("other", "unknown") or "nc" in license_.lower().split("-"):
        out["warnings"].append(f"Its licence is “{license_}”: read it before using the results beyond experiments.")
    if (card.get("library_name") if card else None) not in (None, "") and base.family not in str(card.get("library_name")).lower() \
            and base.family not in " ".join(info.tags or []).lower():
        out["warnings"].append(f"The repository does not mention {base.family}: make sure it really is a checkpoint of that architecture.")
    return out


def add(root: Path, registry: Registry, repo: str, base_id: str, name: str | None = None, api: Any = None) -> dict[str, Any]:
    found = inspect(repo, base_id, registry, api)
    if found["blocking"]:
        raise ValueError(" ".join(found["blocking"]))
    with _LOCK:
        entries = read_entries(root)
        model_id = slug(found["repo"])
        if any(e["id"] == model_id or e["repo"].lower() == found["repo"].lower() for e in entries):
            raise ValueError("This repository is already added.")
        entry = {"id": model_id, "name": (name or found["repo"].split("/")[1]).strip()[:60] or model_id, "repo": found["repo"], "base": base_id,
                 "revision": found["sha"], "params": found.get("params"), "license": found.get("license"), "author": found.get("author"),
                 "size_bytes": found.get("size_bytes"), "added": datetime.now(UTC).isoformat(timespec="seconds")}
        _write(root, [*entries, entry])
        sync(registry, root)
        return entry


def remove(root: Path, registry: Registry, model_id: str) -> str:
    with _LOCK:
        entries = read_entries(root)
        entry = next((e for e in entries if e["id"] == model_id), None)
        if entry is None:
            raise KeyError(f"unknown custom model '{model_id}'")
        _write(root, [e for e in entries if e["id"] != model_id])
        sync(registry, root)
        return entry["repo"]


def _derive(registry: Registry, entry: dict[str, Any]) -> tuple[ModelInfo, Any]:
    base = registry.info(entry["base"])
    info = replace(
        base, id=entry["id"], name=entry["name"], params=entry.get("params") or base.params,
        description=f"Custom checkpoint {entry['repo']} (by {entry.get('author') or 'unknown'}), run by the {base.name} adapter.",
        license=entry.get("license") or "not declared", license_url=f"https://huggingface.co/{entry['repo']}",
        source_url=f"https://huggingface.co/{entry['repo']}",
        details={**base.details, "repo": entry["repo"], "custom": True, "base": entry["base"], "custom_revision": entry["revision"],
                 "vendor_reported": {}, "restriction": None},
        notes=(*CUSTOM_NOTES, *base.notes))

    def make():
        adapter = registry.create(entry["base"])
        adapter.manifest = {**adapter.manifest, "repo": entry["repo"], "id": entry["id"], "name": entry["name"]}
        adapter.info = info
        return adapter

    return info, make


def sync(registry: Registry, root: Path) -> None:
    """Make the registry list exactly the custom models in the file (idempotent; the worker calls it before every job)."""
    with _LOCK:
        wanted = {e["id"]: e for e in read_entries(root)}
        for model_id in [i.id for i in registry.infos() if i.details.get("custom")]:
            if model_id not in wanted:
                registry.remove(model_id)
        for model_id, entry in wanted.items():
            if model_id not in registry.ids() and entry["base"] in registry.ids():
                registry.add(*_derive(registry, entry))
