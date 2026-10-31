"""Dataset catalog used by the API: bundled synthetic scenarios and user-uploaded local files.

A dataset is referred to by a *spec*: ``{"source": "synthetic", "id": "trend", "params": {...}}``,
``{"source": "local", "id": "<dataset id>"}`` or ``{"source": "hub", "id": "<catalog id>", "params": {"max_series": N}}``
(a downloaded entry of the curated online catalog). Uploaded files are stored under ``data/user/`` (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tsfm_lab.data import hub_catalog, synthetic
from tsfm_lab.data.local import (
    READERS,
    ColumnMapping,
    guess_mapping,
    load_dataframe,
    read_table,
)
from tsfm_lab.data.schema import Dataset
from tsfm_lab.isolation import find_root

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
PARAM_LIMITS = {"n_series": (1, 500), "length": (30, 200_000), "seed": (0, 2**31 - 1)}
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
LOCAL_ID = re.compile(r"local-[0-9a-f]{16}")  # the only shape of id a registered dataset gets; anything else never reaches a path


class DatasetCatalog:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else find_root() / "data" / "user"
        (self.root / "uploads").mkdir(parents=True, exist_ok=True)
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, Dataset] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ listing
    def list(self) -> list[dict[str, Any]]:
        out = []
        for sc in synthetic.SCENARIOS.values():
            out.append({"source": "synthetic", "id": sc.id, "name": sc.name, "description": sc.description,
                        "tags": list(sc.tags), "params": sc.defaults, "license": "generated (CC0)",
                        "contamination_risk": "none"})
        for e in hub_catalog.listing():
            if e["downloaded"]:
                src = hub_catalog.sources()[e["source"]]
                out.append({"source": "hub", "id": e["id"], "name": e["title"] + " · " + e["freq_label"] + ("" if e["source"] == "fev" else f" ({src['name']})"),
                            "description": f"{e['n_series']} series, {e['length']['min']}-{e['length']['max']} steps; "
                                           f"{len(e['known'])} known and {len(e['past'])} past covariates. Online benchmark.",
                            "tags": ["online", e["source"], e["family"]], "params": {"max_series": hub_catalog.DEFAULT_MAX_SERIES},
                            "task": e["task"], "benchmark": src["benchmark"],
                            "license": f"{src.get('card_license') or 'other'} (see the catalog entry)", "contamination_risk": "possible"})
        for meta in sorted(self.root.joinpath("datasets").glob("*.json")):
            d = json.loads(meta.read_text(encoding="utf-8"))
            out.append({"source": "local", "id": d["id"], "name": d["name"], "description": d["description"],
                        "tags": ["local"], "summary": d["summary"], "report": d["report"],
                        "license": "user data", "contamination_risk": "unknown"})
        return out

    # ------------------------------------------------------------------ resolution
    def resolve(self, spec: dict[str, Any]) -> Dataset:
        source, sid = spec.get("source"), spec.get("id")
        if source == "synthetic":
            params = self._clean_params(sid, spec.get("params") or {})
            key = f"syn:{sid}:{json.dumps(params, sort_keys=True)}"
            return self._cached(key, lambda: synthetic.build(sid, **params))
        if source == "local":
            return self._cached(f"local:{sid}", lambda: self._load_local(sid))
        if source == "hub":
            e = hub_catalog.entry(sid)
            params = spec.get("params") or {}
            unknown = set(params) - {"max_series"}
            if unknown:
                raise ValueError(f"parameter '{min(unknown)}' is not valid for online datasets")
            limit = int(params.get("max_series", hub_catalog.DEFAULT_MAX_SERIES))
            if not 1 <= limit <= hub_catalog.MAX_SERIES_LIMIT:
                raise ValueError(f"max_series must be between 1 and {hub_catalog.MAX_SERIES_LIMIT}")
            return self._cached(f"hub:{sid}:{limit}", lambda: hub_catalog.read(e, limit))
        raise ValueError("dataset spec needs source 'synthetic', 'local' or 'hub'")

    @staticmethod
    def _clean_params(sid: str, params: dict[str, Any]) -> dict[str, Any]:
        if sid not in synthetic.SCENARIOS:
            raise KeyError(f"unknown synthetic scenario '{sid}'")
        allowed = synthetic.SCENARIOS[sid].defaults
        clean: dict[str, Any] = {}
        for key, value in params.items():
            if key not in allowed:
                raise ValueError(f"parameter '{key}' is not valid for scenario '{sid}'")
            if key in PARAM_LIMITS:
                lo, hi = PARAM_LIMITS[key]
                value = int(value)
                if not lo <= value <= hi:
                    raise ValueError(f"{key} must be between {lo} and {hi}")
            clean[key] = value
        return clean

    def _cached(self, key: str, build) -> Dataset:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        dataset = build()
        with self._lock:
            self._cache[key] = dataset
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
        return dataset

    # ------------------------------------------------------------------ uploads
    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"file too large (limit {MAX_UPLOAD_BYTES // 2**20} MB)")
        name = SAFE_NAME.sub("_", Path(filename).name)[:120] or "upload"
        if Path(name).suffix.lower() not in READERS:  # refused before anything is written
            raise ValueError(f"unsupported file type '{Path(name).suffix}'. Supported: {', '.join(READERS)}")
        file_id = hashlib.sha256(data).hexdigest()[:16]
        folder = self.root / "uploads" / file_id
        existed = folder.is_dir()
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*"):  # the same bytes under another name: one file per id, so the reader always gets this one
            if old.is_file() and old.name != name:
                old.unlink()
        path = folder / name
        path.write_bytes(data)
        try:
            frame = read_table(path)
        except Exception as error:
            if not existed:  # a file that cannot be read is not kept
                shutil.rmtree(folder, ignore_errors=True)
            raise ValueError(f"could not read '{name}': {error}") from error
        mapping = guess_mapping(frame)
        return {"file_id": file_id, "filename": name, "rows": len(frame), "columns": list(map(str, frame.columns)),
                "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
                "head": json.loads(frame.head(8).to_json(orient="records", date_format="iso")),
                "guessed_mapping": asdict(mapping)}

    def _upload_path(self, file_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{16}", file_id):
            raise ValueError("invalid file id")
        files = [p for p in (self.root / "uploads" / file_id).glob("*") if p.is_file()]
        if not files:
            raise KeyError(f"unknown upload '{file_id}'")
        return files[0]

    def register_local(self, file_id: str, mapping: dict[str, Any], name: str | None) -> dict[str, Any]:
        path = self._upload_path(file_id)
        frame = read_table(path)
        cm = ColumnMapping(timestamp=mapping["timestamp"], targets=list(mapping["targets"]), id=mapping.get("id") or None,
                           past_covariates=list(mapping.get("past_covariates") or []),
                           known_covariates=list(mapping.get("known_covariates") or []), freq=mapping.get("freq") or None)
        ds, report = load_dataframe(frame, cm, name or path.stem, path.name)
        record = {"id": ds.id, "name": name or path.stem, "description": ds.description, "file_id": file_id,
                  "filename": path.name, "mapping": asdict(cm), "summary": ds.summary(), "report": report}
        (self.root / "datasets" / f"{ds.id}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        with self._lock:
            self._cache[f"local:{ds.id}"] = ds
        return record

    def _load_local(self, dataset_id: str) -> Dataset:
        if not LOCAL_ID.fullmatch(dataset_id):
            raise KeyError(f"unknown local dataset '{dataset_id}'")
        meta = self.root / "datasets" / f"{dataset_id}.json"
        if not meta.is_file():
            raise KeyError(f"unknown local dataset '{dataset_id}'")
        rec = json.loads(meta.read_text(encoding="utf-8"))
        frame = read_table(self._upload_path(rec["file_id"]))
        ds, _ = load_dataframe(frame, ColumnMapping(**rec["mapping"]), rec["name"], rec["filename"])
        return ds

    def suite_entries(self, dataset_ids: list[str], horizon: int, windows: int) -> list[dict[str, Any]]:
        """Leaderboard entries for the user's own datasets, shaped like the online catalog's: one horizon for all of them."""
        entries = []
        for dataset_id in dict.fromkeys(dataset_ids):
            if not LOCAL_ID.fullmatch(dataset_id) or not (self.root / "datasets" / f"{dataset_id}.json").is_file():
                raise KeyError(f"unknown local dataset '{dataset_id}'")
            rec = json.loads((self.root / "datasets" / f"{dataset_id}.json").read_text(encoding="utf-8"))
            entries.append({"id": dataset_id, "title": rec["name"], "source": "mine", "local": True,
                            "pandas_freq": rec["summary"].get("freq") or "", "task": {"horizon": horizon, "num_windows": windows}})
        return entries

    def clear_cache(self) -> None:
        """Drop every in-memory dataset (after a download replaced a file that one of them was read from)."""
        with self._lock:
            self._cache.clear()

    def forget_hub(self, entry_id: str) -> None:
        """Drop in-memory copies of a catalog dataset (after its file was removed)."""
        with self._lock:
            for key in [k for k in self._cache if k.startswith(f"hub:{entry_id}:")]:
                del self._cache[key]

    def delete_local(self, dataset_id: str) -> None:
        if not LOCAL_ID.fullmatch(dataset_id):
            raise KeyError(f"unknown local dataset '{dataset_id}'")
        meta = self.root / "datasets" / f"{dataset_id}.json"
        if meta.is_file():
            meta.unlink()
        with self._lock:
            self._cache.pop(f"local:{dataset_id}", None)
