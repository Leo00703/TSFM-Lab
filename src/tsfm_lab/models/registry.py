"""Model registry. Built-in models register themselves; third-party packages can add models
through the ``tsfm_lab.models`` entry-point group (each entry point is a ``register(registry)``
callable), so a new model never requires touching this code base.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from tsfm_lab.models.base import ModelAdapter, ModelInfo


class Registry:
    def __init__(self) -> None:
        self._specs: dict[str, tuple[ModelInfo, Callable[[], ModelAdapter]]] = {}

    def add(self, info: ModelInfo, factory: Callable[[], ModelAdapter]) -> None:
        if info.id in self._specs:
            raise ValueError(f"model id already registered: {info.id}")
        self._specs[info.id] = (info, factory)

    def ids(self) -> list[str]:
        return list(self._specs)

    def infos(self) -> list[ModelInfo]:
        return [spec[0] for spec in self._specs.values()]

    def info(self, model_id: str) -> ModelInfo:
        return self._spec(model_id)[0]

    def create(self, model_id: str) -> ModelAdapter:
        return self._spec(model_id)[1]()

    def _spec(self, model_id: str):
        try:
            return self._specs[model_id]
        except KeyError:
            raise KeyError(f"unknown model '{model_id}'. Available: {', '.join(self._specs)}") from None


def build_default_registry() -> Registry:
    from tsfm_lab.models import baselines, chronos, timestral

    registry = Registry()
    baselines.register(registry)
    timestral.register(registry)
    chronos.register(registry)
    for entry in entry_points(group="tsfm_lab.models"):
        entry.load()(registry)
    return registry


_default: Registry | None = None


def default_registry() -> Registry:
    global _default
    if _default is None:
        _default = build_default_registry()
    return _default
