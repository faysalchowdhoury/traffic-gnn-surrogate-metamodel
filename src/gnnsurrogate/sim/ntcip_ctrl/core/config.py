"""Pydantic base model and YAML/JSON loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from .exceptions import ConfigurationError


class StrictModel(BaseModel):
    """Immutable model rejecting unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


ModelT = TypeVar("ModelT", bound=BaseModel)


def read_mapping(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError("file not found", context={"path": str(p)})
    try:
        text = p.read_text(encoding="utf-8")
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot parse: {exc}", context={"path": str(p)}) from exc
    if not isinstance(data, dict):
        raise ConfigurationError("root must be a mapping", context={"path": str(p)})
    return data


def load_model(path: str | Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(read_mapping(path))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid {model.__name__}:\n{exc}", context={"path": str(path)}) from exc
