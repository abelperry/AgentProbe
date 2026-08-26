"""Data adapters — generic, reusable loaders for benchmark datasets."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel as PydanticBaseModel

from agent_probe.core.models import BaseQuestion

Q = TypeVar("Q", bound=BaseQuestion)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------
class BaseAdapter(ABC, Generic[Q]):
    """Interface for loading benchmark question data.

    Adapters are generic and reusable — they are NOT tied to a specific
    benchmark.  Benchmarks reference an adapter *type* in their YAML config.
    """

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return all question IDs (lightweight, no full data loaded)."""

    @abstractmethod
    def load(self, qid: str) -> Q:
        """Load the full Question object for the given *qid*."""


# ---------------------------------------------------------------------------
# Built-in: LocalJsonlAdapter
# ---------------------------------------------------------------------------
class LocalJsonlAdapter(BaseAdapter[Q]):
    """Load questions from a local ``questions.jsonl`` file.

    Each JSON line MUST contain an ``"id"`` field.  On construction the file
    is scanned once to build a *qid → line-number* index so that subsequent
    ``load()`` calls are efficient.
    """

    def __init__(self, data_dir: str, question_type: type[Q]) -> None:
        self.data_dir = Path(data_dir)
        self.question_type = question_type
        self._id_index: dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        jsonl_path = self.data_dir / "questions.jsonl"
        with open(jsonl_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                q = self.question_type.model_validate_json(line)
                self._id_index[q.qid()] = line_no

    def list_ids(self) -> list[str]:
        return list(self._id_index.keys())

    def load(self, qid: str) -> Q:
        line_no = self._id_index[qid]
        jsonl_path = self.data_dir / "questions.jsonl"
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == line_no:
                    return self.question_type.model_validate_json(line)
        raise KeyError(f"Question {qid!r} not found")


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------
ADAPTER_REGISTRY: dict[str, type[BaseAdapter[Any]]] = {
    "local_jsonl": LocalJsonlAdapter,  # type: ignore[dict-item]
}


def create_adapter(adapter_type: str, **kwargs: Any) -> BaseAdapter[Any]:
    """Create an adapter instance by its registered type name."""
    cls = ADAPTER_REGISTRY.get(adapter_type)
    if cls is None:
        raise ValueError(
            f"Unknown adapter type {adapter_type!r}. "
            f"Available: {list(ADAPTER_REGISTRY)}"
        )
    return cls(**kwargs)
