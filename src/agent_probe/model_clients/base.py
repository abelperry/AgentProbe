"""Model client interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

ModelRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ModelMessage:
    role: ModelRole
    content: str


@dataclass(frozen=True)
class ModelResponse:
    content: str
    raw: dict[str, Any]


class BaseModelClient(ABC):
    """Minimal model completion interface used by eval utilities."""

    @abstractmethod
    def complete(
        self,
        messages: list[ModelMessage],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Run one non-streaming completion call."""
