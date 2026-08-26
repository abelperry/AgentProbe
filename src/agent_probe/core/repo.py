"""Repository interfaces for evaluation persistence (DDD-style)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_probe.core.models import JudgeResult, MetricsRecord


class JudgeRepo(ABC):
    """Repository for per-question evaluation results (Q + I + J).

    A single instance manages all results across datasets/agents/models.
    Concrete implementations decide how to map the flat key
    (dataset, agent, model, qid) to storage (file paths, DB rows, etc.).
    """

    @abstractmethod
    def save(
        self,
        dataset: str,
        agent: str,
        model: str,
        qid: str,
        result: JudgeResult,
    ) -> None:
        """Append or upsert a single judge result."""

    @abstractmethod
    def find(
        self,
        dataset: str,
        agent: str,
        model: str,
        qid: str,
    ) -> JudgeResult | None:
        """Find a single result, or None if not found."""

    @abstractmethod
    def find_all(
        self,
        dataset: str,
        agent: str,
        model: str,
    ) -> list[JudgeResult]:
        """Find all results for a (dataset, agent, model) combination."""


class MetricsRepo(ABC):
    """Repository for aggregated metrics per (dataset, agent, model)."""

    @abstractmethod
    def save(self, record: MetricsRecord) -> None:
        """Append or upsert a single metrics record."""
