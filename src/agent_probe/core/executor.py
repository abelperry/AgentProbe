"""BaseTaskExecutor — pure interface for evaluation executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_probe.config import AgentConfig, DatasetConfig, ModelConfig, SandboxConfig


@dataclass(frozen=True)
class EvalUnit:
    """Lightweight scheduling token — only IDs, no data loaded."""

    qid: str
    agent_name: str
    model_name: str
    dataset_name: str


@dataclass(frozen=True)
class EvalContext:
    """Full context for processing a single question — passed to Task methods."""

    unit: EvalUnit
    model_config: ModelConfig
    agent_config: AgentConfig
    dataset_config: DatasetConfig
    sandbox_config: SandboxConfig
    output_dir: Path

    def log_tag(self) -> str:
        """Return a compact tag for log messages, e.g. ``zcb_002|claude-sonnet-4-6``."""
        return f"{self.unit.qid}|{self.unit.model_name}"


class BaseTaskExecutor(ABC):
    """Abstract executor interface.

    Subclasses decide how to build eval units, schedule inference/judge,
    and persist results.  Only ``run()`` is exposed.
    """

    @abstractmethod
    async def run(self) -> None:
        """Execute the full evaluation."""
