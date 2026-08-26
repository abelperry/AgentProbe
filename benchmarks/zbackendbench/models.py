"""ZBackendBench Q/I/J models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, ConfigDict, Field

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion


class ZBackendBenchQuestion(BaseQuestion):
    """Question schema for ZBackendBench records."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        validation_alias=AliasChoices("qid", "task_id"),
    )
    docker_image: str
    workspace_dir: str
    contexts: list[str] = Field(default_factory=list)
    verifier_timeout: int = 1800
    cpu_cores: float = 1.0
    memory_gib: float = 8.0
    categories: list[str] = Field(default_factory=list)

    def qid(self) -> str:
        return self.task_id

    @property
    def prompt(self) -> str:
        return self.contexts[0] if self.contexts else ""


class ZBackendBenchInference(BaseInference):
    """Inference output plus deterministic verifier result."""

    response: str = ""
    deterministic_score: float = 0.0
    patch_path: Path | None = None
    test_log_path: Path | None = None


class ZBackendBenchJudgement(BaseJudgement):
    """Final judgement after deterministic score and optional quality penalty."""

    score: float = 0.0
    deterministic_score: float = 0.0
    judge_output: str = ""
    code_quality_rubric: dict[str, Any] | None = None
