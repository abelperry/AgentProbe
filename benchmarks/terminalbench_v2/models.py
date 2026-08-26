"""TerminalBench v2 Q/I/J models."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, ConfigDict, Field

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion


class TerminalBenchV2Question(BaseQuestion):
    """Question schema for Terminal Bench v2 records."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        validation_alias=AliasChoices("qid", "task_id"),
    )
    docker_image: str
    workspace_dir: str
    contexts: list[str] = Field(default_factory=list)
    verifier_timeout: int = 1800
    cpu_cores: float = 1.0
    memory_gib: float = 2.0
    categories: list[str] = Field(default_factory=list)

    def qid(self) -> str:
        return self.task_id

    @property
    def prompt(self) -> str:
        return self.contexts[0] if self.contexts else ""


class TerminalBenchV2Inference(BaseInference):
    """Inference result with deterministic verifier score."""

    response: str = ""
    score: float = 0.0
    test_log_path: Path | None = None
    tests_status: dict | None = None


class TerminalBenchV2Judgement(BaseJudgement):
    """Offline judgement backed by the verifier score."""

    score: float = 0.0
