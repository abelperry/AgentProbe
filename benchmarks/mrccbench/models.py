"""MRCCBench data models for AgentProbe."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion, Error

# Deployment-specific; questions.jsonl carries judge_docker. Set
# MRCC_JUDGE_IMAGE to supply a default for data that omits it.
DEFAULT_JUDGE_DOCKER = os.environ.get("MRCC_JUDGE_IMAGE", "")
DEFAULT_INFER_DOCKER = "alexgshaw/break-filter-js-from-html:20251031"


class RoundSpec(BaseModel):
    """One main requirement round."""

    round_id: int
    prompt: str
    scenario_tags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CriticalCheck(BaseModel):
    """Optional single-round dependency check judged by a browser-capable judge."""

    description: str = ""
    depended_by_rounds: list[int] = Field(default_factory=list)


class DependencySpec(BaseModel):
    """Dependency settings for one round."""

    round_id: int
    critical_check: CriticalCheck | None = None


class ChecklistItem(BaseModel):
    """Final checklist item."""

    id: int | str
    description: str
    weight: float = 1.0


class RoundRecord(BaseModel):
    """Persisted infer round summary."""

    kind: Literal["main", "repair", "skipped"]
    round_index: int
    round_id: int
    attempt: int = 0
    prompt: str = ""
    result_excerpt: str = ""
    trace_ref: str | None = None
    skip_reason: str = ""
    skipped_due_to_failed_rounds: list[int] = Field(default_factory=list)


class DependencyCheckResult(BaseModel):
    """Persisted dependency/repair check summary."""

    kind: str
    round_index: int
    round_id: int
    attempt: int = 0
    passed: bool
    build_passed: bool
    critical_check_passed: bool
    summary: str = ""
    symptoms: str = ""
    critical_check: CriticalCheck | None = None
    critical_check_symptom: str = ""
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    judge_output_excerpt: str = ""
    trace_ref: str | None = None
    project_root: str = "/workspace"
    blocked_future_rounds: list[int] = Field(default_factory=list)
    next_runnable_round: int | None = None
    scheduling_note: str = ""
    auto_verified: bool = False
    verification_note: str = ""
    repair_disabled_skip: bool = False


class MRCCBenchQuestion(BaseQuestion):
    """One prepared MRCCBench instance.

    The JSONL is expected to be the AgentProbe-native shape produced by
    ``scripts/build_mrccbench_dataset.py``. Raw chatglm-eval ``contexts`` /
    ``dependency`` records should be converted before evaluation.
    """

    task_id: str
    docker: str = DEFAULT_INFER_DOCKER
    workspace_dir: str = "/workspace"
    description: str = ""
    rounds: list[RoundSpec]
    checklist: list[ChecklistItem]
    dependencies: list[DependencySpec] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=lambda: ["mrccbench", "agent", "multi_round"])
    test_mode: Literal["file", "http"] = "http"
    task_description_for_judge: str | None = None
    judge_docker: str = DEFAULT_JUDGE_DOCKER
    http_port: int = 5173
    http_build_timeout: int = 900
    eval_timeout: int = 7200
    eval_concurrent: int = 5
    eval_one_retry_max: int = 1
    judge_one_retry_max: int = 2
    max_repair_attempts: int = 2
    repair: bool = True
    dependency_failure_skips_downstream: bool = False
    save_intermediate_workspace_tar: bool = False

    def qid(self) -> str:
        return self.task_id

    @property
    def task_description(self) -> str:
        if self.task_description_for_judge:
            return self.task_description_for_judge
        if self.description:
            return self.description
        return "\n".join(f"第{item.round_id}轮：{item.prompt}" for item in self.rounds)

    @property
    def category(self) -> str:
        return self.categories[0] if self.categories else "mrccbench"

    def dependency_map(self) -> dict[int, CriticalCheck | None]:
        return {item.round_id: item.critical_check for item in self.dependencies}


class MRCCBenchInference(BaseInference):
    """Inference artifacts for one MRCCBench run."""

    response: str = ""
    workspace_tar_path: Path | None = None
    round_records_path: Path | None = None
    dependency_checks_path: Path | None = None
    round_records: list[RoundRecord] = Field(default_factory=list)
    dependency_checks: list[DependencyCheckResult] = Field(default_factory=list)
    agent_error: Error | None = None


class MRCCCheckResult(BaseModel):
    """Evaluation result for one checklist item."""

    id: int | str
    description: str
    weight: float = 1.0
    score: float | None = None
    reason: str = ""
    duration: float = 0.0
    failure_stage: str | None = None
    failure_kind: str | None = None
    extract_status: str | None = None
    eval_timeout_retries: int = 0
    eval_check_retries: int = 0


class MRCCBenchJudgement(BaseJudgement):
    """Checklist judgement and MRCCBench metadata."""

    category: str = "mrccbench"
    judge_output: str = ""
    check_results: list[MRCCCheckResult] = Field(default_factory=list)
    weighted_score: float | None = None
    response: str = ""
    total_rounds: int = 0
    round_summaries: list[dict] = Field(default_factory=list)
    dependency_summary: dict = Field(default_factory=dict)
    repair_summary: dict = Field(default_factory=dict)
