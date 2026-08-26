"""ZClawBench data models — Question, Inference, Judgement."""

from __future__ import annotations

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion
from pydantic import BaseModel


class CheckItemSpec(BaseModel):
    """Specification for a single checklist item."""

    id: str
    description: str = ""
    method: str  # "auto_script" | "agent_judge" | "agent_pairwise_judge" | "agent_judge_with_playwright"
    weight: float = 1.0
    verify_cmd: str = ""
    files: dict[str, str] = {}


class ZClawBenchQuestion(BaseQuestion):
    task_id: str
    task_description: str = ""
    domain: str = "general"
    checklist: list[CheckItemSpec] = []
    skills: list[str] = []
    mock: list[str] = []
    files: dict[str, str] = {}
    required_files: list[str] = []
    entry_script: str = ""
    infer_image: str = ""
    eval_image: str = ""
    case_dir: str = ""

    def qid(self) -> str:
        return self.task_id


class AutoCheckResult(BaseModel):
    """Result of a single auto_script check executed during inference."""

    check_id: str
    score: float | None = None
    output: str = ""
    error: str = ""


class ZClawBenchInference(BaseInference):
    auto_script_results: list[AutoCheckResult] = []
    output: str = ""


class CheckEvalResult(BaseModel):
    """Result of evaluating a single checklist item."""

    check_id: str
    method: str
    score: float | None = None
    weight: float = 1.0
    raw_output: str = ""
    error: str = ""


class ZClawBenchJudgement(BaseJudgement):
    task_id: str = ""
    domain: str = "general"
    check_results: list[CheckEvalResult] = []
