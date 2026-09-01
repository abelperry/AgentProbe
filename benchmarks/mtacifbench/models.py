"""MTACIFBench data models for AgentProbe.

MTACIFBench measures multi-turn agentic-coding *instruction following*: the
agent works several rounds in one workspace and one conversation, and each
round is scored against that round's constraint checklist.

The JSONL is expected to be the shape published on the Hugging Face Hub (see
``benchmarks/mtacifbench/README.md``). Tolerance for looser input lives in
``scripts/build_mtacifbench_dataset.py``, not here, so a malformed row fails at
load time rather than mid-run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion, Error

DEFAULT_INFER_DOCKER = "alexgshaw/break-filter-js-from-html:20251031"
# No default image: judge containers are deployment-specific, and baking a
# private registry path in here makes the benchmark unusable elsewhere.
# Questions carry judge_docker (the converter always writes it); set
# MTACIF_JUDGE_IMAGE to override for data that omits it.
DEFAULT_JUDGE_DOCKER = os.environ.get("MTACIF_JUDGE_IMAGE", "")


class IFConstraint(BaseModel):
    """One instruction-following constraint for one round.

    ``validation_code`` is the dataset-supplied deterministic checker, stored on
    the constraint itself. Keeping it here rather than in a parallel array
    aligned by index removes a whole class of silent misalignment.
    """

    constraint: str
    validation_code: str = ""
    tags: list[str] = Field(default_factory=list)
    main_id: int | None = None
    type_id: int | None = None


class MTACIFRound(BaseModel):
    """One requirement round.

    ``instruction_following_checklist`` is self-contained: constraints may be
    replaced or reordered between rounds (a later round can forbid what an
    earlier round required), so rounds are never merged or inherited.
    """

    round_id: int
    prompt: str
    instruction_following_checklist: list[IFConstraint] = Field(default_factory=list)


class FunctionChecklistItem(BaseModel):
    """One user-visible functional requirement for the finished workspace.

    Unlike an ``IFConstraint`` these are about the product, not the process:
    "hovering a status light shows a tooltip". They are scored once per task
    against the built and served application, never per round.
    """

    checklist_id: int | str
    description: str
    weight: float = 1.0

    @field_validator("description")
    @classmethod
    def _reject_empty_description(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("function checklist description must not be empty")
        return text


class MTACIFBenchQuestion(BaseQuestion):
    """One prepared MTACIFBench instance."""

    task_id: str
    docker: str = DEFAULT_INFER_DOCKER
    judge_docker: str = DEFAULT_JUDGE_DOCKER
    workspace_dir: str = "/workspace"
    description: str = ""
    system_prompt: str = ""
    rounds: list[MTACIFRound]
    # Scored only when the judge config enables functional evaluation. Empty for
    # a task the dataset never wrote functional requirements for -- which is a
    # coverage fact, not a zero.
    function_checklist: list[FunctionChecklistItem] = Field(default_factory=list)
    task_category: str = ""
    # Functional evaluation serves the built workspace and drives a browser.
    # "file" is the fallback for a project with no HTTP entry point at all.
    test_mode: Literal["http", "file"] = "http"
    http_port: int = 5173
    http_build_timeout: int = 600
    task_description_for_judge: str | None = None
    categories: list[str] = Field(default_factory=lambda: ["mtacifbench", "agent", "multi_round"])
    eval_timeout: int = 3600
    eval_concurrent: int = 5
    judge_parse_retry_max: int = 3
    validation_code_timeout: int = 60

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
        return self.categories[0] if self.categories else "mtacifbench"

    def round_by_id(self, round_id: int) -> MTACIFRound | None:
        for item in self.rounds:
            if item.round_id == round_id:
                return item
        return None

    def checklist_for(self, round_id: int) -> list[IFConstraint]:
        item = self.round_by_id(round_id)
        return list(item.instruction_following_checklist) if item else []


class RoundRecord(BaseModel):
    """Persisted per-round inference summary."""

    kind: Literal["main"] = "main"
    round_index: int
    round_id: int
    attempt: int = 0
    prompt: str = ""
    # Full final reply — the input deterministic validators and the judge score.
    result_response: str = ""
    # Bounded copy for logs and diagnostics only.
    result_excerpt: str = ""
    material_ref: str = ""


class MTACIFBenchInference(BaseInference):
    """Inference artifacts for one MTACIFBench run."""

    response: str = ""
    workspace_tar_path: Path | None = None
    round_records: list[RoundRecord] = Field(default_factory=list)
    material_dir: Path | None = None
    agent_error: Error | None = None


class IFCheckResult(BaseModel):
    """Verdict for one constraint."""

    index: int
    requirement: str
    analysis: str = ""
    conclusion: str
    source: Literal["validation_code", "judge"] = "judge"

    @property
    def passed(self) -> bool:
        return self.conclusion == "[[满足了该要求]]"


class IFRoundResult(BaseModel):
    """Verdict for one round."""

    round_id: int
    passed: bool = False
    parse_failed: bool = False
    summary: str = ""
    symptoms: str = ""
    check_results: list[IFCheckResult] = Field(default_factory=list)
    # Bounded; the full judge text lives under eval/{qid}/instruction_following/.
    raw_output_excerpt: str = ""
    result_ref: str = ""


class FunctionCheckResult(BaseModel):
    """Verdict for one function checklist item.

    ``score`` is deliberately nullable and deliberately binary. ``None`` means
    the check never produced a verdict (build failed, browser died) and must be
    left out of the denominator; a partial credit value would let a broken
    environment read as a half-working product.
    """

    checklist_id: int | str
    description: str
    weight: float = 1.0
    score: float | None = None
    reason: str = ""
    evaluation_error: Error | None = None

    @field_validator("score")
    @classmethod
    def _reject_partial_credit(cls, value: float | None) -> float | None:
        if value is None:
            return None
        score = float(value)
        if score not in {0.0, 1.0}:
            raise ValueError("function checklist score must be 0, 1, or null")
        return score

    @property
    def passed(self) -> bool:
        return self.score == 1.0


class MTACIFBenchJudgement(BaseJudgement):
    """Instruction-following judgement for one task, plus optional functional."""

    category: str = "mtacifbench"
    instruction_following_checks: list[IFRoundResult] = Field(default_factory=list)
    instruction_following_score: float = 0.0
    total_rounds: int = 0
    round_summaries: list[dict[str, object]] = Field(default_factory=list)
    response: str = ""

    # Functional evaluation. ``function_checklist_skipped`` stays True unless
    # the run actually built and inspected the workspace, so a disabled switch
    # is never mistaken for "the product failed every check".
    function_checklist_skipped: bool = True
    function_checks: list[FunctionCheckResult] = Field(default_factory=list)
    function_score: float = 0.0
    build_success: bool | None = None
