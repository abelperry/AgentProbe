"""AgentProbe-native Q-I-J models for MTAC-IFBench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion, Error

DEFAULT_INFER_DOCKER = "alexgshaw/break-filter-js-from-html:20251031"


def _default_judge_docker() -> str:
    """Judge image, read per question rather than snapshotted at import.

    Judge containers are deployment-specific and no released question carries
    judge_docker, so this is normally supplied by MTACIF_JUDGE_IMAGE. Reading it
    lazily means setting the variable after this module is imported still works.
    """
    return os.environ.get("MTACIF_JUDGE_IMAGE", "")


PASS_CONCLUSION = "[[满足了该要求]]"
FAIL_CONCLUSION = "[[没有满足该要求]]"


class IFConstraint(BaseModel):
    """One instruction-following constraint and its optional checker."""

    constraint: str
    validation_code: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("constraint")
    @classmethod
    def _constraint_must_not_be_empty(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("instruction-following constraint must not be empty")
        return value


class MTACIFRound(BaseModel):
    """One ordered contestant round."""

    model_config = ConfigDict(populate_by_name=True)

    round_id: int
    # The raw dataset spells this "instruction"; the built dataset spells it
    # "prompt". Accept both so either shape validates through this model.
    prompt: str = Field(validation_alias=AliasChoices("prompt", "instruction"))
    instruction_following_checklist: list[IFConstraint] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def _prompt_must_not_be_empty(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("round prompt must not be empty")
        return value


class MTACIFBenchQuestion(BaseQuestion):
    """One MTAC-IFBench task in the released dataset shape."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    task_id: int = Field(validation_alias=AliasChoices("task_id", "qid"))
    repository_policy: str
    repository_policy_checklist: list[IFConstraint] = Field(default_factory=list)
    task_category: str = ""
    rounds: list[MTACIFRound]
    function_checklist: list[str] = Field(default_factory=list)

    docker: str = DEFAULT_INFER_DOCKER
    judge_docker: str = Field(default_factory=_default_judge_docker)
    workspace_dir: str = "/workspace"
    test_mode: Literal["http", "file"] = "http"
    http_port: int = 5173
    http_build_timeout: int = 600
    eval_concurrent: int = 5
    eval_timeout: int = 36000
    validation_code_timeout: int = 60
    judge_parse_retry_max: int = 3

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> object:
        # Only reject a bool: pydantic would otherwise coerce True to task_id 1
        # and silently collide with a real question. The value is not bounded —
        # the dataset's size is not the loader's business.
        if isinstance(value, bool):
            raise ValueError("MTAC-IFBench task_id must be an integer, not a bool")
        return value

    @field_validator("repository_policy")
    @classmethod
    def _repository_policy_must_not_be_empty(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("MTAC-IFBench repository_policy must not be empty")
        return value

    @field_validator("function_checklist")
    @classmethod
    def _function_checklist_items_must_not_be_empty(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("function checklist items must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_round_alignment(self) -> MTACIFBenchQuestion:
        if not self.rounds:
            raise ValueError("MTAC-IFBench task must contain at least one round")
        round_ids = [item.round_id for item in self.rounds]
        if len(round_ids) != len(set(round_ids)):
            raise ValueError("MTAC-IFBench round_id values must be unique")
        return self

    def qid(self) -> str:
        return str(self.task_id)

    @property
    def task_description(self) -> str:
        return "\n".join(
            f"第{round_item.round_id}轮：{round_item.prompt}" for round_item in self.rounds
        )

    @property
    def category(self) -> str:
        return self.task_category or "mtacifbench"

    def round_by_id(self, round_id: int) -> MTACIFRound | None:
        for round_item in self.rounds:
            if round_item.round_id == round_id:
                return round_item
        return None

    def checklist_for(self, round_id: int) -> list[IFConstraint]:
        """Return the round's own constraints, exactly as the dataset states them.

        Repository-policy constraints are *not* merged in here. The dataset
        repeats every policy constraint it wants scored into the round it
        applies to, so the round checklist is already the authoritative list;
        merging would double-count the repeated ones under a different
        deduplication rule than the dataset's own.
        """
        round_item = self.round_by_id(round_id)
        if round_item is None:
            return []
        return list(round_item.instruction_following_checklist)

    def validation_codes_for(self, round_id: int) -> list[str]:
        return [item.validation_code for item in self.checklist_for(round_id)]

    @property
    def constraint_count(self) -> int:
        return sum(len(item.instruction_following_checklist) for item in self.rounds)


class RoundRecord(BaseModel):
    """Persisted per-round inference summary."""

    kind: Literal["main"] = "main"
    round_index: int
    round_id: int
    attempt: int = 0
    prompt: str = ""
    result_response: str = ""
    result_excerpt: str = ""
    material_ref: str = ""


class MTACIFBenchInference(BaseInference):
    """Inference artifacts for one MTAC-IFBench run."""

    response: str = ""
    workspace_tar_path: Path | None = None
    round_records: list[RoundRecord] = Field(default_factory=list)
    material_dir: Path | None = None
    agent_error: Error | None = None


class IFCheckResult(BaseModel):
    """Verdict for one instruction-following requirement."""

    index: int
    requirement: str
    analysis: str = ""
    conclusion: Literal["[[满足了该要求]]", "[[没有满足该要求]]"]
    source: Literal["validation_code", "judge"] = "judge"

    @property
    def passed(self) -> bool:
        return self.conclusion == PASS_CONCLUSION


class IFRoundResult(BaseModel):
    """Complete instruction-following judgement for one round."""

    round_id: int
    passed: bool = False
    parse_failed: bool = False
    summary: str = ""
    symptoms: str = ""
    check_results: list[IFCheckResult] = Field(default_factory=list)
    raw_output_excerpt: str = ""
    result_ref: str = ""


class FunctionCheckResult(BaseModel):
    """Verdict for one function-checklist item."""

    id: int | str
    description: str
    score: float | None = None
    reason: str = ""
    evaluation_error: Error | None = None
    duration: float = 0.0

    @field_validator("score")
    @classmethod
    def _score_must_be_binary_or_missing(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if normalized not in {0.0, 1.0}:
            raise ValueError("function checklist score must be 0, 1, or null")
        return normalized


class MTACIFBenchJudgement(BaseJudgement):
    """Instruction-following judgement plus optional function results."""

    category: str = "mtacifbench"
    instruction_following_checks: list[IFRoundResult] = Field(default_factory=list)
    instruction_following_score: float = 0.0
    total_rounds: int = 0
    round_summaries: list[dict[str, object]] = Field(default_factory=list)
    response: str = ""
    checks: list[FunctionCheckResult] = Field(default_factory=list)
    function_score: float = 0.0
    function_checklist_skipped: bool = True
    build_success: bool | None = None

    @model_validator(mode="after")
    def _invalidate_incomplete_function_judgement(self) -> MTACIFBenchJudgement:
        if self.error is not None or self.function_checklist_skipped:
            return self
        if any(item.score is None or item.evaluation_error is not None for item in self.checks):
            self.error = Error(code=-1, message="function checklist evaluation is incomplete")
            return self
        expected_score = sum(float(item.score or 0.0) for item in self.checks) / (
            len(self.checks) or 1
        )
        if abs(self.function_score - expected_score) > 1e-9:
            self.error = Error(code=-1, message="function checklist score is inconsistent")
        return self
