"""ZFrontendBench Q/I/J models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion


class ChecklistItem(BaseModel):
    """Normalized frontend checklist item."""

    id: int | str
    description: str = ""
    weight: float = 1.0


class FrontendCheckResult(BaseModel):
    """Evaluation result for one checklist item."""

    id: int | str
    description: str = ""
    weight: float = 1.0
    score: float | None = None
    reason: str = ""
    duration: float = 0.0


class ZFrontendBenchQuestion(BaseQuestion):
    """Question schema for ZFrontendBench records."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        validation_alias=AliasChoices("qid", "task_id"),
    )
    docker: str
    workspace_dir: str
    contexts: list[str] = Field(default_factory=list)
    unittest_script: str | None = None
    baseline_images: list[str] | None = None
    checklist: list[dict[str, Any] | str] | None = None
    test_targets: list[str] | None = None
    test_mode: Literal["file", "http"] = "http"
    task_description_for_judge: str | None = None
    # Deployment-specific; questions.jsonl carries judge_docker. Set
    # ZFRONT_JUDGE_IMAGE to supply a default for data that omits it.
    judge_docker: str = os.environ.get("ZFRONT_JUDGE_IMAGE", "")
    http_port: int = 5173
    http_build_timeout: int = 600
    eval_timeout: int = 3600
    eval_concurrent: int = 30
    categories: list[str] = Field(default_factory=list)

    def qid(self) -> str:
        return self.task_id

    @property
    def prompt(self) -> str:
        return self.contexts[0] if self.contexts else ""

    @property
    def task_description(self) -> str:
        return self.task_description_for_judge or self.prompt

    @property
    def category(self) -> str:
        return self.categories[0] if self.categories else "unknown"

    def get_checklist(self) -> list[ChecklistItem]:
        if not self.checklist:
            return []
        normalized: list[ChecklistItem] = []
        for idx, item in enumerate(self.checklist):
            if isinstance(item, str):
                normalized.append(ChecklistItem(id=idx, description=item, weight=1.0))
            else:
                normalized.append(
                    ChecklistItem(
                        id=item.get("id", idx),
                        description=str(item.get("description", "")),
                        weight=float(item.get("weight", 1.0)),
                    )
                )
        return normalized


class ZFrontendBenchInference(BaseInference):
    """Inference result with exported workspace artifact."""

    response: str = ""
    workspace_tar_path: Path | None = None


class ZFrontendBenchJudgement(BaseJudgement):
    """Checklist judgement and weighted score."""

    category: str = "unknown"
    judge_output: str = ""
    check_results: list[FrontendCheckResult] = Field(default_factory=list)
    weighted_score: float | None = None
    response: str = ""
