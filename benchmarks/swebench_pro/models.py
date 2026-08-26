"""SWE-bench Pro data models."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion, Error


class SWEBenchProQuestion(BaseQuestion):
    """One prepared SWE-bench Pro instance.

    The dataset is expected to embed the official per-instance run assets
    (`run_script`, `parser_py`, `env_exports`) at data-prep time.
    """

    instance_id: str
    repo: str
    base_commit: str
    dockerhub_tag: str
    prompt: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    selected_test_files: list[str]
    eval_cmd: str
    run_script: str
    parser_py: str
    env_exports: str = ""
    repo_language: str = ""
    issue_categories: list[str] = Field(default_factory=list)
    issue_specificity: list[str] = Field(default_factory=list)
    patch: str = ""
    test_patch: str = ""
    eval_timeout: int = 3600

    def qid(self) -> str:
        return self.instance_id


class SWEBenchProInference(BaseInference):
    """Inference artifacts for one SWE-bench Pro run."""

    patch_path: str = ""
    output_json_path: str = ""
    agent_error: Error | None = None


class SWEBenchProJudgement(BaseJudgement):
    """SWE-bench Pro grading result."""

    score: float = 0.0
    tests_status: dict[str, Any] | None = None
