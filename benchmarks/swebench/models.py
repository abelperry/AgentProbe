"""SWE-bench data models."""

from __future__ import annotations

from typing import Any

from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion


class SWEBenchQuestion(BaseQuestion):
    """One SWE-bench instance in the original dataset shape."""

    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str
    patch: str = ""
    test_patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    environment_setup_commit: str = ""
    hints_text: str = ""
    created_at: str = ""

    def qid(self) -> str:
        return self.instance_id

    def to_swebench_instance(self) -> dict[str, Any]:
        """Return the dict shape expected by swebench.harness.make_test_spec."""
        return self.model_dump(mode="python")


class SWEBenchInference(BaseInference):
    """Minimal inference artifact references for one SWE-bench run."""

    patch_path: str = ""
    test_log_path: str = ""
    output: str = ""


class SWEBenchJudgement(BaseJudgement):
    """Official SWE-bench grading result."""

    score: float = 0.0
    tests_status: dict[str, Any] | None = None
