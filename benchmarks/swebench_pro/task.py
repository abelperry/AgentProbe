"""SWE-bench Pro task implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from agent_probe.core.sandbox import Sandbox
from agent_probe.core.task import BaseTask
from benchmarks.swebench_pro.models import (
    SWEBenchProInference,
    SWEBenchProJudgement,
    SWEBenchProQuestion,
)
from benchmarks.swebench_pro.spec_builder import SWEBenchProInferSpecBuilder

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class SWEBenchProTask(
    BaseTask[SWEBenchProQuestion, SWEBenchProInference, SWEBenchProJudgement]
):
    async def inference(
        self, question: SWEBenchProQuestion, ctx: EvalContext
    ) -> SWEBenchProInference:
        builder = SWEBenchProInferSpecBuilder(question, ctx)
        sandbox_result = await Sandbox(builder.build()).run()
        return builder.to_inference(sandbox_result)

    async def judge(
        self,
        question: SWEBenchProQuestion,
        inference_result: SWEBenchProInference,
        ctx: EvalContext,
        prev_judgement: SWEBenchProJudgement | None = None,
    ) -> SWEBenchProJudgement:
        if inference_result.agent_error:
            return SWEBenchProJudgement(
                score=0.0,
                error=inference_result.agent_error,
                tests_status=None,
            )

        try:
            resolved, tests_status = self._grade(
                question,
                inference_result.output_json_path,
            )
        except Exception as exc:
            logger.exception("[{}] SWE-Pro grading failed", ctx.log_tag())
            resolved = False
            tests_status = {"grading_error": str(exc)}

        return SWEBenchProJudgement(
            score=1.0 if resolved else 0.0,
            tests_status=tests_status,
        )

    def collect_metrics(
        self, judgements: list[SWEBenchProJudgement]
    ) -> tuple[dict[str, float], int]:
        total = len(judgements)
        success_count = sum(1 for judgement in judgements if judgement.error is None)
        if total == 0:
            return {"avg_score": 0.0}, 0

        resolved = sum(1 for judgement in judgements if judgement.score >= 1.0)
        return {"avg_score": resolved / total * 100}, success_count

    @staticmethod
    def _grade(
        question: SWEBenchProQuestion,
        output_json_path: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Official SWE-Pro resolution check."""
        if not output_json_path or not Path(output_json_path).exists():
            return False, {
                "grading_error": "output.json missing (patch not applied or tests crashed)"
            }

        output = json.loads(Path(output_json_path).read_text(encoding="utf-8"))
        tests = output.get("tests", [])
        passed = {test["name"] for test in tests if test.get("status") == "PASSED"}

        f2p_missing = sorted(set(question.fail_to_pass) - passed)
        p2p_missing = sorted(set(question.pass_to_pass) - passed)
        resolved = not f2p_missing and not p2p_missing

        return resolved, {
            "resolved": resolved,
            "num_parsed_tests": len(tests),
            "num_passed": len(passed),
            "f2p_total": len(question.fail_to_pass),
            "f2p_missing_count": len(f2p_missing),
            "f2p_missing": f2p_missing,
            "p2p_total": len(question.pass_to_pass),
            "p2p_missing_count": len(p2p_missing),
            "p2p_missing": p2p_missing,
        }
