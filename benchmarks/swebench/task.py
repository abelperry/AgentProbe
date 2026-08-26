"""SWE-bench task implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from agent_probe.core.sandbox import Sandbox
from agent_probe.core.task import BaseTask
from benchmarks.swebench.models import SWEBenchInference, SWEBenchJudgement, SWEBenchQuestion
from benchmarks.swebench.official import grade_with_official_harness
from benchmarks.swebench.spec_builder import SWEBenchInferSpecBuilder

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class SWEBenchTask(BaseTask[SWEBenchQuestion, SWEBenchInference, SWEBenchJudgement]):
    async def inference(self, question: SWEBenchQuestion, ctx: EvalContext) -> SWEBenchInference:
        builder = SWEBenchInferSpecBuilder(question, ctx)
        sandbox_result = await Sandbox(builder.build()).run()
        return builder.to_inference(sandbox_result)

    async def judge(
        self,
        question: SWEBenchQuestion,
        inference_result: SWEBenchInference,
        ctx: EvalContext,
        prev_judgement: SWEBenchJudgement | None = None,
    ) -> SWEBenchJudgement:
        namespace = str(ctx.dataset_config.options.get("namespace", "swebench"))
        try:
            report = grade_with_official_harness(
                question=question,
                patch_path=inference_result.patch_path,
                test_log_path=inference_result.test_log_path,
                model_name=ctx.model_config.model_name,
                namespace=namespace,
            )
        except Exception as exc:
            logger.exception("[{}] official SWE-bench grading failed", ctx.log_tag())
            return SWEBenchJudgement(
                score=0.0,
                tests_status={"grading_error": str(exc)},
            )

        return SWEBenchJudgement(
            score=1.0 if bool(report.get("resolved", False)) else 0.0,
            tests_status=self._tests_status(report),
        )

    def collect_metrics(self, judgements: list[SWEBenchJudgement]) -> tuple[dict[str, float], int]:
        total = len(judgements)
        success_count = sum(1 for judgement in judgements if judgement.error is None)
        if total == 0:
            return {"avg_score": 0.0}, 0

        resolved = sum(
            1
            for judgement in judgements
            if judgement.error is None and judgement.score >= 1.0
        )
        return {"avg_score": resolved / total * 100}, success_count

    @staticmethod
    def _tests_status(report: dict[str, Any]) -> dict[str, Any] | None:
        tests_status = report.get("tests_status")
        if tests_status is not None:
            return tests_status
        return {
            "patch_is_None": report.get("patch_is_None"),
            "patch_exists": report.get("patch_exists"),
            "patch_successfully_applied": report.get("patch_successfully_applied"),
        }
