"""ZClawBenchTask — thin orchestration layer for inference, judge, and metrics."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from agent_probe.core.sandbox import Sandbox
from agent_probe.core.task import BaseTask

from benchmarks.zclawbench.models import ZClawBenchInference, ZClawBenchJudgement, ZClawBenchQuestion
from benchmarks.zclawbench.spec_builder import InferSpecBuilder

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class ZClawBenchTask(
    BaseTask[ZClawBenchQuestion, ZClawBenchInference, ZClawBenchJudgement]
):
    async def inference(
        self, question: ZClawBenchQuestion, ctx: EvalContext
    ) -> ZClawBenchInference:
        builder = InferSpecBuilder(question, ctx)
        sandbox_result = await Sandbox(builder.build()).run()
        return builder.to_inference(sandbox_result)

    async def judge(
        self,
        question: ZClawBenchQuestion,
        inference_result: ZClawBenchInference,
        ctx: EvalContext,
        prev_judgement: ZClawBenchJudgement | None = None,
    ) -> ZClawBenchJudgement:
        if inference_result.error:
            return ZClawBenchJudgement(
                task_id=question.task_id,
                domain=question.domain,
                error=inference_result.error,
            )

        from benchmarks.zclawbench.eval_strategy import create_eval_strategy
        from benchmarks.zclawbench.models import CheckEvalResult

        from agent_probe.core.models import Error

        # Build cache map from previous judgement for check-item level reuse
        prev_map: dict[str, CheckEvalResult] = {}
        if prev_judgement:
            prev_map = {r.check_id: r for r in prev_judgement.check_results}

        async def _eval_item(item):
            # Reuse successful check result from previous judgement
            cached = prev_map.get(item.id)
            if cached and not cached.error:
                return cached

            try:
                strategy = create_eval_strategy(
                    item.method, item, question, inference_result, ctx
                )
            except ValueError:
                return CheckEvalResult(
                    check_id=item.id,
                    method=item.method,
                    error=f"Unknown method: {item.method}",
                )
            return await strategy.evaluate()

        check_results = await asyncio.gather(
            *[_eval_item(item) for item in question.checklist]
        )

        has_error = any(r.error for r in check_results)
        return ZClawBenchJudgement(
            task_id=question.task_id,
            domain=question.domain,
            check_results=check_results,
            error=Error(code=-1, message="some check items failed") if has_error else None,
        )

    def collect_metrics(
        self, judgements: list[ZClawBenchJudgement]
    ) -> tuple[dict[str, float], int]:
        """Compute ISR, CSR, WCSR, Macro_CSR, Macro_WCSR (overall + per-domain)."""
        success_count = sum(1 for j in judgements if j.error is None)
        # Build per-task scores
        task_scores = []
        for j in judgements:
            if j.error:
                continue
            checks = [
                {
                    "check_id": cr.check_id,
                    "score": cr.score,
                    "weight": cr.weight,
                    "method": cr.method,
                }
                for cr in j.check_results
            ]
            task_scores.append(
                _TaskScore(task_id=j.task_id, domain=j.domain, check_scores=checks)
            )

        # Only consider tasks where all checks are scored
        valid = [t for t in task_scores if t.all_scored()]
        if not valid:
            return {"ISR": 0.0, "CSR": 0.0, "WCSR": 0.0, "Macro_CSR": 0.0, "Macro_WCSR": 0.0}, success_count

        # ISR
        isr = sum(t.isr() for t in valid) / len(valid)

        # CSR (flat unweighted)
        all_checks = [c for t in valid for c in t.check_scores]
        csr = sum(c["score"] for c in all_checks) / len(all_checks) if all_checks else 0.0

        # WCSR (flat weighted)
        total_weight = sum(c["weight"] for c in all_checks)
        wcsr = (
            sum(c["score"] * c["weight"] for c in all_checks) / total_weight
            if total_weight > 0
            else 0.0
        )

        # Macro CSR / WCSR
        macro_csr = sum(t.macro_csr() for t in valid) / len(valid)
        macro_wcsr = sum(t.macro_wcsr() for t in valid) / len(valid)

        metrics: dict[str, float] = {
            "ISR": round(isr, 4),
            "CSR": round(csr, 4),
            "WCSR": round(wcsr, 4),
            "Macro_CSR": round(macro_csr, 4),
            "Macro_WCSR": round(macro_wcsr, 4),
        }

        # Per-domain breakdown
        by_domain: dict[str, list[_TaskScore]] = defaultdict(list)
        for t in valid:
            by_domain[t.domain].append(t)

        for domain, tasks in by_domain.items():
            d_checks = [c for t in tasks for c in t.check_scores]
            d_weight = sum(c["weight"] for c in d_checks)
            metrics[f"{domain}_ISR"] = round(
                sum(t.isr() for t in tasks) / len(tasks), 4
            )
            metrics[f"{domain}_CSR"] = round(
                sum(c["score"] for c in d_checks) / len(d_checks) if d_checks else 0, 4
            )
            metrics[f"{domain}_WCSR"] = round(
                sum(c["score"] * c["weight"] for c in d_checks) / d_weight
                if d_weight > 0
                else 0,
                4,
            )
            metrics[f"{domain}_Macro_CSR"] = round(
                sum(t.macro_csr() for t in tasks) / len(tasks), 4
            )
            metrics[f"{domain}_Macro_WCSR"] = round(
                sum(t.macro_wcsr() for t in tasks) / len(tasks), 4
            )

        return metrics, success_count


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------


class _TaskScore:
    """Lightweight aggregator for per-task check scores."""

    def __init__(
        self, task_id: str, domain: str, check_scores: list[dict]
    ) -> None:
        self.task_id = task_id
        self.domain = domain
        self.check_scores = check_scores

    def all_scored(self) -> bool:
        return all(c["score"] is not None for c in self.check_scores)

    def isr(self) -> int:
        if not self.all_scored():
            return 0
        return 1 if all(c["score"] == 1 for c in self.check_scores) else 0

    def macro_csr(self) -> float:
        scored = [c for c in self.check_scores if c["score"] is not None]
        if not scored:
            return 0.0
        return sum(c["score"] for c in scored) / len(scored)

    def macro_wcsr(self) -> float:
        scored = [c for c in self.check_scores if c["score"] is not None]
        if not scored:
            return 0.0
        total_weight = sum(c["weight"] for c in scored)
        if total_weight == 0:
            return 0.0
        return sum(c["score"] * c["weight"] for c in scored) / total_weight
