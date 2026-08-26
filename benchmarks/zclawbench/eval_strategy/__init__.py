"""Eval strategy interface and factory for ZClawBench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from benchmarks.zclawbench.models import CheckEvalResult, CheckItemSpec, ZClawBenchInference, ZClawBenchQuestion

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class EvalStrategy(ABC):
    """Base class for evaluation strategies."""

    @abstractmethod
    async def evaluate(self) -> CheckEvalResult: ...


def create_eval_strategy(
    method: str,
    item: CheckItemSpec,
    question: ZClawBenchQuestion,
    inference_result: ZClawBenchInference,
    ctx: EvalContext,
) -> EvalStrategy:
    """Factory: build the right EvalStrategy for *method*."""
    from benchmarks.zclawbench.eval_strategy.auto_script import AutoScriptStrategy
    from benchmarks.zclawbench.eval_strategy.agent_judge import AgentJudgeStrategy
    from benchmarks.zclawbench.eval_strategy.pairwise_judge import PairwiseJudgeStrategy
    from benchmarks.zclawbench.eval_strategy.playwright_judge import PlaywrightJudgeStrategy

    if method == "auto_script":
        return AutoScriptStrategy(item=item, inference_result=inference_result)

    if method == "agent_judge":
        return AgentJudgeStrategy(item=item, question=question, ctx=ctx)

    if method == "agent_pairwise_judge":
        return PairwiseJudgeStrategy(item=item, question=question, ctx=ctx)

    if method == "agent_judge_with_playwright":
        return PlaywrightJudgeStrategy(item=item, question=question, ctx=ctx)

    raise ValueError(f"Unknown eval method: {method!r}")
