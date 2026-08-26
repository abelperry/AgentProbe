"""BaseTask — the generic Q-I-J evaluation interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from agent_probe.core.models import I, J, Q

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class BaseTask(Generic[Q, I, J]):
    """Stateless service processor for a single benchmark.

    One ``BaseTask`` subclass corresponds to one benchmark dataset.  It
    defines three strongly-typed data structures (Q, I, J) and the logic
    to drive inference, judgement, and metric aggregation.

    **Important**: Task instances are singletons / reused across questions.
    Per-question state MUST live in local variables captured by closures,
    never as instance attributes.
    """

    async def inference(self, question: Q, ctx: EvalContext) -> I:
        """Run the agent on *question* inside a sandbox and return the Inference result."""
        raise NotImplementedError

    async def judge(
        self,
        question: Q,
        inference_result: I,
        ctx: EvalContext,
        prev_judgement: J | None = None,
    ) -> J:
        """Score the inference result against the ground truth.

        Args:
            prev_judgement: If provided, the previous (incomplete) judgement.
                Subclasses may use this to skip already-successful sub-evaluations
                and only re-run failed ones.
        """
        raise NotImplementedError

    def collect_metrics(self, judgements: list[J]) -> tuple[dict[str, float], int]:
        """Aggregate per-question Judgements into benchmark-level metrics.

        Returns:
            (scores, success_count) where success_count is the number of
            judgements without errors.
        """
        raise NotImplementedError
