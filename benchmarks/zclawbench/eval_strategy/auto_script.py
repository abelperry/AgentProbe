"""AutoScript strategy — reads scores from inference results, no new sandbox."""

from __future__ import annotations

from benchmarks.zclawbench.models import CheckEvalResult, CheckItemSpec, ZClawBenchInference
from benchmarks.zclawbench.eval_strategy import EvalStrategy


class AutoScriptStrategy(EvalStrategy):
    """Reads auto_script scores already collected during inference."""

    def __init__(self, item: CheckItemSpec, inference_result: ZClawBenchInference) -> None:
        self.item = item
        self.inference_result = inference_result

    async def evaluate(self) -> CheckEvalResult:
        for r in self.inference_result.auto_script_results:
            if r.check_id == self.item.id:
                return CheckEvalResult(
                    check_id=self.item.id,
                    method="auto_script",
                    score=r.score,
                    weight=self.item.weight,
                    raw_output=r.output,
                    error=r.error,
                )
        return CheckEvalResult(
            check_id=self.item.id,
            method="auto_script",
            error="Not found in inference results",
        )
