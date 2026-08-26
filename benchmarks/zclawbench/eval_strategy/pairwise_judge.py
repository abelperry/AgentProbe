"""PairwiseJudge strategy — compares target vs baseline in a new sandbox."""

from __future__ import annotations

import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from agent_probe.config import JudgeConfig
from agent_probe.core.sandbox import Sandbox, SandboxSpec

from benchmarks.zclawbench.prompts import PAIRWISE_JUDGE_PROMPT_TEMPLATE
from benchmarks.zclawbench.models import CheckEvalResult, CheckItemSpec, ZClawBenchQuestion
from benchmarks.zclawbench.eval_strategy import EvalStrategy

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class PairwiseJudgeStrategy(EvalStrategy):
    """Creates a new sandbox for pairwise comparison against a baseline."""

    def __init__(self, item: CheckItemSpec, question: ZClawBenchQuestion, ctx: EvalContext) -> None:
        self.item = item
        self.question = question
        self.ctx = ctx
        self.default_image = os.getenv("DEFAULT_EVAL_IMAGE", "boyceyi/claw:base")

    async def evaluate(self) -> CheckEvalResult:
        judge_cfg = self._load_judge_config()
        infer_dir = self.ctx.output_dir / "infer" / self.question.qid()
        target_model_name = self.ctx.unit.model_name
        baseline_model_name = "baseline-model"
        item = self.item
        question = self.question
        ctx = self.ctx

        async def _setup(sb: Sandbox) -> None:
            # Prepare target workspace
            workspace_tar = infer_dir / "workspace.tar.gz"
            if workspace_tar.exists():
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_extract = Path(tmpdir) / "workspace"
                    tmp_extract.mkdir()
                    with tarfile.open(workspace_tar, "r:*") as tar:
                        tar.extractall(path=str(tmp_extract))
                    await sb.upload_directory(
                        tmp_extract, f"/workspace/{target_model_name}/workspace"
                    )

            # Copy target traces
            trace_dir = infer_dir / "traces"
            if trace_dir.exists():
                await sb.upload_directory(
                    trace_dir, f"/workspace/{target_model_name}/traces"
                )

            # Copy baseline files (from check_item.files)
            data_dir = Path(ctx.dataset_config.data_dir)
            tasks_base = data_dir / "tasks" / question.task_id
            for src_rel, container_path in item.files.items():
                host_path = (tasks_base / src_rel).resolve()
                if host_path.exists():
                    if host_path.is_dir():
                        await sb.upload_directory(host_path, container_path)
                    else:
                        await sb.write_file(
                            container_path,
                            host_path.read_bytes().decode("utf-8", errors="replace"),
                        )

        prompt = PAIRWISE_JUDGE_PROMPT_TEMPLATE.format(
            question=question.task_description,
            checkitem=item.description,
            target=target_model_name,
            baseline=baseline_model_name,
        )

        spec = SandboxSpec(
            image=question.eval_image or self.default_image,
            sandbox_config=ctx.sandbox_config,
            prompt=prompt,
            agent_config=judge_cfg.agent,
            model_cfg=judge_cfg.model,
            env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
            output_dir=str(ctx.output_dir / "eval" / question.qid() / item.id),
            workspace="/workspace",
            on_setup=_setup,
        )
        result = await Sandbox(spec).run()
        if result.error:
            logger.warning("pairwise_judge sandbox error for {}: {}", item.id, result.error)
            return CheckEvalResult(
                check_id=item.id,
                method="agent_pairwise_judge",
                score=None,
                weight=item.weight,
                raw_output=result.last_assistant.content_text if result.last_assistant else "",
                error=f"sandbox error: {result.error.message}",
            )
        output = result.last_assistant.content_text if result.last_assistant else ""
        score = self._parse_pairwise_output(
            output, target_model_name, baseline_model_name
        )
        return CheckEvalResult(
            check_id=item.id,
            method="agent_pairwise_judge",
            score=score,
            weight=item.weight,
            raw_output=output,
            error="" if score is not None else "Unable to parse pairwise judge output",
        )

    def _load_judge_config(self) -> JudgeConfig:
        judge_path = Path(self.ctx.dataset_config.get_judge_config_path("agent_pairwise_judge"))
        return JudgeConfig.from_yaml(judge_path)

    @staticmethod
    def _parse_pairwise_output(
        stdout: str, target_name: str, baseline_name: str
    ) -> float | None:
        r"""Parse \boxed{[[...]]} from stdout."""
        if not stdout:
            return None
        match = re.search(r"\\{1,2}boxed\{\[\[(.*?)\]\]\}", stdout)
        if not match:
            return None
        result = (
            match.group(1).replace("\\text", "").replace("{", "").replace("}", "")
        )
        if "平局" in result:
            return 0.5
        if target_name in result:
            return 1.0
        if baseline_name in result:
            return 0.0
        logger.warning("Unrecognized pairwise result: {}", result)
        return None
