"""PlaywrightJudge strategy — Claude Code CLI + Playwright MCP in a browser-enabled sandbox."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from agent_probe.config import JudgeConfig
from agent_probe.core.sandbox import Sandbox, SandboxResult, SandboxSpec

from benchmarks.zclawbench.prompts import PLAYWRIGHT_JUDGE_PROMPT_TEMPLATE
from benchmarks.zclawbench.models import CheckEvalResult, CheckItemSpec, ZClawBenchQuestion
from benchmarks.zclawbench.eval_strategy import EvalStrategy

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class PlaywrightJudgeStrategy(EvalStrategy):
    """Evaluates UI checklist items using Claude Code CLI + Playwright MCP."""

    def __init__(self, item: CheckItemSpec, question: ZClawBenchQuestion, ctx: EvalContext) -> None:
        self.item = item
        self.question = question
        self.ctx = ctx
        # Deployment-specific: set DEFAULT_PLAYWRIGHT_IMAGE for your registry.
        self.default_image = os.getenv("DEFAULT_PLAYWRIGHT_IMAGE", "")
        self.max_retries = 3

    async def evaluate(self) -> CheckEvalResult:
        judge_cfg = self._load_judge_config()
        infer_dir = self.ctx.output_dir / "infer" / self.question.qid()
        item = self.item
        question = self.question

        async def _setup(sb: Sandbox) -> None:
            # Upload workspace.tar.gz and extract to /workspace
            workspace_tar = infer_dir / "workspace.tar.gz"
            if workspace_tar.exists():
                await sb.exec_cmd("mkdir -p /workspace")
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_extract = Path(tmpdir) / "workspace"
                    tmp_extract.mkdir()
                    with tarfile.open(workspace_tar, "r:*") as tar:
                        tar.extractall(path=str(tmp_extract))
                    await sb.upload_directory(tmp_extract, "/workspace")
            else:
                logger.warning("[{}] No workspace.tar.gz found at {}", question.task_id, infer_dir)

        # Retry state shared with on_nextround hook
        attempt = 0
        score: int | None = None
        raw_output = ""

        async def _on_nextround(sb: Sandbox, sandbox_result: SandboxResult) -> str | None:
            nonlocal attempt, score, raw_output
            attempt += 1
            la = sandbox_result.last_assistant
            output = la.content_text if la else ""
            raw_output = output
            score = self._parse_playwright_output(output)
            if score is not None:
                return None  # success, stop
            if attempt >= self.max_retries:
                logger.warning(
                    "playwright_judge all {}/{} attempts failed for {}",
                    attempt, self.max_retries, item.id,
                )
                return None  # exhausted retries, stop
            logger.warning(
                "playwright_judge attempt {}/{} parse failed for {}, retrying",
                attempt, self.max_retries, item.id,
            )
            return prompt  # retry with same prompt

        prompt = PLAYWRIGHT_JUDGE_PROMPT_TEMPLATE.format(
            task_description=question.task_description,
            workspace_path="/workspace",
            checklist_item_description=item.description,
        )

        spec = SandboxSpec(
            image=self.default_image,
            sandbox_config=self.ctx.sandbox_config,
            prompt=prompt,
            agent_config=judge_cfg.agent,
            model_cfg=judge_cfg.model,
            env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
            output_dir=str(self.ctx.output_dir / "eval" / question.qid() / item.id),
            workspace="/workspace",
            on_setup=_setup,
            on_nextround=_on_nextround,
        )

        result = await Sandbox(spec).run()
        logger.debug("[{}/{}] playwright judge score: {}", question.task_id, item.id, score)
        if result.error:
            logger.warning("playwright_judge sandbox error for {}: {}", item.id, result.error)
            return CheckEvalResult(
                check_id=item.id,
                method="agent_judge_with_playwright",
                score=None,
                weight=item.weight,
                raw_output=raw_output,
                error=f"sandbox error: {result.error.message}",
            )
        return CheckEvalResult(
            check_id=item.id,
            method="agent_judge_with_playwright",
            score=score,
            weight=item.weight,
            raw_output=raw_output,
            error="" if score is not None else "Unable to parse playwright judge output",
        )

    def _load_judge_config(self) -> JudgeConfig:
        judge_path = Path(self.ctx.dataset_config.get_judge_config_path("agent_judge_with_playwright"))
        return JudgeConfig.from_yaml(judge_path)

    @staticmethod
    def _parse_playwright_output(output: str) -> int | None:
        """Parse playwright judge output for pass/fail keywords."""
        if not output:
            return None
        if "该项目不符合要求" in output or "该项目不满足要求" in output:
            return 0
        if "该项目符合要求" in output or "该项目满足要求" in output:
            return 1
        return None
