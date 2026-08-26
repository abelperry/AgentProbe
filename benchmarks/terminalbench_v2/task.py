"""TerminalBench v2 task implementation."""

from __future__ import annotations

import json
import re
import shlex
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from agent_probe.core.sandbox import ResourceSpec, Sandbox, SandboxResult, SandboxSpec
from agent_probe.core.task import BaseTask
from benchmarks.terminalbench_v2.models import (
    TerminalBenchV2Inference,
    TerminalBenchV2Judgement,
    TerminalBenchV2Question,
)

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class TerminalBenchV2Task(
    BaseTask[TerminalBenchV2Question, TerminalBenchV2Inference, TerminalBenchV2Judgement]
):
    """Run the agent in the task container and score with bundled tests."""

    async def inference(
        self, question: TerminalBenchV2Question, ctx: EvalContext
    ) -> TerminalBenchV2Inference:
        infer_dir = ctx.output_dir / "infer" / question.qid()
        tasks_dir = Path(ctx.dataset_config.data_dir) / "tasks"
        verifier_score = 0.0
        test_log_path: Path | None = None
        tests_status: dict | None = None

        async def _setup(sb: Sandbox) -> None:
            await self._upload_environment(
                sb,
                tasks_dir / question.qid() / "environment",
                question.workspace_dir,
            )

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            nonlocal verifier_score, test_log_path, tests_status
            if sandbox_result.error:
                return
            verifier_score, test_log_path, tests_status = await self._run_verifier(
                sb,
                tests_dir=tasks_dir / question.qid() / "tests",
                workspace_dir=question.workspace_dir,
                verifier_timeout=question.verifier_timeout,
                output_dir=infer_dir,
            )

        spec = SandboxSpec(
            image=question.docker_image,
            sandbox_config=ctx.sandbox_config,
            prompt=self._agent_prompt(question),
            agent_config=ctx.agent_config,
            model_cfg=ctx.model_config,
            output_dir=str(infer_dir),
            env_vars=ctx.agent_config.envs if ctx.agent_config else {},
            workspace=question.workspace_dir,
            resources=ResourceSpec(
                cpus=int(question.cpu_cores),
                memory_mb=int(question.memory_gib * 1024),
            ),
            timeout_sec=max(ctx.model_config.timeout, question.verifier_timeout + 300),
            on_setup=_setup,
            on_complete=_complete,
        )
        result = await Sandbox(spec).run()
        return self._to_inference(result, verifier_score, test_log_path, tests_status)

    async def judge(
        self,
        question: TerminalBenchV2Question,
        inference_result: TerminalBenchV2Inference,
        ctx: EvalContext,
        prev_judgement: TerminalBenchV2Judgement | None = None,
    ) -> TerminalBenchV2Judgement:
        del question, ctx, prev_judgement
        return TerminalBenchV2Judgement(
            score=inference_result.score,
        )

    def collect_metrics(
        self, judgements: list[TerminalBenchV2Judgement]
    ) -> tuple[dict[str, float], int]:
        num_total = len(judgements)
        valid = [j for j in judgements if j.error is None]
        num_success = len(valid)
        if num_total == 0:
            return {"average": 0.0}, 0
        resolved = sum(1 for j in valid if j.score >= 1.0)
        return {"average": resolved / num_total * 100}, num_success

    def _to_inference(
        self,
        result: SandboxResult,
        score: float,
        test_log_path: Path | None,
        tests_status: dict | None,
    ) -> TerminalBenchV2Inference:
        output = result.last_assistant.content_text if result.last_assistant else ""
        if result.error:
            return TerminalBenchV2Inference(
                response=output,
                score=0.0,
                test_log_path=test_log_path,
                tests_status=tests_status,
                error=result.error,
            )
        return TerminalBenchV2Inference(
            response=output,
            score=score,
            test_log_path=test_log_path,
            tests_status=tests_status,
        )

    def _agent_prompt(self, question: TerminalBenchV2Question) -> str:
        return (
            "Work autonomously. Complete the task end-to-end; do not stop midway "
            "and do not ask the user for confirmation.\n\n"
            f"{question.prompt}"
        )

    async def _upload_environment(
        self, sb: Sandbox, env_dir: Path, workspace_dir: str
    ) -> None:
        """Upload task environment files, excluding Dockerfile."""

        if not env_dir.exists():
            return
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            has_files = False
            for path in env_dir.rglob("*"):
                # Exclude any Dockerfile inside the environment directory
                if path.name == "Dockerfile" or "Dockerfile" in path.parts:
                    continue
                rel = path.relative_to(env_dir)
                dest = tmp_path / rel
                if path.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif path.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    has_files = True
            if has_files:
                await sb.upload_directory(tmp_path, workspace_dir)

    async def _run_verifier(
        self,
        sb: Sandbox,
        tests_dir: Path,
        workspace_dir: str,
        verifier_timeout: int,
        output_dir: Path,
    ) -> tuple[float, Path | None, dict | None]:
        if not tests_dir.exists():
            return 0.0, None, None

        await sb.upload_directory(tests_dir, "/tests")
        cmd = f"cd {shlex.quote(workspace_dir)} && /bin/bash /tests/test.sh 2>&1"
        result = await sb.exec_cmd(f"bash -lc {shlex.quote(cmd)}", verifier_timeout)

        test_log_path = output_dir / "test.log"
        test_log_path.parent.mkdir(parents=True, exist_ok=True)
        test_log_path.write_text(result.stdout or "", encoding="utf-8")
        tests_status = await self._collect_tests_status(sb, result.stdout or "")

        # Use read_file API instead of remote cat command for robustness
        score = 0.0
        try:
            reward_text = await sb.read_file("/logs/verifier/reward.txt")
            score = float(reward_text.strip())
        except Exception:
            score = 0.0

        return score, test_log_path, tests_status

    async def _collect_tests_status(self, sb: Sandbox, log_text: str) -> dict | None:
        # Use read_file API instead of remote cat command for robustness
        ctrf_text = ""
        try:
            ctrf_text = await sb.read_file("/logs/verifier/ctrf.json")
            ctrf_text = ctrf_text.strip()
        except Exception:
            pass

        if ctrf_text:
            try:
                tests = (json.loads(ctrf_text).get("results") or {}).get("tests") or []
                buckets: dict[str, list[str]] = {}
                for test in tests:
                    status = (test.get("status") or "other").lower()
                    name = test.get("name") or ""
                    if name:
                        buckets.setdefault(status, []).append(name)
                if buckets:
                    return buckets
            except json.JSONDecodeError:
                pass

        buckets: dict[str, list[str]] = {}
        line_re = re.compile(
            r"^(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\s+"
            r"(?:\[\d+\]\s+)?"
            r"(\S+)"
        )
        for raw in log_text.splitlines():
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw).lstrip()
            match = line_re.match(line)
            if match:
                buckets.setdefault(match.group(1).lower(), []).append(
                    match.group(2).rstrip(":")
                )
        return buckets or None
