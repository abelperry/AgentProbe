"""ZBackendBench task implementation."""

from __future__ import annotations

import json
import re
import shlex
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from agent_probe.config import JudgeConfig
from agent_probe.core.models import Error
from agent_probe.core.sandbox import ResourceSpec, Sandbox, SandboxResult, SandboxSpec
from agent_probe.core.task import BaseTask
from benchmarks.zbackendbench.models import (
    ZBackendBenchInference,
    ZBackendBenchJudgement,
    ZBackendBenchQuestion,
)
from benchmarks.zbackendbench.prompts import CODE_QUALITY_JUDGE_PROMPT

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


REQUIRED_RUBRIC_KEYS = {
    "A1_new_abstraction",
    "A2_dependency",
    "E1_violate_ocp",
    "E2_over_design",
    "M1_diff_minimized",
    "M2_side_effect",
}


class ZBackendBenchTask(
    BaseTask[ZBackendBenchQuestion, ZBackendBenchInference, ZBackendBenchJudgement]
):
    """Backend code-agent benchmark with deterministic and rubric scoring."""

    _judge_config: JudgeConfig | None = None

    async def inference(
        self, question: ZBackendBenchQuestion, ctx: EvalContext
    ) -> ZBackendBenchInference:
        infer_dir = ctx.output_dir / "infer" / question.qid()
        tasks_dir = Path(ctx.dataset_config.data_dir) / "tasks"
        deterministic_score = 0.0
        patch_path: Path | None = None
        test_log_path: Path | None = None

        async def _setup(sb: Sandbox) -> None:
            await self._upload_environment(
                sb,
                tasks_dir / question.qid() / "environment",
                question.workspace_dir,
            )

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            nonlocal deterministic_score, patch_path, test_log_path
            if sandbox_result.error:
                return
            patch_path = await self._dump_patch(sb, question.workspace_dir, infer_dir)
            deterministic_score, test_log_path = await self._run_verifier(
                sb,
                tests_dir=tasks_dir / question.qid() / "tests",
                workspace_dir=question.workspace_dir,
                verifier_timeout=question.verifier_timeout,
                output_dir=infer_dir,
            )

        spec = SandboxSpec(
            image=question.docker_image,
            sandbox_config=ctx.sandbox_config,
            prompt=question.prompt,
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
        return self._to_inference(result, deterministic_score, patch_path, test_log_path)

    async def judge(
        self,
        question: ZBackendBenchQuestion,
        inference_result: ZBackendBenchInference,
        ctx: EvalContext,
        _prev_judgement: ZBackendBenchJudgement | None = None,
    ) -> ZBackendBenchJudgement:
        if inference_result.error:
            return ZBackendBenchJudgement(
                deterministic_score=inference_result.deterministic_score,
                error=inference_result.error,
            )

        if inference_result.deterministic_score == 0.0:
            return ZBackendBenchJudgement(
                score=0.0,
                deterministic_score=0.0,
                judge_output="deterministic check not pass, deterministic_score is 0",
            )

        judge_cfg = self._get_judge_config(ctx)
        judge_prompt = CODE_QUALITY_JUDGE_PROMPT.format(prompt=question.prompt)

        async def _setup(sb: Sandbox) -> None:
            await self._apply_patch(
                sb, question.workspace_dir, inference_result.patch_path
            )

        spec = SandboxSpec(
            image=question.docker_image,
            sandbox_config=ctx.sandbox_config,
            prompt=judge_prompt,
            agent_config=judge_cfg.agent,
            model_cfg=judge_cfg.model,
            output_dir=str(ctx.output_dir / "eval" / question.qid()),
            env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
            workspace=question.workspace_dir,
            resources=ResourceSpec(
                cpus=int(question.cpu_cores),
                memory_mb=int(question.memory_gib * 1024),
            ),
            timeout_sec=judge_cfg.model.timeout,
            on_setup=_setup,
        )
        result = await Sandbox(spec).run()
        return self._parse_judge_result(inference_result, result)

    def collect_metrics(
        self, judgements: list[ZBackendBenchJudgement]
    ) -> tuple[dict[str, float], int]:
        num_total = len(judgements)
        valid = [j for j in judgements if j.error is None]
        num_success = len(valid)
        metrics = {
            "num_total": float(num_total),
            "num_success": float(num_success),
            "average": 0.0,
            "deterministic_average": 0.0,
        }
        if not valid:
            return metrics, 0
        metrics["average"] = sum(j.score for j in valid) / len(valid) * 100
        metrics["deterministic_average"] = (
            sum(j.deterministic_score for j in valid) / len(valid) * 100
        )
        return metrics, num_success

    def _get_judge_config(self, ctx: EvalContext) -> JudgeConfig:
        if self._judge_config is None:
            self._judge_config = JudgeConfig.from_yaml(
                Path(ctx.dataset_config.get_judge_config_path())
            )
        return self._judge_config

    def _parse_judge_result(
        self,
        inference_result: ZBackendBenchInference,
        sandbox_result: SandboxResult,
    ) -> ZBackendBenchJudgement:
        raw_output = (
            sandbox_result.last_assistant.content_text
            if sandbox_result.last_assistant
            else ""
        )
        if sandbox_result.error:
            return ZBackendBenchJudgement(
                deterministic_score=inference_result.deterministic_score,
                judge_output=raw_output,
                error=sandbox_result.error,
            )

        rubric, parse_error = parse_code_quality_rubric(raw_output)
        if parse_error:
            return ZBackendBenchJudgement(
                deterministic_score=inference_result.deterministic_score,
                judge_output=raw_output,
                error=Error(code=-1, message=parse_error),
            )

        penalty = calculate_quality_penalty(rubric)
        final_score = max(inference_result.deterministic_score - penalty, 0.0)
        return ZBackendBenchJudgement(
            score=final_score,
            deterministic_score=inference_result.deterministic_score,
            judge_output=raw_output,
            code_quality_rubric=rubric,
        )

    def _to_inference(
        self,
        result: SandboxResult,
        deterministic_score: float,
        patch_path: Path | None,
        test_log_path: Path | None,
    ) -> ZBackendBenchInference:
        output = result.last_assistant.content_text if result.last_assistant else ""
        if result.error:
            return ZBackendBenchInference(
                response=output,
                deterministic_score=0.0,
                patch_path=patch_path,
                test_log_path=test_log_path,
                error=result.error,
            )
        return ZBackendBenchInference(
            response=output,
            deterministic_score=deterministic_score,
            patch_path=patch_path,
            test_log_path=test_log_path,
        )

    async def _upload_environment(
        self, sb: Sandbox, env_dir: Path, workspace_dir: str
    ) -> None:
        if not env_dir.exists():
            return
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            has_files = False
            for path in env_dir.rglob("*"):
                rel = path.relative_to(env_dir)
                if rel == Path("Dockerfile"):
                    continue
                dest = tmp_path / rel
                if path.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif path.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    has_files = True
            if has_files:
                await sb.upload_directory(tmp_path, workspace_dir)

    async def _dump_patch(self, sb: Sandbox, workspace_dir: str, output_dir: Path) -> Path:
        remote_patch = "/tmp/agentprobe_changes.patch"
        cmd = (
            f": > {remote_patch}; "
            f"find {workspace_dir} -name .git -type d -maxdepth 2 2>/dev/null "
            "| while read gitdir; do "
            'repo_dir=$(dirname "$gitdir"); '
            'git config --global --add safe.directory "$repo_dir" && '
            'cd "$repo_dir" && '
            "git add -A >/dev/null 2>&1 || true; "
            "git diff --cached -M -C --binary --full-index --unified=10 --no-color HEAD "
            f"2>/dev/null >> {remote_patch}; "
            "done"
        )
        await sb.exec_cmd(cmd)
        patch_content = await sb.read_file(remote_patch)
        patch_path = output_dir / "changes.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch_content, encoding="utf-8")
        return patch_path

    async def _run_verifier(
        self,
        sb: Sandbox,
        tests_dir: Path,
        workspace_dir: str,
        verifier_timeout: int,
        output_dir: Path,
    ) -> tuple[float, Path | None]:
        if not tests_dir.exists():
            return 0.0, None
        await sb.upload_directory(tests_dir, "/tests")
        cmd = f"cd {shlex.quote(workspace_dir)} && /bin/bash /tests/test.sh 2>&1"
        result = await sb.exec_cmd(f"bash -lc {shlex.quote(cmd)}", verifier_timeout)
        test_log_path = output_dir / "test.log"
        test_log_path.parent.mkdir(parents=True, exist_ok=True)
        test_log_path.write_text(result.stdout or "", encoding="utf-8")
        reward = await sb.exec_cmd(
            "cat /logs/verifier/reward.txt 2>/dev/null || echo '0'"
        )
        return (1.0 if reward.stdout.strip() == "1" else 0.0), test_log_path

    async def _apply_patch(
        self, sb: Sandbox, workspace_dir: str, patch_path: Path | None
    ) -> None:
        if (
            patch_path is None
            or not patch_path.exists()
            or not patch_path.read_text().strip()
        ):
            return
        remote_patch = f"{workspace_dir.rstrip('/')}/changes.patch"
        await sb.write_file(remote_patch, patch_path.read_text(encoding="utf-8"))
        cmd = (
            f"find {workspace_dir} -name .git -type d -maxdepth 2 2>/dev/null "
            "| while read gitdir; do "
            'repo_dir=$(dirname "$gitdir"); '
            'git config --global --add safe.directory "$repo_dir" && '
            f'cd "$repo_dir" && git apply {remote_patch} 2>/dev/null; '
            "git add -A >/dev/null 2>&1 || true; "
            "done"
        )
        await sb.exec_cmd(cmd)


def parse_code_quality_rubric(output: str) -> tuple[dict[str, Any], str]:
    """Extract and validate the CCBench code-quality rubric JSON."""

    json_str = _extract_json_blob(output)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return {}, f"JSON decode error: {exc}"

    if not isinstance(parsed, dict):
        return {}, "Rubric output must be a JSON object"

    missing = REQUIRED_RUBRIC_KEYS - set(parsed)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        return {}, f"Missing required dimensions in Code Quality Rubric: {missing_keys}"

    for key in REQUIRED_RUBRIC_KEYS:
        value = parsed.get(key)
        if not isinstance(value, dict):
            return {}, f"Invalid structure for {key!r}: expected dict"
        if value.get("result") not in {"pass", "fail"}:
            return {}, f"Invalid or missing result in {key!r}, must be pass or fail"
        if not isinstance(value.get("evidence"), list):
            return {}, f"Invalid or missing evidence in {key!r}, must be a list"
        if not isinstance(value.get("reason"), str):
            return {}, f"Invalid or missing reason in {key!r}, must be a string"
    return parsed, ""


def calculate_quality_penalty(rubric: dict[str, Any]) -> float:
    if not rubric:
        return 0.0
    penalty_weight = round(0.3 / len(rubric), 2)
    fail_count = sum(1 for value in rubric.values() if value.get("result") == "fail")
    return penalty_weight * fail_count


def _extract_json_blob(output: str) -> str:
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", output)
    if fence:
        return fence.group(1)
    blob = re.search(r"\{[\s\S]*\}", output)
    return blob.group(0) if blob else output
