"""InferSpecBuilder — encapsulates all hook logic for inference."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from agent_probe.core.sandbox import OnCompleteHook, OnSetupHook, SandboxSpec, SandboxResult, Sandbox
from agent_probe.utils.json_extract import extract_json_from_text

from benchmarks.zclawbench.models import AutoCheckResult, ZClawBenchInference, ZClawBenchQuestion

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


class InferSpecBuilder:
    """Builds a SandboxSpec for the inference stage and collects auto_script results."""

    def __init__(self, question: ZClawBenchQuestion, ctx: EvalContext) -> None:
        self.question = question
        self.ctx = ctx
        self._auto_results: list[AutoCheckResult] = []
        self.default_image = os.getenv("DEFAULT_INFER_IMAGE", "")

    def build(self) -> SandboxSpec:
        infer_dir = self.ctx.output_dir / "infer" / self.question.qid()
        tag = self.ctx.log_tag()
        logger.info("[{}] Building sandbox: image={}", tag, self.question.infer_image or self.default_image)
        return SandboxSpec(
            image=self.question.infer_image or self.default_image,
            sandbox_config=self.ctx.sandbox_config,
            prompt=self.question.task_description,
            agent_config=self.ctx.agent_config,
            model_cfg=self.ctx.model_config,
            output_dir=str(infer_dir),
            env_vars=self.ctx.agent_config.envs if self.ctx.agent_config else {},
            workspace="/workspace",
            on_setup=self._make_setup_hook(),
            on_complete=self._make_complete_hook(infer_dir),
        )

    @property
    def auto_results(self) -> list[AutoCheckResult]:
        return self._auto_results

    def to_inference(self, sandbox_result: SandboxResult) -> ZClawBenchInference:
        """Convert sandbox result + auto_results into a ZClawBenchInference."""
        la = sandbox_result.last_assistant
        return ZClawBenchInference(
            auto_script_results=self._auto_results,
            output=la.content_text if la else "",
            error=sandbox_result.error,
        )

    # ------------------------------------------------------------------
    # Setup hook
    # ------------------------------------------------------------------

    def _make_setup_hook(self) -> OnSetupHook:
        question = self.question
        ctx = self.ctx
        data_dir = Path(ctx.dataset_config.data_dir)

        async def _setup(sb: Sandbox) -> None:
            tag = ctx.log_tag()
            logger.info("[{}] Setup: files={}, skills={}, mocks={}",
                        tag, len(question.files), len(question.skills), len(question.mock))
            # 1. Upload task files
            tasks_base = data_dir / "tasks" / question.task_id
            for src_rel, container_path in question.files.items():
                host_path = (tasks_base / src_rel).resolve()
                if host_path.is_dir():
                    await sb.upload_directory(host_path, container_path)
                elif host_path.is_file():
                    content = host_path.read_bytes()
                    await sb.write_file(
                        container_path,
                        content.decode("utf-8", errors="replace"),
                    )

            # 2. Upload skills
            skills_base = data_dir / "skills"
            for skill_name in question.skills:
                skill_dir = skills_base / skill_name
                if not skill_dir.exists():
                    logger.warning("[{}] Skill '{}' not found at {}", tag, skill_name, skill_dir)
                await sb.upload_directory(
                    skill_dir, f"/workspace/skills/{skill_name}"
                )
                logger.debug("[{}] Copied skill {} -> /workspace/skills/{}", tag, skill_name, skill_name)

            # 3. Upload mocks
            mocks_base = data_dir / "mocks"
            for mock_name in question.mock:
                mock_dir = mocks_base / mock_name
                if not mock_dir.exists():
                    logger.warning("[{}] Mock '{}' not found at {}", tag, mock_name, mock_dir)
                await sb.upload_directory(
                    mock_dir, f"/app/mock/{mock_name}"
                )
                logger.debug("[{}] Copied mock {} -> /app/mock/{}", tag, mock_name, mock_name)

            # 4. Run entry_script + extract env vars
            if question.entry_script:
                await sb.exec_cmd(question.entry_script)
                overrides = await _extract_env_overrides(sb, question.entry_script)
                if overrides:
                    sb.env_vars.update(overrides)

        return _setup

    # ------------------------------------------------------------------
    # Complete hook
    # ------------------------------------------------------------------

    def _make_complete_hook(self, infer_dir: Path) -> OnCompleteHook:
        question = self.question
        ctx = self.ctx

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            tag = ctx.log_tag()
            if sandbox_result.error:
                return

            auto_items = [c for c in question.checklist if c.method == "auto_script"]

            if auto_items:
                # 1 Collect required_files into /tmp/eval/artifacts
                if question.required_files:
                    await sb.exec_cmd("mkdir -p /tmp/eval/artifacts")
                    for rf in question.required_files:
                        await sb.exec_cmd(
                            f"cd /workspace && cp -a {rf} /tmp/eval/artifacts/$(basename {rf}) 2>/dev/null || true"
                        )
                    logger.info("[{}] Collected {} artifact(s) to /tmp/eval/artifacts",
                                tag, len(question.required_files))

                # 2 Upload case_dir
                data_dir = Path(ctx.dataset_config.data_dir)
                case_dir = data_dir / "tasks" / question.task_id
                if case_dir.exists():
                    await sb.upload_directory(case_dir, "/tmp/eval/case")

                # Ensure python is available
                await sb.exec_cmd(
                    "command -v python >/dev/null 2>&1 "
                    "|| ln -sf $(command -v python3) /usr/local/bin/python"
                )

                # 3 Execute each auto_script check
                for item in auto_items:
                    # Upload verification files
                    tasks_base = data_dir / "tasks" / question.task_id
                    for src_rel, container_path in item.files.items():
                        host_path = (tasks_base / src_rel).resolve()
                        if host_path.exists():
                            if host_path.is_dir():
                                await sb.upload_directory(host_path, container_path)
                            else:
                                await sb.write_file(
                                    container_path,
                                    host_path.read_bytes().decode(
                                        "utf-8", errors="replace"
                                    ),
                                )

                    # Set eval env vars and execute verify_cmd
                    verify_env = (
                        'export AUTO_CLAWBENCH_TASK_DIR="/tmp/eval" '
                        'AUTO_CLAWBENCH_CASE_DIR="/tmp/eval/case" '
                        'AUTO_CLAWBENCH_ARTIFACTS_DIR="/tmp/eval/artifacts" '
                        'AUTO_CLAWBENCH_WORKSPACE_DIR="/workspace" '
                        'AUTO_CLAWBENCH_INSTRUCTION_PATH="/tmp/eval/case/instruction.md" '
                        'AUTO_CLAWBENCH_ORACLE_PATH="/tmp/eval/case/oracle.json"; '
                    )
                    result = await sb.exec_cmd(verify_env + item.verify_cmd)
                    score = _parse_auto_script_output(result.stdout)
                    self._auto_results.append(
                        AutoCheckResult(
                            check_id=item.id,
                            score=score,
                            output=result.stdout.strip(),
                            error=result.stderr if result.exit_code != 0 else "",
                        )
                    )

            # Stream-export workspace.tar.gz
            workspace_tar = infer_dir / "workspace.tar.gz"
            await sb.download_directory("/workspace", workspace_tar)

        return _complete


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


async def _extract_env_overrides(sb: Sandbox, entry_script: str) -> dict[str, str]:
    """Extract environment variables set by a shell script via before/after diff."""
    env_before_result = await sb.exec_cmd("env")
    env_before = _parse_env_output(env_before_result.stdout)

    script_path = entry_script.split()[-1]
    env_after_result = await sb.exec_cmd(
        f"bash -c 'source {script_path} >/dev/null 2>&1; env'"
    )
    env_after = _parse_env_output(env_after_result.stdout)

    overrides = {}
    for key, value in env_after.items():
        if key not in env_before or env_before[key] != value:
            overrides[key] = value

    if overrides:
        logger.debug("Extracted {} env vars: {}", len(overrides), list(overrides.keys()))
    return overrides


def _parse_env_output(raw: str) -> dict[str, str]:
    """Parse `env` command output into a dict."""
    result = {}
    for line in raw.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            result[key] = value
    return result


def _parse_auto_script_output(stdout: str) -> float | None:
    """Parse auto_script verify_cmd output.

    Supports:
    1. JSON with "weighted_score" field (0.0~1.0)
    2. Legacy PASS/FAIL text (returns 1.0 or 0.0)
    """
    stdout = stdout.strip()
    if not stdout:
        return None

    # Try JSON format first
    try:
        data = json.loads(stdout)
        if isinstance(data, dict) and "weighted_score" in data:
            return float(data["weighted_score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try last line as JSON
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict) and "weighted_score" in data:
                return float(data["weighted_score"])
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Fallback to PASS/FAIL
    if "FAIL" in stdout:
        return 0.0
    if "PASS" in stdout:
        return 1.0
    return None
