"""Build OpenSandbox specs for SWE-bench Pro inference/evaluation."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from loguru import logger

from agent_probe.core.models import Error
from agent_probe.core.sandbox import Sandbox, SandboxResult, SandboxSpec
from benchmarks.swebench_pro.models import SWEBenchProInference, SWEBenchProQuestion

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


_WORKDIR = "/app"
_WORKSPACE = "/workspace"
_EVAL_SH = "/swebench_pro_eval.sh"
_SETUP_SH = "/swebench_pro_setup.sh"
_PATCH_TIMEOUT_SECONDS = 600
_IMAGE_REPO = os.environ.get("SWEBENCH_PRO_IMAGE_REPO", "jefzda/sweap-images")


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections, matching the public SWE-Pro eval script."""
    if not patch:
        return patch

    kept: list[str] = []
    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


class SWEBenchProInferSpecBuilder:
    """Constructs SandboxSpec and parses SandboxResult for one SWE-Pro instance."""

    def __init__(self, question: SWEBenchProQuestion, ctx: EvalContext) -> None:
        self.question = question
        self.ctx = ctx
        self.options = ctx.dataset_config.options
        self.gold = bool(self.options.get("gold", False))
        self.image_repo = str(self.options.get("image_repo", _IMAGE_REPO))
        self.infer_dir = ctx.output_dir / "infer" / question.qid()
        self.patch_path = self.infer_dir / "patch.diff"
        self.test_log_path = self.infer_dir / "test.log"
        self.stderr_log_path = self.infer_dir / "test.stderr.log"
        self.eval_log_path = self.infer_dir / "eval.log"
        self.output_json_path = self.infer_dir / "output.json"
        self._patch_is_empty = True
        self._agent_error: Error | None = None

    def build(self) -> SandboxSpec:
        prompt = (
            "Reply 'done' and exit immediately. Do not modify any files."
            if self.gold
            else self._agent_prompt()
        )
        return SandboxSpec(
            image=f"{self.image_repo}:{self.question.dockerhub_tag}",
            sandbox_config=self.ctx.sandbox_config,
            prompt=prompt,
            workspace=_WORKDIR,
            agent_config=self.ctx.agent_config,
            model_cfg=self.ctx.model_config,
            output_dir=str(self.infer_dir),
            timeout_sec=max(self.ctx.model_config.timeout, self.question.eval_timeout + 300),
            on_setup=self._make_setup_hook(),
            on_complete=self._make_complete_hook(),
        )

    def _make_eval_script(self) -> str:
        question = self.question
        joined_files = ",".join(question.selected_test_files)
        return "\n".join([
            "#!/bin/bash",
            "set -xo pipefail",
            question.env_exports,
            f"cd {_WORKDIR}",
            f"git reset --hard {question.base_commit}",
            "git clean -fd",
            f"git checkout {question.base_commit}",
            f"git apply -v {_WORKSPACE}/patch.diff",
            question.eval_cmd,
            f"bash {_WORKSPACE}/run_script.sh '{joined_files}' "
            f"> {_WORKSPACE}/stdout.log 2> {_WORKSPACE}/stderr.log",
            f"python {_WORKSPACE}/parser.py {_WORKSPACE}/stdout.log "
            f"{_WORKSPACE}/stderr.log {_WORKSPACE}/output.json",
        ])

    def _make_setup_script(self) -> str:
        base = self.question.base_commit
        return "\n".join([
            "#!/bin/bash",
            "set -uxo pipefail",
            f"cd {_WORKDIR}",
            f"git config --global --add safe.directory {_WORKDIR}",
            f"git reset --hard {base}",
            "git clean -fdq",
            f"git checkout -B base_ref {base}",
            "git for-each-ref --format='%(refname)' "
            "| grep -vx 'refs/heads/base_ref' "
            "| xargs -r -n1 git update-ref -d",
            "git reflog expire --all --expire=now || true",
            "rm -rf .git/logs",
        ])

    def _make_setup_hook(self):
        setup_script = self._make_setup_script()

        async def _setup(sb: Sandbox) -> None:
            await sb.write_file(_SETUP_SH, setup_script)
            result = await sb.exec_cmd(f"bash {_SETUP_SH} 2>&1")
            if result.exit_code != 0:
                logger.warning(
                    "[{}] SWE-Pro setup exited non-zero: {}",
                    self.ctx.log_tag(),
                    (result.stdout or result.stderr)[-1000:],
                )

        return _setup

    def _make_complete_hook(self):
        eval_script = self._make_eval_script()

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            self.infer_dir.mkdir(parents=True, exist_ok=True)
            self._agent_error = sandbox_result.error

            patch_text = await self._prepare_patch(sb)
            self.patch_path.write_text(patch_text, encoding="utf-8")
            self._patch_is_empty = not patch_text.strip()
            if self._patch_is_empty:
                logger.info(
                    "[{}] empty prediction patch; skipping SWE-Pro eval",
                    self.ctx.log_tag(),
                )
                return

            await sb.exec_cmd(f"mkdir -p {_WORKSPACE}")
            await sb.write_files([
                (f"{_WORKSPACE}/patch.diff", patch_text),
                (f"{_WORKSPACE}/run_script.sh", self.question.run_script),
                (f"{_WORKSPACE}/parser.py", self.question.parser_py),
                (_EVAL_SH, eval_script),
            ])

            result = await sb.exec_cmd(
                f"bash {_EVAL_SH} 2>&1",
                timeout_sec=self.question.eval_timeout,
            )
            self.eval_log_path.write_text(
                (result.stdout or "") + (f"\n[stderr]\n{result.stderr}" if result.stderr else ""),
                encoding="utf-8",
            )
            if result.exit_code != 0:
                logger.warning("[{}] SWE-Pro eval script exited non-zero", self.ctx.log_tag())

            await self._pull_eval_artifacts(sb)

        return _complete

    async def _prepare_patch(self, sb: Sandbox) -> str:
        if self.gold:
            return self.question.patch or ""

        remote_patch = "/tmp/agentprobe_swepro.patch"
        excludes = " ".join(
            f"':(exclude){pattern}'" for pattern in ("appendonlydir", "*.aof", "*.rdb")
        )
        result = await sb.exec_cmd(
            f"cd {_WORKDIR} && "
            f"git config --global --add safe.directory {_WORKDIR} && "
            "git add -A && "
            "git diff --cached -M -C --binary --full-index --unified=10 --no-color "
            f"{self.question.base_commit} -- . {excludes} > {remote_patch}",
            timeout_sec=_PATCH_TIMEOUT_SECONDS,
        )
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout or "unknown error")[-500:]
            raise RuntimeError(f"failed to generate SWE-Pro patch: {detail}")
        if result.stderr:
            logger.warning("[{}] git diff stderr: {}", self.ctx.log_tag(), result.stderr[-500:])

        try:
            patch = await sb.read_file(remote_patch)
        except Exception as read_error:
            result = await sb.exec_cmd(f"cat {remote_patch}", timeout_sec=60)
            if result.exit_code != 0:
                detail = (result.stderr or result.stdout or "patch file missing")[-500:]
                raise RuntimeError(f"failed to read SWE-Pro patch: {detail}") from read_error
            patch = result.stdout
        stripped = strip_binary_hunks(patch)
        if stripped != patch:
            logger.info("[{}] stripped binary hunks from SWE-Pro patch", self.ctx.log_tag())
        return stripped

    async def _pull_eval_artifacts(self, sb: Sandbox) -> None:
        stdout = await sb.exec_cmd(f"cat {_WORKSPACE}/stdout.log 2>/dev/null || true")
        stderr = await sb.exec_cmd(f"cat {_WORKSPACE}/stderr.log 2>/dev/null || true")
        output_json = await sb.exec_cmd(f"cat {_WORKSPACE}/output.json 2>/dev/null || true")

        self.test_log_path.write_text(stdout.stdout or "", encoding="utf-8")
        self.stderr_log_path.write_text(stderr.stdout or "", encoding="utf-8")
        if output_json.stdout and output_json.stdout.strip():
            self.output_json_path.write_text(output_json.stdout, encoding="utf-8")
        else:
            logger.warning("[{}] output.json missing; see eval.log", self.ctx.log_tag())

    def _agent_prompt(self) -> str:
        return self.question.prompt

    def to_inference(self, sandbox_result: SandboxResult) -> SWEBenchProInference:
        agent_error = self._agent_error or sandbox_result.error
        if agent_error is None and not self.gold and self._patch_is_empty:
            agent_error = Error(code=-2, message="empty prediction patch")

        return SWEBenchProInference(
            patch_path=str(self.patch_path) if self.patch_path.exists() else "",
            output_json_path=(
                str(self.output_json_path) if self.output_json_path.exists() else ""
            ),
            agent_error=agent_error,
        )
