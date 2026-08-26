"""Build OpenSandbox specs for SWE-bench inference/evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from agent_probe.core.models import Error
from agent_probe.core.sandbox import ExecResult, Sandbox, SandboxResult, SandboxSpec
from benchmarks.swebench.models import SWEBenchInference, SWEBenchQuestion
from benchmarks.swebench.official import make_test_spec

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


_WORKDIR = "/testbed"
_EVAL_SH = "/eval.sh"
_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


class SWEBenchInferSpecBuilder:
    """Constructs SandboxSpec and parses SandboxResult for one instance."""

    def __init__(self, question: SWEBenchQuestion, ctx: EvalContext) -> None:
        self.question = question
        self.ctx = ctx
        self.options = ctx.dataset_config.options
        self.namespace = str(self.options.get("namespace", "swebench"))
        self.gold = bool(self.options.get("gold", False))
        self.infer_dir = ctx.output_dir / "infer" / question.qid()
        self.patch_path = self.infer_dir / "patch.diff"
        self.test_log_path = self.infer_dir / "test_output.txt"
        self._test_spec = None

    def _make_test_spec(self):
        if self._test_spec is None:
            self._test_spec = make_test_spec(self.question, namespace=self.namespace)
        return self._test_spec

    def build(self) -> SandboxSpec:
        spec = self._make_test_spec()
        prompt = (
            "Reply 'done' and exit immediately. Do not modify any files."
            if self.gold
            else self._agent_prompt()
        )
        return SandboxSpec(
            image=spec.instance_image_key,
            sandbox_config=self.ctx.sandbox_config,
            prompt=prompt,
            workspace=_WORKDIR,
            agent_config=self.ctx.agent_config,
            model_cfg=self.ctx.model_config,
            output_dir=str(self.infer_dir),
            on_complete=self._make_complete_hook(),
        )

    def _make_complete_hook(self):
        official_spec = self._make_test_spec()
        eval_script = "\n".join(
            ["#!/bin/bash", "set -uxo pipefail", *official_spec.eval_script_list]
        )

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            self.infer_dir.mkdir(parents=True, exist_ok=True)
            await sb.write_file(_EVAL_SH, eval_script)

            patch_text = await self._prepare_patch(sb)
            self.patch_path.write_text(patch_text, encoding="utf-8")

            result = await self._run_eval(sb)
            self.test_log_path.write_text(result.stdout or "", encoding="utf-8")
            if result.stderr:
                logger.warning("[{}] eval stderr: {}", self.ctx.log_tag(), result.stderr[-1000:])

        return _complete

    async def _prepare_patch(self, sb: Sandbox) -> str:
        if self.gold:
            gold_patch = self.question.patch or ""
            if not gold_patch.strip():
                logger.warning("[{}] gold patch is empty", self.ctx.log_tag())
                return gold_patch
            await sb.write_file("/tmp/gold.patch", gold_patch)
            # Do not reset/clean: prebuilt SWE-bench images may contain
            # uncommitted pre_install edits that test parsers rely on.
            result = await sb.exec_cmd(
                f"cd {_WORKDIR} && "
                f"git config --global --add safe.directory {_WORKDIR} && "
                "(git apply --verbose /tmp/gold.patch || "
                " git apply --verbose --reject /tmp/gold.patch || "
                " patch --batch --fuzz=5 -p1 -i /tmp/gold.patch)"
            )
            if result.exit_code != 0:
                logger.warning(
                    "[{}] gold patch apply failed: {}",
                    self.ctx.log_tag(),
                    result.stderr[-500:],
                )
            elif result.stderr:
                logger.debug(
                    "[{}] gold patch apply stderr: {}",
                    self.ctx.log_tag(),
                    result.stderr[-500:],
                )
            return gold_patch

        remote_patch = "/tmp/agentprobe_swebench.patch"
        result = await sb.exec_cmd(
            f"cd {_WORKDIR} && "
            f"git config --global --add safe.directory {_WORKDIR} && "
            "git add -A && "
            "git diff --cached -M -C --binary --full-index --unified=10 --no-color "
            f"{self.question.base_commit} > {remote_patch}"
        )
        if result.stderr:
            logger.warning("[{}] git diff stderr: {}", self.ctx.log_tag(), result.stderr[-500:])
        return await sb.read_file(remote_patch)

    async def _run_eval(self, sb: Sandbox) -> ExecResult:
        clean_env = "unset " + " ".join(_PROXY_ENV_VARS)
        result = await sb.exec_cmd(f"{clean_env}; /bin/bash {_EVAL_SH} 2>&1")
        if result.exit_code != 0:
            logger.warning("[{}] eval.sh exited non-zero", self.ctx.log_tag())
        return result

    def _agent_prompt(self) -> str:
        return (
            "Work autonomously. Apply the fix end-to-end; do not stop midway "
            "and do not ask the user for confirmation.\n\n"
            f"{self.question.problem_statement}"
        )

    def _patch_is_empty(self) -> bool:
        return not self.patch_path.exists() or not self.patch_path.read_text(
            encoding="utf-8"
        ).strip()

    def to_inference(self, sandbox_result: SandboxResult) -> SWEBenchInference:
        output = sandbox_result.last_assistant.content_text if sandbox_result.last_assistant else ""
        if sandbox_result.error:
            return SWEBenchInference(
                patch_path=str(self.patch_path) if self.patch_path.exists() else "",
                test_log_path=str(self.test_log_path) if self.test_log_path.exists() else "",
                output=output,
                error=sandbox_result.error
            )
        error = (
            Error(code=-2, message="empty prediction patch")
            if not self.gold and self._patch_is_empty()
            else None
        )
        return SWEBenchInference(
            patch_path=str(self.patch_path) if self.patch_path.exists() else "",
            test_log_path=str(self.test_log_path) if self.test_log_path.exists() else "",
            output=output,
            error=error,
        )
