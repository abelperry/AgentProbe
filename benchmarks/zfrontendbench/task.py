"""ZFrontendBench task implementation."""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from agent_probe.config import JudgeConfig
from agent_probe.core.models import Error
from agent_probe.core.sandbox import Sandbox, SandboxResult, SandboxSpec
from agent_probe.core.task import BaseTask
from benchmarks.zfrontendbench.frontend import (
    DEFAULT_BUILD_DIRS,
    BuildResult,
    ProjectType,
    calculate_weighted_score,
    detect_project,
    find_entry_html,
    find_project_root,
    get_unique_html_or_svg,
    is_retriable_build_error,
    parse_judge_score,
    wrap_svg_as_html,
)
from benchmarks.zfrontendbench.models import (
    ChecklistItem,
    FrontendCheckResult,
    ZFrontendBenchInference,
    ZFrontendBenchJudgement,
    ZFrontendBenchQuestion,
)
from benchmarks.zfrontendbench.prompts import FILE_EVALUATION_PROMPT, HTTP_EVALUATION_PROMPT

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


CONTAINER_WORKSPACE = "/workspace"

STATIC_SERVER_JS = """
const http = require("http");
const fs = require("fs");
const path = require("path");
const root = "{workspace}";
const port = {port};
const types = {{
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif"
}};
http.createServer((req, res) => {{
  const clean = decodeURIComponent(req.url.split("?")[0]).replace(/^\\/+/, "");
  let filePath = path.join(root, clean || "index.html");
  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {{
    filePath = path.join(filePath, "index.html");
  }}
  if (!fs.existsSync(filePath)) {{
    res.writeHead(404);
    res.end("not found");
    return;
  }}
  res.writeHead(200, {{"Content-Type": types[path.extname(filePath)] || "application/octet-stream"}});
  fs.createReadStream(filePath).pipe(res);
}}).listen(port, "0.0.0.0", () => console.log(`Server running on ${{port}}`));
"""


class ZFrontendBenchTask(
    BaseTask[ZFrontendBenchQuestion, ZFrontendBenchInference, ZFrontendBenchJudgement]
):
    """Frontend code-agent benchmark with Playwright checklist judging."""

    _judge_config: JudgeConfig | None = None

    async def inference(
        self, question: ZFrontendBenchQuestion, ctx: EvalContext
    ) -> ZFrontendBenchInference:
        infer_dir = ctx.output_dir / "infer" / question.qid()
        workspace_tar_path: Path | None = None

        async def _setup(sb: Sandbox) -> None:
            await sb.exec_cmd(f"mkdir -p {question.workspace_dir}")

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            nonlocal workspace_tar_path
            if sandbox_result.error:
                return
            workspace_tar_path = infer_dir / "workspace.tar.gz"
            await sb.download_directory(question.workspace_dir, workspace_tar_path)

        spec = SandboxSpec(
            image=question.docker,
            sandbox_config=ctx.sandbox_config,
            prompt=question.prompt,
            agent_config=ctx.agent_config,
            model_cfg=ctx.model_config,
            output_dir=str(infer_dir),
            env_vars=ctx.agent_config.envs if ctx.agent_config else {},
            workspace=question.workspace_dir,
            timeout_sec=ctx.model_config.timeout,
            on_setup=_setup,
            on_complete=_complete,
        )
        result = await Sandbox(spec).run()
        output = result.last_assistant.content_text if result.last_assistant else ""
        if result.error:
            return ZFrontendBenchInference(
                response=output,
                workspace_tar_path=workspace_tar_path,
                error=result.error,
            )
        if not output.strip():
            return ZFrontendBenchInference(
                response=output,
                workspace_tar_path=None,
                error=Error(code=-1, message="infer produced no assistant output"),
            )
        if workspace_tar_path is None or not workspace_tar_path.exists():
            return ZFrontendBenchInference(
                response=output,
                workspace_tar_path=None,
                error=Error(code=-1, message="workspace artifact was not exported"),
            )
        return ZFrontendBenchInference(response=output, workspace_tar_path=workspace_tar_path)

    async def judge(
        self,
        question: ZFrontendBenchQuestion,
        inference_result: ZFrontendBenchInference,
        ctx: EvalContext,
        prev_judgement: ZFrontendBenchJudgement | None = None,
    ) -> ZFrontendBenchJudgement:
        if inference_result.error:
            return self._error_judgement(
                question,
                inference_result,
                f"Response invalid: {inference_result.error.message}",
            )
        if inference_result.workspace_tar_path is None or not inference_result.workspace_tar_path.exists():
            return self._error_judgement(
                question, inference_result, "workspace.tar.gz not found"
            )

        eval_dir = ctx.output_dir / "eval" / question.qid()
        workspace_path = eval_dir / "workspace"
        self._extract_workspace(inference_result.workspace_tar_path, workspace_path)
        if not self._has_files(workspace_path):
            return self._error_judgement(
                question, inference_result, "workspace is empty after extraction"
            )

        actual_workspace = find_project_root(workspace_path)
        checklist = question.get_checklist()
        if not checklist:
            return ZFrontendBenchJudgement(
                category=question.category,
                weighted_score=0.0,
                response=inference_result.response,
            )

        build_result: BuildResult | None = None
        if question.test_mode == "http":
            build_result = await self._prepare_http_build(question, actual_workspace, eval_dir, ctx)
            if not build_result.success:
                checks = self._build_failed_checks(
                    checklist, build_result.error_message, build_result.code_failure
                )
                weighted_score = calculate_weighted_score(checks)
                error = None if build_result.code_failure else Error(
                    code=-1,
                    message=build_result.error_message,
                )
                return ZFrontendBenchJudgement(
                    category=question.category,
                    judge_output=build_result.error_message,
                    check_results=checks,
                    weighted_score=weighted_score,
                    response=inference_result.response,
                    error=error,
                )
        elif get_unique_html_or_svg(actual_workspace) is None:
            return self._error_judgement(question, inference_result, "No html/svg found")

        previous = self._cached_successful_checks(prev_judgement)
        semaphore = asyncio.Semaphore(max(1, question.eval_concurrent))

        async def _eval(item: ChecklistItem) -> FrontendCheckResult:
            cached = previous.get(str(item.id))
            if cached is not None:
                return cached
            async with semaphore:
                return await self._eval_one(
                    question=question,
                    item=item,
                    ctx=ctx,
                    workspace_path=actual_workspace,
                    build_result=build_result,
                )

        check_results = await asyncio.gather(*[_eval(item) for item in checklist])
        weighted_score = calculate_weighted_score(check_results)
        error = None
        if weighted_score is None:
            error = Error(code=-1, message="Has error checks (eval failed)")

        return ZFrontendBenchJudgement(
            category=question.category,
            judge_output=json.dumps(
                {
                    "weighted_score": weighted_score,
                    "checks": [r.model_dump(mode="json") for r in check_results],
                },
                ensure_ascii=False,
            ),
            check_results=check_results,
            weighted_score=weighted_score,
            response=inference_result.response,
            error=error,
        )

    def collect_metrics(
        self, judgements: list[ZFrontendBenchJudgement]
    ) -> tuple[dict[str, float], int]:
        if not judgements:
            return {
                "average": 0.0,
                "ISR": 0.0,
                "ISR_macro": 0.0,
                "CSR": 0.0,
                "CSR_macro": 0.0,
                "BSR": 0.0,
            }, 0

        valid = [
            j
            for j in judgements
            if j.error is None
            and j.weighted_score is not None
            and all(c.score is not None for c in j.check_results)
        ]
        num_success = len(valid)
        weighted_scores = [j.weighted_score or 0.0 for j in valid]
        full_score_tasks = sum(1 for j in valid if j.weighted_score == 1.0)
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        build_success_tasks = 0

        category_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "num_success": 0,
                "full_score_tasks": 0,
                "build_success_tasks": 0,
                "total_checks": 0,
                "passed_checks": 0,
            }
        )

        for judgement in valid:
            category = judgement.category or "unknown"
            category_stats[category]["num_success"] += 1
            if judgement.weighted_score == 1.0:
                category_stats[category]["full_score_tasks"] += 1

            task_has_build_failure = False
            for check in judgement.check_results:
                total_checks += 1
                category_stats[category]["total_checks"] += 1
                if check.score == 1.0:
                    passed_checks += 1
                    category_stats[category]["passed_checks"] += 1
                else:
                    failed_checks += 1
                    if "build failed" in check.reason.lower():
                        task_has_build_failure = True

            if not task_has_build_failure:
                build_success_tasks += 1
                category_stats[category]["build_success_tasks"] += 1

        metrics = {
            "average": (sum(weighted_scores) / len(weighted_scores) * 100)
            if weighted_scores
            else 0.0,
            "ISR": (full_score_tasks / num_success * 100) if num_success else 0.0,
            "CSR": (passed_checks / total_checks * 100) if total_checks else 0.0,
            "BSR": (build_success_tasks / num_success * 100) if num_success else 0.0,
        }

        category_isrs: list[float] = []
        category_csrs: list[float] = []
        for category, stats in sorted(category_stats.items()):
            cat_success = stats["num_success"]
            cat_total_checks = stats["total_checks"]
            cat_isr = (
                stats["full_score_tasks"] / cat_success * 100
                if cat_success
                else 0.0
            )
            cat_csr = (
                stats["passed_checks"] / cat_total_checks * 100
                if cat_total_checks
                else 0.0
            )
            cat_bsr = (
                stats["build_success_tasks"] / cat_success * 100
                if cat_success
                else 0.0
            )
            metrics[f"ISR_{category}"] = cat_isr
            metrics[f"CSR_{category}"] = cat_csr
            metrics[f"BSR_{category}"] = cat_bsr
            if cat_success:
                category_isrs.append(cat_isr)
                category_csrs.append(cat_csr)

        metrics["ISR_macro"] = (
            sum(category_isrs) / len(category_isrs) if category_isrs else 0.0
        )
        metrics["CSR_macro"] = (
            sum(category_csrs) / len(category_csrs) if category_csrs else 0.0
        )
        metrics["total_checks"] = float(total_checks)
        metrics["passed_checks"] = float(passed_checks)
        metrics["failed_checks"] = float(failed_checks)
        metrics["full_score_tasks"] = float(full_score_tasks)
        metrics["build_success_tasks"] = float(build_success_tasks)
        return metrics, num_success

    async def _prepare_http_build(
        self,
        question: ZFrontendBenchQuestion,
        workspace_path: Path,
        eval_dir: Path,
        ctx: EvalContext,
    ) -> BuildResult:
        project_info = detect_project(workspace_path)
        if project_info.project_type == ProjectType.HTML:
            entry = find_entry_html(project_info.project_dir)
            if entry:
                return BuildResult(
                    success=True,
                    project_type=ProjectType.HTML,
                    source_dir=entry.parent,
                    entry_file=entry,
                )
            return BuildResult(
                success=False,
                project_type=ProjectType.HTML,
                error_message="Cannot identify project type",
                code_failure=True,
            )

        if project_info.project_type == ProjectType.SVG:
            svg = get_unique_html_or_svg(project_info.project_dir)
            if svg and svg.suffix.lower() == ".svg":
                entry = wrap_svg_as_html(svg)
                return BuildResult(
                    success=True,
                    project_type=ProjectType.SVG,
                    source_dir=entry.parent,
                    entry_file=entry,
                )
            return BuildResult(
                success=False,
                project_type=ProjectType.SVG,
                error_message="Cannot identify project type",
                code_failure=True,
            )

        if project_info.project_type == ProjectType.UNKNOWN:
            return BuildResult(
                success=False,
                project_type=ProjectType.UNKNOWN,
                error_message="Cannot identify project type",
                code_failure=True,
            )

        return await self._build_npm_project(
            question, project_info.project_dir, eval_dir, ctx
        )

    async def _eval_one(
        self,
        question: ZFrontendBenchQuestion,
        item: ChecklistItem,
        ctx: EvalContext,
        workspace_path: Path,
        build_result: BuildResult | None,
    ) -> FrontendCheckResult:
        judge_cfg = self._get_judge_config(ctx)
        start = time.time()

        async def _setup(sb: Sandbox) -> None:
            await self._setup_eval_workspace(
                sb, question, workspace_path, build_result
            )

        if question.test_mode == "http":
            prompt = HTTP_EVALUATION_PROMPT.format(
                task_description=question.task_description,
                project_url=f"http://localhost:{question.http_port}",
                workspace_path=CONTAINER_WORKSPACE,
                checklist_item_description=item.description,
            )
        else:
            prompt = FILE_EVALUATION_PROMPT.format(
                task_description=question.task_description,
                workspace_path=CONTAINER_WORKSPACE,
                checklist_item_description=item.description,
            )

        spec = SandboxSpec(
            image=question.judge_docker,
            sandbox_config=ctx.sandbox_config,
            prompt=prompt,
            agent_config=judge_cfg.agent,
            model_cfg=judge_cfg.model,
            output_dir=str(ctx.output_dir / "eval" / question.qid() / f"check_{item.id}"),
            env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
            workspace=CONTAINER_WORKSPACE,
            timeout_sec=question.eval_timeout,
            on_setup=_setup,
        )
        result = await Sandbox(spec).run()
        output = result.last_assistant.content_text if result.last_assistant else ""
        if result.error:
            return FrontendCheckResult(
                id=item.id,
                description=item.description,
                weight=item.weight,
                score=None,
                reason=f"Execution error: {result.error.message}",
                duration=time.time() - start,
            )
        score, reason = parse_judge_score(output)
        return FrontendCheckResult(
            id=item.id,
            description=item.description,
            weight=item.weight,
            score=score,
            reason=reason,
            duration=time.time() - start,
        )

    def _get_judge_config(self, ctx: EvalContext) -> JudgeConfig:
        if self._judge_config is None:
            self._judge_config = JudgeConfig.from_yaml(
                Path(ctx.dataset_config.get_judge_config_path("agent_judge_with_playwright"))
            )
        return self._judge_config

    async def _build_npm_project(
        self,
        question: ZFrontendBenchQuestion,
        project_dir: Path,
        eval_dir: Path,
        ctx: EvalContext,
    ) -> BuildResult:
        build_tar_path = eval_dir / "build_artifacts.tar.gz"
        build_result: BuildResult | None = None

        async def _setup(sb: Sandbox) -> None:
            await sb.upload_directory(project_dir, CONTAINER_WORKSPACE)

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            nonlocal build_result
            if sandbox_result.error:
                return
            build_output = await sb.exec_cmd(
                self._npm_build_command(project_dir), question.http_build_timeout
            )
            if build_output.exit_code != 0:
                tail = (build_output.stderr or build_output.stdout or "")[-500:]
                message = f"error during build: {tail}" if tail else "error during build"
                build_result = BuildResult(
                    success=False,
                    project_type=ProjectType.NPM,
                    error_message=message,
                    code_failure=not is_retriable_build_error(message),
                )
                return

            remote_dir, server_side = await self._find_remote_artifact(sb)
            if remote_dir:
                await sb.download_directory(remote_dir, build_tar_path)
                build_result = BuildResult(
                    success=True,
                    project_type=ProjectType.NPM,
                    artifact_tar_path=build_tar_path,
                    server_side=server_side,
                )
                return

            message = "No build output found"
            build_result = BuildResult(
                success=False,
                project_type=ProjectType.NPM,
                error_message=message,
                code_failure=not is_retriable_build_error(message),
            )

        spec = SandboxSpec(
            image=question.judge_docker,
            sandbox_config=ctx.sandbox_config,
            prompt="",
            output_dir=str(eval_dir / "build"),
            workspace=CONTAINER_WORKSPACE,
            timeout_sec=question.http_build_timeout + 180,
            on_setup=_setup,
            on_complete=_complete,
        )
        result = await Sandbox(spec).run()
        if result.error:
            return BuildResult(
                success=False,
                project_type=ProjectType.NPM,
                error_message=result.error.message,
                code_failure=False,
            )
        if build_result is None:
            return BuildResult(
                success=False,
                project_type=ProjectType.NPM,
                error_message="Build result not available",
                code_failure=False,
            )
        return build_result

    async def _setup_eval_workspace(
        self,
        sb: Sandbox,
        question: ZFrontendBenchQuestion,
        workspace_path: Path,
        build_result: BuildResult | None,
    ) -> None:
        await sb.exec_cmd(f"mkdir -p {CONTAINER_WORKSPACE}")
        if question.test_mode == "http" and build_result:
            if build_result.artifact_tar_path:
                await self._upload_tar(
                    sb, build_result.artifact_tar_path, CONTAINER_WORKSPACE
                )
            elif build_result.source_dir:
                await sb.upload_directory(build_result.source_dir, CONTAINER_WORKSPACE)
                await self._ensure_index_entry(sb, build_result)
            else:
                await sb.upload_directory(workspace_path, CONTAINER_WORKSPACE)
            await self._start_http_server(
                sb, question.http_port, build_result.server_side
            )
        else:
            await sb.upload_directory(workspace_path, CONTAINER_WORKSPACE)

    async def _ensure_index_entry(
        self, sb: Sandbox, build_result: BuildResult
    ) -> None:
        if (
            build_result.source_dir is None
            or build_result.entry_file is None
            or build_result.entry_file.name == "index.html"
        ):
            return
        rel_entry = build_result.entry_file.relative_to(build_result.source_dir)
        remote_entry = f"{CONTAINER_WORKSPACE}/{rel_entry.as_posix()}"
        await sb.exec_cmd(
            f"cp {shlex.quote(remote_entry)} {CONTAINER_WORKSPACE}/index.html"
        )

    async def _upload_tar(self, sb: Sandbox, tar_path: Path, remote_dir: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(tmp_path, filter=self._tar_filter)
            await sb.upload_directory(tmp_path, remote_dir)

    async def _start_http_server(
        self, sb: Sandbox, port: int, server_side: bool
    ) -> None:
        if server_side:
            start_script = f"""
cd {CONTAINER_WORKSPACE}
if [ ! -d node_modules ]; then
  npm install --no-audit --no-fund >/tmp/npm-install.log 2>&1 || true
fi
START_CMD=$(node -e 'const p=require("./package.json"); const s=p.scripts||{{}}; if(s.preview) console.log("npm run preview -- --host 0.0.0.0 --port {port}"); else if(s.start) console.log("npm start -- --port {port}"); else if(s.dev) console.log("npm run dev -- --host 0.0.0.0 --port {port}"); else console.log("npm start");')
PORT={port} setsid bash -lc "$START_CMD" >/tmp/server.log 2>&1 &
"""
            await sb.exec_cmd(start_script)
        else:
            server_script = STATIC_SERVER_JS.format(
                workspace=CONTAINER_WORKSPACE, port=port
            )
            await sb.write_file("/tmp/server.js", server_script)
            await sb.exec_cmd("setsid node /tmp/server.js >/tmp/server.log 2>&1 &")
        await self._wait_for_server(sb, port)

    async def _wait_for_server(
        self,
        sb: Sandbox,
        port: int,
        max_attempts: int = 40,
        in_progress_grace_attempts: int = 30,
    ) -> None:
        deadline_attempts = max_attempts
        grace_extended = False
        in_progress_markers = (
            "starting the development server",
            "compiling...",
            "compiling ",
            "webpack is watching",
            "webpack compiling",
        )
        ready_markers = (
            "Server running",
            "ready started",
            "Ready on",
            "Listening on",
            f":{port}",
            "started server",
            "Local:",
            "Compiled successfully",
            "compiled successfully",
            "compiled with warnings",
            "webpack compiled",
            "You can now view",
        )

        attempt = 0
        while attempt < deadline_attempts:
            await asyncio.sleep(3)
            attempt += 1

            log = await sb.exec_cmd("cat /tmp/server.log 2>/dev/null || echo ''", 15)
            if any(marker in log.stdout for marker in ready_markers):
                return
            if not grace_extended and any(
                marker in log.stdout.lower() for marker in in_progress_markers
            ):
                deadline_attempts += in_progress_grace_attempts
                grace_extended = True

            result = await sb.exec_cmd(
                f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 "
                f"--max-time 10 http://localhost:{port}/ 2>/dev/null || echo '000'",
                20,
            )
            status = result.stdout.strip()
            if status.isdigit() and int(status) > 0:
                return
        log = await sb.exec_cmd("cat /tmp/server.log 2>/dev/null || echo 'no log'")
        raise RuntimeError(f"HTTP server failed to start. Log: {log.stdout[:500]}")

    async def _find_remote_artifact(self, sb: Sandbox) -> tuple[str | None, bool]:
        for dirname in [".next", ".nuxt", ".output"]:
            check = await sb.exec_cmd(f"test -d {CONTAINER_WORKSPACE}/{dirname}")
            if check.exit_code == 0:
                return CONTAINER_WORKSPACE, True
        for dirname in DEFAULT_BUILD_DIRS:
            check = await sb.exec_cmd(f"test -d {CONTAINER_WORKSPACE}/{dirname}")
            if check.exit_code == 0:
                return f"{CONTAINER_WORKSPACE}/{dirname}", False
        check = await sb.exec_cmd(f"test -f {CONTAINER_WORKSPACE}/index.html")
        if check.exit_code == 0:
            return CONTAINER_WORKSPACE, False
        return None, False

    @staticmethod
    def _npm_build_command(project_dir: Path) -> str:
        use_pnpm = (project_dir / "pnpm-lock.yaml").exists()
        install = (
            "pnpm install --no-frozen-lockfile"
            if use_pnpm
            else "npm install --prefer-offline"
        )
        build = "pnpm run build" if use_pnpm else "npm run build"
        return f"""
cd {CONTAINER_WORKSPACE}
export NODE_ENV=development
export npm_config_fund=false
export npm_config_audit=false
export npm_config_progress=false
export npm_config_registry=https://registry.npmmirror.com
export npm_config_fetch_retries=5
export npm_config_fetch_timeout=1200000
if {"true" if use_pnpm else "false"} && ! command -v pnpm >/dev/null 2>&1; then
  npm install -g pnpm
fi
{install} 2>&1
{build} 2>&1
"""

    def _extract_workspace(self, tar_path: Path, workspace_path: Path) -> None:
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        workspace_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(workspace_path, filter=self._tar_filter)

    @staticmethod
    def _tar_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
        if member.issym() or member.islnk():
            return None
        if "node_modules" in Path(member.name).parts:
            return None
        root = Path(path).resolve()
        target = (root / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return None
        return member

    @staticmethod
    def _has_files(path: Path) -> bool:
        return any(item.is_file() for item in path.rglob("*"))

    @staticmethod
    def _build_failed_checks(
        checklist: list[ChecklistItem], reason: str, code_failure: bool
    ) -> list[FrontendCheckResult]:
        score = 0.0 if code_failure else None
        return [
            FrontendCheckResult(
                id=item.id,
                description=item.description,
                weight=item.weight,
                score=score,
                reason=f"Build failed: {reason}",
            )
            for item in checklist
        ]

    @staticmethod
    def _error_judgement(
        question: ZFrontendBenchQuestion,
        inference_result: ZFrontendBenchInference,
        message: str,
    ) -> ZFrontendBenchJudgement:
        return ZFrontendBenchJudgement(
            category=question.category,
            judge_output=message,
            weighted_score=None,
            response=inference_result.response,
            error=Error(code=-1, message=message),
        )

    @staticmethod
    def _cached_successful_checks(
        prev_judgement: ZFrontendBenchJudgement | None,
    ) -> dict[str, FrontendCheckResult]:
        if prev_judgement is None:
            return {}
        return {
            str(result.id): result
            for result in prev_judgement.check_results
            if result.score is not None
        }
