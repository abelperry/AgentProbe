"""Functional evaluation: build the workspace, serve it, and inspect it in a browser.

The other half of MTAC-IFBench, answering a different question from instruction
following. Instruction following asks whether the agent obeyed its constraints;
this asks whether what it built works -- "hovering a status light shows a
tooltip" can only be settled by rendering the page and moving a mouse over it.

That costs a build, an HTTP server and a browser-driving judge per task, which
is why ``JudgeConfig.function_checklist_eval_enabled`` defaults to False. When it
is off, every checklist item is recorded with ``score=None`` and a skip reason
rather than 0 -- an unrun check is a coverage fact, not a failure.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import tarfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from agent_probe.agents.claude_code import ClaudeCodeAgent
from agent_probe.config import AgentConfig, JudgeConfig, ModelConfig
from agent_probe.core.models import Error
from agent_probe.core.sandbox import Sandbox, SandboxSpec
from agent_probe.utils.imports import import_class
from benchmarks.mtacifbench.function_prompts import (
    FUNCTION_EVALUATION_HTTP_PROMPT,
    FUNCTION_EVALUATION_PROMPT,
)
from benchmarks.mtacifbench.models import (
    FunctionCheckResult,
    MTACIFBenchInference,
    MTACIFBenchQuestion,
    resolve_judge_image,
)
from benchmarks.mtacifbench.validation import collect_judge_candidates

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext
    from agent_probe.core.sandbox import SandboxResult


CONTAINER_WORKSPACE = "/workspace"
CONTAINER_JUDGE_WORKDIR = "/tmp"
FUNCTION_CHECKLIST_SKIP_REASON = (
    "Skipped: function_checklist evaluation is disabled for MTACIFBench."
)
DEFAULT_BUILD_DIRS = ("dist", "build", "out", "public", ".next", "_site", "www")
SSR_BUILD_DIRS = (".next", ".nuxt", ".output")
SNAPSHOT_EXCLUDES = ("node_modules", ".git", "__pycache__")
HTTP_SERVER_TEMPLATE = Path(__file__).parent / "templates" / "http_server.js"


class ProjectType(StrEnum):
    NPM = "npm"
    HTML = "html"
    SVG = "svg"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProjectInfo:
    project_type: ProjectType
    project_dir: Path
    framework: str = "unknown"


@dataclass(frozen=True)
class FunctionBuildResult:
    success: bool
    artifact_dir: Path | None = None
    entry_file: Path | None = None
    project_relative: Path = Path(".")
    is_ssr: bool = False
    error_message: str = ""
    build_log: str = ""


@dataclass(frozen=True)
class FunctionEvaluationOutcome:
    checks: list[FunctionCheckResult]
    function_score: float
    function_checklist_skipped: bool
    build_success: bool | None


def skipped_function_evaluation(
    question: MTACIFBenchQuestion,
) -> FunctionEvaluationOutcome:
    checks = [
        FunctionCheckResult(
            id=index,
            description=item,
            score=None,
            reason=FUNCTION_CHECKLIST_SKIP_REASON,
        )
        for index, item in enumerate(question.function_checklist)
    ]
    return FunctionEvaluationOutcome(
        checks=checks,
        function_score=0.0,
        function_checklist_skipped=True,
        build_success=None,
    )


async def evaluate_function_checklist(
    *,
    question: MTACIFBenchQuestion,
    inference_result: MTACIFBenchInference,
    ctx: EvalContext,
    judge_config: JudgeConfig,
    eval_dir: Path,
) -> FunctionEvaluationOutcome:
    """Build the final workspace once, then judge checklist items concurrently."""
    checklist = question.function_checklist
    if not checklist:
        return FunctionEvaluationOutcome(
            checks=[],
            function_score=0.0,
            function_checklist_skipped=False,
            build_success=True,
        )

    function_model, function_agent = judge_config.function_runtime()
    runtime_error = _validate_function_runtime(function_model, function_agent)
    if runtime_error:
        return _incomplete_outcome(checklist, runtime_error)

    try:
        judge_image = resolve_judge_image(question, judge_config)
    except ValueError as exc:
        return _incomplete_outcome(checklist, str(exc))

    workspace_archive = inference_result.workspace_tar_path
    if workspace_archive is None or not workspace_archive.is_file():
        return _incomplete_outcome(checklist, "workspace archive not found")

    function_dir = eval_dir / "function_checklist"
    workspace_dir = function_dir / "workspace"
    try:
        await asyncio.to_thread(
            _replace_directory_from_archive,
            workspace_archive,
            workspace_dir,
        )
    except Exception as exc:
        return _incomplete_outcome(
            checklist,
            f"extract workspace archive failed: {exc}",
        )

    actual_workspace = _strip_single_wrapper(workspace_dir)
    project_info = detect_project(actual_workspace)
    eval_mode = question.test_mode
    build_result: FunctionBuildResult | None = None
    needs_build = project_info.project_type in {ProjectType.NPM, ProjectType.HTML}
    if eval_mode == "http" and needs_build:
        build_result = await _build_workspace(
            question=question,
            ctx=ctx,
            function_dir=function_dir,
            workspace_dir=actual_workspace,
            project_info=project_info,
            judge_image=judge_image,
        )
        if not build_result.success:
            reason = f"Build failed: {build_result.error_message or 'unknown error'}"
            return _failed_build_outcome(checklist, reason)
    elif eval_mode == "http":
        eval_mode = "file"
    elif eval_mode == "file" and not (actual_workspace / "index.html").is_file():
        return _failed_build_outcome(checklist, "Missing root index.html")

    source_dir = (
        build_result.artifact_dir
        if build_result is not None and build_result.artifact_dir is not None
        else actual_workspace
    )
    if source_dir is None or not source_dir.is_dir():
        return _incomplete_outcome(checklist, "functional evaluation workspace is missing")

    semaphore = asyncio.Semaphore(max(1, question.eval_concurrent))
    checks = await asyncio.gather(
        *(
            _evaluate_one(
                question=question,
                item=item,
                item_index=index,
                ctx=ctx,
                function_dir=function_dir,
                source_dir=source_dir,
                eval_mode=eval_mode,
                build_result=build_result,
                function_model=function_model,
                function_agent=function_agent,
                semaphore=semaphore,
                judge_image=judge_image,
            )
            for index, item in enumerate(checklist)
        )
    )
    function_score = sum(float(item.score or 0.0) for item in checks) / len(checks)
    return FunctionEvaluationOutcome(
        checks=checks,
        function_score=function_score,
        function_checklist_skipped=False,
        build_success=True,
    )


def _validate_function_runtime(
    model_config: ModelConfig,
    agent_config: AgentConfig,
) -> str | None:
    try:
        agent_cls = import_class(agent_config.type)
    except Exception as exc:
        return f"invalid functional judge agent: {exc}"
    if not issubclass(agent_cls, ClaudeCodeAgent):
        return "function checklist evaluation requires ClaudeCodeAgent"
    if not agent_config.mcp_host_path:
        return "function checklist evaluation requires a Playwright MCP config"
    mcp_path = Path(agent_config.mcp_host_path)
    if not mcp_path.is_file():
        return f"Playwright MCP config not found: {mcp_path}"
    model_label = f"{model_config.api_name} {model_config.model_name}".lower()
    if "sonnet" not in model_label:
        logger.warning(
            "Functional judge model does not look like Claude Sonnet: {}",
            model_config.model_name or model_config.api_name,
        )
    return None


def _incomplete_outcome(
    checklist: list[str],
    reason: str,
) -> FunctionEvaluationOutcome:
    error = Error(code=-1, message=reason[:2000])
    return FunctionEvaluationOutcome(
        checks=[
            FunctionCheckResult(
                id=index,
                description=item,
                score=None,
                reason=reason,
                evaluation_error=error,
            )
            for index, item in enumerate(checklist)
        ],
        function_score=0.0,
        function_checklist_skipped=False,
        build_success=None,
    )


def _failed_build_outcome(
    checklist: list[str],
    reason: str,
) -> FunctionEvaluationOutcome:
    return FunctionEvaluationOutcome(
        checks=[
            FunctionCheckResult(
                id=index,
                description=item,
                score=0.0,
                reason=reason,
            )
            for index, item in enumerate(checklist)
        ],
        function_score=0.0,
        function_checklist_skipped=False,
        build_success=False,
    )


def detect_project(project_dir: Path) -> ProjectInfo:
    """Match the reference evaluator's shallow-to-deep frontend detection."""
    project_dir = Path(project_dir)

    def npm_project(path: Path) -> tuple[Path, dict[str, Any]] | None:
        package_path = path / "package.json"
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if "build" not in package.get("scripts", {}):
            return None
        return path, package

    candidates: list[tuple[Path, dict[str, Any]]] = []
    root_candidate = npm_project(project_dir)
    if root_candidate:
        candidates.append(root_candidate)
    try:
        for child in sorted(project_dir.iterdir()):
            if child.is_dir() and (candidate := npm_project(child)):
                candidates.append(candidate)
    except OSError:
        pass
    if not candidates:
        package_paths = sorted(
            (
                path
                for path in project_dir.glob("**/package.json")
                if "node_modules" not in path.parts
            ),
            key=lambda path: len(path.parts),
        )
        for package_path in package_paths:
            candidate = npm_project(package_path.parent)
            if candidate:
                candidates.append(candidate)
                break
    if candidates:
        path, package = candidates[0]
        dependencies = {
            **package.get("dependencies", {}),
            **package.get("devDependencies", {}),
        }
        framework = next(
            (
                name
                for name in (
                    "next",
                    "nuxt",
                    "gatsby",
                    "astro",
                    "svelte",
                    "vue",
                    "react",
                    "preact",
                )
                if name in dependencies
            ),
            "other",
        )
        return ProjectInfo(ProjectType.NPM, path, framework)

    svg_files = [path for path in project_dir.glob("**/*.svg") if "node_modules" not in path.parts]
    html_files = [
        path for path in project_dir.glob("**/*.html") if "node_modules" not in path.parts
    ]
    if svg_files and not html_files:
        return ProjectInfo(ProjectType.SVG, min(svg_files, key=lambda path: len(path.parts)).parent)
    if html_files:
        index_files = [path for path in html_files if path.name == "index.html"]
        selected = min(index_files or html_files, key=lambda path: len(path.parts))
        return ProjectInfo(ProjectType.HTML, selected.parent)
    return ProjectInfo(ProjectType.UNKNOWN, project_dir)


async def _build_workspace(
    *,
    question: MTACIFBenchQuestion,
    ctx: EvalContext,
    function_dir: Path,
    workspace_dir: Path,
    project_info: ProjectInfo,
    judge_image: str,
) -> FunctionBuildResult:
    build_dir = function_dir / "build_workspace"

    if project_info.project_type == ProjectType.HTML:
        _replace_directory_copy(workspace_dir, build_dir)
        entry_file = _find_html_entry(build_dir)
        result = FunctionBuildResult(
            success=entry_file is not None,
            artifact_dir=build_dir if entry_file is not None else None,
            entry_file=entry_file,
            error_message="No HTML entry found" if entry_file is None else "",
        )
        return result

    export_tar = function_dir / ".build_workspace.tar.gz"
    if export_tar.exists():
        export_tar.unlink()
    holder: dict[str, Any] = {"error": "", "log": "", "entry": None, "ssr": False}
    relative_project = project_info.project_dir.relative_to(workspace_dir)
    remote_project = (
        CONTAINER_WORKSPACE
        if str(relative_project) == "."
        else f"{CONTAINER_WORKSPACE}/{relative_project.as_posix()}"
    )

    async def setup(build_sb: Sandbox) -> None:
        await build_sb.upload_directory(workspace_dir, CONTAINER_WORKSPACE)
        command = _npm_build_command(project_info.project_dir)
        script = "\n".join(
            (
                "set +e",
                "export NODE_ENV=development",
                "export npm_config_fund=false",
                "export npm_config_audit=false",
                "export npm_config_progress=false",
                "export npm_config_loglevel=error",
                "export npm_config_registry=https://registry.npmmirror.com",
                "export npm_config_fetch_retries=5",
                "export npm_config_fetch_retry_mintimeout=20000",
                "export npm_config_fetch_retry_maxtimeout=120000",
                "export npm_config_fetch_timeout=1200000",
                f"cd {shlex.quote(remote_project)}",
                "rm -rf node_modules package-lock.json",
                "npm install --prefer-offline --no-audit --no-fund",
                command,
            )
        )
        try:
            build_exec = await build_sb.exec_cmd(
                f"bash -lc {shlex.quote(script)}",
                timeout_sec=question.http_build_timeout,
            )
            holder["log"] = "\n".join(
                part for part in (build_exec.stdout, build_exec.stderr) if part
            )
        except Exception as exc:
            holder["log"] = f"Build command failed: {exc}"

        remote_source = ""
        for directory_name in DEFAULT_BUILD_DIRS:
            candidate = f"{remote_project}/{directory_name}"
            check = await build_sb.exec_cmd(
                f"test -f {shlex.quote(f'{candidate}/index.html')}",
                timeout_sec=30,
            )
            if check.exit_code == 0:
                remote_source = candidate
                holder["entry"] = "index.html"
                break

        if not remote_source:
            for directory_name in SSR_BUILD_DIRS:
                candidate = f"{remote_project}/{directory_name}"
                check = await build_sb.exec_cmd(
                    f"test -d {shlex.quote(candidate)}",
                    timeout_sec=30,
                )
                if check.exit_code == 0:
                    remote_source = CONTAINER_WORKSPACE
                    holder["ssr"] = True
                    break

        if not remote_source:
            find_result = await build_sb.exec_cmd(
                f"find {shlex.quote(remote_project)} -maxdepth 2 -name '*.html' "
                "-not -path '*/node_modules/*' | head -n 1",
                timeout_sec=30,
            )
            found = find_result.stdout.strip()
            if found and found.startswith(remote_project):
                remote_source = CONTAINER_WORKSPACE
                project_prefix = relative_project.as_posix()
                relative_entry = found[len(remote_project) :].lstrip("/")
                holder["entry"] = (
                    relative_entry
                    if project_prefix == "."
                    else f"{project_prefix}/{relative_entry}"
                )

        if not remote_source:
            holder["error"] = "No build output found"
            return
        await build_sb.download_directory(
            remote_source,
            export_tar,
            exclude_dirs=SNAPSHOT_EXCLUDES,
        )

    spec = SandboxSpec(
        image=judge_image,
        sandbox_config=ctx.sandbox_config,
        timeout_sec=max(120, question.http_build_timeout + 120),
        on_setup=setup,
    )
    sandbox_result = await Sandbox(spec).run()
    if sandbox_result.error and not export_tar.is_file():
        holder["error"] = sandbox_result.error.message
    if not export_tar.is_file():
        result = FunctionBuildResult(
            success=False,
            error_message=str(holder["error"] or "Build artifacts were not exported"),
            build_log=str(holder["log"]),
        )
        return result

    _replace_directory_from_archive(export_tar, build_dir)
    export_tar.unlink(missing_ok=True)
    result = FunctionBuildResult(
        success=True,
        artifact_dir=build_dir,
        entry_file=Path(holder["entry"]) if holder["entry"] else None,
        project_relative=relative_project if holder["ssr"] else Path("."),
        is_ssr=bool(holder["ssr"]),
        build_log=str(holder["log"]),
    )
    return result


async def _evaluate_one(
    *,
    question: MTACIFBenchQuestion,
    item: str,
    item_index: int,
    ctx: EvalContext,
    function_dir: Path,
    source_dir: Path,
    eval_mode: str,
    build_result: FunctionBuildResult | None,
    function_model: ModelConfig,
    function_agent: AgentConfig,
    semaphore: asyncio.Semaphore,
    judge_image: str,
) -> FunctionCheckResult:
    artifact_dir = function_dir / "checks" / f"check_{item_index}"
    result_path = artifact_dir / "result.json"

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt = _function_prompt(question, item, eval_mode)
    (artifact_dir / "judge_prompt.txt").write_text(prompt, encoding="utf-8")
    start = time.monotonic()

    async def setup(eval_sb: Sandbox) -> None:
        await eval_sb.upload_directory(source_dir, CONTAINER_WORKSPACE)
        if eval_mode != "http":
            return
        if build_result is not None and build_result.is_ssr:
            await _start_ssr_server(
                eval_sb,
                source_dir,
                build_result.project_relative,
                question.http_port,
            )
        else:
            await _start_static_server(
                eval_sb,
                question.http_port,
                build_result.entry_file if build_result else None,
            )
        await _wait_for_http_server(eval_sb, question.http_port)

    check_result: FunctionCheckResult
    try:
        async with semaphore:
            try_dir = artifact_dir / "run"
            spec = SandboxSpec(
                image=judge_image,
                sandbox_config=ctx.sandbox_config,
                prompt=prompt,
                agent_config=function_agent,
                model_cfg=function_model,
                output_dir=str(try_dir),
                env_vars=dict(function_agent.envs),
                workspace=CONTAINER_JUDGE_WORKDIR,
                timeout_sec=question.eval_timeout,
                on_setup=setup,
            )
            sandbox_result = await Sandbox(spec).run()
        score, raw_output, failure = _parse_function_result(sandbox_result, try_dir)
        check_result = FunctionCheckResult(
            id=item_index,
            description=item,
            score=score,
            reason=raw_output or failure or "Not evaluated",
            evaluation_error=(Error(code=-1, message=failure) if failure else None),
            duration=time.monotonic() - start,
        )
        if raw_output:
            (artifact_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
    except Exception as exc:
        message = f"Evaluation failed: {str(exc)[:1500]}"
        logger.warning(
            "[{} check:{}] {}",
            question.qid(),
            item_index,
            message,
        )
        check_result = FunctionCheckResult(
            id=item_index,
            description=item,
            score=None,
            reason=message,
            evaluation_error=Error(code=-1, message=message),
            duration=time.monotonic() - start,
        )
    result_path.write_text(
        json.dumps(check_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return check_result


def _parse_function_result(
    sandbox_result: SandboxResult,
    output_dir: Path,
) -> tuple[float | None, str, str]:
    primary = (
        sandbox_result.last_assistant.content_text
        if sandbox_result.last_assistant is not None
        else ""
    )
    command_output = sandbox_result.last.stdout if sandbox_result.last is not None else ""
    candidates = collect_judge_candidates(primary, command_output, output_dir / "traces")
    raw_output = candidates[0] if candidates else primary or command_output
    if sandbox_result.error:
        return None, raw_output, sandbox_result.error.message[:2000]
    for candidate in candidates:
        score = _parse_function_verdict(candidate)
        if score is not None:
            return score, candidate, ""
    return None, raw_output, "functional judge returned no parseable conclusion"


def _parse_function_verdict(text: str) -> float | None:
    normalized = str(text or "")
    negative = "判断结论：该项目不符合要求"
    positive = "判断结论：该项目符合要求"
    if negative in normalized and positive in normalized:
        return None
    if positive in normalized:
        return 1.0
    if negative in normalized:
        return 0.0
    if (
        "该项目符合要求" in normalized
        and "如果该项目" not in normalized
        and "请输出" not in normalized
    ):
        return 1.0
    if (
        "该项目不符合要求" in normalized
        and "如果该项目" not in normalized
        and "请输出" not in normalized
    ):
        return 0.0
    return None


def _function_prompt(
    question: MTACIFBenchQuestion,
    item: str,
    eval_mode: str,
) -> str:
    if eval_mode == "http":
        return FUNCTION_EVALUATION_HTTP_PROMPT.format(
            task_description=question.task_description,
            workspace_path=CONTAINER_WORKSPACE,
            project_url=f"http://localhost:{question.http_port}",
            checklist_item_description=item,
        )
    return FUNCTION_EVALUATION_PROMPT.format(
        task_description=question.task_description,
        workspace_path=CONTAINER_WORKSPACE,
        checklist_item_description=item,
    )


async def _start_static_server(
    sb: Sandbox,
    port: int,
    entry_file: Path | None,
) -> None:
    if entry_file is not None and entry_file.as_posix() != "index.html":
        root_index = await sb.exec_cmd("test -f /workspace/index.html", timeout_sec=30)
        if root_index.exit_code != 0:
            target = entry_file.as_posix().lstrip("/")
            await sb.write_file(
                "/workspace/index.html",
                f'<meta http-equiv="refresh" content="0;url=/{target}">',
            )
    template = HTTP_SERVER_TEMPLATE.read_text(encoding="utf-8")
    server_script = template.replace("__PORT__", str(port)).replace(
        "__WORKSPACE__", CONTAINER_WORKSPACE
    )
    await sb.write_file("/tmp/server.js", server_script)
    start = await sb.exec_cmd(
        "setsid node /tmp/server.js > /tmp/server.log 2>&1 &",
        timeout_sec=30,
    )
    if start.exit_code != 0:
        raise RuntimeError(start.stderr or "failed to start static HTTP server")


async def _start_ssr_server(
    sb: Sandbox,
    source_dir: Path,
    project_relative: Path,
    port: int,
) -> None:
    package_path = source_dir / project_relative / "package.json"
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        package = {}
    scripts = package.get("scripts", {})
    dependencies = {
        **package.get("dependencies", {}),
        **package.get("devDependencies", {}),
    }
    if "next" in dependencies:
        start_command = f"npm start -- -p {port}"
    elif "nuxt" in dependencies:
        start_command = "npm start"
    elif "start" not in scripts and "preview" in scripts:
        start_command = f"npm run preview -- --port {port}"
    elif "start" not in scripts and "dev" in scripts:
        start_command = f"npm run dev -- --port {port}" if "vite" in dependencies else "npm run dev"
    else:
        start_command = "npm start"
    remote_project = (
        CONTAINER_WORKSPACE
        if str(project_relative) == "."
        else f"{CONTAINER_WORKSPACE}/{project_relative.as_posix()}"
    )
    install = await sb.exec_cmd(
        f"cd {shlex.quote(remote_project)} && npm install --no-audit --no-fund",
        timeout_sec=300,
    )
    if install.exit_code != 0:
        raise RuntimeError(f"SSR npm install failed: {(install.stderr or install.stdout)[-1000:]}")
    start = await sb.exec_cmd(
        f"cd {shlex.quote(remote_project)} && PORT={port} setsid {start_command} "
        "> /tmp/server.log 2>&1 &",
        timeout_sec=60,
    )
    if start.exit_code != 0:
        raise RuntimeError(start.stderr or "failed to start SSR server")


async def _wait_for_http_server(sb: Sandbox, port: int) -> None:
    probe_script = (
        'const http = require("http");'
        f'const request = http.get("http://127.0.0.1:{port}/", (response) => {{'
        "const ok = response.statusCode >= 200 && response.statusCode < 400;"
        "response.resume();"
        'response.on("end", () => process.exit(ok ? 0 : 1));'
        "});"
        "request.setTimeout(3000, () => request.destroy());"
        'request.on("error", () => process.exit(1));'
    )
    for _ in range(60):
        check = await sb.exec_cmd(
            f"node -e {shlex.quote(probe_script)}",
            timeout_sec=10,
        )
        if check.exit_code == 0:
            return
        await asyncio.sleep(1)
    log = await sb.exec_cmd("tail -n 80 /tmp/server.log 2>/dev/null", timeout_sec=30)
    raise RuntimeError(f"HTTP server did not become ready: {(log.stdout or log.stderr)[-2000:]}")


def _npm_build_command(project_dir: Path) -> str:
    try:
        package = json.loads((project_dir / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "npm run build"
    dependencies = {
        **package.get("dependencies", {}),
        **package.get("devDependencies", {}),
    }
    if "vite" in dependencies or "@vitejs/plugin-react" in dependencies:
        return "npm run build -- --base=./"
    if "react-scripts" in dependencies:
        return "PUBLIC_URL=./ npm run build"
    return "npm run build"


def _find_html_entry(directory: Path) -> Path | None:
    root_index = directory / "index.html"
    if root_index.is_file():
        return Path("index.html")
    candidates = sorted(
        (path for path in directory.glob("**/*.html") if "node_modules" not in path.parts),
        key=lambda path: len(path.parts),
    )
    return candidates[0].relative_to(directory) if candidates else None


def _replace_directory_copy(source: Path, destination: Path) -> None:
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.is_dir():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*SNAPSHOT_EXCLUDES),
    )


def _replace_directory_from_archive(archive_path: Path, destination: Path) -> None:
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.is_dir():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:

        def safe_member(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
            try:
                return tarfile.data_filter(member, path)
            except tarfile.FilterError as exc:
                logger.warning("Dropping unsafe tar member {}: {}", member.name, exc)
                return None

        archive.extractall(destination, filter=safe_member)


def _strip_single_wrapper(directory: Path) -> Path:
    entries = list(directory.iterdir())
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else directory
