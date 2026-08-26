"""MRCCBench task implementation."""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_probe.config import JudgeConfig, ModelConfig
from agent_probe.core.models import Error
from agent_probe.core.sandbox import ExecResult, Sandbox, SandboxResult, SandboxSpec
from agent_probe.core.task import BaseTask
from benchmarks.mrccbench.frontend import (
    DEFAULT_BUILD_DIRS,
    BuildResult,
    ProjectType,
    calculate_weighted_score,
    detect_project,
    extract_json_object,
    find_entry_html,
    find_project_root,
    get_unique_html_or_svg,
    is_retriable_build_error,
    summarize_failure_symptoms,
    summarize_failure_title,
    wrap_svg_as_html,
)
from benchmarks.mrccbench.models import (
    ChecklistItem,
    CriticalCheck,
    DependencyCheckResult,
    MRCCBenchInference,
    MRCCBenchJudgement,
    MRCCBenchQuestion,
    MRCCCheckResult,
    RoundRecord,
)
from benchmarks.mrccbench.prompts import (
    DEPENDENCY_EVALUATION_PROMPT_TEMPLATE,
    FILE_EVALUATION_PROMPT,
    HTTP_EVALUATION_PROMPT,
    MULTIROUND_MAIN_PROMPT_TEMPLATE,
    REPAIR_PROMPT_TEMPLATE,
)
from benchmarks.mrccbench.score_extract import (
    EXTRACT_STATUS_NO_MODEL_OUTPUT,
    FAILURE_KIND_AGENT_ERROR,
    FAILURE_KIND_EVAL_TIMEOUT_FINAL,
    FAILURE_KIND_EXTRACT_FAILED,
    FAILURE_KIND_JUDGE_NO_OUTPUT,
    check_has_retriable_eval_only_pending,
    is_eval_check_immediate_retry_kind,
    resolve_check_score,
)

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext


CONTAINER_WORKSPACE = "/workspace"
REPAIR_PENALTY_GAMMA = 0.9
# Deployment-specific; set EXTRACT_API_BASE_URL for your gateway.
DEFAULT_EXTRACT_API_BASE_URL = os.environ.get("EXTRACT_API_BASE_URL", "")

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


class MRCCBenchTask(BaseTask[MRCCBenchQuestion, MRCCBenchInference, MRCCBenchJudgement]):
    """Multi-round coding benchmark with dependency repair and checklist judging."""

    _judge_config: JudgeConfig | None = None

    async def inference(
        self,
        question: MRCCBenchQuestion,
        ctx: EvalContext,
    ) -> MRCCBenchInference:
        infer_dir = ctx.output_dir / "infer" / question.qid()
        round_records_path = infer_dir / "round_records.json"
        dependency_checks_path = infer_dir / "dependency_checks.json"
        workspace_tar_path: Path | None = None
        round_records: list[RoundRecord] = []
        dependency_checks: list[DependencyCheckResult] = []
        completed_main_round_ids: set[int] = set()
        blocked_round_ids: set[int] = set()
        blocked_by_failed_rounds: dict[int, set[int]] = {}
        dependency_map = question.dependency_map()
        processed_round_count = 0
        active_turn: dict[str, Any] | None = self._initial_turn(question)

        async def _setup(sb: Sandbox) -> None:
            await sb.exec_cmd(f"mkdir -p {shlex.quote(question.workspace_dir)}")

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            nonlocal workspace_tar_path
            self._write_json(round_records_path, [r.model_dump(mode="json") for r in round_records])
            self._write_json(
                dependency_checks_path,
                [d.model_dump(mode="json") for d in dependency_checks],
            )
            target = infer_dir / "workspace.tar.gz"
            try:
                await sb.download_directory(question.workspace_dir, target)
                workspace_tar_path = target
            except Exception:
                workspace_tar_path = None

        async def _next_round(sb: Sandbox, sandbox_result: SandboxResult) -> str | None:
            nonlocal active_turn, processed_round_count
            if active_turn is None:
                return None
            if len(sandbox_result.rounds) <= processed_round_count:
                return None

            state = dict(active_turn)
            last_result = sandbox_result.rounds[-1]
            processed_round_count = len(sandbox_result.rounds)
            record = self._record_turn(state, last_result, question)
            round_records.append(record)
            if state["kind"] == "main":
                completed_main_round_ids.add(int(state["round_id"]))

            if question.repair:
                dependency_result = await self._run_dependency_check(
                    question=question,
                    ctx=ctx,
                    sb=sb,
                    state=state,
                    dependency_map=dependency_map,
                    infer_dir=infer_dir,
                )
            else:
                dependency_result = self._dependency_result_skip_checks(
                    question,
                    state,
                    dependency_map,
                )
            dependency_checks.append(dependency_result)

            should_continue, next_state, next_prompt = self._decide_next_turn(
                question=question,
                state=state,
                dependency_result=dependency_result,
                completed_main_round_ids=completed_main_round_ids,
                blocked_round_ids=blocked_round_ids,
                blocked_by_failed_rounds=blocked_by_failed_rounds,
            )
            if should_continue and next_state and next_prompt:
                active_turn = next_state
                return next_prompt
            active_turn = None
            return None

        spec = SandboxSpec(
            image=question.docker,
            sandbox_config=ctx.sandbox_config,
            prompt=self._build_main_prompt(question, 0)
            if question.rounds
            else question.task_description,
            agent_config=ctx.agent_config,
            model_cfg=ctx.model_config,
            output_dir=str(infer_dir),
            env_vars=ctx.agent_config.envs if ctx.agent_config else {},
            workspace=question.workspace_dir,
            timeout_sec=ctx.model_config.timeout,
            # Upstream chatglm-eval runs every round in one agent conversation
            # (axec appends --continue from round 2 on); repair rounds only make
            # sense if the model still remembers what it just built.
            keep_session=True,
            on_setup=_setup,
            on_complete=_complete,
            on_nextround=_next_round,
        )
        result = await Sandbox(spec).run()
        output = result.last_assistant.content_text if result.last_assistant else ""
        agent_error = result.error

        if workspace_tar_path is None or not workspace_tar_path.exists():
            return MRCCBenchInference(
                response=output,
                workspace_tar_path=None,
                round_records_path=round_records_path if round_records_path.exists() else None,
                dependency_checks_path=dependency_checks_path
                if dependency_checks_path.exists()
                else None,
                round_records=round_records,
                dependency_checks=dependency_checks,
                agent_error=agent_error,
                error=Error(code=-1, message="workspace artifact was not exported"),
            )

        return MRCCBenchInference(
            response=output,
            workspace_tar_path=workspace_tar_path,
            round_records_path=round_records_path,
            dependency_checks_path=dependency_checks_path,
            round_records=round_records,
            dependency_checks=dependency_checks,
            agent_error=agent_error,
        )

    async def judge(
        self,
        question: MRCCBenchQuestion,
        inference_result: MRCCBenchInference,
        ctx: EvalContext,
        prev_judgement: MRCCBenchJudgement | None = None,
    ) -> MRCCBenchJudgement:
        if inference_result.error:
            return self._error_judgement(
                question,
                inference_result,
                f"Response invalid: {inference_result.error.message}",
            )
        if (
            inference_result.workspace_tar_path is None
            or not inference_result.workspace_tar_path.exists()
        ):
            return self._error_judgement(question, inference_result, "workspace.tar.gz not found")

        eval_dir = ctx.output_dir / "eval" / question.qid()
        workspace_path = eval_dir / "workspace"
        self._extract_workspace(inference_result.workspace_tar_path, workspace_path)
        if not self._has_files(workspace_path):
            return self._error_judgement(
                question, inference_result, "workspace is empty after extraction"
            )

        actual_workspace = find_project_root(workspace_path)
        checklist = question.checklist
        if not checklist:
            return MRCCBenchJudgement(
                category=question.category,
                weighted_score=0.0,
                response=inference_result.response,
                total_rounds=len(question.rounds),
                round_summaries=self._build_round_summaries(inference_result),
                dependency_summary=self._build_dependency_summary(inference_result),
                repair_summary=self._build_repair_summary(inference_result),
            )

        build_result: BuildResult | None = None
        if question.test_mode == "http":
            build_result = await self._prepare_http_build(question, actual_workspace, eval_dir, ctx)
            if not build_result.success:
                checks = self._build_failed_checks(
                    checklist, build_result.error_message, build_result.code_failure
                )
                weighted_score = calculate_weighted_score(checks)
                error = (
                    None
                    if build_result.code_failure
                    else Error(
                        code=-1,
                        message=build_result.error_message,
                    )
                )
                return MRCCBenchJudgement(
                    category=question.category,
                    judge_output=build_result.error_message,
                    check_results=checks,
                    weighted_score=weighted_score,
                    response=inference_result.response,
                    total_rounds=len(question.rounds),
                    round_summaries=self._build_round_summaries(inference_result),
                    dependency_summary=self._build_dependency_summary(inference_result),
                    repair_summary=self._build_repair_summary(inference_result),
                    error=error,
                )
        elif get_unique_html_or_svg(actual_workspace) is None:
            return self._error_judgement(question, inference_result, "No html/svg found")

        previous = self._cached_successful_checks(prev_judgement)
        check_results: list[MRCCCheckResult] = []
        weighted_score: float | None = None
        max_eval_rounds = max(1, int(question.judge_one_retry_max or 1))

        for eval_round in range(max_eval_rounds):
            semaphore = asyncio.Semaphore(max(1, question.eval_concurrent))

            async def _eval(
                item: ChecklistItem,
                current_semaphore: asyncio.Semaphore = semaphore,
            ) -> MRCCCheckResult:
                cached = previous.get(str(item.id))
                if cached is not None:
                    return cached
                async with current_semaphore:
                    return await self._eval_one(
                        question=question,
                        item=item,
                        ctx=ctx,
                        workspace_path=actual_workspace,
                        build_result=build_result,
                        output_dir=eval_dir / f"check_{item.id}",
                    )

            check_results = await asyncio.gather(*[_eval(item) for item in checklist])
            weighted_score = calculate_weighted_score(check_results)
            self._write_eval_result(eval_dir / "eval_result.json", weighted_score, check_results)
            if weighted_score is not None:
                break
            if not any(check_has_retriable_eval_only_pending(check) for check in check_results):
                break
            if eval_round + 1 >= max_eval_rounds:
                break
            previous.update(self._successful_checks_from_results(check_results))

        error = None
        if weighted_score is None:
            error = Error(code=-1, message="Has error checks (eval failed)")

        judge_output = json.dumps(
            {
                "weighted_score": weighted_score,
                "checks": [r.model_dump(mode="json") for r in check_results],
            },
            ensure_ascii=False,
        )
        return MRCCBenchJudgement(
            category=question.category,
            judge_output=judge_output,
            check_results=check_results,
            weighted_score=weighted_score,
            response=inference_result.response,
            total_rounds=len(question.rounds),
            round_summaries=self._build_round_summaries(inference_result),
            dependency_summary=self._build_dependency_summary(inference_result),
            repair_summary=self._build_repair_summary(inference_result),
            error=error,
        )

    def collect_metrics(self, judgements: list[MRCCBenchJudgement]) -> tuple[dict[str, float], int]:
        if not judgements:
            return {
                "num_main_complete": 0.0,
                "average": 0.0,
                "adj-average": 0.0,
                "ISR": 0.0,
                "FPSR": 0.0,
                "CSR": 0.0,
                "BSR": 0.0,
            }, 0

        total = len(judgements)
        success_count = 0
        main_complete = 0
        pass_success = 0
        first_pass_success = 0
        build_success = 0
        total_checks = 0
        passed_checks = 0
        weighted_scores: list[float] = []
        adjusted_scores: list[float] = []

        for judgement in judgements:
            weighted = judgement.weighted_score if judgement.weighted_score is not None else 0.0
            weighted_scores.append(weighted)
            repair_attempts = int(judgement.repair_summary.get("total_repairs") or 0)
            adjusted_scores.append(weighted * (REPAIR_PENALTY_GAMMA**repair_attempts))
            if (
                judgement.error is None
                and judgement.weighted_score is not None
                and all(check.score is not None for check in judgement.check_results)
            ):
                success_count += 1
            if self._completed_all_main_rounds(judgement):
                main_complete += 1
                if judgement.check_results and all(
                    check.score == 1.0 for check in judgement.check_results
                ):
                    pass_success += 1
                    if repair_attempts == 0:
                        first_pass_success += 1
            if self._is_build_success_task(judgement):
                build_success += 1
            for check in judgement.check_results:
                total_checks += 1
                if check.score == 1.0:
                    passed_checks += 1

        return {
            "num_main_complete": float(main_complete),
            "average": sum(weighted_scores) / total * 100,
            "adj-average": sum(adjusted_scores) / total * 100,
            "ISR": pass_success / total * 100,
            "FPSR": first_pass_success / total * 100,
            "CSR": passed_checks / total_checks * 100 if total_checks else 0.0,
            "BSR": build_success / total * 100,
        }, success_count

    @staticmethod
    def _initial_turn(question: MRCCBenchQuestion) -> dict[str, Any] | None:
        if not question.rounds:
            return None
        return {
            "kind": "main",
            "round_index": 0,
            "round_id": question.rounds[0].round_id,
            "attempt": 0,
            "prompt": "",
        }

    @staticmethod
    def _build_main_prompt(question: MRCCBenchQuestion, round_index: int) -> str:
        return MULTIROUND_MAIN_PROMPT_TEMPLATE.format(
            round_prompt=question.rounds[round_index].prompt
        )

    @staticmethod
    def _build_repair_prompt(dependency_result: DependencyCheckResult, attempt: int) -> str:
        symptoms = (
            dependency_result.symptoms
            or dependency_result.summary
            or "自动验收失败，但没有拿到明确的失败表现。"
        )
        return f"第 {attempt} 次修复尝试。\n\n" + REPAIR_PROMPT_TEMPLATE.format(symptoms=symptoms)

    def _record_turn(
        self,
        state: dict[str, Any],
        result: ExecResult,
        question: MRCCBenchQuestion,
    ) -> RoundRecord:
        prompt = str(state.get("prompt") or "")
        if not prompt and state["kind"] == "main":
            prompt = self._build_main_prompt(question, int(state["round_index"]))
        return RoundRecord(
            kind=state["kind"],
            round_index=int(state["round_index"]),
            round_id=int(state["round_id"]),
            attempt=int(state["attempt"]),
            prompt=prompt,
            result_excerpt=self._result_excerpt(result),
        )

    async def _run_dependency_check(
        self,
        *,
        question: MRCCBenchQuestion,
        ctx: EvalContext,
        sb: Sandbox,
        state: dict[str, Any],
        dependency_map: dict[int, CriticalCheck | None],
        infer_dir: Path,
    ) -> DependencyCheckResult:
        round_id = int(state["round_id"])
        critical_check = dependency_map.get(round_id)
        script = self._dependency_check_shell_script(question)
        output = await sb.exec_cmd(
            script, timeout_sec=min(max(question.http_build_timeout, 60), 900)
        )

        build_passed = output.exit_code == 0
        symptoms = "" if build_passed else summarize_failure_symptoms(output.stdout, output.stderr)
        summary = (
            "dependency 检查通过"
            if build_passed
            else summarize_failure_title(symptoms, f"第 {round_id} 轮后的构建/基础运行检查失败")
        )
        critical_passed = critical_check is None
        critical_symptom = ""
        judge_output = ""
        if build_passed and critical_check:
            critical = await self._run_dependency_critical_check(
                question=question,
                ctx=ctx,
                sb=sb,
                state=state,
                critical_check=critical_check,
                infer_dir=infer_dir,
            )
            critical_passed = bool(critical.get("passed"))
            critical_symptom = str(critical.get("symptom") or critical.get("symptoms") or "")
            judge_output = str(critical.get("raw_output") or "")
            if not critical_passed:
                symptoms = critical_symptom
                summary = summarize_failure_title(symptoms, "critical_check 未通过")

        passed = build_passed and critical_passed
        if passed and critical_check:
            summary = "dependency 检查通过，critical_check 通过"

        return DependencyCheckResult(
            kind=str(state["kind"]),
            round_index=int(state["round_index"]),
            round_id=round_id,
            attempt=int(state["attempt"]),
            passed=passed,
            build_passed=build_passed,
            critical_check_passed=critical_passed,
            summary=summary,
            symptoms=symptoms,
            critical_check=critical_check,
            critical_check_symptom=critical_symptom,
            stdout_excerpt=(output.stdout or "")[-1000:],
            stderr_excerpt=(output.stderr or "")[-1000:],
            judge_output_excerpt=judge_output[-1000:],
            project_root=self._extract_project_root(output.stdout) or question.workspace_dir,
            auto_verified=passed,
            verification_note="已自动执行构建/静态入口检查；若存在 critical_check，则额外执行了单轮关键功能判定。",
        )

    async def _run_dependency_critical_check(
        self,
        *,
        question: MRCCBenchQuestion,
        ctx: EvalContext,
        sb: Sandbox,
        state: dict[str, Any],
        critical_check: CriticalCheck,
        infer_dir: Path,
    ) -> dict[str, Any]:
        round_id = int(state["round_id"])
        attempt = int(state["attempt"])
        critical_dir = infer_dir / "critical_checks" / f"round_{round_id}_attempt_{attempt}"
        critical_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace_tar = tmp_path / "workspace.tar.gz"
            workspace = tmp_path / "workspace"
            try:
                await sb.download_directory(question.workspace_dir, workspace_tar)
                self._extract_workspace(workspace_tar, workspace)
                actual_workspace = find_project_root(workspace)
                build_result = None
                if question.test_mode == "http":
                    build_result = await self._prepare_http_build(
                        question, actual_workspace, critical_dir, ctx
                    )
                    if not build_result.success:
                        return {
                            "passed": False,
                            "summary": build_result.error_message,
                            "symptom": build_result.error_message,
                            "raw_output": build_result.error_message,
                        }
                prompt = DEPENDENCY_EVALUATION_PROMPT_TEMPLATE.format(
                    project_url=f"http://localhost:{question.http_port}"
                    if question.test_mode == "http"
                    else "N/A",
                    workspace_path=CONTAINER_WORKSPACE,
                    task_description=question.task_description,
                    round_no=round_id,
                    critical_check=critical_check.description or "无",
                )

                async def _setup(eval_sb: Sandbox) -> None:
                    await self._setup_eval_workspace(
                        eval_sb, question, actual_workspace, build_result
                    )

                judge_cfg = self._get_judge_config(ctx)
                spec = SandboxSpec(
                    image=question.judge_docker,
                    sandbox_config=ctx.sandbox_config,
                    prompt=prompt,
                    agent_config=judge_cfg.agent,
                    model_cfg=judge_cfg.model,
                    output_dir=str(critical_dir),
                    env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
                    workspace=CONTAINER_WORKSPACE,
                    timeout_sec=min(question.eval_timeout, 3600),
                    on_setup=_setup,
                )
                result = await Sandbox(spec).run()
                output = result.last_assistant.content_text if result.last_assistant else ""
                if result.error:
                    return {
                        "passed": False,
                        "summary": f"critical_check 判定执行失败：{result.error.message[:160]}",
                        "symptom": result.error.message[:1200],
                        "raw_output": output,
                    }
                parsed = extract_json_object(output)
                if not parsed:
                    return {
                        "passed": False,
                        "summary": "critical_check 判定未返回可解析 JSON",
                        "symptom": output[-1200:],
                        "raw_output": output,
                    }
                returned_round_id = int(parsed.get("round_id", round_id) or round_id)
                if returned_round_id != round_id:
                    symptom = f"critical_check 返回 round_id={returned_round_id} 与当前轮次 {round_id} 不一致。"
                    return {
                        "passed": False,
                        "summary": symptom,
                        "symptom": symptom,
                        "raw_output": output,
                    }
                return {
                    "passed": bool(parsed.get("passed")),
                    "summary": "critical_check 通过"
                    if parsed.get("passed")
                    else "critical_check 未通过",
                    "symptom": str(parsed.get("symptom") or ""),
                    "raw_output": output,
                }
            except Exception as exc:
                message = str(exc)
                return {
                    "passed": False,
                    "summary": summarize_failure_title(message, "critical_check 判定执行异常"),
                    "symptom": message[:1200],
                    "raw_output": "",
                }

    @staticmethod
    def _dependency_result_skip_checks(
        question: MRCCBenchQuestion,
        state: dict[str, Any],
        dependency_map: dict[int, CriticalCheck | None],
    ) -> DependencyCheckResult:
        round_id = int(state["round_id"])
        return DependencyCheckResult(
            kind=str(state["kind"]),
            round_index=int(state["round_index"]),
            round_id=round_id,
            attempt=int(state["attempt"]),
            passed=True,
            build_passed=True,
            critical_check_passed=True,
            summary="repair=false：未执行每轮 dependency（build / critical_check）",
            critical_check=dependency_map.get(round_id),
            project_root=question.workspace_dir,
            verification_note="repair=false：跳过自动验收（无 build / critical_check）。",
            repair_disabled_skip=True,
        )

    def _decide_next_turn(
        self,
        *,
        question: MRCCBenchQuestion,
        state: dict[str, Any],
        dependency_result: DependencyCheckResult,
        completed_main_round_ids: set[int],
        blocked_round_ids: set[int],
        blocked_by_failed_rounds: dict[int, set[int]],
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        round_id = int(state["round_id"])
        current_index = int(state["round_index"])
        if not dependency_result.passed:
            if int(state["attempt"]) < question.max_repair_attempts:
                next_attempt = int(state["attempt"]) + 1
                prompt = self._build_repair_prompt(dependency_result, next_attempt)
                return (
                    True,
                    {
                        "kind": "repair",
                        "round_index": current_index,
                        "round_id": round_id,
                        "attempt": next_attempt,
                        "prompt": prompt,
                    },
                    prompt,
                )

            next_round = self._find_next_round(
                question,
                after_round_id=round_id,
                completed_main_round_ids=completed_main_round_ids,
                blocked_round_ids=blocked_round_ids
                if question.dependency_failure_skips_downstream
                else set(),
            )
            if question.dependency_failure_skips_downstream:
                for blocked in (
                    dependency_result.critical_check.depended_by_rounds
                    if dependency_result.critical_check
                    else []
                ):
                    if blocked > round_id:
                        blocked_round_ids.add(blocked)
                        blocked_by_failed_rounds.setdefault(blocked, set()).add(round_id)
                        dependency_result.blocked_future_rounds.append(blocked)
            if next_round is None:
                dependency_result.scheduling_note = (
                    f"第 {round_id} 轮在 repair 后仍未通过，且无后续主轮。"
                )
                return False, None, None
            dependency_result.next_runnable_round = int(next_round["round_id"])
            dependency_result.scheduling_note = (
                f"第 {round_id} 轮在 repair 后仍未通过；继续第 {next_round['round_id']} 轮。"
            )
            prompt = self._build_main_prompt(question, int(next_round["round_index"]))
            return (
                True,
                {
                    "kind": "main",
                    "round_index": int(next_round["round_index"]),
                    "round_id": int(next_round["round_id"]),
                    "attempt": 0,
                    "prompt": prompt,
                },
                prompt,
            )

        next_round = self._find_next_round(
            question,
            after_round_id=round_id,
            completed_main_round_ids=completed_main_round_ids,
            blocked_round_ids=blocked_round_ids,
        )
        if next_round is None:
            dependency_result.scheduling_note = "当前之后无更多可执行主轮。"
            return False, None, None
        dependency_result.next_runnable_round = int(next_round["round_id"])
        dependency_result.scheduling_note = f"继续执行第 {next_round['round_id']} 轮。"
        prompt = self._build_main_prompt(question, int(next_round["round_index"]))
        return (
            True,
            {
                "kind": "main",
                "round_index": int(next_round["round_index"]),
                "round_id": int(next_round["round_id"]),
                "attempt": 0,
                "prompt": prompt,
            },
            prompt,
        )

    @staticmethod
    def _find_next_round(
        question: MRCCBenchQuestion,
        *,
        after_round_id: int,
        completed_main_round_ids: set[int],
        blocked_round_ids: set[int],
    ) -> dict[str, int] | None:
        for index, item in enumerate(question.rounds):
            if item.round_id <= after_round_id:
                continue
            if item.round_id in completed_main_round_ids or item.round_id in blocked_round_ids:
                continue
            return {"round_index": index, "round_id": item.round_id}
        return None

    async def _prepare_http_build(
        self,
        question: MRCCBenchQuestion,
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

        return await self._build_npm_project(question, project_info.project_dir, eval_dir, ctx)

    async def _eval_one(
        self,
        *,
        question: MRCCBenchQuestion,
        item: ChecklistItem,
        ctx: EvalContext,
        workspace_path: Path,
        build_result: BuildResult | None,
        output_dir: Path,
    ) -> MRCCCheckResult:
        judge_cfg = self._get_judge_config(ctx)
        start = time.time()

        async def _setup(sb: Sandbox) -> None:
            await self._setup_eval_workspace(sb, question, workspace_path, build_result)

        prompt_history = self._prompts_history(question)
        if question.test_mode == "http":
            prompt = HTTP_EVALUATION_PROMPT.format(
                task_description=question.task_description,
                project_url=f"http://localhost:{question.http_port}",
                workspace_path=CONTAINER_WORKSPACE,
                checklist_item_description=item.description,
                prompts_history=prompt_history,
            )
        else:
            prompt = FILE_EVALUATION_PROMPT.format(
                task_description=question.task_description,
                workspace_path=CONTAINER_WORKSPACE,
                checklist_item_description=item.description,
                prompts_history=prompt_history,
            )

        retries = 0
        max_retries = max(0, question.eval_one_retry_max)
        timeout_retries = 0
        while True:
            spec = SandboxSpec(
                image=question.judge_docker,
                sandbox_config=ctx.sandbox_config,
                prompt=prompt,
                agent_config=judge_cfg.agent,
                model_cfg=judge_cfg.model,
                output_dir=str(output_dir),
                env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
                workspace=CONTAINER_WORKSPACE,
                timeout_sec=question.eval_timeout,
                on_setup=_setup,
            )
            result = await Sandbox(spec).run()
            output = result.last_assistant.content_text if result.last_assistant else ""
            if result.error:
                is_timeout = "timeout" in result.error.message.lower()
                if is_timeout and timeout_retries == 0:
                    timeout_retries = 1
                    continue
                failure_kind = (
                    FAILURE_KIND_EVAL_TIMEOUT_FINAL if is_timeout else FAILURE_KIND_AGENT_ERROR
                )
                if (
                    not is_timeout
                    and retries < max_retries
                    and is_eval_check_immediate_retry_kind(failure_kind)
                ):
                    retries += 1
                    continue
                return MRCCCheckResult(
                    id=item.id,
                    description=item.description,
                    weight=item.weight,
                    score=0.0 if is_timeout else None,
                    reason=f"Execution error: {result.error.message}",
                    duration=time.time() - start,
                    failure_stage="eval_check",
                    failure_kind=failure_kind,
                    eval_timeout_retries=timeout_retries,
                    eval_check_retries=retries,
                )

            score, reason, extract_status = await asyncio.to_thread(
                resolve_check_score,
                check_description=item.description,
                output_text=output,
                trace_dir=output_dir / "traces",
                extract_api=self._extract_api(judge_cfg),
                qid=question.qid(),
            )
            failure_kind = None
            if score is None:
                failure_kind = (
                    FAILURE_KIND_JUDGE_NO_OUTPUT
                    if extract_status == EXTRACT_STATUS_NO_MODEL_OUTPUT
                    else FAILURE_KIND_EXTRACT_FAILED
                )
            if (
                score is not None
                or retries >= max_retries
                or not is_eval_check_immediate_retry_kind(failure_kind)
            ):
                return MRCCCheckResult(
                    id=item.id,
                    description=item.description,
                    weight=item.weight,
                    score=score,
                    reason=reason,
                    duration=time.time() - start,
                    failure_stage=None if score is not None else "eval_check",
                    failure_kind=failure_kind,
                    extract_status=extract_status,
                    eval_timeout_retries=timeout_retries,
                    eval_check_retries=retries,
                )
            retries += 1

    def _get_judge_config(self, ctx: EvalContext) -> JudgeConfig:
        if self._judge_config is None:
            self._judge_config = JudgeConfig.from_yaml(
                Path(ctx.dataset_config.get_judge_config_path("agent_judge_with_playwright"))
            )
        return self._judge_config

    @staticmethod
    def _extract_api(judge_cfg: JudgeConfig) -> ModelConfig:
        if judge_cfg.extract_api:
            return judge_cfg.extract_api
        return ModelConfig(
            base_url=DEFAULT_EXTRACT_API_BASE_URL,
            api_key=os.environ.get("GATEWAY_API_KEY") or os.environ.get("GLM_API_KEY") or "",
            model_name="deepseek-v4-pro",
            format="openai",
            max_tokens=1024,
            timeout=120,
        )

    async def _build_npm_project(
        self,
        question: MRCCBenchQuestion,
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
                tail = (build_output.stderr or build_output.stdout or "")[-800:]
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
        return build_result or BuildResult(
            success=False,
            project_type=ProjectType.NPM,
            error_message="Build result not available",
            code_failure=False,
        )

    async def _setup_eval_workspace(
        self,
        sb: Sandbox,
        question: MRCCBenchQuestion,
        workspace_path: Path,
        build_result: BuildResult | None,
    ) -> None:
        await sb.exec_cmd(f"mkdir -p {CONTAINER_WORKSPACE}")
        if question.test_mode == "http" and build_result:
            if build_result.artifact_tar_path:
                await self._upload_tar(sb, build_result.artifact_tar_path, CONTAINER_WORKSPACE)
            elif build_result.source_dir:
                await sb.upload_directory(build_result.source_dir, CONTAINER_WORKSPACE)
                await self._ensure_index_entry(sb, build_result)
            else:
                await sb.upload_directory(workspace_path, CONTAINER_WORKSPACE)
            await self._start_http_server(sb, question.http_port, build_result.server_side)
        else:
            await sb.upload_directory(workspace_path, CONTAINER_WORKSPACE)

    async def _ensure_index_entry(self, sb: Sandbox, build_result: BuildResult) -> None:
        if (
            build_result.source_dir is None
            or build_result.entry_file is None
            or build_result.entry_file.name == "index.html"
        ):
            return
        rel_entry = build_result.entry_file.relative_to(build_result.source_dir)
        remote_entry = f"{CONTAINER_WORKSPACE}/{rel_entry.as_posix()}"
        await sb.exec_cmd(f"cp {shlex.quote(remote_entry)} {CONTAINER_WORKSPACE}/index.html")

    async def _upload_tar(self, sb: Sandbox, tar_path: Path, remote_dir: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(tmp_path, filter=self._tar_filter)
            await sb.upload_directory(tmp_path, remote_dir)

    async def _start_http_server(self, sb: Sandbox, port: int, server_side: bool) -> None:
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
            await sb.write_file(
                "/tmp/server.js", STATIC_SERVER_JS.format(workspace=CONTAINER_WORKSPACE, port=port)
            )
            await sb.exec_cmd("setsid node /tmp/server.js >/tmp/server.log 2>&1 &")
        await self._wait_for_server(sb, port)

    async def _wait_for_server(self, sb: Sandbox, port: int, max_attempts: int = 40) -> None:
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
        )
        for _ in range(max_attempts):
            await asyncio.sleep(3)
            log = await sb.exec_cmd("cat /tmp/server.log 2>/dev/null || echo ''", 15)
            if any(marker in log.stdout for marker in ready_markers):
                return
            result = await sb.exec_cmd(
                f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 --max-time 10 http://localhost:{port}/ 2>/dev/null || echo '000'",
                20,
            )
            status = result.stdout.strip()
            if status.isdigit() and int(status) > 0:
                return
        log = await sb.exec_cmd("cat /tmp/server.log 2>/dev/null || echo 'no log'")
        raise RuntimeError(f"HTTP server failed to start. Log: {log.stdout[:500]}")

    async def _find_remote_artifact(self, sb: Sandbox) -> tuple[str | None, bool]:
        for dirname in ["out", "dist", "build", "public", "_site", "www"]:
            check = await sb.exec_cmd(f"test -d {CONTAINER_WORKSPACE}/{dirname}")
            if check.exit_code == 0:
                return f"{CONTAINER_WORKSPACE}/{dirname}", False
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
    def _dependency_check_shell_script(question: MRCCBenchQuestion) -> str:
        cleanup_dirs = " ".join(DEFAULT_BUILD_DIRS + [".turbo"])
        port = question.http_port
        workspace = shlex.quote(question.workspace_dir)
        return f"""set -u
PROJECT_ROOT=$(python - <<'PY'
from pathlib import Path
root = Path({question.workspace_dir!r})
if (root / "workspace").is_dir() and not (root / "package.json").exists() and not (root / "index.html").exists():
    root = root / "workspace"
candidates = [root]
for child in sorted(root.iterdir()) if root.exists() else []:
    if child.is_dir() and child.name not in ("node_modules", ".git", "__pycache__"):
        candidates.append(child)
def score(path):
    s = 0
    if (path / "index.html").exists():
        s += 100
    if (path / "package.json").exists():
        s += 80
    if any(k in path.name.lower() for k in ("frontend", "client", "web", "ui", "app")):
        s += 20
    return s
marked = [p for p in candidates if (p / "package.json").exists() or (p / "index.html").exists()]
print(max(marked, key=score).as_posix() if marked else root.as_posix(), end="")
PY
)
cd "$PROJECT_ROOT"
echo "使用项目根目录: $PROJECT_ROOT"
for BUILD_DIR in {cleanup_dirs}; do
  rm -rf "$BUILD_DIR"
done
if [ -f package.json ]; then
  export NODE_ENV=development
  export npm_config_fund=false
  export npm_config_audit=false
  export npm_config_progress=false
  export npm_config_loglevel=error
  export npm_config_registry=https://registry.npmmirror.com
  export npm_config_fetch_retries=3
  if [ ! -d node_modules ]; then
    npm install --prefer-offline --no-audit --no-fund
  fi
  CHECK_PLAN=$(node - <<'NODE'
const fs = require('fs');
try {{
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const scripts = pkg.scripts || {{}};
  const deps = Object.assign({{}}, pkg.dependencies || {{}}, pkg.devDependencies || {{}});
  let mode = 'none';
  let command = '';
  if (scripts.build) {{
    mode = 'build';
    if (deps.vite || deps['@vitejs/plugin-react']) command = 'npm run build -- --base=./';
    else if (deps['react-scripts']) command = 'PUBLIC_URL=./ npm run build';
    else command = 'npm run build';
  }} else if (scripts.start) {{
    mode = 'start';
    command = 'npm start';
  }} else if (scripts.dev) {{
    mode = 'start';
    if (deps.vite || deps['@vitejs/plugin-react']) command = 'npm run dev -- --host 0.0.0.0 --port {port}';
    else command = 'npm run dev';
  }}
  process.stdout.write(JSON.stringify({{ mode, command }}));
}} catch (err) {{
  process.stdout.write(JSON.stringify({{ mode: 'none', command: '' }}));
}}
NODE
)
  CHECK_MODE=$(python - <<'PY' "$CHECK_PLAN"
import json, sys
print(json.loads(sys.argv[1]).get("mode", ""), end="")
PY
)
  CHECK_CMD=$(python - <<'PY' "$CHECK_PLAN"
import json, sys
print(json.loads(sys.argv[1]).get("command", ""), end="")
PY
)
  if [ "$CHECK_MODE" = "build" ] && [ -n "$CHECK_CMD" ]; then
    eval "$CHECK_CMD"
  elif [ "$CHECK_MODE" = "start" ] && [ -n "$CHECK_CMD" ]; then
    SERVER_LOG="/tmp/mrccbench_dependency_server.log"
    rm -f "$SERVER_LOG"
    PORT={port} setsid sh -lc "$CHECK_CMD" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    READY=0
    for _attempt in $(seq 1 10); do
      sleep 3
      STATUS=$(curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 --max-time 10 http://localhost:{port}/ 2>/dev/null || echo '000')
      if [ "$STATUS" != "000" ]; then READY=1; break; fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    done
    cat "$SERVER_LOG" || true
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    if [ "$READY" -ne 1 ]; then
      echo "start 脚本未在预期时间内就绪" >&2
      exit 1
    fi
  else
    echo "package.json 存在，但没有 build/start/dev 脚本，视为基础结构已就绪"
  fi
else
  if [ ! -f index.html ]; then
    echo "项目根目录缺少 index.html。纯 HTML 题目必须在项目根生成 index.html。" >&2
    exit 1
  fi
  echo "检测到静态 HTML 入口: index.html"
fi
test -d {workspace}
"""

    @staticmethod
    def _npm_build_command(project_dir: Path) -> str:
        use_pnpm = (project_dir / "pnpm-lock.yaml").exists()
        install = (
            "pnpm install --no-frozen-lockfile" if use_pnpm else "npm install --prefer-offline"
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

    @staticmethod
    def _extract_project_root(stdout: str) -> str | None:
        for line in stdout.splitlines():
            if line.startswith("使用项目根目录:"):
                return line.split(":", 1)[1].strip()
        return None

    @staticmethod
    def _prompts_history(question: MRCCBenchQuestion) -> str:
        if not question.rounds:
            return "- 无历史 prompts"
        return "\n".join(
            f"- 第{item.round_id}轮: {item.prompt or '(空)'}" for item in question.rounds
        )

    @staticmethod
    def _result_excerpt(result: ExecResult, limit: int = 2000) -> str:
        text = result.stdout or result.stderr or ""
        return text[-limit:]

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
        checklist: list[ChecklistItem],
        reason: str,
        code_failure: bool,
    ) -> list[MRCCCheckResult]:
        score = 0.0 if code_failure else None
        failure_kind = "model_code_build" if code_failure else "framework_gap"
        return [
            MRCCCheckResult(
                id=item.id,
                description=item.description,
                weight=item.weight,
                score=score,
                reason=f"Build failed: {reason}",
                failure_stage="eval_build",
                failure_kind=failure_kind,
            )
            for item in checklist
        ]

    @staticmethod
    def _cached_successful_checks(
        prev_judgement: MRCCBenchJudgement | None,
    ) -> dict[str, MRCCCheckResult]:
        if prev_judgement is None:
            return {}
        return {
            str(item.id): item for item in prev_judgement.check_results if item.score is not None
        }

    @staticmethod
    def _successful_checks_from_results(
        check_results: list[MRCCCheckResult],
    ) -> dict[str, MRCCCheckResult]:
        return {str(item.id): item for item in check_results if item.score is not None}

    def _write_eval_result(
        self,
        path: Path,
        weighted_score: float | None,
        check_results: list[MRCCCheckResult],
    ) -> None:
        self._write_json(
            path,
            {
                "weighted_score": weighted_score,
                "checks": [r.model_dump(mode="json") for r in check_results],
            },
        )

    @staticmethod
    def _build_round_summaries(inference_result: MRCCBenchInference) -> list[dict]:
        dependency_map = {
            (item.kind, item.round_index, item.attempt): item
            for item in inference_result.dependency_checks
        }
        summaries: list[dict] = []
        for record in inference_result.round_records:
            dependency = dependency_map.get((record.kind, record.round_index, record.attempt))
            summaries.append(
                {
                    "kind": record.kind,
                    "round_index": record.round_index,
                    "round_id": record.round_id,
                    "attempt": record.attempt,
                    "result_excerpt": record.result_excerpt,
                    "check_excerpt": (dependency.symptoms or dependency.summary)
                    if dependency
                    else "",
                    "dependency_passed": dependency.passed if dependency else None,
                    "scheduling_note": dependency.scheduling_note
                    if dependency
                    else record.skip_reason,
                    "skipped_due_to_failed_rounds": record.skipped_due_to_failed_rounds,
                    "trace_ref": record.trace_ref,
                }
            )
        return summaries

    @staticmethod
    def _build_dependency_summary(inference_result: MRCCBenchInference) -> dict:
        total = len(inference_result.dependency_checks)
        passed = sum(1 for item in inference_result.dependency_checks if item.passed)
        return {
            "total_dependency_checks": total,
            "passed_dependency_checks": passed,
            "failed_dependency_checks": total - passed,
        }

    @staticmethod
    def _build_repair_summary(inference_result: MRCCBenchInference) -> dict:
        dependency_map = {
            (item.kind, item.round_index, item.attempt): item
            for item in inference_result.dependency_checks
        }
        records: list[dict] = []
        successful = 0
        for record in inference_result.round_records:
            if record.kind != "repair":
                continue
            dependency = dependency_map.get((record.kind, record.round_index, record.attempt))
            passed = bool(dependency and dependency.passed)
            if passed:
                successful += 1
            records.append(
                {
                    "round_index": record.round_index,
                    "round_id": record.round_id,
                    "attempt": record.attempt,
                    "result_excerpt": record.result_excerpt,
                    "check_excerpt": (dependency.symptoms or dependency.summary)
                    if dependency
                    else "",
                    "passed": passed,
                    "trace_ref": record.trace_ref,
                }
            )
        return {
            "total_repairs": len(records),
            "successful_repairs": successful,
            "failed_repairs": len(records) - successful,
            "records": records,
        }

    @staticmethod
    def _completed_all_main_rounds(judgement: MRCCBenchJudgement) -> bool:
        if judgement.weighted_score is None or not judgement.round_summaries:
            return False
        completed = sum(1 for item in judgement.round_summaries if item.get("kind") == "main")
        return completed == judgement.total_rounds

    @staticmethod
    def _is_build_success_task(judgement: MRCCBenchJudgement) -> bool:
        if judgement.error:
            message = judgement.error.message
            if "构建" in message or "build" in message.lower():
                return False
        for check in judgement.check_results:
            reason = check.reason
            if "构建" in reason or "build failed" in reason.lower():
                return False
        return True

    @staticmethod
    def _error_judgement(
        question: MRCCBenchQuestion,
        inference_result: MRCCBenchInference,
        message: str,
    ) -> MRCCBenchJudgement:
        return MRCCBenchJudgement(
            category=question.category,
            judge_output=message,
            weighted_score=None,
            response=inference_result.response,
            total_rounds=len(question.rounds),
            round_summaries=MRCCBenchTask._build_round_summaries(inference_result),
            dependency_summary=MRCCBenchTask._build_dependency_summary(inference_result),
            repair_summary=MRCCBenchTask._build_repair_summary(inference_result),
            error=Error(code=-1, message=message),
        )

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
