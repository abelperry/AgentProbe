"""MTACIFBench task implementation.

Multi-turn agentic-coding instruction following. One sandbox, one agent
conversation, N rounds in the same workspace; every round is scored against its
own constraint checklist — half of the constraints by dataset-supplied
deterministic checkers, the rest by an LLM judge.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from agent_probe.config import JudgeConfig, ModelConfig
from agent_probe.core.models import Error
from agent_probe.core.sandbox import ExecResult, Sandbox, SandboxResult, SandboxSpec
from agent_probe.core.task import BaseTask
from agent_probe.utils.imports import import_class
from benchmarks.mtacifbench.function_eval import (
    evaluate_function_checklist,
    skipped_function_evaluation,
)
from benchmarks.mtacifbench.models import (
    FAIL_CONCLUSION,
    PASS_CONCLUSION,
    IFCheckResult,
    IFConstraint,
    IFRoundResult,
    MTACIFBenchInference,
    MTACIFBenchJudgement,
    MTACIFBenchQuestion,
    MTACIFRound,
    RoundRecord,
    resolve_judge_image,
)
from benchmarks.mtacifbench.prompts import (
    INSTRUCTION_FOLLOWING_EVALUATION_PROMPT_TEMPLATE,
    MULTIROUND_MAIN_PROMPT_TEMPLATE,
)
from benchmarks.mtacifbench.utils import (
    ROUND_RESULT_EXCERPT_LIMIT,
    diff_round_coverage,
    extract_round_context,
    extract_workspace_archive,
    find_agent_api_error,
    last_assistant_text,
    parse_jsonl_result,
    safe_path_component,
    sanitize_api_error_text,
    write_json,
)
from benchmarks.mtacifbench.validation import (
    collect_judge_candidates,
    parse_check_results,
    run_validation_code,
)

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext

CONTAINER_WORKSPACE = "/workspace"
# The judge agent must not start inside the contestant project, or Claude Code
# would auto-load candidate-controlled CLAUDE.md / .claude/settings as judge
# instructions.
CONTAINER_JUDGE_WORKDIR = "/tmp"
JUDGE_RAW_OUTPUT_EXCERPT_LIMIT = 4000
INCOMPLETE_RERUN_MAX_RETRIES = 2
DEFAULT_PROJECT_INSTRUCTIONS_FILENAME = "CLAUDE.md"
SNAPSHOT_EXCLUDES = ("node_modules", ".git", "__pycache__")
DEFAULT_HTTP_BUILD_TIMEOUT_SEC = 600


class MTACIFBenchTask(BaseTask[MTACIFBenchQuestion, MTACIFBenchInference, MTACIFBenchJudgement]):
    """Multi-round instruction-following benchmark."""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def inference(
        self,
        question: MTACIFBenchQuestion,
        ctx: EvalContext,
    ) -> MTACIFBenchInference:
        """Run the complete multi-round inference, retrying partial attempts."""
        max_attempts = INCOMPLETE_RERUN_MAX_RETRIES + 1
        last_inference: MTACIFBenchInference | None = None
        for attempt in range(1, max_attempts + 1):
            self._reset_inference_attempt(ctx.output_dir, question.qid())
            last_inference = await self._inference_once(question, ctx)
            if last_inference.error is None:
                if attempt > 1:
                    logger.info(
                        "[{}] incomplete inference rerun succeeded on attempt {}/{}",
                        question.qid(),
                        attempt,
                        max_attempts,
                    )
                return last_inference
            if attempt < max_attempts:
                logger.warning(
                    "[{}] inference attempt {}/{} incomplete; retrying: {}",
                    question.qid(),
                    attempt,
                    max_attempts,
                    last_inference.error.message,
                )
        assert last_inference is not None
        return last_inference

    async def _inference_once(
        self,
        question: MTACIFBenchQuestion,
        ctx: EvalContext,
    ) -> MTACIFBenchInference:
        infer_dir = ctx.output_dir / "infer" / safe_path_component(question.qid())
        material_root = infer_dir / "instruction_following"
        round_records: list[RoundRecord] = []
        workspace_tar_path: Path | None = None
        effective_agent_params = dict(ctx.agent_config.params)
        effective_agent_params.pop("append_system_prompt", None)
        effective_agent_config = ctx.agent_config.model_copy(
            update={"params": effective_agent_params}
        )
        project_instruction_candidates = self._project_instruction_candidates(
            effective_agent_config
        )
        project_instructions_filename = project_instruction_candidates[0]
        project_instructions_path = str(
            Path(question.workspace_dir) / project_instructions_filename
        )
        had_original_project_instructions = False
        original_project_instructions_content = ""
        project_instructions_runtime_injected = False
        # Per-question state lives in this closure — the task instance is shared
        # across every question in the dataset.
        state: dict[str, Any] = {
            "round_index": 0 if question.rounds else None,
            "processed_rounds": 0,
            "consumed_messages": 0,
        }

        async def _setup(sb: Sandbox) -> None:
            nonlocal project_instructions_filename
            nonlocal project_instructions_path
            nonlocal had_original_project_instructions
            nonlocal original_project_instructions_content
            nonlocal project_instructions_runtime_injected

            await sb.exec_cmd(f"mkdir -p {shlex.quote(question.workspace_dir)}")
            project_instructions = question.repository_policy.strip()
            for filename in project_instruction_candidates:
                candidate_path = str(Path(question.workspace_dir) / filename)
                quoted_path = shlex.quote(candidate_path)
                exists_result = await sb.exec_cmd(
                    f"if [ -f {quoted_path} ]; then printf 1; else printf 0; fi",
                    timeout_sec=30,
                )
                if exists_result.exit_code != 0:
                    raise RuntimeError(
                        exists_result.stderr or f"failed to inspect {candidate_path}"
                    )
                if not exists_result.stdout.strip().endswith("1"):
                    continue
                project_instructions_filename = filename
                project_instructions_path = candidate_path
                had_original_project_instructions = True
                original_project_instructions_content = await sb.read_file(candidate_path)
                break

            merged_content = self._merge_project_instructions(
                existing_content=original_project_instructions_content,
                project_instructions=project_instructions,
            )
            await sb.write_file(project_instructions_path, merged_content)
            project_instructions_runtime_injected = True

        async def _finish_round(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            """Record the just-finished round and stage its judge material."""
            round_index = state["round_index"]
            if round_index is None:
                return
            if len(sandbox_result.rounds) <= state["processed_rounds"]:
                return
            state["processed_rounds"] = len(sandbox_result.rounds)
            round_spec = question.rounds[round_index]
            exec_result = sandbox_result.rounds[-1]
            context, consumed = self._read_round_context(
                infer_dir=infer_dir,
                session_id=sb.session_id,
                consumed_messages=state["consumed_messages"],
                round_prompt=self._build_main_prompt(question, round_index),
            )
            state["consumed_messages"] = consumed
            response = self._round_response(context, exec_result)
            if not response:
                logger.warning(
                    "[{}] round {} produced no reply",
                    ctx.log_tag(),
                    round_spec.round_id,
                )
            material_dir = await self._stage_round_material(
                sb=sb,
                question=question,
                material_root=material_root,
                round_id=round_spec.round_id,
                context=context,
                response=response,
                project_instructions_filename=project_instructions_filename,
                project_instructions_runtime_injected=project_instructions_runtime_injected,
                had_original_project_instructions=had_original_project_instructions,
                original_project_instructions_content=original_project_instructions_content,
            )
            round_records.append(
                RoundRecord(
                    round_index=round_index,
                    round_id=round_spec.round_id,
                    prompt=self._build_main_prompt(question, round_index),
                    result_response=response,
                    result_excerpt=response[-ROUND_RESULT_EXCERPT_LIMIT:],
                    material_ref=str(material_dir.relative_to(infer_dir)),
                )
            )
            logger.info(
                "[{}] round {} recorded ({} rounds done)",
                ctx.log_tag(),
                round_spec.round_id,
                len(round_records),
            )

        async def _next_round(sb: Sandbox, sandbox_result: SandboxResult) -> str | None:
            await _finish_round(sb, sandbox_result)
            round_index = state["round_index"]
            if round_index is None:
                return None
            # Strictly sequential: no repair, no dependency gating, no skipping.
            next_index = round_index + 1
            if next_index >= len(question.rounds):
                state["round_index"] = None
                return None
            state["round_index"] = next_index
            return self._build_main_prompt(question, next_index)

        async def _complete(sb: Sandbox, sandbox_result: SandboxResult) -> None:
            nonlocal workspace_tar_path
            # Defensive: the engine calls on_nextround after every round, but a
            # trailing round must never be lost if that ever changes.
            await _finish_round(sb, sandbox_result)
            state["round_index"] = None
            write_json(
                infer_dir / "round_records.json",
                [record.model_dump(mode="json") for record in round_records],
            )
            await self._restore_runtime_project_instructions(
                sb=sb,
                project_instructions_path=project_instructions_path,
                project_instructions_runtime_injected=project_instructions_runtime_injected,
                had_original_project_instructions=had_original_project_instructions,
                original_project_instructions_content=original_project_instructions_content,
            )
            target = infer_dir / "workspace.tar.gz"
            try:
                await sb.download_directory(
                    question.workspace_dir,
                    target,
                    exclude_dirs=SNAPSHOT_EXCLUDES,
                )
                workspace_tar_path = target
            except Exception as exc:
                logger.warning("[{}] workspace export failed: {}", ctx.log_tag(), exc)
                workspace_tar_path = None

        spec = SandboxSpec(
            image=question.docker,
            sandbox_config=ctx.sandbox_config,
            prompt=self._build_main_prompt(question, 0)
            if question.rounds
            else question.task_description,
            agent_config=effective_agent_config,
            model_cfg=ctx.model_config,
            output_dir=str(infer_dir),
            env_vars=effective_agent_config.envs,
            workspace=question.workspace_dir,
            timeout_sec=self._inference_execution_timeout(question, ctx.model_config),
            agent_timeout_sec=ctx.model_config.timeout,
            # Constraints span rounds ("keep last round's naming", "every reply
            # must start with ..."), so all rounds share one conversation.
            keep_session=True,
            on_setup=_setup,
            on_complete=_complete,
            on_nextround=_next_round,
        )
        result = await Sandbox(spec).run()
        response = round_records[-1].result_response if round_records else ""

        inference = MTACIFBenchInference(
            response=sanitize_api_error_text(response),
            workspace_tar_path=workspace_tar_path,
            round_records=round_records,
            material_dir=material_root if material_root.exists() else None,
            agent_error=result.error,
        )
        inference.error = self._validate_inference(question, inference, material_root)
        return inference

    @staticmethod
    def _inference_execution_timeout(
        question: MTACIFBenchQuestion,
        model_config: ModelConfig,
    ) -> int:
        round_count = max(1, len(question.rounds))
        per_round_timeout = max(1, int(model_config.timeout))
        build_timeout = max(
            0,
            int(question.http_build_timeout or DEFAULT_HTTP_BUILD_TIMEOUT_SEC),
        )
        setup_and_export_grace = max(600, build_timeout + 300)
        return per_round_timeout * round_count + setup_and_export_grace

    @staticmethod
    def _project_instruction_candidates(agent_config: Any) -> tuple[str, ...]:
        agent_cls = import_class(agent_config.type)
        raw_candidates = getattr(
            agent_cls,
            "project_instruction_filenames",
            (DEFAULT_PROJECT_INSTRUCTIONS_FILENAME,),
        )
        candidates = tuple(str(item).strip() for item in raw_candidates if str(item).strip())
        if not candidates:
            return (DEFAULT_PROJECT_INSTRUCTIONS_FILENAME,)
        if any(Path(item).name != item for item in candidates):
            raise ValueError("project instruction filenames must be plain filenames")
        return candidates

    @staticmethod
    def _merge_project_instructions(
        *,
        existing_content: str,
        project_instructions: str,
    ) -> str:
        normalized_existing = str(existing_content or "").rstrip()
        normalized_instructions = str(project_instructions or "").strip()
        if not normalized_instructions:
            return f"{normalized_existing}\n" if normalized_existing else ""
        if not normalized_existing:
            return f"{normalized_instructions}\n"
        if normalized_existing.endswith(normalized_instructions):
            return f"{normalized_existing}\n"
        return f"{normalized_existing}\n\n{normalized_instructions}\n"

    @staticmethod
    async def _restore_runtime_project_instructions(
        *,
        sb: Sandbox,
        project_instructions_path: str,
        project_instructions_runtime_injected: bool,
        had_original_project_instructions: bool,
        original_project_instructions_content: str,
    ) -> None:
        if not project_instructions_runtime_injected:
            return
        if had_original_project_instructions:
            await sb.write_file(
                project_instructions_path,
                original_project_instructions_content,
            )
            return
        await sb.exec_cmd(
            f"rm -f {shlex.quote(project_instructions_path)}",
            timeout_sec=30,
        )

    @staticmethod
    def _sanitize_snapshot_project_instructions(
        *,
        workspace_path: Path,
        project_instructions_filename: str,
        project_instructions_runtime_injected: bool,
        had_original_project_instructions: bool,
        original_project_instructions_content: str,
    ) -> None:
        if not project_instructions_runtime_injected:
            return
        instructions_path = workspace_path / project_instructions_filename
        if had_original_project_instructions:
            instructions_path.write_text(
                original_project_instructions_content,
                encoding="utf-8",
            )
        elif instructions_path.exists():
            instructions_path.unlink()

    @staticmethod
    def _reset_inference_attempt(output_dir: Path, qid: str) -> None:
        safe_qid = safe_path_component(qid)
        for attempt_path in (
            output_dir / "infer" / safe_qid,
            output_dir / "eval" / safe_qid,
        ):
            if attempt_path.exists():
                shutil.rmtree(attempt_path)

    def _validate_inference(
        self,
        question: MTACIFBenchQuestion,
        inference: MTACIFBenchInference,
        material_root: Path,
    ) -> Error | None:
        """Reject partial inference output so a rerun redoes it.

        Negative codes mean "transient / rerunnable" in this framework, which is
        exactly right here: a half-finished run must not be scored.
        """
        if inference.workspace_tar_path is None or not inference.workspace_tar_path.exists():
            return Error(code=-2, message="workspace archive was not exported")

        missing, unexpected, duplicates = diff_round_coverage(
            inference.round_records,
            [item.round_id for item in question.rounds],
        )
        if missing or unexpected or duplicates:
            return Error(
                code=-2,
                message=(
                    "round records are incomplete: "
                    f"missing={missing}, unexpected={unexpected}, "
                    f"duplicates={duplicates}"
                ),
            )

        for record in inference.round_records:
            material_dir = material_root / f"round_{record.round_id}"
            missing_files = [
                name
                for name in ("context.json", "last_response.txt")
                if not (material_dir / name).is_file()
            ]
            if not (material_dir / "workspace_snapshot").is_dir():
                missing_files.append("workspace_snapshot")
            if missing_files:
                return Error(
                    code=-2,
                    message=(
                        "instruction_following material is incomplete for "
                        f"round_id={record.round_id}: missing={missing_files}"
                    ),
                )
            api_error = find_agent_api_error(record.result_response, record.result_excerpt)
            if api_error:
                return Error(code=-3, message=api_error)
        if inference.agent_error is not None:
            return Error(
                code=inference.agent_error.code,
                message=inference.agent_error.message,
            )
        return None

    @staticmethod
    def _build_main_prompt(question: MTACIFBenchQuestion, round_index: int) -> str:
        return MULTIROUND_MAIN_PROMPT_TEMPLATE.format(
            round_prompt=question.rounds[round_index].prompt
        )

    @staticmethod
    def _round_response(context: str, exec_result: ExecResult) -> str:
        """The round's final reply, as the judge and validators will see it.

        Read it out of *this round's* trace slice, never out of the accumulated
        session: with one shared conversation, ``SandboxResult.last_assistant``
        is the newest reply in the whole session, so a round that produced no
        output would silently inherit the previous round's reply and be scored
        against it. An empty result here is the truth — the round said nothing.
        """
        reply = last_assistant_text(context)
        if reply:
            return sanitize_api_error_text(reply)
        parsed_stdout = parse_jsonl_result(exec_result.stdout)
        parsed_stderr = parse_jsonl_result(exec_result.stderr)
        api_error = find_agent_api_error(
            parsed_stdout,
            parsed_stderr,
            exec_result.stdout,
            exec_result.stderr,
        )
        if api_error:
            return api_error
        return sanitize_api_error_text(
            parsed_stdout
            or parsed_stderr
            or (exec_result.stderr or exec_result.stdout or "").strip()
        )

    @staticmethod
    def _read_round_context(
        infer_dir: Path,
        session_id: str,
        consumed_messages: int,
        round_prompt: str,
    ) -> tuple[str, int]:
        trace_path = infer_dir / "traces" / f"{session_id}.jsonl"
        if not trace_path.is_file():
            logger.warning("trace not found for round slicing: {}", trace_path)
            return "[]", consumed_messages
        trace_text = trace_path.read_text(encoding="utf-8", errors="ignore")
        context, total, used = extract_round_context(trace_text, consumed_messages, round_prompt)
        if not used:
            logger.warning(
                "round context slice is empty (consumed={}, total={})",
                consumed_messages,
                total,
            )
        return context, total

    async def _stage_round_material(
        self,
        *,
        sb: Sandbox,
        question: MTACIFBenchQuestion,
        material_root: Path,
        round_id: int,
        context: str,
        response: str,
        project_instructions_filename: str,
        project_instructions_runtime_injected: bool,
        had_original_project_instructions: bool,
        original_project_instructions_content: str,
    ) -> Path:
        """Snapshot everything the judge will need for this round.

        Staged into a sibling directory and moved into place, so a crash mid-way
        cannot leave a half-written snapshot that later looks complete.
        """
        target_dir = material_root / f"round_{round_id}"
        material_root.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".round_{round_id}.staging-", dir=material_root)
        )
        try:
            snapshot_dir = staging_dir / "workspace_snapshot"
            with tempfile.TemporaryDirectory(prefix="mtacif_snapshot_") as tmp:
                tar_path = Path(tmp) / "workspace.tar.gz"
                await sb.download_directory(
                    question.workspace_dir,
                    tar_path,
                    exclude_dirs=SNAPSHOT_EXCLUDES,
                )
                await asyncio.to_thread(extract_workspace_archive, tar_path, snapshot_dir)
            self._sanitize_snapshot_project_instructions(
                workspace_path=snapshot_dir,
                project_instructions_filename=project_instructions_filename,
                project_instructions_runtime_injected=project_instructions_runtime_injected,
                had_original_project_instructions=had_original_project_instructions,
                original_project_instructions_content=original_project_instructions_content,
            )
            (staging_dir / "context.json").write_text(context, encoding="utf-8")
            (staging_dir / "last_response.txt").write_text(response, encoding="utf-8")

            if target_dir.exists():
                shutil.rmtree(target_dir)
            os.replace(staging_dir, target_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
        return target_dir

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    async def judge(
        self,
        question: MTACIFBenchQuestion,
        inference_result: MTACIFBenchInference,
        ctx: EvalContext,
        prev_judgement: MTACIFBenchJudgement | None = None,
    ) -> MTACIFBenchJudgement:
        # Guard first: an invalid inference must not spin up judge sandboxes to
        # score material that does not exist.
        if inference_result.error is not None:
            return self._error_judgement(
                question,
                inference_result,
                f"inference invalid: {inference_result.error.message}",
            )
        if inference_result.material_dir is None:
            return self._error_judgement(
                question, inference_result, "instruction_following material is missing"
            )
        # A broken judge config fails every round identically; surface it once.
        try:
            judge_config = self._get_judge_config(ctx)
        except Exception as exc:
            return self._error_judgement(
                question,
                inference_result,
                f"invalid judge config: {exc}",
            )
        eval_dir = ctx.output_dir / "eval" / safe_path_component(question.qid())
        material_root = inference_result.material_dir
        cached = self._reusable_round_results(prev_judgement, question)
        semaphore = asyncio.Semaphore(max(1, question.eval_concurrent))

        async def _one(round_spec: MTACIFRound) -> IFRoundResult:
            reused = cached.get(round_spec.round_id)
            if reused is not None:
                logger.info(
                    "[{}] round {} reusing previous verdict",
                    ctx.log_tag(),
                    round_spec.round_id,
                )
                return reused
            async with semaphore:
                return await self._judge_round(
                    question=question,
                    ctx=ctx,
                    round_spec=round_spec,
                    material_dir=material_root / f"round_{round_spec.round_id}",
                    eval_dir=eval_dir,
                )

        # One round raising must not discard the verdicts its siblings earned;
        # a parse_failed round makes _validate_round_judgements fail the
        # question, so it reruns rather than being scored on a partial result.
        settled = await asyncio.gather(
            *[_one(item) for item in question.rounds],
            return_exceptions=True,
        )
        round_results: list[IFRoundResult] = []
        for round_spec, outcome in zip(question.rounds, settled, strict=True):
            # Swallowing CancelledError would defeat an outer timeout.
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                logger.warning(
                    "[{}] round {} judging raised {}: {}",
                    ctx.log_tag(),
                    round_spec.round_id,
                    type(outcome).__name__,
                    outcome,
                )
                round_results.append(
                    IFRoundResult(
                        round_id=round_spec.round_id,
                        passed=False,
                        parse_failed=True,
                        summary="instruction_following judging raised",
                        symptoms=f"{type(outcome).__name__}: {outcome}"[:1200],
                    )
                )
                continue
            round_results.append(outcome)
        if judge_config.function_checklist_eval_enabled:
            function_evaluation = await evaluate_function_checklist(
                question=question,
                inference_result=inference_result,
                ctx=ctx,
                judge_config=judge_config,
                eval_dir=eval_dir,
            )
        else:
            function_evaluation = skipped_function_evaluation(question)
        judgement_error = self._validate_round_judgements(question, round_results)
        all_passed = bool(round_results) and all(item.passed for item in round_results)
        judgement = MTACIFBenchJudgement(
            category=question.category,
            instruction_following_checks=round_results,
            instruction_following_score=1.0 if all_passed else 0.0,
            total_rounds=len(question.rounds),
            round_summaries=self._build_round_summaries(inference_result, round_results),
            response=inference_result.response,
            checks=function_evaluation.checks,
            function_score=function_evaluation.function_score,
            function_checklist_skipped=function_evaluation.function_checklist_skipped,
            build_success=function_evaluation.build_success,
            error=Error(code=-1, message=judgement_error) if judgement_error else None,
        )
        write_json(eval_dir / "eval_result.json", judgement.model_dump(mode="json"))
        return judgement

    @staticmethod
    def _validate_round_judgements(
        question: MTACIFBenchQuestion,
        results: list[IFRoundResult],
    ) -> str | None:
        """Reject a judgement that does not fully cover the dataset checklist.

        Marking the judgement in error keeps the inference and re-runs only the
        judge. Without this, a verdict that silently covers fewer constraints
        than the round declares would be scored as a clean pass.
        """
        expected_ids = [item.round_id for item in question.rounds]
        actual_ids = [item.round_id for item in results]
        if actual_ids != expected_ids:
            return f"IF round coverage mismatch: expected={expected_ids}, actual={actual_ids}"
        for round_item, result in zip(question.rounds, results, strict=True):
            if result.parse_failed:
                return f"round {result.round_id} IF judge output could not be parsed"
            checklist = question.checklist_for(round_item.round_id)
            if len(result.check_results) != len(checklist):
                return (
                    f"round {result.round_id} constraint count mismatch: "
                    f"{len(result.check_results)} != {len(checklist)}"
                )
            for index, (check, expected) in enumerate(
                zip(result.check_results, checklist, strict=True), start=1
            ):
                if check.index != index or check.requirement != expected.constraint:
                    return f"round {result.round_id} requirement {index} does not match dataset"
            expected_passed = all(item.passed for item in result.check_results)
            if result.passed != expected_passed:
                return f"round {result.round_id} aggregate verdict is inconsistent"
        return None

    @staticmethod
    def _reusable_round_results(
        prev_judgement: MTACIFBenchJudgement | None,
        question: MTACIFBenchQuestion,
    ) -> dict[int, IFRoundResult]:
        """Rounds already judged successfully — only unresolved ones re-run."""
        if prev_judgement is None:
            return {}
        expected = {item.round_id: item for item in question.rounds}
        reusable: dict[int, IFRoundResult] = {}
        for result in prev_judgement.instruction_following_checks:
            round_spec = expected.get(result.round_id)
            if round_spec is None or result.parse_failed:
                continue
            if len(result.check_results) != len(question.checklist_for(result.round_id)):
                continue
            reusable[result.round_id] = result
        return reusable

    async def _judge_round(
        self,
        *,
        question: MTACIFBenchQuestion,
        ctx: EvalContext,
        round_spec: MTACIFRound,
        material_dir: Path,
        eval_dir: Path,
    ) -> IFRoundResult:
        round_id = round_spec.round_id
        checklist = question.checklist_for(round_id)
        round_eval_dir = eval_dir / "instruction_following" / f"round_{round_id}"

        if not checklist:
            return self._persist_round_result(
                round_eval_dir,
                eval_dir,
                IFRoundResult(
                    round_id=round_id,
                    passed=True,
                    summary="本轮无 instruction_following 约束",
                ),
                judge_prompt="",
            )

        context_path = material_dir / "context.json"
        response_path = material_dir / "last_response.txt"
        snapshot_dir = material_dir / "workspace_snapshot"
        if not (context_path.is_file() and response_path.is_file() and snapshot_dir.is_dir()):
            return self._persist_round_result(
                round_eval_dir,
                eval_dir,
                IFRoundResult(
                    round_id=round_id,
                    passed=False,
                    parse_failed=True,
                    summary="instruction_following 材料缺失，无法评测",
                    symptoms=f"missing material under {material_dir}",
                ),
                judge_prompt="",
            )

        context = context_path.read_text(encoding="utf-8", errors="ignore")
        response = response_path.read_text(encoding="utf-8", errors="ignore")

        direct, fallback_indices = await self._run_validation_codes(
            question=question,
            round_id=round_id,
            checklist=checklist,
            response=response,
            snapshot_dir=snapshot_dir,
            log_tag=ctx.log_tag(),
        )
        fallback = [checklist[index] for index in fallback_indices]
        if not fallback:
            merged = [direct[index] for index in sorted(direct)]
            return self._persist_round_result(
                round_eval_dir,
                eval_dir,
                self._build_round_result(round_id, merged, ""),
                judge_prompt="",
            )

        judge_prompt = self._build_judge_prompt(question, fallback, context, response)
        parsed: list[IFCheckResult] | None = None
        raw_output = ""
        symptoms = ""
        attempts = max(1, question.judge_parse_retry_max + 1)
        for attempt in range(attempts):
            attempt_dir = round_eval_dir / f"attempt_{attempt}"
            # Rejudging a round that failed to parse lands on the same
            # attempt_N, and collect_judge_candidates scans its traces — so a
            # stale trace could be recorded as this run's raw_output.
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            result = await self._run_judge_sandbox(
                question=question,
                ctx=ctx,
                prompt=judge_prompt,
                snapshot_dir=snapshot_dir,
                output_dir=attempt_dir,
            )
            if result.error is not None:
                symptoms = result.error.message[:1200]
                logger.warning(
                    "[{}] round {} judge run failed (attempt {}): {}",
                    ctx.log_tag(),
                    round_id,
                    attempt,
                    symptoms[:200],
                )
                continue
            primary = result.last_assistant.content_text if result.last_assistant else ""
            command_output = result.last.stdout if result.last else ""
            candidates = collect_judge_candidates(
                primary,
                command_output,
                attempt_dir / "traces",
            )
            raw_output = candidates[0] if candidates else primary or command_output
            for candidate in candidates:
                candidate_results = parse_check_results(candidate, fallback)
                if candidate_results is not None:
                    parsed = candidate_results
                    raw_output = candidate
                    break
            if parsed is not None:
                break
            symptoms = f"期望 {len(fallback)} 项判定，解析失败（attempt {attempt}）"
            logger.warning(
                "[{}] round {} judge output did not parse (attempt {})",
                ctx.log_tag(),
                round_id,
                attempt,
            )

        if parsed is None:
            return self._persist_round_result(
                round_eval_dir,
                eval_dir,
                IFRoundResult(
                    round_id=round_id,
                    passed=False,
                    parse_failed=True,
                    summary="instruction_following 解析失败，需要重判",
                    symptoms=symptoms,
                    raw_output_excerpt=raw_output[-JUDGE_RAW_OUTPUT_EXCERPT_LIMIT:],
                ),
                judge_prompt=judge_prompt,
            )

        merged_map = dict(direct)
        for local_index, original_index in enumerate(fallback_indices):
            item = parsed[local_index]
            merged_map[original_index] = IFCheckResult(
                index=original_index + 1,
                requirement=checklist[original_index].constraint,
                analysis=item.analysis,
                conclusion=item.conclusion,
                source="judge",
            )
        if set(merged_map) != set(range(len(checklist))):
            return self._persist_round_result(
                round_eval_dir,
                eval_dir,
                IFRoundResult(
                    round_id=round_id,
                    passed=False,
                    parse_failed=True,
                    summary="instruction_following 结果合并不完整",
                    symptoms=f"merged={sorted(merged_map)}, expected={len(checklist)}",
                    raw_output_excerpt=raw_output[-JUDGE_RAW_OUTPUT_EXCERPT_LIMIT:],
                ),
                judge_prompt=judge_prompt,
            )
        merged = [merged_map[index] for index in sorted(merged_map)]
        return self._persist_round_result(
            round_eval_dir,
            eval_dir,
            self._build_round_result(round_id, merged, raw_output),
            judge_prompt=judge_prompt,
        )

    async def _run_validation_codes(
        self,
        *,
        question: MTACIFBenchQuestion,
        round_id: int,
        checklist: list[IFConstraint],
        response: str,
        snapshot_dir: Path,
        log_tag: str,
    ) -> tuple[dict[int, IFCheckResult], list[int]]:
        """Score constraints that ship a checker; return the rest for the judge."""
        direct: dict[int, IFCheckResult] = {}
        fallback_indices: list[int] = []
        for index, item in enumerate(checklist):
            code = (item.validation_code or "").strip()
            if not code:
                fallback_indices.append(index)
                continue
            # A malformed verdict line is valid JSON but not an object, so
            # ``verdict.get`` raises; degrade to the judge like every other
            # unknown-verdict path instead of failing the whole question.
            try:
                passed = await asyncio.to_thread(
                    run_validation_code,
                    code,
                    response,
                    snapshot_dir,
                    question.validation_code_timeout,
                    f"[{log_tag} round {round_id} #{index + 1}]",
                )
            except Exception as exc:  # noqa: BLE001 - degrade to the judge
                logger.warning(
                    "[{} round {} #{}] validation code raised {}: {}; falling back to judge",
                    log_tag,
                    round_id,
                    index + 1,
                    type(exc).__name__,
                    exc,
                )
                fallback_indices.append(index)
                continue
            if passed is None:
                fallback_indices.append(index)
                continue
            direct[index] = IFCheckResult(
                index=index + 1,
                requirement=item.constraint,
                analysis=(
                    f"由数据集提供的 instruction_following validation code 判定，返回 {passed}。"
                ),
                conclusion=PASS_CONCLUSION if passed else FAIL_CONCLUSION,
                source="validation_code",
            )
        return direct, fallback_indices

    async def _run_judge_sandbox(
        self,
        *,
        question: MTACIFBenchQuestion,
        ctx: EvalContext,
        prompt: str,
        snapshot_dir: Path,
        output_dir: Path,
    ) -> SandboxResult:
        judge_cfg = self._get_judge_config(ctx)
        judge_image = resolve_judge_image(question, judge_cfg)

        async def _setup(sb: Sandbox) -> None:
            await sb.exec_cmd(f"mkdir -p {CONTAINER_WORKSPACE}")
            await sb.upload_directory(snapshot_dir, CONTAINER_WORKSPACE)

        spec = SandboxSpec(
            image=judge_image,
            sandbox_config=ctx.sandbox_config,
            prompt=prompt,
            agent_config=judge_cfg.agent,
            model_cfg=judge_cfg.model,
            output_dir=str(output_dir),
            env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
            workspace=CONTAINER_JUDGE_WORKDIR,
            timeout_sec=question.eval_timeout,
            on_setup=_setup,
        )
        return await Sandbox(spec).run()

    def _build_judge_prompt(
        self,
        question: MTACIFBenchQuestion,
        checklist: list[IFConstraint],
        context: str,
        response: str,
    ) -> str:
        checklist_text = "\n".join(
            f"[要求{index + 1}]：{item.constraint}" for index, item in enumerate(checklist)
        )
        return INSTRUCTION_FOLLOWING_EVALUATION_PROMPT_TEMPLATE.format(
            workspace_path=CONTAINER_WORKSPACE,
            task_description=question.task_description,
            context=context,
            response=response,
            checklist=checklist_text,
        )

    @staticmethod
    def _build_round_result(
        round_id: int,
        check_results: list[IFCheckResult],
        raw_output: str,
    ) -> IFRoundResult:
        all_passed = all(item.passed for item in check_results)
        return IFRoundResult(
            round_id=round_id,
            passed=all_passed,
            parse_failed=False,
            summary="instruction_following 通过"
            if all_passed
            else "instruction_following 部分约束未满足",
            check_results=check_results,
            raw_output_excerpt=raw_output[-JUDGE_RAW_OUTPUT_EXCERPT_LIMIT:],
        )

    @staticmethod
    def _persist_round_result(
        round_eval_dir: Path,
        eval_dir: Path,
        result: IFRoundResult,
        judge_prompt: str,
    ) -> IFRoundResult:
        """Write judge-owned artifacts under eval/ — never back into infer/."""
        round_eval_dir.mkdir(parents=True, exist_ok=True)
        if judge_prompt:
            (round_eval_dir / "judge_prompt.txt").write_text(judge_prompt, encoding="utf-8")
        result.result_ref = str((round_eval_dir / "round_results.json").relative_to(eval_dir))
        write_json(round_eval_dir / "round_results.json", result.model_dump(mode="json"))
        return result

    @staticmethod
    def _get_judge_config(ctx: EvalContext) -> JudgeConfig:
        config_path = ctx.dataset_config.get_judge_config_path("instruction_following")
        if not config_path:
            raise ValueError("dataset judge_config_path is empty")
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"judge config not found: {path}")
        return JudgeConfig.from_yaml(path)

    @staticmethod
    def _build_round_summaries(
        inference_result: MTACIFBenchInference,
        round_results: list[IFRoundResult],
    ) -> list[dict[str, object]]:
        verdicts = {item.round_id: item for item in round_results}
        summaries: list[dict[str, object]] = []
        for record in inference_result.round_records:
            verdict = verdicts.get(record.round_id)
            summaries.append(
                {
                    "round_index": record.round_index,
                    "round_id": record.round_id,
                    "result_excerpt": record.result_excerpt[-2000:],
                    "passed": None if verdict is None else verdict.passed,
                    "parse_failed": None if verdict is None else verdict.parse_failed,
                    "failed_constraints": []
                    if verdict is None
                    else [item.requirement for item in verdict.check_results if not item.passed],
                }
            )
        return summaries

    def _error_judgement(
        self,
        question: MTACIFBenchQuestion,
        inference_result: MTACIFBenchInference,
        message: str,
    ) -> MTACIFBenchJudgement:
        return MTACIFBenchJudgement(
            category=question.category,
            instruction_following_score=0.0,
            total_rounds=len(question.rounds),
            round_summaries=self._build_round_summaries(inference_result, []),
            response=inference_result.response,
            function_checklist_skipped=True,
            error=Error(code=-1, message=message),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def collect_metrics(
        self, judgements: list[MTACIFBenchJudgement]
    ) -> tuple[dict[str, float], int]:
        total_tasks = len(judgements)
        valid = [
            judgement
            for judgement in judgements
            if judgement is not None and judgement.error is None
        ]
        total_rounds = 0
        passed_rounds = 0
        total_constraints = 0
        passed_constraints = 0

        for judgement in valid:
            rounds = judgement.instruction_following_checks
            for round_result in rounds:
                total_rounds += 1
                if round_result.passed:
                    passed_rounds += 1
                for check in round_result.check_results:
                    total_constraints += 1
                    if check.passed:
                        passed_constraints += 1
        success_count = len(valid)
        scores = {
            "num_total": float(total_tasks),
            "num_success": float(success_count),
            "IFISR": passed_rounds / total_rounds * 100 if total_rounds else 0.0,
            "IFCSR": passed_constraints / total_constraints * 100 if total_constraints else 0.0,
        }
        if any(not item.function_checklist_skipped for item in valid):
            function_valid = [item for item in valid if not item.function_checklist_skipped]
            total_function_checks = sum(len(item.checks) for item in function_valid)
            passed_function_checks = sum(
                check.score == 1.0 for item in function_valid for check in item.checks
            )
            strict_function_passes = sum(
                bool(item.checks) and all(check.score == 1.0 for check in item.checks)
                for item in function_valid
            )
            build_successes = sum(item.build_success is not False for item in function_valid)
            function_count = len(function_valid)
            scores.update(
                {
                    "average": (
                        sum(float(item.function_score) for item in function_valid)
                        / function_count
                        * 100
                        if function_count
                        else 0.0
                    ),
                    "ISR": (
                        strict_function_passes / function_count * 100 if function_count else 0.0
                    ),
                    "CSR": (
                        passed_function_checks / total_function_checks * 100
                        if total_function_checks
                        else 0.0
                    ),
                    "BSR": (build_successes / function_count * 100 if function_count else 0.0),
                }
            )
        return scores, success_count
