"""MTACIFBench task implementation.

Multi-turn agentic-coding instruction following. One sandbox, one agent
conversation, N rounds in the same workspace; every round is scored against its
own constraint checklist — half of the constraints by dataset-supplied
deterministic checkers, the rest by an LLM judge.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from agent_probe.config import JudgeConfig
from agent_probe.core.models import Error
from agent_probe.core.sandbox import ExecResult, Sandbox, SandboxResult, SandboxSpec
from agent_probe.core.task import BaseTask
from benchmarks.mtacifbench.models import (
    IFCheckResult,
    IFConstraint,
    IFRoundResult,
    MTACIFBenchInference,
    MTACIFBenchJudgement,
    MTACIFBenchQuestion,
    MTACIFRound,
    RoundRecord,
)
from benchmarks.mtacifbench.prompts import (
    INSTRUCTION_FOLLOWING_EVALUATION_PROMPT,
    INSTRUCTION_FOLLOWING_JUDGE_SYSTEM_PROMPT,
    MULTIROUND_MAIN_PROMPT_TEMPLATE,
)
from benchmarks.mtacifbench.utils import (
    ROUND_RESULT_EXCERPT_LIMIT,
    diff_round_coverage,
    extract_round_context,
    extract_workspace_archive,
    last_assistant_text,
    safe_path_component,
    sanitize_api_error_text,
    write_json,
)
from benchmarks.mtacifbench.validation import run_validation_code

if TYPE_CHECKING:
    from agent_probe.core.executor import EvalContext

CONTAINER_WORKSPACE = "/workspace"
# The judge agent must not start inside the contestant project, or Claude Code
# would auto-load candidate-controlled CLAUDE.md / .claude/settings as judge
# instructions.
CONTAINER_JUDGE_WORKDIR = "/tmp"
PASS_CONCLUSION = "[[满足了该要求]]"
FAIL_CONCLUSION = "[[没有满足该要求]]"
JUDGE_RAW_OUTPUT_EXCERPT_LIMIT = 4000

_CHECK_BLOCK_PATTERN = re.compile(
    r"(?:\*\*)?\[要求(\d+)-开始\](?:\*\*)?\s*\n"
    r"要求：(.*?)\s*\n"
    r"分析：(.*?)\s*\n"
    r"结论：(.*?)\s*\n"
    r"(?:\*\*)?\[要求\1-结束\](?:\*\*)?",
    re.DOTALL,
)
# Negative forms are matched first: "没有满足了该要求" contains "满足了该要求".
_NEGATIVE_CONCLUSION_PATTERNS = (
    r"\[\[\s*没有满足该要求\s*\]\]",
    r"\[\s*没有满足该要求\s*\]",
    r"没有满足该要求",
    r"\[\[\s*没有满足了该要求\s*\]\]",
    r"\[\s*没有满足了该要求\s*\]",
    r"没有满足了该要求",
)
_POSITIVE_CONCLUSION_PATTERNS = (
    r"\[\[\s*满足了该要求\s*\]\]",
    r"\[\s*满足了该要求\s*\]",
    r"满足了该要求",
)


class MTACIFBenchTask(BaseTask[MTACIFBenchQuestion, MTACIFBenchInference, MTACIFBenchJudgement]):
    """Multi-round instruction-following benchmark."""

    _judge_config: JudgeConfig | None = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def inference(
        self,
        question: MTACIFBenchQuestion,
        ctx: EvalContext,
    ) -> MTACIFBenchInference:
        infer_dir = ctx.output_dir / "infer" / safe_path_component(question.qid())
        material_root = infer_dir / "instruction_following"
        round_records: list[RoundRecord] = []
        workspace_tar_path: Path | None = None
        # Per-question state lives in this closure — the task instance is shared
        # across every question in the dataset.
        state: dict[str, Any] = {
            "round_index": 0 if question.rounds else None,
            "processed_rounds": 0,
            "consumed_messages": 0,
        }

        async def _setup(sb: Sandbox) -> None:
            await sb.exec_cmd(f"mkdir -p {shlex.quote(question.workspace_dir)}")

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
            target = infer_dir / "workspace.tar.gz"
            try:
                await sb.download_directory(question.workspace_dir, target)
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
            agent_config=ctx.agent_config,
            model_cfg=ctx.model_config,
            output_dir=str(infer_dir),
            env_vars=ctx.agent_config.envs if ctx.agent_config else {},
            workspace=question.workspace_dir,
            timeout_sec=ctx.model_config.timeout,
            # Constraints span rounds ("keep last round's naming", "every reply
            # must start with ..."), so all rounds share one conversation.
            keep_session=True,
            append_system_prompt=question.system_prompt,
            on_setup=_setup,
            on_complete=_complete,
            on_nextround=_next_round,
        )
        result = await Sandbox(spec).run()
        response = result.last_assistant.content_text if result.last_assistant else ""

        inference = MTACIFBenchInference(
            response=sanitize_api_error_text(response),
            workspace_tar_path=workspace_tar_path,
            round_records=round_records,
            material_dir=material_root if material_root.exists() else None,
            agent_error=result.error,
        )
        inference.error = self._validate_inference(question, inference, material_root)
        return inference

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
        return sanitize_api_error_text((exec_result.stdout or "").strip())

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
                await sb.download_directory(question.workspace_dir, tar_path)
                await asyncio.to_thread(extract_workspace_archive, tar_path, snapshot_dir)
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

        round_results = list(await asyncio.gather(*[_one(item) for item in question.rounds]))
        unresolved = [item.round_id for item in round_results if item.parse_failed]
        all_passed = bool(round_results) and all(
            item.passed and not item.parse_failed for item in round_results
        )
        return MTACIFBenchJudgement(
            category=question.category,
            instruction_following_checks=round_results,
            instruction_following_score=1.0 if all_passed else 0.0,
            total_rounds=len(question.rounds),
            round_summaries=self._build_round_summaries(inference_result, round_results),
            response=inference_result.response,
            error=Error(
                code=-1,
                message=f"instruction_following verdict unresolved for rounds {unresolved}",
            )
            if unresolved
            else None,
        )

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
            if len(result.check_results) != len(round_spec.instruction_following_checklist):
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
        checklist = list(round_spec.instruction_following_checklist)
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
            for candidate in self._parse_candidates(result):
                candidate_results = self._parse_check_results(candidate, fallback)
                if candidate_results is not None:
                    parsed = candidate_results
                    raw_output = candidate
                    break
            if parsed is not None:
                break
            raw_output = next(iter(self._parse_candidates(result)), "")
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
            passed = await asyncio.to_thread(
                run_validation_code,
                code,
                response,
                snapshot_dir,
                question.validation_code_timeout,
                f"[{log_tag} round {round_id} #{index + 1}]",
            )
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

        async def _setup(sb: Sandbox) -> None:
            await sb.exec_cmd(f"mkdir -p {CONTAINER_WORKSPACE}")
            await sb.upload_directory(snapshot_dir, CONTAINER_WORKSPACE)

        spec = SandboxSpec(
            image=question.judge_docker,
            sandbox_config=ctx.sandbox_config,
            prompt=prompt,
            agent_config=judge_cfg.agent,
            model_cfg=judge_cfg.model,
            output_dir=str(output_dir),
            env_vars=judge_cfg.agent.envs if judge_cfg.agent else {},
            workspace=CONTAINER_JUDGE_WORKDIR,
            timeout_sec=question.eval_timeout,
            append_system_prompt=INSTRUCTION_FOLLOWING_JUDGE_SYSTEM_PROMPT,
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
        return INSTRUCTION_FOLLOWING_EVALUATION_PROMPT.format(
            workspace_path=CONTAINER_WORKSPACE,
            task_description=question.task_description,
            checklist=checklist_text,
            context_fence=self._fence_for(context),
            context=context or "[]",
            response_fence=self._fence_for(response),
            response=response,
        )

    @staticmethod
    def _fence_for(payload: str) -> str:
        """A code fence longer than any backtick run inside *payload*.

        The evidence is model-controlled text. A fixed ``` fence would let a
        reply containing backticks close the block early, promoting the rest of
        the evidence to prompt-level text.
        """
        longest = max((len(run) for run in re.findall(r"`+", payload or "")), default=0)
        return "`" * max(3, longest + 1)

    @staticmethod
    def _parse_candidates(result: SandboxResult) -> list[str]:
        """Judge texts worth trying, most authoritative first."""
        candidates: list[str] = []
        if result.last_assistant and result.last_assistant.content_text.strip():
            candidates.append(result.last_assistant.content_text.strip())
        if result.last and (result.last.stdout or "").strip():
            candidates.append(result.last.stdout.strip())
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    @classmethod
    def _parse_check_results(
        cls,
        output: str,
        checklist: list[IFConstraint],
    ) -> list[IFCheckResult] | None:
        """Parse the judge verdict, fail-closed.

        ``None`` means "we did not obtain a verdict" and triggers a retry. It is
        never silently turned into a pass — a degraded judge must not inflate
        scores.
        """
        expected = len(checklist)
        if expected == 0:
            return []
        if not output:
            return None

        results: list[IFCheckResult] = []
        for match in _CHECK_BLOCK_PATTERN.finditer(output):
            conclusion = cls._normalize_conclusion(match.group(4).strip())
            if conclusion is None:
                return None
            results.append(
                IFCheckResult(
                    index=int(match.group(1)),
                    requirement=match.group(2).strip(),
                    analysis=match.group(3).strip(),
                    conclusion=conclusion,
                    source="judge",
                )
            )
        if len(results) != expected:
            return None

        expected_markers = [str(index) for index in range(1, expected + 1)]
        if re.findall(r"\[要求(\d+)-开始\]", output) != expected_markers:
            return None
        if re.findall(r"\[要求(\d+)-结束\]", output) != expected_markers:
            return None
        if [item.index for item in results] != list(range(1, expected + 1)):
            return None
        # The requirement text must match the trusted checklist verbatim, so a
        # requirement forged inside the untrusted evidence cannot be scored.
        for index, item in enumerate(results):
            if item.requirement != checklist[index].constraint.strip():
                return None
        return results

    @staticmethod
    def _normalize_conclusion(conclusion: str) -> str | None:
        """Absorb formatting drift without ever flipping polarity."""
        text = (conclusion or "").strip()
        for pattern in _NEGATIVE_CONCLUSION_PATTERNS:
            if re.search(pattern, text):
                return FAIL_CONCLUSION
        for pattern in _POSITIVE_CONCLUSION_PATTERNS:
            if re.search(pattern, text):
                return PASS_CONCLUSION
        return None

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

    def _get_judge_config(self, ctx: EvalContext) -> JudgeConfig:
        if self._judge_config is None:
            self._judge_config = JudgeConfig.from_yaml(
                Path(ctx.dataset_config.get_judge_config_path("instruction_following"))
            )
        return self._judge_config

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
            error=Error(code=-1, message=message),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def collect_metrics(
        self, judgements: list[MTACIFBenchJudgement]
    ) -> tuple[dict[str, float], int]:
        # Only judgements that actually produced a verdict enter the denominator;
        # infrastructure failures must show up as missing coverage, not as
        # constraint violations.
        valid = [
            judgement
            for judgement in judgements
            if judgement is not None and judgement.error is None
        ]
        strict_pass = 0
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
            if rounds and all(item.passed for item in rounds):
                strict_pass += 1

        return {
            "IFSSR": strict_pass / len(valid) * 100 if valid else 0.0,
            "IFISR": passed_rounds / total_rounds * 100 if total_rounds else 0.0,
            "IFCSR": passed_constraints / total_constraints * 100 if total_constraints else 0.0,
            "num_strict_pass": float(strict_pass),
            "num_rounds": float(total_rounds),
            "num_constraints": float(total_constraints),
        }, len(valid)
