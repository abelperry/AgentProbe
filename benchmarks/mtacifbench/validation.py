"""Execution of dataset-supplied instruction-following validation code.

Roughly half of MTACIFBench constraints ship a deterministic Python checker.
Those run instead of the LLM judge. The dataset authors are trusted, so this is
not a sandbox boundary — but a checker with a stray ``while`` loop or a blocking
call would otherwise pin a judge worker forever with no error surfacing. Each
checker therefore runs in its own short-lived subprocess with a hard timeout,
and any failure degrades to the judge rather than scoring the constraint 0.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.mtacifbench.models import (
    FAIL_CONCLUSION,
    PASS_CONCLUSION,
    IFCheckResult,
    IFConstraint,
)

# Entry-point names the dataset uses, in priority order.
PREFERRED_VALIDATOR_NAMES = (
    "check",
    "verify",
    "check_requirement",
    "check_requirements",
    "check_response",
)

_DRIVER = '''\
"""Run one dataset validation checker and report a JSON verdict on stdout."""

import inspect
import json
import sys

sys.path.insert(0, {repo_root!r})

from emoji import is_emoji  # noqa: E402

from benchmarks.mtacifbench.utils import count_word, split_sentences  # noqa: E402

PREFERRED_NAMES = {preferred_names!r}
MODULE_NAME = "__instruction_following_validation__"


def can_call(func):
    try:
        inspect.signature(func).bind("", "")
    except (TypeError, ValueError):
        return False
    return True


def resolve(namespace):
    for name in PREFERRED_NAMES:
        candidate = namespace.get(name)
        if inspect.isfunction(candidate) and can_call(candidate):
            return candidate

    owned = [
        value
        for value in namespace.values()
        if inspect.isfunction(value)
        and getattr(value, "__module__", "") == MODULE_NAME
        and can_call(value)
    ]
    # Prefer conventional checker names before any other two-argument function,
    # so a two-argument private helper cannot be mistaken for the entry point.
    named = [f for f in owned if f.__name__.startswith(("check", "verify"))]
    for pool in (named, owned):
        if pool:
            return sorted(pool, key=lambda func: func.__name__)[0]
    raise ValueError("no callable validator found in validation code")


def main():
    code_path, response_path, workspace_path = sys.argv[1:4]
    code = open(code_path, encoding="utf-8").read()
    response = open(response_path, encoding="utf-8").read()

    namespace = {{
        "__builtins__": __builtins__,
        "__name__": MODULE_NAME,
        "is_emoji": is_emoji,
        "count_word": count_word,
        "split_sentences": split_sentences,
    }}
    exec(compile(code, "<instruction_following_validation>", "exec"), namespace, namespace)
    validator = resolve(namespace)
    result = validator(response, workspace_path)
    if not isinstance(result, bool):
        raise TypeError(
            "validation code must return bool, got " + type(result).__name__
        )
    return result


try:
    print(json.dumps({{"ok": True, "passed": main()}}))
except Exception as exc:  # noqa: BLE001 - reported to the caller as a verdict
    print(json.dumps({{"ok": False, "error": "{{}}: {{}}".format(type(exc).__name__, exc)}}))
'''


def _repo_root() -> str:
    # benchmarks/mtacifbench/validation.py -> repo root
    return str(Path(__file__).resolve().parents[2])


def run_validation_code(
    code: str,
    response: str,
    workspace_path: Path,
    timeout: int,
    log_tag: str = "",
) -> bool | None:
    """Run one checker. Return its verdict, or ``None`` to fall back to the judge.

    ``None`` covers a timeout, a crash inside the dataset code, an unresolvable
    entry point and a non-bool return — all cases where we learned nothing about
    the constraint and must not charge the model for it.
    """
    # The checker runs with cwd set to a scratch directory, so the workspace it
    # is handed must be absolute. Experiment configs use a relative output_dir
    # ("./output") by default, and a relative path here silently fails every
    # `os.path.exists(workspace_path)` guard — i.e. every workspace-walking
    # checker returns False no matter what the model wrote.
    workspace_path = Path(workspace_path).resolve()
    with tempfile.TemporaryDirectory(prefix="mtacif_validate_") as tmp:
        tmp_path = Path(tmp)
        code_path = tmp_path / "validation_code.py"
        response_path = tmp_path / "response.txt"
        driver_path = tmp_path / "driver.py"
        code_path.write_text(code, encoding="utf-8")
        response_path.write_text(response, encoding="utf-8")
        driver_path.write_text(
            _DRIVER.format(
                repo_root=_repo_root(),
                preferred_names=PREFERRED_VALIDATOR_NAMES,
            ),
            encoding="utf-8",
        )

        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    sys.executable,
                    str(driver_path),
                    str(code_path),
                    str(response_path),
                    str(workspace_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_path,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "{} validation code timed out after {}s; falling back to judge",
                log_tag,
                timeout,
            )
            return None

        stdout = (completed.stdout or "").strip().splitlines()
        if not stdout:
            logger.warning(
                "{} validation code produced no verdict (exit={}, stderr={}); "
                "falling back to judge",
                log_tag,
                completed.returncode,
                (completed.stderr or "")[-300:],
            )
            return None
        try:
            verdict = json.loads(stdout[-1])
        except json.JSONDecodeError:
            logger.warning(
                "{} validation code verdict is not JSON: {}; falling back to judge",
                log_tag,
                stdout[-1][:300],
            )
            return None

        if not verdict.get("ok"):
            logger.warning(
                "{} validation code failed: {}; falling back to judge",
                log_tag,
                str(verdict.get("error"))[:300],
            )
            return None
        return bool(verdict.get("passed"))


def parse_check_results(
    output: str,
    checklist: list[IFConstraint],
) -> list[IFCheckResult] | None:
    """Parse judge blocks using the standard MTAC-IFBench format tolerance."""
    expected_count = len(checklist)
    if expected_count == 0:
        return []
    if not output:
        return None

    results = _parse_marked_check_blocks(output, checklist)
    if results is None or len(results) != expected_count:
        return None
    return results


def _parse_marked_check_blocks(
    output: str,
    checklist: list[IFConstraint],
) -> list[IFCheckResult] | None:
    start_pattern = re.compile(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:[-*+]\s*)?(?:\*\*)?"
        r"[\[【]\s*要求\s*(\d+)\s*[-－—]\s*开始\s*[\]】](?:\*\*)?",
        re.MULTILINE,
    )
    starts = list(start_pattern.finditer(output))
    if not starts:
        return None

    results_by_index: dict[int, IFCheckResult] = {}
    for position, start_match in enumerate(starts):
        index = int(start_match.group(1))
        next_start = starts[position + 1].start() if position + 1 < len(starts) else len(output)
        block = output[start_match.end() : next_start]
        block = _strip_matching_end_marker(block, index)
        parsed = _parse_single_check_block(index, block, checklist)
        if parsed is not None:
            results_by_index[index] = parsed

    expected_indices = set(range(1, len(checklist) + 1))
    if set(results_by_index) != expected_indices:
        return None
    return [results_by_index[index] for index in range(1, len(checklist) + 1)]


def _strip_matching_end_marker(block: str, index: int) -> str:
    end_pattern = re.compile(
        rf"(?:#{{1,6}}\s*)?(?:[-*+]\s*)?(?:\*\*)?"
        rf"[\[【]\s*要求\s*{index}\s*[-－—]\s*结束\s*[\]】](?:\*\*)?",
        re.MULTILINE,
    )
    return end_pattern.sub("", block).strip()


def _parse_single_check_block(
    index: int,
    block: str,
    checklist: list[IFConstraint],
) -> IFCheckResult | None:
    fields = _extract_labeled_fields(block)
    conclusion_text = _clean_parsed_field(fields.get("结论", ""))
    conclusion = normalize_conclusion(conclusion_text)
    if conclusion is None:
        return None

    requirement = _clean_parsed_field(fields.get("要求", ""))
    analysis = _clean_parsed_field(fields.get("分析", ""))
    if _looks_like_prompt_template_placeholder(requirement, analysis, conclusion_text):
        return None

    if not requirement and 1 <= index <= len(checklist):
        requirement = checklist[index - 1].constraint
    elif 1 <= index <= len(checklist):
        # Verdicts are keyed by index, so a judge that renumbers or reorders its
        # blocks attaches them to the wrong constraints -- and the only trace of
        # that is the requirement text disagreeing with the checklist. Warn
        # rather than discard: judges paraphrase far more often than they
        # misalign, and throwing the verdict away would cost a re-judge every
        # time one does.
        expected = checklist[index - 1].constraint.strip()
        if requirement.strip() != expected:
            logger.warning(
                "judge requirement {} does not match the checklist: {!r} vs {!r}",
                index,
                requirement[:120],
                expected[:120],
            )
    return IFCheckResult(
        index=index,
        requirement=requirement,
        analysis=analysis,
        conclusion=conclusion,
        source="judge",
    )


def _extract_labeled_fields(block: str) -> dict[str, str]:
    field_pattern = re.compile(
        r"(?m)^\s*(?:#{1,6}\s*)?(?:[-*+]\s*)?(?:\*\*)?"
        r"(要求|分析|结论)\s*\d*\s*(?:\*\*)?\s*[:：]\s*(.*)$"
    )
    matches = list(field_pattern.finditer(block))
    fields: dict[str, str] = {}
    for position, match in enumerate(matches):
        label = match.group(1)
        if label in fields and label != "结论":
            continue
        next_start = matches[position + 1].start() if position + 1 < len(matches) else len(block)
        inline_value = match.group(2).strip()
        following_value = block[match.end() : next_start].strip()
        if label == "结论":
            value = inline_value
            verdict_pattern = re.compile(
                r"\[\[?\s*(?:没有)?满足了?该要求\s*\]?\]|(?:没有)?满足了?该要求"
            )
            if following_value and not verdict_pattern.search(value):
                for line in following_value.splitlines():
                    stripped_line = line.strip()
                    if verdict_pattern.search(stripped_line):
                        value = f"{value}\n{stripped_line}".strip() if value else stripped_line
                        break
            if not value and following_value:
                value = next(
                    (line.strip() for line in following_value.splitlines() if line.strip()),
                    "",
                )
            fields[label] = value
            continue
        value = inline_value
        if following_value:
            value = f"{value}\n{following_value}".strip() if value else following_value
        fields[label] = value
    return fields


def _clean_parsed_field(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"```+\s*$", "", value).strip()
    value = re.sub(r"^(?:```[^\n]*\n)+", "", value).strip()
    value = re.sub(
        r"(?:#{1,6}\s*)?(?:[-*+]\s*)?(?:\*\*)?"
        r"[\[【]\s*要求\s*\d+\s*[-－—]\s*结束\s*[\]】](?:\*\*)?",
        "",
        value,
    ).strip()
    return value


def _looks_like_prompt_template_placeholder(*values: str) -> bool:
    text = "\n".join(str(value or "") for value in values)
    placeholder_markers = (
        "此处直接给出要求列表",
        "此处结合人工智能助手",
        "此处只能是 [[满足了该要求]] 或 [[没有满足该要求]]",
    )
    return any(marker in text for marker in placeholder_markers)


def normalize_conclusion(conclusion: str) -> str | None:
    """Accept small format drift while preserving verdict polarity."""
    conclusion = str(conclusion or "").strip()
    negative_patterns = (
        r"\[\[\s*没有满足该要求\s*\]\]",
        r"\[\s*没有满足该要求\s*\]",
        r"没有满足该要求",
        r"\[\[\s*没有满足了该要求\s*\]\]",
        r"\[\s*没有满足了该要求\s*\]",
        r"没有满足了该要求",
    )
    positive_patterns = (
        r"\[\[\s*满足了该要求\s*\]\]",
        r"\[\s*满足了该要求\s*\]",
        r"满足了该要求",
    )
    if any(re.search(pattern, conclusion) for pattern in negative_patterns):
        return FAIL_CONCLUSION
    if any(re.search(pattern, conclusion) for pattern in positive_patterns):
        return PASS_CONCLUSION
    return None


def collect_judge_candidates(
    primary_output: str,
    command_output: str,
    trace_dir: Path,
) -> list[str]:
    """Collect standard parser candidates plus agent transport fallbacks."""
    from benchmarks.mtacifbench.utils import (  # local to keep checker driver lean
        extract_text_content,
        parse_jsonl_result,
        parse_trace_messages,
    )

    candidates: list[str] = []

    def add_candidate(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)
        for derived_text in _derive_parse_candidate_texts(text):
            if derived_text and derived_text not in candidates:
                candidates.append(derived_text)

    add_candidate(parse_jsonl_result(command_output))

    for raw_line in str(command_output or "").splitlines():
        try:
            event = json.loads(raw_line.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "assistant":
            message = event.get("message") or {}
            content = message.get("content") or []
            if isinstance(content, str):
                add_candidate(content)
                continue
            if isinstance(content, list):
                text_parts = [
                    str(item.get("text") or "").strip()
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and str(item.get("text") or "").strip()
                ]
                if text_parts:
                    add_candidate("\n".join(text_parts))
            continue
        if event_type == "result":
            add_candidate(event.get("result"))

    add_candidate(command_output)
    add_candidate(primary_output)
    trace_texts: list[str] = []
    if trace_dir.is_dir():
        for trace_path in sorted(trace_dir.glob("*")):
            if not trace_path.is_file():
                continue
            try:
                messages = parse_trace_messages(
                    trace_path.read_text(encoding="utf-8", errors="ignore")
                )
            except OSError:
                continue
            trace_texts.extend(
                extract_text_content(message.get("content"))
                for message in messages
                if message.get("role") == "assistant"
            )
    for text in reversed(trace_texts):
        add_candidate(text)
    return candidates


def _derive_parse_candidate_texts(text: str) -> list[str]:
    """Slice marker spans and fenced blocks out of a candidate.

    A judge that wraps its verdict in prose or a code fence still parses: the
    marker span and each fenced block are offered as additional candidates.
    """
    if not text:
        return []

    candidates: list[str] = []
    marker_pattern = re.compile(r"[\[【]\s*要求\s*\d+\s*[-－—]\s*开始\s*[\]】]")
    first_marker = marker_pattern.search(text)
    last_end_marker = None
    for match in re.finditer(r"[\[【]\s*要求\s*\d+\s*[-－—]\s*结束\s*[\]】]", text):
        last_end_marker = match
    if first_marker and last_end_marker and last_end_marker.end() > first_marker.start():
        candidates.append(text[first_marker.start() : last_end_marker.end()].strip())

    for fence_match in re.finditer(r"```(?:[^\n]*)\n(.*?)```", text, re.DOTALL):
        fenced = fence_match.group(1).strip()
        if marker_pattern.search(fenced):
            candidates.append(fenced)

    return candidates
