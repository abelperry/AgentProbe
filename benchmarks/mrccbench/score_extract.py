"""MRCCBench judge score extraction helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from agent_probe.config import ModelConfig
from agent_probe.model_clients import ModelMessage, create_model_client
from benchmarks.mrccbench.frontend import parse_mrcc_judge_score

EXTRACT_STATUS_OK = "ok"
EXTRACT_STATUS_PARSE_FAILED = "parse_failed"
EXTRACT_STATUS_NO_MODEL_OUTPUT = "no_model_output"
EXTRACT_STATUS_FALLBACK_LLM = "fallback_llm"

FAILURE_KIND_AGENT_ERROR = "agent_error"
FAILURE_KIND_EVAL_TIMEOUT = "eval_timeout"
FAILURE_KIND_EVAL_TIMEOUT_FINAL = "eval_timeout_final"
FAILURE_KIND_EXTRACT_FAILED = "extract_failed"
FAILURE_KIND_INFRA = "infra"
FAILURE_KIND_JUDGE_NO_OUTPUT = "judge_no_output"

EVAL_CHECK_IMMEDIATE_RETRY_KINDS = frozenset(
    {
        FAILURE_KIND_AGENT_ERROR,
        FAILURE_KIND_EXTRACT_FAILED,
        FAILURE_KIND_INFRA,
        FAILURE_KIND_JUDGE_NO_OUTPUT,
    }
)

EVAL_ONLY_RETRY_KINDS = frozenset(
    {
        FAILURE_KIND_AGENT_ERROR,
        FAILURE_KIND_EVAL_TIMEOUT,
        FAILURE_KIND_EXTRACT_FAILED,
        FAILURE_KIND_INFRA,
        FAILURE_KIND_JUDGE_NO_OUTPUT,
    }
)

_NO_OUTPUT_MARKERS = (
    "no conclusion found",
    "no json lines found",
    "empty stdout",
    "not evaluated",
    "empty judge output",
)

_SYSTEM_ONLY_HINTS = (
    '"type": "system"',
    "task_notification",
    "subtype': 'task_notification",
)

_VERDICT_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(?:final\s+)?(?:verdict|assessment|evaluation\s+result)\b"),
    re.compile(r"(?i)\b(?:final\s+)?(?:judge\s+score|final\s+score|overall\s+score)\b"),
    re.compile(r"评测结论|判断结论|判定结论|评估结论"),
    re.compile(r"(?i)\bmrccbench_verdict_json\b"),
    re.compile(r"\*\*(?:符合|不完全符合|不符合)\s*[\(（]"),
    re.compile(r"(?:符合|不完全符合|不符合)\s*[\(（]\s*[01]\.0"),
    re.compile(r"(?i)\b(?:pass|fail|partial(?:ly)?\s+(?:pass|meet))\b"),
    re.compile(r"(?i)(?:score|rating)\s*[:：=]\s*(?:0\.0|0\.5|1\.0)\b"),
    re.compile(r"(?i)requirements?\s+(?:are|is)\s+(?:not\s+)?(?:fully\s+)?satisf"),
    re.compile(r"(?i)all\s+(?:\d+\s+)?(?:checkpoints?|checks?|criteria)\s+(?:pass|met|verified)"),
    re.compile(r"检查点.*(?:通过|失败|符合|不符合|验证)"),
    re.compile(r"(?i)\bconclusion\s*[:：]"),
)


def has_verdict_signals(text: str) -> bool:
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in _VERDICT_SIGNAL_PATTERNS)


def is_eval_check_immediate_retry_kind(failure_kind: str | None) -> bool:
    return str(failure_kind or "") in EVAL_CHECK_IMMEDIATE_RETRY_KINDS


def check_has_retriable_eval_only_pending(check: Any) -> bool:
    score = getattr(check, "score", None)
    if score is not None:
        return False
    failure_kind = str(getattr(check, "failure_kind", None) or FAILURE_KIND_AGENT_ERROR)
    return failure_kind in EVAL_ONLY_RETRY_KINDS


def resolve_check_score(
    *,
    check_description: str,
    output_text: str,
    trace_dir: Path | None = None,
    extract_api: ModelConfig | None = None,
    qid: str = "",
) -> tuple[float | None, str, str]:
    """Resolve a checklist score using regex, trace aggregation, and optional LLM fallback."""

    candidates = _collect_judge_text_candidates(output_text=output_text, trace_dir=trace_dir)
    for candidate in candidates:
        score, reason = parse_mrcc_judge_score(candidate)
        if score is not None:
            return score, reason, EXTRACT_STATUS_OK

    best_text = _select_best_judge_text(candidates)
    failure_reason = best_text or output_text or "Empty judge output"

    if extract_api and has_verdict_signals(failure_reason):
        llm_score, llm_reason = try_llm_extract_score(
            extract_api,
            check_description=check_description,
            text=failure_reason,
            qid=qid,
        )
        if llm_score is not None:
            return llm_score, llm_reason, EXTRACT_STATUS_FALLBACK_LLM

    if _looks_like_no_model_output(failure_reason):
        return None, failure_reason, EXTRACT_STATUS_NO_MODEL_OUTPUT
    return None, failure_reason, EXTRACT_STATUS_PARSE_FAILED


def try_llm_extract_score(
    extract_api: ModelConfig,
    *,
    check_description: str,
    text: str,
    qid: str = "",
) -> tuple[float | None, str]:
    if not extract_api.api_key:
        return None, "extract_api not configured: missing api_key"

    system_prompt = (
        "You extract MRCCBench checklist scores from judge agent output. "
        'Output ONLY one JSON object: {"score": 0|0.5|1, "confidence": 0.0-1.0}. '
        "score must be exactly 0, 0.5, or 1.0. "
        "If the text contains no evaluative conclusion, return "
        '{"score": null, "confidence": 0}.'
    )
    user_prompt = (
        f"Checklist item:\n{check_description}\n\nJudge output tail:\n{_tail_text(text, 10000)}"
    )

    try:
        client = create_model_client(extract_api)
        raw = client.complete(
            [ModelMessage(role="user", content=user_prompt)],
            system=system_prompt,
            temperature=0,
            max_tokens=extract_api.max_tokens,
        )
    except Exception as exc:
        logger.warning("[{}] MRCC extract_api failed: {}", qid, exc)
        return None, f"extract_api error: {exc}"

    parsed = _extract_json_object(raw.content)
    if not parsed:
        return None, "extract_api non-json"
    if parsed.get("score") is None:
        return None, "extract_api: no score in judge output"
    confidence = _coerce_float(parsed.get("confidence"), default=0.0)
    if confidence < 0.35:
        return None, f"extract_api low confidence ({confidence})"
    score = _coerce_discrete_score(parsed.get("score"))
    if score is None:
        return None, "extract_api invalid score value"
    return score, f"[extract_fallback confidence={confidence}] {raw.content[:200]}"


def _collect_judge_text_candidates(*, output_text: str, trace_dir: Path | None) -> list[str]:
    candidates: list[str] = []
    if output_text and output_text.strip():
        candidates.append(output_text.strip())

    if trace_dir and trace_dir.exists():
        for agent_path in sorted(trace_dir.glob("agent_result_*.txt")):
            try:
                raw = agent_path.read_text(encoding="utf-8")
            except Exception:
                continue
            stdout = _extract_stdout_section_from_agent_result(raw)
            if stdout:
                candidates.append(stdout)

        for trace_path in sorted(trace_dir.glob("*.jsonl")):
            candidates.extend(_collect_assistant_text_from_trace(trace_path))

    seen: set[str] = set()
    unique: list[str] = []
    for text in sorted(candidates, key=len, reverse=True):
        stripped = text.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        unique.append(stripped)
    return unique


def _collect_assistant_text_from_trace(trace_path: Path) -> list[str]:
    texts: list[str] = []
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return texts

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        texts.extend(_message_content_texts(message.get("content")))
    return texts


def _message_content_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif part_type == "thinking" and isinstance(part.get("thinking"), str):
            texts.append(part["thinking"])
    return texts


def _select_best_judge_text(candidates: list[str]) -> str:
    if not candidates:
        return ""
    substantive = [text for text in candidates if not _is_short_parse_error(text)]
    pool = substantive or candidates
    with_signals = [text for text in pool if has_verdict_signals(text)]
    if with_signals:
        return max(with_signals, key=len)
    return max(pool, key=len)


def _extract_stdout_section_from_agent_result(raw: str) -> str:
    marker = ">>>>>>> stdout\n"
    start = raw.find(marker)
    if start < 0:
        return raw.strip()
    start += len(marker)
    end = raw.find("\n>>>>>>> stderr\n", start)
    if end < 0:
        return raw[start:].strip()
    return raw[start:end].strip()


def _looks_like_no_model_output(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if any(marker in lower for marker in _NO_OUTPUT_MARKERS):
        return True
    if has_verdict_signals(stripped):
        return False
    if len(stripped) < 80 and any(hint in lower for hint in _SYSTEM_ONLY_HINTS):
        return True
    if re.search(r"[\u4e00-\u9fff]{20,}", stripped):
        return False
    return len(stripped) < 120


def _is_short_parse_error(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    lower = stripped.lower()
    return len(stripped) < 400 and any(marker in lower for marker in _NO_OUTPUT_MARKERS)


def _tail_text(text: str, max_chars: int) -> str:
    stripped = (text or "").strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", stripped, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_discrete_score(raw: Any) -> float | None:
    value = _coerce_float(raw, default=None)
    if value is None:
        return None
    for allowed in (0.0, 0.5, 1.0):
        if abs(value - allowed) < 1e-6:
            return allowed
    return None


def _coerce_float(raw: Any, *, default: float | None) -> float | None:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
