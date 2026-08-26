"""MRCCBench frontend helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from benchmarks.mrccbench.models import MRCCCheckResult
from benchmarks.zfrontendbench.frontend import (
    DEFAULT_BUILD_DIRS,
    BuildResult,
    ProjectInfo,
    ProjectType,
    detect_project,
    find_entry_html,
    find_project_root,
    get_unique_html_or_svg,
    is_retriable_build_error,
    wrap_svg_as_html,
)

__all__ = [
    "DEFAULT_BUILD_DIRS",
    "BuildResult",
    "ProjectInfo",
    "ProjectType",
    "calculate_weighted_score",
    "detect_project",
    "find_entry_html",
    "find_project_root",
    "get_unique_html_or_svg",
    "is_retriable_build_error",
    "parse_mrcc_judge_score",
    "wrap_svg_as_html",
]

VERDICT_MARKER = "MRCCBENCH_VERDICT_JSON:"
SYMPTOM_SIGNAL_PATTERN = re.compile(
    r"(error|failed|exception|traceback|cannot|can't|missing|not found|syntaxerror|"
    r"typeerror|referenceerror|module not found|eaddrinuse|enoent|err!)",
    re.IGNORECASE,
)


def calculate_weighted_score(results: list[MRCCCheckResult]) -> float | None:
    if any(result.score is None for result in results):
        return None
    total_weight = sum(float(result.weight or 1.0) for result in results) or 1.0
    weighted_sum = sum(
        float(result.score or 0.0) * float(result.weight or 1.0) for result in results
    )
    return weighted_sum / total_weight


def parse_mrcc_judge_score(output: str) -> tuple[float | None, str]:
    """Parse MRCCBench 0/0.5/1 judge output."""

    if not output:
        return None, "Empty judge output"
    score = _extract_marker_score(output)
    if score is not None:
        return score, output
    score = _extract_score_only_json(output)
    if score is not None:
        return score, output
    score = _extract_text_verdict(output)
    if score is not None:
        return score, output
    return None, output


def summarize_failure_symptoms(stdout: str, stderr: str, max_lines: int = 40) -> str:
    blocks: list[str] = []
    for content in (stderr, stdout):
        if not content:
            continue
        lines = [line.rstrip() for line in content.splitlines() if line.strip()]
        if not lines:
            continue
        matched = [idx for idx, line in enumerate(lines) if SYMPTOM_SIGNAL_PATTERN.search(line)]
        if matched:
            last = matched[-1]
            blocks.append("\n".join(lines[max(0, last - 8) : min(len(lines), last + 12)]))
        else:
            blocks.append("\n".join(lines[-max_lines:]))
    merged = "\n\n".join(block for block in blocks if block).strip()
    if not merged:
        return ""
    lines = merged.splitlines()
    return "\n".join(lines[-max_lines:])


def summarize_failure_title(symptoms: str, fallback: str) -> str:
    if not symptoms:
        return fallback
    for line in symptoms.splitlines():
        text = line.strip(" -:\t")
        if text:
            return text[:160]
    return fallback


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_discrete_score(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        try:
            raw = float(raw.strip())
        except ValueError:
            return None
    if not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    for allowed in (0.0, 0.5, 1.0):
        if abs(value - allowed) < 1e-6:
            return allowed
    return None


def _strip_inline_backticks(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text.startswith("`") and text.endswith("`"):
        return text[1:-1].strip()
    return text


def _extract_marker_score(text: str) -> float | None:
    last: float | None = None
    marker_lower = VERDICT_MARKER.lower()
    for line in text.splitlines():
        stripped = _strip_inline_backticks(line.strip())
        idx = stripped.lower().find(marker_lower)
        if idx < 0:
            continue
        payload = _strip_inline_backticks(stripped[idx + len(VERDICT_MARKER) :])
        parsed = extract_json_object(payload)
        if not parsed:
            continue
        score = _coerce_discrete_score(parsed.get("score", parsed.get("mrccbench_judge_score")))
        if score is not None:
            last = score
    return last


def _extract_score_only_json(text: str) -> float | None:
    lines = [line.strip() for line in text.rstrip().splitlines() if line.strip()]
    if not lines:
        return None
    last = _strip_inline_backticks(lines[-1])
    if not (last.startswith("{") and last.endswith("}")):
        return None
    try:
        parsed = json.loads(last)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not set(parsed) <= {"score", "mrccbench_judge_score"}:
        return None
    return _coerce_discrete_score(parsed.get("score", parsed.get("mrccbench_judge_score")))


def _extract_text_verdict(text: str) -> float | None:
    tail = text[-1200:]
    if "如果该项目" in tail or "请输出" in tail:
        return None
    if re.search(r"(判断|判定)结论\s*[:：]\s*该项目不完全符合要求", tail):
        return 0.5
    if re.search(r"(判断|判定)结论\s*[:：]\s*该项目不符合要求", tail):
        return 0.0
    if re.search(r"(判断|判定)结论\s*[:：]\s*该项目符合要求", tail):
        return 1.0
    bold = re.search(r"\*\*(不完全符合|不符合|符合)\s*[（(]\s*(0\.0|0\.5|1\.0)\s*[）)]\*\*", tail)
    if bold:
        label = bold.group(1)
        if label == "不完全符合":
            return 0.5
        if label == "不符合":
            return 0.0
        return 1.0
    explicit = re.search(
        r"(?i)(?:final|judge|overall)?\s*score\s*[:：=]\s*(0\.0|0\.5|1\.0)\b",
        tail,
    )
    if explicit:
        return _coerce_discrete_score(explicit.group(1))
    return None
