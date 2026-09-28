"""MTACIFBench helpers: archive extraction, trace slicing, text utilities."""

from __future__ import annotations

import json
import re
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import pysbd  # type: ignore[import-untyped]
from loguru import logger

TOOL_RESULT_PLACEHOLDER = "[工具返回结果已省略]"
IGNORED_WORKSPACE_DIRS = ("node_modules", ".git", "__pycache__")
ROUND_RESULT_EXCERPT_LIMIT = 20000

# Truncate structured API-error payloads before they reach any artifact, so
# request headers, system prompts and credentials are never persisted.
_API_ERROR_DETAIL_PATTERN = re.compile(
    r"(?:\\n|\n)?(?:"
    r"request url:"
    r"|original request headers:"
    r"|request headers:"
    r"|request payload:"
    r"|response headers:"
    r"|response payload:"
    r"|error traceback:"
    r")",
    re.IGNORECASE,
)


def sanitize_api_error_text(value: Any) -> str:
    """Drop the payload tail of a structured ``API Error:`` message."""
    text = str(value or "")
    if not text.lstrip().startswith("API Error:"):
        return text
    summary = text.strip()
    match = _API_ERROR_DETAIL_PATTERN.search(summary)
    if match:
        summary = summary[: match.start()].rstrip()
    return summary[:2000]


def find_agent_api_error(*values: Any) -> str | None:
    """Return a sanitized Claude Code or OpenCode API error."""
    for value in values:
        text = str(value or "")
        for prefix in ("API Error:", "OpenCode error:"):
            marker = text.find(prefix)
            if marker >= 0:
                return sanitize_api_error_text(text[marker:])
    return None


def parse_jsonl_result(content: str) -> str:
    """Extract the final response from Claude Code or OpenCode JSONL."""
    from agent_probe.agents.opencode_output import extract_opencode_error

    last_text = ""
    final_result = ""
    empty_result_summary = ""
    saw_agent_jsonl = False
    for line in str(content or "").splitlines():
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        event_type = data.get("type")
        if event_type in {
            "system",
            "user",
            "assistant",
            "result",
            "step_start",
            "tool_use",
            "text",
            "reasoning",
            "step_finish",
            "error",
        }:
            saw_agent_jsonl = True
        if event_type == "assistant":
            message = data.get("message") or {}
            content_items = message.get("content") or []
            if isinstance(content_items, str):
                assistant_text = content_items.strip()
            else:
                assistant_text = "\n".join(
                    str(item.get("text") or "").strip()
                    for item in content_items
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and str(item.get("text") or "").strip()
                ).strip()
            if assistant_text:
                last_text = assistant_text
        elif event_type == "result":
            result = sanitize_api_error_text(data.get("result"))
            if result:
                final_result = result
            empty_result_summary = (
                "Agent result is empty "
                f"(subtype={data.get('subtype') or 'unknown'}, "
                f"is_error={data.get('is_error')})"
            )
        elif event_type == "text":
            part = data.get("part")
            text = str(part.get("text") or "").strip() if isinstance(part, dict) else ""
            if text:
                last_text = text
    opencode_error = extract_opencode_error(content)
    if opencode_error:
        return f"OpenCode error: {opencode_error}"
    if final_result:
        return final_result
    if last_text:
        return last_text
    if empty_result_summary:
        return empty_result_summary
    if saw_agent_jsonl:
        return "Agent result is empty; no final assistant text was captured"
    return ""


# ---------------------------------------------------------------------------
# Text helpers exposed to dataset validation code
# ---------------------------------------------------------------------------
def count_word(text: str) -> int:
    """Count CJK characters individually and each ASCII run as one word."""
    count = 0
    found_english = False
    for char in text:
        is_chinese_char = "\u4e00" <= char <= "\u9fff"
        is_english_char = re.match(r"^[a-zA-Z0-9]+$", char) is not None
        is_english_punctuation = (
            re.match(r"""[.,;!?(){}\[\]<>:"'`~\-+*/&^%$#@|\\]""", char) is not None
        )
        if is_chinese_char:
            count += 1
            if found_english:
                count += 1
                found_english = False
        elif is_english_char or is_english_punctuation:
            if not found_english:
                found_english = True
        elif char in {" ", "\n"}:
            if found_english:
                count += 1
                found_english = False
        else:
            count += 1
            if found_english:
                count += 1
                found_english = False
    if found_english:
        count += 1
    return count


@lru_cache(maxsize=1)
def _get_sentence_segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="zh")


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip() for sentence in _get_sentence_segmenter().segment(text) if sentence.strip()
    ]


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------
def extract_workspace_archive(archive_path: Path, target_dir: Path) -> Path:
    """Extract a workspace archive, stripping build dirs and unsafe members.

    The archive is produced from a container whose contents the evaluated model
    fully controls, so extraction must keep tarfile's ``data`` protections
    (absolute paths, ``..`` traversal, escaping symlinks, device files, setuid).
    An unsafe member is *dropped* rather than raised, because aborting would
    discard every well-formed file after it — one stray absolute symlink would
    deterministically destroy the whole case.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    def _filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
        parts = member.name.split("/")
        if any(part in IGNORED_WORKSPACE_DIRS for part in parts):
            return None
        try:
            return tarfile.data_filter(member, path)
        except tarfile.FilterError as exc:
            logger.warning("dropping unsafe tar member {}: {}", member.name, exc)
            return None

    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(target_dir, filter=_filter)
    return target_dir


def has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


# ---------------------------------------------------------------------------
# Claude Code trace → per-round operation flow
# ---------------------------------------------------------------------------
def _normalize_content(content: Any) -> Any:
    """Keep the message structure but blank out tool results.

    Tool output is often huge and can carry the contestant's own text back into
    the judge prompt, so it is replaced by a fixed placeholder.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)

    normalized: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            normalized.append({"type": "text", "text": str(item)})
            continue
        entry = dict(item)
        entry.pop("signature", None)
        if entry.get("type") == "tool_use" and entry.get("id"):
            # One field name for both sides of a tool call.
            entry["tool_use_id"] = entry.pop("id")
        if entry.get("type") == "tool_result":
            entry["content"] = TOOL_RESULT_PLACEHOLDER
        normalized.append(entry)
    return normalized


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": str(message.get("role") or ""),
        "content": _normalize_content(message.get("content", "")),
    }


def parse_trace_messages(trace_text: str) -> list[dict[str, Any]]:
    """Parse Claude Code or OpenCode JSONL into normalized messages."""
    messages: list[dict[str, Any]] = []
    for raw_line in trace_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        message = event.get("message")
        if event_type in {"user", "assistant", "message"} and isinstance(message, dict):
            if message.get("role") not in {"user", "assistant"}:
                continue
            messages.append(_normalize_message(message))
            continue
        if event_type in {"text", "reasoning"}:
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            text = str(part.get("text") or "").strip()
            if text:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                )
            continue
        if event_type == "tool_use":
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            raw_state = part.get("state")
            state: dict[str, Any] = raw_state if isinstance(raw_state, dict) else {}
            tool_use_id = str(part.get("id") or "")
            tool_input = state.get("input")
            if not isinstance(tool_input, dict):
                tool_input = {} if tool_input is None else {"value": tool_input}
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "tool_use_id": tool_use_id,
                                "name": str(part.get("tool") or "unknown"),
                                "input": tool_input,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": TOOL_RESULT_PLACEHOLDER,
                                "is_error": state.get("status") == "error",
                            }
                        ],
                    },
                ]
            )
    return messages


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and str(item.get("type") or "") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    if str(message.get("role") or "") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("type") or "") == "tool_result" for item in content
    )


def _normalize_prompt_text(prompt_text: str) -> str:
    return "\n".join(line.strip() for line in str(prompt_text).splitlines() if line.strip())


def _keep_agent_activity(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only what the agent did — never the user instruction itself."""
    return [
        message
        for message in messages
        if str(message.get("role") or "") == "assistant" or _is_tool_result_message(message)
    ]


def _slice_after_prompt(
    messages: list[dict[str, Any]], round_prompt: str
) -> list[dict[str, Any]] | None:
    """Return the messages after the last user message carrying *round_prompt*."""
    normalized_prompt = _normalize_prompt_text(round_prompt)
    if not normalized_prompt:
        return None
    boundary: int | None = None
    for index, message in enumerate(messages):
        if str(message.get("role") or "") != "user":
            continue
        text = _normalize_prompt_text(extract_text_content(message.get("content", "")))
        if not text:
            continue
        if normalized_prompt in text or text in normalized_prompt:
            boundary = index
    if boundary is None:
        return None
    return messages[boundary + 1 :]


def extract_round_context(
    trace_text: str,
    consumed_messages: int,
    round_prompt: str,
) -> tuple[str, int, int]:
    """Slice one round's operation flow out of an accumulating session trace.

    Rounds share one Claude Code session (``keep_session``), so the trace file
    grows across rounds. The primary slice is positional — everything after the
    messages already consumed by earlier rounds — which is exact and needs no
    text matching. If that yields nothing usable (the CLI rewrote or compacted
    the file, so offsets shifted), fall back to slicing after the last
    occurrence of this round's prompt.

    Returns ``(context_json, total_messages, used_message_count)``.
    """
    messages = parse_trace_messages(trace_text)
    total = len(messages)

    activity: list[dict[str, Any]] = []
    if 0 <= consumed_messages <= total:
        activity = _keep_agent_activity(messages[consumed_messages:])
    # A stale offset (the CLI rewrote or compacted the session file) must not
    # degrade into "hand the judge every round" — fall back to the prompt
    # boundary instead, which is still scoped to this round.
    if not activity:
        fallback = _slice_after_prompt(messages, round_prompt)
        if fallback:
            activity = _keep_agent_activity(fallback)
    return (
        json.dumps(activity, ensure_ascii=False, indent=2),
        total,
        len(activity),
    )


def last_assistant_text(context_json: str) -> str:
    """Return the last non-empty assistant text from a serialized context."""
    try:
        messages = json.loads(context_json)
    except json.JSONDecodeError:
        return ""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "assistant":
            continue
        text = extract_text_content(message.get("content", "")).strip()
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_path_component(value: str) -> str:
    """Validate a value used as a single filesystem path component."""
    normalized = str(value or "").strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).is_absolute()
        or Path(normalized).name != normalized
        or any(char in normalized for char in ("/", "\\", "\0", "*", "?", "[", "]"))
    ):
        raise ValueError(f"unsafe MTACIFBench path component: {value!r}")
    return normalized


def diff_round_coverage(
    round_records: list[Any],
    expected_round_ids: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Return (missing, unexpected, duplicate) round ids.

    Coverage is decided per unique round id, not by record count: a duplicated
    record must not mask a missing round.
    """
    expected = set(expected_round_ids)
    seen: set[int] = set()
    duplicates: set[int] = set()
    for record in round_records:
        round_id = int(getattr(record, "round_id", -1))
        if round_id in seen:
            duplicates.add(round_id)
        seen.add(round_id)
    return (
        sorted(expected - seen),
        sorted(seen - expected),
        sorted(duplicates),
    )
