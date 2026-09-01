"""Parse OpenCode's JSONL protocol, and decide whether a turn actually finished.

``opencode run --format json`` streams one JSON object per line. Reading the
final text out of that is easy; the hard question is whether the agent *stopped
because it was done*, which is what a benchmark has to know before scoring the
workspace it left behind.

Neither obvious signal answers it:

* The exit code is 0 both when the model finished and when it ran out of output
  tokens mid-sentence.
* Assistant text exists in both cases too -- an agent that narrates "now I'll
  update the config" and then dies has produced text and no result.

So completeness is read off the *terminal step*: a turn is finished when the
last ``step_finish`` reports a non-truncating reason and that step ended on
text rather than on an unanswered tool call. A step that ends holding a tool
call means the agent was still working when the process went away.
"""

from __future__ import annotations

import json
from typing import Any

_EVENT_TYPES = {
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
}


def _assistant_event_has_text(data: dict[str, Any]) -> bool:
    message = data.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("type") == "text"
        and bool(str(item.get("text") or "").strip())
        for item in content
    )


def _assistant_event_has_tool_use(data: dict[str, Any]) -> bool:
    message = data.get("message")
    if not isinstance(message, dict):
        return False
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return True
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") in {"tool_call", "tool_use"}
        for item in content
    )


def inspect_opencode_jsonl(content: str | None) -> dict[str, Any]:
    """Inspect the terminal OpenCode step without treating tool chatter as a reply."""
    saw_jsonl = False
    saw_step_start = False
    saw_step_finish = False
    current_step_has_text = False
    current_step_has_tool_use = False
    final_step_has_text = False
    final_step_has_tool_use = False
    protocol_final_text = False
    any_assistant_text = False
    has_tool_use = False
    has_error_event = False
    finish_reason: str | None = None

    for line in str(content or "").splitlines():
        try:
            data = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue

        event_type = data.get("type")
        if event_type not in _EVENT_TYPES:
            continue
        saw_jsonl = True

        if event_type == "step_start":
            saw_step_start = True
            current_step_has_text = False
            current_step_has_tool_use = False
        elif event_type == "text":
            part = data.get("part")
            text = str(part.get("text") or "").strip() if isinstance(part, dict) else ""
            if text:
                current_step_has_text = True
                any_assistant_text = True
        elif event_type == "assistant":
            if _assistant_event_has_text(data):
                current_step_has_text = True
                any_assistant_text = True
            if _assistant_event_has_tool_use(data):
                current_step_has_tool_use = True
                has_tool_use = True
        elif event_type == "result":
            result = data.get("result")
            if isinstance(result, str) and result.strip():
                protocol_final_text = True
                any_assistant_text = True
        elif event_type == "tool_use":
            has_tool_use = True
            current_step_has_tool_use = True
        elif event_type == "error":
            has_error_event = True
        elif event_type == "step_finish":
            saw_step_finish = True
            final_step_has_text = current_step_has_text
            final_step_has_tool_use = current_step_has_tool_use
            part = data.get("part")
            reason = part.get("reason") if isinstance(part, dict) else None
            finish_reason = str(reason) if reason is not None else None

    if protocol_final_text:
        has_assistant_text = True
    elif saw_step_finish:
        has_assistant_text = final_step_has_text
    elif saw_step_start:
        has_assistant_text = current_step_has_text
    elif saw_jsonl:
        has_assistant_text = any_assistant_text
    else:
        has_assistant_text = bool(str(content or "").strip())

    return {
        "saw_jsonl": saw_jsonl,
        "has_terminal_event": protocol_final_text or saw_step_finish,
        "has_assistant_text": has_assistant_text,
        "finish_reason": finish_reason,
        "has_tool_use": has_tool_use,
        "final_step_has_tool_use": final_step_has_tool_use,
        "protocol_final_text": protocol_final_text,
        "has_error_event": has_error_event,
    }


def extract_opencode_error(content: str | None) -> str:
    """Return the last structured OpenCode error message, if one exists."""
    error_message = ""
    for line in str(content or "").splitlines():
        try:
            data = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or data.get("type") != "error":
            continue
        error = data.get("error")
        if isinstance(error, str):
            message = error.strip()
        elif isinstance(error, dict):
            detail = error.get("data")
            message = str(detail.get("message") or "").strip() if isinstance(detail, dict) else ""
            message = message or str(error.get("message") or error.get("name") or "").strip()
        else:
            message = str(error or "").strip()
        if message:
            error_message = message
    return error_message[:2000]


def extract_opencode_final_text(content: str | None) -> str:
    """Extract the final assistant text from OpenCode or Claude-compatible JSONL."""
    last_assistant_text = ""
    final_result = ""

    for line in str(content or "").splitlines():
        try:
            data = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        event_type = data.get("type")

        if event_type == "assistant":
            message = data.get("message")
            if not isinstance(message, dict):
                continue
            content_items = message.get("content")
            if isinstance(content_items, str):
                text = content_items.strip()
            elif isinstance(content_items, list):
                text = "\n".join(
                    str(item.get("text") or "").strip()
                    for item in content_items
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and str(item.get("text") or "").strip()
                ).strip()
            else:
                text = ""
            if text:
                last_assistant_text = text
        elif event_type == "result":
            result = data.get("result")
            if isinstance(result, str) and result.strip():
                final_result = result.strip()
        elif event_type == "text":
            part = data.get("part")
            text = str(part.get("text") or "").strip() if isinstance(part, dict) else ""
            if text:
                last_assistant_text = text

    return final_result or last_assistant_text
