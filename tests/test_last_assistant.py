"""Tests for OpenClawAgent.collect_last_assistant and ClaudeCodeAgent.collect_last_assistant.

Data-driven: fixtures under tests/fixtures/last_assistant/ are either distilled
from real trace files (openclaw/*.jsonl, claude_code/normal.jsonl) or crafted to
cover edge cases (edge/*.jsonl).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
import pytest

from agent_probe.agents.claude_code import ClaudeCodeAgent
from agent_probe.agents.openclaw import OpenClawAgent
from agent_probe.core.models import LastAssistant

FIXTURES = Path(__file__).parent / "fixtures" / "last_assistant"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class Case:
    fixture: str              # relative path under FIXTURES
    stop_reason: str | None
    error_message: str | None
    text_head: str | None     # first ~30 chars of content_text; None = must be empty
    text_len_min: int = 0     # lower-bound length check for non-trivial bodies


class _StubSandbox:
    """Minimal stand-in; collect_last_assistant only reads sb.session_id."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


def _make_agent(agent_cls):
    return agent_cls.__new__(agent_cls)


async def _run(agent_cls, fixture_path: Path, tmp_path: Path) -> LastAssistant | None:
    """Drop the fixture under tmp_path/traces/{sid}.jsonl and invoke the parser."""
    session_id = "sid_for_test"
    traces = tmp_path / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture_path, traces / f"{session_id}.jsonl")
    agent = _make_agent(agent_cls)
    return await agent.collect_last_assistant(_StubSandbox(session_id), tmp_path)


def _assert_matches(result: LastAssistant | None, case: Case) -> None:
    assert result is not None, f"expected LastAssistant, got None ({case.fixture})"
    assert result.stop_reason == case.stop_reason, (
        f"stop_reason mismatch: {result.stop_reason!r} vs {case.stop_reason!r}"
    )
    assert result.error_message == case.error_message, (
        f"error_message mismatch: {result.error_message!r} vs {case.error_message!r}"
    )
    if case.text_head is None:
        assert result.content_text == ""
    else:
        assert result.content_text.startswith(case.text_head), (
            f"content_text head mismatch; got {result.content_text[:60]!r}"
        )
        assert len(result.content_text) >= case.text_len_min


# ---------------------------------------------------------------------------
# OpenClaw — data-driven cases from real distilled traces
# ---------------------------------------------------------------------------

OPENCLAW_CASES: list[Case] = [
    # Normal completion: last assistant has text and stop=stop
    Case(
        fixture="openclaw/zcb_001.jsonl",
        stop_reason="stop",
        error_message=None,
        text_head="已写好，",
        text_len_min=100,
    ),
    # Network abort: last assistant has empty content + stopReason=error;
    # content_text must fall back to an earlier non-empty assistant turn.
    Case(
        fixture="openclaw/zcb_002.jsonl",
        stop_reason="error",
        error_message="The operation was aborted",
        text_head="百度搜出来的都是",
        text_len_min=10,
    ),
    # Long normal completion
    Case(
        fixture="openclaw/zcb_028.jsonl",
        stop_reason="stop",
        error_message=None,
        text_head="全部完成",
        text_len_min=200,
    ),
    # Error with non-empty text: last assistant has BOTH text and stopReason=error
    Case(
        fixture="openclaw/zcb_078.jsonl",
        stop_reason="error",
        error_message="Network connection lost.",
        text_head="The file was truncated",
        text_len_min=30,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", OPENCLAW_CASES, ids=lambda c: c.fixture)
async def test_openclaw_real(case: Case, tmp_path: Path) -> None:
    result = await _run(OpenClawAgent, FIXTURES / case.fixture, tmp_path)
    _assert_matches(result, case)


# ---------------------------------------------------------------------------
# OpenClaw — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openclaw_missing_trace(tmp_path: Path) -> None:
    """No trace file ⇒ returns None without raising."""
    agent = _make_agent(OpenClawAgent)
    result = await agent.collect_last_assistant(_StubSandbox("nonexistent"), tmp_path)
    assert result is None


@pytest.mark.asyncio
async def test_openclaw_empty_file(tmp_path: Path) -> None:
    """Empty trace file ⇒ returns None."""
    result = await _run(OpenClawAgent, FIXTURES / "edge/empty.jsonl", tmp_path)
    assert result is None


@pytest.mark.asyncio
async def test_openclaw_no_assistant_messages(tmp_path: Path) -> None:
    """Only session/non-message events ⇒ returns None."""
    result = await _run(
        OpenClawAgent, FIXTURES / "edge/openclaw_session_only.jsonl", tmp_path,
    )
    assert result is None


@pytest.mark.asyncio
async def test_openclaw_tooluse_only(tmp_path: Path) -> None:
    """Assistant messages exist but none has text ⇒ content_text is empty,
    but stop_reason/error_message still reflect the last assistant."""
    result = await _run(
        OpenClawAgent, FIXTURES / "edge/openclaw_tooluse_only.jsonl", tmp_path,
    )
    assert result is not None
    assert result.stop_reason == "toolUse"
    assert result.error_message is None
    assert result.content_text == ""


@pytest.mark.asyncio
async def test_openclaw_malformed_lines(tmp_path: Path) -> None:
    """Malformed JSON lines are skipped; valid ones still parsed."""
    result = await _run(
        OpenClawAgent, FIXTURES / "edge/openclaw_malformed.jsonl", tmp_path,
    )
    assert result is not None
    assert result.stop_reason == "stop"
    assert result.content_text == "hello"


# ---------------------------------------------------------------------------
# Claude Code — real + edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claude_code_normal(tmp_path: Path) -> None:
    """Real distilled claude-code trace: last assistant has text, no api error."""
    result = await _run(ClaudeCodeAgent, FIXTURES / "claude_code/normal.jsonl", tmp_path)
    assert result is not None
    assert result.error_message is None
    # stop_reason in this particular fixture happens to be None (pairwise_judge cut);
    # content must still carry the final text.
    assert len(result.content_text) > 100


@pytest.mark.asyncio
async def test_claude_code_api_error(tmp_path: Path) -> None:
    """isApiErrorMessage=true flips stop_reason to 'error' and promotes text
    to error_message; content_text falls back to prior non-error assistant."""
    result = await _run(
        ClaudeCodeAgent, FIXTURES / "edge/claude_code_api_error.jsonl", tmp_path,
    )
    assert result is not None
    assert result.stop_reason == "error"
    assert result.error_message == "API Error: 529 Overloaded"
    assert result.content_text == "Let me help with that."


@pytest.mark.asyncio
async def test_claude_code_missing_trace(tmp_path: Path) -> None:
    agent = _make_agent(ClaudeCodeAgent)
    result = await agent.collect_last_assistant(_StubSandbox("nope"), tmp_path)
    assert result is None


@pytest.mark.asyncio
async def test_claude_code_tooluse_only(tmp_path: Path) -> None:
    """Only tool_use blocks ⇒ content_text empty, stop_reason reflects last event."""
    result = await _run(
        ClaudeCodeAgent, FIXTURES / "edge/claude_code_tooluse_only.jsonl", tmp_path,
    )
    assert result is not None
    assert result.stop_reason == "tool_use"
    assert result.error_message is None
    assert result.content_text == ""
