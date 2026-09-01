"""OpenCodeAgent behaviour that a benchmark depends on.

The pinned properties are the ones whose breakage is silent: a system prompt
that reaches the model through the wrong channel still runs, a second round
that quietly starts a fresh conversation still produces output, and a truncated
turn still looks like a finished one. All three would show up as a worse model,
not as a broken harness.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_probe.agents.opencode import OpenCodeAgent
from agent_probe.config import AgentConfig, ModelConfig
from agent_probe.core.sandbox import ExecResult, SandboxSpec

_DONE = "\n".join(
    [
        json.dumps({"type": "step_start"}),
        json.dumps({"type": "text", "part": {"text": "所有改动已完成"}}),
        json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
    ]
)
_TRUNCATED = "\n".join(
    [
        json.dumps({"type": "step_start"}),
        json.dumps({"type": "tool_use"}),
        json.dumps({"type": "step_finish", "part": {"reason": "length"}}),
    ]
)


class _RecordingSandbox:
    """Sandbox stand-in that records commands and the files an agent writes."""

    def __init__(self, spec: Any, stdout: str = _DONE) -> None:
        self.spec = spec
        self.session_id = "sid-1"
        self.commands: list[str] = []
        self.files: dict[str, str] = {}
        self.env_vars: dict[str, str] = {}
        self._stdout = stdout

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        self.commands.append(cmd)
        if "opencode run" in cmd:
            return ExecResult(stdout=self._stdout, stderr="", exit_code=0)
        return ExecResult(stdout="", stderr="", exit_code=0)


def _agent(**params: Any) -> OpenCodeAgent:
    return OpenCodeAgent(
        agent_config=AgentConfig(
            type="agent_probe.agents.opencode.OpenCodeAgent", params=params
        ),
        model_config=ModelConfig(
            base_url="https://example.test", api_key="k", model_name="m"
        ),
    )


@pytest.mark.asyncio
async def test_system_prompt_travels_out_of_band_not_through_the_workspace() -> None:
    """The workspace is the artifact under evaluation; instructions stay out of it."""
    spec = SandboxSpec(
        image="img", workspace="/workspace", append_system_prompt="每轮回复以喵开头"
    )
    sb = _RecordingSandbox(spec)
    agent = _agent()

    await agent.install(sb)
    await agent.run_prompt(sb, "round 0")

    config = json.loads(sb.files["/root/.config/opencode/opencode.json"])
    assert config["agent"]["agentprobe"]["prompt"] == "每轮回复以喵开头"
    run_cmd = next(cmd for cmd in sb.commands if "opencode run" in cmd)
    assert "--agent agentprobe" in run_cmd
    # Nothing may be written under the workspace -- an AGENTS.md there would
    # end up inside the snapshot the constraints are scored against.
    assert not [path for path in sb.files if path.startswith("/workspace")]


@pytest.mark.asyncio
async def test_no_benchmark_agent_when_there_is_no_system_prompt() -> None:
    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    agent = _agent()

    await agent.install(sb)
    await agent.run_prompt(sb, "go")

    config = json.loads(sb.files["/root/.config/opencode/opencode.json"])
    assert "agentprobe" not in config["agent"]
    assert "--agent" not in next(cmd for cmd in sb.commands if "opencode run" in cmd)


@pytest.mark.asyncio
async def test_later_rounds_continue_the_same_conversation() -> None:
    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace", keep_session=True))
    agent = _agent()

    await agent.run_prompt(sb, "round 0")
    await agent.run_prompt(sb, "round 1")

    runs = [cmd for cmd in sb.commands if "opencode run" in cmd]
    assert "--continue" not in runs[0]
    assert "--continue" in runs[1]


@pytest.mark.asyncio
async def test_a_truncated_turn_is_not_reported_as_complete() -> None:
    """Exit code 0 plus a step_finish is not enough: the step ended on a tool call."""
    sb = _RecordingSandbox(
        SandboxSpec(image="img", workspace="/workspace"), stdout=_TRUNCATED
    )
    agent = _agent()

    await agent.run_prompt(sb, "go")
    last = await agent.collect_last_assistant(sb, output_dir=None)  # type: ignore[arg-type]

    assert last is not None
    assert last.is_complete_response is False
    assert "complete final response" in (last.error_message or "")


@pytest.mark.asyncio
async def test_a_finished_turn_is_reported_as_complete() -> None:
    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    agent = _agent()

    await agent.run_prompt(sb, "go")
    last = await agent.collect_last_assistant(sb, output_dir=None)  # type: ignore[arg-type]

    assert last is not None
    assert last.is_complete_response is True
    assert last.content_text == "所有改动已完成"
    assert last.error_message is None


@pytest.mark.asyncio
async def test_gateway_proxy_is_off_by_default() -> None:
    """The proxy works around one gateway; it must not be the default route."""
    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    await _agent().install(sb)

    config = json.loads(sb.files["/root/.config/opencode/opencode.json"])
    assert config["provider"]["agentprobe"]["options"]["baseURL"] == "https://example.test/v1"
    assert not [cmd for cmd in sb.commands if "gateway_proxy" in cmd]
    assert sb.env_vars["OPENAI_BASE_URL"] == "https://example.test/v1"


@pytest.mark.asyncio
async def test_gateway_proxy_reroutes_when_enabled() -> None:
    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    await _agent(gateway_proxy=True).install(sb)

    config = json.loads(sb.files["/root/.config/opencode/opencode.json"])
    assert config["provider"]["agentprobe"]["options"]["baseURL"] == "http://127.0.0.1:18080/v1"
    assert any("gateway_proxy" in cmd for cmd in sb.commands)


def test_version_ignores_the_claude_code_default() -> None:
    """AgentConfig.version defaults to a Claude Code release; opencode has no such tag."""
    assert _agent()._version() == OpenCodeAgent._DEFAULT_VERSION
    assert _agent(opencode_version="1.2.3")._version() == "1.2.3"


def test_each_round_is_bounded_by_the_sandbox_lifetime() -> None:
    """A per-round cap above the sandbox cap would just get the sandbox killed."""
    agent = _agent()
    spec = SandboxSpec(image="img", timeout_sec=3600)
    assert spec.agent_timeout_sec is None
    assert 3600 - agent._POST_RUN_BUFFER_SEC == 3000
