"""Tests for ClaudeCodeAgent installation bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_probe.agents.claude_code import ClaudeCodeAgent
from agent_probe.config import AgentConfig, ModelConfig
from agent_probe.core.sandbox import ExecResult, Sandbox, SandboxSpec


class FakeSandbox:
    def __init__(self, results: list[ExecResult]) -> None:
        self.results = results
        self.commands: list[str] = []
        self.env_vars: dict[str, str] = {}
        self.writes: dict[str, str] = {}

    async def exec_cmd(self, cmd: str) -> ExecResult:
        self.commands.append(cmd)
        if not self.results:
            raise AssertionError(f"unexpected command: {cmd}")
        return self.results.pop(0)

    async def write_file(self, path: str, content: str) -> None:
        self.writes[path] = content


class FakeTraceSandbox:
    def __init__(self, home: str, trace_path: str, trace_content: str) -> None:
        self.session_id = "sid-for-test"
        self.home = home
        self.trace_path = trace_path
        self.trace_content = trace_content
        self.commands: list[str] = []
        self.read_paths: list[str] = []

    async def exec_cmd(self, cmd: str) -> ExecResult:
        self.commands.append(cmd)
        if cmd == "printf '%s' \"$HOME\"":
            return ExecResult(stdout=self.home, stderr="", exit_code=0)
        return ExecResult(stdout=f"{self.trace_path}\n", stderr="", exit_code=0)

    async def read_file(self, path: str) -> str:
        self.read_paths.append(path)
        return self.trace_content


def _agent(
    version: str = "2.1.199",
    *,
    offline: bool = False,
    offline_package_dir: str = "",
    model_name: str = "claude-test",
) -> ClaudeCodeAgent:
    return ClaudeCodeAgent(
        agent_config=AgentConfig(
            type="agent_probe.agents.claude_code.ClaudeCodeAgent",
            version=version,
            offline=offline,
            offline_package_dir=offline_package_dir,
        ),
        model_config=ModelConfig(
            base_url="https://example.test",
            api_key="key",
            model_name=model_name,
        ),
    )


def test_get_env_uses_current_model_config_for_claude_code_models() -> None:
    agent = _agent(
        model_name="glm-5.1",
    )

    env = agent._get_env()

    assert env["ANTHROPIC_MODEL"] == "glm-5.1"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-5.1"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.1"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5.1"


@pytest.mark.asyncio
async def test_install_uses_existing_npm() -> None:
    sandbox = FakeSandbox(
        [
            ExecResult(stdout="/usr/bin/npm", stderr="", exit_code=0),
            ExecResult(stdout="installed", stderr="", exit_code=0),
            ExecResult(stdout="ExitPlanMode patch applied: 1", stderr="", exit_code=0),
        ]
    )

    result = await _agent("1.2.3").install(sandbox)  # type: ignore[arg-type]

    assert result.exit_code == 0
    assert sandbox.commands[:2] == [
        "command -v npm",
        "npm i -g @anthropic-ai/claude-code@1.2.3",
    ]
    assert "ExitPlanMode patch" in sandbox.commands[2]
    assert 'behavior:"ask",message:"Exit plan mode?"' in sandbox.commands[2]
    assert 'behavior:"allow",message:""' in sandbox.commands[2]
    assert sandbox.env_vars["ANTHROPIC_API_KEY"] == "key"


@pytest.mark.asyncio
async def test_install_bootstraps_node_when_npm_missing() -> None:
    sandbox = FakeSandbox(
        [
            ExecResult(stdout="", stderr="", exit_code=1),
            ExecResult(stdout="updated", stderr="", exit_code=0),
            ExecResult(stdout="deps installed", stderr="", exit_code=0),
            ExecResult(stdout="nodesource configured", stderr="", exit_code=0),
            ExecResult(stdout="node installed", stderr="", exit_code=0),
            ExecResult(stdout="claude installed", stderr="", exit_code=0),
            ExecResult(stdout="ExitPlanMode patch applied: 1", stderr="", exit_code=0),
        ]
    )

    result = await _agent().install(sandbox)  # type: ignore[arg-type]

    assert result.exit_code == 0
    assert sandbox.commands[:6] == [
        "command -v npm",
        "apt-get update",
        "apt-get install -y curl gnupg sudo",
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm i -g @anthropic-ai/claude-code@2.1.199",
    ]
    assert "ExitPlanMode patch" in sandbox.commands[6]


@pytest.mark.asyncio
async def test_install_offline_uses_default_versions(tmp_path: Path) -> None:
    sandbox = FakeSandbox(
        [
            ExecResult(stdout="/tmp/offline_package/node/bin/claude", stderr="", exit_code=0),
            ExecResult(stdout="ExitPlanMode patch applied: 1", stderr="", exit_code=0),
        ]
    )

    result = await _agent(
        offline=True,
        offline_package_dir=str(tmp_path),
    ).install(sandbox)  # type: ignore[arg-type]

    assert result.exit_code == 0
    assert len(sandbox.commands) == 2
    cmd = sandbox.commands[0]
    assert "anthropic-ai-claude-code-linux-" in cmd
    assert '"${platform}"-2.1.199.tgz' in cmd
    assert "anthropic-ai-claude-code-linux-x64-musl-2.1.199.tgz" not in cmd
    assert "/mnt/offline_package" in cmd
    assert "npm" not in cmd
    assert "tar -xOzf" in cmd
    assert "package/claude" in cmd
    assert sandbox.env_vars["PATH"] == "/tmp/offline_package/bin:$PATH"
    assert "ExitPlanMode patch" in sandbox.commands[1]


@pytest.mark.asyncio
async def test_install_offline_uses_configured_version(tmp_path: Path) -> None:
    sandbox = FakeSandbox(
        [
            ExecResult(stdout="/tmp/offline_package/node/bin/claude", stderr="", exit_code=0),
            ExecResult(stdout="ExitPlanMode patch applied: 1", stderr="", exit_code=0),
        ]
    )

    result = await _agent(
        "1.2.3",
        offline=True,
        offline_package_dir=str(tmp_path),
    ).install(sandbox)  # type: ignore[arg-type]

    assert result.exit_code == 0
    cmd = sandbox.commands[0]
    assert '"${platform}"-1.2.3.tgz' in cmd


@pytest.mark.asyncio
async def test_install_returns_bootstrap_failure() -> None:
    sandbox = FakeSandbox(
        [
            ExecResult(stdout="", stderr="", exit_code=1),
            ExecResult(stdout="", stderr="apt failed", exit_code=1),
        ]
    )

    result = await _agent().install(sandbox)  # type: ignore[arg-type]

    assert result.exit_code == 1
    assert result.stderr == "apt failed"
    assert len(sandbox.commands) == 2


def test_offline_agent_adds_host_volume(tmp_path: Path) -> None:
    agent_config = AgentConfig(
        type="agent_probe.agents.claude_code.ClaudeCodeAgent",
        offline=True,
        offline_package_dir=str(tmp_path),
    )
    spec = SandboxSpec(image="example:latest", agent_config=agent_config)

    volumes = Sandbox(spec)._build_volumes()

    assert volumes is not None
    assert len(volumes) == 1
    volume = volumes[0]
    assert volume.host is not None
    assert volume.host.path == str(tmp_path)
    assert volume.mount_path == "/mnt/offline_package"
    assert volume.read_only is True


@pytest.mark.asyncio
async def test_collect_traces_finds_non_root_home(tmp_path: Path) -> None:
    trace_path = "/home/devuser/.claude/projects/-workspace/sid-for-test.jsonl"
    sandbox = FakeTraceSandbox("/home/devuser", trace_path, '{"type":"assistant"}\n')

    await _agent().collect_traces(sandbox, tmp_path)  # type: ignore[arg-type]

    trace_file = tmp_path / "traces" / "sid-for-test.jsonl"
    assert trace_file.read_text(encoding="utf-8") == '{"type":"assistant"}\n'
    assert sandbox.read_paths == [trace_path]
    assert sandbox.commands[0] == "printf '%s' \"$HOME\""
    assert "/home/devuser/.claude/projects" in sandbox.commands[1]
    assert "dir=/root/.claude/projects" in sandbox.commands[1]


def test_env_passes_output_and_thinking_budgets() -> None:
    """max_tokens must reach the CLI; the CLI default applies otherwise."""
    agent = ClaudeCodeAgent(
        agent_config=AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent"),
        model_config=ModelConfig(
            base_url="https://example.test",
            api_key="key",
            model_name="glm-5.2",
            max_tokens=128000,
            max_thinking_tokens=96000,
        ),
    )
    env = agent._get_env()
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "128000"
    assert env["MAX_THINKING_TOKENS"] == "96000"


def test_thinking_budget_is_clamped_below_the_output_cap() -> None:
    agent = ClaudeCodeAgent(
        agent_config=AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent"),
        model_config=ModelConfig(
            base_url="https://example.test",
            api_key="key",
            max_tokens=8000,
            max_thinking_tokens=64000,
        ),
    )
    assert agent._get_env()["MAX_THINKING_TOKENS"] == "7998"
