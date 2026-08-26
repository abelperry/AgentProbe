"""Tests for Sandbox command helpers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_probe.config import SandboxConfig
from agent_probe.core.models import ErrorCode, LastAssistant
from agent_probe.core.sandbox import ExecResult, Sandbox, SandboxSpec


class FakeCommands:
    def __init__(self) -> None:
        self.command: str | None = None
        self.opts = None

    async def run(self, command: str, *, opts=None):
        self.command = command
        self.opts = opts
        return SimpleNamespace(
            logs=SimpleNamespace(
                stdout=[SimpleNamespace(text="ok")],
                stderr=[],
            ),
            error=None,
            exit_code=0,
        )


class FakeOpenSandbox:
    def __init__(self) -> None:
        self.commands = FakeCommands()


class FakeAgent:
    async def install(self, sb: Sandbox) -> ExecResult:
        return ExecResult(stdout="", stderr="", exit_code=0)

    async def run_prompt(self, sb: Sandbox, prompt: str) -> ExecResult:
        return ExecResult(stdout="", stderr="Non-zero exit code", exit_code=1)

    async def collect_traces(self, sb: Sandbox, output_dir: Path) -> None:
        return None

    async def collect_last_assistant(
        self, sb: Sandbox, output_dir: Path
    ) -> LastAssistant:
        return LastAssistant(stop_reason="error", error_message="real agent error")


@pytest.mark.asyncio
async def test_exec_cmd_passes_timeout_to_opensandbox() -> None:
    fake = FakeOpenSandbox()
    sandbox = Sandbox(SandboxSpec(image="example:latest"))
    sandbox.os_sandbox = fake  # type: ignore[assignment]

    result = await sandbox.exec_cmd("echo ok", timeout_sec=12)

    assert result.stdout == "ok"
    assert fake.commands.command == "echo ok"
    assert fake.commands.opts.timeout == timedelta(seconds=12)


def test_sandbox_passes_api_key_to_connection_config() -> None:
    sandbox = Sandbox(
        SandboxSpec(
            image="example:latest",
            sandbox_config=SandboxConfig(api_key="sandbox-secret"),
        )
    )

    assert sandbox.connection_config.api_key == "sandbox-secret"


@pytest.mark.asyncio
async def test_nonzero_exit_error_prefers_last_assistant_message(
    tmp_path: Path,
) -> None:
    sandbox = Sandbox(
        SandboxSpec(
            image="example:latest",
            prompt="do task",
            output_dir=str(tmp_path),
        )
    )
    sandbox.os_sandbox = SimpleNamespace(id="sandbox-id")  # type: ignore[assignment]

    result = await sandbox._run(FakeAgent())  # type: ignore[arg-type]

    assert result.error is not None
    assert result.error.code == ErrorCode.AGENT_EXIT_NONZERO
    assert result.error.message == "real agent error"
