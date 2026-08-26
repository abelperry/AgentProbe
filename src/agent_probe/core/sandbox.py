"""Sandbox engine — wraps the OpenSandbox Python SDK."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional
import aiofiles

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from opensandbox import Sandbox as OpenSandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.sandboxes import Host, Volume

from agent_probe.config import AgentConfig, ModelConfig, SandboxConfig
from agent_probe.core.models import Error, ErrorCode, LastAssistant
from agent_probe.utils.imports import import_class



if TYPE_CHECKING:
    from agent_probe.core.agent import BaseAgent

# ---------------------------------------------------------------------------
# Hook type aliases
# ---------------------------------------------------------------------------
OnSetupHook = Callable[["Sandbox"], Awaitable[None]]
OnCompleteHook = Callable[["Sandbox", "SandboxResult"], Awaitable[None]]
OnNextRoundHook = Callable[["Sandbox", "SandboxResult"], Awaitable[Optional[str]]]

# Agent factory: (agent_config, model_config) -> BaseAgent
AgentFactory = Callable[["AgentConfig", "ModelConfig"], "BaseAgent"]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class ResourceSpec(BaseModel):
    cpus: int = 4
    memory_mb: int = 4096
    storage_mb: int = 10240
    gpus: int = 0


class SandboxSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    image: str
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    prompt: str = ""
    env_vars: dict[str, str] = Field(default_factory=dict)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    timeout_sec: int = 10800
    workspace: Optional[str] = None
    volumes: list[Volume] = Field(default_factory=list)

    entrypoint: Optional[list[str]] = None  # None = SDK default (tail -f /dev/null)

    # Keep one agent conversation across all rounds instead of starting a fresh
    # one each time. Multi-turn benchmarks whose constraints span rounds (e.g.
    # "reuse last round's naming") need this; single-turn ones must not set it.
    keep_session: bool = False
    # Extra system-prompt text handed to the agent CLI (claude
    # --append-system-prompt). Used for per-question system constraints and for
    # judge anti-injection instructions.
    append_system_prompt: str = ""

    agent_config: Optional[AgentConfig] = None
    model_cfg: Optional[ModelConfig] = None
    output_dir: Optional[str] = None

    on_setup: Optional[OnSetupHook] = None
    on_complete: Optional[OnCompleteHook] = None
    on_nextround: Optional[OnNextRoundHook] = None

    def model_post_init(self, __context: Any) -> None:
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class SandboxResult:
    """Return type of Sandbox.run() — collects all round results."""

    rounds: list[ExecResult] = field(default_factory=list)
    error: Error | None = None
    last_assistant: LastAssistant | None = None

    @property
    def last(self) -> ExecResult | None:
        return self.rounds[-1] if self.rounds else None


# ---------------------------------------------------------------------------
# Sandbox — stateless wrapper around the OpenSandbox SDK
# ---------------------------------------------------------------------------
class Sandbox:
    """Stateless client that wraps the *opensandbox* Python SDK.

    It does NOT store any ``sandbox_id``; every helper method receives the id
    as a parameter so that multiple sandboxes can be managed concurrently
    through a single ``Sandbox`` instance.
    """

    def __init__(self, spec: SandboxSpec) -> None:
        self.spec = spec
        self.connection_config = ConnectionConfig(
            domain=spec.sandbox_config.host,
            api_key=spec.sandbox_config.api_key or None,
            request_timeout=timedelta(seconds=spec.sandbox_config.request_timeout),
        )
        self.env_vars = spec.env_vars
        self.session_id: str = str(uuid.uuid4())
        self.os_sandbox: OpenSandbox | None = None


    def _build_agent(self, agent_config: AgentConfig, model_config: ModelConfig) -> BaseAgent:
        agent_cls = import_class(agent_config.type)
        return agent_cls(agent_config=agent_config, model_config=model_config)

    def _build_volumes(self) -> Optional[list[Volume]]:
        volumes = list(self.spec.volumes)
        if self.spec.agent_config and self.spec.agent_config.offline:
            volumes.append(
                Volume(
                    name="agent-offline-packages",
                    host=Host(path=self.spec.agent_config.offline_package_dir),
                    mount_path=self.spec.agent_config.offline_mount_path,
                    read_only=True,
                )
            )
        return volumes or None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> SandboxResult:
        """Orchestrate with client-side timeout guard.

        Uses ``asyncio.wait_for`` (timeout_sec - 30s) so we get a clean
        timeout error *before* the SDK server kills the container.
        """
        # 1. Create sandbox (outside wait_for — creation itself shouldn't count)
        agent: BaseAgent | None = None
        if self.spec.agent_config and self.spec.model_cfg:
            agent = self._build_agent(self.spec.agent_config, self.spec.model_cfg)

        logger.debug("Creating sandbox: image={}", self.spec.image)
        self.os_sandbox = await OpenSandbox.create(
            self.spec.image,
            connection_config=self.connection_config,
            timeout=timedelta(seconds=self.spec.timeout_sec),
            ready_timeout=timedelta(seconds=120),
            env=self.env_vars,
            resource={
                "cpu": str(self.spec.resources.cpus),
                "memory": f"{self.spec.resources.memory_mb}Mi",
            },
            entrypoint=self.spec.entrypoint,
            volumes=self._build_volumes(),
        )
        logger.debug("Sandbox created: id={}", self.os_sandbox.id)

        try:
            return await asyncio.wait_for(
                self._run(agent), timeout=self.spec.timeout_sec - 30,
            )
        except asyncio.TimeoutError:
            logger.warning("Sandbox {} timed out after {}s", self.os_sandbox.id, self.spec.timeout_sec - 30)
            return SandboxResult(error=Error(code=ErrorCode.SANDBOX_TIMEOUT, message=f"Sandbox timeout after {self.spec.timeout_sec - 30}s"))
        except Exception as e:
            logger.error("Sandbox {} unexpected error: {}", self.os_sandbox.id, e)
            return SandboxResult(error=Error(code=ErrorCode.SANDBOX_UNKNOWN, message=str(e)))
        finally:
            await self.os_sandbox.kill()
            await self.os_sandbox.close()

    async def _run(self, agent: BaseAgent | None) -> SandboxResult:
        """Core orchestration logic: install → setup → execute → collect."""
        result = SandboxResult()

        # 2. Setup
        if self.spec.on_setup:
            await self.spec.on_setup(self)

        # 3. Install agent (after setup so gateway inherits env vars)
        if agent:
            logger.debug("Installing agent in sandbox {}", self.os_sandbox.id)
            install_result = await agent.install(self)
            if install_result.exit_code != 0:
                logger.error("Agent install failed in sandbox {}: {}", self.os_sandbox.id, install_result.stderr)
                result.rounds.append(install_result)
                result.error = Error(
                    code=ErrorCode.AGENT_INSTALL,
                    message=install_result.stderr or "Agent install failed",
                )
                return result

        # 4. Multi-round execution — trace collection happens per round so
        # on_nextround can observe the freshest last_assistant.
        if agent:
            prompt = self.spec.prompt
            output_dir = Path(self.spec.output_dir) if self.spec.output_dir else None
            while prompt:
                exec_result = await agent.run_prompt(self, prompt)
                result.rounds.append(exec_result)

                if output_dir:
                    try:
                        await agent.collect_traces(self, output_dir)
                        result.last_assistant = await agent.collect_last_assistant(
                            self, output_dir,
                        )
                    except Exception as e:
                        logger.warning("per-round trace collection failed: {}", e)

                # Check for next round
                if self.spec.on_nextround:
                    next_prompt = await self.spec.on_nextround(self, result)
                    prompt = next_prompt or ""
                    if prompt and not self.spec.keep_session:
                        # Fresh session per round: agents that reject session_id
                        # reuse (e.g. claude-code CLI) get a clean slate.
                        self.session_id = str(uuid.uuid4())
                else:
                    break

        # 5. Parse error — exec exit_code first, then trace stop_reason
        if result.last and result.last.exit_code != 0:
            last_assistant_error = (
                result.last_assistant.error_message if result.last_assistant else None
            )
            result.error = Error(
                code=ErrorCode.AGENT_EXIT_NONZERO,
                message=last_assistant_error
                or (result.last.stderr or "")[-500:]
                or "Non-zero exit code",
            )
        elif result.last_assistant and result.last_assistant.stop_reason == "error":
            result.error = Error(
                code=ErrorCode.AGENT_STOP_ERROR,
                message=result.last_assistant.error_message or "agent stopped with error",
            )

        # 6. Complete (side-effect only, e.g. cleanup)
        if self.spec.on_complete:
            await self.spec.on_complete(self, result)

        return result

    # ------------------------------------------------------------------
    # Atomic operations — used by Hooks and Agents
    # ------------------------------------------------------------------

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        """Execute a shell command inside the sandbox."""
        envs = list(self.env_vars.items())
        dump_env = "".join([f'export {k}="{v}" && ' for k, v in envs])
        cmd = f"{dump_env}{cmd}"

        opts = None
        if timeout_sec is not None:
            opts = RunCommandOpts(timeout=timedelta(seconds=timeout_sec))
        execution = await self.os_sandbox.commands.run(cmd, opts=opts)
        stdout = "\n".join(msg.text for msg in execution.logs.stdout)
        stderr = "\n".join(msg.text for msg in execution.logs.stderr)
        exit_code = execution.exit_code
        if exit_code is None:
            exit_code = 0 if execution.error is None else 1
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        return await self.os_sandbox.files.read_file(path)


    async def write_file(self, path: str, content: str) -> None:
        """Write a file into the sandbox."""
        await self.os_sandbox.files.write_file(path, content)

    async def write_files(self, entries: list[tuple[str, str | bytes]]) -> None:
        """Batch write files. entries: [(remote_path, content), ...]."""
        from opensandbox.models.filesystem import WriteEntry

        write_entries = []
        for remote_path, content in entries:
            write_entries.append(WriteEntry(path=remote_path, data=content))
        if write_entries:
            await self.os_sandbox.files.write_files(write_entries)

    async def create_directories(self, paths: list[str]) -> None:
        """Batch create directories."""
        from opensandbox.models.filesystem import WriteEntry

        if paths:
            entries = [WriteEntry(path=p, data=None) for p in paths]
            await self.os_sandbox.files.create_directories(entries)

    async def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        """Recursively upload a local directory to the sandbox."""
        dirs_to_create: list[str] = []
        files_to_write: list[tuple[str, str | bytes]] = []

        for local_path in sorted(local_dir.rglob("*")):
            rel = local_path.relative_to(local_dir)
            remote_path = f"{remote_dir}/{rel}"
            if local_path.is_dir():
                dirs_to_create.append(remote_path)
            elif local_path.is_file():
                files_to_write.append((remote_path, local_path.read_bytes()))

        await self.create_directories([remote_dir] + dirs_to_create)

        # Upload in batches of 50 files
        batch_size = 50
        for i in range(0, len(files_to_write), batch_size):
            batch = files_to_write[i : i + batch_size]
            await self.write_files(batch)

    async def download_directory(self, remote_dir: str, local_tar_path: Path) -> None:
        """Stream-download a sandbox directory as tar.gz to a local file."""
        check = await self.exec_cmd(f"test -d {remote_dir}")
        if check.exit_code != 0:
            raise FileNotFoundError(f"Remote directory not found: {remote_dir}")

        local_tar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_tar = "/tmp/_export.tar.gz"
        tar_result = await self.exec_cmd(f"tar czf {tmp_tar} -C {remote_dir} .")
        if tar_result.exit_code != 0:
            raise RuntimeError(
                f"tar failed for {remote_dir}: {tar_result.stderr.strip()[-500:]}"
            )

        aiter = await self.os_sandbox.files.read_bytes_stream(tmp_tar)
        async with aiofiles.open(local_tar_path, "wb") as f:
            async for chunk in aiter:
                await f.write(chunk)

        await self.exec_cmd(f"rm -f {tmp_tar}")

    async def search_files(self, path: str, pattern: str) -> list[str]:
        """Search sandbox for files matching a glob pattern."""
        from opensandbox.models.filesystem import SearchEntry

        results = await self.os_sandbox.files.search(
            SearchEntry(path=path, pattern=pattern)
        )
        return [entry.path for entry in results]
