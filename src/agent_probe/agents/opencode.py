"""OpenCode agent runtime implemented directly on AgentProbe sandboxes.

Two things about OpenCode shape this file rather than being chosen:

**It is a Node program.** The Claude Code CLI ships as a self-contained native
binary; ``opencode`` is npm packages that need a Node runtime in the sandbox, so
offline installs have to carry Node too -- ``scripts/init.sh --opencode``.

**It has no ``--append-system-prompt``.** That is fine here: MTAC-IFBench
defines ``repository_policy`` as the content of a repository policy file
(``AGENTS.md`` / ``CLAUDE.md``), so the benchmark writes it into the workspace
where the agent is meant to read it. Nothing needs an out-of-band channel.
"""

from __future__ import annotations

import json
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from agent_probe.agents.opencode_output import (
    extract_opencode_error,
    extract_opencode_final_text,
    inspect_opencode_jsonl,
)
from agent_probe.core.agent import BaseAgent
from agent_probe.core.models import LastAssistant
from agent_probe.core.sandbox import ExecResult

if TYPE_CHECKING:
    from agent_probe.core.sandbox import Sandbox


_INCOMPLETE_FINISH_REASONS = {
    "length",
    "max_output_tokens",
    "max_tokens",
    "pause_turn",
    "tool-calls",
    "tool_calls",
    "tool_use",
}


class OpenCodeAgent(BaseAgent):
    """Run the OpenCode CLI directly on AgentProbe sandboxes."""

    project_instruction_filenames = ("AGENTS.md", "CLAUDE.md", "CONTEXT.md")

    _DEFAULT_VERSION = "1.1.21"
    _DEFAULT_NPM_PACKAGE = "opencode-ai"
    _DEFAULT_EXTRA_PACKAGES = ("opencode-linux-x64",)
    _OFFLINE_INSTALL_PATH = "/tmp/offline_package"
    _NPM_CHECK_COMMAND = "command -v node >/dev/null 2>&1 && " "command -v npm >/dev/null 2>&1"
    _GATEWAY_NORMALIZER_PATH = "/tmp/agentprobe_opencode_gateway_proxy.mjs"
    _GATEWAY_NORMALIZER_LOG = "/tmp/agentprobe_opencode_gateway_proxy.log"
    _GATEWAY_NORMALIZER_PID = "/tmp/agentprobe_opencode_gateway_proxy.pid"
    _GATEWAY_NORMALIZER_PORT = 18080
    # Seconds left between killing opencode and the sandbox's own cap, so
    # on_complete still gets to export the workspace.
    _POST_RUN_BUFFER_SEC = 600

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active_sessions: set[str] = set()
        self._trace_lines: dict[str, list[str]] = {}
        self._last_assistants: dict[str, LastAssistant] = {}

    def _version(self) -> str:
        configured = str(self.agent_config.params.get("opencode_version") or "").strip()
        if configured:
            return configured
        version = str(self.agent_config.version or "").strip()
        # AgentConfig's default is a Claude Code version. Do not accidentally
        # use that unrelated value when an OpenCode config omits its version.
        if version in {"", "2.1.14", "2.1.199"}:
            return self._DEFAULT_VERSION
        return version

    def _connection(self) -> tuple[str, str]:
        return self.model_config.base_url, self.model_config.api_key

    @staticmethod
    def _api_base_url(base_url: str) -> str:
        normalized = str(base_url or "").rstrip("/")
        return normalized if normalized.endswith("/v1") else f"{normalized}/v1"

    def _provider_config(self, base_url: str, api_key: str) -> dict[str, Any]:
        model_name = self.model_config.model_name
        output_limit = int(self.model_config.max_tokens)
        context_limit = int(self.agent_config.params.get("context_window", 200_000))
        model_options: dict[str, Any] = {
            "name": model_name,
            "limit": {"context": context_limit, "output": output_limit},
        }
        if (
            self.model_config.format == "anthropic"
            and self.model_config.thinking
            and self.model_config.thinking != "off"
        ):
            model_options["options"] = {
                "thinking": {
                    "type": "enabled",
                    "budgetTokens": min(
                        max(1, output_limit - 2),
                        int(self.model_config.max_thinking_tokens),
                    ),
                }
            }

        provider_npm = (
            "@ai-sdk/anthropic"
            if self.model_config.format == "anthropic"
            else "@ai-sdk/openai-compatible"
        )
        provider_name = (
            "AgentProbe Anthropic Provider"
            if self.model_config.format == "anthropic"
            else "AgentProbe OpenAI-compatible Provider"
        )
        return {
            "npm": provider_npm,
            "name": provider_name,
            "options": {
                "baseURL": self._api_base_url(base_url),
                "apiKey": api_key,
            },
            "models": {model_name: model_options},
        }

    async def install(self, sb: Sandbox) -> ExecResult:
        base_url, api_key = self._connection()
        sb.env_vars.update(
            {
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_CONFIG": "/root/.config/opencode/opencode.json",
                "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(self.model_config.max_tokens),
            }
        )
        result = (
            await self._install_offline(sb)
            if self.agent_config.offline
            else await self._install_online(sb)
        )
        if result.exit_code != 0:
            return result

        # On by default: OpenCode's providers ask for SSE, and a gateway that
        # answers a streaming request with one JSON body leaves the SDK waiting.
        # Set params.gateway_proxy: false to talk to an endpoint that streams
        # properly -- the shim converts the call to non-streaming, which is a
        # cost worth avoiding when it buys nothing.
        local_base_url = base_url
        if self.agent_config.params.get("gateway_proxy", True):
            normalizer = await self._start_gateway_normalizer(sb, base_url)
            if normalizer.exit_code != 0:
                return normalizer
            local_base_url = f"http://127.0.0.1:{self._GATEWAY_NORMALIZER_PORT}"
        api_base_url = self._api_base_url(local_base_url)
        if self.model_config.format == "anthropic":
            sb.env_vars.update(
                {
                    "ANTHROPIC_API_KEY": api_key,
                    "ANTHROPIC_BASE_URL": api_base_url,
                }
            )
        else:
            sb.env_vars.update(
                {
                    "OPENAI_API_KEY": api_key,
                    "OPENAI_BASE_URL": api_base_url,
                }
            )

        model_name = self.model_config.model_name
        opencode_config = {
            "model": f"agentprobe/{model_name}",
            "small_model": f"agentprobe/{model_name}",
            "agent": {"title": {"disable": True}},
            "provider": {
                "agentprobe": self._provider_config(local_base_url, api_key),
            },
            "permission": {
                "*": "allow",
                "external_directory": "allow",
                "bash": "allow",
                "edit": "allow",
                "read": "allow",
                "write": "allow",
                "question": "deny",
            },
        }
        await sb.exec_cmd("mkdir -p /root/.config/opencode", timeout_sec=30)
        await sb.write_file(
            "/root/.config/opencode/opencode.json",
            json.dumps(opencode_config, ensure_ascii=False, indent=2),
        )
        verify = await sb.exec_cmd("opencode --version", timeout_sec=30)
        if verify.exit_code != 0:
            return verify
        if self.agent_config.mcp_host_path:
            logger.warning("OpenCode MCP injection is not supported; ignoring mcp_host_path")
        logger.debug(
            "OpenCode configured: version={}, model={}, format={}",
            verify.stdout.strip() or self._version(),
            model_name,
            self.model_config.format,
        )
        return verify

    async def _start_gateway_normalizer(self, sb: Sandbox, upstream_base_url: str) -> ExecResult:
        proxy_source = (
            Path(__file__).with_name("opencode_gateway_proxy.mjs").read_text(encoding="utf-8")
        )
        await sb.write_file(self._GATEWAY_NORMALIZER_PATH, proxy_source)
        start_command = " ".join(
            (
                f"AGENTPROBE_UPSTREAM_BASE_URL={shlex.quote(self._api_base_url(upstream_base_url))}",
                f"AGENTPROBE_PROXY_PORT={self._GATEWAY_NORMALIZER_PORT}",
                "nohup node",
                shlex.quote(self._GATEWAY_NORMALIZER_PATH),
                f">{shlex.quote(self._GATEWAY_NORMALIZER_LOG)} 2>&1 &",
                f"echo $! >{shlex.quote(self._GATEWAY_NORMALIZER_PID)}",
            )
        )
        started = await sb.exec_cmd(start_command, timeout_sec=30)
        if started.exit_code != 0:
            return started

        health_url = f"http://127.0.0.1:{self._GATEWAY_NORMALIZER_PORT}/healthz"
        health_script = (
            f"fetch({json.dumps(health_url)})"
            ".then(response => process.exit(response.ok ? 0 : 1))"
            ".catch(() => process.exit(1))"
        )
        health_command = (
            "for attempt in $(seq 1 50); do "
            f"node -e {shlex.quote(health_script)} && exit 0; "
            "sleep 0.1; "
            "done; "
            f"tail -50 {shlex.quote(self._GATEWAY_NORMALIZER_LOG)} >&2; exit 1"
        )
        return await sb.exec_cmd(health_command, timeout_sec=30)

    async def _install_online(self, sb: Sandbox) -> ExecResult:
        npm_result = await self._ensure_npm(sb)
        if npm_result.exit_code != 0:
            return npm_result
        package_name = str(self.agent_config.params.get("npm_package") or self._DEFAULT_NPM_PACKAGE)
        return await sb.exec_cmd(
            "npm --fetch-retries=3 --fetch-retry-mintimeout=2000 "
            "--fetch-retry-maxtimeout=10000 i -g "
            f"{shlex.quote(package_name)}@{shlex.quote(self._version())}"
        )

    async def _ensure_npm(self, sb: Sandbox) -> ExecResult:
        check = await sb.exec_cmd(self._NPM_CHECK_COMMAND)
        if check.exit_code == 0:
            return check
        result = check
        for command in (
            "apt-get update -o Acquire::Retries=3",
            (
                "DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "--no-install-recommends curl ca-certificates gnupg"
            ),
            (
                "curl --retry 3 --retry-delay 2 --retry-all-errors -fsSL "
                "https://deb.nodesource.com/setup_22.x | bash -"
            ),
            ("DEBIAN_FRONTEND=noninteractive apt-get install -y " "--no-install-recommends nodejs"),
        ):
            result = await sb.exec_cmd(command)
            if result.exit_code != 0:
                return result
        verification = await sb.exec_cmd(self._NPM_CHECK_COMMAND)
        if verification.exit_code == 0:
            return verification
        fallback = await sb.exec_cmd("DEBIAN_FRONTEND=noninteractive apt-get install -y npm")
        if fallback.exit_code != 0:
            return fallback
        return await sb.exec_cmd(self._NPM_CHECK_COMMAND)

    async def _install_offline(self, sb: Sandbox) -> ExecResult:
        version = self._version()
        node_version = self.agent_config.offline_node_version
        node_archive_name = f"node-v{node_version}-linux-x64"
        package_name = str(self.agent_config.params.get("npm_package") or self._DEFAULT_NPM_PACKAGE)
        extra_packages = self.agent_config.params.get(
            "offline_extra_packages", list(self._DEFAULT_EXTRA_PACKAGES)
        )
        if not isinstance(extra_packages, list):
            raise ValueError("offline_extra_packages must be a list")
        mount_path = self.agent_config.offline_mount_path.rstrip("/")
        install_path = self._OFFLINE_INSTALL_PATH
        node_bin = f"{install_path}/{node_archive_name}/bin"
        archives = [*(str(item) for item in extra_packages), package_name]

        commands = [
            "set -e",
            f"mkdir -p {shlex.quote(install_path)}",
            (
                f"test -f {shlex.quote(f'{mount_path}/{node_archive_name}.tar.gz')} "
                "|| { echo 'missing offline Node.js archive' >&2; exit 1; }"
            ),
            (
                f"cp {shlex.quote(f'{mount_path}/{node_archive_name}.tar.gz')} "
                f"{shlex.quote(install_path)}/"
            ),
            (
                f"tar -xzf {shlex.quote(f'{install_path}/{node_archive_name}.tar.gz')} "
                f"-C {shlex.quote(install_path)}"
            ),
            (f"chmod +x {shlex.quote(f'{node_bin}/node')} " f"{shlex.quote(f'{node_bin}/npm')}"),
            f"export PATH={shlex.quote(node_bin)}:$PATH",
        ]
        for package in archives:
            archive_name = f"{package.replace('@', '').replace('/', '-')}-{version}.tgz"
            source = f"{mount_path}/{archive_name}"
            destination = f"{install_path}/{archive_name}"
            commands.extend(
                [
                    (
                        f"test -f {shlex.quote(source)} || "
                        f"{{ echo 'missing offline package: {source}' >&2; exit 1; }}"
                    ),
                    f"cp {shlex.quote(source)} {shlex.quote(destination)}",
                    f"{shlex.quote(f'{node_bin}/npm')} install -g {shlex.quote(destination)}",
                ]
            )
        commands.append("command -v opencode")
        result = await sb.exec_cmd(f"bash -lc {shlex.quote('; '.join(commands))}")
        if result.exit_code == 0:
            current_path = await sb.exec_cmd("printf '%s' \"$PATH\"")
            inherited_path = current_path.stdout.strip() if current_path.exit_code == 0 else ""
            sb.env_vars["PATH"] = ":".join(
                part for part in (node_bin, f"{install_path}/bin", inherited_path) if part
            )
        return result

    async def run_prompt(self, sb: Sandbox, prompt: str) -> ExecResult:
        session_id = sb.session_id
        continue_session = session_id in self._active_sessions
        # Bound each round by what is left of the sandbox, not by the model
        # timeout: running past the sandbox cap just gets the container killed
        # before on_complete can export the workspace. Mirrors ClaudeCodeAgent.
        turn_timeout = sb.spec.agent_timeout_sec or max(
            60, sb.spec.timeout_sec - self._POST_RUN_BUFFER_SEC
        )
        trace_lines = [
            json.dumps(
                {"type": "user", "message": {"role": "user", "content": prompt}},
                ensure_ascii=False,
            )
        ]
        result = await self._run_once(
            sb,
            prompt,
            continue_session=continue_session,
            timeout_sec=turn_timeout,
        )
        self._active_sessions.add(session_id)
        stdout = result.stdout
        trace_lines.extend(stdout.splitlines())
        inspection = inspect_opencode_jsonl(stdout)
        finish_reason = str(inspection["finish_reason"] or "").strip().lower()
        incomplete_response = bool(
            result.exit_code == 0
            and not inspection["has_error_event"]
            and inspection["saw_jsonl"]
            and (
                inspection["final_step_has_tool_use"]
                or finish_reason in _INCOMPLETE_FINISH_REASONS
                or not inspection["has_assistant_text"]
                or not inspection["has_terminal_event"]
            )
        )
        incomplete_detail = ""
        if incomplete_response:
            incomplete_detail = (
                "OpenCode exited without a complete final assistant response "
                f"(finish_reason={inspection['finish_reason']!r}, "
                f"terminal_tool_use={inspection['final_step_has_tool_use']})"
            )

        self._trace_lines.setdefault(session_id, []).extend(trace_lines)
        inspection = inspect_opencode_jsonl(stdout)
        protocol_error = extract_opencode_error(stdout)
        process_error = ""
        if not protocol_error and result.exit_code != 0:
            process_error = result.stderr.strip() or f"OpenCode exited with code {result.exit_code}"
        error_message = protocol_error or process_error or incomplete_detail
        text = extract_opencode_final_text(stdout)
        if not inspection["saw_jsonl"] and stdout.strip():
            text = stdout.strip()
        finish_reason = str(inspection["finish_reason"] or "").strip().lower()
        has_assistant_text = bool(
            inspection["has_assistant_text"] if inspection["saw_jsonl"] else text
        )
        has_terminal_event = bool(
            inspection["has_terminal_event"] if inspection["saw_jsonl"] else text
        )
        complete = bool(
            result.exit_code == 0
            and text
            and not error_message
            and not inspection["has_error_event"]
            and has_assistant_text
            and not inspection["final_step_has_tool_use"]
            and finish_reason not in _INCOMPLETE_FINISH_REASONS
            and has_terminal_event
        )
        self._last_assistants[session_id] = LastAssistant(
            stop_reason="error" if error_message else inspection["finish_reason"],
            error_message=error_message or None,
            content_text=text,
            is_complete_response=complete,
        )
        return ExecResult(
            stdout=stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    async def _run_once(
        self,
        sb: Sandbox,
        prompt: str,
        *,
        continue_session: bool,
        timeout_sec: int,
    ) -> ExecResult:
        prompt_path = f"/tmp/prompt_{uuid.uuid4().hex[:8]}.txt"
        await sb.write_file(prompt_path, prompt)
        workspace = sb.spec.workspace or "/root"
        command_parts = [
            f"cd {shlex.quote(workspace)} &&",
            f"cat {shlex.quote(prompt_path)} |",
            "opencode run",
            f"--model {shlex.quote(f'agentprobe/{self.model_config.model_name}')}",
        ]
        if str(self.agent_config.params.get("output_format", "json")) == "json":
            command_parts.extend(("--format", "json"))
        agent_name = str(self.agent_config.params.get("agent_name") or "").strip()
        if agent_name:
            command_parts.extend(("--agent", shlex.quote(agent_name)))
        extra_flags = self.agent_config.params.get("extra_flags") or []
        if isinstance(extra_flags, str):
            extra_flags = shlex.split(extra_flags)
        if not isinstance(extra_flags, list):
            raise ValueError("OpenCode extra_flags must be a string or list")
        command_parts.extend(shlex.quote(str(flag)) for flag in extra_flags)
        if continue_session:
            command_parts.append("--continue")
        return await sb.exec_cmd(
            " ".join(command_parts),
            timeout_sec=timeout_sec,
        )

    async def collect_traces(self, sb: Sandbox, output_dir: Path) -> None:
        lines = self._trace_lines.get(sb.session_id, [])
        if not lines:
            logger.warning("[{}] OpenCode trace is empty", sb.session_id)
            return
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{sb.session_id}.jsonl").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    async def collect_last_assistant(
        self,
        sb: Sandbox,
        output_dir: Path,
    ) -> LastAssistant | None:
        del output_dir
        return self._last_assistants.get(sb.session_id)
