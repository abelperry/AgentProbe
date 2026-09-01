"""OpenCode agent — the second CLI AgentProbe can drive.

Two things differ from the Claude Code agent, and both are forced by OpenCode
rather than chosen:

**It is a Node program.** The Claude Code CLI ships as a self-contained native
binary; ``opencode`` is npm packages (``opencode-ai`` plus a platform build)
that need a Node runtime in the sandbox. Offline installs therefore have to
carry Node itself — see ``scripts/init.sh --opencode``.

**It has no ``--append-system-prompt``.** The equivalent is a named agent
defined in ``opencode.json`` with a ``prompt`` field, selected at run time with
``--agent``. That matters: it keeps a benchmark's standing instructions *out of
the workspace*, exactly like Claude Code's flag, so the workspace stays a clean
record of what the model built. Writing them into ``AGENTS.md`` instead would
put benchmark text inside the artifact under evaluation, and would then have to
be scrubbed back out of every snapshot before scoring.
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


# Reasons that mean "the model stopped early", not "the model was done".
_INCOMPLETE_FINISH_REASONS = {
    "length",
    "max_output_tokens",
    "max_tokens",
    "pause_turn",
    "tool-calls",
    "tool_calls",
    "tool_use",
}

#: Name of the agent definition we generate to carry a benchmark's system
#: prompt. Only created when there is one to carry.
_BENCHMARK_AGENT_NAME = "agentprobe"


class OpenCodeAgent(BaseAgent):
    """Run the OpenCode CLI inside an AgentProbe sandbox."""

    # OpenCode reads AGENTS.md; CLAUDE.md is honoured as a fallback. Listed for
    # benchmarks that must inject into the workspace — this agent does not.
    project_instruction_filenames = ("AGENTS.md", "CLAUDE.md")

    _DEFAULT_VERSION = "1.1.21"
    _DEFAULT_NPM_PACKAGE = "opencode-ai"
    _DEFAULT_EXTRA_PACKAGES = ("opencode-linux-x64",)
    _OFFLINE_INSTALL_PATH = "/tmp/offline_package"
    _CONFIG_PATH = "/root/.config/opencode/opencode.json"
    _NPM_CHECK = "command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1"
    # Seconds left between killing opencode and the sandbox's own cap, so
    # on_complete (workspace export, verification) still gets to run. Mirrors
    # ClaudeCodeAgent so both agents are bounded the same way.
    _POST_RUN_BUFFER_SEC = 600

    _PROXY_PATH = "/tmp/agentprobe_opencode_gateway_proxy.mjs"
    _PROXY_LOG = "/tmp/agentprobe_opencode_gateway_proxy.log"
    _PROXY_PID = "/tmp/agentprobe_opencode_gateway_proxy.pid"
    _PROXY_PORT = 18080

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._started_sessions: set[str] = set()
        self._trace_lines: dict[str, list[str]] = {}
        self._last_assistants: dict[str, LastAssistant] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _version(self) -> str:
        configured = str(self.agent_config.params.get("opencode_version") or "").strip()
        if configured:
            return configured
        version = str(self.agent_config.version or "").strip()
        # AgentConfig.version defaults to a Claude Code release. Using that as
        # an OpenCode version would fail at install with a confusing 404.
        return self._DEFAULT_VERSION if not version or version.startswith("2.1.") else version

    def _uses_gateway_proxy(self) -> bool:
        """Whether to route model traffic through the in-sandbox normalizer.

        Off by default: the proxy exists to paper over a gateway that mishandles
        streaming, and pointing every deployment at a workaround for one network
        would be wrong. Turn it on with ``params.gateway_proxy: true``.
        """
        return bool(self.agent_config.params.get("gateway_proxy", False))

    @staticmethod
    def _api_base_url(base_url: str) -> str:
        normalized = str(base_url or "").rstrip("/")
        return normalized if normalized.endswith("/v1") else f"{normalized}/v1"

    def _provider_config(self, base_url: str, api_key: str) -> dict[str, Any]:
        model_name = self.model_config.model_name
        output_limit = int(self.model_config.max_tokens)
        model_options: dict[str, Any] = {
            "name": model_name,
            "limit": {
                "context": int(self.agent_config.params.get("context_window", 200_000)),
                "output": output_limit,
            },
        }
        if self.model_config.format == "anthropic" and self.model_config.thinking != "off":
            model_options["options"] = {
                "thinking": {
                    "type": "enabled",
                    # Must stay under the output cap or the request is rejected.
                    "budgetTokens": min(
                        max(1, output_limit - 2),
                        int(self.model_config.max_thinking_tokens),
                    ),
                }
            }
        anthropic = self.model_config.format == "anthropic"
        return {
            "npm": "@ai-sdk/anthropic" if anthropic else "@ai-sdk/openai-compatible",
            "name": "AgentProbe provider",
            "options": {"baseURL": self._api_base_url(base_url), "apiKey": api_key},
            "models": {model_name: model_options},
        }

    def _opencode_config(self, base_url: str, api_key: str, system_prompt: str) -> dict[str, Any]:
        model_ref = f"agentprobe/{self.model_config.model_name}"
        config: dict[str, Any] = {
            "model": model_ref,
            "small_model": model_ref,
            # Session titling spends a model call per turn on something no
            # benchmark reads.
            "agent": {"title": {"disable": True}},
            "provider": {"agentprobe": self._provider_config(base_url, api_key)},
            # The sandbox is the isolation boundary; an interactive permission
            # prompt inside it would just hang the run. "question" stays denied
            # so the agent answers rather than asking and stalling.
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
        if system_prompt:
            config["agent"][_BENCHMARK_AGENT_NAME] = {
                "description": "AgentProbe benchmark agent",
                "mode": "primary",
                "prompt": system_prompt,
            }
        return config

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    async def install(self, sb: Sandbox) -> ExecResult:
        base_url, api_key = self.model_config.base_url, self.model_config.api_key
        sb.env_vars.update(
            {
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_CONFIG": self._CONFIG_PATH,
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

        effective_base_url = base_url
        if self._uses_gateway_proxy():
            proxy = await self._start_gateway_proxy(sb, base_url)
            if proxy.exit_code != 0:
                return proxy
            effective_base_url = f"http://127.0.0.1:{self._PROXY_PORT}"

        api_base_url = self._api_base_url(effective_base_url)
        if self.model_config.format == "anthropic":
            sb.env_vars.update({"ANTHROPIC_API_KEY": api_key, "ANTHROPIC_BASE_URL": api_base_url})
        else:
            sb.env_vars.update({"OPENAI_API_KEY": api_key, "OPENAI_BASE_URL": api_base_url})

        config = self._opencode_config(
            effective_base_url, api_key, str(sb.spec.append_system_prompt or "").strip()
        )
        await sb.exec_cmd(f"mkdir -p {shlex.quote(str(Path(self._CONFIG_PATH).parent))}")
        await sb.write_file(self._CONFIG_PATH, json.dumps(config, ensure_ascii=False, indent=2))

        if self.agent_config.mcp_host_path:
            logger.warning("OpenCode MCP injection is not implemented; ignoring mcp_host_path")

        verify = await sb.exec_cmd("opencode --version", timeout_sec=60)
        if verify.exit_code == 0:
            logger.debug(
                "OpenCode ready: version={}, model={}, proxy={}",
                verify.stdout.strip() or self._version(),
                self.model_config.model_name,
                self._uses_gateway_proxy(),
            )
        return verify

    async def _install_online(self, sb: Sandbox) -> ExecResult:
        npm_result = await self._ensure_npm(sb)
        if npm_result.exit_code != 0:
            return npm_result
        package = str(self.agent_config.params.get("npm_package") or self._DEFAULT_NPM_PACKAGE)
        return await sb.exec_cmd(
            "npm --fetch-retries=3 --fetch-retry-mintimeout=2000 "
            f"--fetch-retry-maxtimeout=10000 i -g {shlex.quote(f'{package}@{self._version()}')}"
        )

    async def _ensure_npm(self, sb: Sandbox) -> ExecResult:
        check = await sb.exec_cmd(self._NPM_CHECK)
        if check.exit_code == 0:
            return check
        logger.info("node/npm missing; installing Node.js 22.x")
        result = check
        for cmd in (
            "apt-get update -o Acquire::Retries=3",
            "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
            "curl ca-certificates gnupg",
            "curl --retry 3 --retry-delay 2 --retry-all-errors -fsSL "
            "https://deb.nodesource.com/setup_22.x | bash -",
            "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nodejs",
        ):
            result = await sb.exec_cmd(cmd)
            if result.exit_code != 0:
                return result
        return await sb.exec_cmd(self._NPM_CHECK)

    async def _install_offline(self, sb: Sandbox) -> ExecResult:
        """Install Node and the OpenCode packages from the mounted tarballs.

        Unlike Claude Code there is no single self-contained binary to unpack:
        Node has to be laid down first, then ``npm install -g`` each ``.tgz``.
        """
        version = self._version()
        node_dir = f"node-v{self.agent_config.offline_node_version}-linux-x64"
        package = str(self.agent_config.params.get("npm_package") or self._DEFAULT_NPM_PACKAGE)
        extra = self.agent_config.params.get(
            "offline_extra_packages", list(self._DEFAULT_EXTRA_PACKAGES)
        )
        if not isinstance(extra, list):
            raise ValueError("offline_extra_packages must be a list")

        mount = self.agent_config.offline_mount_path.rstrip("/")
        install_path = self._OFFLINE_INSTALL_PATH
        node_bin = f"{install_path}/{node_dir}/bin"
        node_archive = f"{mount}/{node_dir}.tar.gz"

        commands = [
            "set -e",
            f"mkdir -p {shlex.quote(install_path)}",
            f"test -f {shlex.quote(node_archive)} || "
            f"{{ echo 'missing offline Node archive: {node_archive}' >&2; exit 1; }}",
            f"tar -xzf {shlex.quote(node_archive)} -C {shlex.quote(install_path)}",
            f"export PATH={shlex.quote(node_bin)}:$PATH",
        ]
        for name in [*(str(item) for item in extra), package]:
            archive = f"{mount}/{name.replace('@', '').replace('/', '-')}-{version}.tgz"
            commands.extend(
                [
                    f"test -f {shlex.quote(archive)} || "
                    f"{{ echo 'missing offline package: {archive}' >&2; exit 1; }}",
                    f"{shlex.quote(f'{node_bin}/npm')} install -g {shlex.quote(archive)}",
                ]
            )
        commands.append("command -v opencode")

        result = await sb.exec_cmd(f"bash -lc {shlex.quote('; '.join(commands))}")
        if result.exit_code == 0:
            current = await sb.exec_cmd("printf '%s' \"$PATH\"")
            inherited = current.stdout.strip() if current.exit_code == 0 else ""
            sb.env_vars["PATH"] = ":".join(p for p in (node_bin, inherited) if p)
        return result

    async def _start_gateway_proxy(self, sb: Sandbox, upstream_base_url: str) -> ExecResult:
        """Start the in-sandbox normalizer that fakes SSE for a non-streaming upstream."""
        source = Path(__file__).with_name("opencode_gateway_proxy.mjs").read_text(encoding="utf-8")
        await sb.write_file(self._PROXY_PATH, source)
        started = await sb.exec_cmd(
            " ".join(
                (
                    f"AGENTPROBE_UPSTREAM_BASE_URL="
                    f"{shlex.quote(self._api_base_url(upstream_base_url))}",
                    f"AGENTPROBE_PROXY_PORT={self._PROXY_PORT}",
                    "nohup node",
                    shlex.quote(self._PROXY_PATH),
                    f">{shlex.quote(self._PROXY_LOG)} 2>&1 &",
                    f"echo $! >{shlex.quote(self._PROXY_PID)}",
                )
            ),
            timeout_sec=30,
        )
        if started.exit_code != 0:
            return started

        health = json.dumps(f"http://127.0.0.1:{self._PROXY_PORT}/healthz")
        probe = (
            f"fetch({health}).then(r => process.exit(r.ok ? 0 : 1))"
            ".catch(() => process.exit(1))"
        )
        return await sb.exec_cmd(
            f"for _ in $(seq 1 50); do node -e {shlex.quote(probe)} && exit 0; sleep 0.2; done; "
            f"tail -50 {shlex.quote(self._PROXY_LOG)} >&2; exit 1",
            timeout_sec=60,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run_prompt(self, sb: Sandbox, prompt: str) -> ExecResult:
        session_id = sb.session_id
        resume = session_id in self._started_sessions

        prompt_path = f"/tmp/prompt_{uuid.uuid4().hex[:8]}.txt"
        await sb.write_file(prompt_path, prompt)

        parts = [
            f"cd {shlex.quote(sb.spec.workspace or '/root')} &&",
            f"cat {shlex.quote(prompt_path)} |",
            "opencode run",
            f"--model {shlex.quote(f'agentprobe/{self.model_config.model_name}')}",
            "--format json",
        ]
        agent_name = self._agent_flag(sb)
        if agent_name:
            parts.extend(("--agent", shlex.quote(agent_name)))
        extra_flags = self.agent_config.params.get("extra_flags") or []
        if isinstance(extra_flags, str):
            extra_flags = shlex.split(extra_flags)
        if not isinstance(extra_flags, list):
            raise ValueError("OpenCode extra_flags must be a string or a list")
        parts.extend(shlex.quote(str(flag)) for flag in extra_flags)
        if resume:
            # Same conversation as the previous round; multi-turn benchmarks
            # whose constraints reference earlier rounds depend on this.
            parts.append("--continue")

        timeout = sb.spec.agent_timeout_sec or max(
            60, sb.spec.timeout_sec - self._POST_RUN_BUFFER_SEC
        )
        result = await sb.exec_cmd(" ".join(parts), timeout_sec=max(1, int(timeout)))
        self._started_sessions.add(session_id)

        self._record_round(session_id, prompt, result)
        return result

    def _agent_flag(self, sb: Sandbox) -> str:
        configured = str(self.agent_config.params.get("agent_name") or "").strip()
        if configured:
            return configured
        return _BENCHMARK_AGENT_NAME if str(sb.spec.append_system_prompt or "").strip() else ""

    def _record_round(self, session_id: str, prompt: str, result: ExecResult) -> None:
        """Fold one round's stdout into the session trace and last-assistant state."""
        stdout = result.stdout
        lines = self._trace_lines.setdefault(session_id, [])
        lines.append(
            json.dumps(
                {"type": "user", "message": {"role": "user", "content": prompt}},
                ensure_ascii=False,
            )
        )
        lines.extend(stdout.splitlines())

        inspection = inspect_opencode_jsonl(stdout)
        protocol_error = extract_opencode_error(stdout)
        process_error = ""
        if not protocol_error and result.exit_code != 0:
            process_error = result.stderr.strip() or f"OpenCode exited with {result.exit_code}"

        text = extract_opencode_final_text(stdout)
        if not inspection["saw_jsonl"] and stdout.strip():
            # No JSONL at all: the CLI printed something plain, keep it rather
            # than reporting an empty reply.
            text = stdout.strip()

        finish_reason = str(inspection["finish_reason"] or "").strip().lower()
        complete = bool(
            result.exit_code == 0
            and text
            and not protocol_error
            and not process_error
            and not inspection["has_error_event"]
            and (inspection["has_assistant_text"] if inspection["saw_jsonl"] else text)
            and (inspection["has_terminal_event"] if inspection["saw_jsonl"] else True)
            and not inspection["final_step_has_tool_use"]
            and finish_reason not in _INCOMPLETE_FINISH_REASONS
        )
        error_message = protocol_error or process_error
        if not error_message and not complete and inspection["saw_jsonl"]:
            error_message = (
                "OpenCode stopped without a complete final response "
                f"(finish_reason={inspection['finish_reason']!r}, "
                f"terminal_tool_use={inspection['final_step_has_tool_use']})"
            )

        self._last_assistants[session_id] = LastAssistant(
            stop_reason="error" if (protocol_error or process_error) else finish_reason or None,
            error_message=error_message or None,
            content_text=text,
            is_complete_response=complete,
        )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def collect_traces(self, sb: Sandbox, output_dir: Path) -> None:
        lines = self._trace_lines.get(sb.session_id, [])
        if not lines:
            logger.warning("[{}] OpenCode trace is empty", sb.session_id)
            return
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{sb.session_id}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    async def collect_last_assistant(
        self, sb: Sandbox, output_dir: Path
    ) -> LastAssistant | None:
        del output_dir  # OpenCode's reply is parsed from stdout, not from disk
        return self._last_assistants.get(sb.session_id)
