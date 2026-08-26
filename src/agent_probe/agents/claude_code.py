"""Claude Code agent — installs and runs the claude CLI inside a sandbox."""

from __future__ import annotations

import json
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from agent_probe.core.agent import BaseAgent
from agent_probe.core.models import LastAssistant
from agent_probe.core.sandbox import ExecResult

if TYPE_CHECKING:
    from agent_probe.config import AgentConfig, ModelConfig
    from agent_probe.core.sandbox import Sandbox


class ClaudeCodeAgent(BaseAgent):
    """Claude Code agent backed by the ``@anthropic-ai/claude-code`` CLI.

    Uses the current ``model_config.model_name`` for all Claude Code model env vars::

        ANTHROPIC_MODEL
        ANTHROPIC_DEFAULT_HAIKU_MODEL
        ANTHROPIC_DEFAULT_SONNET_MODEL
        ANTHROPIC_DEFAULT_OPUS_MODEL

    Authentication and base URL come from ``model_config``.
    """

    _MODEL_ENV_KEYS = (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
    )
    _NPM_PACKAGE = "@anthropic-ai/claude-code"
    _PACKAGE_BASENAME = "anthropic-ai-claude-code"
    _OFFLINE_INSTALL_PATH = "/tmp/offline_package"
    # Seconds reserved between killing claude and the sandbox wait_for cap,
    # so on_complete (patch + judge eval) still gets to run.
    _POST_CLAUDE_BUFFER_SEC = 600
    # Accepted values of ``ModelConfig.thinking`` beyond "off".
    _EFFORT_LEVELS = ("low", "medium", "high", "max")

    def __init__(
        self,
        agent_config: AgentConfig,
        model_config: ModelConfig,
    ) -> None:
        super().__init__(agent_config=agent_config, model_config=model_config)
        # Session ids already created by this agent, so a repeated id resumes
        # rather than fails. One agent instance serves one Sandbox.run().
        self._started_sessions: set[str] = set()
        self._thinking_flag = self._build_thinking_flag()

    def _build_thinking_flag(self) -> str:
        """Turn ``ModelConfig.thinking`` into CLI flags.

        ``off`` (the default) emits nothing rather than ``--thinking disabled``,
        so benchmarks that never set the field keep the CLI's own default.
        """
        effort = (self.model_config.thinking or "off").strip().lower()
        if effort in ("", "off"):
            return ""
        if effort not in self._EFFORT_LEVELS:
            raise ValueError(
                f"ModelConfig.thinking must be 'off' or one of "
                f"{self._EFFORT_LEVELS}, got {self.model_config.thinking!r}"
            )
        return f" --thinking adaptive --effort {effort}"

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _get_env(self) -> dict[str, str]:
        # The thinking budget has to stay under the output cap, or the CLI
        # rejects the request outright.
        thinking_budget = min(
            self.model_config.max_thinking_tokens,
            max(1, self.model_config.max_tokens - 2),
        )
        env: dict[str, str] = {
            "ANTHROPIC_API_KEY": self.model_config.api_key,
            "ANTHROPIC_AUTH_TOKEN": self.model_config.api_key,
            # Without this the CLI silently applies its own default and
            # ``ModelConfig.max_tokens`` has no effect at all.
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.model_config.max_tokens),
            "MAX_THINKING_TOKENS": str(thinking_budget),
            "IS_SANDBOX": "1",
        }

        if self.model_config.base_url:
            env["ANTHROPIC_BASE_URL"] = self.model_config.base_url

        # Keep model selection scoped to the current eval unit.
        if self.model_config.model_name:
            for env_key in self._MODEL_ENV_KEYS:
                env[env_key] = self.model_config.model_name

        return {k: v for k, v in env.items() if v}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def install(self, sb: Sandbox) -> ExecResult:
        envs = self._get_env()
        sb.env_vars.update(envs)

        if self.agent_config.offline:
            result = await self._install_offline(sb)
        else:
            npm_result = await self._ensure_npm(sb)
            if npm_result.exit_code != 0:
                return npm_result

            result = await sb.exec_cmd(f"npm i -g {self._NPM_PACKAGE}@{self.agent_config.version}")

        if result.exit_code != 0:
            return result

        await self._patch_exit_plan_mode(sb)

        # Upload MCP config if provided
        if self.agent_config.mcp_host_path:
            content = Path(self.agent_config.mcp_host_path).read_text(encoding="utf-8")
            await sb.write_file("/tmp/mcp_config.json", content)
            logger.debug("MCP config uploaded to /tmp/mcp_config.json")
        return result

    async def _ensure_npm(self, sb: Sandbox) -> ExecResult:
        """Ensure npm is available before installing Claude Code."""
        check = await sb.exec_cmd("command -v npm")
        if check.exit_code == 0:
            return check

        logger.info("npm not found, installing Node.js 22.x...")
        result = check
        for cmd in [
            "apt-get update",
            "apt-get install -y curl gnupg sudo",
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
        ]:
            result = await sb.exec_cmd(cmd)
            if result.exit_code != 0:
                return result
        return result

    async def _install_offline(self, sb: Sandbox) -> ExecResult:
        version = self.agent_config.version
        mount_path = self.agent_config.offline_mount_path.rstrip("/")
        install_path = self._OFFLINE_INSTALL_PATH
        install_bin = f"{install_path}/bin"
        native_archive_prefix = f"{mount_path}/{self._PACKAGE_BASENAME}-linux-"

        logger.info(
            "Installing Claude Code offline: version={}",
            version,
        )
        script = (
            "set -e; "
            "platform=x64; "
            "if [ -e /lib/ld-musl-x86_64.so.1 ]; then platform=x64-musl; fi; "
            f"native_archive={shlex.quote(native_archive_prefix)}"
            f'"${{platform}}"-{shlex.quote(version)}.tgz; '
            'test -f "$native_archive" || '
            '(echo "missing offline package: $native_archive" >&2; exit 1); '
            f"mkdir -p {shlex.quote(install_bin)}; "
            f'tar -xOzf "$native_archive" package/claude > '
            f"{shlex.quote(f'{install_bin}/claude')}; "
            f"chmod +x {shlex.quote(f'{install_bin}/claude')}; "
            f"export PATH={shlex.quote(install_bin)}:$PATH; "
            "claude --version; "
            "command -v claude"
        )
        result = await sb.exec_cmd(f"sh -c {shlex.quote(script)}")
        if result.exit_code == 0:
            sb.env_vars["PATH"] = f"{install_bin}:$PATH"
        return result

    async def _patch_exit_plan_mode(self, sb: Sandbox) -> None:
        script = r"""
set -e
claude_bin=$(command -v claude)
target=$(readlink -f "$claude_bin")
python_bin=$(command -v python3 || command -v python || true)
if [ -z "$python_bin" ]; then
    echo "ExitPlanMode patch skipped: python not found" >&2
    exit 1
fi
"$python_bin" - "$target" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
old = b'behavior:"ask",message:"Exit plan mode?"'
new = b'behavior:"allow",message:""             '
assert len(old) == len(new)
data = path.read_bytes()
count = data.count(old)
if count:
    path.write_bytes(data.replace(old, new))
    patched = path.read_bytes()
    assert old not in patched and patched.count(new) >= count
    print(f"ExitPlanMode patch applied: {count}")
elif new in data:
    print("ExitPlanMode patch already applied")
else:
    print("ExitPlanMode patch skipped: pattern not found")
PY
""".strip()
        result = await sb.exec_cmd(f"sh -c {shlex.quote(script)}")
        if result.exit_code != 0:
            logger.warning(
                "ExitPlanMode patch failed: {}",
                (result.stderr or result.stdout)[-500:],
            )

    async def run_prompt(self, sb: Sandbox, prompt: str) -> ExecResult:
        prompt_file = f"/tmp/prompt_{uuid.uuid4().hex[:8]}.txt"
        await sb.write_file(prompt_file, prompt)
        cd_prefix = f"cd {sb.spec.workspace} && " if sb.spec.workspace else ""
        mcp_flag = " --mcp-config /tmp/mcp_config.json" if self.agent_config.mcp_host_path else ""
        system_prompt_flag = (
            f" --append-system-prompt {shlex.quote(sb.spec.append_system_prompt)}"
            if sb.spec.append_system_prompt
            else ""
        )
        # A session id may only be *created* once. When the caller keeps one
        # conversation across rounds, later rounds must resume it instead so the
        # model still sees the earlier turns.
        resuming = sb.session_id in self._started_sessions
        if resuming:
            # The previous round's CLI process can linger and hold the session
            # file; drop it before resuming. The bracket keeps the pattern from
            # matching the shell that runs pkill (its own cmdline contains the
            # pattern text), which would otherwise kill this command.
            await sb.exec_cmd("pkill -f '[c]laude.*-p' || true")
            session_flag = f"--resume {sb.session_id}"
        else:
            session_flag = f"--session-id {sb.session_id}"

        deadline = max(60, sb.spec.timeout_sec - self._POST_CLAUDE_BUFFER_SEC)
        # Feed the prompt on stdin rather than inlining it into argv: Linux caps
        # a single argument at 128 KiB (MAX_ARG_STRLEN), and judge prompts that
        # embed a full round transcript exceed that, failing with
        # "Argument list too long" before claude even starts.
        cmd = (
            f"{cd_prefix}"
            f"cat {prompt_file} | claude -p "
            f"{session_flag} "
            f"--dangerously-skip-permissions"
            f"{mcp_flag}"
            f"{self._thinking_flag}"
            f"{system_prompt_flag}"
        )
        result = await sb.exec_cmd(cmd, timeout_sec=deadline)
        self._started_sessions.add(sb.session_id)
        if result.exit_code != 0:
            logger.warning(
                "[{}] claude exit_code={} stderr={}",
                sb.session_id,
                result.exit_code,
                (result.stderr or "")[-500:],
            )
            if result.exit_code == -1:
                # Signal-killed (timeout or OOM): workspace may still have
                # usable artifacts, let judge evaluate instead of discarding.
                result.exit_code = 0
        return result

    async def collect_traces(self, sb: Sandbox, output_dir: Path) -> None:
        """Copy the current session's trace file from the sandbox.

        Claude writes under ~/.claude/projects/{cwd-encoded}/{sid}.jsonl; the
        cwd encoding isn't part of our contract, so locate the file by session id.
        """
        sid = sb.session_id
        home_result = await sb.exec_cmd("printf '%s' \"$HOME\"")
        home = home_result.stdout.strip() or "/root"
        trace_root = f"{home.rstrip('/')}/.claude/projects"
        find_script = (
            f"dir={shlex.quote(trace_root)}; "
            '[ -d "$dir" ] || dir=/root/.claude/projects; '
            '[ -d "$dir" ] && '
            f'find "$dir" -name {shlex.quote(f"{sid}.jsonl")} -type f 2>/dev/null '
            "| head -n 1"
        )
        find_result = await sb.exec_cmd(f"bash -lc {shlex.quote(find_script)}")
        paths = [p.strip() for p in find_result.stdout.splitlines() if p.strip()]
        if not paths:
            logger.warning("[{}] Claude Code trace not found in sandbox", sid)
            return

        traces_dir = output_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        content = await sb.read_file(paths[0])
        (traces_dir / f"{sid}.jsonl").write_text(content, encoding="utf-8")

    async def collect_last_assistant(self, sb: Sandbox, output_dir: Path) -> LastAssistant | None:
        """Parse last assistant event from the current session jsonl.

        Fields are resolved independently:
            stop_reason / error_message — absolute last assistant event
            content_text                — last non-error assistant with non-empty text
        """
        trace_path = output_dir / "traces" / f"{sb.session_id}.jsonl"
        if not trace_path.is_file():
            logger.warning("Claude Code trace not found: {}", trace_path)
            return None

        last_event: dict | None = None
        last_text: str = ""
        try:
            with trace_path.open("r", encoding="utf-8") as f:
                for line in f:
                    event = self._parse_assistant_event(line)
                    if event is None:
                        continue
                    last_event = event
                    if event.get("isApiErrorMessage"):
                        continue
                    text = self._extract_text(event.get("message") or {})
                    if text:
                        last_text = text
        except OSError as e:
            logger.warning("Failed to read claude-code trace {}: {}", trace_path, e)
            return None

        if last_event is None:
            return None

        msg = last_event.get("message") or {}
        if last_event.get("isApiErrorMessage"):
            stop_reason: str | None = "error"
            error_message: str | None = self._extract_text(msg) or None
        else:
            stop_reason = msg.get("stop_reason")
            error_message = None
        return LastAssistant(
            stop_reason=stop_reason,
            error_message=error_message,
            content_text=last_text,
        )

    @staticmethod
    def _parse_assistant_event(line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        return event if event.get("type") == "assistant" else None

    @staticmethod
    def _extract_text(msg: dict) -> str:
        return "".join(
            block.get("text", "")
            for block in (msg.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
