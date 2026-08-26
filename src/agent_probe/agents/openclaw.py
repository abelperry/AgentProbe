"""OpenClaw agent — installs and runs the OpenClaw CLI inside a sandbox."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from agent_probe.core.agent import BaseAgent
from agent_probe.core.models import LastAssistant

if TYPE_CHECKING:
    from agent_probe.config import AgentConfig, ModelConfig
    from agent_probe.core.sandbox import ExecResult, Sandbox


class OpenClawAgent(BaseAgent):
    """OpenClaw agent backed by the ``openclaw`` CLI.

    Supports both OpenAI and Anthropic provider formats via
    ``model_config.format``.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def install(self, sb: Sandbox) -> ExecResult:
        """Verify OpenClaw is available; if not, install Node.js + npm."""
        # Check if openclaw already exists
        check = await sb.exec_cmd("command -v openclaw")
        if check.exit_code == 0:
            logger.debug("OpenClaw found in sandbox")
            return await self._configure(sb)

        logger.info("OpenClaw not found, installing via npm...")

        # Ensure Node.js is available
        node_check = await sb.exec_cmd("command -v node")
        if node_check.exit_code != 0:
            logger.info("Node.js not found, installing Node.js 22.x...")
            result = await sb.exec_cmd(
                "bash -c 'curl -fsSL https://deb.nodesource.com/setup_22.x | bash - "
                "&& apt-get install -y nodejs'"
            )
            if result.exit_code != 0:
                return result

        # Install OpenClaw
        version = self.agent_config.version
        install_cmd = f"npm install -g openclaw@{version}"
        result = await sb.exec_cmd(install_cmd)
        if result.exit_code != 0:
            return result

        return await self._configure(sb)

    async def _configure(self, sb: Sandbox) -> ExecResult:
        """Configure OpenClaw with model API settings."""
        mc = self.model_config
        workspace = sb.spec.workspace or "/workspace"

        # Onboard (non-interactive)
        result = await sb.exec_cmd(
            "openclaw onboard --non-interactive --accept-risk "
            "--auth-choice skip --skip-daemon --skip-channels "
            "--skip-skills --skip-health --skip-ui "
            f"--workspace {workspace}"
        )
        if result.exit_code != 0:
            logger.warning("OpenClaw onboard non-zero exit: {}", result.stderr)

        # Build provider config based on format
        if mc.format == "openai":
            provider_config = {
                "baseUrl": mc.base_url,
                "apiKey": mc.api_key,
                "models": [
                    {
                        "id": mc.model_name,
                        "name": mc.model_name,
                        "compat": {
                            "supportsDeveloperRole": False,
                            "maxTokensField": "max_tokens",
                        },
                        "api": "openai-completions",
                        "reasoning": mc.thinking != "off",
                    }
                ],
            }
        else:
            provider_config = {
                "baseUrl": mc.base_url,
                "apiKey": mc.api_key,
                "models": [
                    {
                        "id": mc.model_name,
                        "name": mc.model_name,
                        "api": "anthropic-messages",
                        "reasoning": mc.thinking != "off",
                        "maxTokens": mc.max_tokens,
                        "compat": {"maxTokensField": "max_tokens"},
                    }
                ],
            }

        provider_json = json.dumps(provider_config, ensure_ascii=False)
        result = await sb.exec_cmd(
            f"openclaw config set 'models.providers.zclawbench' '{provider_json}' --json"
        )
        if result.exit_code != 0:
            return result

        result = await sb.exec_cmd(
            f"openclaw models set zclawbench/{mc.model_name}"
        )
        if result.exit_code != 0:
            return result

        # Enable full tool profile (web_search, browser, web_fetch, etc.)
        await sb.exec_cmd("openclaw config set tools.profile full")

        # Start gateway in background (required for agent_judge tasks etc.)
        result = await self._start_gateway(sb)

        logger.info("[{}] OpenClaw configured: model={}, format={}, thinking={}",
                    sb.session_id, mc.model_name, mc.format, mc.thinking)
        return result

    async def _start_gateway(self, sb: Sandbox) -> ExecResult:
        """Start OpenClaw gateway in background."""
        result = await sb.exec_cmd(
            "bash -c '"
            "nohup openclaw gateway run --allow-unconfigured "
            "--bind loopback --port 18789 "
            "> /tmp/openclaw-gateway.log 2>&1 & "
            "GWPID=$!; sleep 2; "
            "if kill -0 $GWPID 2>/dev/null; then "
            "  echo \"gateway started (pid=$GWPID)\"; "
            "else "
            "  echo \"gateway failed\" >&2; "
            "  tail -20 /tmp/openclaw-gateway.log >&2; exit 1; "
            "fi'"
        )
        if result.exit_code != 0:
            logger.warning("[{}] OpenClaw gateway failed to start: {}", sb.session_id, result.stderr)
        else:
            logger.info("[{}] OpenClaw gateway started", sb.session_id)
        return result

    async def run_prompt(self, sb: Sandbox, prompt: str) -> ExecResult:
        """Execute a task using openclaw agent CLI."""
        prompt_file = f"/tmp/prompt_{uuid.uuid4().hex[:8]}.txt"
        await sb.write_file(prompt_file, prompt)

        timeout = self.model_config.timeout
        cd_prefix = f"cd {sb.spec.workspace} && " if sb.spec.workspace else ""
        cmd = (
            f"{cd_prefix}"
            f"openclaw agent "
            f"--session-id {sb.session_id} "
            f'--message "$(cat {prompt_file})" '
            f"--json "
            f"--thinking {self.model_config.thinking} "
            f"--timeout {timeout}"
        )
        logger.info("[{}] Running prompt: workspace={}, timeout={}s, prompt_len={}",
                    sb.session_id, sb.spec.workspace, timeout, len(prompt))
        return await sb.exec_cmd(cmd)

    async def collect_traces(self, sb: Sandbox, output_dir: Path) -> None:
        """Copy the current session's trace file from the sandbox."""
        sid = sb.session_id
        remote = f"/root/.openclaw/agents/main/sessions/{sid}.jsonl"

        check = await sb.exec_cmd(f"test -f {remote}")
        if check.exit_code != 0:
            logger.warning("[{}] OpenClaw trace not found at {}", sid, remote)
            return

        traces_dir = output_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        content = await sb.read_file(remote)
        (traces_dir / f"{sid}.jsonl").write_text(content, encoding="utf-8")

    async def collect_last_assistant(
        self, sb: Sandbox, output_dir: Path
    ) -> LastAssistant | None:
        """Parse last assistant message from the current session trace.

        Fields are resolved independently:
            stop_reason / error_message — absolute last assistant message
            content_text                — last assistant message with non-empty text
        """
        trace_path = output_dir / "traces" / f"{sb.session_id}.jsonl"
        if not trace_path.is_file():
            logger.warning("OpenClaw trace not found: {}", trace_path)
            return None

        last_msg: dict | None = None
        last_text: str = ""
        try:
            with trace_path.open("r", encoding="utf-8") as f:
                for line in f:
                    msg = self._parse_assistant_message(line)
                    if msg is None:
                        continue
                    last_msg = msg
                    text = self._extract_text(msg)
                    if text:
                        last_text = text
        except OSError as e:
            logger.warning("Failed to read openclaw trace {}: {}", trace_path, e)
            return None

        if last_msg is None:
            return None
        return LastAssistant(
            stop_reason=last_msg.get("stopReason"),
            error_message=last_msg.get("errorMessage"),
            content_text=last_text,
        )

    @staticmethod
    def _parse_assistant_message(line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if event.get("type") != "message":
            return None
        msg = event.get("message") or {}
        return msg if msg.get("role") == "assistant" else None

    @staticmethod
    def _extract_text(msg: dict) -> str:
        return "".join(
            block.get("text", "")
            for block in (msg.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
