"""mini-swe-agent integration — drives the upstream DefaultAgent in-process.

The upstream agent loop is synchronous; we run it on a worker thread via
``asyncio.to_thread``. Bash commands hop back into AgentProbe's asyncio loop
through ``run_coroutine_threadsafe`` and execute inside the OpenSandbox.

Upstream protocols satisfied:
- ``minisweagent.Environment``: ``SandboxEnvironment`` below
- ``minisweagent.Model``: reuses ``minisweagent.models.get_model`` (default
  ``LitellmTextbasedModel`` from the backticks config)
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from agent_probe.core.agent import BaseAgent
from agent_probe.core.models import LastAssistant
from agent_probe.core.sandbox import ExecResult

# Silence mini-swe-agent's startup banner — it prints a version banner to stdout
# on first import. Must be set before any minisweagent.* import.
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

if TYPE_CHECKING:
    from agent_probe.core.sandbox import Sandbox


_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_CONFIG_PATH = Path("benchmarks/swebench/data/swebench_backticks.yaml")


class SandboxEnvironment:
    """Adapter satisfying ``minisweagent.Environment`` over an AgentProbe Sandbox.

    ``execute`` is synchronous (mini's contract) and bounces the bash command
    back to the asyncio loop via ``run_coroutine_threadsafe``.
    """

    def __init__(
        self,
        sb: Sandbox,
        loop: asyncio.AbstractEventLoop,
        *,
        cwd: str = "/testbed",
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> None:
        self._sb = sb
        self._loop = loop
        self._cwd = cwd
        self._timeout = timeout
        self._env = env or {}

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        from minisweagent.exceptions import Submitted

        command = action.get("command", "")
        target_cwd = cwd or self._cwd
        env_prefix = "".join(f'export {k}={_shell_quote(v)} && ' for k, v in self._env.items())
        full_cmd = f"cd {target_cwd} && {env_prefix}{command}"
        cmd_timeout = timeout or self._timeout

        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._sb.exec_cmd(full_cmd, timeout_sec=cmd_timeout), self._loop
            )
            # Add a margin over the bash timeout so the SDK has time to return
            # a clean timeout result rather than us cancelling it.
            result = fut.result(timeout=cmd_timeout + 30)
            output = {
                "output": result.stdout or "",
                "returncode": int(result.exit_code) if result.exit_code is not None else 0,
                "exception_info": "",
            }
            if result.stderr:
                output["output"] = (output["output"] + "\n" + result.stderr).strip()
        except Exception as e:  # transport / loop errors — surface to model
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"sandbox exec failed: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }

        # Submission protocol: bash output starts with the marker, returncode 0.
        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == _SUBMIT_MARKER and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"cwd": self._cwd, "timeout": self._timeout, **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "cwd": self._cwd,
                        "timeout": self._timeout,
                        "env": self._env,
                    },
                    "environment_type": "agent_probe.agents.mini_swe_agent.SandboxEnvironment",
                }
            }
        }


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


class MiniSWEAgent(BaseAgent):
    """Run mini-swe-agent's ``DefaultAgent`` in-process against an AgentProbe sandbox.

    Reads the upstream config (system/instance templates, step/cost limits,
    observation template) from ``benchmarks/swebench/data/swebench_backticks.yaml``
    relative to AgentProbe's run cwd.
    """

    async def install(self, sb: Sandbox) -> ExecResult:
        # Loop runs on the host; nothing to install inside the sandbox.
        return ExecResult(stdout="", stderr="", exit_code=0)

    async def run_prompt(self, sb: Sandbox, prompt: str) -> ExecResult:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.models import get_model

        cfg = self._load_mini_config()
        agent_cfg = cfg.get("agent", {}) or {}
        env_cfg = cfg.get("environment", {}) or {}
        model_cfg = cfg.get("model", {}) or {}

        loop = asyncio.get_running_loop()
        env = SandboxEnvironment(
            sb,
            loop,
            cwd=env_cfg.get("cwd", sb.spec.workspace or "/testbed"),
            timeout=int(env_cfg.get("timeout", 60)),
            env=dict(env_cfg.get("env") or {}),
        )

        traj_path = self._traj_path(sb)
        traj_path.parent.mkdir(parents=True, exist_ok=True)

        # Set wall-clock budget that fires *before* the AgentProbe sandbox timeout.
        wall = int(agent_cfg.get("wall_time_limit_seconds", 0))
        if wall <= 0:
            wall = max(60, sb.spec.timeout_sec - 120)

        agent = DefaultAgent(
            model=get_model(config=self._build_mini_model_config(model_cfg)),
            env=env,
            **{
                **agent_cfg,
                "wall_time_limit_seconds": wall,
                "output_path": traj_path,
            },
        )

        try:
            info = await asyncio.to_thread(agent.run, task=prompt)
        except Exception as e:
            logger.exception("[{}] mini-swe-agent run raised", sb.session_id)
            agent.save(traj_path)
            return ExecResult(stdout="", stderr=str(e), exit_code=1)

        exit_status = info.get("exit_status") or ""
        submission = info.get("submission") or ""
        rc = 0 if exit_status == "Submitted" else 1
        return ExecResult(stdout=submission, stderr=exit_status, exit_code=rc)

    async def collect_traces(self, sb: Sandbox, output_dir: Path) -> None:
        # DefaultAgent.save() already wrote the trajectory; nothing to copy
        # out of the sandbox.
        return

    async def collect_last_assistant(self, sb: Sandbox, output_dir: Path) -> LastAssistant | None:
        traj_path = self._traj_path(sb, output_dir)
        if not traj_path.is_file():
            logger.warning("mini-swe-agent trajectory not found: {}", traj_path)
            return None
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse mini-swe-agent trajectory {}: {}", traj_path, e)
            return None

        info = (data.get("info") or {})
        exit_status = info.get("exit_status") or ""
        submission = info.get("submission") or ""
        error = None
        if exit_status and exit_status != "Submitted":
            error = info.get("exception_str") or exit_status

        # last assistant text — fall back to submission text
        last_text = submission
        for msg in reversed(data.get("messages") or []):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    last_text = content
                    break

        return LastAssistant(
            stop_reason=exit_status.lower() or None,
            error_message=str(error) if error else None,
            content_text=last_text or "",
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _traj_path(self, sb: Sandbox, output_dir: Path | None = None) -> Path:
        base = Path(output_dir) if output_dir is not None else Path(sb.spec.output_dir or ".")
        return base / "traces" / f"{sb.session_id}.traj.json"

    def _load_mini_config(self) -> dict[str, Any]:
        if not _CONFIG_PATH.is_file():
            raise FileNotFoundError(
                f"mini-swe-agent config not found at {_CONFIG_PATH} (run from AgentProbe repo root)"
            )
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _build_mini_model_config(self, file_model_cfg: dict[str, Any]) -> dict[str, Any]:
        """Translate AgentProbe ModelConfig + yaml model block into mini get_model() args."""
        cfg = copy.deepcopy(file_model_cfg)

        provider = "anthropic" if self.model_config.format == "anthropic" else "openai"
        cfg["model_name"] = f"{provider}/{self.model_config.model_name}"

        # Per-call credentials override env. litellm honours api_key/api_base in kwargs.
        kwargs = dict(cfg.get("model_kwargs") or {})
        kwargs.setdefault("api_key", self.model_config.api_key)
        if self.model_config.base_url:
            kwargs.setdefault("api_base", self.model_config.base_url)
            kwargs.setdefault("base_url", self.model_config.base_url)
        kwargs.setdefault("timeout", self.model_config.timeout)
        if self.model_config.max_tokens:
            kwargs.setdefault("max_tokens", self.model_config.max_tokens)
        cfg["model_kwargs"] = kwargs

        # Cost tracking off by default: we run on private gateways litellm can't price.
        cfg.setdefault("cost_tracking", "ignore_errors")
        return cfg
