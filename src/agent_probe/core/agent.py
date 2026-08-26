"""BaseAgent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_probe.config import AgentConfig, ModelConfig
    from agent_probe.core.models import LastAssistant
    from agent_probe.core.sandbox import ExecResult, Sandbox


class BaseAgent(ABC):
    """Abstract agent that can be installed and executed inside a Sandbox.

    Agents are instantiated by ``Sandbox.run()`` via an agent factory.
    They receive the full ``AgentConfig`` and ``ModelConfig`` for the current
    eval unit, and use them to configure environment variables, CLI flags, etc.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        model_config: ModelConfig,
    ) -> None:
        self.agent_config = agent_config
        self.model_config = model_config

    @abstractmethod
    async def install(self, sb: Sandbox) -> ExecResult:
        """Install the agent runtime inside the sandbox (e.g. npm install, pip install)."""

    @abstractmethod
    async def run_prompt(self, sb: Sandbox, prompt: str) -> ExecResult:
        """Send a prompt to the agent and wait for completion."""

    @abstractmethod
    async def collect_traces(self, sb: Sandbox, output_dir: Path) -> None:
        """Copy trace files from the sandbox to ``output_dir/traces/`` on the host."""

    async def collect_last_assistant(
        self, sb: Sandbox, output_dir: Path
    ) -> "LastAssistant | None":
        """Parse the last assistant message from collected traces."""
        return None
