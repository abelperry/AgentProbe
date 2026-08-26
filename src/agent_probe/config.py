"""Experiment configuration models."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    base_url: str
    api_key: str
    timeout: int = 10800
    max_tokens: int = 32000
    model_name: str = ""  # filled by factory from YAML key if empty
    format: str = "openai"  # "openai" | "anthropic"
    thinking: str = "off"  # "high" | "medium" | "low" | "off"
    max_thinking_tokens: int = 10000  # make it 1/3 of max_tokens, save the rest for output


class DatasetConfig(BaseModel):
    name: str
    adapter_type: str = "local_jsonl"
    data_dir: str = ""
    task_type: str = ""
    judge_config_path: str | dict[str, str] = ""
    options: dict[str, Any] = Field(default_factory=dict)

    def get_judge_config_path(self, method: str = "default") -> str:
        """Resolve judge config path for a given eval method.

        If judge_config_path is a string, it is used for all methods.
        If it is a dict, look up by method name, falling back to "default".
        """
        if isinstance(self.judge_config_path, str):
            return self.judge_config_path
        return self.judge_config_path.get(method, self.judge_config_path.get("default", ""))


class AgentConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: str  # dotted path, e.g. "agent_probe.agents.claude_code.ClaudeCodeAgent"
    envs: dict[str, str] = Field(default_factory=dict)
    version: str = "2.1.199"  # agent install version
    mcp_host_path: str = ""  # host path to MCP config JSON file
    offline: bool = False  # install agent from local offline packages
    offline_package_dir: str = ""  # host dir mounted read-only when offline=True
    offline_mount_path: str = "/mnt/offline_package"


class JudgeConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelConfig
    agent: AgentConfig
    prompt_template: str = ""
    extract_api: ModelConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> JudgeConfig:
        raw = path.read_text(encoding="utf-8")
        expanded = _expand_env_vars(raw)
        data = yaml.safe_load(expanded)
        return cls.model_validate(data)


class SandboxConfig(BaseModel):
    host: str = "localhost:8080"
    api_key: str = ""
    request_timeout: int = 600  # seconds


class EvalExperimentConfig(BaseModel):
    name: str
    concurrency: int = 10
    output_dir: str = "./output"
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    models: dict[str, ModelConfig]
    datasets: dict[str, DatasetConfig]
    agents: dict[str, AgentConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load config from a YAML file, expanding ``${ENV_VAR}`` references."""
        raw = path.read_text(encoding="utf-8")
        expanded = _expand_env_vars(raw)
        data = yaml.safe_load(expanded)
        return cls.model_validate(data)


_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(text: str) -> str:
    """Replace ``${VAR}`` placeholders with their environment variable values."""

    def _replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable {var_name!r} referenced in config but not set")
        return value

    return _ENV_PATTERN.sub(_replacer, text)
