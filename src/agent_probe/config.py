"""Experiment configuration models."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    base_url: str = ""
    api_key: str = ""
    api_name: str = ""
    auth_file: str = ""
    timeout: int = 10800
    max_tokens: int = 32000
    model_name: str = ""  # filled by factory from YAML key if empty
    format: str = "openai"  # "openai" | "anthropic"
    extra_body: dict[str, Any] = Field(default_factory=dict)
    thinking: str = "off"  # "high" | "medium" | "low" | "off"
    max_thinking_tokens: int = 10000  # make it 1/3 of max_tokens, save the rest for output

    _AUTH_KEY_MAP: ClassVar[dict[str, str]] = {
        "openai": "openai",
        "loli": "gateway",
        "gateway": "gateway",
        "oneapi": "oneapi",
        "ipo": "api.zhipuai-infra.cn",
        "zhipu": "zhipuai",
        "zhipuai": "zhipuai",
        "qingyan": "qingyan",
    }
    _URL_BASED_TYPES: ClassVar[set[str]] = {"tp", "tgi"}

    @classmethod
    def from_mapping(cls, value: Any, *, config_path: Path) -> ModelConfig:
        """Resolve WBS-style ``api_name``/``auth_file`` model entries."""
        if not isinstance(value, dict):
            return cls.model_validate(value)
        data = dict(value)
        api_name = str(data.get("api_name") or "").strip()
        if not api_name:
            return cls.model_validate(data)
        if ":" not in api_name:
            raise ValueError(f"invalid api_name {api_name!r}; expected 'api_type:model'")

        api_type, resolved_model_name = api_name.split(":", 1)
        resolved_model_name = resolved_model_name.split("@", 1)[0]
        data.setdefault("model_name", resolved_model_name)
        # An explicit pair needs no credentials file, but the url-based types
        # still have to be normalized below.
        if (
            api_type not in cls._URL_BASED_TYPES
            and str(data.get("base_url") or "").strip()
            and str(data.get("api_key") or "").strip()
        ):
            return cls.model_validate(data)

        if api_type in cls._URL_BASED_TYPES:
            raw_url = str(data.get("base_url") or data.get("url") or "").strip()
            if not raw_url:
                raise ValueError(f"api_type {api_type!r} requires base_url or url")
            if api_type == "tp":
                resolved_base_url = raw_url.rstrip("/").rsplit("/v1/chat/completions", 1)[0]
            else:
                parsed_url = urlparse(raw_url)
                resolved_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            resolved_api_key = str(data.get("api_key") or "no-key")
            # Assigned directly: raw_url usually comes *from* data["base_url"],
            # so the `or` below would short-circuit on the un-normalized value.
            data["base_url"] = resolved_base_url
            data["api_key"] = str(data.get("api_key") or resolved_api_key)
            return cls.model_validate(data)
        else:
            auth_path = _resolve_auth_file(data.get("auth_file"), config_path)
            auth_dict = _load_auth_dict(auth_path)
            auth_key = (
                resolved_model_name if api_type == "custom" else cls._AUTH_KEY_MAP.get(api_type, "")
            )
            if not auth_key or auth_key not in auth_dict:
                raise ValueError(
                    f"auth entry {auth_key!r} for api_name {api_name!r} "
                    f"not found in {auth_path}"
                )
            auth_entry = auth_dict[auth_key]
            if api_type in {"custom", "ipo"}:
                resolved_base_url = (
                    str(auth_entry["url"]).rstrip("/").rsplit("/chat/completions", 1)[0]
                )
            else:
                resolved_base_url = str(auth_entry["base_url"])
            api_keys = auth_entry.get("api_keys") or []
            if not api_keys or not str(api_keys[0]):
                raise ValueError(f"auth entry {auth_key!r} in {auth_path} has no API key")
            resolved_api_key = str(api_keys[0])
            data["auth_file"] = str(auth_path)

        data["base_url"] = str(data.get("base_url") or resolved_base_url)
        data["api_key"] = str(data.get("api_key") or resolved_api_key)
        return cls.model_validate(data)


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
    params: dict[str, Any] = Field(default_factory=dict)
    version: str = "2.1.199"  # agent install version
    mcp_host_path: str = ""  # host path to MCP config JSON file
    offline: bool = False  # install agent from local offline packages
    offline_package_dir: str = ""  # host dir mounted read-only when offline=True
    offline_mount_path: str = "/mnt/offline_package"
    offline_node_version: str = "22.21.1"


class JudgeConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelConfig
    agent: AgentConfig
    docker: str = ""
    prompt_template: str = ""
    extract_api: ModelConfig | None = None
    function_checklist_eval_enabled: bool = False
    function_model: ModelConfig | None = None
    function_agent: AgentConfig | None = None

    def function_runtime(self) -> tuple[ModelConfig, AgentConfig]:
        """Return the dedicated functional judge, falling back to the main judge."""
        return self.function_model or self.model, self.function_agent or self.agent

    @classmethod
    def from_yaml(cls, path: Path) -> JudgeConfig:
        raw = path.read_text(encoding="utf-8")
        expanded = _expand_env_vars(raw)
        data = yaml.safe_load(expanded)
        data["model"] = ModelConfig.from_mapping(data["model"], config_path=path)
        data["agent"] = _resolve_agent_paths(data["agent"], config_path=path)
        if data.get("extract_api") is not None:
            data["extract_api"] = ModelConfig.from_mapping(data["extract_api"], config_path=path)
        if data.get("function_model") is not None:
            data["function_model"] = ModelConfig.from_mapping(
                data["function_model"], config_path=path
            )
        if data.get("function_agent") is not None:
            data["function_agent"] = _resolve_agent_paths(data["function_agent"], config_path=path)
        return cls.model_validate(data)


class SandboxConfig(BaseModel):
    host: str = "localhost:8080"
    api_key: str = ""
    request_timeout: int = 600  # seconds
    use_server_proxy: bool = False


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
        data["models"] = {
            name: ModelConfig.from_mapping(model, config_path=path)
            for name, model in data["models"].items()
        }
        return cls.model_validate(data)


_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_auth_file(value: Any, config_path: Path) -> Path:
    configured = str(value or os.environ.get("GLM_API_AUTH_FILE") or "").strip()
    if not configured:
        raise ValueError(
            "api_name requires auth_file or the GLM_API_AUTH_FILE environment variable"
        )
    auth_path = Path(configured).expanduser()
    if not auth_path.is_absolute():
        auth_path = config_path.parent / auth_path
    auth_path = auth_path.resolve()
    if not auth_path.is_file():
        raise FileNotFoundError(f"API authorization file not found: {auth_path}")
    return auth_path


def _load_auth_dict(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"API authorization file must contain an object: {path}")
    return data


def _resolve_agent_paths(value: Any, *, config_path: Path) -> Any:
    if not isinstance(value, dict):
        return value
    data = dict(value)
    raw_mcp_path = str(data.get("mcp_host_path") or "").strip()
    if not raw_mcp_path:
        return data
    mcp_path = Path(raw_mcp_path).expanduser()
    if not mcp_path.is_absolute():
        cwd_candidate = (Path.cwd() / mcp_path).resolve()
        config_candidate = (config_path.parent / mcp_path).resolve()
        mcp_path = cwd_candidate if cwd_candidate.is_file() else config_candidate
    data["mcp_host_path"] = str(mcp_path)
    return data


def _expand_env_vars(text: str) -> str:
    """Replace ``${VAR}`` placeholders with their environment variable values."""

    def _replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable {var_name!r} referenced in config but not set")
        return value

    return _ENV_PATTERN.sub(_replacer, text)
