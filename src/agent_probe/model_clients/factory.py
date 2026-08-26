"""Factory for model API clients."""

from __future__ import annotations

from agent_probe.config import ModelConfig
from agent_probe.model_clients.base import BaseModelClient
from agent_probe.model_clients.http import HttpModelClient


def create_model_client(model_config: ModelConfig, *, backend: str = "http") -> BaseModelClient:
    """Create a model client implementation.

    ``backend`` is intentionally explicit so callers can later switch to SDK-backed
    implementations without changing benchmark code.
    """

    if backend == "http":
        return HttpModelClient(model_config)
    raise ValueError(f"Unsupported model client backend: {backend}")
