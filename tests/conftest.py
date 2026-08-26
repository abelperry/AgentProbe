"""Shared fixtures for AgentProbe tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_probe.config import AgentConfig, EvalExperimentConfig, ModelConfig, DatasetConfig


@pytest.fixture
def tmp_jsonl(tmp_path: Path):
    """Factory fixture: write a list of question dicts to a temp questions.jsonl."""

    def _write(questions: list[dict[str, Any]]) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        jsonl = data_dir / "questions.jsonl"
        with open(jsonl, "w") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        return data_dir

    return _write


@pytest.fixture
def sample_config(tmp_path: Path) -> EvalExperimentConfig:
    """A minimal in-memory EvalExperimentConfig for testing."""
    return EvalExperimentConfig(
        name="test-exp",
        concurrency=2,
        output_dir=str(tmp_path / "output"),
        models={
            "model_a": ModelConfig(base_url="http://a", api_key="key_a"),
            "model_b": ModelConfig(base_url="http://b", api_key="key_b"),
        },
        datasets={
            "bench1": DatasetConfig(
                name="bench1",
                adapter_type="local_jsonl",
                data_dir="",
                task_type="",
            ),
        },
        agents={
            "agent_x": AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent"),
            "agent_y": AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent"),
        },
    )
