"""Tests for agent_probe.config — YAML parsing, env-var expansion, validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_probe.config import EvalExperimentConfig


# ---------------------------------------------------------------------------
# Parametrized: env-var expansion
# ---------------------------------------------------------------------------

_YAML_WITH_ENV = """\
name: exp-1
models:
  m1:
    base_url: "http://host"
    api_key: "${MY_KEY}"
datasets:
  ds1:
    name: ds1
agents:
  a1:
    type: "some.Agent"
"""


@pytest.mark.parametrize(
    "env_val, expected_key",
    [
        ("secret-123", "secret-123"),
        ("", ""),
        ("with spaces", "with spaces"),
    ],
    ids=["normal", "empty", "spaces"],
)
def test_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_val: str, expected_key: str):
    monkeypatch.setenv("MY_KEY", env_val)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_YAML_WITH_ENV)

    cfg = EvalExperimentConfig.from_yaml(cfg_file)
    assert cfg.models["m1"].api_key == expected_key


def test_sandbox_api_key_expands_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_KEY", "sandbox-secret")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """\
name: exp
sandbox:
  host: localhost:8080
  api_key: "${SANDBOX_KEY}"
models:
  m:
    base_url: "http://host"
    api_key: "model-key"
datasets:
  d:
    name: d
agents:
  a:
    type: "some.Agent"
"""
    )

    cfg = EvalExperimentConfig.from_yaml(cfg_file)

    assert cfg.sandbox.api_key == "sandbox-secret"


def test_env_var_missing_raises(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_YAML_WITH_ENV)

    with pytest.raises(ValueError, match="MY_KEY"):
        EvalExperimentConfig.from_yaml(cfg_file)


# ---------------------------------------------------------------------------
# Parametrized: full YAML → config round-trip
# ---------------------------------------------------------------------------

_AGENT_A = {"type": "x.Agent"}

_MINIMAL_CASES: list[tuple[str, dict]] = [
    (
        "single_model_single_agent",
        {
            "name": "exp",
            "models": {"m": {"base_url": "http://a", "api_key": "k"}},
            "datasets": {"d": {"name": "d"}},
            "agents": {"a": _AGENT_A},
        },
    ),
    (
        "multiple_models_and_agents",
        {
            "name": "multi",
            "concurrency": 5,
            "output_dir": "/tmp/out",
            "models": {
                "m1": {"base_url": "http://a", "api_key": "k1"},
                "m2": {"base_url": "http://b", "api_key": "k2", "timeout": 60},
            },
            "datasets": {
                "ds1": {"name": "ds1", "adapter_type": "local_jsonl"},
                "ds2": {"name": "ds2"},
            },
            "agents": {"a1": _AGENT_A, "a2": _AGENT_A},
        },
    ),
]


@pytest.mark.parametrize("label,data", _MINIMAL_CASES, ids=[c[0] for c in _MINIMAL_CASES])
def test_yaml_round_trip(tmp_path: Path, label: str, data: dict):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(data))

    cfg = EvalExperimentConfig.from_yaml(cfg_file)
    assert cfg.name == data["name"]
    assert set(cfg.models) == set(data["models"])
    assert set(cfg.datasets) == set(data["datasets"])
    assert set(cfg.agents) == set(data["agents"])


# ---------------------------------------------------------------------------
# Parametrized: defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field, default_val",
    [
        ("concurrency", 10),
        ("output_dir", "./output"),
    ],
)
def test_defaults(tmp_path: Path, field: str, default_val):
    data = {
        "name": "e",
        "models": {"m": {"base_url": "u", "api_key": "k"}},
        "datasets": {"d": {"name": "d"}},
        "agents": {"a": {"type": "x.A"}},
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(data))

    cfg = EvalExperimentConfig.from_yaml(cfg_file)
    assert getattr(cfg, field) == default_val


# ---------------------------------------------------------------------------
# Validation: required fields missing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "missing_field",
    ["name", "models", "datasets", "agents"],
)
def test_missing_required_field(tmp_path: Path, missing_field: str):
    data = {
        "name": "e",
        "models": {"m": {"base_url": "u", "api_key": "k"}},
        "datasets": {"d": {"name": "d"}},
        "agents": {"a": {"type": "x.A"}},
    }
    del data[missing_field]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(data))

    with pytest.raises(Exception):
        EvalExperimentConfig.from_yaml(cfg_file)
