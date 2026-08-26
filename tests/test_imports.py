"""Tests for agent_probe.utils.imports — dynamic class loading."""

from __future__ import annotations

import pytest

from agent_probe.utils.imports import import_class


# ---------------------------------------------------------------------------
# Parametrized: valid imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dotted_path,expected_name",
    [
        ("agent_probe.core.models.BaseQuestion", "BaseQuestion"),
        ("agent_probe.core.models.BaseInference", "BaseInference"),
        ("agent_probe.core.adapter.LocalJsonlAdapter", "LocalJsonlAdapter"),
        ("agent_probe.config.EvalExperimentConfig", "EvalExperimentConfig"),
        ("pathlib.Path", "Path"),
    ],
    ids=["BaseQuestion", "BaseInference", "LocalJsonlAdapter", "Config", "stdlib-Path"],
)
def test_import_valid(dotted_path: str, expected_name: str):
    cls = import_class(dotted_path)
    assert cls.__name__ == expected_name


# ---------------------------------------------------------------------------
# Parametrized: invalid imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dotted_path,error_match",
    [
        ("NoModule", "Invalid dotted path"),
        ("agent_probe.core.models.NonExistentClass", "not found in module"),
        ("totally.fake.module.Cls", "No module named"),
    ],
    ids=["no-dot", "bad-class", "bad-module"],
)
def test_import_invalid(dotted_path: str, error_match: str):
    with pytest.raises(ImportError, match=error_match):
        import_class(dotted_path)
