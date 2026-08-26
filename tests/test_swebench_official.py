"""Tests for SWE-bench official harness helpers."""

from __future__ import annotations

import importlib.util

import pytest
from benchmarks.swebench.official import patched_instance_image_mode


@pytest.mark.skipif(importlib.util.find_spec("swebench") is None, reason="swebench not installed")
def test_patched_instance_image_mode_restores_functions() -> None:
    import swebench.harness.test_spec.create_scripts as create_scripts
    import swebench.harness.test_spec.test_spec as test_spec_mod

    old_create_env = create_scripts.make_env_script_list
    old_create_repo = create_scripts.make_repo_script_list
    old_spec_env = test_spec_mod.make_env_script_list
    old_spec_repo = test_spec_mod.make_repo_script_list

    with patched_instance_image_mode():
        assert create_scripts.make_env_script_list("x") == []
        assert create_scripts.make_repo_script_list("x") == []
        assert test_spec_mod.make_env_script_list("x") == []
        assert test_spec_mod.make_repo_script_list("x") == []

    assert create_scripts.make_env_script_list is old_create_env
    assert create_scripts.make_repo_script_list is old_create_repo
    assert test_spec_mod.make_env_script_list is old_spec_env
    assert test_spec_mod.make_repo_script_list is old_spec_repo
