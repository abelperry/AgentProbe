"""Tests for SWE-bench Q/I/J models."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.swebench.models import (
    SWEBenchInference,
    SWEBenchJudgement,
    SWEBenchQuestion,
)
from benchmarks.swebench.spec_builder import SWEBenchInferSpecBuilder
from benchmarks.swebench.task import SWEBenchTask

from agent_probe.core.models import resolve_types
from agent_probe.core.sandbox import SandboxResult


def _record() -> dict:
    return {
        "repo": "django/django",
        "instance_id": "django__django-11179",
        "base_commit": "abc123",
        "patch": "diff --git a/a.py b/a.py\n",
        "test_patch": "diff --git a/tests.py b/tests.py\n",
        "problem_statement": "fix bug",
        "hints_text": "",
        "created_at": "",
        "version": "3.0",
        "FAIL_TO_PASS": ["test_new"],
        "PASS_TO_PASS": ["test_old"],
        "environment_setup_commit": "def456",
    }


def test_question_accepts_original_swebench_shape() -> None:
    question = SWEBenchQuestion.model_validate(_record())
    assert question.qid() == "django__django-11179"
    assert question.FAIL_TO_PASS == ["test_new"]
    assert question.PASS_TO_PASS == ["test_old"]

    instance = question.to_swebench_instance()
    assert instance["instance_id"] == question.instance_id
    assert instance["FAIL_TO_PASS"] == ["test_new"]
    assert instance["PASS_TO_PASS"] == ["test_old"]
    assert instance["problem_statement"] == "fix bug"


def test_resolve_types_swebench() -> None:
    q_cls, i_cls, j_cls = resolve_types(SWEBenchTask)
    assert q_cls is SWEBenchQuestion
    assert i_cls is SWEBenchInference
    assert j_cls is SWEBenchJudgement


def test_swebench_agent_prompt_adds_anti_stall_prefix(tmp_path) -> None:
    builder = SWEBenchInferSpecBuilder(
        SWEBenchQuestion.model_validate(_record()),
        _ctx(tmp_path),
    )

    prompt = builder._agent_prompt()

    assert prompt.startswith("Work autonomously.")
    assert prompt.endswith("fix bug")


def test_swebench_empty_patch_marks_inference_error(tmp_path) -> None:
    builder = SWEBenchInferSpecBuilder(
        SWEBenchQuestion.model_validate(_record()),
        _ctx(tmp_path),
    )
    builder.infer_dir.mkdir(parents=True)
    builder.patch_path.write_text("", encoding="utf-8")

    inference = builder.to_inference(SandboxResult())

    assert inference.error is not None
    assert inference.error.message == "empty prediction patch"


def _ctx(tmp_path):
    return SimpleNamespace(
        dataset_config=SimpleNamespace(options={}),
        output_dir=tmp_path / "output",
    )
