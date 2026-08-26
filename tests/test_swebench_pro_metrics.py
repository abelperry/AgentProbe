"""Tests for SWE-bench Pro grading and metrics."""

from __future__ import annotations

import json

import pytest
from benchmarks.swebench_pro.models import SWEBenchProJudgement, SWEBenchProQuestion
from benchmarks.swebench_pro.task import SWEBenchProTask

from agent_probe.core.models import Error


def _question() -> SWEBenchProQuestion:
    return SWEBenchProQuestion.model_validate({
        "instance_id": "instance_repo-abc-vnan",
        "repo": "org/repo",
        "base_commit": "base",
        "dockerhub_tag": "org.repo-instance",
        "prompt": "fix",
        "fail_to_pass": ["new_test"],
        "pass_to_pass": ["old_test"],
        "selected_test_files": ["test.py"],
        "eval_cmd": "git checkout gold -- test.py",
        "run_script": "#!/bin/bash",
        "parser_py": "print(1)",
    })


def test_grade_swebench_pro_resolved(tmp_path) -> None:
    output = tmp_path / "output.json"
    output.write_text(
        json.dumps({
            "tests": [
                {"name": "new_test", "status": "PASSED"},
                {"name": "old_test", "status": "PASSED"},
            ]
        }),
        encoding="utf-8",
    )

    resolved, status = SWEBenchProTask._grade(_question(), str(output))

    assert resolved is True
    assert status["f2p_missing_count"] == 0
    assert status["p2p_missing_count"] == 0


def test_grade_swebench_pro_missing_tests(tmp_path) -> None:
    output = tmp_path / "output.json"
    output.write_text(
        json.dumps({"tests": [{"name": "old_test", "status": "PASSED"}]}),
        encoding="utf-8",
    )

    resolved, status = SWEBenchProTask._grade(_question(), str(output))

    assert resolved is False
    assert status["f2p_missing"] == ["new_test"]


def test_grade_swebench_pro_missing_output_json() -> None:
    resolved, status = SWEBenchProTask._grade(_question(), "")

    assert resolved is False
    assert "grading_error" in status


def test_collect_metrics_counts_error_judgements_in_denominator() -> None:
    judgements = [
        SWEBenchProJudgement(score=1.0),
        SWEBenchProJudgement(score=0.0, error=Error(code=-2, message="empty")),
        SWEBenchProJudgement(score=0.0),
    ]

    scores, success_count = SWEBenchProTask().collect_metrics(judgements)

    assert scores == {"avg_score": pytest.approx(1 / 3 * 100)}
    assert success_count == 2
