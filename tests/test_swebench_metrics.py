"""Tests for SWE-bench metrics."""

from __future__ import annotations

import pytest
from benchmarks.swebench.models import SWEBenchJudgement
from benchmarks.swebench.task import SWEBenchTask

from agent_probe.core.models import Error


def test_collect_metrics_empty() -> None:
    scores, success_count = SWEBenchTask().collect_metrics([])
    assert scores == {"avg_score": 0.0}
    assert success_count == 0


def test_collect_metrics_mixed() -> None:
    judgements = [
        SWEBenchJudgement(score=1.0),
        SWEBenchJudgement(score=0.0),
        SWEBenchJudgement(score=1.0),
    ]
    scores, success_count = SWEBenchTask().collect_metrics(judgements)
    assert scores == {"avg_score": pytest.approx(2 / 3 * 100)}
    assert success_count == 3


def test_collect_metrics_error_counts_as_not_successful() -> None:
    judgements = [
        SWEBenchJudgement(score=1.0),
        SWEBenchJudgement(score=0.0, error=Error(code=-1, message="boom")),
    ]
    scores, success_count = SWEBenchTask().collect_metrics(judgements)
    assert scores == {"avg_score": pytest.approx(1 / 2 * 100)}
    assert success_count == 1
