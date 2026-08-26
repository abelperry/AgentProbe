"""Tests for PipelineExecutor — unit expansion and member methods."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_probe.config import AgentConfig, DatasetConfig, EvalExperimentConfig, ModelConfig
from agent_probe.core.executor import EvalUnit
from agent_probe.core.models import BaseInference, BaseJudgement, BaseQuestion, JudgeResult
from agent_probe.core.adapter import BaseAdapter
from agent_probe.core.repo import JudgeRepo, MetricsRepo
from agent_probe.executors.pipeline_executor import PipelineExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeQuestion(BaseQuestion):
    id: str
    text: str

    def qid(self) -> str:
        return self.id


class FakeInference(BaseInference):
    answer: str


class FakeJudgement(BaseJudgement):
    score: float


class FakeAdapter(BaseAdapter[FakeQuestion]):
    def __init__(self, qids: list[str]):
        self._qids = qids
        self._data = {qid: FakeQuestion(id=qid, text=f"q-{qid}") for qid in qids}

    def list_ids(self) -> list[str]:
        return self._qids

    def load(self, qid: str) -> FakeQuestion:
        return self._data[qid]


def _make_executor(
    tmp_path: Path,
    qids: list[str],
    agent_names: list[str],
    model_names: list[str],
    dataset_names: list[str] | None = None,
) -> PipelineExecutor:
    dataset_names = dataset_names or ["ds1"]
    config = EvalExperimentConfig(
        name="test",
        concurrency=2,
        output_dir=str(tmp_path / "output"),
        models={m: ModelConfig(base_url="http://x", api_key="k") for m in model_names},
        datasets={d: DatasetConfig(name=d) for d in dataset_names},
        agents={a: AgentConfig(type="x.Agent") for a in agent_names},
    )
    adapters = {d: FakeAdapter(qids) for d in dataset_names}
    tasks = {d: MagicMock() for d in dataset_names}
    judge_repo = MagicMock(spec=JudgeRepo)
    metrics_repo = MagicMock(spec=MetricsRepo)

    return PipelineExecutor(
        config=config,
        adapters=adapters,
        tasks=tasks,
        judge_repo=judge_repo,
        metrics_repo=metrics_repo,
    )


# ---------------------------------------------------------------------------
# _build_units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,qids,agents,models,datasets,expected_count",
    [
        ("1x1x1x1", ["q1"], ["a1"], ["m1"], ["ds1"], 1),
        ("2x1x1x1", ["q1", "q2"], ["a1"], ["m1"], ["ds1"], 2),
        ("2x2x2x1", ["q1", "q2"], ["a1", "a2"], ["m1", "m2"], ["ds1"], 8),
        ("1x1x1x2", ["q1"], ["a1"], ["m1"], ["ds1", "ds2"], 2),
        ("3x2x2x2", ["q1", "q2", "q3"], ["a1", "a2"], ["m1", "m2"], ["ds1", "ds2"], 24),
    ],
    ids=["1x1x1x1", "2x1x1x1", "2x2x2x1", "1x1x1x2", "3x2x2x2"],
)
def test_build_units_count(
    tmp_path: Path, label: str, qids: list[str],
    agents: list[str], models: list[str], datasets: list[str],
    expected_count: int,
):
    executor = _make_executor(tmp_path, qids, agents, models, datasets)
    assert len(executor._units) == expected_count


def test_build_units_content(tmp_path: Path):
    executor = _make_executor(tmp_path, ["q1", "q2"], ["a1"], ["m1", "m2"], ["ds1"])
    combos = {(u.qid, u.agent_name, u.model_name, u.dataset_name) for u in executor._units}
    assert combos == {
        ("q1", "a1", "m1", "ds1"),
        ("q1", "a1", "m2", "ds1"),
        ("q2", "a1", "m1", "ds1"),
        ("q2", "a1", "m2", "ds1"),
    }


# ---------------------------------------------------------------------------
# JudgeResult.valid_inference / valid_judgement
# ---------------------------------------------------------------------------

def test_judge_result_valid_inference():
    q = FakeQuestion(id="q1", text="hello")
    i = FakeInference(answer="world")
    r = JudgeResult(question=q, inference=i)
    assert r.valid_inference is True
    assert r.valid_judgement is False  # judgement is None


def test_judge_result_valid_judgement():
    from agent_probe.core.models import Error

    q = FakeQuestion(id="q1", text="hello")
    i = FakeInference(answer="world")
    j = FakeJudgement(score=1.0)
    r = JudgeResult(question=q, inference=i, judgement=j)
    assert r.valid_inference is True
    assert r.valid_judgement is True

    # With error
    j_err = FakeJudgement(score=0.0, error=Error(code=1, message="fail"))
    r2 = JudgeResult(question=q, inference=i, judgement=j_err)
    assert r2.valid_judgement is False
