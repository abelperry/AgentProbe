"""Tests for FileJudgeRepo and FileMetricsRepo."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field

from agent_probe.config import DatasetConfig
from agent_probe.core.models import (
    BaseInference,
    BaseJudgement,
    BaseQuestion,
    Error,
    JudgeResult,
    MetricsRecord,
)
from agent_probe.core.task import BaseTask
from agent_probe.repos.file_judge_repo import FileJudgeRepo
from agent_probe.repos.file_metrics_repo import FileMetricsRepo

# ---------------------------------------------------------------------------
# Concrete subclass types (simulate a benchmark's models)
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


class FakeTask(BaseTask[FakeQuestion, FakeInference, FakeJudgement]):
    pass


class AliasQuestion(BaseQuestion):
    task_id: str = Field(
        validation_alias=AliasChoices("qid", "task_id"),
    )
    text: str

    def qid(self) -> str:
        return self.task_id


class AliasTask(BaseTask[AliasQuestion, FakeInference, FakeJudgement]):
    pass


_FAKE_DATASETS = {
    "ds": DatasetConfig(name="ds", task_type="tests.test_file_repos.FakeTask"),
    "ds1": DatasetConfig(name="ds1", task_type="tests.test_file_repos.FakeTask"),
    "ds2": DatasetConfig(name="ds2", task_type="tests.test_file_repos.FakeTask"),
    "alias": DatasetConfig(name="alias", task_type="tests.test_file_repos.AliasTask"),
}


# ---------------------------------------------------------------------------
# FileJudgeRepo
# ---------------------------------------------------------------------------


class TestFileJudgeRepo:
    def _make_result(self, qid: str, has_error: bool = False) -> JudgeResult:
        q = FakeQuestion(id=qid, text=f"question-{qid}")
        i = FakeInference(
            answer=f"answer-{qid}",
            error=Error(code=1, message="fail") if has_error else None,
        )
        j = None if has_error else FakeJudgement(score=1.0)
        return JudgeResult(question=q, inference=i, judgement=j)

    def test_save_and_find(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        result = self._make_result("q1")
        repo.save("ds", "agent", "model", "q1", result)

        found = repo.find("ds", "agent", "model", "q1")
        assert found is not None
        assert found.question.id == "q1"
        assert found.valid_inference

    def test_find_preserves_concrete_types(self, tmp_path: Path):
        """Verify that subclass fields survive the save/load round-trip."""
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        result = self._make_result("q1")
        repo.save("ds", "agent", "model", "q1", result)

        found = repo.find("ds", "agent", "model", "q1")
        assert found is not None
        # Concrete subclass fields are preserved
        assert isinstance(found.question, FakeQuestion)
        assert found.question.text == "question-q1"
        assert isinstance(found.inference, FakeInference)
        assert found.inference.answer == "answer-q1"
        assert isinstance(found.judgement, FakeJudgement)
        assert found.judgement.score == 1.0

    def test_find_round_trips_qid_alias_questions(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        result = JudgeResult(
            question=AliasQuestion.model_validate(
                {"qid": "alias-q1", "text": "question"}
            ),
            inference=FakeInference(answer="answer"),
            judgement=FakeJudgement(score=1.0),
        )

        repo.save("alias", "agent", "model", "alias-q1", result)

        found = repo.find("alias", "agent", "model", "alias-q1")
        assert found is not None
        assert isinstance(found.question, AliasQuestion)
        assert found.question.qid() == "alias-q1"

    def test_find_not_found(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        assert repo.find("ds", "agent", "model", "q999") is None

    def test_save_overwrites(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        repo.save("ds", "agent", "model", "q1", self._make_result("q1", has_error=True))
        repo.save("ds", "agent", "model", "q1", self._make_result("q1", has_error=False))

        found = repo.find("ds", "agent", "model", "q1")
        assert found is not None
        assert found.valid_inference

    def test_find_all(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        repo.save("ds", "a", "m", "q1", self._make_result("q1"))
        repo.save("ds", "a", "m", "q2", self._make_result("q2"))
        repo.save("ds", "a", "m", "q3", self._make_result("q3"))

        results = repo.find_all("ds", "a", "m")
        assert len(results) == 3
        ids = {r.question.id for r in results}
        assert ids == {"q1", "q2", "q3"}

    def test_find_all_empty(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        assert repo.find_all("ds", "a", "m") == []

    def test_file_path_structure(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        repo.save("ds1", "claude", "zhipu", "q1", self._make_result("q1"))
        expected = tmp_path / "ds1" / "claude" / "zhipu" / "result" / "q1.json"
        assert expected.exists()

    def test_find_all_ignores_other_combinations(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)
        repo.save("ds", "a1", "m1", "q1", self._make_result("q1"))
        repo.save("ds", "a2", "m1", "q2", self._make_result("q2"))

        results = repo.find_all("ds", "a1", "m1")
        assert len(results) == 1
        assert results[0].question.id == "q1"

    def test_valid_inference_and_judgement(self, tmp_path: Path):
        repo = FileJudgeRepo(tmp_path, _FAKE_DATASETS)

        # Error case
        repo.save("ds", "a", "m", "q1", self._make_result("q1", has_error=True))
        found = repo.find("ds", "a", "m", "q1")
        assert found is not None
        assert not found.valid_inference
        assert not found.valid_judgement

        # Success case
        repo.save("ds", "a", "m", "q2", self._make_result("q2"))
        found = repo.find("ds", "a", "m", "q2")
        assert found is not None
        assert found.valid_inference
        assert found.valid_judgement


# ---------------------------------------------------------------------------
# FileMetricsRepo
# ---------------------------------------------------------------------------


class TestFileMetricsRepo:
    def test_save_and_read_file(self, tmp_path: Path):
        repo = FileMetricsRepo(tmp_path, _FAKE_DATASETS)
        record = MetricsRecord(dataset="ds", agent="a", model="m", scores={"accuracy": 0.8})
        repo.save(record)

        path = tmp_path / "ds" / "metrics.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_save_multiple_combinations(self, tmp_path: Path):
        repo = FileMetricsRepo(tmp_path, _FAKE_DATASETS)
        repo.save(MetricsRecord(dataset="ds", agent="a1", model="m1", scores={"acc": 0.7}))
        repo.save(MetricsRecord(dataset="ds", agent="a1", model="m2", scores={"acc": 0.8}))
        repo.save(MetricsRecord(dataset="ds", agent="a2", model="m1", scores={"acc": 0.9}))

        lines = (tmp_path / "ds" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_save_appends(self, tmp_path: Path):
        repo = FileMetricsRepo(tmp_path, _FAKE_DATASETS)
        repo.save(MetricsRecord(dataset="ds", agent="a", model="m", scores={"acc": 0.5}))
        repo.save(MetricsRecord(dataset="ds", agent="a", model="m", scores={"acc": 0.9}))

        lines = (tmp_path / "ds" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_different_datasets_separate_files(self, tmp_path: Path):
        repo = FileMetricsRepo(tmp_path, _FAKE_DATASETS)
        repo.save(MetricsRecord(dataset="ds1", agent="a", model="m", scores={"acc": 0.7}))
        repo.save(MetricsRecord(dataset="ds2", agent="a", model="m", scores={"acc": 0.8}))

        assert (tmp_path / "ds1" / "metrics.jsonl").exists()
        assert (tmp_path / "ds2" / "metrics.jsonl").exists()
