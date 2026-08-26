"""Tests for agent_probe.core.adapter — LocalJsonlAdapter and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_probe.core.adapter import LocalJsonlAdapter, create_adapter
from agent_probe.core.models import BaseQuestion


class SimpleQuestion(BaseQuestion):
    id: str
    text: str

    def qid(self) -> str:
        return self.id


# ---------------------------------------------------------------------------
# Data-driven test cases
# ---------------------------------------------------------------------------

_QUESTIONS_CASES: list[tuple[str, list[dict[str, Any]]]] = [
    (
        "single",
        [{"id": "q1", "text": "hello"}],
    ),
    (
        "multiple",
        [
            {"id": "q1", "text": "a"},
            {"id": "q2", "text": "b"},
            {"id": "q3", "text": "c"},
        ],
    ),
    (
        "unicode",
        [{"id": "u1", "text": "你好世界"}, {"id": "u2", "text": "émojis 🎉"}],
    ),
]


# ---------------------------------------------------------------------------
# list_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,questions", _QUESTIONS_CASES, ids=[c[0] for c in _QUESTIONS_CASES]
)
def test_list_ids(tmp_jsonl, label: str, questions: list[dict]):
    data_dir = tmp_jsonl(questions)
    adapter = LocalJsonlAdapter(data_dir=str(data_dir), question_type=SimpleQuestion)
    assert adapter.list_ids() == [q["id"] for q in questions]


# ---------------------------------------------------------------------------
# load — round-trip through each question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,questions", _QUESTIONS_CASES, ids=[c[0] for c in _QUESTIONS_CASES]
)
def test_load_each(tmp_jsonl, label: str, questions: list[dict]):
    data_dir = tmp_jsonl(questions)
    adapter = LocalJsonlAdapter(data_dir=str(data_dir), question_type=SimpleQuestion)

    for q in questions:
        loaded = adapter.load(q["id"])
        assert loaded.qid() == q["id"]
        assert loaded.text == q["text"]


# ---------------------------------------------------------------------------
# load — missing qid raises KeyError
# ---------------------------------------------------------------------------


def test_load_missing_qid(tmp_jsonl):
    data_dir = tmp_jsonl([{"id": "q1", "text": "x"}])
    adapter = LocalJsonlAdapter(data_dir=str(data_dir), question_type=SimpleQuestion)

    with pytest.raises(KeyError):
        adapter.load("nonexistent")


# ---------------------------------------------------------------------------
# create_adapter factory
# ---------------------------------------------------------------------------


def test_create_adapter_local_jsonl(tmp_jsonl):
    data_dir = tmp_jsonl([{"id": "q1", "text": "hi"}])
    adapter = create_adapter(
        "local_jsonl", data_dir=str(data_dir), question_type=SimpleQuestion
    )
    assert isinstance(adapter, LocalJsonlAdapter)
    assert adapter.list_ids() == ["q1"]


def test_create_adapter_unknown():
    with pytest.raises(ValueError, match="Unknown adapter type"):
        create_adapter("nonexistent_adapter")


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------


def test_empty_jsonl(tmp_jsonl):
    data_dir = tmp_jsonl([])
    adapter = LocalJsonlAdapter(data_dir=str(data_dir), question_type=SimpleQuestion)
    assert adapter.list_ids() == []
