"""Tests for TerminalBench v2 metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.terminalbench_v2.models import TerminalBenchV2Judgement, TerminalBenchV2Question
from benchmarks.terminalbench_v2.task import TerminalBenchV2Task

from agent_probe.core.models import Error
from agent_probe.core.sandbox import ExecResult


class FakeVerifierSandbox:
    def __init__(self, test_log: str, ctrf: str = "") -> None:
        self.test_log = test_log
        self.ctrf = ctrf
        self.commands: list[tuple[str, int | None]] = []
        self.uploads: list[tuple[Path, str]] = []

    async def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        self.uploads.append((local_dir, remote_dir))

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        self.commands.append((cmd, timeout_sec))
        if cmd.startswith("cat /logs/verifier/ctrf.json"):
            return ExecResult(stdout=self.ctrf, stderr="", exit_code=0)
        if cmd.startswith("cat /logs/verifier/reward.txt"):
            return ExecResult(stdout="1", stderr="", exit_code=0)
        return ExecResult(stdout=self.test_log, stderr="", exit_code=0)


def _j(qid: str, score: float, error: Error | None = None) -> TerminalBenchV2Judgement:
    del qid
    return TerminalBenchV2Judgement(score=score, error=error)


def test_question_uses_terminalbench_jsonl_fields() -> None:
    q = TerminalBenchV2Question.model_validate(
        {
            "qid": "tb_001",
            "docker_image": "image:latest",
            "workspace_dir": "/workspace",
            "contexts": ["do it"],
        }
    )
    assert q.qid() == "tb_001"
    assert q.docker_image == "image:latest"
    assert q.prompt == "do it"


def test_agent_prompt_adds_anti_stall_prefix() -> None:
    q = TerminalBenchV2Question.model_validate(
        {
            "qid": "tb_001",
            "docker_image": "image:latest",
            "workspace_dir": "/workspace",
            "contexts": ["do it"],
        }
    )
    prompt = TerminalBenchV2Task()._agent_prompt(q)

    assert prompt.startswith("Work autonomously.")
    assert prompt.endswith("do it")


def test_collect_metrics_empty() -> None:
    scores, success_count = TerminalBenchV2Task().collect_metrics([])
    assert scores == {"num_total": 0, "num_success": 0, "average": 0.0}
    assert success_count == 0


def test_collect_metrics_uses_total_denominator_and_success_count() -> None:
    scores, success_count = TerminalBenchV2Task().collect_metrics(
        [
            _j("q1", 1.0),
            _j("q2", 0.0),
            _j("q3", 1.0, Error(code=-1, message="boom")),
        ]
    )
    assert scores["num_total"] == 3
    assert scores["num_success"] == 2
    assert scores["average"] == pytest.approx(1 / 3 * 100)
    assert success_count == 2


@pytest.mark.asyncio
async def test_run_verifier_collects_ctrf_status(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    ctrf = json.dumps(
        {
            "results": {
                "tests": [
                    {"name": "tests/test_a.py::test_ok", "status": "passed"},
                    {"name": "tests/test_b.py::test_bad", "status": "failed"},
                ]
            }
        }
    )
    sandbox = FakeVerifierSandbox("ignored log", ctrf)

    score, test_log_path, tests_status = await TerminalBenchV2Task()._run_verifier(
        sandbox,  # type: ignore[arg-type]
        tests_dir=tests_dir,
        workspace_dir="/workspace",
        verifier_timeout=123,
        output_dir=tmp_path / "out",
    )

    assert score == 1.0
    assert test_log_path is not None
    assert test_log_path.read_text(encoding="utf-8") == "ignored log"
    assert tests_status == {
        "passed": ["tests/test_a.py::test_ok"],
        "failed": ["tests/test_b.py::test_bad"],
    }
    assert sandbox.uploads == [(tests_dir, "/tests")]
    assert sandbox.commands[0][1] == 123


@pytest.mark.asyncio
async def test_run_verifier_falls_back_to_log_status(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    sandbox = FakeVerifierSandbox(
        "\x1b[32mPASSED tests/test_a.py::test_ok\x1b[0m\n"
        "FAILED tests/test_b.py::test_bad - AssertionError\n"
        "SKIPPED [3] tests/test_c.py:42: reason\n"
    )

    _, _, tests_status = await TerminalBenchV2Task()._run_verifier(
        sandbox,  # type: ignore[arg-type]
        tests_dir=tests_dir,
        workspace_dir="/workspace",
        verifier_timeout=123,
        output_dir=tmp_path / "out",
    )

    assert tests_status == {
        "passed": ["tests/test_a.py::test_ok"],
        "failed": ["tests/test_b.py::test_bad"],
        "skipped": ["tests/test_c.py:42"],
    }
