"""Tests for ZBackendBench metrics and rubric parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.zbackendbench.models import (
    ZBackendBenchInference,
    ZBackendBenchJudgement,
    ZBackendBenchQuestion,
)
from benchmarks.zbackendbench.task import (
    ZBackendBenchTask,
    calculate_quality_penalty,
    parse_code_quality_rubric,
)

from agent_probe.core.models import Error
from agent_probe.core.sandbox import ExecResult


class FakeSandbox:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int | None]] = []
        self.uploads: list[tuple[Path, str]] = []

    async def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        self.uploads.append((local_dir, remote_dir))

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        self.commands.append((cmd, timeout_sec))
        if cmd.startswith("cat /logs/verifier/reward.txt"):
            return ExecResult(stdout="1", stderr="", exit_code=0)
        return ExecResult(stdout="test log", stderr="", exit_code=0)


class FakePatchSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: dict[str, str] = {}

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        del timeout_sec
        self.commands.append(cmd)
        return ExecResult(stdout="", stderr="", exit_code=0)


class FakeDumpPatchSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files = {"/tmp/agentprobe_changes.patch": "diff --git a/a b/a"}

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        del timeout_sec
        self.commands.append(cmd)
        return ExecResult(stdout="line log that should not become patch", stderr="", exit_code=0)

    async def read_file(self, path: str) -> str:
        return self.files[path]


def _rubric(*failed: str) -> dict:
    keys = {
        "A1_new_abstraction",
        "A2_dependency",
        "E1_violate_ocp",
        "E2_over_design",
        "M1_diff_minimized",
        "M2_side_effect",
    }
    return {
        key: {
            "result": "fail" if key in failed else "pass",
            "evidence": [],
            "reason": "ok",
        }
        for key in keys
    }


def _j(
    _qid: str,
    score: float,
    deterministic_score: float,
    error: Error | None = None,
) -> ZBackendBenchJudgement:
    return ZBackendBenchJudgement(
        score=score,
        deterministic_score=deterministic_score,
        error=error,
    )


def test_question_uses_zbackend_jsonl_fields() -> None:
    q = ZBackendBenchQuestion.model_validate(
        {
            "qid": "zbe_001",
            "docker_image": "image:latest",
            "workspace_dir": "/workspace",
            "contexts": ["do backend work"],
        }
    )
    assert q.qid() == "zbe_001"
    assert q.docker_image == "image:latest"
    assert q.prompt == "do backend work"


def test_parse_code_quality_rubric_from_fenced_json() -> None:
    raw = "```json\n" + json.dumps(_rubric("A1_new_abstraction")) + "\n```"
    parsed, error = parse_code_quality_rubric(raw)
    assert error == ""
    assert parsed["A1_new_abstraction"]["result"] == "fail"


def test_parse_code_quality_rubric_requires_all_dimensions() -> None:
    parsed, error = parse_code_quality_rubric('{"A1_new_abstraction": {}}')
    assert parsed == {}
    assert "Missing required dimensions" in error


def test_calculate_quality_penalty() -> None:
    penalty = calculate_quality_penalty(_rubric("A1_new_abstraction", "M2_side_effect"))
    assert penalty == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_judge_zero_skips_llm_judge() -> None:
    question = ZBackendBenchQuestion.model_validate(
        {
            "qid": "zbe_001",
            "docker_image": "image:latest",
            "workspace_dir": "/workspace",
            "contexts": ["do backend work"],
        }
    )
    inference = ZBackendBenchInference(response="done", deterministic_score=0.0)
    judgement = await ZBackendBenchTask().judge(
        question,
        inference,
        ctx=None,  # type: ignore[arg-type]
    )
    assert judgement.score == 0.0
    assert judgement.deterministic_score == 0.0
    assert "deterministic check not pass" in judgement.judge_output


def test_collect_metrics() -> None:
    scores, success_count = ZBackendBenchTask().collect_metrics(
        [
            _j("q1", 1.0, 1.0),
            _j("q2", 0.7, 1.0),
            _j("q3", 0.0, 0.0, Error(code=-1, message="judge failed")),
        ]
    )
    assert scores["num_total"] == 3
    assert scores["num_success"] == 2
    assert scores["average"] == pytest.approx(85.0)
    assert scores["deterministic_average"] == pytest.approx(100.0)
    assert success_count == 2


@pytest.mark.asyncio
async def test_run_verifier_passes_timeout_to_exec_cmd(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    sandbox = FakeSandbox()

    score, test_log_path = await ZBackendBenchTask()._run_verifier(
        sandbox,  # type: ignore[arg-type]
        tests_dir=tests_dir,
        workspace_dir="/workspace",
        verifier_timeout=123,
        output_dir=tmp_path / "out",
    )

    assert score == 1.0
    assert test_log_path is not None
    assert test_log_path.read_text(encoding="utf-8") == "test log"
    assert sandbox.uploads == [(tests_dir, "/tests")]
    assert sandbox.commands[0][1] == 123
    assert not sandbox.commands[0][0].startswith("timeout ")
    assert sandbox.commands[1][1] is None


@pytest.mark.asyncio
async def test_apply_patch_stages_added_files_for_judge(tmp_path: Path) -> None:
    patch_path = tmp_path / "changes.patch"
    patch_path.write_text("diff --git a/new.py b/new.py\n", encoding="utf-8")
    sandbox = FakePatchSandbox()

    await ZBackendBenchTask()._apply_patch(
        sandbox,  # type: ignore[arg-type]
        "/workspace",
        patch_path,
    )

    assert sandbox.files["/workspace/changes.patch"] == patch_path.read_text(
        encoding="utf-8"
    )
    assert sandbox.commands
    assert "git apply /workspace/changes.patch" in sandbox.commands[0]
    assert "git add -A" in sandbox.commands[0]


@pytest.mark.asyncio
async def test_dump_patch_reads_patch_file_instead_of_exec_stdout(tmp_path: Path) -> None:
    sandbox = FakeDumpPatchSandbox()

    patch_path = await ZBackendBenchTask()._dump_patch(
        sandbox,  # type: ignore[arg-type]
        "/workspace",
        tmp_path,
    )

    assert patch_path.read_text(encoding="utf-8") == "diff --git a/a b/a"
    assert sandbox.commands
    assert ">> /tmp/agentprobe_changes.patch" in sandbox.commands[0]
