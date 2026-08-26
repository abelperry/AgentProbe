"""Tests for ZFrontendBench parsing, scoring, and metrics."""

from __future__ import annotations

import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.zfrontendbench.frontend import (
    BuildResult,
    ProjectType,
    calculate_weighted_score,
    parse_judge_score,
)
from benchmarks.zfrontendbench.models import (
    FrontendCheckResult,
    ZFrontendBenchInference,
    ZFrontendBenchJudgement,
    ZFrontendBenchQuestion,
)
from benchmarks.zfrontendbench.task import ZFrontendBenchTask

from agent_probe.config import ModelConfig, SandboxConfig
from agent_probe.core.models import Error
from agent_probe.core.sandbox import ExecResult, SandboxResult


def _check(
    check_id: int,
    score: float | None,
    weight: float = 1.0,
    reason: str = "",
) -> FrontendCheckResult:
    return FrontendCheckResult(
        id=check_id,
        description=f"c{check_id}",
        score=score,
        weight=weight,
        reason=reason,
    )


def _j(
    _qid: str,
    category: str,
    weighted_score: float | None,
    checks: list[FrontendCheckResult],
    error: Error | None = None,
) -> ZFrontendBenchJudgement:
    return ZFrontendBenchJudgement(
        category=category,
        weighted_score=weighted_score,
        check_results=checks,
        error=error,
    )


def test_question_normalizes_checklist_and_metadata() -> None:
    q = ZFrontendBenchQuestion.model_validate(
        {
            "qid": "zfe_001",
            "docker": "image:latest",
            "workspace_dir": "/workspace",
            "contexts": ["build ui"],
            "checklist": [
                {"id": 10, "description": "renders", "weight": 2},
                "works on click",
            ],
            "categories": ["frontend"],
        }
    )
    checklist = q.get_checklist()
    assert q.qid() == "zfe_001"
    assert q.task_description == "build ui"
    assert q.category == "frontend"
    assert checklist[0].id == 10
    assert checklist[0].weight == 2
    assert checklist[1].description == "works on click"


@pytest.mark.parametrize(
    "output,expected",
    [
        ("判断结论：该项目符合要求", 1.0),
        ("判断结论：该项目不符合要求", 0.0),
        ("没有明确结论", None),
    ],
)
def test_parse_judge_score(output: str, expected: float | None) -> None:
    score, _ = parse_judge_score(output)
    assert score == expected


def test_calculate_weighted_score() -> None:
    assert calculate_weighted_score([_check(1, 1.0, 2), _check(2, 0.0, 1)]) == pytest.approx(2 / 3)
    assert calculate_weighted_score([_check(1, None)]) is None


def test_collect_metrics_with_category_breakdown_and_build_success() -> None:
    scores, success_count = ZFrontendBenchTask().collect_metrics(
        [
            _j("q1", "ui", 1.0, [_check(1, 1.0), _check(2, 1.0)]),
            _j("q2", "ui", 0.5, [_check(1, 1.0), _check(2, 0.0, reason="Build failed: syntax")]),
            _j("q3", "svg", 1.0, [_check(1, 1.0)]),
            _j("q4", "ui", None, [_check(1, None)], Error(code=-1, message="retry")),
        ]
    )

    assert success_count == 3
    assert scores["num_total"] == 4
    assert scores["num_success"] == 3
    assert scores["average"] == pytest.approx((1.0 + 0.5 + 1.0) / 3 * 100)
    assert scores["ISR"] == pytest.approx(2 / 3 * 100)
    assert scores["CSR"] == pytest.approx(4 / 5 * 100)
    assert scores["BSR"] == pytest.approx(2 / 3 * 100)
    assert scores["ISR_ui"] == pytest.approx(1 / 2 * 100)
    assert scores["CSR_ui"] == pytest.approx(3 / 4 * 100)
    assert scores["BSR_ui"] == pytest.approx(1 / 2 * 100)
    assert scores["ISR_svg"] == pytest.approx(100.0)
    assert scores["CSR_svg"] == pytest.approx(100.0)
    assert scores["BSR_svg"] == pytest.approx(100.0)


class FakeCommandSandbox:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int | None]] = []
        self.uploads: list[tuple[Path, str]] = []

    async def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        self.uploads.append((local_dir, remote_dir))

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        self.commands.append((cmd, timeout_sec))
        return ExecResult(stdout="", stderr="", exit_code=1)


class FakeSandboxRunner:
    command_sandbox: FakeCommandSandbox | None = None

    def __init__(self, spec) -> None:
        self.spec = spec

    async def run(self) -> SandboxResult:
        sandbox = FakeCommandSandbox()
        type(self).command_sandbox = sandbox
        if self.spec.on_setup:
            await self.spec.on_setup(sandbox)
        if self.spec.on_complete:
            await self.spec.on_complete(sandbox, SandboxResult())
        return SandboxResult()


class FakeInferenceSandbox:
    command_sandbox: FakeCommandSandbox | None = None

    def __init__(self, spec) -> None:
        self.spec = spec

    async def run(self) -> SandboxResult:
        sandbox = FakeCommandSandbox()
        type(self).command_sandbox = sandbox
        if self.spec.on_setup:
            await self.spec.on_setup(sandbox)
        return SandboxResult(
            error=Error(code=-1, message="stop after setup"),
        )


class FakeServerReadySandbox:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int | None]] = []

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        self.commands.append((cmd, timeout_sec))
        if cmd.startswith("cat /tmp/server.log"):
            return ExecResult(stdout="Server running on 5173", stderr="", exit_code=0)
        return ExecResult(stdout="000", stderr="", exit_code=0)


@pytest.mark.asyncio
async def test_build_npm_project_passes_build_timeout(monkeypatch, tmp_path: Path) -> None:
    import benchmarks.zfrontendbench.task as task_module

    monkeypatch.setattr(task_module, "Sandbox", FakeSandboxRunner)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    question = ZFrontendBenchQuestion.model_validate(
        {
            "qid": "zfe_001",
            "docker": "image:latest",
            "workspace_dir": "/workspace",
            "contexts": ["build ui"],
            "checklist": ["renders"],
            "http_build_timeout": 77,
        }
    )

    await ZFrontendBenchTask()._build_npm_project(
        question,
        project_dir,
        tmp_path / "eval",
        SimpleNamespace(sandbox_config=SandboxConfig()),  # type: ignore[arg-type]
    )

    sandbox = FakeSandboxRunner.command_sandbox
    assert sandbox is not None
    assert sandbox.uploads == [(project_dir, "/workspace")]
    assert sandbox.commands[0][1] == 77
    assert "npm run build" in sandbox.commands[0][0]
    assert len(sandbox.commands) == 1


@pytest.mark.asyncio
async def test_inference_creates_missing_workspace(monkeypatch, tmp_path: Path) -> None:
    import benchmarks.zfrontendbench.task as task_module

    monkeypatch.setattr(task_module, "Sandbox", FakeInferenceSandbox)
    question = ZFrontendBenchQuestion.model_validate(
        {
            "qid": "zfe_001",
            "docker": "image:latest",
            "workspace_dir": "/workspace",
            "contexts": ["build ui"],
            "checklist": ["renders"],
        }
    )
    ctx = SimpleNamespace(
        output_dir=tmp_path,
        sandbox_config=SandboxConfig(),
        agent_config=None,
        model_config=ModelConfig(base_url="http://example", api_key="key", timeout=60),
    )

    result = await ZFrontendBenchTask().inference(
        question,
        ctx,  # type: ignore[arg-type]
    )

    sandbox = FakeInferenceSandbox.command_sandbox
    assert isinstance(result, ZFrontendBenchInference)
    assert sandbox is not None
    assert sandbox.commands[0] == ("mkdir -p /workspace", None)


@pytest.mark.asyncio
async def test_wait_for_server_accepts_ready_log(monkeypatch) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    import benchmarks.zfrontendbench.task as task_module

    monkeypatch.setattr(task_module.asyncio, "sleep", no_sleep)
    sandbox = FakeServerReadySandbox()

    await ZFrontendBenchTask()._wait_for_server(
        sandbox,  # type: ignore[arg-type]
        5173,
        max_attempts=1,
    )

    assert sandbox.commands == [
        ("cat /tmp/server.log 2>/dev/null || echo ''", 15)
    ]


def test_tar_filter_rejects_links_and_path_escape(tmp_path: Path) -> None:
    task = ZFrontendBenchTask()
    normal = tarfile.TarInfo("index.html")
    symlink = tarfile.TarInfo("link.html")
    symlink.type = tarfile.SYMTYPE
    hardlink = tarfile.TarInfo("hard.html")
    hardlink.type = tarfile.LNKTYPE
    escape = tarfile.TarInfo("../secret.txt")

    assert task._tar_filter(normal, str(tmp_path)) is normal
    assert task._tar_filter(symlink, str(tmp_path)) is None
    assert task._tar_filter(hardlink, str(tmp_path)) is None
    assert task._tar_filter(escape, str(tmp_path)) is None


@pytest.mark.asyncio
async def test_non_index_html_entry_is_copied_to_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "site"
    source_dir.mkdir()
    entry = source_dir / "game.html"
    entry.write_text("<html></html>", encoding="utf-8")
    sandbox = FakeCommandSandbox()

    await ZFrontendBenchTask()._ensure_index_entry(
        sandbox,  # type: ignore[arg-type]
        BuildResult(
            success=True,
            project_type=ProjectType.HTML,
            source_dir=source_dir,
            entry_file=entry,
        ),
    )

    assert sandbox.commands == [
        ("cp /workspace/game.html /workspace/index.html", None)
    ]
