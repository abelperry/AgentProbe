"""Tests for SWE-bench Pro Q/I/J models and spec builder helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from benchmarks.swebench_pro.models import (
    SWEBenchProInference,
    SWEBenchProJudgement,
    SWEBenchProQuestion,
)
from benchmarks.swebench_pro.spec_builder import (
    SWEBenchProInferSpecBuilder,
    strip_binary_hunks,
)
from benchmarks.swebench_pro.task import SWEBenchProTask

from agent_probe.core.models import Error, resolve_types
from agent_probe.core.sandbox import SandboxResult


def _record() -> dict:
    return {
        "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
        "repo": "NodeBB/NodeBB",
        "base_commit": "base123",
        "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-abc",
        "prompt": "fix the bug",
        "fail_to_pass": ["test/file.js | should pass"],
        "pass_to_pass": ["test/file.js | should stay green"],
        "selected_test_files": ["test/file.js"],
        "eval_cmd": "git checkout gold123 -- test/file.js",
        "run_script": "#!/bin/bash\n",
        "parser_py": "print('parser')\n",
        "env_exports": "export A=1",
        "repo_language": "js",
        "patch": "diff --git a/a b/a\n",
        "test_patch": "diff --git a/test b/test\n",
    }


def test_question_accepts_prepared_jsonl_shape() -> None:
    question = SWEBenchProQuestion.model_validate(_record())

    assert question.qid() == "instance_NodeBB__NodeBB-abc-vnan"
    assert question.fail_to_pass == ["test/file.js | should pass"]
    assert question.pass_to_pass == ["test/file.js | should stay green"]
    assert question.selected_test_files == ["test/file.js"]
    assert question.prompt == "fix the bug"


def test_resolve_types_swebench_pro() -> None:
    assert resolve_types(SWEBenchProTask) == (
        SWEBenchProQuestion,
        SWEBenchProInference,
        SWEBenchProJudgement,
    )


def test_setup_script_hides_post_base_refs_without_pruning(tmp_path) -> None:
    builder = SWEBenchProInferSpecBuilder(
        SWEBenchProQuestion.model_validate(_record()),
        _ctx(tmp_path),
    )

    script = builder._make_setup_script()

    assert "git reset --hard base123" in script
    assert "git checkout -B base_ref base123" in script
    assert "git update-ref -d" in script
    assert "gc --prune" not in script


def test_eval_script_matches_swepro_flow(tmp_path) -> None:
    builder = SWEBenchProInferSpecBuilder(
        SWEBenchProQuestion.model_validate(_record()),
        _ctx(tmp_path),
    )

    script = builder._make_eval_script()

    assert "git reset --hard base123" in script
    assert "git apply -v /workspace/patch.diff" in script
    assert "git checkout gold123 -- test/file.js" in script
    assert "bash /workspace/run_script.sh 'test/file.js'" in script
    assert "parser.py /workspace/stdout.log /workspace/stderr.log" in script


def test_swepro_agent_prompt_uses_prepared_prompt(tmp_path) -> None:
    builder = SWEBenchProInferSpecBuilder(
        SWEBenchProQuestion.model_validate(_record()),
        _ctx(tmp_path),
    )

    assert builder._agent_prompt() == "fix the bug"


def test_strip_binary_hunks_removes_binary_sections() -> None:
    patch = (
        "diff --git a/a b/a\n"
        "--- a/a\n"
        "+++ b/a\n"
        "@@\n"
        "+text\n"
        "diff --git a/img b/img\n"
        "GIT binary patch\n"
        "literal 1\n"
        "x\n"
    )

    stripped = strip_binary_hunks(patch)

    assert "a/a" in stripped
    assert "GIT binary patch" not in stripped


def test_empty_patch_is_agent_error_but_not_inference_error(tmp_path) -> None:
    builder = SWEBenchProInferSpecBuilder(
        SWEBenchProQuestion.model_validate(_record()),
        _ctx(tmp_path),
    )
    builder.infer_dir.mkdir(parents=True)
    builder.patch_path.write_text("", encoding="utf-8")

    inference = builder.to_inference(SandboxResult())

    assert inference.error is None
    assert inference.agent_error == Error(code=-2, message="empty prediction patch")


class _PatchSandbox:
    def __init__(self, patch: str, *, read_fails: bool = False) -> None:
        self.patch = patch
        self.read_fails = read_fails
        self.calls: list[tuple[str, int | None]] = []

    async def exec_cmd(self, command: str, timeout_sec: int | None = None):
        self.calls.append((command, timeout_sec))
        if command.startswith("cat "):
            return SimpleNamespace(exit_code=0, stdout=self.patch, stderr="")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def read_file(self, path: str) -> str:
        if self.read_fails:
            raise FileNotFoundError(path)
        return self.patch


@pytest.mark.asyncio
async def test_prepare_patch_uses_long_timeout_and_read_fallback(tmp_path) -> None:
    builder = SWEBenchProInferSpecBuilder(
        SWEBenchProQuestion.model_validate(_record()),
        _ctx(tmp_path),
    )
    sandbox = _PatchSandbox("diff --git a/a b/a\n", read_fails=True)

    patch = await builder._prepare_patch(sandbox)

    assert patch == "diff --git a/a b/a\n"
    assert sandbox.calls[0][1] == 600
    assert sandbox.calls[1] == ("cat /tmp/agentprobe_swepro.patch", 60)


def _ctx(tmp_path):
    return SimpleNamespace(
        dataset_config=SimpleNamespace(options={}),
        output_dir=tmp_path / "output",
        model_config=SimpleNamespace(timeout=7200),
        sandbox_config=SimpleNamespace(),
        agent_config=SimpleNamespace(),
        log_tag=lambda: "qid|model",
    )
