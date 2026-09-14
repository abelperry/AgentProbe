"""Tests for MTACIFBench models, trace slicing, judge parsing, and metrics."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from benchmarks.mtacifbench.function_eval import (
    ProjectType,
    _parse_function_verdict,
    detect_project,
)
from benchmarks.mtacifbench.models import (
    FunctionCheckResult,
    IFCheckResult,
    IFConstraint,
    IFRoundResult,
    MTACIFBenchInference,
    MTACIFBenchJudgement,
    MTACIFBenchQuestion,
    RoundRecord,
)
from benchmarks.mtacifbench.task import (
    FAIL_CONCLUSION,
    PASS_CONCLUSION,
    MTACIFBenchTask,
)
from benchmarks.mtacifbench.utils import (
    diff_round_coverage,
    extract_round_context,
    extract_workspace_archive,
    parse_jsonl_result,
    safe_path_component,
    sanitize_api_error_text,
)
from benchmarks.mtacifbench.validation import (
    normalize_conclusion,
    parse_check_results,
    run_validation_code,
)

from agent_probe.agents.claude_code import ClaudeCodeAgent
from agent_probe.agents.opencode import OpenCodeAgent
from agent_probe.agents.opencode_output import (
    extract_opencode_final_text,
    inspect_opencode_jsonl,
)
from agent_probe.config import AgentConfig, ModelConfig
from agent_probe.core.models import Error, resolve_types
from agent_probe.core.sandbox import ExecResult

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPT = REPO_ROOT / "scripts" / "build_mtacifbench_dataset.py"


# ---------------------------------------------------------------------------
# Models / wiring
# ---------------------------------------------------------------------------
def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "task_id": 1,
        "docker": "infer:latest",
        "judge_docker": "judge:latest",
        "workspace_dir": "/workspace",
        "repository_policy": "每轮回复以喵开头",
        "repository_policy_checklist": [
            {"constraint": "回复以喵开头", "validation_code": "", "tags": ["Content"]}
        ],
        "task_category": "mtacifbench",
        "function_checklist": [],
        "rounds": [
            {
                "round_id": 0,
                "prompt": "build a table",
                "instruction_following_checklist": [
                    {"constraint": "回复以喵开头", "validation_code": "", "tags": ["内容"]},
                ],
            },
            {
                "round_id": 1,
                "prompt": "add filters",
                "instruction_following_checklist": [
                    {"constraint": "回复以喵开头", "validation_code": ""},
                    {"constraint": "禁止 ESLint", "validation_code": ""},
                ],
            },
        ],
    }
    record.update(overrides)
    return record


def test_resolve_types_and_question_parsing() -> None:
    assert resolve_types(MTACIFBenchTask) == (
        MTACIFBenchQuestion,
        MTACIFBenchInference,
        MTACIFBenchJudgement,
    )
    question = MTACIFBenchQuestion.model_validate(_record())
    assert question.qid() == "1"
    assert len(question.checklist_for(1)) == 2
    assert question.checklist_for(99) == []
    # No explicit description: fall back to the concatenated round prompts.
    assert "第0轮：build a table" in question.task_description


def test_rounds_are_independent_not_inherited() -> None:
    """A later round may forbid what an earlier round required."""
    question = MTACIFBenchQuestion.model_validate(_record())
    assert [item.constraint for item in question.checklist_for(0)] == ["回复以喵开头"]
    assert [item.constraint for item in question.checklist_for(1)] == [
        "回复以喵开头",
        "禁止 ESLint",
    ]


def test_round_checklist_is_the_dataset_round_checklist_verbatim() -> None:
    """Repository-policy constraints are not merged into the round checklist.

    The dataset repeats each policy constraint into the rounds it applies to,
    so the round checklist is already authoritative. A policy constraint the
    round omits (``global-b`` here) is deliberately not scored for that round.
    """
    question = MTACIFBenchQuestion.model_validate(
        _record(
            repository_policy_checklist=[
                {"constraint": "global-a", "validation_code": "repo-a"},
                {"constraint": "global-b", "validation_code": "repo-b"},
            ],
            rounds=[
                {
                    "round_id": 0,
                    "instruction": "build",
                    "instruction_following_checklist": [
                        {"constraint": "global-a", "validation_code": "round-a"},
                        {"constraint": "local", "validation_code": "local-code"},
                    ],
                }
            ],
        )
    )
    checklist = question.checklist_for(0)
    assert [item.constraint for item in checklist] == ["global-a", "local"]
    # The round's own validation_code wins over the repository-policy copy.
    assert [item.validation_code for item in checklist] == ["round-a", "local-code"]
    assert question.validation_codes_for(0) == ["round-a", "local-code"]
    assert question.constraint_count == 2


def test_wbs_question_dump_contains_only_the_new_schema() -> None:
    payload = MTACIFBenchQuestion.model_validate(_record()).model_dump(mode="json")
    assert payload["task_id"] == 1
    assert "repository_policy" in payload
    assert "repository_policy_checklist" in payload
    assert "system_prompt" not in payload
    constraint = payload["repository_policy_checklist"][0]
    assert set(constraint) == {"constraint", "validation_code", "tags"}
    assert "weight" not in constraint


def test_project_instruction_merge_preserves_existing_content() -> None:
    assert (
        MTACIFBenchTask._merge_project_instructions(
            existing_content="# Existing\n",
            project_instructions="Follow policy",
        )
        == "# Existing\n\nFollow policy\n"
    )


def test_function_evaluator_detects_npm_project_and_binary_verdict(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "scripts": {"build": "vite build"},
                "dependencies": {"react": "latest", "vite": "latest"},
            }
        ),
        encoding="utf-8",
    )
    project = detect_project(tmp_path)
    assert project.project_type is ProjectType.NPM
    assert project.framework == "react"
    assert _parse_function_verdict("判断结论：该项目符合要求") == 1.0
    assert _parse_function_verdict("判断结论：该项目不符合要求") == 0.0


def test_experiment_factory_wires_mtacifbench(tmp_path: Path) -> None:
    from agent_probe.config import DatasetConfig, EvalExperimentConfig
    from agent_probe.core.factory import ExperimentFactory

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "questions.jsonl").write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cfg = EvalExperimentConfig(
        name="mtacif-test",
        concurrency=1,
        output_dir=str(tmp_path / "output"),
        models={"m": ModelConfig(base_url="http://example", api_key="k")},
        agents={"a": AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent")},
        datasets={
            "mtacifbench": DatasetConfig(
                name="mtacifbench",
                data_dir=str(data_dir),
                task_type="benchmarks.mtacifbench.task.MTACIFBenchTask",
                judge_config_path={"instruction_following": "unused.yaml"},
            )
        },
    )
    executor = ExperimentFactory().create(cfg)
    assert [unit.qid for unit in executor._units] == ["1"]


def test_safe_path_component_rejects_traversal() -> None:
    assert safe_path_component("verified_1") == "verified_1"
    for bad in ("..", ".", "a/b", "a\\b", "*", "q?", "[x]", "", "  "):
        with pytest.raises(ValueError):
            safe_path_component(bad)


def test_sanitize_api_error_text_drops_payload() -> None:
    text = "API Error: 500 upstream\nrequest payload: {'key': 'secret'}"
    assert sanitize_api_error_text(text) == "API Error: 500 upstream"
    # Ordinary replies are untouched.
    assert sanitize_api_error_text("喵～ 完成了") == "喵～ 完成了"


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------
def _tar_with_unsafe_members(tmp_path: Path) -> Path:
    payload = tmp_path / "payload"
    (payload / "node_modules" / "pkg").mkdir(parents=True)
    (payload / "node_modules" / "pkg" / "index.js").write_text("junk", encoding="utf-8")
    (payload / "src").mkdir()
    (payload / "src" / "ok.txt").write_text("kept", encoding="utf-8")
    (payload / "second.txt").write_text("also kept", encoding="utf-8")

    archive = tmp_path / "workspace.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload / "node_modules", arcname="node_modules")
        # An escaping symlink placed before the good files: aborting here would
        # silently drop everything that follows.
        link = tarfile.TarInfo("evil_link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
        traversal = tarfile.TarInfo("../escaped.txt")
        traversal.size = 4
        import io

        tar.addfile(traversal, io.BytesIO(b"evil"))
        tar.add(payload / "src" / "ok.txt", arcname="src/ok.txt")
        tar.add(payload / "second.txt", arcname="second.txt")
    return archive


def test_extract_workspace_archive_drops_unsafe_members_and_keeps_rest(
    tmp_path: Path,
) -> None:
    archive = _tar_with_unsafe_members(tmp_path)
    target = tmp_path / "out"
    extract_workspace_archive(archive, target)

    extracted = sorted(
        str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()
    )
    assert extracted == ["second.txt", "src/ok.txt"]
    assert (target / "src" / "ok.txt").read_text(encoding="utf-8") == "kept"
    assert not (target / "node_modules").exists()
    assert not (target / "evil_link").exists()
    assert not (tmp_path / "escaped.txt").exists()


# ---------------------------------------------------------------------------
# Round coverage
# ---------------------------------------------------------------------------
def _round_record(round_id: int) -> RoundRecord:
    return RoundRecord(round_index=round_id, round_id=round_id)


def test_diff_round_coverage_reports_missing_unexpected_duplicate() -> None:
    assert diff_round_coverage([_round_record(0), _round_record(1)], [0, 1]) == ([], [], [])
    assert diff_round_coverage([_round_record(0)], [0, 1]) == ([1], [], [])
    assert diff_round_coverage([_round_record(0), _round_record(5)], [0]) == ([], [5], [])
    # A duplicate must not mask a missing round.
    assert diff_round_coverage([_round_record(0), _round_record(0)], [0, 1]) == ([1], [], [0])


# ---------------------------------------------------------------------------
# Trace slicing
# ---------------------------------------------------------------------------
def _trace_line(role: str, content: Any) -> str:
    return json.dumps(
        {"type": role, "message": {"role": role, "content": content}},
        ensure_ascii=False,
    )


def _round0_trace() -> str:
    """What the session file holds right after round 0 finishes."""
    return "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            _trace_line("user", "# 需求：\nbuild a table\n"),
            _trace_line("assistant", [{"type": "text", "text": "round0 done"}]),
            _trace_line(
                "user",
                [{"type": "tool_result", "tool_use_id": "t1", "content": "HUGE OUTPUT"}],
            ),
        ]
    )


def _round1_trace() -> str:
    """The same file after round 1 — one shared session, so it grows."""
    return "\n".join(
        [
            _round0_trace(),
            _trace_line("user", "# 需求：\nadd filters\n"),
            _trace_line(
                "assistant",
                [
                    {"type": "tool_use", "id": "t2", "name": "Edit", "signature": "sig"},
                    {"type": "text", "text": "round1 done"},
                ],
            ),
        ]
    )


def test_extract_round_context_slices_by_offset() -> None:
    first, total_after_first, used_first = extract_round_context(
        _round0_trace(), 0, "build a table"
    )
    second, total_after_second, used_second = extract_round_context(
        _round1_trace(), total_after_first, "add filters"
    )

    first_messages = json.loads(first)
    second_messages = json.loads(second)
    assert (total_after_first, used_first) == (3, 2)
    assert (total_after_second, used_second) == (5, 1)

    # Round 0's flow keeps the assistant reply and the tool result placeholder,
    # never the user instruction itself.
    assert [item["role"] for item in first_messages] == ["assistant", "user"]
    assert first_messages[1]["content"][0]["content"] == "[工具返回结果已省略]"
    assert "round1 done" not in first
    # Round 1's slice is exactly the tail, with the tool_use id renamed and the
    # signature stripped.
    assert len(second_messages) == 1
    assert second_messages[0]["content"][0]["tool_use_id"] == "t2"
    assert "signature" not in second_messages[0]["content"][0]
    assert "round0 done" not in second
    # system/init events never reach the judge.
    assert "init" not in first


def test_extract_round_context_falls_back_to_prompt_boundary() -> None:
    """A stale offset must still yield this round only, never every round."""
    context, _total, used = extract_round_context(_round1_trace(), 99, "add filters")
    assert used == 1
    assert "round1 done" in context
    assert "round0 done" not in context


# ---------------------------------------------------------------------------
# Judge output parsing
# ---------------------------------------------------------------------------
def _checklist(*constraints: str) -> list[IFConstraint]:
    return [IFConstraint(constraint=item) for item in constraints]


def _block(index: int, requirement: str, conclusion: str) -> str:
    return (
        f"[要求{index}-开始]\n"
        f"要求：{requirement}\n"
        f"分析：详细分析\n"
        f"结论：{conclusion}\n"
        f"[要求{index}-结束]"
    )


def test_parse_check_results_happy_path() -> None:
    checklist = _checklist("回复以喵开头", "禁止 ESLint")
    output = "\n\n".join(
        [
            _block(1, "回复以喵开头", PASS_CONCLUSION),
            _block(2, "禁止 ESLint", FAIL_CONCLUSION),
        ]
    )
    parsed = parse_check_results(output, checklist)
    assert parsed is not None
    assert [item.passed for item in parsed] == [True, False]


@pytest.mark.parametrize(
    "output_builder",
    [
        pytest.param(lambda c: "", id="empty"),
        pytest.param(lambda c: "分析完了，看起来还不错", id="no_markers"),
        pytest.param(lambda c: _block(1, c[0].constraint, PASS_CONCLUSION), id="too_few_blocks"),
        pytest.param(
            lambda c: "\n\n".join(
                [
                    _block(1, c[0].constraint, PASS_CONCLUSION),
                    _block(1, c[1].constraint, PASS_CONCLUSION),
                ]
            ),
            id="duplicate_index",
        ),
        pytest.param(
            lambda c: "\n\n".join(
                [
                    _block(1, c[0].constraint, "看起来还行"),
                    _block(2, c[1].constraint, PASS_CONCLUSION),
                ]
            ),
            id="unparsable_conclusion",
        ),
        pytest.param(
            lambda c: "\n\n".join(
                [
                    _block(1, c[0].constraint, "此处只能是 [[满足了该要求]] 或 [[没有满足该要求]]"),
                    _block(2, c[1].constraint, PASS_CONCLUSION),
                ]
            ),
            id="echoed_prompt_template",
        ),
    ],
)
def test_parse_check_results_is_fail_closed(output_builder: Any) -> None:
    """A missing or undecidable verdict must never become a pass."""
    checklist = _checklist("回复以喵开头", "禁止 ESLint")
    assert parse_check_results(output_builder(checklist), checklist) is None


def test_parse_check_results_empty_checklist_returns_empty() -> None:
    assert parse_check_results("anything", []) == []


@pytest.mark.parametrize(
    "output_builder",
    [
        # Blocks keyed by their own index, so emission order does not matter.
        pytest.param(
            lambda c: "\n\n".join(
                [
                    _block(2, c[1].constraint, PASS_CONCLUSION),
                    _block(1, c[0].constraint, PASS_CONCLUSION),
                ]
            ),
            id="out_of_order",
        ),
        # Markdown drift around the markers and field labels.
        pytest.param(
            lambda c: "\n\n".join(
                [
                    f"### **[要求1-开始]**\n"
                    f"- **要求**：{c[0].constraint}\n"
                    f"- **分析**：详细分析\n"
                    f"- **结论**：{PASS_CONCLUSION}\n"
                    f"**[要求1-结束]**",
                    _block(2, c[1].constraint, PASS_CONCLUSION),
                ]
            ),
            id="markdown_decorated",
        ),
        # The verdict on the line after the 结论 label.
        pytest.param(
            lambda c: "\n\n".join(
                [
                    f"[要求1-开始]\n"
                    f"要求：{c[0].constraint}\n"
                    f"分析：详细分析\n"
                    f"结论：\n{PASS_CONCLUSION}\n"
                    f"[要求1-结束]",
                    _block(2, c[1].constraint, PASS_CONCLUSION),
                ]
            ),
            id="conclusion_on_next_line",
        ),
        # The whole verdict wrapped in prose and a code fence.
        pytest.param(
            lambda c: "我先说明一下思路。\n\n```\n"
            + "\n\n".join(
                [
                    _block(1, c[0].constraint, PASS_CONCLUSION),
                    _block(2, c[1].constraint, PASS_CONCLUSION),
                ]
            )
            + "\n```\n\n以上是我的判断。",
            id="fenced_with_prose",
        ),
    ],
)
def test_parse_check_results_absorbs_judge_format_drift(output_builder: Any) -> None:
    """Format tolerance: a real verdict must not be discarded over layout."""
    checklist = _checklist("回复以喵开头", "禁止 ESLint")
    parsed = parse_check_results(output_builder(checklist), checklist)
    assert parsed is not None
    assert [item.index for item in parsed] == [1, 2]
    assert all(item.passed for item in parsed)


def test_parse_check_results_keeps_the_judge_requirement_text() -> None:
    """WBS behaviour: the judge's own requirement text is recorded as-is.

    The checklist text is only used to fill an omitted 要求 field, so a
    mismatch is visible in the artifact rather than discarding the verdict.
    """
    checklist = _checklist("回复以喵开头", "禁止 ESLint")
    output = "\n\n".join(
        [
            _block(1, "回复必须以喵开头", PASS_CONCLUSION),
            _block(2, "禁止 ESLint", PASS_CONCLUSION),
        ]
    )
    parsed = parse_check_results(output, checklist)
    assert parsed is not None
    assert parsed[0].requirement == "回复必须以喵开头"


def test_parse_check_results_backfills_an_omitted_requirement() -> None:
    checklist = _checklist("回复以喵开头")
    output = "[要求1-开始]\n分析：详细分析\n结论：" + PASS_CONCLUSION + "\n[要求1-结束]"
    parsed = parse_check_results(output, checklist)
    assert parsed is not None
    assert parsed[0].requirement == "回复以喵开头"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[[满足了该要求]]", PASS_CONCLUSION),
        ("[满足了该要求]", PASS_CONCLUSION),
        ("满足了该要求", PASS_CONCLUSION),
        ("[[没有满足该要求]]", FAIL_CONCLUSION),
        # "没有满足了该要求" contains "满足了该要求" — polarity must not flip.
        ("[[没有满足了该要求]]", FAIL_CONCLUSION),
        ("没有满足了该要求", FAIL_CONCLUSION),
        ("无法判断", None),
        ("", None),
    ],
)
def test_normalize_conclusion(raw: str, expected: str | None) -> None:
    assert normalize_conclusion(raw) == expected


# ---------------------------------------------------------------------------
# Dataset validation code execution
# ---------------------------------------------------------------------------
def _has_validation_deps() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import emoji"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


requires_validation_deps = pytest.mark.skipif(
    not _has_validation_deps(), reason="emoji not installed in this interpreter"
)


@requires_validation_deps
def test_validation_code_returns_verdict(tmp_path: Path) -> None:
    code = "def check(response, workspace_path):\n    return response.startswith('喵')\n"
    assert run_validation_code(code, "喵～ ok", tmp_path, timeout=30) is True
    assert run_validation_code(code, "ok", tmp_path, timeout=30) is False


@requires_validation_deps
def test_validation_code_can_read_the_workspace(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text("// -*- coding: utf-8 -*-\n", encoding="utf-8")
    code = (
        "import os\n"
        "def check(response, workspace_path):\n"
        "    path = os.path.join(workspace_path, 'app.js')\n"
        "    return open(path, encoding='utf-8').read().startswith('// -*-')\n"
    )
    assert run_validation_code(code, "", tmp_path, timeout=30) is True


@requires_validation_deps
def test_validation_code_helpers_are_injected(tmp_path: Path) -> None:
    code = (
        "def check(response, workspace_path):\n"
        "    return count_word(response) > 0 and is_emoji('🙂')\n"
    )
    assert run_validation_code(code, "你好 world", tmp_path, timeout=30) is True


@requires_validation_deps
@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("def check(response, workspace_path):\n    while True:\n        pass\n", "timeout"),
        ("def check(response, workspace_path):\n    raise RuntimeError('boom')\n", "raises"),
        ("def check(response, workspace_path):\n    return 'yes'\n", "non_bool"),
        ("VALUE = 1\n", "no_entry_point"),
    ],
)
def test_validation_code_degrades_to_judge(tmp_path: Path, code: str, reason: str) -> None:
    """A checker we cannot trust must fall back, never score the constraint 0."""
    assert run_validation_code(code, "x", tmp_path, timeout=3) is None


@requires_validation_deps
def test_validation_code_prefers_conventional_entry_point(tmp_path: Path) -> None:
    """A two-argument private helper must not be mistaken for the entry point."""
    code = (
        "def apply_rules(response, workspace_path):\n"
        "    return 'helper'\n"
        "def check_requirement(response, workspace_path):\n"
        "    return True\n"
    )
    assert run_validation_code(code, "x", tmp_path, timeout=30) is True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _judgement(
    rounds: list[tuple[int, bool, list[bool]]],
    error: Error | None = None,
) -> MTACIFBenchJudgement:
    checks = [
        IFRoundResult(
            round_id=round_id,
            passed=passed,
            check_results=[
                IFCheckResult(
                    index=index + 1,
                    requirement=f"c{index}",
                    conclusion=PASS_CONCLUSION if ok else FAIL_CONCLUSION,
                )
                for index, ok in enumerate(constraints)
            ],
        )
        for round_id, passed, constraints in rounds
    ]
    return MTACIFBenchJudgement(
        instruction_following_checks=checks,
        instruction_following_score=1.0 if all(item.passed for item in checks) else 0.0,
        error=error,
    )


def test_collect_metrics_excludes_invalid_rows() -> None:
    task = MTACIFBenchTask()
    judgements = [
        _judgement([(0, True, [True, True]), (1, True, [True])]),
        _judgement([(0, False, [True, False]), (1, True, [True])]),
        # Infrastructure failure: must not count as a constraint violation.
        _judgement([(0, False, [False])], error=Error(code=-1, message="judge crashed")),
    ]
    scores, success = task.collect_metrics(judgements)

    assert success == 2
    assert scores["IFISR"] == pytest.approx(75.0)
    # 6 constraints across the two valid rows, 5 satisfied.
    assert scores["IFCSR"] == pytest.approx(100 * 5 / 6)
    assert scores["num_total"] == 3.0
    assert scores["num_success"] == 2.0
    assert "IFSSR" not in scores
    assert "num_strict_pass" not in scores


def test_collect_metrics_on_empty_input() -> None:
    scores, success = MTACIFBenchTask().collect_metrics([])
    assert success == 0
    assert scores["num_total"] == 0.0
    assert scores["num_success"] == 0.0
    assert scores["IFISR"] == 0.0
    assert scores["IFCSR"] == 0.0


def test_collect_metrics_includes_function_scores_when_enabled() -> None:
    judgement = _judgement([(0, True, [True])]).model_copy(
        update={
            "checks": [
                FunctionCheckResult(id=0, description="works", score=1.0),
                FunctionCheckResult(id=1, description="fails", score=0.0),
            ],
            "function_score": 0.5,
            "function_checklist_skipped": False,
            "build_success": False,
        }
    )
    scores, success = MTACIFBenchTask().collect_metrics([judgement])
    assert success == 1
    assert scores["average"] == 50.0
    assert scores["ISR"] == 0.0
    assert scores["CSR"] == 50.0
    assert scores["BSR"] == 0.0


def test_result_summary_uses_scored_checklists_and_recorded_build_status(
    tmp_path: Path,
) -> None:
    from scripts.summarize_mtacifbench_results import summarize

    result_dir = tmp_path / "model" / "result"
    result_dir.mkdir(parents=True)
    payload = {
        "question": {
            "repository_policy_checklist": [
                {"constraint": "global", "tags": ["Content", "Format"]}
            ],
            "rounds": [
                {
                    "round_id": 0,
                    "instruction_following_checklist": [
                        {"constraint": "local", "tags": ["Style", "Naming"]}
                    ],
                }
            ],
            "function_checklist": ["works"],
        },
        "inference": {"error": None},
        "judgement": {
            "instruction_following_checks": [
                {
                    "round_id": 0,
                    "passed": True,
                    "check_results": [
                        {
                            "index": 1,
                            "requirement": "global",
                            "conclusion": PASS_CONCLUSION,
                        },
                        {
                            "index": 2,
                            "requirement": "local",
                            "conclusion": PASS_CONCLUSION,
                        },
                    ],
                }
            ],
            "instruction_following_score": 1.0,
            "checks": [{"id": 0, "description": "works", "score": 0.0}],
            "function_score": 0.0,
            "function_checklist_skipped": False,
            "build_success": True,
            "error": None,
        },
    }
    (result_dir / "1.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = summarize(result_dir, expected_total=None, include_cases=False)
    stats = summary["model_stats"]
    assert "IF_SSR_all" not in stats
    assert "strict_success_cases" not in stats
    assert stats["BSR"] == 1.0
    assert stats["build_success_cases"] == 1
    # Only the round's own checklist is scored, so it supplies the one entry
    # that can be categorised; the judge's extra verdict has no dataset entry
    # to read tags or a source label from.
    categories = summary["instruction_following_by_primary_category"]
    assert categories["Style"]["constraints_total"] == 1
    assert categories["未识别"]["constraints_total"] == 1
    assert "Content" not in categories
    sources = summary["instruction_following_by_constraint_source"]
    assert sources["new_added"]["constraints_total"] == 1
    assert sources["unknown"]["constraints_total"] == 1
    assert "repository_policy" not in sources


def _round_verdict(round_id: int, *constraints: tuple[str, str], passed: bool) -> IFRoundResult:
    return IFRoundResult(
        round_id=round_id,
        passed=passed,
        check_results=[
            IFCheckResult(index=index, requirement=requirement, conclusion=conclusion)
            for index, (requirement, conclusion) in enumerate(constraints, start=1)
        ],
    )


def test_validate_round_judgements_accepts_full_coverage() -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    results = [
        _round_verdict(0, ("回复以喵开头", PASS_CONCLUSION), passed=True),
        _round_verdict(
            1,
            ("回复以喵开头", PASS_CONCLUSION),
            ("禁止 ESLint", PASS_CONCLUSION),
            passed=True,
        ),
    ]
    assert MTACIFBenchTask._validate_round_judgements(question, results) is None


@pytest.mark.parametrize(
    ("results_builder", "expected_message"),
    [
        pytest.param(lambda: [], "round coverage mismatch", id="missing_rounds"),
        pytest.param(
            lambda: [
                _round_verdict(0, ("回复以喵开头", PASS_CONCLUSION), passed=True),
                # Round 1 declares two constraints; only one was scored.
                _round_verdict(1, ("回复以喵开头", PASS_CONCLUSION), passed=True),
            ],
            "constraint count mismatch",
            id="under_covered_round",
        ),
        pytest.param(
            lambda: [
                _round_verdict(0, ("回复以喵开头", PASS_CONCLUSION), passed=True),
                _round_verdict(
                    1,
                    ("回复以喵开头", PASS_CONCLUSION),
                    ("禁止 ESLint", FAIL_CONCLUSION),
                    passed=True,
                ),
            ],
            "aggregate verdict is inconsistent",
            id="passed_contradicts_checks",
        ),
        pytest.param(
            lambda: [
                _round_verdict(0, ("回复以喵开头", PASS_CONCLUSION), passed=True),
                _round_verdict(
                    1,
                    ("回复以喵开头", PASS_CONCLUSION),
                    ("伪造的要求", PASS_CONCLUSION),
                    passed=True,
                ),
            ],
            "does not match dataset",
            id="requirement_drifted_from_dataset",
        ),
        pytest.param(
            lambda: [
                _round_verdict(0, ("回复以喵开头", PASS_CONCLUSION), passed=True),
                IFRoundResult(round_id=1, passed=False, parse_failed=True),
            ],
            "could not be parsed",
            id="unresolved_round",
        ),
    ],
)
def test_validate_round_judgements_rejects_incomplete_coverage(
    results_builder: Any, expected_message: str
) -> None:
    """An under-covered or self-contradictory judgement must not score as a pass."""
    question = MTACIFBenchQuestion.model_validate(_record())
    message = MTACIFBenchTask._validate_round_judgements(question, results_builder())
    assert message is not None
    assert expected_message in message


def test_reusable_round_results_skips_unresolved_rounds() -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    prev = MTACIFBenchJudgement(
        instruction_following_checks=[
            IFRoundResult(
                round_id=0,
                passed=True,
                check_results=[
                    IFCheckResult(index=1, requirement="回复以喵开头", conclusion=PASS_CONCLUSION)
                ],
            ),
            IFRoundResult(round_id=1, passed=False, parse_failed=True),
        ]
    )
    reusable = MTACIFBenchTask._reusable_round_results(prev, question)
    assert set(reusable) == {0}


def test_reusable_round_results_rejects_stale_checklist_length() -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    prev = MTACIFBenchJudgement(
        instruction_following_checks=[
            # Round 1 now has two constraints; a one-item verdict is stale.
            IFRoundResult(
                round_id=1,
                passed=True,
                check_results=[
                    IFCheckResult(index=1, requirement="回复以喵开头", conclusion=PASS_CONCLUSION)
                ],
            )
        ]
    )
    assert MTACIFBenchTask._reusable_round_results(prev, question) == {}


# ---------------------------------------------------------------------------
# Inference validation
# ---------------------------------------------------------------------------
def _material(root: Path, round_id: int) -> None:
    material = root / f"round_{round_id}"
    (material / "workspace_snapshot").mkdir(parents=True)
    (material / "context.json").write_text("[]", encoding="utf-8")
    (material / "last_response.txt").write_text("喵～", encoding="utf-8")


def test_validate_inference_accepts_complete_output(tmp_path: Path) -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    tar = tmp_path / "workspace.tar.gz"
    tar.write_bytes(b"x")
    material_root = tmp_path / "instruction_following"
    _material(material_root, 0)
    _material(material_root, 1)
    inference = MTACIFBenchInference(
        workspace_tar_path=tar,
        round_records=[_round_record(0), _round_record(1)],
    )
    assert MTACIFBenchTask()._validate_inference(question, inference, material_root) is None


@pytest.mark.parametrize(
    "breakage",
    ["no_tar", "missing_round", "missing_material", "agent_error"],
)
def test_validate_inference_rejects_partial_output(tmp_path: Path, breakage: str) -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    tar = tmp_path / "workspace.tar.gz"
    tar.write_bytes(b"x")
    material_root = tmp_path / "instruction_following"
    _material(material_root, 0)
    _material(material_root, 1)
    records = [_round_record(0), _round_record(1)]
    agent_error = None

    if breakage == "no_tar":
        tar.unlink()
    elif breakage == "missing_round":
        records = [_round_record(0)]
    elif breakage == "missing_material":
        (material_root / "round_1" / "context.json").unlink()
    else:
        agent_error = Error(code=-1, message="agent stopped with error")

    inference = MTACIFBenchInference(
        workspace_tar_path=tar,
        round_records=records,
        agent_error=agent_error,
    )
    error = MTACIFBenchTask()._validate_inference(question, inference, material_root)
    assert error is not None
    assert error.code < 0


# ---------------------------------------------------------------------------
# Round judging: validation-code verdicts merged with judge verdicts
# ---------------------------------------------------------------------------
def _mixed_round_question() -> MTACIFBenchQuestion:
    return MTACIFBenchQuestion.model_validate(
        _record(
            rounds=[
                {
                    "round_id": 0,
                    "prompt": "build a table",
                    "instruction_following_checklist": [
                        # index 0: judged by code, passes
                        {
                            "constraint": "回复以喵开头",
                            "validation_code": (
                                "def check(response, workspace_path):\n"
                                "    return response.startswith('喵')\n"
                            ),
                        },
                        # index 1: no code, must go to the judge
                        {"constraint": "注释必须是英文", "validation_code": ""},
                        # index 2: code crashes, must degrade to the judge
                        {
                            "constraint": "禁止 TODO",
                            "validation_code": (
                                "def check(response, workspace_path):\n"
                                "    raise RuntimeError('broken checker')\n"
                            ),
                        },
                    ],
                }
            ]
        )
    )


def _material_dir(tmp_path: Path, response: str) -> Path:
    material = tmp_path / "material" / "round_0"
    (material / "workspace_snapshot").mkdir(parents=True)
    (material / "context.json").write_text("[]", encoding="utf-8")
    (material / "last_response.txt").write_text(response, encoding="utf-8")
    return material


def _judge_sandbox_result(text: str) -> Any:
    from agent_probe.core.models import LastAssistant
    from agent_probe.core.sandbox import SandboxResult

    return SandboxResult(
        rounds=[ExecResult(stdout=text, stderr="", exit_code=0)],
        last_assistant=LastAssistant(stop_reason="end_turn", content_text=text),
    )


class _FakeEvalContext:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def log_tag(self) -> str:
        return "test|model"


def _judging_ctx(tmp_path: Path) -> _FakeEvalContext:
    """A context whose judge config actually resolves.

    ``judge()`` reads ``ctx.dataset_config`` before judging any round, so a
    context without one fails the question early and never reaches the per-round
    logic under test.
    """
    from agent_probe.config import DatasetConfig

    judge_yaml = tmp_path / "judge.yaml"
    judge_yaml.write_text(
        "model:\n"
        '  base_url: "http://judge.invalid"\n'
        '  api_key: "k"\n'
        '  model_name: "judge-model"\n'
        "agent:\n"
        '  type: "agent_probe.agents.claude_code.ClaudeCodeAgent"\n',
        encoding="utf-8",
    )
    ctx = _FakeEvalContext(tmp_path / "out")
    ctx.dataset_config = DatasetConfig(name="mtacifbench", judge_config_path=str(judge_yaml))
    return ctx


@requires_validation_deps
@pytest.mark.asyncio
async def test_judge_round_merges_code_and_judge_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = _mixed_round_question()
    material = _material_dir(tmp_path, "喵～ 已完成")
    prompts: list[str] = []

    # The judge only sees the two constraints the checkers could not decide, so
    # its own numbering restarts at 1 and must be mapped back.
    judge_text = "\n\n".join(
        [
            _block(1, "注释必须是英文", PASS_CONCLUSION),
            _block(2, "禁止 TODO", FAIL_CONCLUSION),
        ]
    )

    async def _fake_judge(self: Any, **kwargs: Any) -> Any:
        prompts.append(kwargs["prompt"])
        return _judge_sandbox_result(judge_text)

    monkeypatch.setattr(MTACIFBenchTask, "_run_judge_sandbox", _fake_judge)

    result = await MTACIFBenchTask()._judge_round(
        question=question,
        ctx=_FakeEvalContext(tmp_path / "out"),
        round_spec=question.rounds[0],
        material_dir=material,
        eval_dir=tmp_path / "out" / "eval",
    )

    assert len(prompts) == 1
    # The prompt lists exactly the undecided constraints, renumbered.
    assert "[要求1]：注释必须是英文" in prompts[0]
    assert "[要求2]：禁止 TODO" in prompts[0]
    # The constraint its checker already decided is not re-sent to the judge.
    assert "回复以喵开头" not in prompts[0].split("## 要求列表")[1]

    assert not result.parse_failed
    assert result.passed is False
    assert [item.index for item in result.check_results] == [1, 2, 3]
    assert [item.requirement for item in result.check_results] == [
        "回复以喵开头",
        "注释必须是英文",
        "禁止 TODO",
    ]
    assert [item.source for item in result.check_results] == [
        "validation_code",
        "judge",
        "judge",
    ]
    assert [item.passed for item in result.check_results] == [True, True, False]
    # Judge-owned artifacts land under eval/, never back in the infer material.
    assert (
        tmp_path / "out" / "eval" / "instruction_following" / "round_0" / "round_results.json"
    ).is_file()
    assert sorted(item.name for item in material.iterdir()) == [
        "context.json",
        "last_response.txt",
        "workspace_snapshot",
    ]


@requires_validation_deps
@pytest.mark.asyncio
async def test_judge_round_skips_judge_when_all_constraints_have_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = MTACIFBenchQuestion.model_validate(
        _record(
            rounds=[
                {
                    "round_id": 0,
                    "prompt": "p",
                    "instruction_following_checklist": [
                        {
                            "constraint": "回复以喵开头",
                            "validation_code": (
                                "def check(response, workspace_path):\n"
                                "    return response.startswith('喵')\n"
                            ),
                        }
                    ],
                }
            ]
        )
    )
    material = _material_dir(tmp_path, "喵～ ok")

    async def _fail(self: Any, **kwargs: Any) -> Any:
        raise AssertionError("judge sandbox must not start")

    monkeypatch.setattr(MTACIFBenchTask, "_run_judge_sandbox", _fail)

    result = await MTACIFBenchTask()._judge_round(
        question=question,
        ctx=_FakeEvalContext(tmp_path / "out"),
        round_spec=question.rounds[0],
        material_dir=material,
        eval_dir=tmp_path / "out" / "eval",
    )
    assert result.passed is True
    assert result.check_results[0].source == "validation_code"


@pytest.mark.asyncio
async def test_judge_round_retries_then_reports_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    material = _material_dir(tmp_path, "ok")
    calls: list[int] = []

    async def _garbage(self: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return _judge_sandbox_result("我觉得都挺好的")

    monkeypatch.setattr(MTACIFBenchTask, "_run_judge_sandbox", _garbage)

    result = await MTACIFBenchTask()._judge_round(
        question=question,
        ctx=_FakeEvalContext(tmp_path / "out"),
        round_spec=question.rounds[0],
        material_dir=material,
        eval_dir=tmp_path / "out" / "eval",
    )
    # judge_parse_retry_max defaults to 3, so 4 attempts total.
    assert len(calls) == 4
    assert result.parse_failed is True
    assert result.passed is False


@pytest.mark.asyncio
async def test_judge_round_reports_missing_material(tmp_path: Path) -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    result = await MTACIFBenchTask()._judge_round(
        question=question,
        ctx=_FakeEvalContext(tmp_path / "out"),
        round_spec=question.rounds[0],
        material_dir=tmp_path / "missing",
        eval_dir=tmp_path / "out" / "eval",
    )
    assert result.parse_failed is True


@pytest.mark.asyncio
async def test_judge_skips_sandboxes_for_invalid_inference(tmp_path: Path) -> None:
    """An invalid inference must not burn judge budget on missing material."""
    question = MTACIFBenchQuestion.model_validate(_record())
    inference = MTACIFBenchInference(error=Error(code=-2, message="workspace missing"))
    judgement = await MTACIFBenchTask().judge(
        question, inference, _FakeEvalContext(tmp_path / "out")
    )
    assert judgement.error is not None
    assert judgement.instruction_following_score == 0.0
    assert judgement.instruction_following_checks == []


@pytest.mark.asyncio
async def test_judge_marks_unresolved_rounds_for_rejudging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = MTACIFBenchQuestion.model_validate(_record())
    material_root = tmp_path / "material"
    for round_id in (0, 1):
        _material(material_root, round_id)
    inference = MTACIFBenchInference(
        material_dir=material_root,
        round_records=[_round_record(0), _round_record(1)],
    )

    async def _garbage(self: Any, **kwargs: Any) -> Any:
        return _judge_sandbox_result("no verdict here")

    monkeypatch.setattr(MTACIFBenchTask, "_run_judge_sandbox", _garbage)

    judgement = await MTACIFBenchTask().judge(
        question, inference, _FakeEvalContext(tmp_path / "out")
    )
    # error set => the framework keeps the inference and re-runs only the judge.
    assert judgement.error is not None
    assert judgement.instruction_following_score == 0.0
    assert all(item.parse_failed for item in judgement.instruction_following_checks)


@pytest.mark.asyncio
async def test_judge_keeps_sibling_verdicts_when_one_round_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raise in one round must not discard the verdicts its siblings earned.

    Rounds are judged concurrently, so a bare gather would let one exception
    take down the whole question's judgement. The raising round degrades to
    parse_failed, which fails the question with an error so it re-judges.
    """
    question = MTACIFBenchQuestion.model_validate(_record())
    material_root = tmp_path / "material"
    for round_id in (0, 1):
        _material(material_root, round_id)
    inference = MTACIFBenchInference(
        material_dir=material_root,
        round_records=[_round_record(0), _round_record(1)],
    )

    async def _round_1_raises(self: Any, **kwargs: Any) -> Any:
        if kwargs["round_spec"].round_id == 1:
            raise RuntimeError("judge container vanished")
        return IFRoundResult(round_id=0, passed=True)

    monkeypatch.setattr(MTACIFBenchTask, "_judge_round", _round_1_raises)

    judgement = await MTACIFBenchTask().judge(question, inference, _judging_ctx(tmp_path))

    # Round 0's verdict survives; round 1 is recorded as unresolved rather than
    # silently dropped or scored as a failure the model caused.
    by_id = {item.round_id: item for item in judgement.instruction_following_checks}
    assert sorted(by_id) == [0, 1]
    assert by_id[0].passed is True
    assert by_id[0].parse_failed is False
    assert by_id[1].parse_failed is True
    assert "RuntimeError" in by_id[1].symptoms
    # error set => re-judge, instead of scoring a partial judgement as a pass.
    assert judgement.error is not None
    assert judgement.instruction_following_score == 0.0


@pytest.mark.asyncio
async def test_judge_propagates_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CancelledError must not be swallowed by the per-round exception guard.

    return_exceptions=True captures it like any other exception; turning it into
    a parse_failed round would defeat an outer timeout or shutdown.
    """
    question = MTACIFBenchQuestion.model_validate(_record())
    material_root = tmp_path / "material"
    for round_id in (0, 1):
        _material(material_root, round_id)
    inference = MTACIFBenchInference(
        material_dir=material_root,
        round_records=[_round_record(0), _round_record(1)],
    )

    async def _cancelled(self: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(MTACIFBenchTask, "_judge_round", _cancelled)

    with pytest.raises(asyncio.CancelledError):
        await MTACIFBenchTask().judge(question, inference, _judging_ctx(tmp_path))


@pytest.mark.asyncio
async def test_judge_round_ignores_stale_trace_from_earlier_rejudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-judge must not pick up the previous run's trace as its own output.

    Each attempt within one judge() call gets a fresh path, but a re-judge of a
    round that previously failed to parse lands on the same attempt_N — and
    collect_judge_candidates scans attempt_dir/"traces".
    """
    question = MTACIFBenchQuestion.model_validate(_record())
    material = _material_dir(tmp_path, "ok")
    eval_dir = tmp_path / "out" / "eval"

    # A stale trace left by an earlier judge run, holding a parseable verdict.
    stale_verdict = _block(1, "回复以喵开头", PASS_CONCLUSION)
    stale_traces = eval_dir / "instruction_following" / "round_0" / "attempt_0" / "traces"
    stale_traces.mkdir(parents=True)
    (stale_traces / "old.jsonl").write_text(
        json.dumps({"type": "result", "result": stale_verdict}) + "\n",
        encoding="utf-8",
    )

    # This run's judge produces nothing parseable and writes no trace of its own.
    async def _no_trace(self: Any, **kwargs: Any) -> Any:
        return _judge_sandbox_result("")

    monkeypatch.setattr(MTACIFBenchTask, "_run_judge_sandbox", _no_trace)

    result = await MTACIFBenchTask()._judge_round(
        question=question,
        ctx=_FakeEvalContext(tmp_path / "out"),
        round_spec=question.rounds[0],
        material_dir=material,
        eval_dir=eval_dir,
    )

    # Without the rmtree the stale verdict would be adopted as this run's own.
    assert result.parse_failed is True
    assert PASS_CONCLUSION not in result.raw_output_excerpt
    assert not stale_traces.joinpath("old.jsonl").exists()


@requires_validation_deps
@pytest.mark.asyncio
async def test_run_validation_codes_degrades_when_verdict_is_not_an_object(
    tmp_path: Path,
) -> None:
    """A checker whose last stdout line is non-object JSON must reach the judge.

    run_validation_code json.loads()es the last stdout line and then calls
    ``verdict.get("ok")``; valid JSON that is not an object raises AttributeError
    from inside it, past its own error handling. Unguarded that escapes the
    gather and takes down the whole question's judgement.
    """
    question = MTACIFBenchQuestion.model_validate(
        _record(
            rounds=[
                {
                    "round_id": 0,
                    "prompt": "p",
                    "instruction_following_checklist": [
                        {
                            "constraint": "回复以喵开头",
                            # The driver prints its verdict object, then this
                            # atexit hook prints valid-but-not-an-object JSON
                            # after it — so the last stdout line is a list.
                            "validation_code": (
                                "import atexit\n"
                                "atexit.register(lambda: print('[1, 2]'))\n"
                                "def check(response, workspace_path):\n"
                                "    return True\n"
                            ),
                        },
                    ],
                }
            ]
        )
    )
    (tmp_path / "snapshot").mkdir()

    direct, fallback_indices = await MTACIFBenchTask()._run_validation_codes(
        question=question,
        round_id=0,
        checklist=question.checklist_for(0),
        response="喵～",
        snapshot_dir=tmp_path / "snapshot",
        log_tag="test|model",
    )

    # Degraded to the judge, exactly as every other unknown-verdict path does.
    assert direct == {}
    assert fallback_indices == [0]


def test_judge_prompt_carries_context_response_and_checklist() -> None:
    """The judge prompt is the WBS template, filled in the WBS order."""
    question = MTACIFBenchQuestion.model_validate(_record())
    response = "喵～ 见 ```js\nconst a = 1\n``` 汪～"
    prompt = MTACIFBenchTask()._build_judge_prompt(
        question, _checklist("回复以喵开头", "禁止 ESLint"), "[]", response
    )
    assert response in prompt
    assert "## 人工智能助手的操作流程" in prompt
    assert "## 人工智能助手的最终回复" in prompt
    # The checklist is the last section, numbered from 1.
    assert prompt.rstrip().endswith("[要求1]：回复以喵开头\n[要求2]：禁止 ESLint")
    # The operation flow precedes the final reply, which precedes the checklist.
    assert prompt.index("## 人工智能助手的操作流程") < prompt.index("## 人工智能助手的最终回复")
    assert prompt.index("## 人工智能助手的最终回复") < prompt.index("## 要求列表")


# ---------------------------------------------------------------------------
# Multi-turn session continuity (core contract this benchmark depends on)
# ---------------------------------------------------------------------------
class _RecordingSandbox:
    """Minimal Sandbox stand-in that records the commands an agent issues."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.session_id = "sid-1"
        self.commands: list[str] = []
        self.env_vars: dict[str, str] = {}

    async def write_file(self, path: str, content: str) -> None:
        return None

    async def exec_cmd(self, cmd: str, timeout_sec: int | None = None) -> ExecResult:
        self.commands.append(cmd)
        return ExecResult(stdout="done", stderr="", exit_code=0)


def _claude_agent() -> ClaudeCodeAgent:
    return ClaudeCodeAgent(
        agent_config=AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent"),
        model_config=ModelConfig(base_url="https://example.test", api_key="k"),
    )


@pytest.mark.asyncio
async def test_claude_code_resumes_a_reused_session() -> None:
    from agent_probe.core.sandbox import SandboxSpec

    spec = SandboxSpec(image="img", keep_session=True, workspace="/workspace")
    sb = _RecordingSandbox(spec)
    agent = _claude_agent()

    await agent.run_prompt(sb, "round 0")
    await agent.run_prompt(sb, "round 1")

    claude_cmds = [cmd for cmd in sb.commands if "claude -p" in cmd]
    assert "--session-id sid-1" in claude_cmds[0]
    assert "--resume" not in claude_cmds[0]
    # Round 2 must continue the same conversation, not start a new one.
    assert "--resume sid-1" in claude_cmds[1]
    assert "--session-id" not in claude_cmds[1]
    assert any("pkill" in cmd for cmd in sb.commands)


@pytest.mark.asyncio
async def test_claude_code_passes_append_system_prompt() -> None:
    from agent_probe.core.sandbox import SandboxSpec

    spec = SandboxSpec(image="img", append_system_prompt="每轮回复以喵开头", workspace="/workspace")
    sb = _RecordingSandbox(spec)
    await _claude_agent().run_prompt(sb, "go")
    claude_cmd = next(cmd for cmd in sb.commands if "claude -p" in cmd)
    assert "--append-system-prompt '每轮回复以喵开头'" in claude_cmd


@pytest.mark.asyncio
async def test_claude_code_omits_system_prompt_flag_when_unset() -> None:
    from agent_probe.core.sandbox import SandboxSpec

    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    await _claude_agent().run_prompt(sb, "go")
    claude_cmd = next(cmd for cmd in sb.commands if "claude -p" in cmd)
    assert "--append-system-prompt" not in claude_cmd


# ---------------------------------------------------------------------------
# Dataset converter
# ---------------------------------------------------------------------------
def _upstream_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_id": 1,
        "system_prompt": "以喵开头",
        "system_prompt_checklist": [{"约束内容": "以喵开头", "validation_code": "CODE_A"}],
        "rounds": [
            {
                "round_id": 0,
                "instruction": "build a table",
                "instruction_following_checklist": [
                    {
                        "约束内容": "以喵开头",
                        "validation_code": "CODE_A",
                        "tag": ["内容", "人设"],
                        "main_id": 9,
                        "type_id": 0,
                    },
                    {"约束内容": "全部用中文", "validation_code": ""},
                ],
            }
        ],
        "instruction_following_validation_codes": [["CODE_A", ""]],
    }
    record.update(overrides)
    return record


def _run_converter(
    tmp_path: Path, records: list[dict[str, Any]]
) -> subprocess.CompletedProcess[str]:
    src = tmp_path / "src.jsonl"
    src.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--src",
            str(src),
            "--out",
            str(tmp_path / "questions.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


def test_converter_produces_strict_questions(tmp_path: Path) -> None:
    completed = _run_converter(tmp_path, [_upstream_record()])
    assert completed.returncode == 0, completed.stderr

    rows = (tmp_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    question = MTACIFBenchQuestion.model_validate_json(rows[0])
    assert question.task_id == 1
    # The runtime model provides the public inference fallback. The judge image
    # is deployment-specific: empty unless --judge-docker / MTACIF_JUDGE_IMAGE
    # supplies one, so the benchmark carries no private registry path.
    assert question.docker
    assert question.judge_docker == ""
    checklist = question.checklist_for(0)
    assert [item.constraint for item in checklist] == ["以喵开头", "全部用中文"]
    assert checklist[0].validation_code == "CODE_A"
    assert checklist[0].tags == ["内容", "人设"]
    assert checklist[1].validation_code == ""
    # The old field is renamed and the repository checklist remains available.
    assert "system_prompt_checklist" not in json.loads(rows[0])
    assert "repository_policy_checklist" in json.loads(rows[0])


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        pytest.param(
            lambda r: r.update({"instruction_following_validation_codes": [["CODE_A"]]}),
            "code_count_mismatch",
            id="code_count_mismatch",
        ),
        pytest.param(
            lambda r: r.update({"instruction_following_validation_codes": [["OTHER", ""]]}),
            "inline_disagrees",
            id="inline_disagrees",
        ),
        pytest.param(
            lambda r: r["rounds"].append(dict(r["rounds"][0])),
            "duplicate_round_id",
            id="duplicate_round_id",
        ),
        pytest.param(lambda r: r.update({"task_id": "../evil"}), "bad_task_id", id="bad_task_id"),
        pytest.param(lambda r: r.update({"rounds": []}), "no_rounds", id="no_rounds"),
    ],
)
def test_converter_fails_hard_on_inconsistent_input(
    tmp_path: Path, mutate: Any, reason: str
) -> None:
    record = _upstream_record()
    mutate(record)
    completed = _run_converter(tmp_path, [record])
    assert completed.returncode != 0, f"{reason} should not convert silently"


def test_converter_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    completed = _run_converter(tmp_path, [_upstream_record(), _upstream_record()])
    assert completed.returncode != 0


def test_split_sentences_handles_cjk_and_terminator_runs() -> None:
    from benchmarks.mtacifbench.utils import split_sentences

    assert split_sentences("我完成了表格。接下来做筛选！还有问题吗？") == [
        "我完成了表格。",
        "接下来做筛选！",
        "还有问题吗？",
    ]
    # pysbd treats a "……" run as intra-sentence, so no split happens here.
    assert split_sentences("省略号……然后继续。") == ["省略号……然后继续。"]
    assert split_sentences("第一句。\n\n第二句！") == ["第一句。", "第二句！"]
    assert split_sentences("没有终止符") == ["没有终止符"]
    assert split_sentences("") == []


@pytest.mark.asyncio
async def test_claude_code_feeds_prompt_on_stdin() -> None:
    """A judge prompt with a full round transcript exceeds argv's 128 KiB cap."""
    from agent_probe.core.sandbox import SandboxSpec

    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    await _claude_agent().run_prompt(sb, "x" * 200_000)
    claude_cmd = next(cmd for cmd in sb.commands if "claude -p" in cmd)
    assert "| claude -p" in claude_cmd
    # The prompt must not be inlined into argv.
    assert "$(cat" not in claude_cmd
    assert "x" * 100 not in claude_cmd


def test_round_response_never_inherits_a_previous_round_reply() -> None:
    """A silent round must score as silent, not as the previous round's reply."""
    empty_slice = "[]"
    silent = ExecResult(stdout="", stderr="", exit_code=0)
    assert MTACIFBenchTask._round_response(empty_slice, silent) == ""

    # Its own slice wins over stdout noise.
    own_slice = json.dumps(
        [{"role": "assistant", "content": [{"type": "text", "text": "喵～ 本轮完成"}]}]
    )
    noisy = ExecResult(stdout="npm warn deprecated\n喵～ 本轮完成", stderr="", exit_code=0)
    assert MTACIFBenchTask._round_response(own_slice, noisy) == "喵～ 本轮完成"

    # Slice empty but the CLI printed the reply: fall back to stdout.
    assert MTACIFBenchTask._round_response(empty_slice, ExecResult("汪～ ok", "", 0)) == "汪～ ok"


def test_parse_jsonl_result_supports_opencode_text_events() -> None:
    content = "\n".join(
        [
            json.dumps({"type": "step_start"}),
            json.dumps({"type": "text", "part": {"text": "done"}}),
            json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
        ]
    )
    assert parse_jsonl_result(content) == "done"


def test_opencode_output_tracks_terminal_text_and_tool_use() -> None:
    complete = "\n".join(
        [
            json.dumps({"type": "text", "part": {"text": "finished"}}),
            json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
        ]
    )
    inspection = inspect_opencode_jsonl(complete)
    assert inspection["has_assistant_text"] is True
    assert inspection["has_terminal_event"] is True
    assert inspection["final_step_has_tool_use"] is False
    assert extract_opencode_final_text(complete) == "finished"

    incomplete = json.dumps(
        {
            "type": "tool_use",
            "part": {"id": "tool-1", "tool": "bash", "state": {"status": "running"}},
        }
    )
    assert inspect_opencode_jsonl(incomplete)["has_tool_use"] is True


@pytest.mark.asyncio
async def test_opencode_incomplete_response_is_an_error(tmp_path: Path) -> None:
    class _IncompleteOpenCodeAgent(OpenCodeAgent):
        async def _run_once(
            self,
            sb: Any,
            prompt: str,
            *,
            continue_session: bool,
            timeout_sec: int,
        ) -> ExecResult:
            del sb, prompt, continue_session, timeout_sec
            return ExecResult(
                stdout=json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "id": "tool-1",
                            "tool": "bash",
                            "state": {"status": "running"},
                        },
                    }
                ),
                stderr="",
                exit_code=0,
            )

    class _Spec:
        agent_timeout_sec = 30

    class _Sandbox:
        session_id = "sid-opencode"
        spec = _Spec()

    agent = _IncompleteOpenCodeAgent(
        agent_config=AgentConfig(type="agent_probe.agents.opencode.OpenCodeAgent"),
        model_config=ModelConfig(base_url="https://example.test", api_key="k"),
    )
    sandbox = _Sandbox()
    result = await agent.run_prompt(sandbox, "go")  # type: ignore[arg-type]
    last_assistant = await agent.collect_last_assistant(  # type: ignore[arg-type]
        sandbox, tmp_path
    )

    assert result.exit_code == 0
    assert last_assistant is not None
    assert last_assistant.stop_reason == "error"
    assert last_assistant.is_complete_response is False
    assert "without a complete final assistant response" in (last_assistant.error_message or "")


def _agent_with_thinking(level: str) -> ClaudeCodeAgent:
    return ClaudeCodeAgent(
        agent_config=AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent"),
        model_config=ModelConfig(base_url="https://example.test", api_key="k", thinking=level),
    )


@pytest.mark.asyncio
async def test_claude_code_passes_thinking_effort() -> None:
    from agent_probe.core.sandbox import SandboxSpec

    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    await _agent_with_thinking("max").run_prompt(sb, "go")
    cmd = next(c for c in sb.commands if "claude -p" in c)
    assert "--thinking adaptive --effort max" in cmd


@pytest.mark.asyncio
async def test_thinking_off_emits_no_flag() -> None:
    """The default must not change behaviour for benchmarks that never set it."""
    from agent_probe.core.sandbox import SandboxSpec

    sb = _RecordingSandbox(SandboxSpec(image="img", workspace="/workspace"))
    await _agent_with_thinking("off").run_prompt(sb, "go")
    cmd = next(c for c in sb.commands if "claude -p" in c)
    assert "--thinking" not in cmd and "--effort" not in cmd


def test_invalid_thinking_level_fails_fast() -> None:
    with pytest.raises(ValueError, match="thinking"):
        _agent_with_thinking("turbo")


def test_converter_honours_explicit_judge_docker(tmp_path: Path) -> None:
    """The judge image must come from config, not a baked-in registry path."""
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps(_upstream_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "questions.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--src",
            str(src),
            "--out",
            str(out),
            "--judge-docker",
            "registry.example.test/judge:v1",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    question = MTACIFBenchQuestion.model_validate_json(
        out.read_text(encoding="utf-8").splitlines()[0]
    )
    assert question.judge_docker == "registry.example.test/judge:v1"


@requires_validation_deps
def test_validation_code_accepts_a_relative_workspace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checker runs from a scratch cwd, so the workspace must be absolutised.

    Experiment configs default to a relative output_dir, so the snapshot path
    reaching the validator is relative too. Left as-is it fails every
    `os.path.exists(workspace_path)` guard and marks the constraint unmet no
    matter what the model wrote.
    """
    workspace = tmp_path / "out" / "snap"
    workspace.mkdir(parents=True)
    (workspace / "app.js").write_text("// -*- coding: utf-8 -*-\n", encoding="utf-8")
    code = (
        "import os\n"
        "def check(response, workspace_path):\n"
        "    if not os.path.exists(workspace_path):\n"
        "        return False\n"
        "    for root, _dirs, files in os.walk(workspace_path):\n"
        "        for name in files:\n"
        "            with open(os.path.join(root, name), encoding='utf-8') as handle:\n"
        "                if '-*- coding: utf-8 -*-' not in handle.readline():\n"
        "                    return False\n"
        "    return True\n"
    )
    monkeypatch.chdir(tmp_path)
    assert run_validation_code(code, "", Path("out/snap"), timeout=60) is True
    assert run_validation_code(code, "", workspace, timeout=60) is True
