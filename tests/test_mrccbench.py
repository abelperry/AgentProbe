"""Tests for MRCCBench models, parsing, conversion, and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.mrccbench import MRCCBenchTask as ExportedTask
from benchmarks.mrccbench.frontend import calculate_weighted_score, parse_mrcc_judge_score
from benchmarks.mrccbench.models import (
    MRCCBenchInference,
    MRCCBenchJudgement,
    MRCCBenchQuestion,
    MRCCCheckResult,
    RoundRecord,
)
from benchmarks.mrccbench.score_extract import (
    EXTRACT_STATUS_FALLBACK_LLM,
    EXTRACT_STATUS_OK,
    resolve_check_score,
)
from benchmarks.mrccbench.task import MRCCBenchTask

from agent_probe.config import JudgeConfig, ModelConfig
from agent_probe.core.models import Error, resolve_types
from agent_probe.model_clients import ModelResponse


def _record() -> dict:
    return {
        "task_id": "mrcc_001",
        "docker": "infer:latest",
        "workspace_dir": "/workspace",
        "description": "multi-round UI",
        "rounds": [
            {"round_id": 0, "prompt": "create app"},
            {"round_id": 1, "prompt": "add filter"},
        ],
        "checklist": [
            {"id": 0, "description": "renders", "weight": 2},
            {"id": 1, "description": "filters", "weight": 1},
        ],
        "dependencies": [
            {
                "round_id": 0,
                "critical_check": {
                    "description": "app loads",
                    "depended_by_rounds": [1],
                },
            }
        ],
    }


def _check(score: float | None, reason: str = "", weight: float = 1.0) -> MRCCCheckResult:
    return MRCCCheckResult(id=0, description="c", score=score, reason=reason, weight=weight)


def test_question_accepts_prepared_shape() -> None:
    question = MRCCBenchQuestion.model_validate(_record())

    assert question.qid() == "mrcc_001"
    assert question.task_description == "multi-round UI"
    assert question.rounds[1].prompt == "add filter"
    assert question.dependency_map()[0] is not None
    assert question.checklist[0].weight == 2


def test_resolve_types_mrccbench() -> None:
    assert resolve_types(MRCCBenchTask) == (
        MRCCBenchQuestion,
        MRCCBenchInference,
        MRCCBenchJudgement,
    )
    assert ExportedTask is MRCCBenchTask


@pytest.mark.parametrize(
    "output,expected",
    [
        ('MRCCBENCH_VERDICT_JSON:{"score":1.0}', 1.0),
        ('`MRCCBENCH_VERDICT_JSON:{"score":0.5}`', 0.5),
        ("判断结论：该项目不符合要求", 0.0),
        ("**不完全符合 (0.5)**", 0.5),
        ("no verdict", None),
    ],
)
def test_parse_mrcc_judge_score(output: str, expected: float | None) -> None:
    score, _ = parse_mrcc_judge_score(output)
    assert score == expected


def test_resolve_check_score_reads_assistant_trace(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        '判断结论：该项目不完全符合要求\nMRCCBENCH_VERDICT_JSON:{"score":0.5}'
                    ),
                }
            ]
        },
    }
    (trace_dir / "trace.jsonl").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")

    score, _, status = resolve_check_score(
        check_description="works",
        output_text="",
        trace_dir=trace_dir,
    )

    assert score == 0.5
    assert status == EXTRACT_STATUS_OK


def test_resolve_check_score_uses_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.mrccbench import score_extract

    class FakeClient:
        def complete(self, *args, **kwargs) -> ModelResponse:
            return ModelResponse(content='{"score": 0.5, "confidence": 0.9}', raw={})

    monkeypatch.setattr(
        score_extract,
        "create_model_client",
        lambda *args, **kwargs: FakeClient(),
    )

    score, _, status = resolve_check_score(
        check_description="works",
        output_text="Final verdict: the requirement is only partially satisfied.",
        extract_api=ModelConfig(base_url="http://example.com/v1", api_key="k", model_name="m"),
    )

    assert score == 0.5
    assert status == EXTRACT_STATUS_FALLBACK_LLM


def test_judge_config_extract_api_uses_model_config(tmp_path: Path) -> None:
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(
        """
model:
  base_url: "https://open.bigmodel.cn/api/anthropic/"
  api_key: "judge-key"
  model_name: "glm-5.1"
agent:
  type: "agent_probe.agents.claude_code.ClaudeCodeAgent"
extract_api:
  base_url: "https://gateway.example.test/v1"
  api_key: "extract-key"
  model_name: "deepseek-v4-pro"
  format: "openai"
  timeout: 120
  max_tokens: 1024
""",
        encoding="utf-8",
    )

    config = JudgeConfig.from_yaml(config_path)

    assert isinstance(config.extract_api, ModelConfig)
    assert config.extract_api.model_name == "deepseek-v4-pro"
    assert config.extract_api.max_tokens == 1024


def test_calculate_weighted_score() -> None:
    assert calculate_weighted_score(
        [_check(1.0, weight=2), _check(0.5, weight=1)]
    ) == pytest.approx(2.5 / 3)
    assert calculate_weighted_score([_check(None)]) is None


def test_collect_metrics_counts_errors_in_denominator() -> None:
    task = MRCCBenchTask()
    judgements = [
        MRCCBenchJudgement(
            weighted_score=1.0,
            check_results=[_check(1.0), _check(1.0)],
            total_rounds=2,
            round_summaries=[{"kind": "main"}, {"kind": "main"}],
            repair_summary={"total_repairs": 0},
        ),
        MRCCBenchJudgement(
            weighted_score=0.5,
            check_results=[_check(1.0), _check(0.0, reason="Build failed: syntax")],
            total_rounds=2,
            round_summaries=[{"kind": "main"}, {"kind": "main"}],
            repair_summary={"total_repairs": 1},
        ),
        MRCCBenchJudgement(
            weighted_score=None,
            check_results=[_check(None)],
            total_rounds=2,
            round_summaries=[{"kind": "main"}],
            error=Error(code=-1, message="retry"),
        ),
    ]

    scores, success_count = task.collect_metrics(judgements)

    assert success_count == 2
    assert scores["num_total"] == 3
    assert scores["num_success"] == 2
    assert scores["num_main_complete"] == 2
    assert scores["average"] == pytest.approx((1.0 + 0.5 + 0.0) / 3 * 100)
    assert scores["adj-average"] == pytest.approx((1.0 + 0.5 * 0.9 + 0.0) / 3 * 100)
    assert scores["ISR"] == pytest.approx(1 / 3 * 100)
    assert scores["FPSR"] == pytest.approx(1 / 3 * 100)
    assert scores["CSR"] == pytest.approx(3 / 5 * 100)
    assert scores["BSR"] == pytest.approx(2 / 3 * 100)


def test_converter_outputs_agentprobe_shape(tmp_path: Path) -> None:
    from scripts.build_mrccbench_dataset import build

    source = tmp_path / "raw.jsonl"
    output_dir = tmp_path / "out"
    raw = {
        "qid": "raw_001",
        "description": "desc",
        "contexts": [{"round_id": 0, "prompt": "do it"}],
        "checklist": [{"checklist_id": 7, "description": "works", "weight": 3}],
        "dependency": [
            {
                "round_id": 0,
                "critical_check": {
                    "description": "loads",
                    "depended_by_rounds": [1],
                },
            }
        ],
    }
    source.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")

    build(source, output_dir, None)

    converted = json.loads((output_dir / "questions.jsonl").read_text(encoding="utf-8"))
    question = MRCCBenchQuestion.model_validate(converted)
    assert question.qid() == "raw_001"
    assert question.rounds[0].prompt == "do it"
    assert question.checklist[0].id == 7


def test_round_summary_links_dependency_results() -> None:
    inference = MRCCBenchInference(
        round_records=[RoundRecord(kind="main", round_index=0, round_id=0, result_excerpt="done")],
        dependency_checks=[],
    )

    summary = MRCCBenchTask._build_round_summaries(inference)

    assert summary == [
        {
            "kind": "main",
            "round_index": 0,
            "round_id": 0,
            "attempt": 0,
            "result_excerpt": "done",
            "check_excerpt": "",
            "dependency_passed": None,
            "scheduling_note": "",
            "skipped_due_to_failed_rounds": [],
            "trace_ref": None,
        }
    ]
