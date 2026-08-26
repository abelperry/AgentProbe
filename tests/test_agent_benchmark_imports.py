"""Import and factory tests for migrated agent benchmarks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarks.swebench_pro.models import (
    SWEBenchProInference,
    SWEBenchProJudgement,
    SWEBenchProQuestion,
)
from benchmarks.swebench_pro.task import SWEBenchProTask
from benchmarks.terminalbench_v2.models import (
    TerminalBenchV2Inference,
    TerminalBenchV2Judgement,
    TerminalBenchV2Question,
)
from benchmarks.terminalbench_v2.task import TerminalBenchV2Task
from benchmarks.zbackendbench.models import (
    ZBackendBenchInference,
    ZBackendBenchJudgement,
    ZBackendBenchQuestion,
)
from benchmarks.zbackendbench.task import ZBackendBenchTask
from benchmarks.zfrontendbench.models import (
    ZFrontendBenchInference,
    ZFrontendBenchJudgement,
    ZFrontendBenchQuestion,
)
from benchmarks.zfrontendbench.task import ZFrontendBenchTask

from agent_probe.config import AgentConfig, DatasetConfig, EvalExperimentConfig, ModelConfig
from agent_probe.core.factory import ExperimentFactory
from agent_probe.core.models import resolve_types


def test_resolve_types_for_migrated_agent_benchmarks() -> None:
    assert resolve_types(TerminalBenchV2Task) == (
        TerminalBenchV2Question,
        TerminalBenchV2Inference,
        TerminalBenchV2Judgement,
    )
    assert resolve_types(ZBackendBenchTask) == (
        ZBackendBenchQuestion,
        ZBackendBenchInference,
        ZBackendBenchJudgement,
    )
    assert resolve_types(ZFrontendBenchTask) == (
        ZFrontendBenchQuestion,
        ZFrontendBenchInference,
        ZFrontendBenchJudgement,
    )
    assert resolve_types(SWEBenchProTask) == (
        SWEBenchProQuestion,
        SWEBenchProInference,
        SWEBenchProJudgement,
    )


def test_experiment_factory_loads_migrated_agent_benchmarks(tmp_path: Path) -> None:
    terminal_data = _write_questions_jsonl(
        tmp_path / "terminal_data",
        [
            {
                "qid": "terminal_001",
                "docker_image": "terminal:latest",
                "workspace_dir": "/workspace",
                "contexts": ["run terminal task"],
            }
        ],
    )
    zbackend_data = _write_questions_jsonl(
        tmp_path / "zbackend_data",
        [
            {
                "qid": "backend_001",
                "docker_image": "backend:latest",
                "workspace_dir": "/workspace",
                "contexts": ["run backend task"],
            }
        ],
    )
    zfrontend_data = _write_questions_jsonl(
        tmp_path / "zfrontend_data",
        [
            {
                "qid": "frontend_001",
                "docker": "frontend:latest",
                "workspace_dir": "/workspace",
                "contexts": ["run frontend task"],
                "checklist": ["renders"],
            }
        ],
    )
    swepro_data = _write_questions_jsonl(
        tmp_path / "swepro_data",
        [
            {
                "instance_id": "instance_repo-abc-vnan",
                "repo": "org/repo",
                "base_commit": "base",
                "dockerhub_tag": "org.repo-instance",
                "prompt": "fix",
                "fail_to_pass": ["new_test"],
                "pass_to_pass": ["old_test"],
                "selected_test_files": ["test.py"],
                "eval_cmd": "git checkout gold -- test.py",
                "run_script": "#!/bin/bash",
                "parser_py": "print(1)",
            }
        ],
    )

    cfg = EvalExperimentConfig(
        name="agent-benchmarks-test",
        concurrency=1,
        output_dir=str(tmp_path / "output"),
        models={"m": ModelConfig(base_url="http://example", api_key="k")},
        agents={"a": AgentConfig(type="agent_probe.agents.claude_code.ClaudeCodeAgent")},
        datasets={
            "terminalbench_v2": DatasetConfig(
                name="terminalbench_v2",
                data_dir=str(terminal_data),
                task_type="benchmarks.terminalbench_v2.task.TerminalBenchV2Task",
            ),
            "zbackendbench": DatasetConfig(
                name="zbackendbench",
                data_dir=str(zbackend_data),
                task_type="benchmarks.zbackendbench.task.ZBackendBenchTask",
            ),
            "zfrontendbench": DatasetConfig(
                name="zfrontendbench",
                data_dir=str(zfrontend_data),
                task_type="benchmarks.zfrontendbench.task.ZFrontendBenchTask",
            ),
            "swebench_pro": DatasetConfig(
                name="swebench_pro",
                data_dir=str(swepro_data),
                task_type="benchmarks.swebench_pro.task.SWEBenchProTask",
            ),
        },
    )

    executor = ExperimentFactory().create(cfg)

    counts = Counter(unit.dataset_name for unit in executor._units)
    assert set(counts) == {
        "terminalbench_v2",
        "zbackendbench",
        "zfrontendbench",
        "swebench_pro",
    }
    assert all(count >= 1 for count in counts.values())


def _write_questions_jsonl(data_dir: Path, rows: list[dict]) -> Path:
    data_dir.mkdir(parents=True)
    jsonl = data_dir / "questions.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return data_dir
