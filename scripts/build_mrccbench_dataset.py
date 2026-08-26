#!/usr/bin/env python3
"""Convert chatglm-eval MRCCBench questions.jsonl to AgentProbe format.

The output is intentionally strict and matches
``benchmarks.mrccbench.models.MRCCBenchQuestion``. Raw compatibility with
``contexts`` / ``dependency`` lives here instead of in runtime models.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def default_judge_docker() -> str:
    # Deployment-specific: pass --judge-docker or set MRCC_JUDGE_IMAGE.
    return os.environ.get("MRCC_JUDGE_IMAGE", "")


def as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def convert_rounds(record: dict[str, Any]) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for index, item in enumerate(record.get("contexts") or []):
        if not isinstance(item, dict):
            raise ValueError(f"context at index {index} is not an object")
        rounds.append(
            {
                "round_id": int(item.get("round_id", index)),
                "prompt": str(item.get("prompt") or "").strip(),
                "scenario_tags": as_list_of_str(item.get("scenario_tags")),
                "tags": as_list_of_str(item.get("tags")),
            }
        )
    rounds.sort(key=lambda item: item["round_id"])
    return rounds


def convert_checklist(record: dict[str, Any]) -> list[dict[str, Any]]:
    checklist: list[dict[str, Any]] = []
    for index, item in enumerate(record.get("checklist") or []):
        if isinstance(item, str):
            checklist.append({"id": index, "description": item, "weight": 1.0})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"checklist at index {index} is not an object")
        checklist.append(
            {
                "id": item.get("id", item.get("checklist_id", index)),
                "description": str(item.get("description") or ""),
                "weight": float(item.get("weight", 1.0) or 1.0),
            }
        )
    return checklist


def convert_dependencies(record: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    for index, item in enumerate(record.get("dependency") or []):
        if not isinstance(item, dict):
            raise ValueError(f"dependency at index {index} is not an object")
        critical = item.get("critical_check")
        converted_critical = None
        if isinstance(critical, dict):
            converted_critical = {
                "description": str(critical.get("description") or ""),
                "depended_by_rounds": [
                    int(round_id) for round_id in critical.get("depended_by_rounds", [])
                ],
            }
        dependencies.append(
            {
                "round_id": int(item.get("round_id", 0)),
                "critical_check": converted_critical,
            }
        )
    dependencies.sort(key=lambda item: item["round_id"])
    return dependencies


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    qid = str(record.get("qid") or record.get("task_id") or "").strip()
    if not qid:
        raise ValueError("missing qid/task_id")
    rounds = convert_rounds(record)
    if not rounds:
        raise ValueError(f"{qid}: no contexts/rounds found")
    checklist = convert_checklist(record)
    if not checklist:
        raise ValueError(f"{qid}: no checklist found")

    output: dict[str, Any] = {
        "task_id": qid,
        "docker": str(record.get("docker") or "alexgshaw/break-filter-js-from-html:20251031"),
        "workspace_dir": str(record.get("workspace_dir") or "/workspace"),
        "description": str(record.get("description") or ""),
        "rounds": rounds,
        "checklist": checklist,
        "dependencies": convert_dependencies(record),
        "categories": as_list_of_str(record.get("categories"))
        or ["mrccbench", "agent", "multi_round"],
        "test_mode": str(record.get("test_mode") or "http"),
        "task_description_for_judge": record.get("task_description_for_judge")
        or record.get("description")
        or "\n".join(f"第{item['round_id']}轮：{item['prompt']}" for item in rounds),
        "judge_docker": str(record.get("judge_docker") or default_judge_docker()),
        "http_port": int(record.get("http_port", 5173) or 5173),
        "http_build_timeout": int(record.get("http_build_timeout", 900) or 900),
        "eval_timeout": int(record.get("eval_timeout", 7200) or 7200),
        "eval_concurrent": int(record.get("eval_concurrent", 5) or 5),
        "eval_one_retry_max": int(record.get("eval_one_retry_max", 1) or 1),
        "judge_one_retry_max": int(record.get("judge_one_retry_max", 2) or 2),
        "max_repair_attempts": int(record.get("max_repair_attempts", 2) or 2),
        "repair": bool(record.get("repair", True)),
        "dependency_failure_skips_downstream": bool(
            record.get("dependency_failure_skips_downstream", False)
        ),
        "save_intermediate_workspace_tar": bool(
            record.get("save_intermediate_workspace_tar", False)
        ),
    }
    return output


def build(input_jsonl: Path, output_dir: Path, limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions.jsonl"
    converted: list[dict[str, Any]] = []
    skipped: list[tuple[int, str]] = []

    with input_jsonl.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(converted) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                converted.append(convert_record(json.loads(line)))
            except Exception as exc:
                skipped.append((line_no, str(exc)))

    with questions_path.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    images = sorted(
        {str(item["docker"]) for item in converted}
        | {str(item["judge_docker"]) for item in converted}
    )
    (output_dir / "images.txt").write_text(
        "\n".join(images) + ("\n" if images else ""),
        encoding="utf-8",
    )

    print(f"Wrote {len(converted)} questions -> {questions_path}")
    print(f"Wrote {len(images)} image refs -> {output_dir / 'images.txt'}")
    if skipped:
        print(f"Skipped {len(skipped)} records")
        for line_no, reason in skipped[:20]:
            print(f"  line {line_no}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "benchmarks/mrccbench/data",
        type=Path,
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        raise SystemExit(f"input jsonl not found: {args.input_jsonl}")
    build(args.input_jsonl, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
