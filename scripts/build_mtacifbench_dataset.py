#!/usr/bin/env python3
"""Convert chatglm-eval MTACIFBench questions.jsonl to the AgentProbe format.

The output matches ``benchmarks.mtacifbench.models.MTACIFBenchQuestion`` exactly.
Every tolerance for the upstream shape lives here, so the runtime models stay
strict:

* ``rounds[*].instruction`` becomes ``rounds[*].prompt``.
* Each constraint's ``validation_code`` is kept **on the constraint**. Upstream
  also carries a parallel ``instruction_following_validation_codes[i][j]`` array
  aligned by index; this script asserts the two agree and then drops the array,
  which removes the whole class of index-misalignment bugs.
* ``system_prompt_checklist`` is dropped: it is a prefix of each round's own
  checklist, and nothing scores it separately.
* Container images are absent upstream (hardcoded defaults) and are materialised
  into the output so the dataset is self-describing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_INFER_DOCKER = "alexgshaw/break-filter-js-from-html:20251031"
# Deployment-specific: pass --judge-docker or set MTACIF_JUDGE_IMAGE.
DEFAULT_JUDGE_DOCKER = os.environ.get("MTACIF_JUDGE_IMAGE", "")


def as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def convert_constraint(item: Any, where: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{where}: constraint is not an object")
    constraint = str(item.get("约束内容") or item.get("constraint") or "").strip()
    if not constraint:
        raise ValueError(f"{where}: constraint text is empty")
    return {
        "constraint": constraint,
        "validation_code": str(item.get("validation_code") or ""),
        "tags": as_list_of_str(item.get("tag") or item.get("tags")),
        "main_id": item.get("main_id"),
        "type_id": item.get("type_id"),
    }


def convert_rounds(record: dict[str, Any], task_id: str) -> list[dict[str, Any]]:
    raw_rounds = record.get("rounds")
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise ValueError(f"{task_id}: rounds is missing or empty")

    parallel_codes = record.get("instruction_following_validation_codes")
    if parallel_codes is not None:
        if not isinstance(parallel_codes, list) or len(parallel_codes) != len(raw_rounds):
            raise ValueError(
                f"{task_id}: instruction_following_validation_codes has "
                f"{len(parallel_codes) if isinstance(parallel_codes, list) else 'non-list'} "
                f"entries for {len(raw_rounds)} rounds"
            )

    rounds: list[dict[str, Any]] = []
    seen_round_ids: set[int] = set()
    for index, raw_round in enumerate(raw_rounds):
        if not isinstance(raw_round, dict):
            raise ValueError(f"{task_id}: round at index {index} is not an object")
        round_id = int(raw_round.get("round_id", index))
        if round_id in seen_round_ids:
            raise ValueError(f"{task_id}: duplicate round_id {round_id}")
        seen_round_ids.add(round_id)

        prompt = str(raw_round.get("instruction") or raw_round.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{task_id}: round {round_id} has an empty instruction")

        raw_checklist = raw_round.get("instruction_following_checklist") or []
        if not isinstance(raw_checklist, list):
            raise ValueError(f"{task_id}: round {round_id} checklist is not a list")
        checklist = [
            convert_constraint(item, f"{task_id} round {round_id} #{position + 1}")
            for position, item in enumerate(raw_checklist)
        ]

        if parallel_codes is not None:
            codes = parallel_codes[index]
            if not isinstance(codes, list) or len(codes) != len(checklist):
                raise ValueError(
                    f"{task_id}: round {round_id} has {len(checklist)} constraints but "
                    f"{len(codes) if isinstance(codes, list) else 'non-list'} validation codes"
                )
            for position, (item, code) in enumerate(zip(checklist, codes, strict=True)):
                if item["validation_code"] != str(code or ""):
                    raise ValueError(
                        f"{task_id}: round {round_id} #{position + 1} inline "
                        "validation_code disagrees with "
                        "instruction_following_validation_codes"
                    )

        rounds.append(
            {
                "round_id": round_id,
                "prompt": prompt,
                "instruction_following_checklist": checklist,
            }
        )
    return rounds


def convert_record(
    record: dict[str, Any],
    infer_docker: str,
    judge_docker: str,
) -> dict[str, Any]:
    task_id = str(record.get("task_id") or record.get("qid") or "").strip()
    if not task_id or task_id in {".", ".."} or "/" in task_id or "\\" in task_id:
        raise ValueError(f"invalid task_id: {record.get('task_id')!r}")

    return {
        "task_id": task_id,
        "docker": str(record.get("docker") or infer_docker),
        "judge_docker": str(record.get("judge_docker") or judge_docker),
        "workspace_dir": str(record.get("workspace_dir") or "/workspace"),
        "description": str(record.get("description") or ""),
        "system_prompt": str(record.get("system_prompt") or "").strip(),
        "rounds": convert_rounds(record, task_id),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="upstream questions.jsonl")
    parser.add_argument(
        "--out",
        default="benchmarks/mtacifbench/data/questions.jsonl",
        help="output questions.jsonl",
    )
    parser.add_argument("--infer-docker", default=DEFAULT_INFER_DOCKER)
    parser.add_argument("--judge-docker", default=DEFAULT_JUDGE_DOCKER)
    parser.add_argument(
        "--limit", type=int, default=0, help="keep only the first N tasks (0 = all)"
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated task_id allowlist",
    )
    args = parser.parse_args()

    allow = {item.strip() for item in args.only.split(",") if item.strip()}
    src_path = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(src_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{src_path}:{line_no} is not valid JSON: {exc}") from exc
        question = convert_record(record, args.infer_docker, args.judge_docker)
        if allow and question["task_id"] not in allow:
            continue
        if question["task_id"] in seen_ids:
            raise ValueError(f"duplicate task_id across lines: {question['task_id']}")
        seen_ids.add(question["task_id"])
        converted.append(question)
        if args.limit and len(converted) >= args.limit:
            break

    with out_path.open("w", encoding="utf-8") as handle:
        for question in converted:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")

    rounds = sum(len(item["rounds"]) for item in converted)
    constraints = sum(
        len(round_item["instruction_following_checklist"])
        for item in converted
        for round_item in item["rounds"]
    )
    with_code = sum(
        1
        for item in converted
        for round_item in item["rounds"]
        for constraint in round_item["instruction_following_checklist"]
        if constraint["validation_code"]
    )
    print(
        f"wrote {len(converted)} tasks / {rounds} rounds / {constraints} constraints "
        f"({with_code} with validation code) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
