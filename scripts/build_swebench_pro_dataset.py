#!/usr/bin/env python3
"""Build AgentProbe SWE-bench Pro questions.jsonl from official sources.

Inputs:
  1. HF dataset ScaleAI/SWE-bench_Pro parquet.
  2. A local clone of https://github.com/scaleapi/SWE-bench_Pro-os.

The output JSONL is the clean AgentProbe shape consumed by
benchmarks.swebench_pro.models.SWEBenchProQuestion: list fields are real lists,
the prompt is pre-rendered, and per-instance run_script/parser assets are
embedded.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

HF_PARQUET_URL = (
    "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/"
    "resolve/main/data/test-00000-of-00001.parquet"
)


def maybe_json_decode(value: Any) -> str:
    if not isinstance(value, str):
        return "" if value is None else str(value)
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, str):
            return decoded
    return value


def parse_list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def create_prompt(problem_statement: str, requirements: str, interface: str) -> str:
    return f"""{problem_statement}

Requirements:
{requirements}

New interfaces introduced:
{interface}"""


def extract_env_exports(repo_dir: Path, instance_id: str) -> str:
    env_cmds: list[str] = []
    for kind in ("base_dockerfile", "instance_dockerfile"):
        dockerfile = repo_dir / "dockerfiles" / kind / instance_id / "Dockerfile"
        if not dockerfile.exists():
            print(f"  WARN missing {kind} for {instance_id}")
            continue
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ENV"):
                env_cmds.append(line.replace("ENV", "export", 1))
    return "\n".join(env_cmds)


def build(parquet: str, repo_dir: Path, output_dir: Path, eval_timeout: int) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required for this conversion script") from exc

    df = pd.read_parquet(parquet).fillna("")
    print(f"Loaded {len(df)} SWE-Pro instances")

    questions: list[dict[str, Any]] = []
    skipped: list[str] = []
    for _, row in df.iterrows():
        iid = str(row["instance_id"])
        run_script_path = repo_dir / "run_scripts" / iid / "run_script.sh"
        parser_path = repo_dir / "run_scripts" / iid / "parser.py"
        if not run_script_path.exists() or not parser_path.exists():
            skipped.append(iid)
            print(f"  SKIP {iid}: run_script.sh/parser.py missing")
            continue

        before_repo_set_cmd = str(row["before_repo_set_cmd"]).strip()
        if not before_repo_set_cmd:
            skipped.append(iid)
            print(f"  SKIP {iid}: empty before_repo_set_cmd")
            continue

        problem_statement = maybe_json_decode(row["problem_statement"])
        requirements = maybe_json_decode(row["requirements"])
        interface = maybe_json_decode(row["interface"])

        questions.append({
            "instance_id": iid,
            "repo": str(row["repo"]),
            "base_commit": str(row["base_commit"]),
            "dockerhub_tag": str(row["dockerhub_tag"]),
            "prompt": create_prompt(problem_statement, requirements, interface),
            "fail_to_pass": parse_list_field(row["fail_to_pass"]),
            "pass_to_pass": parse_list_field(row["pass_to_pass"]),
            "selected_test_files": parse_list_field(row["selected_test_files_to_run"]),
            "eval_cmd": before_repo_set_cmd.splitlines()[-1],
            "run_script": run_script_path.read_text(encoding="utf-8"),
            "parser_py": parser_path.read_text(encoding="utf-8"),
            "env_exports": extract_env_exports(repo_dir, iid),
            "repo_language": str(row["repo_language"]),
            "issue_categories": parse_list_field(row["issue_categories"]),
            "issue_specificity": parse_list_field(row["issue_specificity"]),
            "patch": str(row["patch"]),
            "test_patch": str(row["test_patch"]),
            "eval_timeout": eval_timeout,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    questions_file = output_dir / "questions.jsonl"
    with questions_file.open("w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")

    images = sorted({f"jefzda/sweap-images:{q['dockerhub_tag']}" for q in questions})
    (output_dir / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")

    print(f"Wrote {len(questions)} questions -> {questions_file}")
    print(f"Wrote {len(images)} image refs -> {output_dir / 'images.txt'}")
    if skipped:
        print(f"Skipped {len(skipped)} instances")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet", default=HF_PARQUET_URL)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "benchmarks/swebench_pro/data",
        type=Path,
    )
    parser.add_argument("--eval-timeout", default=3600, type=int)
    args = parser.parse_args()

    if not (args.repo_dir / "run_scripts").exists():
        raise SystemExit(f"run_scripts/ not found under {args.repo_dir}")
    if not (args.repo_dir / "dockerfiles").exists():
        raise SystemExit(f"dockerfiles/ not found under {args.repo_dir}")

    build(args.parquet, args.repo_dir, args.output_dir, args.eval_timeout)


if __name__ == "__main__":
    main()
