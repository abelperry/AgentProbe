#!/usr/bin/env python3
"""Pull AgentProbe benchmark datasets from the Hugging Face Hub into ``benchmarks/<bench>/data``.

Benchmark data (``questions.jsonl``, ``tasks/``, judge configs, ...) is *not* committed to this
repo (see ``.gitignore``); it is published as one Hugging Face dataset repo per benchmark and
downloaded on demand with this script.

Each HF dataset repo maps 1:1 into ``benchmarks/<bench>/data/``.

Examples
--------
Pull every benchmark from an org::

    python scripts/pull_benchmarks.py --org your-org

Pull only a couple::

    python scripts/pull_benchmarks.py --org your-org zbackendbench zfrontendbench

Override a single repo id::

    python scripts/pull_benchmarks.py --repo swebench=someone/swe-verified
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# benchmark name -> default HF repo suffix (joined with --org as "<org>/<suffix>")
BENCHMARKS: dict[str, str] = {
    "swebench": "swebench",
    "swebench_pro": "swebench_pro",
    "terminalbench_v2": "terminalbench_v2",
    "zbackendbench": "zbackendbench",
    "zfrontendbench": "zfrontendbench",
    "zclawbench": "zclawbench",
    "mrccbench": "mrccbench",
    "mtacifbench": "mtacifbench",
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_ids(org: str | None, overrides: dict[str, str]) -> dict[str, str]:
    repos: dict[str, str] = {}
    for bench, suffix in BENCHMARKS.items():
        if bench in overrides:
            repos[bench] = overrides[bench]
        elif org:
            repos[bench] = f"{org}/{suffix}"
    return repos


def pull(bench: str, repo_id: str, *, revision: str, force: bool) -> None:
    from huggingface_hub import snapshot_download

    data_dir = REPO_ROOT / "benchmarks" / bench / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{bench}] downloading {repo_id} -> {data_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(data_dir),
        force_download=force,
        # keep the dataset card / git metadata out of the runtime data dir
        ignore_patterns=["README.md", ".gitattributes"],
    )
    questions = data_dir / "questions.jsonl"
    status = "ok" if questions.exists() else "WARNING: questions.jsonl missing"
    print(f"[{bench}] done ({status})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "benches",
        nargs="*",
        choices=[*BENCHMARKS, []],
        help="benchmarks to pull (default: all)",
    )
    parser.add_argument(
        "--org",
        default=None,
        help="Hugging Face org/user that hosts the dataset repos (e.g. your-org)",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="BENCH=REPO_ID",
        help="override a single repo id, e.g. --repo swebench=someone/swe-verified",
    )
    parser.add_argument("--revision", default="main", help="git revision to download")
    parser.add_argument("--force", action="store_true", help="force re-download")
    args = parser.parse_args()

    overrides: dict[str, str] = {}
    for item in args.repo:
        if "=" not in item:
            parser.error(f"--repo expects BENCH=REPO_ID, got: {item}")
        bench, repo_id = item.split("=", 1)
        if bench not in BENCHMARKS:
            parser.error(f"unknown benchmark in --repo: {bench}")
        overrides[bench] = repo_id

    if not args.org and not overrides:
        parser.error("provide --org and/or --repo BENCH=REPO_ID")

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 2

    repos = resolve_repo_ids(args.org, overrides)
    selected = args.benches or list(BENCHMARKS)
    missing = [b for b in selected if b not in repos]
    if missing:
        parser.error(f"no repo id resolved for: {', '.join(missing)} (pass --org or --repo)")

    for bench in selected:
        pull(bench, repos[bench], revision=args.revision, force=args.force)
    print(f"\nPulled {len(selected)} benchmark(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
