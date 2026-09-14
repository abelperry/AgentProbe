#!/usr/bin/env python3
"""Summarize completed MTAC-IFBench results in the paper-analysis format."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.mtacifbench.models import (  # noqa: E402
    PASS_CONCLUSION,
    MTACIFBenchJudgement,
)

ROUND_BANDS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))

COMPACT_MODEL_STAT_KEYS = (
    "valid_cases",
    "skipped_cases",
    "IF_CSR_all",
    "IF_ISR_all",
)
OPTIONAL_COVERAGE_STAT_KEYS = (
    "expected_cases",
    "not_completed_cases",
)
FUNCTION_MODEL_STAT_KEYS = (
    "FC_CSR",
    "FC_ISR",
    "function_success_cases",
    "function_total_cases",
    "function_score_sum",
    "function_total_checks",
    "BSR",
    "build_success_cases",
    "build_total_cases",
)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _percentage(value: float | None) -> float:
    return value * 100 if value is not None else 0.0


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _constraint_key(entry: Any) -> tuple[Any, ...]:
    if not isinstance(entry, dict):
        return ("raw", _stable_json(entry))
    return ("constraint", entry.get("constraint"))


def _constraint_tag(entry: Any) -> tuple[str, str]:
    tags = entry.get("tags") if isinstance(entry, dict) else None
    if isinstance(tags, list) and len(tags) >= 2:
        primary = tags[0] if isinstance(tags[0], str) and tags[0] else "未识别"
        secondary = tags[1] if isinstance(tags[1], str) and tags[1] else "未识别"
        return primary, f"{primary}/{secondary}"
    if isinstance(tags, list) and len(tags) == 1 and isinstance(tags[0], str) and tags[0]:
        return tags[0], f"{tags[0]}/未识别"
    return "未识别", "未识别/unknown"


def _classify_constraint_sources(
    repository_policy_checklist: list[Any],
    rounds: list[Any],
) -> list[list[str]]:
    """Match the analysis labels: repository_policy/new_added/replaced."""
    repository_policy_keys = {_constraint_key(entry) for entry in repository_policy_checklist}
    previous_user_keys: set[tuple[Any, ...]] = set()
    previous_source_by_key: dict[tuple[Any, ...], str] = {}
    sources_by_round: list[list[str]] = []

    for round_item in rounds:
        raw_checklist = (
            round_item.get("instruction_following_checklist")
            if isinstance(round_item, dict)
            else []
        )
        checklist = raw_checklist if isinstance(raw_checklist, list) else []
        entries: list[tuple[Any, ...]] = []
        for entry in checklist:
            key = _constraint_key(entry)
            entries.append(key)

        current_user_keys = set(entries) - repository_policy_keys
        replacement_quota = len(previous_user_keys - current_user_keys)

        round_sources: list[str] = []
        next_source_by_key: dict[tuple[Any, ...], str] = {}
        for key in entries:
            if key in repository_policy_keys:
                source = "repository_policy"
            elif key in previous_source_by_key:
                source = previous_source_by_key[key]
            elif replacement_quota > 0:
                source = "replaced"
                replacement_quota -= 1
            else:
                source = "new_added"
            round_sources.append(source)
            if source != "repository_policy":
                next_source_by_key[key] = source

        sources_by_round.append(round_sources)
        previous_user_keys = current_user_keys
        previous_source_by_key = next_source_by_key

    return sources_by_round


def _completed_judgement(
    payload: dict[str, Any],
) -> tuple[MTACIFBenchJudgement | None, str]:
    inference = payload.get("inference")
    if not isinstance(inference, dict):
        return None, "invalid_inference"
    if inference.get("error") is not None:
        return None, "inference_error"

    raw_judgement = payload.get("judgement")
    if not isinstance(raw_judgement, dict):
        return None, "missing_judgement"
    try:
        judgement = MTACIFBenchJudgement.model_validate(raw_judgement)
    except Exception:
        return None, "invalid_judgement"
    if judgement.error is not None:
        return None, "judgement_error"
    return judgement, "completed"


def _result_error_reason(payload: dict[str, Any], status: str) -> str:
    inference = payload.get("inference")
    judgement = payload.get("judgement")
    error: Any = inference.get("error") if isinstance(inference, dict) else None
    if status != "inference_error":
        error = judgement.get("error") if isinstance(judgement, dict) else None
    return str(error.get("message") or "") if isinstance(error, dict) else ""


def _empty_if_bucket() -> dict[str, int]:
    return {
        "constraints_ok": 0,
        "constraints_total": 0,
        "rounds_ok": 0,
        "rounds_total": 0,
    }


def _empty_constraint_bucket() -> dict[str, int]:
    return {"constraints_ok": 0, "constraints_total": 0}


def _if_bucket_row(bucket: dict[str, int]) -> dict[str, int | float | None]:
    return {
        "CSR": _ratio(bucket["constraints_ok"], bucket["constraints_total"]),
        "ISR": _ratio(bucket["rounds_ok"], bucket["rounds_total"]),
        **bucket,
    }


def _constraint_bucket_row(bucket: dict[str, int]) -> dict[str, int | float | None]:
    return {
        "CSR": _ratio(bucket["constraints_ok"], bucket["constraints_total"]),
        **bucket,
    }


def _functional_check_candidates(payload: dict[str, Any]) -> tuple[list[Any] | None, bool]:
    """Return raw function checks and whether an evaluation field was present."""
    containers: list[Any] = [
        payload.get("evaluation"),
        payload.get("functional_evaluation"),
        payload.get("judgement"),
    ]
    present = False
    for container in containers:
        if isinstance(container, list):
            present = True
            return container, present
        if not isinstance(container, dict):
            continue
        if container.get("error") is not None:
            present = True
            continue
        for key in ("checks", "function_checks", "function_checklist_results"):
            if key in container:
                present = True
                checks = container.get(key)
                if isinstance(checks, list):
                    return checks, present
    return None, present


def _completed_function_evaluation(
    payload: dict[str, Any],
) -> tuple[list[float] | None, str, bool]:
    question = payload.get("question")
    checklist = question.get("function_checklist") if isinstance(question, dict) else None
    expected = checklist if isinstance(checklist, list) else []
    judgement = payload.get("judgement")
    if isinstance(judgement, dict) and judgement.get("function_checklist_skipped") is True:
        return None, "skipped", True
    raw_checks, present = _functional_check_candidates(payload)
    if not expected:
        return None, "missing_function_checklist", present
    if raw_checks is None:
        return None, "missing_evaluation", present

    expected_ids = [
        str(item.get("checklist_id", index)) if isinstance(item, dict) else str(index)
        for index, item in enumerate(expected)
    ]
    by_id: dict[str, Any] = {}
    for index, item in enumerate(raw_checks):
        if not isinstance(item, dict):
            continue
        identifier = item.get("id", item.get("checklist_id", item.get("index", index)))
        by_id[str(identifier)] = item

    if all(identifier in by_id for identifier in expected_ids):
        ordered = [by_id[identifier] for identifier in expected_ids]
    elif len(raw_checks) == len(expected) and all(isinstance(item, dict) for item in raw_checks):
        ordered = raw_checks
    else:
        return None, f"incomplete_evaluation:{len(raw_checks)}/{len(expected)}", True

    completed: list[float] = []
    for item in ordered:
        score = item.get("score")
        if not _is_number(score):
            return None, "evaluation_score_missing", True
        completed.append(float(score))
    return completed, "completed", True


def _model_name(result_dir: Path) -> str:
    if result_dir.name == "result" and result_dir.parent.name:
        return result_dir.parent.name
    return result_dir.name


def summarize(
    result_dir: Path,
    expected_total: int | None,
    include_cases: bool,
) -> dict[str, Any]:
    model_name = _model_name(result_dir)
    completed: list[tuple[str, dict[str, Any], MTACIFBenchJudgement]] = []
    excluded: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    paths = sorted(result_dir.glob("*.json"))

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            status_counts["invalid_json"] += 1
            excluded.append({"case_id": path.stem, "status": "invalid_json", "reason": str(exc)})
            continue
        if not isinstance(payload, dict):
            status_counts["invalid_result"] += 1
            excluded.append(
                {
                    "case_id": path.stem,
                    "status": "invalid_result",
                    "reason": "root is not an object",
                }
            )
            continue

        judgement, status = _completed_judgement(payload)
        status_counts[status] += 1
        if judgement is None:
            excluded.append(
                {
                    "case_id": path.stem,
                    "status": status,
                    "reason": _result_error_reason(payload, status),
                }
            )
            continue
        completed.append((path.stem, payload, judgement))

    round_buckets: dict[int, dict[str, int]] = defaultdict(_empty_if_bucket)
    band_buckets: dict[str, dict[str, int]] = defaultdict(_empty_if_bucket)
    primary_buckets: dict[str, dict[str, int]] = defaultdict(_empty_constraint_bucket)
    secondary_buckets: dict[str, dict[str, int]] = defaultdict(_empty_constraint_bucket)
    source_buckets: dict[str, dict[str, int]] = defaultdict(_empty_constraint_bucket)
    round_source_buckets: dict[
        int,
        dict[str, dict[str, int]],
    ] = defaultdict(lambda: defaultdict(_empty_constraint_bucket))

    total_rounds = 0
    passed_rounds = 0
    total_constraints = 0
    passed_constraints = 0
    function_status_counts: Counter[str] = Counter()
    function_evaluation_seen = False
    function_cases = 0
    function_success_cases = 0
    function_score_sum = 0.0
    function_total_checks = 0
    build_success_cases = 0
    case_rows: list[dict[str, Any]] = []

    for case_id, payload, judgement in completed:
        question = payload.get("question")
        raw_rounds = question.get("rounds") if isinstance(question, dict) else None
        question_rounds = raw_rounds if isinstance(raw_rounds, list) else []
        repository_policy_checklist = (
            question.get("repository_policy_checklist") if isinstance(question, dict) else None
        )
        if not isinstance(repository_policy_checklist, list):
            repository_policy_checklist = []
        sources_by_round = _classify_constraint_sources(
            repository_policy_checklist,
            question_rounds,
        )
        round_index_by_id: dict[int, int] = {}
        for index, item in enumerate(question_rounds):
            raw_round_id = item.get("round_id") if isinstance(item, dict) else None
            if _is_number(raw_round_id):
                round_index_by_id[int(raw_round_id)] = index

        case_round_pass = 0
        case_constraint_pass = 0
        case_constraint_total = 0
        for fallback_index, round_result in enumerate(judgement.instruction_following_checks):
            question_index = round_index_by_id.get(round_result.round_id, fallback_index)
            round_number = question_index + 1
            question_round = (
                question_rounds[question_index]
                if 0 <= question_index < len(question_rounds)
                else {}
            )
            raw_checklist = (
                question_round.get("instruction_following_checklist")
                if isinstance(question_round, dict)
                else None
            )
            checklist = raw_checklist if isinstance(raw_checklist, list) else []
            sources = (
                sources_by_round[question_index]
                if 0 <= question_index < len(sources_by_round)
                else []
            )

            bucket = round_buckets[round_number]
            bucket["rounds_total"] += 1
            bucket["rounds_ok"] += int(round_result.passed)
            total_rounds += 1
            passed_rounds += int(round_result.passed)
            case_round_pass += int(round_result.passed)

            for low, high in ROUND_BANDS:
                if low <= round_number <= high:
                    band = band_buckets[f"{low}-{high}"]
                    band["rounds_total"] += 1
                    band["rounds_ok"] += int(round_result.passed)
                    break

            for check_index, check in enumerate(round_result.check_results):
                passed = check.conclusion == PASS_CONCLUSION
                bucket["constraints_total"] += 1
                bucket["constraints_ok"] += int(passed)
                total_constraints += 1
                passed_constraints += int(passed)
                case_constraint_total += 1
                case_constraint_pass += int(passed)
                for low, high in ROUND_BANDS:
                    if low <= round_number <= high:
                        band = band_buckets[f"{low}-{high}"]
                        band["constraints_total"] += 1
                        band["constraints_ok"] += int(passed)
                        break

                entry = checklist[check_index] if check_index < len(checklist) else {}
                primary, secondary = _constraint_tag(entry)
                source = sources[check_index] if check_index < len(sources) else "unknown"
                for category_bucket in (
                    primary_buckets[primary],
                    secondary_buckets[secondary],
                    source_buckets[source],
                    round_source_buckets[round_number][source],
                ):
                    category_bucket["constraints_total"] += 1
                    category_bucket["constraints_ok"] += int(passed)

        function_checks, function_status, evaluation_present = _completed_function_evaluation(
            payload
        )
        function_evaluation_seen = function_evaluation_seen or evaluation_present
        function_status_counts[function_status] += 1
        if function_checks is not None:
            function_cases += 1
            case_score = sum(function_checks)
            function_score_sum += case_score
            function_total_checks += len(function_checks)
            function_success_cases += int(all(score >= 1.0 for score in function_checks))
            build_success_cases += int(judgement.build_success is not False)

        if include_cases:
            case_rows.append(
                {
                    "case_id": case_id,
                    "strict_pass": judgement.instruction_following_score == 1.0,
                    "rounds_ok": case_round_pass,
                    "rounds_total": len(judgement.instruction_following_checks),
                    "ISR": _ratio(
                        case_round_pass,
                        len(judgement.instruction_following_checks),
                    ),
                    "constraints_ok": case_constraint_pass,
                    "constraints_total": case_constraint_total,
                    "CSR": _ratio(case_constraint_pass, case_constraint_total),
                }
            )

    model_stats: dict[str, Any] = {
        "result_dir": str(result_dir.resolve()),
        "result_files": len(paths),
        "valid_cases": len(completed),
        "skipped_cases": len(excluded),
        "IF_ISR_all": _ratio(passed_rounds, total_rounds),
        "IF_CSR_all": _ratio(passed_constraints, total_constraints),
        "rounds_ok": passed_rounds,
        "rounds_total": total_rounds,
        "constraints_ok": passed_constraints,
        "constraints_total": total_constraints,
    }
    if expected_total is not None:
        model_stats["expected_cases"] = expected_total
        model_stats["not_completed_cases"] = max(expected_total - len(completed), 0)

    function_metrics_enabled = function_cases > 0
    if function_metrics_enabled:
        model_stats.update(
            {
                "FC_CSR": _ratio(function_score_sum, function_total_checks),
                "FC_ISR": _ratio(function_success_cases, function_cases),
                "function_success_cases": function_success_cases,
                "function_total_cases": function_cases,
                "function_evaluation_completed_cases": function_cases,
                "function_evaluation_excluded_cases": len(completed) - function_cases,
                "function_score_sum": round(function_score_sum, 6),
                "function_total_checks": function_total_checks,
                "BSR": _ratio(build_success_cases, function_cases),
                "build_success_cases": build_success_cases,
                "build_total_cases": function_cases,
            }
        )
    if function_evaluation_seen:
        model_stats["function_evaluation_status_counts"] = dict(
            sorted(function_status_counts.items())
        )

    return {
        "model": model_name,
        "model_stats": model_stats,
        "instruction_following_by_round": {
            str(round_number): _if_bucket_row(bucket)
            for round_number, bucket in sorted(round_buckets.items())
        },
        "instruction_following_by_round_bands": {
            f"{low}-{high}": _if_bucket_row(band_buckets[f"{low}-{high}"])
            for low, high in ROUND_BANDS
        },
        "instruction_following_by_primary_category": {
            name: _constraint_bucket_row(bucket) for name, bucket in sorted(primary_buckets.items())
        },
        "instruction_following_by_secondary_category": {
            name: _constraint_bucket_row(bucket)
            for name, bucket in sorted(secondary_buckets.items())
        },
        "instruction_following_by_constraint_source": {
            name: _constraint_bucket_row(bucket) for name, bucket in sorted(source_buckets.items())
        },
        "instruction_following_by_round_and_constraint_source": {
            str(round_number): {
                name: _constraint_bucket_row(bucket) for name, bucket in sorted(source_map.items())
            }
            for round_number, source_map in sorted(round_source_buckets.items())
        },
        "status_counts": dict(sorted(status_counts.items())),
        "excluded": excluded,
        "cases": case_rows,
        "function_metrics_enabled": function_metrics_enabled,
    }


def build_report(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    model_stats: dict[str, Any] = {}
    sections: dict[str, dict[str, Any]] = {
        "instruction_following_by_round": {},
        "instruction_following_by_round_bands": {},
        "instruction_following_by_primary_category": {},
        "instruction_following_by_secondary_category": {},
        "instruction_following_by_constraint_source": {},
        "instruction_following_by_round_and_constraint_source": {},
    }
    skipped_cases: dict[str, Any] = {}
    skipped_details: list[dict[str, str]] = []
    cases: dict[str, Any] = {}

    for summary in summaries:
        model = summary["model"]
        if model in model_stats:
            raise ValueError(f"duplicate model name inferred from result directories: {model}")
        model_stats[model] = summary["model_stats"]
        for section in sections:
            sections[section][model] = summary[section]
        skipped_cases[model] = {
            "skipped_cases": summary["model_stats"]["skipped_cases"],
            "reasons": [
                {"reason": reason, "count": count}
                for reason, count in summary["status_counts"].items()
                if reason != "completed"
            ],
        }
        skipped_details.extend({"model": model, **item} for item in summary["excluded"])
        if summary["cases"]:
            cases[model] = summary["cases"]

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).astimezone().isoformat(),
        "function_metrics_enabled": any(
            summary["function_metrics_enabled"] for summary in summaries
        ),
        "policy": {
            "case_scope": (
                "Only fully generated and fully judged cases are included in IF metrics."
            ),
            "IF_CSR_all": "Passed constraints divided by completed constraints.",
            "IF_ISR_all": "Fully passed rounds divided by completed rounds.",
            "function_metrics": (
                "FC_CSR, FC_ISR and BSR are emitted per model only when complete "
                "function evaluation checks exist."
            ),
            "BSR": (
                "Among cases with complete function evaluation, a case is build-successful "
                "unless its recorded build_success verdict is false."
            ),
        },
        "summary": {
            "model_count": len(model_stats),
            "total_valid_cases": sum(row["valid_cases"] for row in model_stats.values()),
            "total_skipped_cases": sum(row["skipped_cases"] for row in model_stats.values()),
        },
        "model_stats": model_stats,
        **sections,
        "skipped_cases": skipped_cases,
        "skipped_case_details": skipped_details,
    }
    if cases:
        report["cases"] = cases
    return report


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the default metrics artifact limited to paper-facing key results."""
    compact_model_stats: dict[str, dict[str, Any]] = {}
    for model, stats in report["model_stats"].items():
        keys = [
            *COMPACT_MODEL_STAT_KEYS,
            *OPTIONAL_COVERAGE_STAT_KEYS,
        ]
        if "FC_CSR" in stats:
            keys.extend(FUNCTION_MODEL_STAT_KEYS)
        compact_model_stats[model] = {key: stats[key] for key in keys if key in stats}

    return {
        "generated_at": report["generated_at"],
        "function_metrics_enabled": report["function_metrics_enabled"],
        "summary": report["summary"],
        "model_stats": compact_model_stats,
    }


def print_human(report: dict[str, Any], details: bool) -> None:
    for model, stats in report["model_stats"].items():
        print(model)
        print(
            f"  valid/skipped: {stats['valid_cases']}/{stats['skipped_cases']}  "
            f"result files: {stats['result_files']}"
        )
        if "expected_cases" in stats:
            print(
                f"  expected: {stats['expected_cases']}  "
                f"not completed: {stats['not_completed_cases']}"
            )
        print(
            f"  IF_ISR: {_percentage(stats['IF_ISR_all']):.6f}%  "
            f"IF_CSR: {_percentage(stats['IF_CSR_all']):.6f}%"
        )
        if "BSR" in stats:
            print(
                f"  FC_CSR: {_percentage(stats['FC_CSR']):.6f}%  "
                f"FC_ISR: {_percentage(stats['FC_ISR']):.6f}%  "
                f"BSR: {_percentage(stats['BSR']):.6f}%"
            )
        if details:
            for item in report["skipped_case_details"]:
                if item["model"] != model:
                    continue
                suffix = f": {item['reason']}" if item["reason"] else ""
                print(f"  excluded {item['case_id']}: {item['status']}{suffix}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paper-facing MTAC-IFBench statistics from fully generated and "
            "fully judged cases. Failed, missing, invalid and incomplete cases are "
            "excluded from score denominators."
        )
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="One or more .../<model>/result directories.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help="Expected case count per model, used only for coverage reporting.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-valid-case scores and print excluded-case details.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human-readable summary.",
    )
    args = parser.parse_args()
    if args.expected_total is not None and args.expected_total < 0:
        parser.error("--expected-total must be non-negative")
    for result_dir in args.result_dirs:
        if not result_dir.is_dir():
            parser.error(f"result directory does not exist: {result_dir}")
    return args


def main() -> None:
    args = parse_args()
    summaries = [
        summarize(result_dir, args.expected_total, args.details) for result_dir in args.result_dirs
    ]
    try:
        report = build_report(summaries)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        output_report = report if args.details else compact_report(report)
        print(json.dumps(output_report, ensure_ascii=False, indent=2))
    else:
        print_human(report, args.details)


if __name__ == "__main__":
    main()
