"""Execution of dataset-supplied instruction-following validation code.

Roughly half of MTACIFBench constraints ship a deterministic Python checker.
Those run instead of the LLM judge. The dataset authors are trusted, so this is
not a sandbox boundary — but a checker with a stray ``while`` loop or a blocking
call would otherwise pin a judge worker forever with no error surfacing. Each
checker therefore runs in its own short-lived subprocess with a hard timeout,
and any failure degrades to the judge rather than scoring the constraint 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Entry-point names the dataset uses, in priority order.
PREFERRED_VALIDATOR_NAMES = (
    "check",
    "verify",
    "check_requirement",
    "check_requirements",
    "check_response",
)

_DRIVER = '''\
"""Run one dataset validation checker and report a JSON verdict on stdout."""

import inspect
import json
import sys

sys.path.insert(0, {repo_root!r})

from emoji import is_emoji  # noqa: E402

from benchmarks.mtacifbench.utils import count_word, split_sentences  # noqa: E402

PREFERRED_NAMES = {preferred_names!r}
MODULE_NAME = "__instruction_following_validation__"


def can_call(func):
    try:
        inspect.signature(func).bind("", "")
    except (TypeError, ValueError):
        return False
    return True


def resolve(namespace):
    for name in PREFERRED_NAMES:
        candidate = namespace.get(name)
        if inspect.isfunction(candidate) and can_call(candidate):
            return candidate

    owned = [
        value
        for value in namespace.values()
        if inspect.isfunction(value)
        and getattr(value, "__module__", "") == MODULE_NAME
        and can_call(value)
    ]
    # Prefer conventional checker names before any other two-argument function,
    # so a two-argument private helper cannot be mistaken for the entry point.
    named = [f for f in owned if f.__name__.startswith(("check", "verify"))]
    for pool in (named, owned):
        if pool:
            return sorted(pool, key=lambda func: func.__name__)[0]
    raise ValueError("no callable validator found in validation code")


def main():
    code_path, response_path, workspace_path = sys.argv[1:4]
    code = open(code_path, encoding="utf-8").read()
    response = open(response_path, encoding="utf-8").read()

    namespace = {{
        "__builtins__": __builtins__,
        "__name__": MODULE_NAME,
        "is_emoji": is_emoji,
        "count_word": count_word,
        "split_sentences": split_sentences,
    }}
    exec(compile(code, "<instruction_following_validation>", "exec"), namespace, namespace)
    validator = resolve(namespace)
    result = validator(response, workspace_path)
    if not isinstance(result, bool):
        raise TypeError(
            "validation code must return bool, got " + type(result).__name__
        )
    return result


try:
    print(json.dumps({{"ok": True, "passed": main()}}))
except Exception as exc:  # noqa: BLE001 - reported to the caller as a verdict
    print(json.dumps({{"ok": False, "error": "{{}}: {{}}".format(type(exc).__name__, exc)}}))
'''


def _repo_root() -> str:
    # benchmarks/mtacifbench/validation.py -> repo root
    return str(Path(__file__).resolve().parents[2])


def run_validation_code(
    code: str,
    response: str,
    workspace_path: Path,
    timeout: int,
    log_tag: str = "",
) -> bool | None:
    """Run one checker. Return its verdict, or ``None`` to fall back to the judge.

    ``None`` covers a timeout, a crash inside the dataset code, an unresolvable
    entry point and a non-bool return — all cases where we learned nothing about
    the constraint and must not charge the model for it.
    """
    # The checker runs with cwd set to a scratch directory, so the workspace it
    # is handed must be absolute. Experiment configs use a relative output_dir
    # ("./output") by default, and a relative path here silently fails every
    # `os.path.exists(workspace_path)` guard — i.e. every workspace-walking
    # checker returns False no matter what the model wrote.
    workspace_path = Path(workspace_path).resolve()
    with tempfile.TemporaryDirectory(prefix="mtacif_validate_") as tmp:
        tmp_path = Path(tmp)
        code_path = tmp_path / "validation_code.py"
        response_path = tmp_path / "response.txt"
        driver_path = tmp_path / "driver.py"
        code_path.write_text(code, encoding="utf-8")
        response_path.write_text(response, encoding="utf-8")
        driver_path.write_text(
            _DRIVER.format(
                repo_root=_repo_root(),
                preferred_names=PREFERRED_VALIDATOR_NAMES,
            ),
            encoding="utf-8",
        )

        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    sys.executable,
                    str(driver_path),
                    str(code_path),
                    str(response_path),
                    str(workspace_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_path,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "{} validation code timed out after {}s; falling back to judge",
                log_tag,
                timeout,
            )
            return None

        stdout = (completed.stdout or "").strip().splitlines()
        if not stdout:
            logger.warning(
                "{} validation code produced no verdict (exit={}, stderr={}); "
                "falling back to judge",
                log_tag,
                completed.returncode,
                (completed.stderr or "")[-300:],
            )
            return None
        try:
            verdict = json.loads(stdout[-1])
        except json.JSONDecodeError:
            logger.warning(
                "{} validation code verdict is not JSON: {}; falling back to judge",
                log_tag,
                stdout[-1][:300],
            )
            return None

        if not verdict.get("ok"):
            logger.warning(
                "{} validation code failed: {}; falling back to judge",
                log_tag,
                str(verdict.get("error"))[:300],
            )
            return None
        return bool(verdict.get("passed"))
