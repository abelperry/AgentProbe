"""Small wrappers around the official SWE-bench harness."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks.swebench.models import SWEBenchQuestion


@contextmanager
def patched_instance_image_mode() -> Iterator[None]:
    """Avoid generating env/repo build scripts for prebuilt instance images.

    SWE-bench remote instance images already contain the repository and
    environment. We still need the official eval script, but not env/repo script
    generation. The monkey patch is scoped to this context to avoid leaking into
    other benchmark code running in the same process.
    """
    import swebench.harness.test_spec.create_scripts as create_scripts
    import swebench.harness.test_spec.test_spec as test_spec_mod

    old_values = (
        create_scripts.make_env_script_list,
        create_scripts.make_repo_script_list,
        test_spec_mod.make_env_script_list,
        test_spec_mod.make_repo_script_list,
    )

    def empty_list(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    create_scripts.make_env_script_list = empty_list
    create_scripts.make_repo_script_list = empty_list
    test_spec_mod.make_env_script_list = empty_list
    test_spec_mod.make_repo_script_list = empty_list
    try:
        yield
    finally:
        (
            create_scripts.make_env_script_list,
            create_scripts.make_repo_script_list,
            test_spec_mod.make_env_script_list,
            test_spec_mod.make_repo_script_list,
        ) = old_values


def make_test_spec(question: SWEBenchQuestion, namespace: str = "swebench"):
    """Build an official SWE-bench TestSpec for a prebuilt instance image."""
    from swebench.harness.test_spec.test_spec import make_test_spec as official_make_test_spec

    with patched_instance_image_mode():
        return official_make_test_spec(question.to_swebench_instance(), namespace=namespace)


def read_patch_or_none(path: str) -> str | None:
    if not path:
        return None
    patch_path = Path(path)
    if not patch_path.exists():
        return None
    text = patch_path.read_text(encoding="utf-8")
    return text if text.strip() else None


def grade_with_official_harness(
    question: SWEBenchQuestion,
    patch_path: str,
    test_log_path: str,
    model_name: str,
    namespace: str = "swebench",
) -> dict[str, Any]:
    """Return the per-instance official SWE-bench report entry."""
    from swebench.harness.grading import get_eval_report

    if not test_log_path or not Path(test_log_path).exists():
        return {
            "patch_is_None": read_patch_or_none(patch_path) is None,
            "patch_exists": read_patch_or_none(patch_path) is not None,
            "patch_successfully_applied": False,
            "resolved": False,
            "tests_status": {"missing_log": True},
        }

    prediction = {
        "instance_id": question.instance_id,
        "model_name_or_path": model_name,
        "model_patch": read_patch_or_none(patch_path),
    }
    report = get_eval_report(
        test_spec=make_test_spec(question, namespace=namespace),
        prediction=prediction,
        test_log_path=test_log_path,
        include_tests_status=True,
    )
    return report[question.instance_id]
