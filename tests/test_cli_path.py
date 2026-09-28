"""The project root has to be *first* on sys.path, not merely on it.

pysbd — pulled in for sentence segmentation — ships a top-level package called
``benchmarks``. An editable install leaves this repo's root on ``sys.path`` but
behind site-packages, so without reordering that package wins and every
benchmark import fails with ``No module named 'benchmarks.mtacifbench'``.

The bug was invisible to the rest of the suite: under ``python -c`` sys.path[0]
is ``''`` and the repo wins regardless. It only appeared through the installed
console script, which is how anyone actually runs an experiment.
"""

from __future__ import annotations

import sys

import pytest

from agent_probe.cli.main import _put_project_root_first


def test_a_root_already_present_is_moved_to_the_front(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression: "insert if absent" skipped exactly the broken case."""
    monkeypatch.setattr(
        sys, "path", ["/site-packages", "/project", "/elsewhere"], raising=False
    )

    _put_project_root_first("/project")

    assert sys.path[0] == "/project"
    # Moved, not duplicated.
    assert sys.path.count("/project") == 1
    assert sys.path == ["/project", "/site-packages", "/elsewhere"]


def test_an_absent_root_is_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", ["/site-packages"], raising=False)

    _put_project_root_first("/project")

    assert sys.path == ["/project", "/site-packages"]


def test_duplicate_entries_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "path", ["/project", "/site-packages", "/project"], raising=False
    )

    _put_project_root_first("/project")

    assert sys.path == ["/project", "/site-packages"]
