"""Dynamic import utilities."""

from __future__ import annotations

import importlib
from typing import Any


def import_class(dotted_path: str) -> type[Any]:
    """Import a class given its fully-qualified dotted path.

    Example::

        cls = import_class("benchmarks.zbackendbench.models.ZBackendBenchQuestion")
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid dotted path: {dotted_path!r}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name!r} not found in module {module_path!r}")
    return cls  # type: ignore[no-any-return]
