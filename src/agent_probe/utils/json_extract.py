"""Shared JSON extraction utilities."""

from __future__ import annotations

import json
import re


def extract_json_from_text(text: str) -> dict | None:
    """Extract a JSON object from text, handling ```json``` fences and bare objects.

    Returns the first valid JSON dict found, or None.
    """
    # 1. Try ```json ... ``` fenced block
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 2. Try outermost { ... } block
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = text

    # 3. Parse
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    return None
