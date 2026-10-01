"""Extract JSON objects from free text (a pasted chat answer, a model reply...)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"'})


def _loads(text: str) -> Any | None:
    for candidate in (text, _TRAILING_COMMA.sub(r"\1", text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _balanced_objects(text: str) -> list[str]:
    """Top-level {...} spans, skipping braces inside strings."""
    spans, depth, start, in_str, escape = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """All JSON objects found in `text`: fenced code blocks first, then bare objects."""
    found: list[dict[str, Any]] = []
    blocks = _FENCE.findall(text)
    for chunk in blocks or [text]:
        obj = _loads(chunk.strip())
        if isinstance(obj, dict):
            found.append(obj)
            continue
        for span in _balanced_objects(chunk):
            obj = _loads(span)
            if obj is None:
                obj = _loads(span.translate(_SMART_QUOTES))
            if isinstance(obj, dict):
                found.append(obj)
    return found
