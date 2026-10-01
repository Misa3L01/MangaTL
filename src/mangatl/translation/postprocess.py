"""Deterministic clean-up of translator output (Spanish punctuation, stray Japanese marks)."""

from __future__ import annotations

import re

_JP_BRACKETS = str.maketrans({"「": "", "」": "", "『": "«", "』": "»", "（": "(", "）": ")"})
_FULLWIDTH = str.maketrans({"！": "!", "？": "?", "、": ",", "。": ".", "・": "", "　": " "})
_PAIRS = {"¡": "!", "¿": "?"}


def _drop_nested_openers(text: str) -> str:
    """'¡No te detengas, ¡haz la transfusión!' -> '¡No te detengas, haz la transfusión!'."""
    out: list[str] = []
    open_marks: list[str] = []
    for ch in text:
        if ch in _PAIRS:
            if _PAIRS[ch] in [_PAIRS[m] for m in open_marks]:
                continue  # same kind already open in this sentence
            open_marks.append(ch)
        elif ch in ("!", "?"):
            open_marks = [m for m in open_marks if _PAIRS[m] != ch]
        elif ch in ".…" and not open_marks:
            pass
        out.append(ch)
    return "".join(out)


def _add_missing_openers(text: str) -> str:
    """Add '¡'/'¿' to sentences that end in '!'/'?' without an opening mark."""
    sentences = re.split(r"(?<=[!?.…])\s+", text)
    fixed = []
    for s in sentences:
        stripped = s.rstrip()
        end = stripped[-1:] if stripped else ""
        if end in ("!", "?"):
            opener = "¡" if end == "!" else "¿"
            if opener not in s:
                lead = re.match(r"^[\"'«(—-]*", s).group(0)
                s = f"{lead}{opener}{s[len(lead) :]}"
        fixed.append(s)
    return " ".join(fixed)


_VERTICAL_ELLIPSIS = str.maketrans({"︙": "…", "⋮": "…", "︰": "…", "‥": "…"})


# Second-person plural forms of Spain (sepáis, tenéis, vosotros, os...).
_VOSOTROS = re.compile(r"\b(vosotr[oa]s|os)\b|\b\w+[áé]is\b", re.IGNORECASE)


def vosotros_forms(text: str) -> list[str]:
    """Vosotros forms found in `text` (wrong in es-419, es-MX and es-AR)."""
    return [m.group(0) for m in _VOSOTROS.finditer(text or "")]


def clean_translation(text: str) -> str:
    text = text.translate(_JP_BRACKETS).translate(_FULLWIDTH).translate(_VERTICAL_ELLIPSIS)
    text = re.sub(r"\s+([,.;:!?…»)])", r"\1", text)  # first, so "… ::" becomes "…::"
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"…[.…:：]*[()]?", "…", text)  # "……", "…." "…::" "…:(" (JP pauses) -> "…"
    text = re.sub(r"([!?])\.+", r"\1", text)  # "¿…?." -> "¿…?"
    text = re.sub(r"([!?])[–—-]+([!?])", r"\1\2", text)  # "¡Buenos días!–!" -> "¡Buenos días!!"
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"([¡¿«(])\s+", r"\1", text)
    text = _drop_nested_openers(text)
    text = _add_missing_openers(text)
    return re.sub(r"\s+", " ", text).strip()
