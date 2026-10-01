"""Repair answers whose translations were attached to the wrong region id.

Small local models sometimes shift every translation by one balloon (typically after
merging a sentence split across two balloons). Each answer echoes the original text of its
region, so we can match answers to regions by text similarity instead of trusting ids.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

from mangatl.models import Region
from mangatl.translation.schema import RegionTranslation

log = logging.getLogger(__name__)

_NOISE = re.compile(r"[\s\W_]+", re.UNICODE)
MATCH = 0.55  # similarity needed to (re)assign an answer to a region
KEEP_OWN = 0.3  # an answer whose echo is this similar to its own region keeps its id


def normalize(text: str) -> str:
    return _NOISE.sub("", unicodedata.normalize("NFKC", text or "")).lower()


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb, autojunk=False).ratio()


def realign(
    regions: list[Region], answers: list[RegionTranslation]
) -> tuple[list[RegionTranslation], int]:
    """Return answers re-keyed to the regions they actually translate, and the number moved.

    Answers without an echo keep their id. Answers whose echo matches no region well enough
    are dropped (their region will be requested again).
    """
    by_id = {r.id: r for r in regions}
    echoed = [a for a in answers if a.source_text_corrected and a.id in by_id]
    plain = [a for a in answers if not a.source_text_corrected and a.id in by_id]
    if not echoed:
        return [a for a in answers if a.id in by_id], 0

    pairs = sorted(
        (
            (similarity(a.source_text_corrected or "", r.ocr_text), i, r.id)
            for i, a in enumerate(echoed)
            for r in regions
        ),
        reverse=True,
    )
    assigned: dict[int, str] = {}
    taken: set[str] = set()
    for score, i, rid in pairs:
        if score < MATCH:
            break
        if i in assigned or rid in taken:
            continue
        assigned[i] = rid
        taken.add(rid)

    result: list[RegionTranslation] = []
    moved = 0
    for i, answer in enumerate(echoed):
        rid = assigned.get(i)
        if rid is None:
            own = by_id[answer.id]
            if (
                answer.id in taken
                or similarity(answer.source_text_corrected or "", own.ocr_text) < KEEP_OWN
            ):
                continue  # unreliable answer: let the region be requested again
            rid = answer.id
            taken.add(rid)
        if rid != answer.id:
            moved += 1
            answer = answer.model_copy(update={"id": rid})
        result.append(answer)
    for answer in plain:
        if answer.id not in taken:
            result.append(answer)
            taken.add(answer.id)
    if moved:
        log.warning(
            "Se corrigió la alineación de %d traducciones (estaban en el globo equivocado)", moved
        )
    return result, moved
