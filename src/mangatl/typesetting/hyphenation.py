"""Spanish hyphenation that never splits proper names, numbers or short words."""

from __future__ import annotations

import re
from functools import lru_cache

import pyphen

_CORE = re.compile(
    r"^(?P<lead>[^\wáéíóúüñÁÉÍÓÚÜÑ]*)(?P<core>[\wáéíóúüñÁÉÍÓÚÜÑ]+?)(?P<trail>[^\wáéíóúüñÁÉÍÓÚÜÑ]*)$"
)
MIN_WORD_LEN = 6


@lru_cache(maxsize=4)
def _dictionary(lang: str) -> pyphen.Pyphen:
    # At least 3 letters on each side: avoids ugly breaks such as "se-mana".
    return pyphen.Pyphen(lang=lang, left=3, right=3)


class Hyphenator:
    def __init__(self, lang: str = "es", protected: set[str] | None = None) -> None:
        self.dic = _dictionary(lang)
        self.protected = {w.lower() for w in (protected or set())}

    def can_split(self, word: str) -> bool:
        m = _CORE.match(word)
        if not m:
            return False
        core = m.group("core")
        return (
            len(core) >= MIN_WORD_LEN
            and core.isalpha()
            and not core[0].isupper()  # proper names (and sentence starts) stay whole
            and core.lower() not in self.protected
        )

    def splits(self, word: str) -> list[tuple[str, str]]:
        """Possible (head + '-', tail) pairs, longest head first."""
        if not self.can_split(word):
            return []
        m = _CORE.match(word)
        assert m is not None
        lead, core, trail = m.group("lead"), m.group("core"), m.group("trail")
        positions = sorted(self.dic.positions(core), reverse=True)
        return [(f"{lead}{core[:p]}-", f"{core[p:]}{trail}") for p in positions]
