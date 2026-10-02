"""Prompt building shared by every translation backend."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from string import Template

from mangatl.models import ChapterMeta, GlossaryEntry, Region

VARIANTS = {
    "es-419": (
        "latinoamericano neutro",
        "Variant: neutral Latin American Spanish. Use 'tú' for informal address and 'ustedes' "
        "for plural. NEVER use vosotros forms (vosotros, os, sepáis, tenéis, estáis, venid): "
        "write 'ustedes saben', 'ustedes tienen'. Avoid regionalisms of any single country and "
        "Spain-only words (vale, tío, coger, ordenador, móvil, gilipollas). Casual speech is "
        "fine as long as any Latin American reader understands it.",
    ),
    "es-MX": (
        "español de México",
        "Variant: Mexican Spanish. 'tú' and 'ustedes'; NEVER vosotros forms (os, sepáis, "
        "tenéis). Mexican colloquial "
        "expressions are welcome in casual dialogue, matching each character's register.",
    ),
    "es-ES": (
        "español de España",
        "Variant: Castilian Spanish from Spain. 'tú' and 'vosotros' for informal plural, "
        "'usted/ustedes' for formal address. Spain colloquialisms are welcome in casual dialogue.",
    ),
    "es-AR": (
        "español rioplatense",
        "Variant: Rioplatense Spanish (Argentina). Use voseo ('vos tenés', 'vení') for "
        "informal address and 'ustedes' for plural. Argentine colloquialisms are welcome in "
        "casual dialogue.",
    ),
}

HONORIFICS = {
    "keep": (
        "Honorifics: KEEP Japanese honorifics attached with a hyphen (Tanaka-san, Kenji-kun, "
        "Yui-chan, Saitō-sensei, senpai, onii-chan, -sama). Do not translate them."
    ),
    "adapt": (
        "Honorifics: ADAPT them to natural Spanish (señor/señora, doctor, profesor, "
        "affectionate nicknames, or drop them). No Japanese honorifics in the output."
    ),
    "mixed": (
        "Honorifics: keep only senpai, sensei and -sama when meaningful; adapt or drop "
        "-san, -kun, -chan and family terms (onii-chan -> hermano/hermanito)."
    ),
}

SFX = {
    "ignore": "Sound effects (type sfx or evident onomatopoeia): leave `translation` empty.",
    "annotate": (
        "Sound effects (type sfx or evident onomatopoeia): give a very short Spanish "
        "rendering (e.g. '¡Pum!', 'Silencio…', 'Riiing') with style sfx; it is lettered small "
        "next to the original."
    ),
    "replace": (
        "Sound effects (type sfx or evident onomatopoeia): give a short, punchy Spanish "
        "onomatopoeia with style sfx; it will replace the original lettering."
    ),
}

SOURCE_EXTRA = {
    "ja": (
        "Source is Japanese: vertical text was already joined into lines; "
        "furigana may appear inline."
    ),
    "en": (
        "Source is English: detect whether it is already a translation of a Japanese manga "
        "and recover evident nuances (honorifics, forms of address) without inventing."
    ),
}


# Speaker and name rules: (original wording, translator.clear_speaker_rules = true). With the
# original wording local models copy the example "Shūhei" as the speaker of many balloons.
SPEAKER_RULES = (
    'who says it (a character name from context or the glossary) or "desconocido".',
    "who says it: only a name that appears in the text, the glossary, the previous lines or "
    "the story so far, written in the same romanization as your translation (never in "
    'Japanese script). Caption boxes (narration_box) are "narración" unless a character is '
    'clearly speaking. When you cannot tell, write "desconocido"; never invent a name.',
)
NAMES_RULES = (
    "keep the original order (family name first if the original does) and Hepburn "
    "romanization (Saitō, Shūhei, Tōkyō... or without macrons if the glossary says so). "
    "Glossary entries are mandatory.",
    "keep the original order (family name first if the original does) and Hepburn "
    "romanization with macrons for long vowels (as in the place names Tōkyō, Ōsaka; or "
    "without macrons if the glossary says so). Glossary entries are mandatory.",
)

PIVOT_EXTRA = (
    "The text is an English DRAFT made from a Japanese original: each region also carries "
    "`ja` with the original. Translate the meaning into natural Spanish, using the Japanese "
    "to resolve ambiguities, names, honorifics and tone. In `source_text_corrected` copy the "
    "English draft of that region."
)


def system_prompt(meta: ChapterMeta, pivot: bool = False, clear_speakers: bool = False) -> str:
    """Style-guide system prompt. `pivot`: second pass of JA -> EN -> ES.
    `clear_speakers`: stricter speaker rules (translator.clear_speaker_rules)."""
    template = Template(resources.files(__package__).joinpath("system.md").read_text("utf-8"))
    variant_name, variant_rules = VARIANTS.get(meta.target_variant, VARIANTS["es-419"])
    return template.substitute(
        speaker_rule=SPEAKER_RULES[clear_speakers],
        names_rule=NAMES_RULES[clear_speakers],
        source_language="Japanese" if meta.source_lang == "ja" and not pivot else "English",
        variant_name=variant_name,
        variant_rules=variant_rules,
        honorifics_rules=HONORIFICS.get(meta.honorifics, HONORIFICS["keep"]),
        sfx_rules=SFX.get(meta.sfx_mode, SFX["annotate"]),
        source_extra=PIVOT_EXTRA if pivot else SOURCE_EXTRA[meta.source_lang],
    )


def pivot_english_prompt() -> str:
    """System prompt of the first pivot pass (JA -> EN draft)."""
    return resources.files(__package__).joinpath("system_en.md").read_text("utf-8")


def region_payload(region: Region) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": region.id,
        "type": region.type,
        "text": region.text_for_translation,
        "ocr_conf": round(region.ocr_confidence or 0.0, 2),
        "order": region.reading_order,
        "pos": region.position,
    }
    if region.panel is not None:
        payload["panel"] = region.panel
    if region.capacity_chars:
        payload["max_chars"] = region.capacity_chars
    return payload


@dataclass
class BlockContext:
    series: str
    chapter: str
    pages: list[int]
    regions: list[Region]
    glossary: list[GlossaryEntry] = field(default_factory=list)
    story_so_far: str = ""
    previous_lines: list[Region] = field(default_factory=list)
    # Extra per-region fields merged into the payload (e.g. {"ja": ...} in pivot mode).
    extra: dict[str, dict[str, object]] = field(default_factory=dict)
    previous_chapters: list[str] = field(default_factory=list)


def glossary_lines(entries: list[GlossaryEntry]) -> list[str]:
    lines = []
    for e in entries:
        tag = "" if e.status == "approved" else " (provisional)"
        note = f" — {e.notes}" if e.notes else ""
        lines.append(f"- {e.source} → {e.target} [{e.category}]{tag}{note}")
    return lines


def block_message(ctx: BlockContext) -> str:
    parts = [f"SERIES: {ctx.series} — CHAPTER {ctx.chapter}"]
    parts.append("GLOSSARY (mandatory renderings):")
    parts.extend(glossary_lines(ctx.glossary) or ["(empty)"])
    if ctx.previous_chapters:
        parts.append("PREVIOUS CHAPTERS:")
        parts.extend(ctx.previous_chapters)
    parts.append("STORY SO FAR:")
    parts.append(ctx.story_so_far or "(this is the beginning of the chapter)")
    if ctx.previous_lines:
        parts.append("PREVIOUS LINES (already translated, for continuity):")
        for r in ctx.previous_lines:
            who = f"{r.speaker}: " if r.speaker and r.speaker != "desconocido" else ""
            parts.append(f"{r.id} {r.text_for_translation} => {who}{r.translation}")
    parts.append(f"PAGES IN THIS BLOCK: {', '.join(map(str, ctx.pages))}")
    parts.append("REGIONS (one JSON object per line, in reading order):")
    parts.extend(
        json.dumps(region_payload(r) | ctx.extra.get(r.id, {}), ensure_ascii=False)
        for r in ctx.regions
    )
    ids = ", ".join(r.id for r in ctx.regions)
    parts.append(
        f"Translate every region. Return one entry for each of these ids: {ids}. "
        "Answer with the JSON object only."
    )
    return "\n".join(parts)
