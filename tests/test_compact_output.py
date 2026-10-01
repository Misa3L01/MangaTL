"""translator.compact_output: short JSON keys in the model's answer, same parsed result."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from mangatl.config import Settings
from mangatl.models import Project
from mangatl.translation.ollama_backend import (
    COMPACT_SCHEMA,
    OllamaTranslator,
    expand_compact,
    parse_block,
)
from mangatl.translation.prompts import BlockContext
from tests.test_translation import fake_answer

COMPACT_REPLY = {
    "regions": [
        {"id": "P001-B01", "src": "はい！", "who": "Saitō", "es": "¡Sí!", "st": "shout", "c": 0.9},
        {
            "id": "P001-B02",
            "src": "x",
            "who": "desconocido",
            "es": "Texto largo",
            "st": "normal",
            "c": 0.4,
            "alt": "Corto",
        },
    ],
    "block_summary": "Resumen.",
    "new_glossary_entries": [
        {"source": "斉藤", "target": "Saitō", "category": "character", "notes": "residente"}
    ],
}


def test_only_the_region_keys_are_short() -> None:
    # Same top level as the regular schema (the glossary list is optional there too).
    assert set(COMPACT_SCHEMA["required"]) == {"regions", "block_summary"}
    assert "new_glossary_entries" in COMPACT_SCHEMA["properties"]
    item = COMPACT_SCHEMA["properties"]["regions"]["items"]
    assert set(item["required"]) == {"id", "src", "who", "es", "st", "c"}
    glossary = COMPACT_SCHEMA["properties"]["new_glossary_entries"]["items"]
    assert "target" in glossary["properties"]


def test_first_version_with_one_letter_top_keys_is_still_read() -> None:
    old = {
        "r": [{"id": "P1", "src": "a", "who": "x", "es": "b", "st": "normal", "c": 1}],
        "sum": "S.",
        "gl": [{"src": "斉藤", "es": "Saitō", "cat": "character"}],
    }
    block = parse_block(json.dumps(old, ensure_ascii=False), compact=True)
    assert block.block_summary == "S." and block.new_glossary_entries[0].target == "Saitō"


def test_expand_compact_maps_every_key() -> None:
    block = parse_block(json.dumps(COMPACT_REPLY, ensure_ascii=False), compact=True)
    first, second = block.regions
    assert (first.id, first.source_text_corrected, first.speaker) == ("P001-B01", "はい！", "Saitō")
    assert (first.translation, first.style, first.confidence) == ("¡Sí!", "shout", 0.9)
    assert second.shorter_alternative == "Corto" and second.fits_capacity is False
    assert block.block_summary == "Resumen."
    entry = block.new_glossary_entries[0]
    assert (entry.source, entry.target, entry.category, entry.notes) == (
        "斉藤",
        "Saitō",
        "character",
        "residente",
    )


def test_compact_parser_accepts_long_keys_and_text_around() -> None:
    long_form = {"regions": [], "block_summary": "Nada.", "new_glossary_entries": []}
    assert parse_block(json.dumps(long_form), compact=True).block_summary == "Nada."
    wrapped = "Aquí va:\n" + json.dumps(COMPACT_REPLY, ensure_ascii=False) + "\nFin."
    assert len(parse_block(wrapped, compact=True).regions) == 2
    with pytest.raises(ValidationError):
        parse_block("esto no es json", compact=True)
    assert expand_compact({})["regions"] == []


def test_compact_request_uses_short_schema_and_instructions(
    tmp_path, small_project: Project
) -> None:
    settings = Settings(root=tmp_path)
    settings.translator.compact_output = True
    ctx = BlockContext("S", "1", [1], small_project.pages[0].regions)
    full = fake_answer(ctx)
    reply = {
        "regions": [
            {"id": r.id, "src": r.source_text_corrected, "who": r.speaker, "es": r.translation}
            | {"st": r.style, "c": r.confidence}
            for r in full.regions
        ],
        "block_summary": full.block_summary,
        "new_glossary_entries": [],
    }
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        content = json.dumps(reply, ensure_ascii=False)
        return httpx.Response(200, json={"message": {"content": content}, "eval_count": 1})

    tr = OllamaTranslator(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = tr.translate_block("sys", ctx)
    assert [r.id for r in result.regions] == [r.id for r in ctx.regions]
    assert sent[0]["format"] == COMPACT_SCHEMA
    assert "Compact output" in sent[0]["messages"][0]["content"]
