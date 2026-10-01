"""Chapter driver, Ollama backend (mocked HTTP) and manual backend."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from mangatl.config import Settings
from mangatl.models import GlossaryEntry, Project
from mangatl.project_io import ProjectPaths
from mangatl.translation.base import Translator, translate_chapter
from mangatl.translation.manual_backend import export_prompt, import_translation
from mangatl.translation.ollama_backend import SCHEMA, OllamaTranslator
from mangatl.translation.prompts import BlockContext, block_message, system_prompt
from mangatl.translation.schema import BlockTranslation, GlossaryProposal, RegionTranslation


def fake_answer(ctx: BlockContext, skip: set[str] | None = None) -> BlockTranslation:
    return BlockTranslation(
        regions=[
            RegionTranslation(
                id=r.id,
                source_text_corrected=r.ocr_text,
                translation=f"ES {r.id}",
                confidence=0.9,
            )
            for r in ctx.regions
            if r.id not in (skip or set())
        ],
        block_summary=f"Resumen páginas {ctx.pages}",
        new_glossary_entries=[
            GlossaryProposal(source="斉藤", target="Saitō", category="character"),
            GlossaryProposal(source="Yoroshiku", target="¡Inventado!"),  # not in the text
        ],
    )


class FakeTranslator(Translator):
    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[BlockContext] = []

    def translate_block(self, system: str, ctx: BlockContext) -> BlockTranslation:
        self.contexts.append(ctx)
        return fake_answer(ctx)


# ------------------------------------------------------------------ driver


def test_chapter_is_translated_in_blocks_with_context(small_project: Project) -> None:
    fake = FakeTranslator()
    stats = translate_chapter(small_project, fake, [], pages_per_block=2, previous_lines=2)
    assert [c.pages for c in fake.contexts] == [[1, 2], [3]]
    second = fake.contexts[1]
    assert "Resumen páginas [1, 2]" in second.story_so_far
    assert [r.id for r in second.previous_lines] == ["P002-B01", "P002-B02"]
    assert stats.regions == 6
    assert small_project.find_region("P003-B01").translation == "ES P003-B01"
    assert small_project.chapter_summary.startswith("Resumen páginas [1, 2]")


def test_glossary_keeps_only_terms_found_in_the_text(small_project: Project) -> None:
    translate_chapter(small_project, FakeTranslator(), [], 4, 2)
    assert [(g.source, g.target, g.status) for g in small_project.pending_glossary] == [
        ("斉藤", "Saitō", "pending")
    ]


def test_glossary_is_sent_to_later_blocks(small_project: Project) -> None:
    fake = FakeTranslator()
    approved = [GlossaryEntry(source="永大", target="Eidai", status="approved")]
    translate_chapter(small_project, fake, approved, 1, 0)
    assert [g.source for g in fake.contexts[0].glossary] == ["永大"]
    assert [g.source for g in fake.contexts[1].glossary] == ["永大", "斉藤"]


def test_region_missing_from_answer_is_flagged(small_project: Project) -> None:
    class Forgetful(FakeTranslator):
        def translate_block(self, system: str, ctx: BlockContext) -> BlockTranslation:
            return fake_answer(ctx, skip={"P001-B02"})

    stats = translate_chapter(small_project, Forgetful(), [], 4, 2)
    region = small_project.find_region("P001-B02")
    assert stats.missing == ["P001-B02"]
    assert region.status == "needs_review" and region.translation is None


def test_edited_regions_are_not_retranslated(small_project: Project) -> None:
    region = small_project.find_region("P001-B01")
    region.status, region.translation = "edited", "Mi versión"
    translate_chapter(small_project, FakeTranslator(), [], 4, 2)
    assert region.translation == "Mi versión"


def test_clear_speaker_rules_only_change_the_speaker_and_name_lines(
    small_project: Project,
) -> None:
    current = system_prompt(small_project.meta)
    assert "(Saitō, Shūhei, Tōkyō..." in current  # default wording is untouched
    assert "a character name from context or the glossary" in current
    clear = system_prompt(small_project.meta, clear_speakers=True)
    assert "Shūhei" not in clear and "never invent a name" in clear
    changed = [a for a, b in zip(current.splitlines(), clear.splitlines(), strict=True) if a != b]
    assert len(changed) == 2


def test_prompts_carry_the_style_choices(small_project: Project) -> None:
    system = system_prompt(small_project.meta)
    assert "neutral Latin American Spanish" in system
    assert "KEEP Japanese honorifics" in system
    assert "lettered small" in system  # sfx annotate
    ctx = BlockContext("S", "1", [1], small_project.pages[0].regions)
    message = block_message(ctx)
    assert '"id": "P001-B01"' in message and '"max_chars": 60' in message


# ------------------------------------------------------------------ Ollama backend (mocked)


def ollama_reply(content: str, done_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "done_reason": done_reason,
            "prompt_eval_count": 100,
            "eval_count": 50,
            "eval_duration": 1_000_000_000,
        },
    )


def make_ollama(settings: Settings, replies: list) -> tuple[OllamaTranslator, list[dict]]:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        reply = replies.pop(0)
        return reply(body) if callable(reply) else reply

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaTranslator(settings, client=client), requests


def ctx_for(project: Project, page: int = 1) -> BlockContext:
    return BlockContext("S", "1", [page], project.pages[page - 1].regions)


def test_ollama_request_uses_schema_num_ctx_and_no_thinking(
    settings: Settings, small_project: Project
) -> None:
    ctx = ctx_for(small_project)
    tr, reqs = make_ollama(settings, [ollama_reply(fake_answer(ctx).model_dump_json())])
    result = tr.translate_block("sys", ctx)
    assert len(result.regions) == 3
    body = reqs[0]
    assert body["format"] == SCHEMA and body["think"] is False and body["stream"] is False
    assert body["options"]["num_ctx"] == settings.translator.ollama.num_ctx
    assert "source_text_corrected" in SCHEMA["properties"]["regions"]["items"]["required"]
    assert tr.stats.output_tokens == 50


def test_ollama_retries_invalid_json_with_the_error(
    settings: Settings, small_project: Project
) -> None:
    ctx = ctx_for(small_project)
    tr, reqs = make_ollama(
        settings,
        [
            ollama_reply('{"regions": "no es una lista"}'),
            ollama_reply(fake_answer(ctx).model_dump_json()),
        ],
    )
    result = tr.translate_block("sys", ctx)
    assert len(result.regions) == 3
    assert "not valid" in reqs[1]["messages"][-1]["content"]


def test_ollama_asks_again_only_for_missing_ids(settings: Settings, small_project: Project) -> None:
    ctx = ctx_for(small_project)
    first = fake_answer(ctx, skip={"P001-B03"})
    tr, reqs = make_ollama(
        settings,
        [
            ollama_reply(first.model_dump_json()),
            lambda body: ollama_reply(
                fake_answer(BlockContext("S", "1", [1], ctx.regions[2:])).model_dump_json()
            ),
        ],
    )
    result = tr.translate_block("sys", ctx)
    assert sorted(r.id for r in result.regions) == ["P001-B01", "P001-B02", "P001-B03"]
    retry_prompt = reqs[1]["messages"][1]["content"]
    assert "P001-B03" in retry_prompt and '"id": "P001-B01"' not in retry_prompt


def test_ollama_splits_block_when_output_is_truncated(
    settings: Settings, small_project: Project
) -> None:
    ctx = ctx_for(small_project)

    def answer(body: dict) -> httpx.Response:
        ids = [
            line for line in body["messages"][1]["content"].splitlines() if line.startswith('{"id"')
        ]
        sub = [r for r in ctx.regions if any(f'"{r.id}"' in line for line in ids)]
        return ollama_reply(fake_answer(BlockContext("S", "1", [1], sub)).model_dump_json())

    tr, reqs = make_ollama(settings, [ollama_reply('{"regions": [', "length"), answer, answer])
    result = tr.translate_block("sys", ctx)
    assert len(reqs) == 3
    assert sorted(r.id for r in result.regions) == ["P001-B01", "P001-B02", "P001-B03"]


def test_ollama_shorten_sends_budgets_and_returns_only_requested_ids(
    settings: Settings, small_project: Project
) -> None:
    region = small_project.find_region("P002-B01")
    region.translation = (
        "Como es la primera noche, esta noche hay un médico de guardia extra asignado."
    )
    reply = {
        "regions": [
            {"id": "P002-B01", "translation": "Hoy hay otro de guardia."},
            {"id": "X", "translation": "?"},
        ]
    }
    tr, reqs = make_ollama(settings, [ollama_reply(json.dumps(reply, ensure_ascii=False))])
    result = tr.shorten("sys", [(region, 30)])
    assert result == {"P002-B01": "Hoy hay otro de guardia."}
    body = reqs[0]
    assert body["format"]["required"] == ["regions"]
    assert '"max_chars": 30' in body["messages"][1]["content"]


def test_ollama_shorten_survives_bad_json(settings: Settings, small_project: Project) -> None:
    region = small_project.find_region("P001-B01")
    region.translation = "Largo"
    tr, _ = make_ollama(settings, [ollama_reply("esto no es json")])
    assert tr.shorten("sys", [(region, 10)]) == {}


def test_pivot_translates_through_an_english_draft(small_project: Project) -> None:
    class Recorder(FakeTranslator):
        def __init__(self) -> None:
            super().__init__()
            self.systems: list[str] = []

        def translate_block(self, system: str, ctx: BlockContext) -> BlockTranslation:
            self.systems.append(system)
            return super().translate_block(system, ctx)

    fake = Recorder()
    translate_chapter(small_project, fake, [], pages_per_block=4, previous_lines=0, pivot=True)
    assert len(fake.contexts) == 2  # one block, two passes
    assert "FIRST of two passes" in fake.systems[0]
    second = fake.contexts[1]
    # The second pass reads the English draft and keeps the Japanese as reference.
    assert second.regions[0].ocr_text == "ES P001-B01"
    assert second.extra["P001-B01"]["ja"] == "研修医というのは要するに見習いだ"
    region = small_project.find_region("P001-B01")
    assert region.translation == "ES P001-B01" and region.source_text_corrected is None


def test_previous_chapters_reach_only_the_first_block(small_project: Project) -> None:
    fake = FakeTranslator()
    translate_chapter(small_project, fake, [], 1, 0, previous_chapters=["Capítulo 0: prólogo"])
    assert fake.contexts[0].previous_chapters == ["Capítulo 0: prólogo"]
    assert all(not c.previous_chapters for c in fake.contexts[1:])
    assert "PREVIOUS CHAPTERS:" in block_message(fake.contexts[0])


def test_ollama_close_unloads_the_model(settings: Settings) -> None:
    tr, reqs = make_ollama(settings, [httpx.Response(200, json={})])
    tr.close()
    assert reqs == [{"model": settings.translator.ollama.model, "keep_alive": 0}]


# ------------------------------------------------------------------ manual backend


def test_export_prompt_contains_guide_ids_and_splits_parts(
    small_project: Project, tmp_path: Path
) -> None:
    paths = ProjectPaths(tmp_path / "bj-1.mangatl.json")
    files = export_prompt(small_project, paths, [], max_chars=300)
    assert len(files) >= 2
    text = "\n".join(f.read_text(encoding="utf-8") for f in files)
    for _, region in small_project.regions():
        assert region.id in text
    assert "Style guide" in text and "```json" in text
    assert "Parte 1 de" in files[0].read_text(encoding="utf-8")


def test_import_pasted_answer_reports_missing_unknown_invalid(small_project: Project) -> None:
    answer = {
        "regions": [
            {"id": "P001-B01", "translation": "Un residente es, en pocas palabras, un aprendiz."},
            {"id": "P001-B02", "translation": "¡Abre más el campo, Saitō!", "style": "shout"},
            {"id": "P099-B01", "translation": "No existe"},
            {"id": "P002-B01"},
        ],
        "chapter_summary": "Saitō empieza su residencia.",
        "new_glossary_entries": [{"source": "斉藤", "target": "Saitō", "category": "character"}],
    }
    pasted = f"¡Aquí tienes!\n```json\n{json.dumps(answer, ensure_ascii=False)}\n```\nSaludos."
    report = import_translation(small_project, pasted)
    assert report.applied == ["P001-B01", "P001-B02"]
    assert report.unknown == ["P099-B01"]
    assert len(report.invalid) == 1 and report.invalid[0].startswith("P002-B01")
    assert set(report.missing) == {"P001-B03", "P002-B01", "P002-B02", "P003-B01"}
    assert report.glossary_added == 1
    assert small_project.find_region("P001-B02").style == "shout"
    assert small_project.chapter_summary == "Saitō empieza su residencia."


def test_import_accepts_several_parts(small_project: Project) -> None:
    part1 = {"regions": [{"id": "P001-B01", "translation": "Uno"}]}
    part2 = {"regions": [{"id": "P003-B01", "translation": "¡La presión está bajando!"}]}
    report = import_translation(small_project, f"{json.dumps(part1)}\n\n{json.dumps(part2)}")
    assert report.applied == ["P001-B01", "P003-B01"]


@pytest.mark.parametrize("junk", ["", "no hay json", '{"otra": "cosa"}'])
def test_import_without_regions_applies_nothing(small_project: Project, junk: str) -> None:
    assert import_translation(small_project, junk).applied == []
