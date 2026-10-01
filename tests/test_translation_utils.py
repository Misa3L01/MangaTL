"""JSON extraction, punctuation clean-up and answer realignment."""

from __future__ import annotations

import pytest

from mangatl.models import BBox, Region
from mangatl.translation.alignment import realign
from mangatl.translation.json_extract import extract_json_objects
from mangatl.translation.postprocess import clean_translation
from mangatl.translation.schema import RegionTranslation

# ------------------------------------------------------------------ JSON extraction


def test_extracts_json_from_fenced_block_with_text_around() -> None:
    text = 'Claro, aquí va:\n```json\n{"regions": [{"id": "P001-B01"}]}\n```\n¡Suerte!'
    assert extract_json_objects(text) == [{"regions": [{"id": "P001-B01"}]}]


def test_extracts_bare_object_surrounded_by_prose() -> None:
    text = 'La traducción es {"a": 1, "b": "llave } dentro de texto"} y nada más.'
    assert extract_json_objects(text) == [{"a": 1, "b": "llave } dentro de texto"}]


def test_tolerates_trailing_commas_and_multiple_objects() -> None:
    text = '{"a": [1, 2,],}\notra parte\n{"b": 2}'
    assert extract_json_objects(text) == [{"a": [1, 2]}, {"b": 2}]


def test_several_fenced_blocks() -> None:
    text = '```json\n{"p": 1}\n```\nparte 2\n```\n{"p": 2}\n```'
    assert [o["p"] for o in extract_json_objects(text)] == [1, 2]


def test_no_json_returns_empty_list() -> None:
    assert extract_json_objects("sin json por aquí") == []


# ------------------------------------------------------------------ punctuation


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("¡No te detengas, ¡haz la transfusión!", "¡No te detengas, haz la transfusión!"),
        ("Hola... ¿qué tal?", "Hola… ¿qué tal?"),
        ("Qué haces?", "¿Qué haces?"),
        ("Ya voy! Espera.", "¡Ya voy! Espera."),
        ("「¡Sí！」", "¡Sí!"),
        ("Bien , gracias .", "Bien, gracias."),
        ("¿Estás bien? ¡Responde!", "¿Estás bien? ¡Responde!"),
        ("Ee……… Soy Gōda.", "Ee… Soy Gōda."),
        ("Yo……:::", "Yo…"),
        ("Yo… ::", "Yo…"),
        ("¡Buenos días!–!", "¡Buenos días!"),
        ("¡¡Ayuda!!", "¡Ayuda!"),
        ("¿Un residente…?.", "¿Un residente…?"),
    ],
)
def test_clean_translation(raw: str, clean: str) -> None:
    assert clean_translation(raw) == clean


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Quizás lo sepáis, pero no es así.", ["sepáis"]),
        ("¿Vosotros tenéis hambre? Os espero.", ["Vosotros", "tenéis", "Os"]),
        ("Ustedes saben que el país tiene seis hospitales.", []),
    ],
)
def test_vosotros_detection(text: str, found: list[str]) -> None:
    from mangatl.translation.postprocess import vosotros_forms

    assert vosotros_forms(text) == found


# ------------------------------------------------------------------ realignment


def region(rid: str, text: str) -> Region:
    return Region(id=rid, type="speech_bubble", bbox=BBox(x0=0, y0=0, x1=1, y1=1), ocr_text=text)


def answer(rid: str, echo: str | None, tr: str) -> RegionTranslation:
    return RegionTranslation(id=rid, source_text_corrected=echo, translation=tr)


REGIONS = [
    region("P009-B05", "実技試験などは含まれていない"),
    region("P009-B06", "そこで医師免許を取得した者の大半は"),
    region("P009-B07", "もっと術野を広げろ斉藤"),
    region("P009-B08", "はい！"),
]


def test_shifted_answers_are_moved_back_to_their_regions() -> None:
    shifted = [
        answer("P009-B05", "そこで医師免許を取得した者の大半は", "Por eso la mayoría…"),
        answer("P009-B06", "もっと術野を広げろ斉藤", "¡Abre más el campo, Saitō!"),
        answer("P009-B07", "はい！", "¡Sí!"),
    ]
    fixed, moved = realign(REGIONS, shifted)
    assert moved == 3
    assert {a.id: a.translation for a in fixed} == {
        "P009-B06": "Por eso la mayoría…",
        "P009-B07": "¡Abre más el campo, Saitō!",
        "P009-B08": "¡Sí!",
    }


def test_correct_answers_are_untouched_and_ocr_fixes_still_match() -> None:
    answers = [
        answer("P009-B05", "実技試験などは含まれていない", "No incluye pruebas prácticas."),
        answer("P009-B07", "もっと術野を広げろ斎藤", "¡Abre más el campo, Saitō!"),  # OCR fix
    ]
    fixed, moved = realign(REGIONS, answers)
    assert moved == 0
    assert [a.id for a in fixed] == ["P009-B05", "P009-B07"]


def test_answers_without_echo_keep_their_id() -> None:
    fixed, moved = realign(REGIONS, [answer("P009-B08", None, "¡Sí!")])
    assert moved == 0 and fixed[0].id == "P009-B08"


def test_unmatched_garbage_answer_is_dropped() -> None:
    fixed, _ = realign(REGIONS, [answer("P009-B05", "まったく関係ない文章です", "???")])
    assert fixed == []
