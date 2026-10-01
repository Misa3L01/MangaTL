"""Local translation with Ollama (default backend).

- Output shape is enforced with `format` (JSON schema) and validated with Pydantic.
- Invalid JSON: retry passing the validation error back to the model.
- Missing ids: ask again only for those.
- Truncated output (context full): split the block in two.
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import replace

import httpx
from pydantic import ValidationError

from mangatl.config import Settings
from mangatl.models import Region
from mangatl.translation.alignment import realign
from mangatl.translation.base import TranslationError, Translator
from mangatl.translation.json_extract import extract_json_objects
from mangatl.translation.prompts import BlockContext, block_message
from mangatl.translation.schema import BlockTranslation, inline_schema

log = logging.getLogger(__name__)


def _ollama_schema() -> dict:
    """Block schema where every region must echo its source text (see alignment.py)."""
    schema = inline_schema(BlockTranslation)
    item = schema["properties"]["regions"]["items"]
    item["properties"]["source_text_corrected"] = {"type": "string"}
    item["required"] = sorted(
        {*item.get("required", []), "source_text_corrected", "speaker", "style"}
    )
    # The model tends to omit the category of glossary proposals ("other" by default).
    glossary_item = schema["properties"]["new_glossary_entries"]["items"]
    glossary_item["required"] = sorted({*glossary_item.get("required", []), "category"})
    return schema


SCHEMA = _ollama_schema()
SHORTEN_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "translation": {"type": "string"}},
                "required": ["id", "translation"],
            },
        }
    },
    "required": ["regions"],
}


class TruncatedOutputError(TranslationError):
    pass


class RepetitionLoopError(TranslationError):
    """Ollama aborted the answer: the model kept repeating a token (e.g. '¡¡¡¡…')."""


def parse_block(content: str) -> BlockTranslation:
    """Validate the model reply, tolerating text around the JSON object."""
    try:
        return BlockTranslation.model_validate_json(content)
    except ValidationError as first_error:
        for obj in extract_json_objects(content):
            try:
                return BlockTranslation.model_validate(obj)
            except ValidationError:
                continue
        raise first_error


class OllamaTranslator(Translator):
    name = "ollama"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        super().__init__()
        self.cfg = settings.translator.ollama
        self.model = self.cfg.model
        self.max_retries = settings.translator.max_retries
        self.base_url = self.cfg.host.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0, read=1800.0))
        self.realigned = 0

    # ------------------------------------------------------------------ transport
    def _chat(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
        options: dict[str, float] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema or SCHEMA,
            "stream": False,
            "think": self.cfg.think,
            "keep_alive": "15m",
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
                **(options or {}),
            },
        }
        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = f"Ollama respondió {exc.response.status_code}: {exc.response.text[:300]}"
            if "repeat limit" in exc.response.text:
                raise RepetitionLoopError(message) from exc
            raise TranslationError(message) from exc
        except httpx.HTTPError as exc:
            raise TranslationError(
                f"No se pudo contactar a Ollama en {self.base_url}: {exc}"
            ) from exc
        data = resp.json()
        st = self.stats
        st.requests += 1
        prompt, output = int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))
        st.prompt_tokens += prompt
        st.output_tokens += output
        st.generation_seconds += data.get("eval_duration", 0) / 1e9
        st.context_peak = max(st.context_peak, prompt + output)
        if prompt + output > 0.9 * self.cfg.num_ctx:
            log.warning(
                "La petición usó %d de %d tokens de contexto: baja translator.pages_per_block "
                "o sube translator.ollama.num_ctx",
                prompt + output,
                self.cfg.num_ctx,
            )
        if data.get("done_reason") == "length":
            raise TruncatedOutputError("La respuesta del modelo se cortó (contexto lleno)")
        return data.get("message", {}).get("content", "")

    def _ask(self, system: str, user: str) -> BlockTranslation:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        options: dict[str, float] = {}
        for attempt in range(self.max_retries + 1):
            try:
                content = self._chat(messages, options=options)
            except RepetitionLoopError:
                if not self.cfg.recover_repeat_loops or attempt == self.max_retries:
                    raise
                # Another seed and a little more temperature usually get out of the loop.
                log.warning("El modelo entró en un bucle de repetición: se reintenta")
                options = {
                    "seed": attempt + 1,
                    "temperature": min(1.0, self.cfg.temperature + 0.2 * (attempt + 1)),
                }
                continue
            try:
                return parse_block(content)
            except ValidationError as exc:
                last_error = exc
                errors = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8]
                )
                log.warning("Respuesta inválida del modelo (intento %d): %s", attempt + 1, errors)
                messages += [
                    {"role": "assistant", "content": content[:4000]},
                    {
                        "role": "user",
                        "content": (
                            f"Your answer is not valid: {errors}. Reply again with ONLY the JSON "
                            "object following the required schema, covering every region id."
                        ),
                    },
                ]
        raise TranslationError(
            f"El modelo no devolvió JSON válido tras {self.max_retries + 1} intentos: {last_error}"
        )

    # ------------------------------------------------------------------ block
    def _split(self, system: str, ctx: BlockContext) -> BlockTranslation:
        half = len(ctx.regions) // 2
        first = self.translate_block(system, replace(ctx, regions=ctx.regions[:half]))
        second = self.translate_block(
            system, replace(ctx, regions=ctx.regions[half:], previous_lines=ctx.regions[:half])
        )
        return BlockTranslation(
            regions=first.regions + second.regions,
            block_summary=f"{first.block_summary} {second.block_summary}".strip(),
            new_glossary_entries=first.new_glossary_entries + second.new_glossary_entries,
        )

    def translate_block(self, system: str, ctx: BlockContext) -> BlockTranslation:
        try:
            result = self._ask(system, block_message(ctx))
        except TruncatedOutputError:
            if len(ctx.regions) <= 1:
                raise
            log.warning("Bloque demasiado largo para el contexto: se divide en dos")
            return self._split(system, ctx)
        except RepetitionLoopError:
            if not self.cfg.recover_repeat_loops:
                raise
            if len(ctx.regions) <= 1:
                # Isolated: this region stays untranslated and is flagged for review.
                log.warning(
                    "%s: el modelo no sale del bucle; queda sin traducir", ctx.regions[0].id
                )
                return BlockTranslation(regions=[], block_summary="")
            log.warning("Bucle de repetición persistente: el bloque se divide en dos")
            return self._split(system, ctx)

        expected = [r.id for r in ctx.regions]
        unknown = {rt.id for rt in result.regions} - set(expected)
        if unknown:
            log.debug("IDs no solicitados en la respuesta: %s", ", ".join(sorted(unknown)))
        result.regions, moved = realign(ctx.regions, result.regions)
        self.realigned += moved
        got = {rt.id for rt in result.regions}

        for attempt in range(self.max_retries):
            missing = [r for r in ctx.regions if r.id not in got]
            if not missing:
                break
            log.warning(
                "Faltan %d regiones en la respuesta (%s); se vuelven a pedir",
                len(missing),
                ", ".join(r.id for r in missing[:6]),
            )
            try:
                sub = self._ask(system, block_message(replace(ctx, regions=missing)))
            except RepetitionLoopError:
                if not self.cfg.recover_repeat_loops:
                    raise
                break  # the missing regions stay flagged for review
            sub_regions, moved = realign(missing, sub.regions)
            self.realigned += moved
            fresh = [rt for rt in sub_regions if rt.id not in got]
            result.regions.extend(fresh)
            result.new_glossary_entries.extend(sub.new_glossary_entries)
            got |= {rt.id for rt in fresh}
            if not fresh and attempt == self.max_retries - 1:
                break
        return result

    def shorten(self, system: str, items: list[tuple[Region, int]]) -> dict[str, str]:
        """Ask once for shorter versions of translations that do not fit their balloons."""
        if not items:
            return {}
        lines = [
            json.dumps(
                {
                    "id": r.id,
                    "original": r.text_for_translation,
                    "translation": r.translation,
                    "max_chars": limit,
                },
                ensure_ascii=False,
            )
            for r, limit in items
        ]
        user = (
            "These Spanish translations do not fit in their balloons. Rewrite each one shorter, "
            "with at most `max_chars` characters, keeping the meaning, the tone, the names and "
            "the Spanish punctuation. Drop filler words or rephrase; never leave it unfinished.\n"
            + "\n".join(lines)
            + '\nReturn JSON: {"regions": [{"id": ..., "translation": ...}]}'
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            content = self._chat(messages, schema=SHORTEN_SCHEMA)
            data = json.loads(content)
        except (TranslationError, json.JSONDecodeError) as exc:
            log.warning("No se pudo obtener la versión corta: %s", exc)
            return {}
        wanted = {r.id for r, _ in items}
        return {
            str(item.get("id")): str(item.get("translation", "")).strip()
            for item in data.get("regions", [])
            if isinstance(item, dict) and item.get("id") in wanted and item.get("translation")
        }

    def close(self) -> None:
        # Free VRAM right away (keep_alive=0); the server itself is stopped by its owner.
        with suppress(httpx.HTTPError):
            self._client.post(
                f"{self.base_url}/api/generate", json={"model": self.model, "keep_alive": 0}
            )

    def describe(self) -> str:
        return json.dumps({"model": self.model, "num_ctx": self.cfg.num_ctx})
