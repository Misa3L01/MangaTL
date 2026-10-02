# MangaTL

Traductor local de manga JA/EN → ES (Windows, GPU NVIDIA 6 GB, todo dentro de la carpeta del proyecto).

## Cómo trabajar
- Antes: `. .\env.ps1` (variables solo de la sesión; nada va a C:).
- Código, identificadores y comentarios en inglés; README, CLI y logs en español.
- Entrada: `src/mangatl/cli.py` (`uv run mangatl --help`). Pipeline: `pipeline.py`.
- Un test: `uv run pytest tests/test_translation.py -q --tb=short -x`
  (los de GPU llevan `@pytest.mark.gpu`; sin GPU: `-m "not gpu"`). Lint: `uv run ruff check .`

## Módulos clave (`src/mangatl/`)
- `config.py`: ajustes (defaults < config.toml < env `MANGATL_*`). Toda opción nueva va aquí y en `config.example.toml` (un test exige que coincidan).
- `ingest/`: carpeta/zip/cbz/pdf → páginas. `detection/`: RT-DETR, máscaras de globos (`regions.py`).
- `ocr/`: manga-ocr (ja), PP-OCR vía RapidOCR (en). `ordering/`: orden por viñetas.
- `translation/`: prompts, backend Ollama (`ollama_backend.py`), backend manual, glosario.
- `ollama_runtime.py`: Ollama portable (`.local/ollama`), arranque/parada, import de GGUF.
- `inpainting/`: relleno + LaMa. `typesetting/` y `stages/render.py`: rotulado.
- `server/` + `web/`: editor web (FastAPI + React). `bootstrap.py`: `mangatl setup`.
- `scripts/compare_models.py` (mide traducción, `--set clave=valor`) y `scripts/reclean.py` (mide limpieza).

## Probar sin servicios reales
Los tests de traducción usan Ollama simulado (httpx MockTransport); no hace falta GPU ni modelos.

## Contratos que no se rompen
- Estado del capítulo en un `.mangatl.json` (guardado atómico); las ediciones `edited` no se retraducen.
- Nunca dos modelos pesados en VRAM a la vez (visión en subproceso → LLM → LaMa).
- Cambios de comportamiento detrás de opciones de config; descargas: mostrar tamaño y dejar ≥3 GB libres.
- Repo público: nada de tomos del usuario, `series/`, `output/`, `config.toml`, `.env` ni rutas personales.
- Licencia MIT: no copiar código GPL (manga-image-translator, BallonsTranslator, comic-text-detector).

## No leer
`models/ .local/ .venv/ input/ output/ tmp/ logs/ series/ uv.lock web/node_modules web/dist docs/images tests/fixtures`, imágenes, PDFs, pesos. Del README: grep de la sección, nunca entero.
