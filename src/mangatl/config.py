"""Application settings: defaults < config.toml < MANGATL_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

SpanishVariant = Literal["es-419", "es-MX", "es-ES", "es-AR"]
HonorificsMode = Literal["keep", "adapt", "mixed"]
SfxMode = Literal["ignore", "annotate", "replace"]
TranslatorBackend = Literal["ollama", "manual"]


def find_project_root() -> Path:
    """Locate the project root: $MANGATL_ROOT, else the repo containing this package, else cwd."""
    env_root = os.environ.get("MANGATL_ROOT")
    if env_root:
        return Path(env_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        return repo_root
    return Path.cwd().resolve()


class PathsConfig(BaseModel):
    models_dir: Path = Path("models")
    tmp_dir: Path = Path("tmp")
    logs_dir: Path = Path("logs")
    series_dir: Path = Path("series")
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    fonts_dir: Path = Path("assets/fonts")


class GpuConfig(BaseModel):
    device: Literal["cuda", "cpu"] = "cuda"
    # Warn before loading the LLM if less free VRAM than this is available.
    warn_free_vram_mb: int = 4000


class DetectionConfig(BaseModel):
    model_repo: str = "ogkalu/comic-text-and-bubble-detector"
    # Pinned commit for reproducible results; set to "main" to follow upstream.
    model_revision: str = "16e8a622f91fabc6b5b65c96d32d1183f8843546"
    model_files: list[str] = Field(
        default_factory=lambda: ["config.json", "model.safetensors", "preprocessor_config.json"]
    )
    # Raw score cut-off; per-class thresholds below decide what becomes a region.
    threshold: float = Field(0.3, ge=0.0, le=1.0)
    bubble_threshold: float = Field(0.5, ge=0.0, le=1.0)
    text_threshold: float = Field(0.45, ge=0.0, le=1.0)
    # Reading order by panels (falls back to rows when no panels are found).
    panel_order: bool = True


class OcrConfig(BaseModel):
    ja_model_repo: str = "kha-white/manga-ocr-base"
    ja_model_revision: str = "aa6573bd10b0d446cbf622e29c3e084914df9741"
    batch_size: int = Field(16, ge=1)
    # Regions whose OCR text is empty or below this confidence are flagged for review.
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)


class InpaintConfig(BaseModel):
    # Pixels kept untouched along the inside of every bubble outline.
    border_protect_px: int = Field(3, ge=0)
    # Growth of the detected text box when searching glyph pixels.
    text_box_grow_px: int = Field(6, ge=0)
    # A pixel is "ink" when it is this much darker than the bubble background.
    text_contrast: int = Field(30, ge=1, le=255)
    # Dilation of the glyph mask to also remove anti-aliased edges.
    text_dilate_px: int = Field(3, ge=0)
    # Minimum share of the ink the fill must reach; below it the region goes to LaMa.
    min_coverage: float = Field(0.6, ge=0.0, le=1.0)
    # LaMa for text on artwork, screentone and lettering that crosses the outline.
    use_lama: bool = True
    lama_url: str = (
        "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/"
        "anime-manga-big-lama.pt"
    )
    lama_file: Path = Path("models/lama/anime-manga-big-lama.pt")
    lama_max_side: int = Field(1024, ge=256)
    # Experimental: also clean the bright areas of a bubble split off by a line of text that
    # touches the outline on both sides (their glyphs and the dividing line were left).
    join_text_areas: bool = False


class GgufImport(BaseModel):
    """Build the Ollama model from a GGUF on Hugging Face instead of the Ollama library."""

    repo: str
    file: str
    revision: str = "main"
    # Ollama's built-in prompt renderer/parser for the family (e.g. "qwen3.5").
    renderer: str | None = None
    parser: str | None = None
    parameters: dict[str, float | int] = Field(default_factory=dict)


class OllamaConfig(BaseModel):
    host: str = "http://127.0.0.1:11434"
    # Chosen after the Phase 2 comparison: clearly more accurate than qwen3.5:4b (names,
    # terms, meaning) at ~4x the time; runs partly in RAM on a 6 GB GPU.
    model: str = "qwen3.5:9b"
    num_ctx: int = Field(16384, ge=2048)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    # Reasoning ("thinking") mode: off by default, it multiplies latency for little gain here.
    think: bool = False
    # Start the portable server automatically when a command needs it, and stop it afterwards.
    auto_start: bool = True
    startup_timeout_s: float = 60.0
    install_dir: Path = Path(".local/ollama")
    version: str = "v0.34.4"
    models_dir: Path = Path("models/ollama")
    flash_attention: bool = True
    kv_cache_type: Literal["f16", "q8_0", "q4_0"] = "q8_0"
    # When Ollama aborts an answer because the model loops ("¡¡¡¡…", HTTP 500 "token repeat
    # limit"): retry with another seed, then split the block, instead of failing the chapter.
    recover_repeat_loops: bool = False
    # Extra Ollama options for every translation request, e.g. {num_gpu = 27} (layers on the
    # GPU) or {presence_penalty = 0.0}. Empty: the model's own defaults.
    extra_options: dict[str, float | int] = Field(default_factory=dict)
    # Optional: `mangatl setup` creates `model` from this GGUF instead of pulling it. With
    # the text-only Qwen3.5 9B the vision encoder (~1.3 GB of VRAM) is not loaded, so more
    # layers fit on a 6 GB GPU (23 of 33 instead of 18 of 34).
    gguf: GgufImport | None = None


class TranslatorConfig(BaseModel):
    backend: TranslatorBackend = "ollama"
    target_variant: SpanishVariant = "es-419"
    honorifics: HonorificsMode = "keep"
    sfx_mode: SfxMode = "annotate"
    pages_per_block: int = Field(4, ge=1, le=8)
    # Lines of the previous block (source + translation) repeated as context.
    previous_lines: int = Field(6, ge=0)
    # Retries when the model returns invalid JSON or misses region IDs.
    max_retries: int = Field(2, ge=0, le=5)
    # Manual backend: split the exported prompt into parts of at most this many characters.
    manual_part_chars: int = Field(20000, ge=2000)
    # Two passes per block: Japanese -> English draft -> Spanish (about twice as slow).
    pivot_english: bool = False
    # Experimental: stricter speaker rules in the prompt (no invented names, "narración" for
    # caption boxes) and name examples that cannot be taken for characters.
    clear_speaker_rules: bool = False
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


class FontsConfig(BaseModel):
    """Font file per text style. Relative paths are resolved against the project root."""

    normal: Path = Path("assets/fonts/ComicNeue-Bold.ttf")
    shout: Path = Path("assets/fonts/Bangers-Regular.ttf")
    whisper: Path = Path("assets/fonts/ComicNeue-Italic.ttf")
    thought: Path = Path("assets/fonts/ComicNeue-BoldItalic.ttf")
    narration: Path = Path("assets/fonts/ComicNeue-Regular.ttf")
    sfx: Path = Path("assets/fonts/Bangers-Regular.ttf")


class TypesettingConfig(BaseModel):
    fonts: FontsConfig = Field(default_factory=FontsConfig)
    # Font sizes are given in pixels for a page of this height and scaled to each page.
    reference_page_height: int = Field(1600, ge=100)
    min_font_px: float = Field(15, gt=0)
    base_font_px: float = Field(22, gt=0)  # used to estimate each bubble's capacity
    max_font_px: float = Field(32, gt=0)
    line_spacing: float = Field(1.12, ge=0.8, le=2.0)
    # Inner margin, as a fraction of the interior's shorter side.
    margin_ratio: float = Field(0.08, ge=0.0, le=0.4)
    # No bubble may use a font larger than this factor times the page median.
    page_size_spread: float = Field(1.2, ge=1.0)
    hyphenate: bool = True
    text_color: str = "#000000"
    # White outline around text lettered on artwork (fraction of the font size).
    stroke_ratio: float = Field(0.12, ge=0.0, le=0.5)


class ExportConfig(BaseModel):
    # PNG pages are always written; these are the packaged formats.
    formats: list[Literal["cbz", "pdf"]] = Field(default_factory=lambda: ["cbz", "pdf"])


class PipelineConfig(BaseModel):
    # Run detection + OCR in a child process: when it exits, the CUDA context (~0.3-0.5 GB)
    # is freed too, which torch.cuda.empty_cache() alone cannot do.
    isolate_vision: bool = True


class SetupConfig(BaseModel):
    # Never let a download leave less than this much free space on the project drive.
    min_free_disk_gb: float = Field(3.0, ge=0.0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MANGATL_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    root: Path = Field(default_factory=find_project_root)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    inpaint: InpaintConfig = Field(default_factory=InpaintConfig)
    translator: TranslatorConfig = Field(default_factory=TranslatorConfig)
    typesetting: TypesettingConfig = Field(default_factory=TypesettingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    setup: SetupConfig = Field(default_factory=SetupConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # .env is reserved for optional API keys (Phase 4); it is not a config source.
        return init_settings, env_settings, TomlConfigSettingsSource(settings_cls)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the project root."""
        return path if path.is_absolute() else (self.root / path).resolve()


def apply_overrides(settings: Settings, overrides: list[str]) -> Settings:
    """Copy of `settings` with dotted `key=value` overrides (values parsed as TOML literals;
    bare words are strings). Raises ValueError on malformed items or unknown keys."""
    import tomllib

    data = settings.model_dump()
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Se esperaba CLAVE=VALOR: {item}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # e.g. translator.ollama.model=qwen3.5:4b
        node = data
        *parents, leaf = key.strip().split(".")
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"Clave desconocida: {key}")
            node = node[part]
        if leaf not in node:
            raise ValueError(f"Clave desconocida: {key}")
        node[leaf] = value
    return Settings.model_validate(data)


def default_config_path(root: Path | None = None) -> Path:
    env_path = os.environ.get("MANGATL_CONFIG")
    if env_path:
        return Path(env_path)
    return (root or find_project_root()) / "config.toml"


def load_settings(config_file: Path | None = None) -> Settings:
    """Load settings from `config_file` (or the default config.toml, if present)."""
    path = config_file or default_config_path()
    if config_file is not None and not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    toml_file = path if path.is_file() else None

    class _FileSettings(Settings):
        model_config = SettingsConfigDict(toml_file=toml_file)

    return _FileSettings()
