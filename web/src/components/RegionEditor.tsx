import { useEffect, useState } from "react";
import type { Fonts, Region, RegionPatch, TextStyle } from "../types";
import { STATUS_COLORS } from "./PageCanvas";

const STYLE_LABELS: Record<TextStyle, string> = {
  normal: "Normal",
  shout: "Grito",
  whisper: "Susurro",
  thought: "Pensamiento",
  narration: "Narración",
  sfx: "Onomatopeya",
};

const TYPE_LABELS: Record<Region["type"], string> = {
  speech_bubble: "Globo",
  narration_box: "Cuadro de texto",
  text_on_art: "Texto sobre el dibujo",
  sfx: "Onomatopeya",
};

export const STATUS_LABELS: Record<Region["status"], string> = {
  auto: "Automática",
  edited: "Editada",
  needs_review: "A revisar",
  skipped: "Omitida (original)",
};

const CLEAN_LABELS: Record<Region["clean_method"], string> = {
  fill: "relleno",
  lama: "LaMa",
  none: "sin limpieza",
};

interface Props {
  region: Region;
  fonts: Fonts | null;
  busy: boolean;
  onSave: (patch: RegionPatch) => void;
}

function fontName(path: string) {
  return path.split("/").pop()?.replace(/\.(ttf|otf|ttc)$/i, "") ?? path;
}

export function RegionEditor({ region, fonts, busy, onSave }: Props) {
  const [translation, setTranslation] = useState(region.translation ?? "");
  const [shorter, setShorter] = useState(region.shorter_alternative ?? "");
  const [style, setStyle] = useState<TextStyle>(region.style);
  const [font, setFont] = useState(region.font ?? "");
  const [autoSize, setAutoSize] = useState(!region.font_size_fixed);
  const [size, setSize] = useState(region.font_size ?? 24);

  useEffect(() => {
    setTranslation(region.translation ?? "");
    setShorter(region.shorter_alternative ?? "");
    setStyle(region.style);
    setFont(region.font ?? "");
    setAutoSize(!region.font_size_fixed);
    setSize(region.font_size ?? 24);
  }, [region]);

  const save = () => {
    const patch: RegionPatch = {};
    if (translation !== (region.translation ?? "")) patch.translation = translation;
    if (shorter !== (region.shorter_alternative ?? "")) patch.shorter_alternative = shorter;
    if (style !== region.style) patch.style = style;
    if (font !== (region.font ?? "")) patch.font = font;
    if (autoSize && region.font_size_fixed) patch.auto_size = true;
    if (!autoSize && (!region.font_size_fixed || size !== region.font_size)) patch.font_size = size;
    onSave(patch);
  };

  const original = region.source_text_corrected ?? region.ocr_text;
  const styleFont = fonts?.styles[style];

  return (
    <div
      className="editor"
      onKeyDown={(e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          save();
        }
      }}
    >
      <div className="editor-head">
        <span className="region-id">{region.id}</span>
        <span className="chip" style={{ background: STATUS_COLORS[region.status] }}>
          {STATUS_LABELS[region.status]}
        </span>
      </div>
      <div className="muted small">
        {TYPE_LABELS[region.type]}
        {region.panel != null && ` · viñeta ${region.panel}`}
        {` · limpieza: ${CLEAN_LABELS[region.clean_method]}`}
        {region.ocr_confidence != null && ` · OCR ${Math.round(region.ocr_confidence * 100)} %`}
        {region.speaker && region.speaker !== "desconocido" && ` · habla: ${region.speaker}`}
      </div>

      {region.notes.length > 0 && (
        <ul className="notes">
          {region.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}

      <label className="field">
        <span>Texto original</span>
        <div className="original">{original || "—"}</div>
      </label>

      <label className="field">
        <span>Traducción</span>
        <textarea rows={4} value={translation} onChange={(e) => setTranslation(e.target.value)} />
      </label>

      <label className="field">
        <span>Versión corta (se usa si la traducción no cabe)</span>
        <input value={shorter} onChange={(e) => setShorter(e.target.value)} />
      </label>

      {region.translator_note && <p className="translator-note">Nota: {region.translator_note}</p>}

      <div className="row">
        <label className="field">
          <span>Estilo</span>
          <select value={style} onChange={(e) => setStyle(e.target.value as TextStyle)}>
            {(Object.keys(STYLE_LABELS) as TextStyle[]).map((s) => (
              <option key={s} value={s}>
                {STYLE_LABELS[s]}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Fuente</span>
          <select value={font} onChange={(e) => setFont(e.target.value)}>
            <option value="">{styleFont ? `La del estilo (${fontName(styleFont)})` : "La del estilo"}</option>
            {fonts?.available.map((f) => (
              <option key={f} value={f}>
                {fontName(f)}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="row size-row">
        <label className="check">
          <input type="checkbox" checked={autoSize} onChange={(e) => setAutoSize(e.target.checked)} />
          Tamaño automático
          {autoSize && region.font_size ? <span className="muted"> (ahora {region.font_size} px)</span> : null}
        </label>
        <input
          type="number"
          min={6}
          max={400}
          value={size}
          disabled={autoSize}
          onChange={(e) => setSize(Number(e.target.value))}
          aria-label="Tamaño de fuente en píxeles"
        />
        <span className="muted">px</span>
      </div>

      {region.fits === false && (
        <p className="warn">La traducción no cabe al tamaño mínimo: acórtala, usa la versión corta o agranda la caja.</p>
      )}

      <div className="field">
        <span>Caja de texto</span>
        <div className="muted small">
          {region.text_box_override
            ? "Movida a mano (arrastra o estira el recuadro rosa en la página)."
            : "Sigue la forma del globo. Arrastra o estira el recuadro rosa para cambiarla."}
        </div>
        {region.text_box_override && (
          <button className="link" disabled={busy} onClick={() => onSave({ reset_text_box: true })}>
            Volver a la forma del globo
          </button>
        )}
      </div>

      <div className="actions">
        <button className="primary" disabled={busy} onClick={save} title="Ctrl+Enter">
          Guardar y re-renderizar
        </button>
        <div className="row wrap">
          {region.status !== "edited" && (
            <button disabled={busy} onClick={() => onSave({ status: "edited" })}>
              Marcar como revisada
            </button>
          )}
          {region.status !== "skipped" ? (
            <button disabled={busy} onClick={() => onSave({ status: "skipped" })}>
              Conservar el original
            </button>
          ) : (
            <button disabled={busy} onClick={() => onSave({ status: "auto" })}>
              Volver a traducir aquí
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
