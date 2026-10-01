export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type RegionStatus = "auto" | "edited" | "needs_review" | "skipped";
export type TextStyle = "normal" | "shout" | "whisper" | "thought" | "narration" | "sfx";
export type RegionType = "speech_bubble" | "narration_box" | "text_on_art" | "sfx";

export interface Region {
  id: string;
  type: RegionType;
  bbox: BBox;
  bubble_bbox: BBox | null;
  reading_order: number;
  panel: number | null;
  ocr_text: string;
  ocr_confidence: number | null;
  source_text_corrected: string | null;
  speaker: string | null;
  translation: string | null;
  shorter_alternative: string | null;
  translator_note: string | null;
  translation_confidence: number | null;
  style: TextStyle;
  font: string | null;
  font_size: number | null;
  font_size_fixed: boolean;
  text_box_override: BBox | null;
  clean_method: "fill" | "lama" | "none";
  fits: boolean | null;
  used_shorter: boolean;
  status: RegionStatus;
  notes: string[];
}

export interface Page {
  number: number;
  source_name: string;
  width: number;
  height: number;
  is_spread: boolean;
  has_rendered: boolean;
  has_clean: boolean;
  review: number;
  regions: Region[];
}

export interface GlossaryEntry {
  source: string;
  target: string;
  category: "character" | "place" | "technique" | "term" | "other";
  notes: string | null;
  status: "approved" | "pending";
  chapter: string | null;
}

export interface Project {
  id: string;
  meta: {
    series: string;
    chapter: string;
    source_lang: string;
    target_variant: string;
    translator_backend: string;
    translator_model: string | null;
  };
  chapter_summary: string | null;
  pages: Page[];
}

export interface ProjectSummary {
  id: string;
  series: string;
  chapter: string;
  source_lang: string;
  model: string | null;
  pages: number;
  regions: number;
  review: number;
  translated: boolean;
  updated: string;
}

export interface Fonts {
  styles: Record<TextStyle, string>;
  available: string[];
}

export interface RegionPatch {
  translation?: string;
  shorter_alternative?: string;
  style?: TextStyle;
  font?: string;
  font_size?: number;
  auto_size?: boolean;
  text_box_override?: BBox;
  reset_text_box?: boolean;
  status?: RegionStatus;
}
