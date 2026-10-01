import { useEffect, useRef, useState } from "react";
import { Group, Image as KImage, Layer, Rect, Stage, Text, Transformer } from "react-konva";
import type Konva from "konva";
import type { BBox, Page, Region } from "../types";

export const STATUS_COLORS: Record<Region["status"], string> = {
  auto: "#2e9d5b",
  edited: "#2f6fdb",
  needs_review: "#e08a00",
  skipped: "#8a8a8a",
};

function useHtmlImage(url: string | null): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) return;
    const image = new window.Image();
    image.onload = () => setImg(image);
    image.src = url;
    return () => {
      image.onload = null;
    };
  }, [url]);
  return img;
}

function useSize(ref: React.RefObject<HTMLDivElement | null>) {
  const [size, setSize] = useState({ w: 800, h: 600 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      setSize({ w: entry.contentRect.width, h: entry.contentRect.height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [ref]);
  return size;
}

export function textBox(region: Region): BBox {
  return region.text_box_override ?? region.bubble_bbox ?? region.bbox;
}

interface Props {
  page: Page;
  imageUrl: string;
  sideImageUrl: string | null; // original shown on the left in side-by-side mode
  selectedId: string | null;
  showBoxes: boolean;
  onSelect: (id: string | null) => void;
  onMoveBox: (region: Region, box: BBox) => void;
}

export function PageCanvas({ page, imageUrl, sideImageUrl, selectedId, showBoxes, onSelect, onMoveBox }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const { w, h } = useSize(wrapRef);
  const image = useHtmlImage(imageUrl);
  const boxRef = useRef<Konva.Rect>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const selected = page.regions.find((r) => r.id === selectedId) ?? null;

  const panels = sideImageUrl ? 2 : 1;
  const gap = sideImageUrl ? 12 : 0;
  const scale = Math.max(0.05, Math.min((w - gap) / panels / page.width, h / page.height));
  const stageW = Math.round(page.width * scale);
  const stageH = Math.round(page.height * scale);

  useEffect(() => {
    const tr = trRef.current;
    if (!tr) return;
    tr.nodes(boxRef.current && selected ? [boxRef.current] : []);
    tr.getLayer()?.batchDraw();
  }, [selected, image, scale]);

  const commit = () => {
    const node = boxRef.current;
    if (!node || !selected) return;
    const x = node.x();
    const y = node.y();
    const bw = node.width() * node.scaleX();
    const bh = node.height() * node.scaleY();
    node.scaleX(1);
    node.scaleY(1);
    const clamp = (v: number, max: number) => Math.round(Math.min(Math.max(v, 0), max));
    onMoveBox(selected, {
      x0: clamp(x, page.width),
      y0: clamp(y, page.height),
      x1: clamp(x + bw, page.width),
      y1: clamp(y + bh, page.height),
    });
  };

  const box = selected ? textBox(selected) : null;

  return (
    <div className="canvas-wrap" ref={wrapRef}>
      <div className="canvas-row" style={{ gap }}>
        {sideImageUrl && (
          <img className="side-image" src={sideImageUrl} width={stageW} height={stageH} alt="Página original" />
        )}
        <Stage
          width={stageW}
          height={stageH}
          onMouseDown={(e) => {
            if (e.target === e.target.getStage() || e.target.name() === "page") onSelect(null);
          }}
        >
          <Layer scaleX={scale} scaleY={scale}>
            {image && <KImage image={image} name="page" width={page.width} height={page.height} />}
            {showBoxes &&
              page.regions.map((r) => {
                const b = r.bubble_bbox ?? r.bbox;
                const color = STATUS_COLORS[r.status];
                const isSel = r.id === selectedId;
                return (
                  <Group key={r.id} onClick={() => onSelect(r.id)} onTap={() => onSelect(r.id)}>
                    <Rect
                      x={b.x0}
                      y={b.y0}
                      width={b.x1 - b.x0}
                      height={b.y1 - b.y0}
                      stroke={color}
                      strokeWidth={(isSel ? 3 : 1.5) / scale}
                      dash={r.type === "text_on_art" ? [8 / scale, 6 / scale] : undefined}
                      fill={isSel ? "rgba(47,111,219,0.08)" : "rgba(0,0,0,0.001)"}
                    />
                    <Rect
                      x={b.x0}
                      y={b.y0}
                      width={Math.max(22, String(r.reading_order).length * 11) / scale}
                      height={18 / scale}
                      fill={color}
                    />
                    <Text
                      x={b.x0 + 4 / scale}
                      y={b.y0 + 2 / scale}
                      text={String(r.reading_order)}
                      fontSize={14 / scale}
                      fontStyle="bold"
                      fill="#fff"
                    />
                  </Group>
                );
              })}
            {selected && box && (
              <Rect
                ref={boxRef}
                x={box.x0}
                y={box.y0}
                width={box.x1 - box.x0}
                height={box.y1 - box.y0}
                stroke="#d6007a"
                strokeWidth={2 / scale}
                dash={[6 / scale, 4 / scale]}
                draggable
                onDragEnd={commit}
                onTransformEnd={commit}
              />
            )}
            <Transformer
              ref={trRef}
              rotateEnabled={false}
              keepRatio={false}
              ignoreStroke
              anchorSize={9}
              borderStroke="#d6007a"
              anchorStroke="#d6007a"
              boundBoxFunc={(oldBox, newBox) => (newBox.width < 16 || newBox.height < 16 ? oldBox : newBox)}
            />
          </Layer>
        </Stage>
      </div>
    </div>
  );
}
