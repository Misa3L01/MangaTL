import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { GlossaryView } from "./components/GlossaryView";
import { PageCanvas, STATUS_COLORS } from "./components/PageCanvas";
import { RegionEditor, STATUS_LABELS } from "./components/RegionEditor";
import type { BBox, Fonts, Page, Project, ProjectSummary, Region, RegionPatch } from "./types";

type View = "rendered" | "original" | "clean" | "side";
type Tab = "editor" | "glossary";

const VIEW_LABELS: Record<View, string> = {
  rendered: "Traducida",
  original: "Original",
  clean: "Limpia",
  side: "Lado a lado",
};

interface Toast {
  text: string;
  error: boolean;
}

// Deep links: #<project>?page=66&region=P066-B02&view=side&tab=glossary
const params = new URLSearchParams(location.search);
const initialPage = Number(params.get("page")) || null;
const initialRegion = params.get("region");
const initialView = (params.get("view") as View | null) ?? "rendered";
const initialTab = (params.get("tab") as Tab | null) ?? "editor";

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [projectId, setProjectId] = useState<string | null>(() => decodeURIComponent(location.hash.slice(1)) || null);
  const [project, setProject] = useState<Project | null>(null);
  const [pageNo, setPageNo] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<View>(initialView in VIEW_LABELS ? initialView : "rendered");
  const [tab, setTab] = useState<Tab>(initialTab);
  const [version, setVersion] = useState(() => Date.now());
  const [busy, setBusy] = useState<string | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const [onlyReview, setOnlyReview] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [fonts, setFonts] = useState<Fonts | null>(null);

  const notify = useCallback((text: string, error = false) => {
    setToast({ text, error });
    window.setTimeout(() => setToast((t) => (t?.text === text ? null : t)), error ? 7000 : 3500);
  }, []);

  useEffect(() => {
    api.projects().then(setProjects).catch((e: Error) => notify(e.message, true));
    api.fonts().then(setFonts).catch(() => undefined);
  }, [notify]);

  useEffect(() => {
    if (!projectId && projects?.length) setProjectId(projects[0].id);
  }, [projects, projectId]);

  useEffect(() => {
    if (!projectId) return;
    location.hash = encodeURIComponent(projectId);
    setProject(null);
    setSelectedId(null);
    api
      .project(projectId)
      .then((p) => {
        setProject(p);
        const wanted = p.pages.find((pg) => pg.number === initialPage);
        setPageNo(wanted?.number ?? p.pages[0]?.number ?? null);
        if (wanted && initialRegion && wanted.regions.some((r) => r.id === initialRegion)) {
          setSelectedId(initialRegion);
        }
        setVersion(Date.now());
      })
      .catch((e: Error) => notify(e.message, true));
  }, [projectId, notify]);

  const page = useMemo(() => project?.pages.find((p) => p.number === pageNo) ?? null, [project, pageNo]);
  const region = page?.regions.find((r) => r.id === selectedId) ?? null;
  const reviewList = useMemo(
    () => project?.pages.flatMap((p) => p.regions.filter((r) => r.status === "needs_review").map((r) => ({ page: p, region: r }))) ?? [],
    [project],
  );

  const replacePage = (updated: Page) =>
    setProject((p) => (p ? { ...p, pages: p.pages.map((pg) => (pg.number === updated.number ? updated : pg)) } : p));

  const replaceRegion = (updated: Region) =>
    setProject((p) =>
      p
        ? {
            ...p,
            pages: p.pages.map((pg) => ({ ...pg, regions: pg.regions.map((r) => (r.id === updated.id ? updated : r)) })),
          }
        : p,
    );

  const rerender = async (number: number) => {
    if (!projectId) return;
    setBusy("Re-renderizando la página…");
    try {
      replacePage(await api.renderPage(projectId, number));
      setVersion(Date.now());
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setBusy(null);
    }
  };

  const saveRegion = async (target: Region, patch: RegionPatch) => {
    if (!projectId || !page) return;
    if (Object.keys(patch).length === 0) {
      notify("No hay cambios que guardar");
      return;
    }
    setBusy("Guardando…");
    try {
      replaceRegion(await api.patchRegion(projectId, target.id, patch));
    } catch (e) {
      notify((e as Error).message, true);
      setBusy(null);
      return;
    }
    await rerender(page.number);
    notify(`${target.id} guardada`);
  };

  const moveBox = (target: Region, box: BBox) => saveRegion(target, { text_box_override: box });

  const exportProject = async () => {
    if (!projectId) return;
    setBusy("Exportando…");
    try {
      const out = await api.exportProject(projectId);
      notify(`Exportado: ${out.pages} páginas${out.cbz ? " · CBZ" : ""}${out.pdf ? " · PDF" : ""}`);
      api.projects().then(setProjects).catch(() => undefined);
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setBusy(null);
    }
  };

  const goRegion = (delta: number) => {
    if (!page) return;
    const idx = page.regions.findIndex((r) => r.id === selectedId);
    const next = page.regions[idx + delta];
    if (next) setSelectedId(next.id);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (!project || !page) return;
      const idx = project.pages.findIndex((p) => p.number === page.number);
      if (e.key === "ArrowRight" || e.key === "PageDown") setPageNo(project.pages[Math.min(idx + 1, project.pages.length - 1)].number);
      if (e.key === "ArrowLeft" || e.key === "PageUp") setPageNo(project.pages[Math.max(idx - 1, 0)].number);
      if (e.key === "ArrowDown") goRegion(1);
      if (e.key === "ArrowUp") goRegion(-1);
      if (e.key === "Escape") setSelectedId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const kind = view === "side" ? "rendered" : view;
  const summary = projects?.find((p) => p.id === projectId);

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">MangaTL</span>
        <select
          className="project-select"
          value={projectId ?? ""}
          onChange={(e) => setProjectId(e.target.value)}
          aria-label="Proyecto"
        >
          {projects?.map((p) => (
            <option key={p.id} value={p.id}>
              {p.series} · cap. {p.chapter} · {p.pages} págs.{p.review ? ` · ${p.review} a revisar` : ""}
            </option>
          ))}
        </select>
        <nav className="tabs">
          <button className={tab === "editor" ? "active" : ""} onClick={() => setTab("editor")}>
            Editor
          </button>
          <button className={tab === "glossary" ? "active" : ""} onClick={() => setTab("glossary")}>
            Glosario
          </button>
        </nav>
        <span className="spacer" />
        {busy && <span className="busy">{busy}</span>}
        {summary && !summary.translated && <span className="muted small">Sin traducir todavía</span>}
        <button className="primary" disabled={!project || !!busy} onClick={exportProject}>
          Exportar PNG/CBZ/PDF
        </button>
      </header>

      {projects?.length === 0 && (
        <div className="empty">
          <h2>No hay proyectos</h2>
          <p>
            Traduce un capítulo con <code>uv run mangatl translate …</code> y vuelve a abrir el editor.
          </p>
        </div>
      )}

      {tab === "glossary" && project && <GlossaryView series={project.meta.series} notify={notify} />}

      {tab === "editor" && project && (
        <main className="layout">
          <aside className="sidebar">
            <label className="check">
              <input type="checkbox" checked={onlyReview} onChange={(e) => setOnlyReview(e.target.checked)} />
              Solo regiones a revisar ({reviewList.length})
            </label>
            {onlyReview ? (
              <ul className="review-list">
                {reviewList.map(({ page: p, region: r }) => (
                  <li key={r.id}>
                    <button
                      className={r.id === selectedId ? "active" : ""}
                      onClick={() => {
                        setPageNo(p.number);
                        setSelectedId(r.id);
                      }}
                    >
                      <strong>{r.id}</strong>
                      <span className="muted small">{r.notes[0] ?? ""}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <ul className="page-list">
                {project.pages.map((p) => (
                  <li key={p.number}>
                    <button
                      className={p.number === pageNo ? "active" : ""}
                      onClick={() => {
                        setPageNo(p.number);
                        setSelectedId(null);
                      }}
                    >
                      <span>Página {p.number}</span>
                      {p.review > 0 && <span className="badge">{p.review}</span>}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </aside>

          <section className="center">
            <div className="view-bar">
              {(Object.keys(VIEW_LABELS) as View[]).map((v) => (
                <button key={v} className={view === v ? "active" : ""} onClick={() => setView(v)}>
                  {VIEW_LABELS[v]}
                </button>
              ))}
              <label className="check">
                <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
                Mostrar regiones
              </label>
              <span className="spacer" />
              {page && (
                <button disabled={!!busy} onClick={() => rerender(page.number)}>
                  Re-renderizar página
                </button>
              )}
            </div>
            {page ? (
              <PageCanvas
                page={page}
                imageUrl={api.imageUrl(project.id, page.number, kind, version)}
                sideImageUrl={view === "side" ? api.imageUrl(project.id, page.number, "original", version) : null}
                selectedId={selectedId}
                showBoxes={showBoxes}
                onSelect={setSelectedId}
                onMoveBox={moveBox}
              />
            ) : (
              <p className="muted">Elige una página.</p>
            )}
            <p className="hint muted small">
              ← → cambiar de página · ↑ ↓ cambiar de región · clic en una región para editarla · arrastra el recuadro rosa
              para mover el texto
            </p>
          </section>

          <aside className="inspector">
            {region ? (
              <RegionEditor region={region} fonts={fonts} busy={!!busy} onSave={(patch) => saveRegion(region, patch)} />
            ) : (
              <div className="region-list">
                <h3>Regiones de la página {page?.number}</h3>
                <ul>
                  {page?.regions.map((r) => (
                    <li key={r.id}>
                      <button onClick={() => setSelectedId(r.id)}>
                        <span className="dot" style={{ background: STATUS_COLORS[r.status] }} title={STATUS_LABELS[r.status]} />
                        <span className="order">{r.reading_order}</span>
                        <span className="snippet">{r.translation || r.ocr_text || "—"}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </aside>
        </main>
      )}

      {toast && (
        <div className={`toast ${toast.error ? "error" : ""}`} role="status">
          {toast.text}
        </div>
      )}
    </div>
  );
}
