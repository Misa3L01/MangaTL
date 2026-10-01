import type { Fonts, GlossaryEntry, Page, Project, ProjectSummary, Region, RegionPatch } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* body is not JSON */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  projects: () => request<ProjectSummary[]>("/api/projects"),
  project: (id: string) => request<Project>(`/api/projects/${enc(id)}`),
  imageUrl: (id: string, page: number, kind: "original" | "clean" | "rendered", version: number) =>
    `/api/projects/${enc(id)}/pages/${page}/image/${kind}?v=${version}`,
  patchRegion: (id: string, regionId: string, patch: RegionPatch) =>
    request<Region>(`/api/projects/${enc(id)}/regions/${enc(regionId)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  renderPage: (id: string, page: number) =>
    request<Page>(`/api/projects/${enc(id)}/pages/${page}/render`, { method: "POST" }),
  exportProject: (id: string) =>
    request<Record<string, string | number>>(`/api/projects/${enc(id)}/export`, { method: "POST" }),
  glossary: (series: string) => request<GlossaryEntry[]>(`/api/series/${enc(series)}/glossary`),
  approve: (series: string, sources: string[] | null) =>
    request<{ approved: string[] }>(`/api/series/${enc(series)}/glossary/approve`, {
      method: "POST",
      body: JSON.stringify({ sources }),
    }),
  upsertTerm: (
    series: string,
    source: string,
    target: string,
    notes: string | null,
    category?: GlossaryEntry["category"],
  ) =>
    request<GlossaryEntry>(`/api/series/${enc(series)}/glossary/${enc(source)}`, {
      method: "PUT",
      body: JSON.stringify({ target, notes, category }),
    }),
  removeTerm: (series: string, source: string) =>
    request<{ removed: string }>(`/api/series/${enc(series)}/glossary/${enc(source)}`, {
      method: "DELETE",
    }),
  fonts: () => request<Fonts>("/api/fonts"),
};
