import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { GlossaryEntry } from "../types";

const CATEGORY_LABELS: Record<GlossaryEntry["category"], string> = {
  character: "Personaje",
  place: "Lugar",
  technique: "Técnica",
  term: "Término",
  other: "Otro",
};

interface Props {
  series: string;
  notify: (text: string, error?: boolean) => void;
}

function Row({ entry, series, onChanged, notify }: { entry: GlossaryEntry; series: string; onChanged: () => void; notify: Props["notify"] }) {
  const [target, setTarget] = useState(entry.target);
  const [notes, setNotes] = useState(entry.notes ?? "");
  const [category, setCategory] = useState(entry.category);
  useEffect(() => {
    setTarget(entry.target);
    setNotes(entry.notes ?? "");
    setCategory(entry.category);
  }, [entry]);
  const dirty = target !== entry.target || notes !== (entry.notes ?? "") || category !== entry.category;
  const run = async (action: () => Promise<unknown>, done: string) => {
    try {
      await action();
      notify(done);
      onChanged();
    } catch (e) {
      notify((e as Error).message, true);
    }
  };
  return (
    <tr className={entry.status === "pending" ? "pending" : undefined}>
      <td className="ja">{entry.source}</td>
      <td>
        <input value={target} onChange={(e) => setTarget(e.target.value)} aria-label={`Traducción de ${entry.source}`} />
      </td>
      <td>
        <select value={category} onChange={(e) => setCategory(e.target.value as GlossaryEntry["category"])} aria-label={`Tipo de ${entry.source}`}>
          {(Object.keys(CATEGORY_LABELS) as GlossaryEntry["category"][]).map((c) => (
            <option key={c} value={c}>
              {CATEGORY_LABELS[c]}
            </option>
          ))}
        </select>
      </td>
      <td>
        <input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Cómo habla, cómo trata a otros…" aria-label={`Notas de ${entry.source}`} />
      </td>
      <td>{entry.status === "approved" ? "Aprobada" : "Pendiente"}</td>
      <td className="actions-cell">
        {(dirty || entry.status === "pending") && (
          <button
            className="primary"
            onClick={() =>
              run(
                () =>
                  dirty
                    ? api.upsertTerm(series, entry.source, target, notes || null, category)
                    : api.approve(series, [entry.source]),
                dirty ? "Entrada guardada y aprobada" : "Entrada aprobada",
              )
            }
          >
            {dirty ? "Guardar" : "Aprobar"}
          </button>
        )}
        <button onClick={() => run(() => api.removeTerm(series, entry.source), "Entrada eliminada")}>Eliminar</button>
      </td>
    </tr>
  );
}

export function GlossaryView({ series, notify }: Props) {
  const [entries, setEntries] = useState<GlossaryEntry[] | null>(null);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");

  const reload = useCallback(() => {
    api
      .glossary(series)
      .then(setEntries)
      .catch((e: Error) => notify(e.message, true));
  }, [series, notify]);
  useEffect(reload, [reload]);

  const pending = entries?.filter((e) => e.status === "pending").length ?? 0;

  return (
    <div className="glossary">
      <div className="glossary-head">
        <h2>Glosario · {series}</h2>
        <p className="muted">
          Las entradas <strong>aprobadas</strong> se envían al traductor como obligatorias en los próximos capítulos. Las
          propuestas por el modelo quedan pendientes hasta que las apruebes.
        </p>
        {pending > 0 && (
          <button
            className="primary"
            onClick={() =>
              api
                .approve(series, null)
                .then((r) => {
                  notify(`Aprobadas ${r.approved.length} entradas`);
                  reload();
                })
                .catch((e: Error) => notify(e.message, true))
            }
          >
            Aprobar las {pending} pendientes
          </button>
        )}
      </div>
      <form
        className="add-term"
        onSubmit={(e) => {
          e.preventDefault();
          if (!source.trim() || !target.trim()) return;
          api
            .upsertTerm(series, source.trim(), target.trim(), null)
            .then(() => {
              setSource("");
              setTarget("");
              notify("Entrada añadida");
              reload();
            })
            .catch((err: Error) => notify(err.message, true));
        }}
      >
        <input placeholder="Original (p. ej. 斉藤)" value={source} onChange={(e) => setSource(e.target.value)} />
        <input placeholder="Traducción fija (p. ej. Saitō)" value={target} onChange={(e) => setTarget(e.target.value)} />
        <button type="submit">Añadir</button>
      </form>
      {entries === null ? (
        <p className="muted">Cargando…</p>
      ) : entries.length === 0 ? (
        <p className="muted">Todavía no hay entradas para esta serie.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Original</th>
              <th>Traducción</th>
              <th>Tipo</th>
              <th>Notas</th>
              <th>Estado</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <Row key={e.source} entry={e} series={series} onChanged={reload} notify={notify} />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
