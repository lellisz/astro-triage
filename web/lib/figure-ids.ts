// Ids das figuras de cada pagina, derivados do JSON da pagina com a mesma regra do export
// (astro_triage.web_export.figures.charts.builders). Sem disco: serve a pagina e a view.
import type { Compare, Graficos, Overview } from "./schema";

export const zFigureId = (cls: string): string => `taxa-por-z-${cls.toLowerCase()}`;
export const gainFigureId = (model: string): string => `ganho-${model.replace(/_/g, "-")}`;

export function graficosFigureIds(g: Graficos): string[] {
  return [
    "calibracao",
    "taxa-por-g",
    ...g.error_by_z.series.map((s) => zFigureId(s.cls)),
    "confusao",
    ...(g.cmd === null ? [] : ["cmd-taxa"]),
  ];
}

export function overviewFigureIds(technical: Overview["technical"]): string[] {
  return ["quasar-pureza", "quasar-completude", "dsc-diferencas", ...technical.models.map((m) => gainFigureId(m.model))];
}

export const galaxiesFigureIds = (): string[] => ["galaxia-erros"];

export function compareFigureIds(c: Compare): string[] {
  return c.status === "ok" ? ["pureza-sistemas", "decisao-diferencas"] : [];
}
