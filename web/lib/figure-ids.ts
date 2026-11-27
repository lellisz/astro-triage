// Ids das figuras de cada pagina, derivados do JSON da pagina com a mesma regra do export
// (astro_triage.web_export.figures.charts.builders). Sem disco: serve a pagina e a view.
import type { Compare, Graficos, Meta, Overview } from "./schema";

export const zFigureId = (cls: string): string => `taxa-por-z-${cls.toLowerCase()}`;
export const gainFigureId = (model: string): string => `ganho-${model.replace(/_/g, "-")}`;
export const quasarSummaryFigureId = (rule: string, prior: string): string =>
  `quasar-resumo-${rule}-${prior}`.replace(/_/g, "-").toLowerCase();

export function graficosFigureIds(g: Graficos): string[] {
  return [
    "calibracao",
    "taxa-por-g",
    "taxa-por-g-simples",
    ...g.error_by_z.series.map((s) => zFigureId(s.cls)),
    "confusao",
    "confusao-simples",
    ...(g.cmd === null ? [] : ["cmd-taxa"]),
  ];
}

export function overviewFigureIds(technical: Overview["technical"], states: Meta["states"]): string[] {
  return [
    "quasar-pureza",
    "quasar-completude",
    ...states.map((s) => quasarSummaryFigureId(s.rule, s.prior)),
    "dsc-diferencas",
    ...technical.models.map((m) => gainFigureId(m.model)),
  ];
}

export const galaxiesFigureIds = (): string[] => ["galaxia-erros", "galaxia-erros-simples"];

export function compareFigureIds(c: Compare): string[] {
  return c.status === "ok" ? ["pureza-sistemas", "pureza-sistemas-simples", "decisao-diferencas"] : [];
}
