// Testes independentes do modo simples das figuras: roteamento por pagina (sem e com tecnico=1)
// e paridade entre lib/figure-ids.ts e o figures.json gerado por builders() na fixture.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { CompareView } from "@/components/pages/CompareView";
import { GalaxiesView } from "@/components/pages/GalaxiesView";
import { GraficosView } from "@/components/pages/GraficosView";
import { OverviewView } from "@/components/pages/OverviewView";
import {
  compareFigureIds,
  galaxiesFigureIds,
  graficosFigureIds,
  overviewFigureIds,
  quasarSummaryFigureId,
} from "@/lib/figure-ids";
import type { Compare, Galaxies, Graficos, Meta, Overview } from "@/lib/schema";
import { FIGS } from "./fixture-figures";

const nav = vi.hoisted(() => ({ search: "" }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => "/",
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

const FIX = fileURLToPath(new URL("./fixtures", import.meta.url));
const load = <T,>(name: string): T =>
  JSON.parse(readFileSync(path.join(FIX, "fixture_v0", `${name}.json`), "utf-8")) as T;
const meta = load<Meta>("meta");
const overview = load<Overview>("overview");
const galaxies = load<Galaxies>("galaxies");
const graficos = load<Graficos>("graficos");
const compare = load<Compare>("compare");

const isSimple = (id: string) => id.endsWith("-simples") || id.startsWith("quasar-resumo-");
const ids = (html: string) => [...html.matchAll(/data-figure="([^"]+)"/g)].map((m) => m[1]);
const at = (search: string, el: () => React.ReactElement) => {
  nav.search = search;
  try {
    return ids(renderToStaticMarkup(el()));
  } finally {
    nav.search = "";
  }
};

const pages = {
  overview: () => (
    <OverviewView
      meta={meta}
      rows={overview.rows}
      accuracyNote={overview.accuracy_note}
      technical={overview.technical}
      figures={FIGS}
    />
  ),
  graficos: () => <GraficosView meta={meta} graficos={graficos} figures={FIGS} />,
  galaxias: () => <GalaxiesView meta={meta} galaxies={galaxies} figures={FIGS} />,
  comparacao: () => <CompareView meta={meta} compare={compare} figures={FIGS} />,
};
const pageIds = {
  overview: overviewFigureIds(overview.technical, meta.states),
  graficos: graficosFigureIds(graficos),
  galaxias: galaxiesFigureIds(),
  comparacao: compareFigureIds(compare),
};
const SIMPLE: Record<keyof typeof pages, string[]> = {
  overview: [quasarSummaryFigureId(meta.default_state.rule, meta.default_state.prior)],
  graficos: ["taxa-por-g-simples", "confusao-simples"],
  galaxias: ["galaxia-erros-simples"],
  comparacao: ["pureza-sistemas-simples"],
};

describe("criterio 5: roteamento modo simples x tecnico=1", () => {
  it("pre-condicoes da fixture: compare ok, estado padrao as_shipped/P2, 4 estados", () => {
    expect(compare.status).toBe("ok");
    expect(SIMPLE.overview).toEqual(["quasar-resumo-as-shipped-p2"]);
    expect(meta.states).toHaveLength(4);
  });

  for (const page of Object.keys(pages) as (keyof typeof pages)[]) {
    it(`${page}: sem tecnico, so os ids simples esperados`, () => {
      for (const search of ["", "tecnico=0"]) {
        expect(at(search, pages[page]), `${page} ?${search}`).toEqual(SIMPLE[page]);
      }
    });

    it(`${page}: tecnico=1 renderiza os ids tecnicos e nenhum simples`, () => {
      const got = at("tecnico=1", pages[page]);
      expect(got.filter(isSimple)).toEqual([]);
      // cada figura tecnica da pagina aparece exatamente uma vez
      const tech = pageIds[page].filter((id) => !isSimple(id));
      expect([...got].sort()).toEqual([...tech].sort());
    });
  }

  it("visao geral simples mostra o quasar-resumo do estado da URL (2 estados diferentes)", () => {
    const p3 = at("regra=prior_matched&prior=P3", pages.overview);
    const p1 = at("regra=prior_matched&prior=P1", pages.overview);
    expect(p3).toEqual(["quasar-resumo-prior-matched-p3"]);
    expect(p1).toEqual(["quasar-resumo-prior-matched-p1"]);
    // e o SVG embutido e o do estado, nao outro
    nav.search = "regra=prior_matched&prior=P3";
    const html = renderToStaticMarkup(pages.overview());
    nav.search = "";
    expect(html).toContain(FIGS["quasar-resumo-prior-matched-p3"].svg.slice(0, 200));
    expect(html).not.toContain(FIGS["quasar-resumo-as-shipped-p2"].svg.slice(0, 200));
  });

  it("estado invalido na URL cai no padrao; tecnico=1 com estado nao mostra resumo", () => {
    expect(at("regra=xx&prior=P9", pages.overview)).toEqual(SIMPLE.overview);
    expect(at("regra=prior_matched&prior=P1&tecnico=1", pages.overview).filter(isSimple)).toEqual([]);
  });
});

describe("criterio 6: paridade figure-ids.ts x figures.json (builders) na fixture", () => {
  const manifest = JSON.parse(
    readFileSync(path.join(FIX, "fixture_v0", "figures", "figures.json"), "utf-8"),
  ) as { figures: Record<string, unknown> };
  const all = Object.values(pageIds).flat();

  it("sem id repetido entre paginas", () => {
    expect(new Set(all).size).toBe(all.length);
  });

  it("uniao dos ids das paginas == chaves do figures.json gerado", () => {
    expect([...all].sort()).toEqual(Object.keys(manifest.figures).sort());
  });

  it("um quasar-resumo por item de meta.states, com o mesmo nome do Python", () => {
    expect(pageIds.overview.filter((id) => id.startsWith("quasar-resumo-"))).toEqual([
      "quasar-resumo-as-shipped-p2",
      "quasar-resumo-prior-matched-p1",
      "quasar-resumo-prior-matched-p2",
      "quasar-resumo-prior-matched-p3",
    ]);
  });
});
