import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Tabs, type TabItem } from "@/components/kit/Tabs";
import { CompareView } from "@/components/pages/CompareView";
import { GalaxiesView } from "@/components/pages/GalaxiesView";
import MethodPage from "@/app/metodo/page";
import SkyPage from "@/app/ceu/page";
import { OverviewTechnical } from "@/components/pages/OverviewTechnical";
import { SkyView } from "@/components/pages/SkyView";
import { StateControls } from "@/components/shell/StateControls";
import {
  chartLabels,
  chartsHref,
  compareHref,
  compareLabels,
  controls,
  galaxiesLabels as L_gal,
  methodLabels,
  overviewLabels as L_ov,
  skyLabels as L_sky,
} from "@/content/labels";
import { ptInt, ptPct } from "@/lib/fmt";
import type { Compare, Est, Galaxies, Meta, Method, Overview, Sky } from "@/lib/schema";
import { FIGS } from "./fixture-figures";

const nav = vi.hoisted(() => ({ search: "", path: "/" }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => nav.path,
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

// /metodo e um componente de servidor que le JSON em tempo de build: o teste injeta objetos.
const store = vi.hoisted(() => ({ pages: {} as Record<string, unknown> }));
vi.mock("@/lib/data", () => ({ loadPage: (key: string) => store.pages[key] }));

// Sem DOM (ambiente node): o teste das abas executa o componente com hooks minimos.
// Fora do teste de abas (h.on = false) os hooks reais sao usados.
const h = vi.hoisted(() => ({ on: false, slots: [] as unknown[], i: 0 }));
vi.mock("react", async (orig) => {
  const actual = await orig<typeof import("react")>();
  const real = ((actual as unknown as { default?: typeof actual }).default ?? actual) as typeof actual;
  return {
    ...real,
    default: real,
    useState: (init: unknown) => {
      if (!h.on) return real.useState(init);
      const at = h.i++;
      if (!(at in h.slots)) h.slots[at] = init;
      const set = (v: unknown) => {
        h.slots[at] = typeof v === "function" ? (v as (p: unknown) => unknown)(h.slots[at]) : v;
      };
      return [h.slots[at], set];
    },
    useRef: (init: unknown) => {
      if (!h.on) return real.useRef(init);
      const at = h.i++;
      if (!(at in h.slots)) h.slots[at] = { current: init };
      return h.slots[at];
    },
    useId: () => (h.on ? "tid" : real.useId()),
    useSyncExternalStore: (sub: never, get: () => unknown, srv: () => unknown) =>
      h.on ? get() : real.useSyncExternalStore(sub, get, srv),
  };
});

const FIX = fileURLToPath(new URL("./fixtures", import.meta.url));
const CHARTS_CSS = fileURLToPath(new URL("../components/charts/charts.module.css", import.meta.url));
const load = <T,>(name: string): T =>
  JSON.parse(readFileSync(path.join(FIX, "fixture_v0", `${name}.json`), "utf-8")) as T;
const meta = load<Meta>("meta");
const galaxies = load<Galaxies>("galaxies");
const compare = load<Compare>("compare");
const method = load<Method>("method");
const overview = load<Overview>("overview");
const sky = load<Sky>("sky");

type Ok = Extract<Compare, { status: "ok" }>;
const ok = compare as Ok;
const NA_EST: Est = { point: null, lo: null, hi: null, text: "n/a", ci_text: "n/a" };
const clone = <T,>(v: T): T => structuredClone(v);

const decode = (s: string) =>
  s
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
const textOf = (html: string) => decode(html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " "));
const count = (html: string, needle: string | RegExp) =>
  typeof needle === "string" ? html.split(needle).length - 1 : (html.match(needle) ?? []).length;
// Elemento <g ... data-row="<key>"> completo (balanceando <g>/</g>); um item por ocorrencia.
const rowChunks = (html: string, key: string): string[] => {
  const out: string[] = [];
  const marker = `data-row="${key}"`;
  let from = 0;
  for (;;) {
    const at = html.indexOf(marker, from);
    if (at < 0) return out;
    const start = html.lastIndexOf("<g", at);
    let depth = 0;
    let end = start;
    for (const m of html.slice(start).matchAll(/<g[\s>]|<\/g>/g)) {
      depth += m[0] === "</g>" ? -1 : 1;
      if (depth === 0) {
        end = start + m.index + m[0].length;
        break;
      }
    }
    out.push(html.slice(start, end));
    from = at + marker.length;
  }
};
const escHtml = (x: string) =>
  x.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
const escRe = (x: string) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// Tabela da DataTable cuja legenda (<p id=X>) tem o texto dado: a <table aria-labelledby=X>.
const tableAfterCaption = (html: string, caption: string): string | undefined => {
  const p = html.match(new RegExp(`<p id="([^"]+)"[^>]*>${escRe(escHtml(caption))}</p>`));
  if (!p) return undefined;
  const all = [...html.matchAll(/<table[^>]*>[\s\S]*?<\/table>/g)].map((m) => m[0]);
  return all.find((x) => x.includes(`aria-labelledby="${p[1]}"`));
};
// Conteudo do <section aria-labelledby=id> (Panel do kit); os paineis testados nao aninham section.
const panel = (html: string, id: string): string => {
  const m = html.match(new RegExp(`<section[^>]*aria-labelledby="${escRe(id)}"[^>]*>[\\s\\S]*?</section>`));
  if (!m) throw new Error(`painel ${id} ausente`);
  return m[0];
};

const figure = (html: string, id: string): string => {
  const m = html.match(new RegExp(`<figure[^>]*id="${id}"[\\s\\S]*?</figure>`));
  if (!m) throw new Error(`figure ${id} ausente`);
  return m[0];
};

const noFig = (html: string): string => html.replace(/<figure[^>]*data-figure="[^"]*"[\s\S]*?<\/figure>/g, "");

const renderGal = (g: Galaxies, search = "") => {
  nav.search = search;
  return renderToStaticMarkup(<GalaxiesView figures={FIGS} meta={meta} galaxies={g} />);
};

describe("A1 Galaxias: Est com point null vira n/a, sem zero, barra ou ponto", () => {
  const g = clone(galaxies);
  // prior_matched P2 sem estimativa nas duas series (pureza e completude)
  g.blocks.prior_matched[1].purity = NA_EST;
  g.blocks.prior_matched[1].completeness = NA_EST;
  const html = renderGal(g, "tecnico=1");

  it("barras por prior: a linha nula mostra n/a, sem barra, marcador, intervalo ou valor", () => {
    const chunks = rowChunks(html, "prior_matched-P2");
    expect(chunks).toHaveLength(2); // pureza e completude
    for (const c of chunks) {
      expect(c).not.toContain("data-bar");
      expect(c).not.toContain("data-ci");
      expect(c).not.toMatch(/<(circle|rect|polygon|polyline)\b/);
      expect(c).toContain('aria-label="não disponível"');
      expect(textOf(c)).toContain("n/a");
      expect(textOf(c)).not.toMatch(/\d\s*%/);
    }
  });

  it("barras por prior: as vizinhas nao-nulas continuam com barra e IC", () => {
    for (const key of ["prior_matched-P1", "prior_matched-P3", "as_shipped-P2"]) {
      const chunks = rowChunks(html, key);
      expect(chunks, key).toHaveLength(2);
      for (const c of chunks) {
        expect(c).toContain("data-bar");
        expect(c).toContain("data-ci");
      }
    }
  });

  it("erros gerais viram a figura galaxia-erros; G e z sao so tabela (Tarefa 37)", () => {
    const errors = panel(html, "gal-errors");
    expect(count(errors, 'data-figure="galaxia-erros"')).toBe(1);
    expect(count(html, /data-row="err-\d+"/g)).toBe(0);
    expect(html).not.toContain('id="gal-g-chart"');
    expect(html).not.toContain('id="gal-z-chart"');
    expect(count(html, /data-row="(g|z)-\d+"/g)).toBe(0);
    // as faixas nulas de G e z viram n/a na tabela, sem numero de taxa
    const tables = [...html.matchAll(/<table[^>]*>[\s\S]*?<\/table>/g)].map((m) => m[0]);
    const nullRows = [...galaxies.technical.by_g, ...galaxies.technical.by_z].filter((r) => r.rate.point === null);
    expect(nullRows.length).toBeGreaterThan(0);
    const trs = tables.join("").split("<tr>").map((x) => textOf(x.split("</tr>")[0]));
    for (const r of nullRows) {
      const row = trs.find((t) => t.includes(r.label));
      expect(row, r.label).toBeDefined();
      expect(row).toContain("n/a");
    }
  });

  it("faixas abaixo do piso nao somem: linhas das tabelas = contagem do JSON, com IC (Tarefa 37)", () => {
    expect(galaxies.technical.by_g.length).toBe(5);
    expect(galaxies.technical.by_z.length).toBe(7);
    expect(galaxies.technical.by_g.some((r) => !r.reportable)).toBe(true);
    expect(galaxies.technical.by_z.some((r) => !r.reportable)).toBe(true);
    const tech = html.slice(html.indexOf('id="gal-tech"'));
    const bodies = [...tech.matchAll(/<tbody>([\s\S]*?)<\/tbody>/g)].map((m) => m[1]);
    expect(bodies.map((b) => count(b, "<tr>"))).toEqual([5, 7]);
    // coluna de IC rotulada como Wilson por objeto, com o ci_text de cada faixa reportavel
    expect(tech).toContain("Intervalo de confiança (Wilson, por objeto)");
    for (const r of [...galaxies.technical.by_g, ...galaxies.technical.by_z]) {
      expect(textOf(tech)).toContain(r.label);
      if (r.rate.point !== null) expect(textOf(tech)).toContain(r.rate.ci_text);
    }
    expect(textOf(tech)).toContain(L_gal.galBandsJoinNote);
  });
});

describe("A2 Galaxias: P3 parcial, marcador distinto e series separadas", () => {
  nav.search = "";
  const html = renderGal(galaxies);
  const css = readFileSync(CHARTS_CSS, "utf-8");

  it("precondicao: o estado padrao nao seleciona P3", () => {
    expect(meta.default_state.prior).not.toBe("P3");
    expect(galaxies.blocks.as_shipped.find((b) => b.prior === "P3")?.partial).toBe(true);
    expect(galaxies.blocks.prior_matched.find((b) => b.prior === "P3")?.partial).toBe(true);
  });

  it("Tarefa 38: 6 barras em cada grafico (2 regras x P1, P2, P3), P3 marcado como parcial", () => {
    const blocks = [...galaxies.blocks.as_shipped, ...galaxies.blocks.prior_matched];
    expect(blocks).toHaveLength(6);
    for (const id of ["gal-purity-chart", "gal-completeness-chart"]) {
      const fig = figure(html, id);
      const keys = [...fig.matchAll(/data-row="([^"]+)"/g)].map((m) => m[1]);
      expect(keys, id).toEqual([
        "as_shipped-P1",
        "as_shipped-P2",
        "as_shipped-P3",
        "prior_matched-P1",
        "prior_matched-P2",
        "prior_matched-P3",
      ]);
      expect(count(fig, 'data-partial="true"'), id).toBe(2);
      expect(count(textOf(fig), `P3 (${chartLabels.partialTag})`), id).toBe(2);
      for (const rule of ["as_shipped", "prior_matched"]) {
        expect(rowChunks(fig, `${rule}-P3`)[0]).toContain('data-partial="true"');
        expect(rowChunks(fig, `${rule}-P1`)[0]).not.toContain("data-partial");
      }
    }
  });

  it("Tarefa 38: rotulo de meta.states quando existe, senao rule_labels + prior; valor e IC do bloco", () => {
    const fig = figure(html, "gal-purity-chart");
    for (const rule of ["as_shipped", "prior_matched"] as const) {
      for (const b of galaxies.blocks[rule]) {
        const c = rowChunks(fig, `${rule}-${b.prior}`)[0];
        const st = meta.states.find((s) => s.rule === rule && s.prior === b.prior);
        expect(textOf(c), `${rule}-${b.prior}`).toContain(st ? st.label : `${meta.rule_labels[rule]} · ${b.prior}`);
        if (b.purity.point !== null) {
          expect(textOf(c)).toContain(b.purity.text);
          expect(c).toContain("data-ci");
        }
      }
    }
    // a barra selecionada e a do estado padrao
    const sel = [...fig.matchAll(/data-row="([^"]+)"[^>]*aria-current="true"/g)].map((m) => m[1]);
    expect(sel).toEqual([`${meta.default_state.rule}-${meta.default_state.prior}`]);
  });

  it("Tarefas 38/39: PriorSeries saiu do codigo (componente apagado e sem import)", () => {
    const web = fileURLToPath(new URL("..", import.meta.url));
    expect(existsSync(path.join(web, "components", "charts", "PriorSeries.tsx"))).toBe(false);
    expect(existsSync(path.join(web, "components", "charts", "HistogramPair.tsx"))).toBe(false);
    const view = readFileSync(path.join(web, "components", "pages", "GalaxiesView.tsx"), "utf-8");
    expect(view).not.toMatch(/PriorSeries/);
    expect(html).not.toContain("data-series=");
  });

  it("nenhum atributo inline de traco no markup", () => {
    expect(html).not.toMatch(/stroke-dasharray=/i);
  });

  it("nenhuma cor fora de tokens/currentColor: atributos, style inline e css do grafico", () => {
    // As figuras matplotlib trazem as cores proprias no SVG; a regra vale para o markup React.
    const page = noFig(html);
    const attrs = [...page.matchAll(/\s(fill|stroke)="([^"]*)"/g)].map((m) => m[2]);
    for (const v of attrs) expect(v).toMatch(/^(none|currentColor|var\(--[\w-]+\))$/);
    expect(page).not.toMatch(/style="[^"]*(#|rgb|hsl)/i);
    const decls = [...css.matchAll(/(?:^|[;{\s])(?:fill|stroke)\s*:\s*([^;}]+)/g)].map((m) => m[1].trim());
    expect(decls.length).toBeGreaterThan(5);
    for (const v of decls) expect(v, v).toMatch(/^(none|currentColor|inherit|var\(--[\w-]+\))$/);
    expect(css).not.toMatch(/(fill|stroke)\s*:\s*(#|rgb|hsl)/i);
  });
});

describe("A3 Galaxias: tabelas tecnicas so com tecnico=1; power_note", () => {
  const nets = Object.values(galaxies.networks).filter((n) => n !== null);
  const panelOf = (html: string, id: string) => {
    const at = html.indexOf(`id="${id}"`);
    expect(at, id).toBeGreaterThan(-1);
    const next = html.indexOf("<section", at + 1);
    return html.slice(at, next < 0 ? undefined : next);
  };

  it("sem tecnico: so a tabela das redes; sem painel tecnico nem graficos G/z", () => {
    const html = renderGal(galaxies, "");
    expect(html).not.toContain('id="gal-tech"');
    expect(count(html, "<table")).toBe(1);
    expect(panelOf(html, "gal-networks")).toContain("<table");
    expect(html).not.toContain('id="gal-g-chart"');
    expect(html).not.toContain('id="gal-z-chart"');
  });

  it("tecnico=0 explicito tambem esconde o painel tecnico", () => {
    const html = renderGal(galaxies, "tecnico=0");
    expect(html).not.toContain('id="gal-tech"');
    expect(count(html, "<table")).toBe(1);
  });

  it("tecnico=1 mostra o painel e duas tabelas tecnicas com uma linha por faixa do JSON", () => {
    const html = renderGal(galaxies, "tecnico=1");
    expect(html).toContain('id="gal-tech"');
    expect(count(html, "<table")).toBe(3);
    const bodyRows = [...html.matchAll(/<tbody>([\s\S]*?)<\/tbody>/g)].map((m) => count(m[1], "<tr>"));
    expect(bodyRows).toEqual([nets.length, galaxies.technical.by_g.length, galaxies.technical.by_z.length]);
  });

  it("Tarefa 36: redes como tabela (uma linha por rede, com D, IC, delta e veredito), sem #gal-net, com link", () => {
    const html = renderGal(galaxies, "");
    const panel = panelOf(html, "gal-networks");
    expect(html).not.toContain('id="gal-net"');
    expect(nets).toHaveLength(2);
    const body = panel.match(/<tbody>([\s\S]*?)<\/tbody>/)![1];
    expect(count(body, "<tr>")).toBe(2);
    for (const n of nets) {
      for (const s of [n.label, n.diff.text, n.diff.ci_text, n.word]) expect(textOf(body), s).toContain(s);
    }
    expect(panel).toMatch(/href="\/comparacao-dl\/?(\?[^"]*)?"/);
  });

  // REV-01: a coluna "Margem δ" (4a celula: rede, D, IC, delta, veredito) usa a unidade de D.
  const netCells = (g: Galaxies) => {
    const panel = panelOf(renderGal(g, ""), "gal-networks");
    const body = panel.match(/<tbody>([\s\S]*?)<\/tbody>/)![1];
    return [...body.matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map((r) =>
      [...r[1].matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/g)].map((c) => textOf(c[1]).trim()),
    );
  };

  it("REV-01: Margem δ da fixture (0.02) e exatamente '2,00 p.p.', na unidade de D, sem '0.02'", () => {
    expect(nets.map((n) => n.delta)).toEqual([0.02, 0.02]);
    const rows = netCells(galaxies);
    expect(rows).toHaveLength(2);
    for (const cells of rows) {
      expect(cells).toHaveLength(5);
      expect(cells[3]).toBe("2,00 p.p.");
      expect(cells[3]).not.toContain("0.02");
      expect(cells[1].endsWith(" p.p."), `D="${cells[1]}"`).toBe(true);
    }
  });

  it("REV-01 controle: δ = 0.005 vira '0,50 p.p.' (x100, 2 casas, nao fixo)", () => {
    const g = clone(galaxies);
    for (const n of Object.values(g.networks)) if (n) n.delta = 0.005;
    const rows = netCells(g);
    expect(rows).toHaveLength(2);
    for (const cells of rows) expect(cells[3]).toBe("0,50 p.p.");
  });

  it("Tarefa 35: ponto fixo dentro de #gal-networks e fora do bloco de contexto", () => {
    const html = renderGal(galaxies, "");
    const fixed = galaxies.fixed_point_text;
    expect(fixed).toContain("ponto fixo da comparação com as redes");
    expect(count(html, fixed)).toBe(1);
    expect(panelOf(html, "gal-networks")).toContain(fixed);
    const ctx = html.match(/<div class="[^"]*context[^"]*">[\s\S]*?<\/div>/)![0];
    expect(ctx).not.toContain(fixed);
    expect(textOf(html)).toContain(L_gal.errorsSub);
    expect(L_gal.errorsSub).toContain("classe de maior probabilidade (regra como entregue)");
  });

  it("power_note nao nulo aparece uma vez; nulo nao deixa rastro", () => {
    const g = clone(galaxies);
    g.power_note = "NOTA-DE-PODER-XYZ";
    const html = renderGal(g, "");
    expect(count(textOf(html), "NOTA-DE-PODER-XYZ")).toBe(1);
    expect(galaxies.power_note).toBeNull();
    expect(textOf(renderGal(galaxies, ""))).not.toContain("NOTA-DE-PODER");
  });
});

describe("A4 Comparacao: unavailable/error e 3 cartoes", () => {
  const down = (status: "unavailable" | "error", error_text: string | null): Compare => ({
    ...compare,
    status,
    error_text,
    fixed_point: null,
    answer_text: null,
    reasons_text: null,
    protocol_accepted: null,
    pairs: null,
    grid: null,
    power_note: null,
    purity: null,
    rule_lines: null,
    descriptive: null,
    limitations: null,
  });

  it.each([
    ["unavailable", null],
    ["error", "Falha controlada ao ler o resultado."],
  ] as const)("%s: estado claro, sem digito, sem cartoes, sem grafico", (status, err) => {
    const html = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={down(status, err)} />);
    const text = textOf(html);
    expect(text).toContain(compareLabels.unavailable);
    expect(text).toContain(compareLabels.unavailableHint);
    expect(text).not.toMatch(/\d/);
    expect(html).not.toContain("data-system");
    expect(html).not.toContain("<svg");
    expect(html).not.toContain("<table");
    expect(html).not.toContain('role="tab"');
    if (err) {
      expect(text).toContain(err);
      expect(html).toContain('role="status"');
    } else {
      expect(html).not.toContain('role="status"');
    }
  });

  it("status error com sobras de purity/grid no objeto: nenhum numero vaza", () => {
    const leaky = { ...compare, status: "error", error_text: "Falha controlada." } as unknown as Compare;
    const html = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={leaky} />);
    const text = textOf(html);
    expect(text).not.toMatch(/\d/);
    expect(html).not.toContain("data-system");
    expect(text).not.toContain("40,00");
  });

  const three = clone(ok);
  three.purity = [
    ok.purity[1],
    { ...ok.purity[1], system: "ftt_full", net: "FTT", label: "FT-Transformer completo", median: 0.5, low: 0.4, high: 0.6 },
    ok.purity[0],
  ];
  const html3 = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={three} />);

  it("ok com 3 sistemas: 3 cartoes, exatamente um de referencia, referencia primeiro", () => {
    const cards = [...html3.matchAll(/<li class="([^"]*)" data-system="([^"]+)"/g)];
    expect(cards.map((m) => m[2])).toEqual(["lgbm_full", "mlp_full", "ftt_full"]);
    expect(cards.filter((m) => /cardRef/.test(m[1]))).toHaveLength(1);
    expect(cards[0][1]).toMatch(/cardRef/);
    expect(count(html3, `>${compareLabels.referenceTag}</p>`)).toBe(1);
    expect(textOf(html3)).toContain(compareLabels.referenceTag);
    // so os nao-referencia mostram "Veredito do par"
    expect(count(textOf(html3), `${compareLabels.pairVerdict}:`)).toBe(2);
  });

  it("rede sem par no JSON mostra a frase de par ausente, nunca vazio", () => {
    const noPair = clone(three);
    noPair.pairs = noPair.pairs.filter((p) => p.net !== "FTT");
    const t = textOf(renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={noPair} />));
    expect(t).toContain(compareLabels.pairMissing);
  });
});

describe("A5 Comparacao: faixa min-max rotulada 'nao e IC'", () => {
  // A faixa min-max (figura pureza-sistemas) e do modo tecnico.
  nav.search = "tecnico=1";
  const html = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={ok} />);
  nav.search = "";
  const panel = (() => {
    const m = html.match(/<div role="tabpanel" id="[^"]*-panel-pureza"[\s\S]*?(?=<div role="tabpanel"|$)/);
    if (!m) throw new Error("painel de pureza ausente");
    return m[0];
  })();
  const attrs = [...panel.matchAll(/(?:aria-label|aria-labelledby)="([^"]*)"/g)].map((m) => m[1]).join(" ");
  const visible = `${textOf(panel)} ${decode(attrs)}`;

  it("o texto da faixa traz 'nao e IC'", () => {
    expect(compareLabels.notCi).toBe("não é IC");
    const p = panel.match(/<p class="[^"]*callout[^"]*">([\s\S]*?)<\/p>/);
    expect(p).not.toBeNull();
    expect(textOf(p![1])).toContain(`(${compareLabels.notCi})`);
    expect(textOf(p![1])).toContain("mínimo ao máximo");
    // cada detalhe de sistema tambem
    for (const s of ok.purity) expect(textOf(panel)).toContain(s.detail_text);
  });

  it("fora de 'nao e IC' nao ha IC isolado, 'intervalo de confianca' nem CI no painel (texto e aria)", () => {
    const stripped = visible.replace(/não é IC/g, "");
    expect(stripped).not.toMatch(/\bIC\b/);
    expect(stripped).not.toMatch(/intervalo de confiança/i);
    expect(stripped).not.toMatch(/\bCI\b/);
    expect(count(visible, "não é IC")).toBeGreaterThanOrEqual(1 + ok.purity.length);
  });

  it("a faixa min-max e a figura pureza-sistemas, uma vez no painel", () => {
    expect(count(panel, 'data-figure="pureza-sistemas"')).toBe(1);
    expect(count(panel, "<figure")).toBe(1);
  });
});

describe("A6 Comparacao: abas", () => {
  nav.search = "";
  const html = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={ok} />);

  it("markup: 4 abas, cada aria-controls aponta para um tabpanel existente e vice-versa", () => {
    const tabs = [...html.matchAll(/<button[^>]*role="tab"[^>]*>/g)].map((m) => m[0]);
    expect(tabs).toHaveLength(4);
    const panels = [...html.matchAll(/<div role="tabpanel" id="([^"]+)" aria-labelledby="([^"]+)"/g)];
    expect(panels).toHaveLength(4);
    for (const t of tabs) {
      const controls = t.match(/aria-controls="([^"]+)"/)![1];
      const id = t.match(/\sid="([^"]+)"/)![1];
      const p = panels.find((x) => x[1] === controls);
      expect(p, controls).toBeDefined();
      expect(p![2]).toBe(id);
    }
    // sem JS (SSR) nenhum painel fica oculto: o conteudo nao depende da troca de aba
    expect(html).not.toMatch(/role="tabpanel"[^>]*hidden/);
  });

  const items: TabItem[] = ["a", "b", "c", "d"].map((k) => ({ key: k, label: k, content: k }));
  type El = { props: Record<string, unknown> };
  const walk = (n: unknown, out: El[]): El[] => {
    if (Array.isArray(n)) n.forEach((c) => walk(c, out));
    else if (n && typeof n === "object" && "props" in n) {
      out.push(n as El);
      walk((n as El).props.children, out);
    }
    return out;
  };
  const mount = () => {
    h.on = true;
    h.slots = [];
    const focus = items.map(() => vi.fn());
    const render = () => {
      h.i = 0;
      const els = walk(Tabs({ label: "Secoes", tabs: items }), []);
      const tabs = els.filter((e) => e.props.role === "tab");
      tabs.forEach((e, i) => (e.props.ref as (x: unknown) => void)({ focus: focus[i] }));
      return { tabs, panels: els.filter((e) => e.props.role === "tabpanel") };
    };
    const press = (from: number, key: string) => {
      const ev = { key, preventDefault: vi.fn() };
      (render().tabs[from].props.onKeyDown as (e: unknown) => void)(ev);
      return ev;
    };
    return { render, press, focus };
  };
  afterEach(() => {
    h.on = false;
  });

  const state = (r: ReturnType<ReturnType<typeof mount>["render"]>) => ({
    selected: r.tabs.map((t) => t.props["aria-selected"]),
    tabindex: r.tabs.map((t) => t.props.tabIndex),
  });

  it("estado inicial: so a primeira aba selecionada e com tabindex 0; paineis ocultos exceto o ativo", () => {
    const { render } = mount();
    const r = render();
    expect(state(r)).toEqual({ selected: [true, false, false, false], tabindex: [0, -1, -1, -1] });
    expect(r.panels.map((p) => p.props.hidden)).toEqual([false, true, true, true]);
    for (const [i, t] of r.tabs.entries()) {
      expect(r.panels.some((p) => p.props.id === t.props["aria-controls"]), `tab ${i}`).toBe(true);
    }
  });

  it("ArrowRight/ArrowLeft movem selecao e foco, com volta circular; preventDefault so em tecla tratada", () => {
    const { render, press, focus } = mount();
    const ev = press(0, "ArrowRight");
    expect(ev.preventDefault).toHaveBeenCalledOnce();
    expect(state(render())).toEqual({ selected: [false, true, false, false], tabindex: [-1, 0, -1, -1] });
    expect(focus[1]).toHaveBeenCalledOnce();
    press(1, "ArrowRight");
    press(2, "ArrowRight");
    expect(state(render()).selected).toEqual([false, false, false, true]);
    press(3, "ArrowRight"); // volta ao inicio
    expect(state(render()).selected).toEqual([true, false, false, false]);
    expect(focus[0]).toHaveBeenCalledOnce();
    press(0, "ArrowLeft"); // volta ao fim
    expect(state(render())).toEqual({ selected: [false, false, false, true], tabindex: [-1, -1, -1, 0] });
    press(3, "ArrowLeft");
    expect(state(render()).selected).toEqual([false, false, true, false]);
  });

  it("Home e End", () => {
    const { render, press, focus } = mount();
    press(0, "End");
    expect(state(render()).selected).toEqual([false, false, false, true]);
    expect(focus[3]).toHaveBeenCalledOnce();
    press(3, "Home");
    expect(state(render()).selected).toEqual([true, false, false, false]);
    expect(focus[0]).toHaveBeenCalledOnce();
    // exatamente uma aba com tabindex 0 e um painel visivel em qualquer estado
    const r = render();
    expect(r.tabs.filter((t) => t.props.tabIndex === 0)).toHaveLength(1);
    expect(r.panels.filter((p) => p.props.hidden === false)).toHaveLength(1);
  });

  it("tecla nao tratada: nada muda e o evento nao e cancelado", () => {
    const { render, press, focus } = mount();
    const ev = press(0, "a");
    expect(ev.preventDefault).not.toHaveBeenCalled();
    expect(state(render()).selected).toEqual([true, false, false, false]);
    for (const f of focus) expect(f).not.toHaveBeenCalled();
  });
});

describe("A7 Comparacao: forest plot de diferencas", () => {
  const v = clone(ok);
  v.grid[2].diff = NA_EST; // celula sem d
  v.descriptive = [
    { pair: "MLP", role: "pior", cls: "QSO", d: -0.03, lo: -0.05, hi: -0.01, diff: { point: -0.03, lo: -0.05, hi: -0.01, text: "-0,0300", ci_text: "IC 95%: -0,0500 a -0,0100" }, text: "t" },
    { pair: "FTT", role: "desc", cls: "GALAXY", d: 0, lo: 0, hi: 0, diff: NA_EST, text: "t" },
    { pair: "FTT", role: "desc", cls: "QSO", d: 0.02, lo: 0.01, hi: 0.03, diff: { point: 0.02, lo: 0.01, hi: 0.03, text: "+0,0200", ci_text: "IC 95%: +0,0100 a +0,0300" }, text: "t" },
  ];
  // A figura de decisao e a faixa min-max sao do modo tecnico.
  nav.search = "tecnico=1";
  const html = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={v} />);
  nav.search = "";

  it("decisao: a figura decisao-diferencas aparece uma vez", () => {
    expect(count(html, 'data-figure="decisao-diferencas"')).toBe(1);
  });

  it("descritivo (Tarefa 31): sem grafico #cmp-descriptive; tabela com uma linha por item e a coluna de IC", () => {
    expect(html).not.toContain('id="cmp-descriptive"');
    expect(count(html, 'data-figure="decisao-diferencas"')).toBe(1);
    expect(count(html, 'data-figure="pureza-sistemas"')).toBe(1);
    expect(count(html, "<figure")).toBe(2);
    // A DataTable poe a legenda num <p> antes da tabela (aria-labelledby), nao num <caption>.
    const t = tableAfterCaption(html, compareLabels.descriptiveChart);
    expect(t).toBeDefined();
    const body = t!.match(/<tbody>([\s\S]*?)<\/tbody>/)![1];
    expect(count(body, "<tr>")).toBe(v.descriptive.length);
    const bodyRows = body.split("<tr>").slice(1).map((r) => textOf(r));
    v.descriptive.forEach((d, i) => {
      expect(bodyRows[i]).toContain(d.diff.text);
      expect(bodyRows[i]).toContain(d.diff.ci_text);
    });
    expect(bodyRows[1]).toContain("n/a");
  });

  it("descritivo com 14 itens (como o full_v0): 14 linhas, cada uma com IC", () => {
    const many = clone(ok);
    many.descriptive = Array.from({ length: 14 }, (_, k) => ({
      ...v.descriptive[0],
      pair: k % 2 ? "FTT" : "MLP",
      diff: { ...v.descriptive[0].diff, ci_text: `IC-${k}` },
    }));
    const h14 = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={many} />);
    expect(h14).not.toContain('id="cmp-descriptive"');
    const t = [...h14.matchAll(/<table[^>]*>[\s\S]*?<\/table>/g)].map((m) => m[0]).find((x) => x.includes("IC-0"))!;
    expect(count(t.match(/<tbody>([\s\S]*?)<\/tbody>/)![1], "<tr>")).toBe(14);
    for (let k = 0; k < 14; k++) expect(t).toContain(`IC-${k}<`);
  });

  it("sem itens descritivos: mensagem, sem segundo grafico", () => {
    nav.search = "tecnico=1";
    const h2 = renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={ok} />);
    nav.search = "";
    expect(h2).not.toContain('id="cmp-descriptive"');
    expect(textOf(h2)).toContain(compareLabels.noDescriptive);
    expect(count(h2, 'data-figure="decisao-diferencas"')).toBe(1);
  });
});

describe("A8 Metodo: todos os campos do method.json aparecem", () => {
  const withMethod = (m: Method): string => {
    store.pages = { meta, method: m };
    nav.search = "";
    return renderToStaticMarkup(<MethodPage />);
  };
  const html = withMethod(method);
  const text = textOf(html);

  it("texto corrido, fatos, mistura, limites, listas tecnicas, glossario e fontes", () => {
    for (const s of [method.lead, method.method_text, method.mix_text, method.simple]) expect(text).toContain(s);
    for (const k of Object.keys(method.facts) as (keyof Method["facts"])[]) {
      expect(text, `facts.${k}`).toContain(method.facts[k].text);
      expect(text).toContain(methodLabels.facts[k]);
    }
    for (const c of ["STAR", "GALAXY", "QSO"] as const) expect(text).toContain(method.mix[c].text);
    for (const s of [...method.three_limits, ...method.reading, ...method.usage, ...method.rule_lines]) {
      expect(text).toContain(s);
    }
    for (const l of method.limitations) {
      expect(text).toContain(l.text);
      expect(text).toContain(l.original);
    }
    for (const g of method.glossary) {
      expect(text, g.term).toContain(g.term);
      if (g.definition) expect(text).toContain(g.definition);
    }
    for (const s of method.sources) expect(text).toContain(s);
  });

  it("cada link https aparece com href, rotulo e nota, e abre com rel seguro", () => {
    expect(method.links.length).toBeGreaterThan(3);
    for (const l of method.links) {
      expect(l.url.startsWith("https://")).toBe(true);
      expect(html).toContain(`href="${l.url}"`);
      expect(text).toContain(l.label);
      expect(text).toContain(l.note);
    }
    expect(count(html, 'rel="noopener noreferrer"')).toBe(method.links.length);
  });

  it("links nao-https (http, javascript:, invalido) sao descartados", () => {
    const m = clone(method);
    m.links = [
      ...m.links,
      { label: "LINK-HTTP", note: "n", url: "http://exemplo.org/x" },
      { label: "LINK-JS", note: "n", url: "javascript:alert(1)" },
      { label: "LINK-RUIM", note: "n", url: "nao e url" },
    ];
    const out = withMethod(m);
    for (const bad of ["LINK-HTTP", "LINK-JS", "LINK-RUIM"]) expect(out).not.toContain(bad);
    expect(out).not.toContain("javascript:");
    expect(count(out, 'rel="noopener noreferrer"')).toBe(method.links.length);
  });

  it("Tarefa 19: o export web grava usage vazio e a secao 'uso' nao aparece", () => {
    expect(method.usage).toEqual([]);
    expect(html).not.toContain('id="uso"');
    expect(text).not.toMatch(/dispersão por objeto/i);
  });

  it("details/summary para leitura, regras e limitacoes; original aninhado so se difere", () => {
    for (const id of ["leitura", "regras", "limitacoes"]) {
      expect(html, id).toMatch(new RegExp(`<details[^>]*id="${id}"`));
    }
    const differing = method.limitations.filter((l) => l.original !== l.text).length;
    expect(differing).toBeGreaterThan(0);
    expect(count(html, "<details")).toBe(3 + differing);
    expect(count(html, "<summary")).toBe(3 + differing);
    expect(count(html, /<summary[^>]*>[^<]+<\/summary>/g)).toBe(3 + differing);
    // com usage preenchido a secao volta (o componente nao a esconde por engano)
    const withUsage = withMethod({ ...clone(method), usage: ["USO-XYZ"] });
    expect(withUsage).toMatch(/<details[^>]*id="uso"/);
    expect(textOf(withUsage)).toContain("USO-XYZ");
  });

  it("fechados por padrao e abertos com tecnico=1", () => {
    expect(html).not.toMatch(/<details[^>]*\sopen/);
    store.pages = { meta, method };
    nav.search = "tecnico=1";
    const open = renderToStaticMarkup(<MethodPage />);
    expect(count(open, /<details[^>]*\sopen=""/g)).toBe(3);
  });

  it("Tarefa 20: a leitura rotula o IC por faixa como Wilson por objeto", () => {
    const r = method.reading.find((s) => s.includes("Wilson"));
    expect(r).toContain("método de Wilson, por objeto");
    expect(r).toContain("regiões vizinhas");
    expect(text).toContain(r!);
  });

  it("listas vazias nao geram details vazio", () => {
    const m = clone(method);
    m.reading = [];
    m.usage = [];
    const out = withMethod(m);
    expect(out).not.toContain('id="leitura"');
    expect(out).not.toContain('id="uso"');
    expect(out).toContain('id="regras"');
  });

  it("Num com value null vira 'nao disponivel', nunca 0 (mesmo com text enganoso)", () => {
    const m = clone(method);
    m.facts.n_boot = { value: null, text: "0" };
    m.facts.n_test = { value: null, text: "n/a" };
    m.mix.QSO = { value: null, text: "0%" };
    const out = withMethod(m);
    const kpi = (label: string) => out.match(new RegExp(`${label}</p><p[^>]*>([^<]*)</p>`))?.[1];
    expect(kpi(methodLabels.facts.n_boot)).toBe(methodLabels.notAvailable);
    expect(kpi(methodLabels.facts.n_test)).toBe(methodLabels.notAvailable);
    expect(kpi(methodLabels.classes.QSO)).toBe(methodLabels.notAvailable);
    // e um zero verdadeiro continua sendo zero
    const z = clone(method);
    z.facts.n_boot = { value: 0, text: "0" };
    expect(withMethod(z).match(new RegExp(`${methodLabels.facts.n_boot}</p><p[^>]*>([^<]*)</p>`))?.[1]).toBe("0");
  });
});

describe("Metodo: frase do modo simples so fora do modo tecnico", () => {
  const render = (search: string): string => {
    store.pages = { meta, method };
    nav.search = search;
    return textOf(renderToStaticMarkup(<MethodPage />));
  };

  it("aparece no modo simples e some com tecnico=1", () => {
    expect(render("")).toContain(method.simple);
    expect(render("tecnico=1")).not.toContain(method.simple);
  });
});

describe("Comparacao: sem barra de ajustes (a pagina nao tem modo tecnico)", () => {
  const bar = (path: string): string => {
    nav.path = path;
    nav.search = "";
    const out = renderToStaticMarkup(<StateControls meta={meta} />);
    nav.path = "/";
    return out;
  };

  it("em /comparacao-dl, com ou sem barra final, nada e renderizado", () => {
    expect(bar(compareHref)).toBe("");
    expect(bar(`${compareHref}/`)).toBe("");
  });

  it("nas outras paginas o checkbox do modo tecnico continua", () => {
    expect(bar("/")).toContain(controls.tech);
    expect(bar("/galaxias")).toContain(controls.tech);
  });
});

describe("Graficos: barra sem regra nem prior, com modo tecnico", () => {
  const bar = (path: string): string => {
    nav.path = path;
    nav.search = "";
    const out = renderToStaticMarkup(<StateControls meta={meta} />);
    nav.path = "/";
    return out;
  };

  it("em /graficos, com ou sem barra final, so o checkbox tecnico aparece", () => {
    for (const p of [chartsHref, chartsHref.replace(/\/$/, "")]) {
      const html = bar(p);
      expect(html).toContain(controls.tech);
      expect(html).toContain('type="checkbox"');
      expect(html).not.toContain('type="radio"');
      expect(html).not.toContain(controls.rule);
      expect(html).not.toContain(controls.prior);
    }
  });

  it("nas outras rotas os controles de regra e prior continuam", () => {
    for (const p of ["/", "/galaxias"]) {
      const html = bar(p);
      expect(html).toContain('type="radio"');
      expect(html).toContain(controls.rule);
      expect(html).toContain(controls.prior);
    }
    expect(bar(compareHref)).toBe("");
  });
});

describe("Tarefa 30: status do protocolo dentro do Resultado da Comparacao", () => {
  it("'em proposta' aparece dentro de #cmp-result quando protocol_accepted e false", () => {
    expect(ok.protocol_accepted).toBe(false);
    const res = textOf(panel(renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={ok} />), "cmp-result"));
    expect(res).toContain("em proposta");
    expect(res).toContain(compareLabels.protocolProposed);
  });

  it("aceito e desconhecido mudam o texto do painel; 'em proposta' some", () => {
    const acc = clone(ok);
    acc.protocol_accepted = true;
    const a = textOf(panel(renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={acc} />), "cmp-result"));
    expect(a).toContain(compareLabels.protocolAccepted);
    expect(a).not.toContain("em proposta");
    const unk = clone(ok);
    unk.protocol_accepted = null;
    const u = textOf(panel(renderToStaticMarkup(<CompareView meta={meta} figures={FIGS} compare={unk} />), "cmp-result"));
    expect(u).toContain(compareLabels.protocolUnknown);
    expect(u).not.toContain("em proposta");
  });
});

describe("Tarefa 32: /ceu/ sem regra nem prior", () => {
  afterEach(() => {
    nav.path = "/";
    nav.search = "";
  });

  it("barra de ajustes em /ceu/: sem radios de regra e prior, com o checkbox tecnico", () => {
    for (const where of ["/ceu/", "/ceu"]) {
      nav.path = where;
      const html = renderToStaticMarkup(<StateControls meta={meta} />);
      expect(html, where).not.toMatch(/name="regra"/);
      expect(html, where).not.toMatch(/name="prior"/);
      expect(html, where).toMatch(/type="checkbox"/);
      expect(html, where).toContain(escHtml(controls.tech));
    }
  });

  it("controle: em /galaxias/ os radios de regra continuam", () => {
    nav.path = "/galaxias/";
    expect(renderToStaticMarkup(<StateControls meta={meta} />)).toMatch(/name="regra"/);
  });

  it("pagina /ceu/ sem o chip de estado", () => {
    store.pages = { meta, sky };
    nav.path = "/ceu/";
    const html = renderToStaticMarkup(<SkyPage />);
    expect(html).toContain(escHtml(sky.texts.title));
    expect(html).not.toContain(`aria-label="${escHtml(controls.currentState)}"`);
    expect(html).not.toMatch(/name="regra"/);
  });
});

describe("Tarefa 33: faixas do ceu com coluna de IC e banda abaixo do piso em n/a", () => {
  const bandsTable = (s: Sky) => {
    nav.search = "tecnico=1";
    const html = panel(renderToStaticMarkup(<SkyView meta={meta} sky={s} />), "sky-bands");
    nav.search = "";
    return html;
  };
  const bodyRows = (html: string) => html.match(/<tbody>([\s\S]*?)<\/tbody>/)![1].split("<tr>").slice(1);

  it("coluna de IC; reportavel mostra taxa e IC, abaixo do piso mostra n/a com o rotulo do piso", () => {
    const html = bandsTable(sky);
    expect(textOf(html.match(/<thead>[\s\S]*?<\/thead>/)![0])).toContain(L_sky.colRateCi);
    const rows = bodyRows(html).map(textOf);
    expect(rows).toHaveLength(sky.bands.length);
    expect(sky.bands[0].band).toBe("A");
    expect(sky.bands[0].reportable).toBe(false);
    expect(sky.bands.some((b) => b.reportable)).toBe(true);
    sky.bands.forEach((b, i) => {
      if (b.reportable) {
        expect(b.rate.ci_text).toMatch(/^IC 95%/);
        expect(rows[i]).toContain(b.rate.text);
        expect(rows[i]).toContain(b.rate.ci_text);
        expect(rows[i]).not.toContain(L_sky.bandBelowFloor);
      } else {
        expect(rows[i]).toContain(`n/a (${L_sky.bandBelowFloor})`);
        expect(rows[i]).not.toMatch(/\d,\d%/);
      }
    });
  });

  it("banda A com objetos mas abaixo do piso: contagens ficam, taxa n/a, sem IC", () => {
    const s = clone(sky);
    s.bands[0] = { ...s.bands[0], n_pixels: 13, n: 1284, errors: 29, reportable: false, rate: NA_EST };
    const rowA = textOf(bodyRows(bandsTable(s))[0]);
    expect(rowA).toContain(ptInt(1284));
    expect(rowA).toContain(`n/a (${L_sky.bandBelowFloor})`);
    expect(rowA).not.toMatch(/IC 95%/);
  });
});

describe("Tarefas 40b, 40c e validacao: Visao geral tecnica", () => {
  const tech = overview.technical;
  const body = (t: string) => t.match(/<tbody>([\s\S]*?)<\/tbody>/)![1].split("<tr>").slice(1);

  it("40b: DSC sem c_sdsc no DOM e sem #ov-dsc-f1; populacao e prior em portugues; tabela de F1 com IC", () => {
    const t = clone(tech);
    t.dsc.population = "c_sdsc"; // o campo cru nunca deve ser exibido
    t.dsc.prior = "c_sdsc";
    const html = renderToStaticMarkup(<OverviewTechnical figures={FIGS} technical={t} />);
    expect(html).not.toContain("c_sdsc");
    expect(html).not.toContain('id="ov-dsc-f1"');
    expect(html).not.toContain('id="ov-calibration"');
    const text = textOf(html);
    expect(text).toContain(t.dsc.population_text);
    expect(text).toContain(t.dsc.prior_text);
    const f1 = tableAfterCaption(html, L_ov.dscScoreTable);
    expect(f1).toBeDefined();
    const rows = body(f1!).map(textOf);
    expect(rows).toHaveLength(t.dsc.macro_f1.length);
    t.dsc.macro_f1.forEach((m, i) => {
      expect(rows[i]).toContain(m.label);
      expect(rows[i]).toContain(m.est.text);
      expect(rows[i]).toContain(m.est.ci_text);
    });
  });

  it("40c: ConfidenceTable com uma linha por faixa; contagens e soma dos erros lidas da fixture", () => {
    const hist = tech.confidence_hist;
    expect(hist).toHaveLength(8);
    const html = renderToStaticMarkup(<OverviewTechnical figures={FIGS} technical={tech} />);
    const rows = body(html.match(/<table id="ov-confidence"[\s\S]*?<\/table>/)![0]);
    expect(rows).toHaveLength(hist.length);
    let wrongSum = 0;
    rows.forEach((r, i) => {
      const cells = r.split("<td>").slice(1).map((x) => textOf(x).trim());
      expect(cells).toHaveLength(2);
      expect(cells[0]).toBe(ptInt(hist[i].n_right));
      expect(cells[1]).toBe(ptInt(hist[i].n_wrong));
      expect(textOf(r)).toContain(`${ptPct(hist[i].lo, 0)} – ${ptPct(hist[i].hi, 0)}`);
      wrongSum += Number(cells[1].replace(/\./g, ""));
    });
    const expected = hist.reduce((acc, b) => acc + b.n_wrong, 0);
    expect(expected).toBeGreaterThan(0);
    expect(wrongSum).toBe(expected);
  });

  it("40c: cada coluna tem a propria escala (o maximo de cada coluna ocupa a largura toda)", () => {
    const t = clone(tech);
    t.confidence_hist = [
      { lo: 0, hi: 0.5, n_right: 1000, n_wrong: 10 },
      { lo: 0.5, hi: 1, n_right: 500, n_wrong: 5 },
    ];
    const html = renderToStaticMarkup(<OverviewTechnical figures={FIGS} technical={t} />);
    const widths = body(html.match(/<table id="ov-confidence"[\s\S]*?<\/table>/)![0]).map((r) =>
      [...r.matchAll(/<rect[^>]*width="([\d.]+)"/g)].map((m) => Number(m[1])),
    );
    const full = widths[0][0];
    expect(full).toBeGreaterThan(0);
    // com escala unica, 10 erros contra 1000 acertos dariam 1% da largura
    expect(widths[0][1]).toBeCloseTo(full);
    expect(widths[1][0]).toBeCloseTo(full / 2);
    expect(widths[1][1]).toBeCloseTo(full / 2);
  });

  it("validacao: com o objeto montado no teste o prior P2 aparece; com null o bloco some", () => {
    expect(tech.validation).toBeNull();
    const t = clone(tech);
    t.validation = {
      purity: { text: "71,3%", value: 0.713 },
      completeness: { text: "80,0%", value: 0.8 },
      prior: "P2",
    } as unknown as NonNullable<Overview["technical"]["validation"]>;
    const text = textOf(renderToStaticMarkup(<OverviewTechnical figures={FIGS} technical={t} />));
    expect(text).toContain(L_ov.validationTitle);
    expect(text).toContain(`${L_ov.validationPrior}: P2`);
    expect(text).toContain("71,3%");
    expect(text).toContain("80,0%");
    const none = textOf(renderToStaticMarkup(<OverviewTechnical figures={FIGS} technical={tech} />));
    expect(none).not.toContain(L_ov.validationTitle);
    expect(none).not.toContain(`${L_ov.validationPrior}: P2`);
  });
});
