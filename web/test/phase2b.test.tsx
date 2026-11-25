import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Tabs, type TabItem } from "@/components/kit/Tabs";
import { CompareView } from "@/components/pages/CompareView";
import { GalaxiesView } from "@/components/pages/GalaxiesView";
import MethodPage from "@/app/metodo/page";
import { StateControls } from "@/components/shell/StateControls";
import { chartLabels, compareHref, compareLabels, controls, methodLabels } from "@/content/labels";
import type { Compare, Est, Galaxies, Meta, Method } from "@/lib/schema";

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
const figure = (html: string, id: string): string => {
  const m = html.match(new RegExp(`<figure[^>]*id="${id}"[\\s\\S]*?</figure>`));
  if (!m) throw new Error(`figure ${id} ausente`);
  return m[0];
};

const renderGal = (g: Galaxies, search = "") => {
  nav.search = search;
  return renderToStaticMarkup(<GalaxiesView meta={meta} galaxies={g} />);
};

describe("A1 Galaxias: Est com point null vira n/a, sem zero, barra ou ponto", () => {
  const g = clone(galaxies);
  // prior_matched P2 sem estimativa nas duas series (pureza e completude)
  g.blocks.prior_matched[1].purity = NA_EST;
  g.blocks.prior_matched[1].completeness = NA_EST;
  const html = renderGal(g, "tecnico=1");

  it("series por prior: a linha nula mostra n/a e nenhum marcador, linha, intervalo ou valor", () => {
    const chunks = rowChunks(html, "prior_matched-P2");
    expect(chunks).toHaveLength(2); // pureza e completude
    for (const c of chunks) {
      expect(c).not.toMatch(/<(circle|rect|polygon|polyline|line)\b/);
      expect(c).not.toContain("data-ci");
      expect(c).toContain('aria-label="não disponível"');
      expect(textOf(c).trim()).toBe("n/a");
    }
  });

  it("series por prior: o ponto nulo nao entra no caminho e o vizinho nao-nulo continua desenhado", () => {
    const poly = [...html.matchAll(/<g data-series="prior_matched">([\s\S]*?)<\/g><\/g>/g)];
    expect(poly.length).toBeGreaterThan(0);
    expect(rowChunks(html, "prior_matched-P1")[0]).toMatch(/<circle|<rect/);
    expect(rowChunks(html, "prior_matched-P3")[0]).toMatch(/<polygon/);
  });

  it("barras de erro (geral, G e z): linhas nulas sem barra, ponto ou intervalo", () => {
    const groups: [string, number[], typeof galaxies.error_rows][] = [
      ["err", [], galaxies.error_rows],
      ["g", [], galaxies.technical.by_g],
      ["z", [], galaxies.technical.by_z],
    ];
    let nulls = 0;
    for (const [prefix, , rows] of groups) {
      rows.forEach((r, i) => {
        if (r.rate.point !== null) return;
        nulls += 1;
        const chunks = rowChunks(html, `${prefix}-${i}`);
        expect(chunks, `${prefix}-${i}`).toHaveLength(1);
        const c = chunks[0];
        expect(c).not.toContain("data-bar");
        expect(c).not.toContain("data-ci");
        expect(c).not.toMatch(/<circle/);
        expect(c).toContain("n/a");
        expect(textOf(c)).not.toMatch(/\d\s*%/);
      });
    }
    expect(nulls).toBeGreaterThan(0);
  });

  it("faixas abaixo do piso nao somem: contagem de linhas = contagem do JSON", () => {
    expect(count(html, /data-row="err-\d+"/g)).toBe(galaxies.error_rows.length);
    expect(count(html, /data-row="g-\d+"/g)).toBe(galaxies.technical.by_g.length);
    expect(count(html, /data-row="z-\d+"/g)).toBe(galaxies.technical.by_z.length);
    expect(galaxies.technical.by_g.some((r) => !r.reportable)).toBe(true);
    expect(galaxies.technical.by_z.some((r) => !r.reportable)).toBe(true);
    // e cada rotulo do JSON aparece
    for (const r of [...galaxies.technical.by_g, ...galaxies.technical.by_z]) {
      expect(textOf(html)).toContain(r.label);
    }
  });

  it("uma estimativa 0 real (point 0) NAO e confundida com n/a: continua desenhada", () => {
    const z = clone(galaxies);
    z.error_rows[0].rate = { point: 0, lo: 0, hi: 0.1, text: "0,0%", ci_text: "IC 95%: 0,0% a 10,0%" };
    const c = rowChunks(renderGal(z), "err-0")[0];
    expect(c).toContain("data-bar");
    expect(textOf(c)).toContain("0,0%");
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

  it("P3 tem rotulo de parcial nos dois graficos e marcador triangulo nas duas regras", () => {
    expect(count(textOf(html), `P3 (${chartLabels.partialTag})`)).toBeGreaterThanOrEqual(2);
    for (const rule of ["as_shipped", "prior_matched"]) {
      const chunks = rowChunks(html, `${rule}-P3`);
      expect(chunks, rule).toHaveLength(2);
      for (const c of chunks) {
        expect(c).toContain('data-partial="true"');
        expect(c).toMatch(/<polygon/);
        expect(c).not.toMatch(/<circle|<rect/);
      }
    }
    // nao-parcial nunca usa triangulo
    for (const rule of ["as_shipped", "prior_matched"]) {
      for (const c of rowChunks(html, `${rule}-P1`)) expect(c).not.toMatch(/<polygon/);
    }
  });

  it("as_shipped e prior_matched sao series separadas, com forma de marcador e traco diferentes", () => {
    expect(count(html, 'data-series="as_shipped"')).toBe(2);
    expect(count(html, 'data-series="prior_matched"')).toBe(2);
    expect(rowChunks(html, "as_shipped-P1")[0]).toMatch(/<circle/);
    expect(rowChunks(html, "prior_matched-P1")[0]).toMatch(/<rect/);
    const cls = (rule: string) =>
      [...html.matchAll(new RegExp(`data-series="${rule}"><polyline class="([^"]*)"`, "g"))].map((m) => m[1]);
    expect(cls("as_shipped")).toHaveLength(2);
    expect(cls("prior_matched")).toHaveLength(2);
    for (const c of cls("as_shipped")) expect(c).toMatch(/seriesSolid/);
    for (const c of cls("prior_matched")) expect(c).toMatch(/seriesDashed/);
    // legenda tambem distingue
    expect(html).toMatch(/data-legend="as_shipped"><line class="[^"]*seriesSolid/);
    expect(html).toMatch(/data-legend="prior_matched"><line class="[^"]*seriesDashed/);
  });

  it("stroke-dasharray existe so na regra tracejada (via classe do css module, nao atributo)", () => {
    const rule = (name: string) => css.match(new RegExp(`\\.${name}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
    expect(rule("seriesDashed")).toMatch(/stroke-dasharray\s*:\s*\S/);
    expect(rule("seriesSolid")).not.toMatch(/stroke-dasharray/);
    expect(rule("seriesSolid")).not.toBe("");
    // nenhum atributo inline de traco no markup
    expect(html).not.toMatch(/stroke-dasharray=/i);
  });

  it("nenhuma cor fora de tokens/currentColor: atributos, style inline e css do grafico", () => {
    const attrs = [...html.matchAll(/\s(fill|stroke)="([^"]*)"/g)].map((m) => m[2]);
    for (const v of attrs) expect(v).toMatch(/^(none|currentColor|var\(--[\w-]+\))$/);
    expect(html).not.toMatch(/style="[^"]*(#|rgb|hsl)/i);
    const decls = [...css.matchAll(/(?:^|[;{\s])(?:fill|stroke)\s*:\s*([^;}]+)/g)].map((m) => m[1].trim());
    expect(decls.length).toBeGreaterThan(5);
    for (const v of decls) expect(v, v).toMatch(/^(none|currentColor|inherit|var\(--[\w-]+\))$/);
    expect(css).not.toMatch(/(fill|stroke)\s*:\s*(#|rgb|hsl)/i);
  });
});

describe("A3 Galaxias: tabelas tecnicas so com tecnico=1; power_note", () => {
  it("sem tecnico nao ha painel tecnico, tabelas nem graficos G/z", () => {
    const html = renderGal(galaxies, "");
    expect(html).not.toContain('id="gal-tech"');
    expect(html).not.toContain("<table");
    expect(html).not.toContain('id="gal-g-chart"');
    expect(html).not.toContain('id="gal-z-chart"');
  });

  it("tecnico=0 explicito tambem esconde", () => {
    const html = renderGal(galaxies, "tecnico=0");
    expect(html).not.toContain('id="gal-tech"');
    expect(html).not.toContain("<table");
  });

  it("tecnico=1 mostra o painel e duas tabelas com uma linha por faixa do JSON", () => {
    const html = renderGal(galaxies, "tecnico=1");
    expect(html).toContain('id="gal-tech"');
    expect(count(html, "<table")).toBe(2);
    expect(count(html, /<tbody>[\s\S]*?<\/tbody>/g)).toBe(2);
    const bodyRows = [...html.matchAll(/<tbody>([\s\S]*?)<\/tbody>/g)].map((m) => count(m[1], "<tr>"));
    expect(bodyRows).toEqual([galaxies.technical.by_g.length, galaxies.technical.by_z.length]);
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
    const html = renderToStaticMarkup(<CompareView compare={down(status, err)} />);
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
    const html = renderToStaticMarkup(<CompareView compare={leaky} />);
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
  const html3 = renderToStaticMarkup(<CompareView compare={three} />);

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
    const t = textOf(renderToStaticMarkup(<CompareView compare={noPair} />));
    expect(t).toContain(compareLabels.pairMissing);
  });
});

describe("A5 Comparacao: faixa min-max rotulada 'nao e IC'", () => {
  const html = renderToStaticMarkup(<CompareView compare={ok} />);
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

  it("o grafico de faixa nao reaproveita a classe/data-ci de intervalo de confianca", () => {
    const fig = figure(html, "cmp-purity-chart");
    expect(fig).not.toContain("data-ci");
    expect(fig).toContain("data-range");
    expect(count(fig, "data-row=")).toBe(ok.purity.length);
  });
});

describe("A6 Comparacao: abas", () => {
  nav.search = "";
  const html = renderToStaticMarkup(<CompareView compare={ok} />);

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
  const html = renderToStaticMarkup(<CompareView compare={v} />);

  it("decisao: linha de zero, uma linha por celula do grid, celula sem d vira n/a sem ponto", () => {
    const fig = figure(html, "cmp-decision");
    expect(count(fig, "data-zero")).toBe(1);
    expect(count(fig, "data-row=")).toBe(ok.grid.length);
    const na = rowChunks(fig, `g-${ok.grid[2].net}-${ok.grid[2].cls}`)[0];
    expect(na).not.toContain("data-point");
    expect(na).not.toContain("data-ci");
    expect(na).toContain("n/a");
    expect(textOf(na)).not.toMatch(/[+-]?0,\d/);
    expect(count(fig, "data-point")).toBe(ok.grid.length - 1);
  });

  it("descritivo: linha de zero, uma linha por item, n/a sem ponto", () => {
    const fig = figure(html, "cmp-descriptive");
    expect(count(fig, "data-zero")).toBe(1);
    expect(count(fig, "data-row=")).toBe(v.descriptive.length);
    const na = rowChunks(fig, "d-1")[0];
    expect(na).not.toContain("data-point");
    expect(na).toContain("n/a");
    expect(count(fig, "data-point")).toBe(2);
    // a tabela descritiva traz uma linha por item
    const rows = [...html.matchAll(/<table[^>]*>[\s\S]*?<\/table>/g)].map((m) => m[0]);
    expect(rows.some((t) => count(t, "<tr>") === v.descriptive.length + 1)).toBe(true);
  });

  it("sem itens descritivos: mensagem, sem segundo grafico", () => {
    const h2 = renderToStaticMarkup(<CompareView compare={ok} />);
    expect(h2).not.toContain('id="cmp-descriptive"');
    expect(textOf(h2)).toContain(compareLabels.noDescriptive);
    expect(count(figure(h2, "cmp-decision"), "data-zero")).toBe(1);
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

  it("details/summary para leitura, uso, regras e limitacoes; original aninhado so se difere", () => {
    for (const id of ["leitura", "uso", "regras", "limitacoes"]) {
      expect(html, id).toMatch(new RegExp(`<details[^>]*id="${id}"`));
    }
    const differing = method.limitations.filter((l) => l.original !== l.text).length;
    expect(differing).toBeGreaterThan(0);
    expect(count(html, "<details")).toBe(4 + differing);
    expect(count(html, "<summary")).toBe(4 + differing);
    expect(count(html, /<summary[^>]*>[^<]+<\/summary>/g)).toBe(4 + differing);
  });

  it("fechados por padrao e abertos com tecnico=1", () => {
    expect(html).not.toMatch(/<details[^>]*\sopen/);
    store.pages = { meta, method };
    nav.search = "tecnico=1";
    const open = renderToStaticMarkup(<MethodPage />);
    expect(count(open, /<details[^>]*\sopen=""/g)).toBe(4);
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
