import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { HBarCI, type HBarCIRow } from "@/components/charts/HBarCI";
import { DataTable } from "@/components/kit/DataTable";
import { NA } from "@/components/kit/NA";
import { OverviewTechnical } from "@/components/pages/OverviewTechnical";
import { OverviewView } from "@/components/pages/OverviewView";
import { SkyView } from "@/components/pages/SkyView";
import { PageHeader } from "@/components/shell/PageHeader";
import { MapLegend } from "@/components/sky/MapLegend";
import { MollweideMap } from "@/components/sky/MollweideMap";
import { RegionTooltip, regionDescription } from "@/components/sky/RegionTooltip";
import { chartLabels, overviewLabels, skyLabels, skyMapLabels } from "@/content/labels";
import { ptInt, ptPct } from "@/lib/fmt";
import type { Est, Meta, Overview, Sky } from "@/lib/schema";
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
const sky = load<Sky>("sky");

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const count = (html: string, re: RegExp) => (html.match(re) ?? []).length;
const countText = (html: string, needle: string) => html.split(needle).length - 1;
const naEst: Est = { point: null, lo: null, hi: null, text: "n/a", ci_text: "n/a" };

function overviewRows(pick: "purity" | "completeness", selectedIdx: number): HBarCIRow[] {
  return overview.rows.map((r, i) => ({
    key: `${r.rule}-${r.prior}`,
    label: meta.states[i].label,
    est: r[pick],
    selected: i === selectedIdx,
    partial: r.partial,
  }));
}

describe("A1 HBarCI: estimativa sem ponto", () => {
  const html = renderToStaticMarkup(
    <HBarCI
      id="t"
      title="Titulo"
      partialTag={chartLabels.partialTag}
      rows={[{ key: "a", label: "Linha A", est: naEst, selected: false, partial: false }]}
    />,
  );

  it("mostra n/a de est.text, sem barra nem IC", () => {
    expect(html).not.toContain("data-bar");
    expect(html).not.toContain("data-ci");
    expect(html).toContain(">n/a<");
  });

  it("nunca mostra 0% para a linha (so ha 0% como tick do eixo)", () => {
    const fo = html.match(/<foreignObject[\s\S]*?<\/foreignObject>/);
    expect(fo).not.toBeNull();
    expect(fo![0]).not.toContain("%");
    // o resumo acessivel tambem usa est.text, nao zero
    expect(html).toMatch(/aria-label="Titulo\. Linha A: n\/a"/);
  });

  it("linha mista: so a linha com ponto tem barra e IC", () => {
    const real = overview.rows[0].purity;
    const mixed = renderToStaticMarkup(
      <HBarCI
        id="t"
        title="T"
        partialTag="p"
        rows={[
          { key: "a", label: "A", est: naEst, selected: false, partial: false },
          { key: "b", label: "B", est: real, selected: false, partial: false },
        ]}
      />,
    );
    expect(count(mixed, /data-bar=/g)).toBe(1);
    expect(count(mixed, /data-ci=/g)).toBe(1);
    expect(count(mixed, /<foreignObject/g)).toBe(1);
  });
});

describe("A2 HBarCI com as linhas da fixture overview", () => {
  const render = (pick: "purity" | "completeness") =>
    renderToStaticMarkup(
      <HBarCI id={`ov-${pick}`} title={pick} partialTag={chartLabels.partialTag} rows={overviewRows(pick, 3)} />,
    );
  const purity = render("purity");
  const completeness = render("completeness");

  it("a fixture tem 4 linhas e P3 e a unica parcial", () => {
    expect(overview.rows).toHaveLength(4);
    expect(overview.rows.filter((r) => r.partial).map((r) => r.prior)).toEqual(["P3"]);
  });

  it("P3 tem data-partial=true e a tag parcial; so ela", () => {
    expect(count(purity, /data-partial="true"/g)).toBe(1);
    const g = purity.match(/<g[^>]*data-row="prior_matched-P3"[^>]*>/);
    expect(g).not.toBeNull();
    expect(g![0]).toContain('data-partial="true"');
    expect(purity).toContain(`${meta.states[3].label} (${chartLabels.partialTag})`);
    expect(count(purity, new RegExp(`\\(${chartLabels.partialTag}\\)`, "g"))).toBeGreaterThanOrEqual(1);
  });

  it("a linha selecionada tem aria-current=true; so uma", () => {
    expect(count(purity, /aria-current="true"/g)).toBe(1);
    const g = purity.match(/<g[^>]*aria-current="true"[^>]*>/);
    expect(g![0]).toContain('data-row="prior_matched-P3"');
  });

  it("pureza e completude tem o mesmo viewBox", () => {
    const vb = (h: string) => h.match(/viewBox="([^"]+)"/)![1];
    expect(vb(purity)).toBe(vb(completeness));
  });
});

describe("A3 Visao geral", () => {
  const props = {
    meta,
    rows: overview.rows,
    accuracyNote: overview.accuracy_note,
    technical: overview.technical,
  };
  const render = (search: string) => {
    nav.search = search;
    return renderToStaticMarkup(<OverviewView figures={FIGS} {...props} />);
  };
  const kpi = (html: string, label: string) =>
    html.match(new RegExp(`${label}</p><p[^>]*>([^<]*)</p>`))?.[1];

  it("estado default: 3 KPIs com o .text da linha default", () => {
    const html = render("");
    const row = overview.rows[0];
    expect(kpi(html, overviewLabels.purity)).toBe(row.purity.text);
    expect(kpi(html, overviewLabels.completeness)).toBe(row.completeness.text);
    expect(kpi(html, overviewLabels.accuracy)).toBe(row.accuracy.text);
    expect(row.accuracy.text).not.toBe("n/a");
  });

  it("prior_matched: KPI de acuracia mostra o motivo (accuracyNA) e pureza muda", () => {
    const html = render("regra=prior_matched&prior=P2");
    expect(overview.rows[2].accuracy.value).toBeNull();
    expect(overviewLabels.accuracyNA).toMatch(/mistura do teste/);
    expect(kpi(html, overviewLabels.accuracy)).toBe(overviewLabels.accuracyNA);
    expect(kpi(html, overviewLabels.purity)).toBe(overview.rows[2].purity.text);
  });

  it("REV-03: as_shipped com accuracy null e sem motivo mostra o texto exportado, nao accuracyNA", () => {
    const base = overview.rows[0];
    expect(base.rule).toBe("as_shipped");
    const row = {
      ...base,
      accuracy: { value: null, text: "n/a" },
      accuracy_na_reason: null,
    };
    const rows = [row, ...overview.rows.slice(1)];
    nav.search = "";
    const html = renderToStaticMarkup(<OverviewView figures={FIGS} {...props} rows={rows} />);
    expect(kpi(html, overviewLabels.accuracy)).toBe(row.accuracy.text);
    expect(kpi(html, overviewLabels.accuracy)).toBe("n/a");
    expect(kpi(html, overviewLabels.accuracy)).not.toBe(overviewLabels.accuracyNA);
    // contraprova: prior_matched com "not_test_mix" continua mostrando accuracyNA
    const pm = overview.rows[2];
    expect(pm.rule).toBe("prior_matched");
    expect(pm.accuracy_na_reason).toBe("not_test_mix");
    expect(kpi(render("regra=prior_matched&prior=P2"), overviewLabels.accuracy)).toBe(
      overviewLabels.accuracyNA,
    );
  });

  it("prior_matched P1 (Tarefa 29): KPI de acuracia mostra accuracyNA, nunca 'n/a' cru", () => {
    const row = overview.rows.find((r) => r.rule === "prior_matched" && r.prior === "P1")!;
    expect(row.accuracy.value).toBeNull();
    const html = render("regra=prior_matched&prior=P1");
    expect(kpi(html, overviewLabels.accuracy)).toBe(overviewLabels.accuracyNA);
    // e na regra como entregue o valor real continua
    expect(kpi(render(""), overviewLabels.accuracy)).not.toBe(overviewLabels.accuracyNA);
  });

  it("partial_notice: aparece como aviso so quando o estado e P3", () => {
    const notice = escapeHtml(overview.rows[3].partial_notice!);
    // As figuras matplotlib repetem a nota dentro do SVG; conta-se so o HTML da pagina.
    const noFig = (h: string) => h.replace(/<figure[^>]*data-figure="[^"]*"[\s\S]*?<\/figure>/g, "");
    const p3 = noFig(render("regra=prior_matched&prior=P3"));
    const def = noFig(render(""));
    // no default aparece so a nota fixa do grafico; em P3 ha tambem o callout
    expect(countText(def, notice)).toBe(1);
    expect(countText(p3, notice)).toBe(2);
    expect(p3).toMatch(/callout[^"]*">/);
    expect(p3.includes(`>${notice}</p>`)).toBe(true);
    expect(def).not.toMatch(/callout[^"]*">/);
  });

  it("links para as quatro paginas", () => {
    const html = render("");
    for (const href of ["/ceu/", "/galaxias/", "/comparacao-dl/", "/metodo/"]) {
      expect(html, href).toMatch(new RegExp(`href="${href.replace(/\/$/, "")}/?(\\?[^"]*)?"`));
    }
  });

  it("mapa proprio do ceu: so regioes reportaveis, sem <img, com legenda", () => {
    nav.search = "";
    const html = renderToStaticMarkup(<OverviewView figures={FIGS} {...props} sky={sky} />);
    const fig = html.match(/<figure class="[^"]*thumb[^"]*">[\s\S]*?<\/figure>/)![0];
    const reportable = sky.regions.filter((r) => !r.below_floor && r.rate !== null).length;
    expect(reportable).toBeGreaterThan(0);
    expect(count(fig, /<circle/g)).toBe(reportable);
    expect(fig).not.toContain("<img");
    expect(fig).toContain(escapeHtml(overviewLabels.skyAria));
    expect(fig).toContain(escapeHtml(overviewLabels.skyCaption));
    expect(render("")).not.toContain('id="ov-sky"');
    expect(html).toContain('id="ov-sky"');
  });

  it("sem tecnico nao ha conteudo tecnico; com tecnico=1 ele aparece", () => {
    const off = render("");
    const on = render("tecnico=1");
    for (const marker of ['data-figure="dsc-diferencas"', 'data-figure="ganho-lgbm-full"', 'id="ov-confidence"']) {
      expect(off, marker).not.toContain(marker);
      expect(on, marker).toContain(marker);
    }
    // Tarefa 40a: a calibracao saiu da visao geral; no lugar, um link para /graficos/
    for (const html of [off, on]) {
      expect(html).not.toContain("data-cal-point");
      expect(html).not.toContain('id="ov-calibration"');
    }
    expect(on).toMatch(/href="\/graficos\/?(\?[^"]*)?"[^>]*>[^<]*Calibra/);
    expect(on).toContain(overviewLabels.calibrationLink);
    // Modo simples: so o resumo do estado selecionado; as duas figuras por regra e prior sao tecnicas.
    const summary = `data-figure="quasar-resumo-${meta.default_state.rule}-${meta.default_state.prior}"`
      .replace(/_/g, "-")
      .toLowerCase();
    for (const marker of ['data-figure="quasar-pureza"', 'data-figure="quasar-completude"']) {
      expect(off, marker).not.toContain(marker);
      expect(on, marker).toContain(marker);
    }
    expect(off).toContain(summary);
    expect(on).not.toContain(summary);
    expect(off).not.toContain(overviewLabels.techTitle);
    expect(on).toContain(overviewLabels.techTitle);
    // tecnico=0 explicito tambem e modo simples
    expect(render("tecnico=0")).not.toContain('id="ov-confidence"');
  });

  it("OverviewTechnical direto: tabela de erros tem uma linha por direcao", () => {
    const html = renderToStaticMarkup(<OverviewTechnical figures={FIGS} technical={overview.technical} />);
    const tbody = html.match(/<tbody>([\s\S]*?)<\/tbody>/)![1];
    expect(count(tbody, /<tr>/g)).toBe(overview.technical.error_directions.length);
  });
});

describe("A4 selo provisorio", () => {
  const prevRel = process.env.ASTRO_RELEASE;
  const prevRoot = process.env.ASTRO_WEB_DATA;
  beforeAll(() => {
    process.env.ASTRO_RELEASE = "fixture_v0";
    process.env.ASTRO_WEB_DATA = FIX;
  });
  afterAll(() => {
    if (prevRel === undefined) delete process.env.ASTRO_RELEASE;
    else process.env.ASTRO_RELEASE = prevRel;
    if (prevRoot === undefined) delete process.env.ASTRO_WEB_DATA;
    else process.env.ASTRO_WEB_DATA = prevRoot;
  });

  it("PageHeader com e sem state chip: um role=status", async () => {
    for (const chip of [false, true]) {
      nav.search = "";
      const html = renderToStaticMarkup(
        <PageHeader eyebrow="e" title="t" showStateChip={chip} meta={meta} />,
      );
      expect(count(html, /role="status"/g), `chip=${chip}`).toBe(1);
    }
  });

  it("paginas renderizadas (visao geral e ceu), tecnico off e on: um role=status", async () => {
    const { default: OverviewPage } = await import("@/app/page");
    const { default: SkyPage } = await import("@/app/ceu/page");
    for (const search of ["", "tecnico=1"]) {
      nav.search = search;
      for (const Page of [OverviewPage, SkyPage]) {
        const html = renderToStaticMarkup(<Page />);
        expect(count(html, /role="status"/g), `${Page.name} ?${search}`).toBe(1);
        expect(html).toContain(escapeHtml(meta.seal_text));
      }
    }
  });
});

// Abaixo do piso a regiao nao tem hp (SEC-02): todo circle do mapa e uma regiao, com ou sem data-hp.
const tags = (html: string) => html.match(/<circle[^>]*>/g) ?? [];
const mapHtml = (
  regions: Sky["regions"],
  mode: "rate" | "count",
  hideBelow: boolean,
  colorCap: Sky["color_cap"] = sky.color_cap,
) =>
  renderToStaticMarkup(
    <MollweideMap
      regions={regions}
      graticule={sky.graticule}
      colorCap={colorCap}
      bands={sky.bands}
      mode={mode}
      hideBelow={hideBelow}
      labels={skyMapLabels}
      summary="resumo"
    />,
  );

describe("A5 MollweideMap", () => {
  const below = sky.regions.filter((r) => r.below_floor);

  it("a fixture tem regioes abaixo do piso", () => {
    expect(sky.regions.length).toBe(4);
    expect(below.length).toBeGreaterThan(0);
  });

  it.each(["rate", "count"] as const)("%s: um circle por regiao", (mode) => {
    expect(tags(mapHtml(sky.regions, mode, false))).toHaveLength(sky.regions.length);
  });

  it.each(["rate", "count"] as const)("%s: data-hp so nas regioes acima do piso", (mode) => {
    const t = tags(mapHtml(sky.regions, mode, false));
    expect(t.filter((x) => x.includes("data-hp="))).toHaveLength(sky.regions.length - below.length);
    for (const x of t.filter((y) => y.includes('data-below="true"'))) {
      expect(x).not.toContain("data-hp");
    }
    expect(mapHtml(sky.regions, mode, false)).not.toContain("data-hp=\"null\"");
  });

  it("rate: anel vazio abaixo do piso sai sem n nem hp; tooltip e descricao dizem nao disponivel", () => {
    const ring = tags(mapHtml(sky.regions, "rate", false)).filter((x) => x.includes('fill="none"'));
    expect(ring).toHaveLength(below.length);
    for (const x of ring) expect(x).not.toMatch(/data-(hp|n)=/);
    const html = renderToStaticMarkup(
      <RegionTooltip region={below[0]} bandLabel="B" labels={skyMapLabels} leftPct={10} topPct={10} />,
    );
    const objects = html.match(new RegExp(`${skyMapLabels.objects}</span><span>([^<]*)</span>`))!;
    expect(objects[1]).toBe(skyMapLabels.notAvailable);
    expect(regionDescription(below[0], "B", skyMapLabels)).toContain(
      `${skyMapLabels.objects}: ${skyMapLabels.notAvailable}.`,
    );
  });

  it.each(["rate", "count"] as const)("%s com hideBelow: len - abaixo do piso", (mode) => {
    const t = tags(mapHtml(sky.regions, mode, true));
    expect(t).toHaveLength(sky.regions.length - below.length);
    expect(t.some((x) => x.includes('data-below="true"'))).toBe(false);
  });

  it("rate: todo data-below=true tem fill=none; os demais nao", () => {
    const t = tags(mapHtml(sky.regions, "rate", false));
    const marked = t.filter((x) => x.includes('data-below="true"'));
    expect(marked).toHaveLength(below.length);
    for (const x of marked) expect(x).toContain('fill="none"');
    for (const x of t.filter((y) => !y.includes('data-below="true"'))) {
      expect(x).not.toContain('fill="none"');
      expect(x).toContain("fill-opacity");
    }
  });

  it("rate: regiao abaixo do piso nao tem texto de taxa; tooltip diz nao disponivel", () => {
    const html = renderToStaticMarkup(
      <RegionTooltip region={below[0]} bandLabel="B" labels={skyMapLabels} leftPct={10} topPct={10} />,
    );
    expect(html).toContain(`${skyMapLabels.notAvailable} (${skyMapLabels.belowFloor})`);
    expect(html.replace(/style="[^"]*"/, "")).not.toMatch(/\d%/);
    // regiao normal mostra a taxa formatada
    const ok = sky.regions.find((r) => !r.below_floor)!;
    const okHtml = renderToStaticMarkup(
      <RegionTooltip region={ok} bandLabel="B" labels={skyMapLabels} leftPct={1} topPct={1} />,
    );
    expect(okHtml).toContain(ptPct(ok.rate!, 1));
  });

  it("count: nenhum fill=none, mesmo para regiao abaixo do piso", () => {
    const html = mapHtml(sky.regions, "count", false);
    expect(html).not.toContain('fill="none"');
    expect(tags(html)).toHaveLength(sky.regions.length);
  });

  describe("variante: 2 abaixo do piso e color_cap.rate = null", () => {
    const regions: Sky["regions"] = sky.regions.map((r, i) =>
      i === 0 || i === 1
        ? { ...r, below_floor: false, hp: r.hp ?? 900 + i, n: r.n ?? 9, rate: r.rate ?? 0.5, errors: r.errors ?? 1 }
        : { ...r, below_floor: true, hp: null, n: null, rate: null, errors: null },
    );
    const cap: Sky["color_cap"] = { rate: null, count: sky.color_cap.count };

    it("rate: 2 aneis vazios, os outros preenchidos", () => {
      const t = tags(mapHtml(regions, "rate", false, cap));
      expect(t).toHaveLength(4);
      expect(t.filter((x) => x.includes('fill="none"'))).toHaveLength(2);
      expect(t.filter((x) => x.includes('data-below="true"'))).toHaveLength(2);
    });

    it("legenda em rate: maximo nao disponivel, nunca zero", () => {
      const html = renderToStaticMarkup(
        <MapLegend mode="rate" colorCap={cap} howText="como" labels={skyMapLabels} saturated={false} />,
      );
      const m = html.match(/legendBar[^>]*><span>([^<]*)<\/span>[\s\S]*?<\/svg><span>([^<]*)<\/span>/)!;
      expect(m[2]).toBe(skyMapLabels.notAvailable);
      expect(m[2]).not.toMatch(/^0/);
      expect(html).toContain(skyMapLabels.legendRing);
    });

    it("legenda em count nao e afetada por cap.rate nulo", () => {
      const html = renderToStaticMarkup(
        <MapLegend mode="count" colorCap={cap} howText="como" labels={skyMapLabels} saturated={false} />,
      );
      expect(html).toContain(`<span>${ptInt(cap.count!)}</span>`);
      expect(html).not.toContain(skyMapLabels.legendRing);
    });
  });
});

describe("A6 MapLegend: minimo e maximo explicitos", () => {
  const bar = (mode: "rate" | "count") => {
    const html = renderToStaticMarkup(
      <MapLegend mode={mode} colorCap={sky.color_cap} howText="como" labels={skyMapLabels} saturated={false} />,
    );
    const m = html.match(/legendBar[^>]*><span>([^<]*)<\/span>[\s\S]*?<\/svg><span>([^<]*)<\/span>/)!;
    return { min: m[1], max: m[2] };
  };

  it("rate: 0% a ptPct(color_cap.rate)", () => {
    expect(bar("rate")).toEqual({ min: ptPct(0, 0), max: ptPct(sky.color_cap.rate!, 1) });
  });

  it("count: 0 a ptInt(color_cap.count)", () => {
    expect(bar("count")).toEqual({ min: ptInt(0), max: ptInt(sky.color_cap.count!) });
  });

  it("rate e count independentes: valores diferentes", () => {
    expect(bar("rate").max).not.toBe(bar("count").max);
    expect(bar("rate").min).not.toBe(bar("count").min);
  });

  it("saturado: maximo ganha o prefixo 'pelo menos'", () => {
    const html = renderToStaticMarkup(
      <MapLegend mode="count" colorCap={sky.color_cap} howText="c" labels={skyMapLabels} saturated />,
    );
    expect(html).toContain(`${skyMapLabels.atLeast} ${ptInt(sky.color_cap.count!)}`);
  });
});

describe("A7 SkyView", () => {
  const render = (search: string) => {
    nav.search = search;
    return renderToStaticMarkup(<SkyView meta={meta} sky={sky} />);
  };

  it("tecnico=1: 2 tabelas com len(top_regions) e len(bands) linhas e a caixa de ocultar", () => {
    const html = render("tecnico=1");
    expect(count(html, /<table/g)).toBe(2);
    const bodies = [...html.matchAll(/<tbody>([\s\S]*?)<\/tbody>/g)].map((m) => count(m[1], /<tr>/g));
    expect(bodies).toEqual([sky.top_regions.length, sky.bands.length]);
    expect(html).toContain('type="checkbox"');
    expect(html).toContain(skyLabels.hideBelow);
  });

  it("sem tecnico: nenhuma tabela e nenhuma caixa de ocultar", () => {
    for (const s of ["", "tecnico=0"]) {
      const html = render(s);
      expect(html).not.toContain("<table");
      expect(html).not.toContain('type="checkbox"');
      expect(html).not.toContain(skyLabels.hideBelow);
    }
  });

  it("textos de faixa vem de bands[].label", () => {
    const html = render("tecnico=1");
    for (const b of sky.bands) expect(html).toContain(escapeHtml(b.label));
  });
});

describe("kit e graficos: fronteira", () => {
  it("NA: texto de entrada, rotulo acessivel nao disponivel", () => {
    const html = renderToStaticMarkup(<NA text="n/a" />);
    expect(html).toContain(">n/a<");
    expect(html).toContain('aria-label="não disponível"');
  });

  it("DataTable vazia: cabecalho sem linhas", () => {
    const html = renderToStaticMarkup(
      <DataTable caption="c" columns={[{ key: "a", label: "A" }]} rows={[]} />,
    );
    expect(html).toContain("<th");
    expect(count(html, /<tr>/g)).toBe(1);
  });
});
