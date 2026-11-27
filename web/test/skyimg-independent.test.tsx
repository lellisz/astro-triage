// Testes independentes (escritos a partir dos criterios, nao da implementacao) da
// remocao de imagem de terceiros e do mapa proprio da Visao geral.
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { OverviewView } from "@/components/pages/OverviewView";
import { SkyView } from "@/components/pages/SkyView";
import { MollweideMap } from "@/components/sky/MollweideMap";
import { SkyThumb } from "@/components/sky/SkyThumb";
import * as scale from "@/components/sky/scale";
import { overviewLabels, skyLabels, skyMapLabels } from "@/content/labels";
import type { Meta, Overview, Sky } from "@/lib/schema";
import { MollweideMapBase } from "./fixtures/legacy/MollweideMap.base";
import { FIGS } from "./fixture-figures";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(""),
  usePathname: () => "/",
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

const WEB = fileURLToPath(new URL("..", import.meta.url));
const FIX = path.join(WEB, "test", "fixtures", "fixture_v0");
const load = <T,>(n: string): T => JSON.parse(readFileSync(path.join(FIX, `${n}.json`), "utf-8")) as T;
const meta = load<Meta>("meta");
const overview = load<Overview>("overview");
const sky = load<Sky>("sky");

const RASTER = /\.(png|jpe?g|webp|gif|avif)$/i;
function walk(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)],
  );
}
// Guarda reutilizavel: devolve os rasters encontrados sob `dir`.
const rastersIn = (dir: string) => walk(dir).filter((f) => RASTER.test(f));

const props = {
  meta,
  rows: overview.rows,
  accuracyNote: overview.accuracy_note,
  technical: overview.technical,
};
const circles = (html: string) =>
  [...html.matchAll(/<circle\b[^>]*>/g)].map((m) => ({
    cx: Number(m[0].match(/\bcx="([^"]+)"/)![1]),
    cy: Number(m[0].match(/\bcy="([^"]+)"/)![1]),
    op: Number(m[0].match(/fill-opacity="([^"]+)"/)?.[1] ?? NaN),
  }));

// Ceu sintetico com regioes abaixo do piso, taxa nula, taxa acima do teto e zero.
const synth: Sky = {
  ...sky,
  color_cap: { count: 20, rate: 0.5 },
  regions: [
    { band: "B", below_floor: false, errors: 0, hp: 1, n: 30, rate: 0, x: -1, y: 0.1 },
    { band: "B", below_floor: false, errors: 5, hp: 2, n: 30, rate: 0.1, x: -0.5, y: 0.2 },
    { band: "B", below_floor: false, errors: 9, hp: 3, n: 30, rate: 0.3, x: 0, y: 0.3 },
    { band: "B", below_floor: false, errors: 15, hp: 4, n: 30, rate: 0.5, x: 0.5, y: 0.4 },
    { band: "B", below_floor: false, errors: 25, hp: 5, n: 30, rate: 0.99, x: 1, y: 0.5 },
    { band: "B", below_floor: true, errors: null, hp: null, n: null, rate: null, x: 1.5, y: 0.6 },
    // incoerente de proposito: below_floor=true com taxa preenchida nunca deve ser desenhado
    { band: "B", below_floor: true, errors: 1, hp: 7, n: 3, rate: 0.9, x: 2, y: 0.7 },
    // incoerente de proposito: acima do piso mas sem taxa
    { band: "B", below_floor: false, errors: null, hp: 8, n: 30, rate: null, x: 2.5, y: 0.8 },
  ] as Sky["regions"],
};
const thumbProps = (s: Sky) => ({
  graticule: s.graticule,
  regions: s.regions,
  colorCap: s.color_cap,
  aria: "ARIA-X",
  caption: "CAP-X",
});

describe("(1) sem raster de terceiros", () => {
  it("web/public/gaia nao existe", () => {
    expect(existsSync(path.join(WEB, "public", "gaia"))).toBe(false);
  });
  it("nenhum raster em web/public nem em web/out", () => {
    expect(rastersIn(path.join(WEB, "public"))).toEqual([]);
    expect(rastersIn(path.join(WEB, "out"))).toEqual([]);
  });
  it("falsificador: o guarda detecta um raster plantado", () => {
    const tmp = mkdtempSync(path.join(os.tmpdir(), "rastercheck-"));
    try {
      expect(rastersIn(tmp)).toEqual([]);
      for (const ext of ["png", "JPG", "jpeg", "webp", "gif", "avif"]) {
        writeFileSync(path.join(tmp, `x.${ext}`), "x");
      }
      expect(rastersIn(tmp)).toHaveLength(6);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
});

describe("(2) HTML do export sem imagem de terceiros", () => {
  const pages = walk(path.join(WEB, "out")).filter((f) => /\.(html|txt)$/.test(f));
  it("o export da fixture existe", () => {
    expect(pages.some((f) => f.endsWith("index.html"))).toBe(true);
    expect(existsSync(path.join(WEB, "out", "ceu", "index.html"))).toBe(true);
  });
  it("sem <img, /gaia/, gaia-skymap ou dominio externo de imagem", () => {
    for (const f of pages.filter((p) => p.endsWith(".html"))) {
      const html = readFileSync(f, "utf-8");
      const rel = path.relative(WEB, f);
      expect(html, rel).not.toMatch(/<img\b/i);
      // caminho local do ativo removido (link de texto para a pagina da ESA nao conta)
      expect(html, rel).not.toMatch(/(?:src|href|srcset|url\()=?["'(]?\/gaia\//i);
      expect(html, rel).not.toMatch(/gaia-skymap/i);
      expect(html, rel).not.toMatch(/(src|srcset)=["']https?:\/\//i);
    }
  });
  it("a Visao geral exportada tem o mapa proprio (svg) e nao o painel antigo", () => {
    const html = readFileSync(path.join(WEB, "out", "index.html"), "utf-8");
    expect(html).toContain('id="ov-sky"');
    expect(html).not.toContain("ov-gaia");
  });
});

describe("(3) mapa da Visao geral (SkyThumb)", () => {
  const html = renderToStaticMarkup(<SkyThumb {...thumbProps(synth)} />);
  const c = circles(html);
  it("desenha so regioes reportaveis, nas posicoes de sky.json", () => {
    expect(c.map((p) => p.cx)).toEqual([-1, -0.5, 0, 0.5, 1]);
    expect(c.map((p) => p.cy)).toEqual([-0.1, -0.2, -0.3, -0.4, -0.5]);
  });
  it("nunca emite n, hp ou marcacao de regiao abaixo do piso", () => {
    expect(html).not.toMatch(/data-hp|data-below|<text|<img/);
    for (const r of synth.regions.filter((x) => x.below_floor || x.rate === null)) {
      expect(html).not.toContain(`cx="${r.x}"`);
    }
    // nenhum dos n (30, 3) nem das taxas aparece como texto
    expect(html).not.toMatch(/>\s*30\s*</);
  });
  it("opacidade monotona na taxa e saturada no teto", () => {
    const ops = c.map((p) => p.op);
    for (let i = 1; i < ops.length; i++) expect(ops[i]).toBeGreaterThanOrEqual(ops[i - 1]);
    expect(ops[1]).toBeGreaterThan(ops[0]);
    expect(ops[2]).toBeGreaterThan(ops[1]);
    expect(ops[3]).toBeGreaterThan(ops[2]);
    // taxa == teto e taxa > teto: mesma opacidade, e nunca acima de 1
    expect(ops[4]).toBe(ops[3]);
    expect(Math.max(...ops)).toBeLessThanOrEqual(1);
    expect(Math.min(...ops)).toBeGreaterThan(0);
  });
  it("o teto vem de color_cap.rate: mudar o teto muda a opacidade", () => {
    const hi = circles(
      renderToStaticMarkup(<SkyThumb {...thumbProps({ ...synth, color_cap: { count: 20, rate: 1 } })} />),
    );
    expect(hi[3].op).toBeLessThan(c[3].op); // 0.5/1 < 0.5/0.5
    expect(hi[4].op).toBeLessThan(c[4].op); // 0.99/1 < saturado
  });
  it("fixture real: tantos pontos quantas regioes reportaveis, com aria e legenda", () => {
    const h = renderToStaticMarkup(<SkyThumb {...thumbProps(sky)} />);
    const n = sky.regions.filter((r) => !r.below_floor && r.rate !== null).length;
    expect(circles(h)).toHaveLength(n);
    expect(h).toContain('role="img"');
    expect(h).toContain("ARIA-X");
    expect(h).toContain("CAP-X");
  });
  it("todas abaixo do piso: nenhum ponto", () => {
    const none = { ...synth, regions: synth.regions.filter((r) => r.below_floor) } as Sky;
    expect(circles(renderToStaticMarkup(<SkyThumb {...thumbProps(none)} />))).toHaveLength(0);
  });
  it("Visao geral: sem a prop sky nao ha painel e o resto e identico ao com sky", () => {
    const without = renderToStaticMarkup(<OverviewView figures={FIGS} {...props} />);
    const withSky = renderToStaticMarkup(<OverviewView figures={FIGS} {...props} sky={sky} />);
    expect(without).not.toContain("ov-sky");
    expect(without).not.toContain(overviewLabels.skyTitle);
    expect(without).not.toContain("<svg class=\"" + "sky");
    expect(withSky).toContain("ov-sky");
    // remover o painel do HTML com sky devolve exatamente o HTML sem sky
    const panel = withSky.match(/<section[^>]*aria-labelledby="ov-sky"[\s\S]*?<\/section>/);
    expect(panel, "painel ov-sky delimitado por <section>").not.toBeNull();
    expect(withSky.replace(panel![0], "")).toBe(without);
  });
});

describe("(4) /ceu sem imagem de contexto e com mapa principal", () => {
  const html = renderToStaticMarkup(<SkyView meta={meta} sky={sky} />);
  it("sem painel de contexto, <img ou /gaia/", () => {
    expect(html).not.toContain("sky-context");
    expect(html).not.toMatch(/<img\b|\/gaia\//i);
    expect(html).not.toContain("Imagem de contexto");
  });
  it("mantem o mapa principal interativo (circulos com data-hp)", () => {
    expect(html).toContain(skyMapLabels.mapAria);
    expect(html).toMatch(/<circle[^>]*data-hp=/);
  });
  it("labels de contexto removidos nao existem mais", () => {
    expect(skyLabels).not.toHaveProperty("contextTitle");
    expect(skyLabels).not.toHaveProperty("contextSub");
    expect(overviewLabels).not.toHaveProperty("gaiaTitle");
    expect(overviewLabels).not.toHaveProperty("gaiaSub");
  });
});

describe("(5) textos novos do painel", () => {
  const texts = [overviewLabels.skyTitle, overviewLabels.skySub, overviewLabels.skyAria, overviewLabels.skyCaption];
  it("existem e nao estao vazios", () => {
    for (const t of texts) expect(t.trim().length).toBeGreaterThan(5);
  });
  it("sem empate, streamlit, ADR", () => {
    for (const t of texts) {
      expect(t).not.toMatch(/empate/i);
      expect(t).not.toMatch(/streamlit/i);
      expect(t).not.toMatch(/\bADR\b/i);
    }
  });
  it("sem numero formatado (digito, porcentagem)", () => {
    for (const t of texts) expect(t).not.toMatch(/\d|%/);
  });
});

describe("(6) refatoracao: o mapa de /ceu e identico ao da base", () => {
  const regionsMixed = [...sky.regions, ...synth.regions] as Sky["regions"];
  for (const mode of ["rate", "count"] as const) {
    for (const hideBelow of [false, true]) {
      it(`markup identico: modo=${mode} hideBelow=${hideBelow}`, () => {
        const common = {
          regions: regionsMixed,
          graticule: sky.graticule,
          colorCap: sky.color_cap,
          bands: sky.bands,
          mode,
          hideBelow,
          labels: skyMapLabels,
          summary: "resumo",
        };
        const now = renderToStaticMarkup(<MollweideMap {...common} />);
        const base = renderToStaticMarkup(<MollweideMapBase {...common} />);
        expect(circles(now).length).toBeGreaterThan(0);
        expect(now).toBe(base);
      });
    }
  }
  it("scale.ts: moldura e opacidade mantem os valores da base", () => {
    expect([scale.VIEW_X, scale.VIEW_Y, scale.VIEW_W, scale.VIEW_H]).toEqual([-3.35, -1.52, 6.7, 3.04]);
    expect(scale.MIN_OPACITY).toBe(0.15);
    expect(scale.opacityFor(null, 1)).toBe(0.15);
    expect(scale.opacityFor(1, null)).toBe(0.15);
    expect(scale.opacityFor(1, 0)).toBe(0.15);
    expect(scale.opacityFor(0, 1)).toBeCloseTo(0.15);
    expect(scale.opacityFor(2, 1)).toBeCloseTo(1);
    expect(scale.opacityFor(-5, 1)).toBeCloseTo(0.15);
    expect(scale.opacityFor(0.5, 1)).toBeCloseTo(0.575);
    expect(scale.points([[1, 2], [3, -4]])).toBe("1,-2 3,4");
  });
});
