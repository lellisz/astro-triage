// Regressoes da ADR 0008 Fase 2a (P2-T11): REV-P2-01, TST-03, REV-P2-21..24, TST-02.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { OverviewView } from "@/components/pages/OverviewView";
import { SkyView } from "@/components/pages/SkyView";
import { PageHeader } from "@/components/shell/PageHeader";
import { MapLegend } from "@/components/sky/MapLegend";
import { MollweideMap } from "@/components/sky/MollweideMap";
import { skyMapLabels } from "@/content/labels";
import type { Meta, Overview, Sky } from "@/lib/schema";

const nav = vi.hoisted(() => ({ search: "" }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => "/",
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

const WEB = fileURLToPath(new URL("../", import.meta.url));
const FIX = path.join(WEB, "test", "fixtures", "fixture_v0");
const load = <T,>(name: string): T =>
  JSON.parse(fs.readFileSync(path.join(FIX, `${name}.json`), "utf-8")) as T;
const meta = load<Meta>("meta");
const overview = load<Overview>("overview");
const sky = load<Sky>("sky");

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const stripTags = (s: string) => s.replace(/<[^>]*>/g, "").replace(/<!-- -->/g, "");

describe("REV-P2-01: rotulo do grafico segue o par regra/prior, nao o indice", () => {
  const props = {
    rows: overview.rows,
    accuracyNote: overview.accuracy_note,
    technical: overview.technical,
  };
  const labelOf = (html: string, key: string) =>
    stripTags(html.match(new RegExp(`data-row="${key}"[^>]*>\\s*<text[^>]*>([\\s\\S]*?)</text>`))![1]);

  it("states em ordem invertida: cada data-row mantem o rotulo do seu par", () => {
    nav.search = "";
    const reversed: Meta = { ...meta, states: [...meta.states].reverse() };
    expect(reversed.states[0].label).not.toBe(meta.states[0].label);
    for (const m of [meta, reversed]) {
      const html = renderToStaticMarkup(<OverviewView {...props} meta={m} />);
      for (const r of overview.rows) {
        const want = meta.states.find((s) => s.rule === r.rule && s.prior === r.prior)!.label;
        const got = labelOf(html, `${r.rule}-${r.prior}`);
        expect(got.includes(esc(want)) || got.includes(want), `${r.rule}-${r.prior}`).toBe(true);
      }
    }
  });
});

describe("TST-03: rotulos do graticule usam o kind do JSON", () => {
  const render = (labels: Sky["graticule"]["labels"]) =>
    renderToStaticMarkup(
      <MollweideMap
        regions={sky.regions}
        graticule={{ ...sky.graticule, labels }}
        colorCap={sky.color_cap}
        bands={sky.bands}
        mode="rate"
        hideBelow={false}
        labels={skyMapLabels}
        summary="r"
      />,
    );
  const parse = (html: string) =>
    [...html.matchAll(/<text[^>]*data-kind="([^"]+)"[^>]*>([^<]*)<\/text>/g)].map((m) => [m[1], m[2]]);

  it("cada label renderiza com data-kind igual ao kind do JSON", () => {
    expect(parse(render(sky.graticule.labels))).toEqual(sky.graticule.labels.map((l) => [l.kind, l.text]));
  });

  it("kind trocado no JSON troca o data-kind (nao deriva do texto)", () => {
    const flipped = sky.graticule.labels.map((l) => ({ ...l, kind: l.kind === "lat" ? "lon" : "lat" })) as Sky["graticule"]["labels"];
    expect(parse(render(flipped))).toEqual(flipped.map((l) => [l.kind, l.text]));
  });
});

describe("REV-P2-24: opacidade minima 0.15 so em scale.ts", () => {
  const walk = (dir: string): string[] =>
    fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) return walk(p);
      return /\.(ts|tsx|css)$/.test(e.name) ? [p] : [];
    });

  it("nenhum outro arquivo de app/components/lib/content contem 0.15", () => {
    const hits = ["app", "components", "lib", "content"]
      .flatMap((d) => walk(path.join(WEB, d)))
      .filter((f) => /0\.15/.test(fs.readFileSync(f, "utf-8")))
      .map((f) => path.relative(WEB, f).replace(/\\/g, "/"));
    expect(hits).toEqual(["components/sky/scale.ts"]);
  });
});

describe("REV-P2-21: anel abaixo do piso recebe eventos de ponteiro", () => {
  const css = fs.readFileSync(path.join(WEB, "components", "sky", "sky.module.css"), "utf-8");

  it("regra .dotRing declara pointer-events: all (fill none nao captura por padrao)", () => {
    const block = css.match(/\.dotRing\s*\{([^}]*)\}/);
    expect(block).not.toBeNull();
    expect(block![1]).toMatch(/fill:\s*none/);
    expect(block![1]).toMatch(/pointer-events:\s*all/);
  });

  it("o circulo de regiao abaixo do piso em rate usa a classe dotRing", () => {
    const html = renderToStaticMarkup(
      <MollweideMap
        regions={sky.regions}
        graticule={sky.graticule}
        colorCap={sky.color_cap}
        bands={sky.bands}
        mode="rate"
        hideBelow={false}
        labels={skyMapLabels}
        summary="r"
      />,
    );
    const ring = html.match(/<circle[^>]*data-below="true"[^>]*>/)![0];
    expect(ring).toMatch(/class="[^"]*dotRing/);
  });
});

describe("REV-P2-22 e REV-P2-23: legenda", () => {
  const legend = (mode: "rate" | "count", hideBelowCount?: boolean) =>
    renderToStaticMarkup(
      <MapLegend
        mode={mode}
        colorCap={sky.color_cap}
        howText="texto de leitura"
        labels={skyMapLabels}
        saturated={false}
        hideBelowCount={hideBelowCount}
      />,
    );

  it("o texto de leitura comeca com 'Como ler:'", () => {
    for (const mode of ["rate", "count"] as const) {
      const p = legend(mode).match(/<p[^>]*howText[^>]*>([\s\S]*?)<\/p>/)![1];
      expect(stripTags(p).startsWith("Como ler: texto de leitura")).toBe(true);
    }
  });

  it("nota de ocultos aparece com hideBelow+count e nao aparece com rate", () => {
    const note = esc(skyMapLabels.hiddenInCountNote);
    expect(skyMapLabels.hiddenInCountNote.length).toBeGreaterThan(0);
    expect(legend("count", true)).toContain(note);
    expect(legend("count", false)).not.toContain(note);
    expect(legend("rate", false)).not.toContain(note);
    expect(legend("rate")).not.toContain(note);
  });
});

describe("TST-02: texts.lead visivel uma vez em /ceu", () => {
  it("cabecalho + corpo mostram o lead uma unica vez (fora de title/desc de SVG)", () => {
    nav.search = "";
    const html = renderToStaticMarkup(
      <>
        <PageHeader eyebrow="e" title={sky.texts.title} lead={sky.texts.lead} showStateChip meta={meta} />
        <SkyView meta={meta} sky={sky} />
      </>,
    );
    const visible = html
      .replace(/<desc>[\s\S]*?<\/desc>/g, "")
      .replace(/<title>[\s\S]*?<\/title>/g, "")
      .replace(/aria-label="[^"]*"/g, "");
    expect(sky.texts.lead.length).toBeGreaterThan(0);
    expect(visible.split(esc(sky.texts.lead)).length - 1).toBe(1);
  });
});
