// Testes independentes de MplFigure (figuras matplotlib embutidas no build), escritos a partir
// dos criterios de aceite: render a partir de figures.json da fixture e recusa de SVG perigoso.
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MplFigure } from "@/components/charts/MplFigure";
import { loadFigure, loadFigures } from "@/lib/figures";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const FIX_ROOT = path.join(WEB, "test", "fixtures");
const REL = "fixture_v0";
type Entry = { title: string; caption: string; aria_label: string };
const manifest = JSON.parse(
  readFileSync(path.join(FIX_ROOT, REL, "figures", "figures.json"), "utf-8"),
) as { figures: Record<string, Entry> };

// Mesmo escape que o React aplica em texto e atributos.
const esc = (s: string) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("criterio 8: MplFigure renderiza a partir de figures.json da fixture", () => {
  beforeEach(() => {
    vi.stubEnv("ASTRO_RELEASE", REL);
    vi.stubEnv("ASTRO_WEB_DATA", FIX_ROOT);
  });

  it("a fixture tem as 14 figuras", () => {
    expect(Object.keys(manifest.figures)).toHaveLength(14);
  });

  it.each(Object.keys(manifest.figures))("%s: figure, aria-label e SVG, sem legenda repetida", (id) => {
    const e = manifest.figures[id];
    const html = renderToStaticMarkup(<MplFigure id={id} figures={loadFigures([id])} />);
    expect(html.startsWith("<figure")).toBe(true);
    const open = html.slice(0, html.indexOf(">") + 1);
    expect(open).toContain(`aria-label="${esc(e.aria_label)}"`);
    expect(open).toContain(`data-figure="${id}"`);
    expect(html).not.toContain("<figcaption");
    expect(html).not.toContain("PROVISÓRIO");
    const svg = readFileSync(path.join(FIX_ROOT, REL, "figures", `${id}.svg`), "utf-8");
    expect(html).toContain(svg.slice(0, 200));
    expect(html).toContain(`<title id="${id}-title">`);
    expect(html.endsWith("</figure>")).toBe(true);
  });

  it("release do manifesto diferente de ASTRO_RELEASE e recusada", () => {
    const root = mkdtempSync(path.join(os.tmpdir(), "mplfig-"));
    try {
      const dir = path.join(root, "outra", "figures");
      mkdirSync(dir, { recursive: true });
      writeFileSync(path.join(dir, "ok.svg"), "<svg></svg>");
      writeFileSync(
        path.join(dir, "figures.json"),
        JSON.stringify({ release: REL, provisional: true, figures: { ok: { title: "t", caption: "c", aria_label: "a" } } }),
      );
      vi.stubEnv("ASTRO_RELEASE", "outra");
      vi.stubEnv("ASTRO_WEB_DATA", root);
      expect(() => loadFigure("ok")).toThrow(/difere de ASTRO_RELEASE/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

// Pasta temporaria com <rel>/figures/<id>.svg e figures.json validos; so o SVG varia.
function withSvg(svg: string, run: () => void, id = "fig") {
  const root = mkdtempSync(path.join(os.tmpdir(), "mplfig-"));
  try {
    const dir = path.join(root, REL, "figures");
    mkdirSync(dir, { recursive: true });
    writeFileSync(path.join(dir, `${id}.svg`), svg);
    writeFileSync(
      path.join(dir, "figures.json"),
      JSON.stringify({
        release: REL,
        provisional: true,
        figures: { [id]: { title: "T", caption: "C", aria_label: "A" } },
      }),
    );
    vi.stubEnv("ASTRO_RELEASE", REL);
    vi.stubEnv("ASTRO_WEB_DATA", root);
    run();
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

const CLEAN = '<svg xmlns="http://www.w3.org/2000/svg" role="img"><title>T</title><text>PROVISÓRIO</text></svg>';
const inject = (s: string) => CLEAN.replace("</svg>", `${s}</svg>`);

describe("criterio 9: MplFigure recusa SVG perigoso", () => {
  it("controle: o SVG limpo e aceito (o erro dos casos abaixo vem do conteudo)", () => {
    withSvg(CLEAN, () => {
      expect(loadFigure("fig").svg).toBe(CLEAN);
    });
  });

  it.each([
    ["<script>", inject("<script>alert(1)</script>"), /<script/],
    ["<SCRIPT> maiusculo", inject("<SCRIPT>alert(1)</SCRIPT>"), /<script/],
    ["onload=", CLEAN.replace("<svg ", '<svg onload="alert(1)" '), /on\*=/],
    ["onclick com espaco", inject('<g ONCLICK = "alert(1)"></g>'), /on\*=/],
    ["javascript:", inject('<a href="javascript:alert(1)"><text>x</text></a>'), /javascript:/],
    ["<foreignObject>", inject("<foreignObject><div>x</div></foreignObject>"), /foreignObject/],
    ["<style> (REV-03)", inject("<style>text{fill:red}</style>"), /<style>/],
    ["prefixo que nao e <svg", `<?xml version="1.0"?>\n${CLEAN}`, /comeca com <svg/],
    ["espaco antes de <svg", ` ${CLEAN}`, /comeca com <svg/],
  ])("%s", (_name, svg, why) => {
    withSvg(svg, () => {
      expect(() => loadFigure("fig")).toThrow(why);
      expect(() => renderToStaticMarkup(<MplFigure id="fig" figures={loadFigures(["fig"])} />)).toThrow(why);
    });
  });

  it.each(["../x", "..", "a/b", "X", "-a", "", "a b", "fig.svg"])("id invalido %j", (id) => {
    // Um SVG valido existe em <rel>/x.svg: sem a validacao, "../x" o carregaria.
    const root = mkdtempSync(path.join(os.tmpdir(), "mplfig-"));
    try {
      mkdirSync(path.join(root, REL, "figures"), { recursive: true });
      writeFileSync(path.join(root, REL, "x.svg"), CLEAN);
      writeFileSync(
        path.join(root, REL, "figures", "figures.json"),
        JSON.stringify({ release: REL, provisional: true, figures: { [id]: { title: "T", caption: "C", aria_label: "A" } } }),
      );
      vi.stubEnv("ASTRO_RELEASE", REL);
      vi.stubEnv("ASTRO_WEB_DATA", root);
      expect(() => loadFigure(id)).toThrow(/id de figura invalido/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("criterio 10 (SEC-01): MplFigure recusa link externo, javascript codificado e DTD", () => {
  it.each([
    ["href javascript&#58;", inject('<a href="javascript&#58;alert(1)"><text>x</text></a>')],
    ["xlink:href javascript&#x3A;", inject('<a xlink:href="javascript&#x3A;alert(1)"><text>x</text></a>')],
    ["<a href> externo", inject('<a href="https://example.com/"><text>x</text></a>')],
    ["<!DOCTYPE", `<svg xmlns="http://www.w3.org/2000/svg"><!DOCTYPE svg><text>x</text></svg>`],
    ["<!ENTITY", `<svg xmlns="http://www.w3.org/2000/svg"><!ENTITY x "y"><text>x</text></svg>`],
  ])("%s", (_name, svg) => {
    withSvg(svg, () => {
      expect(() => loadFigure("fig")).toThrow();
    });
  });
});
