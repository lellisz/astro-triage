import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { GaiaFigure } from "@/components/media/GaiaFigure";
import { OverviewView } from "@/components/pages/OverviewView";
import { SkyView } from "@/components/pages/SkyView";
import { gaiaSkymap } from "@/content/images";
import { kitLabels, skyLabels } from "@/content/labels";
import type { Meta, Overview, Sky } from "@/lib/schema";

const nav = vi.hoisted(() => ({ search: "" }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => "/",
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

const FIX = fileURLToPath(new URL("./fixtures", import.meta.url));
const COMPONENTS = fileURLToPath(new URL("../components", import.meta.url));
const load = <T,>(name: string): T =>
  JSON.parse(readFileSync(path.join(FIX, "fixture_v0", `${name}.json`), "utf-8")) as T;
const meta = load<Meta>("meta");
const overview = load<Overview>("overview");
const sky = load<Sky>("sky");

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const countText = (html: string, needle: string) => html.split(needle).length - 1;

function cssFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = path.join(dir, n);
    return statSync(p).isDirectory() ? cssFiles(p) : p.endsWith(".module.css") ? [p] : [];
  });
}

// Unica forma de trilho fluido permitida fora de minmax(0, ...): grade responsiva
// repeat(auto-fit|auto-fill, minmax(<n>rem, 1fr)). Lista explicita (arquivo:linha).
const ALLOWED_AUTO_FIT = new Set([
  "pages/compare.module.css",
  "pages/method.module.css",
  "pages/page.module.css",
]);

describe("(a) TST-06: trilhos de grid nao estouram o conteudo", () => {
  const files = cssFiles(COMPONENTS);

  it("ha arquivos css para inspecionar", () => {
    expect(files.length).toBeGreaterThan(10);
  });

  it("nenhum trilho em fr fora de minmax(0, ...) (exceto auto-fit/auto-fill com minmax(<n>rem, 1fr))", () => {
    const offenders: string[] = [];
    const seenAuto: string[] = [];
    for (const f of files) {
      const rel = path.relative(COMPONENTS, f).replace(/\\/g, "/");
      readFileSync(f, "utf-8")
        .split(/\r?\n/)
        .forEach((line, i) => {
          const m = line.match(/grid-template-columns\s*:\s*([^;]+);/);
          if (!m) return;
          let v = m[1];
          const auto = v.match(/repeat\(\s*auto-(?:fit|fill)\s*,\s*minmax\(\s*[\d.]+rem\s*,\s*1fr\s*\)\s*\)/);
          if (auto) {
            seenAuto.push(rel);
            v = v.replace(auto[0], "");
          }
          // remove trilhos seguros e confere se sobrou algum fr
          const rest = v.replace(/minmax\(\s*0(?:px)?\s*,\s*[\d.]+fr\s*\)/g, "");
          if (/[\d.]*fr\b/.test(rest)) offenders.push(`${rel}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(offenders).toEqual([]);
    // as grades auto-fit existentes sao exatamente as listadas
    expect(new Set(seenAuto)).toEqual(ALLOWED_AUTO_FIT);
  });

  it("detector: pega 1fr solto e aceita minmax(0, 1fr)", () => {
    const bad = "grid-template-columns: 1fr 2fr;";
    const ok = "grid-template-columns: repeat(3, minmax(0, 1fr));";
    const flag = (s: string) => /[\d.]*fr\b/.test(s.replace(/minmax\(\s*0(?:px)?\s*,\s*[\d.]+fr\s*\)/g, ""));
    expect(flag(bad)).toBe(true);
    expect(flag(ok)).toBe(false);
  });

  it(".main de Shell.module.css tem min-width: 0", () => {
    const css = readFileSync(path.join(COMPONENTS, "shell/Shell.module.css"), "utf-8");
    const block = css.match(/(^|\n)\.main\s*\{([^}]*)\}/);
    expect(block).not.toBeNull();
    expect(block![2]).toMatch(/min-width:\s*0\s*;/);
  });
});

describe("SkyView com tecnico=1", () => {
  nav.search = "tecnico=1";
  const html = renderToStaticMarkup(<SkyView meta={meta} sky={sky} />);

  it("(b) REV-P2-30: cada titulo de tabela tecnica aparece exatamente 1 vez", () => {
    for (const t of [skyLabels.topTitle, skyLabels.bandsTitle]) {
      // so o <h2> do painel; o caption da tabela entra apenas no aria-label da regiao
      expect(countText(html, `>${escapeHtml(t)}</`), t).toBe(1);
      expect(html).not.toMatch(new RegExp(`<p[^>]*>${escapeHtml(t)}</p>`));
    }
  });

  it("(b) todo <table aria-labelledby=X> aponta para um id X existente", () => {
    const tables = [...html.matchAll(/<table[^>]*aria-labelledby="([^"]+)"/g)].map((m) => m[1]);
    expect(tables.length).toBe(2);
    for (const id of tables) {
      expect(countText(html, `id="${id}"`), id).toBe(1);
    }
  });

  it("(c) position_note, neighbour_note e spread_note estao dentro do <section> do mapa", () => {
    const m = html.match(/<section[^>]*aria-labelledby="sky-map"[^>]*>([\s\S]*?)<\/section>/);
    expect(m).not.toBeNull();
    const section = m![1];
    for (const k of ["position_note", "neighbour_note", "spread_note"] as const) {
      const text = sky.texts[k];
      expect(text, k).toBeTruthy();
      expect(section, k).toContain(escapeHtml(text));
      expect(countText(html, escapeHtml(text)), `${k} so uma vez`).toBe(1);
    }
  });
});

describe("(d) OverviewView: ContextStrip", () => {
  nav.search = "";
  const html = renderToStaticMarkup(
    <OverviewView
      meta={meta}
      rows={overview.rows}
      accuracyNote={overview.accuracy_note}
      technical={overview.technical}
    />,
  );

  it("tem 3 chips com os .text de n_test, n_pixels e n_pairs", () => {
    const box = overview.technical.data_box;
    const ul = html.match(/<ul class="[^"]*strip[^"]*">([\s\S]*?)<\/ul>/);
    expect(ul).not.toBeNull();
    const chips = [...ul![1].matchAll(/<li[^>]*>([\s\S]*?)<\/li>/g)].map((m) => m[1]);
    expect(chips).toHaveLength(3);
    const texts = [box.n_test.text, box.n_pixels.text, box.n_pairs.text];
    texts.forEach((t, i) => {
      expect(t).toBeTruthy();
      expect(chips[i], `chip ${i}`).toContain(escapeHtml(t));
    });
  });
});

describe("(e) GaiaFigure split", () => {
  const html = renderToStaticMarkup(<GaiaFigure variant="split" />);

  it("mantem alt, legenda de nao-saida e credito", () => {
    expect(gaiaSkymap.alt.length).toBeGreaterThan(10);
    expect(html).toContain(`alt="${escapeHtml(gaiaSkymap.alt)}"`);
    expect(gaiaSkymap.caption).toMatch(/não representa a saída do classificador/);
    expect(html).toContain(escapeHtml(gaiaSkymap.caption));
    expect(html).toContain(`${kitLabels.creditPrefix}: ${escapeHtml(gaiaSkymap.credit)}`);
    expect(html).toMatch(/<figure[^>]*>[\s\S]*<img[\s\S]*<figcaption/);
  });
});
