import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// O texto dentro de um SVG escala com a largura renderizada: tamanho real =
// tamanho no viewBox * (largura / largura do viewBox). Sem piso e teto na
// largura, o tamanho de tokens.css ("nenhum tamanho abaixo de 15px") nao vale:
// medido no navegador em 2026-11-25, /galaxias ia de 13,1px (390) a 31,1px (1440).

const WEB = path.join(__dirname, "..");
const read = (...p: string[]) => fs.readFileSync(path.join(WEB, ...p), "utf8");
const stripCss = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, "");

const TEXT_SM = Number(/--text-sm:\s*(\d+(?:\.\d+)?)px/.exec(read("styles", "tokens.css"))![1]);
// Teto de crescimento: ate 25% acima do token (15px -> 18,75px).
const MAX_GROWTH = 1.25;

function width(rule: string, prop: "min-width" | "max-width"): number {
  const m = new RegExp(`${prop}:\\s*(\\d+(?:\\.\\d+)?)px`).exec(rule);
  expect(m, `${prop} ausente em: ${rule.trim()}`).not.toBeNull();
  return Number(m![1]);
}

function ruleOf(css: string, selector: string): string {
  // Espaco em branco do seletor (inclusive CRLF) casa com qualquer espaco do arquivo.
  const esc = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*");
  const m = new RegExp(`${esc}\\s*\\{([^}]*)\\}`).exec(stripCss(css));
  expect(m, `regra ${selector} ausente`).not.toBeNull();
  return m![1];
}

describe("escala do texto dos graficos SVG", () => {
  const chartDir = path.join(WEB, "components", "charts");
  const viewWidths = fs
    .readdirSync(chartDir)
    .filter((f) => f.endsWith(".tsx"))
    .flatMap((f) => {
      const m = /const (?:VIEW_W|VIEWBOX_W) = (\d+);/.exec(fs.readFileSync(path.join(chartDir, f), "utf8"));
      return m ? [{ file: f, w: Number(m[1]) }] : [];
    });

  it("os 8 graficos declaram a largura do viewBox", () => {
    expect(viewWidths.length).toBe(8);
  });

  // RateByBin (720) so existe em /graficos; as demais paginas nao o usam.
  const GRAFICOS_ONLY = ["RateByBin.tsx"];
  const others = viewWidths.map((v) => v.file).filter((f) => !GRAFICOS_ONLY.includes(f));
  const pageCharts: [string, string, string[]][] = [
    ["overview", ".scroll svg", others],
    ["galaxies", ".scroll svg", others],
    ["compare", ".scroll svg", others],
    ["graficos", ".scroll svg", ["RateByBin.tsx"]],
    ["graficos", ".scrollCal svg", ["ReliabilityPlot.tsx"]],
  ];

  for (const [page, selector, files] of pageCharts) {
    it(`${page} ${selector}: piso e teto do SVG mantem o texto entre ${TEXT_SM}px e ${TEXT_SM * MAX_GROWTH}px`, () => {
      const rule = ruleOf(read("components", "pages", `${page}.module.css`), selector);
      const [min, max] = [width(rule, "min-width"), width(rule, "max-width")];
      for (const { file, w } of viewWidths.filter((v) => files.includes(v.file))) {
        expect(min / w, `${page}: piso em ${file}`).toBeGreaterThanOrEqual(1);
        expect(max / w, `${page}: teto em ${file}`).toBeLessThanOrEqual(MAX_GROWTH);
      }
    });
  }

  it("mapa Mollweide: rotulo do grafico entre o token e o teto", () => {
    const sky = read("components", "sky", "sky.module.css");
    const label = Number(/font-size:\s*(\d*\.?\d+)px/.exec(ruleOf(sky, ".gratLat,\n.gratLon"))![1]);
    const viewW = Number(/const VIEW_W = ([\d.]+);/.exec(read("components", "sky", "MollweideMap.tsx"))![1]);
    const wrap = ruleOf(sky, ".wrap");
    const at = (px: number) => (label * px) / viewW;
    expect(at(width(wrap, "min-width"))).toBeGreaterThanOrEqual(TEXT_SM);
    expect(at(width(wrap, "max-width"))).toBeLessThanOrEqual(TEXT_SM * MAX_GROWTH);
  });
});
