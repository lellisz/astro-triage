import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const WEB = path.join(__dirname, "..");
const TOKENS = path.join(WEB, "styles", "tokens.css");
const OUT = path.join(WEB, "out");

function walk(dir: string, exts: string[]): string[] {
  if (!fs.existsSync(dir)) return [];
  const res: string[] = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) res.push(...walk(p, exts));
    else if (exts.some((x) => e.name.endsWith(x))) res.push(p);
  }
  return res;
}

const stripCssComments = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, "");
const stripTsComments = (s: string) =>
  s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");

function lum(hex: string): number {
  const h = hex.replace("#", "");
  const c = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const l = c.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2];
}
function ratio(a: string, b: string): number {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

const tokensRaw = fs.readFileSync(TOKENS, "utf8");
const tokensCode = stripCssComments(tokensRaw);
const vars = new Map<string, string>();
for (const m of tokensCode.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) vars.set(m[1], m[2].trim());

describe("tokens.css", () => {
  it("toda cor declarada e neutra (sem matiz)", () => {
    const colors: string[] = [];
    for (const m of tokensCode.matchAll(/#([0-9a-fA-F]+)\b/g)) {
      const h = m[1];
      expect([3, 6, 8]).toContain(h.length);
      const full = h.length === 3 ? [...h].map((c) => c + c).join("") : h.slice(0, 6);
      const [r, g, b] = [0, 2, 4].map((i) => full.slice(i, i + 2));
      expect(`${m[0]}:${g}`).toBe(`${m[0]}:${r}`);
      expect(`${m[0]}:${b}`).toBe(`${m[0]}:${r}`);
      colors.push(m[0]);
    }
    expect(colors.length).toBeGreaterThan(10);
    for (const m of tokensCode.matchAll(/rgba?\(([^)]*)\)/g)) {
      const parts = m[1].split(",").map((s) => s.trim());
      const rgb = parts.slice(0, 3).join(",");
      expect(["0,0,0", "255,255,255"], `cor com matiz: ${m[0]}`).toContain(rgb);
    }
    expect(tokensCode).not.toMatch(/\b(hsl|hsla|hwb|lab|lch|oklab|oklch|color)\(/);
  });

  it("define --target-min: 44px", () => {
    expect(vars.get("--target-min")).toBe("44px");
  });

  it("texto secundario e todos os tamanhos de texto >= 15px", () => {
    const sizes = [...vars].filter(([k]) => /^--text-(sm|base|md|lg|[2-9]?xl)$/.test(k));
    expect(sizes.length).toBeGreaterThanOrEqual(5);
    for (const [k, v] of sizes) {
      const px = parseFloat(v);
      expect(v, k).toMatch(/px$/);
      expect(px, k).toBeGreaterThanOrEqual(15);
    }
  });

  it("pares documentados no cabecalho: contraste calculado respeita os minimos", () => {
    const header = tokensRaw.slice(0, tokensRaw.indexOf("*/"));
    const lines = header.split("\n");
    let min = 0;
    let n = 0;
    for (const line of lines) {
      if (/texto sobre fundo/.test(line)) min = 4.5;
      else if (/grafico e componente/.test(line)) min = 3;
      else if (/^\s*\*\s+foco\s*$/.test(line)) min = 3;
      else if (/Decorativos/.test(line)) min = 0;
      const m = line.match(/(--[\w-]+)\s+(#[0-9a-f]{6})\s*\/\s*(--[\w-]+)\s+(#[0-9a-f]{6})\s+([\d.]+)/i);
      if (!m) continue;
      expect(min, `par fora de secao com minimo: ${line}`).toBeGreaterThan(0);
      const [, fgVar, fgHex, bgVar, bgHex, doc] = m;
      // o hex documentado precisa ser o valor real do token
      expect(vars.get(fgVar), fgVar).toBe(fgHex);
      expect(vars.get(bgVar), bgVar).toBe(bgHex);
      const r = ratio(fgHex, bgHex);
      expect(r, `${fgVar}/${bgVar}`).toBeGreaterThanOrEqual(min);
      expect(Math.abs(r - parseFloat(doc)), `documentado ${doc} vs calculado ${r.toFixed(2)}`).toBeLessThan(0.02);
      n++;
    }
    expect(n).toBeGreaterThanOrEqual(15);
  });
});

describe("codigo-fonte", () => {
  const cssFiles = [...walk(path.join(WEB, "app"), [".css"]), ...walk(path.join(WEB, "components"), [".css"])];
  const tsxFiles = [...walk(path.join(WEB, "app"), [".tsx"]), ...walk(path.join(WEB, "components"), [".tsx"])];

  it("ha arquivos para varrer", () => {
    expect(cssFiles.length).toBeGreaterThan(3);
    expect(tsxFiles.length).toBeGreaterThan(3);
  });

  it("nenhuma cor literal em css/tsx", () => {
    const bad: string[] = [];
    for (const f of [...cssFiles, ...tsxFiles]) {
      const src = f.endsWith(".css") ? stripCssComments(fs.readFileSync(f, "utf8")) : stripTsComments(fs.readFileSync(f, "utf8"));
      for (const m of src.matchAll(/#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/g)) bad.push(`${path.relative(WEB, f)}: ${m[0]}`);
    }
    expect(bad).toEqual([]);
  });

  it("content/labels.ts nao tem algarismos", () => {
    const src = fs.readFileSync(path.join(WEB, "content", "labels.ts"), "utf8");
    expect(src.match(/\d/g) ?? []).toEqual([]);
  });

  it('nenhum "ADR" em content, app e components', () => {
    const bad: string[] = [];
    for (const d of ["content", "app", "components"]) {
      for (const f of walk(path.join(WEB, d), [".ts", ".tsx", ".css", ".json", ".md"])) {
        if (/ADR/.test(fs.readFileSync(f, "utf8"))) bad.push(path.relative(WEB, f));
      }
    }
    expect(bad).toEqual([]);
  });

  it("nenhum href http: em codigo", () => {
    const bad: string[] = [];
    for (const d of ["app", "components", "content", "lib"]) {
      for (const f of walk(path.join(WEB, d), [".ts", ".tsx"])) {
        if (/href\s*[=:]\s*\{?\s*["'`]http:/.test(fs.readFileSync(f, "utf8"))) bad.push(path.relative(WEB, f));
      }
    }
    expect(bad).toEqual([]);
  });
});

function sealCandidates(): string[] {
  const metas = [
    path.join(WEB, "test", "fixtures", "fixture_v0", "meta.json"),
    path.join(WEB, "data", "full_v0", "meta.json"),
  ];
  const rel = process.env.ASTRO_RELEASE;
  const root = process.env.ASTRO_WEB_DATA;
  if (rel && root) metas.unshift(path.join(root, rel, "meta.json"));
  const seals: string[] = [];
  for (const p of metas) {
    if (!fs.existsSync(p)) continue;
    const t = (JSON.parse(fs.readFileSync(p, "utf8")) as { seal_text?: string }).seal_text;
    if (t) seals.push(t);
  }
  return seals;
}

const PAGES: [string, string, string][] = [
  ["index.html", "/", "inicio"],
  ["ceu/index.html", "/ceu/", "ceu"],
  ["galaxias/index.html", "/galaxias/", "galaxias"],
  ["comparacao-dl/index.html", "/comparacao-dl/", "comparacao"],
  ["metodo/index.html", "/metodo/", "metodo"],
];

describe("saida estatica (web/out)", () => {
  if (!fs.existsSync(OUT)) {
    it.skip("web/out nao existe: rode `npm run build` antes (teste pulado)", () => {});
    console.warn("static.test: web/out ausente, testes da saida estatica pulados");
    return;
  }
  const escape = (s: string) => s.replace(/&/g, "&amp;");

  for (const [file, , name] of PAGES) {
    it(`${file}: lang pt-BR, selo, 5 links, sem CDN de fontes`, () => {
      const p = path.join(OUT, file);
      expect(fs.existsSync(p), p).toBe(true);
      const html = fs.readFileSync(p, "utf8");
      expect(html, name).toMatch(/<html[^>]*\blang="pt-BR"/);
      const seals = sealCandidates();
      expect(seals.length).toBeGreaterThan(0);
      expect(seals.some((s) => html.includes(s) || html.includes(escape(s))), `selo ausente; esperado um de ${JSON.stringify(seals)}`).toBe(true);
      for (const [, href] of PAGES) {
        const re = new RegExp(`href="${href}(\\?[^"]*)?"`);
        expect(html, `link ${href}`).toMatch(re);
      }
      expect(html).not.toMatch(/fonts\.googleapis|fonts\.gstatic/);
      expect(html).toMatch(/\/_next\/static\/media\/[^"')]+\.woff2/);
    });
  }

  it("fontes servidas de /_next/static/media e presentes no disco", () => {
    const media = path.join(OUT, "_next", "static", "media");
    expect(fs.existsSync(media)).toBe(true);
    expect(fs.readdirSync(media).filter((f) => f.endsWith(".woff2")).length).toBeGreaterThanOrEqual(1);
    const css = walk(path.join(OUT, "_next", "static"), [".css"]).map((f) => fs.readFileSync(f, "utf8")).join("\n");
    expect(css).not.toMatch(/fonts\.googleapis|fonts\.gstatic/);
    // o CSS gerado referencia as fontes por caminho relativo (../media/) ao
    // proprio /_next/static/css: equivale a /_next/static/media.
    expect(css).toMatch(/url\((\.\.|\/_next\/static)\/media\/[^)]+\.woff2\)/);
    expect(css).not.toMatch(/url\(\s*["']?https?:/);
  });
});
