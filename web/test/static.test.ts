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
  ["graficos/index.html", "/graficos/", "graficos"],
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
    it(`${file}: lang pt-BR, selo, 6 links, sem CDN de fontes`, () => {
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

// ---- P2-T9: invariantes estaticos adicionais (A8, A9, A11) ----

const rel = (f: string) => path.relative(WEB, f).split(path.sep).join("/");
const COMPONENT_DIRS = ["charts", "sky", "kit", "media", "pages"];
const componentTsx = COMPONENT_DIRS.flatMap((d) => walk(path.join(WEB, "components", d), [".tsx"]));

describe("A11 varredura de cor literal cobre os diretorios novos", () => {
  it("charts, sky, kit, media e pages entram na lista de tsx e css varridos", () => {
    const tsx = walk(path.join(WEB, "components"), [".tsx"]).map(rel);
    const css = walk(path.join(WEB, "components"), [".css"]).map(rel);
    for (const d of COMPONENT_DIRS) {
      expect(tsx.some((f) => f.startsWith(`components/${d}/`)), `tsx em ${d}`).toBe(true);
    }
    for (const d of ["charts", "sky", "kit", "media"]) {
      expect(css.some((f) => f.startsWith(`components/${d}/`)), `css em ${d}`).toBe(true);
    }
  });

  it("cor literal e detectada se aparecesse (a varredura nao e cega)", () => {
    const re = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/g;
    expect(`fill="#fff"`.match(re)).not.toBeNull();
    expect(`color: rgb(0,0,0)`.match(re)).not.toBeNull();
    expect(`fill="var(--ink)"`.match(re)).toBeNull();
  });
});

describe("A8 nenhum algarismo de dado escrito a mao em componentes", () => {
  const PCT = /["'`>][^"'`<\n]*\d+[,.]?\d*\s*%/;
  const NA_LIT = /["'`>][^"'`<\n]*n\/a/;

  it("ha componentes para varrer", () => {
    expect(componentTsx.length).toBeGreaterThan(10);
  });

  it("nenhum literal com 'n/a' ou com numero seguido de %", () => {
    const bad: string[] = [];
    for (const f of componentTsx) {
      const src = stripTsComments(fs.readFileSync(f, "utf8"));
      src.split("\n").forEach((line, i) => {
        if (PCT.test(line) || NA_LIT.test(line)) bad.push(`${rel(f)}:${i + 1}: ${line.trim()}`);
      });
    }
    expect(bad).toEqual([]);
  });

  it("os padroes pegariam os casos reais (teste do teste)", () => {
    expect(PCT.test(`<span>{x}</span><p>12,5%</p>`)).toBe(true);
    expect(PCT.test(`const t = "85%";`)).toBe(true);
    expect(PCT.test(`const t = \`taxa 7 %\`;`)).toBe(true);
    expect(NA_LIT.test(`const t = "n/a";`)).toBe(true);
    expect(NA_LIT.test(`<span>n/a</span>`)).toBe(true);
    expect(PCT.test(`viewBox="0 0 640 208" r={6}`)).toBe(false);
    expect(PCT.test(`{ptPct(f * axisMax, 0)}`)).toBe(false);
  });
});

describe("A9 TST-01 invariantes de repositorio", () => {
  const sourceFiles = ["app", "components", "content"].flatMap((d) => walk(path.join(WEB, d), [".ts", ".tsx"]));

  it("nenhum <img com src http em web/out/**/*.html", () => {
    const htmls = walk(OUT, [".html"]);
    if (htmls.length === 0) {
      console.warn("static.test: web/out sem html, verificacao de <img ignorada");
      return;
    }
    const bad = htmls.filter((f) => /<img\b[^>]*\bsrc\s*=\s*["']?https?:/i.test(fs.readFileSync(f, "utf8"))).map(rel);
    expect(bad).toEqual([]);
  });

  it("nenhum <img src=http nem src: http em app, components e content", () => {
    expect(sourceFiles.length).toBeGreaterThan(10);
    const bad = sourceFiles
      .filter((f) => {
        const s = stripTsComments(fs.readFileSync(f, "utf8"));
        return /<img\b[^>]*\bsrc\s*=\s*\{?\s*["'`]https?:/i.test(s) || /\bsrc\s*:\s*["'`]https?:/i.test(s);
      })
      .map(rel);
    expect(bad).toEqual([]);
  });

  it("web/public contem so gaia/gaia-skymap-monochrome.png e gaia/CREDITS.md", () => {
    const files = walk(path.join(WEB, "public"), [""]).map((f) => path.relative(path.join(WEB, "public"), f).split(path.sep).join("/"));
    expect(files.sort()).toEqual(["gaia/CREDITS.md", "gaia/gaia-skymap-monochrome.png"]);
  });

  it("nenhum arquivo por objeto em web/data e web/test/fixtures", () => {
    const MAX_JSON_PER_RELEASE = 20;
    for (const root of [path.join(WEB, "data"), path.join(WEB, "test", "fixtures")]) {
      const all = walk(root, [""]);
      for (const f of all) expect(path.basename(f), rel(f)).not.toMatch(/source_?id/i);
      const perDir = new Map<string, number>();
      for (const f of all.filter((x) => x.endsWith(".json"))) {
        const d = path.dirname(f);
        perDir.set(d, (perDir.get(d) ?? 0) + 1);
        expect(fs.readFileSync(f, "utf8"), `${rel(f)} contem a chave source_id`).not.toMatch(/"source_id"\s*:/);
      }
      for (const [d, n] of perDir) expect(n, `${rel(d)}: ${n} .json`).toBeLessThanOrEqual(MAX_JSON_PER_RELEASE);
    }
  });

  it("a deteccao de source_id funcionaria (teste do teste)", () => {
    expect(/"source_id"\s*:/.test('{"source_id": 1}')).toBe(true);
    expect(/"source_id"\s*:/.test('{"n_sources": 1}')).toBe(false);
  });

  it("web/components/sky nao usa Math.sin/cos/asin/acos/atan/sqrt", () => {
    const files = walk(path.join(WEB, "components", "sky"), [".ts", ".tsx"]);
    expect(files.length).toBeGreaterThan(0);
    const bad = files.filter((f) => /\bMath\.(sin|cos|asin|acos|atan2?|sqrt)\b/.test(stripTsComments(fs.readFileSync(f, "utf8")))).map(rel);
    expect(bad).toEqual([]);
  });
});
