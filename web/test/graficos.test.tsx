import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { GraficosView } from "@/components/pages/GraficosView";
import { ptPct } from "@/lib/fmt";
import type { Graficos, Meta } from "@/lib/schema";

const nav = vi.hoisted(() => ({ search: "" }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => "/graficos/",
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

const FIX = fileURLToPath(new URL("./fixtures", import.meta.url));
const load = <T,>(name: string): T =>
  JSON.parse(readFileSync(path.join(FIX, "fixture_v0", `${name}.json`), "utf-8")) as T;
const meta = load<Meta>("meta");
const graficos = load<Graficos>("graficos");

const render = (search: string, data: Graficos = graficos): string => {
  nav.search = search;
  return renderToStaticMarkup(<GraficosView meta={meta} graficos={data} />);
};
const count = (html: string, needle: string) => html.split(needle).length - 1;
const decode = (s: string) =>
  s
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
const textOf = (html: string) => decode(html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " "));
const labelsOf = (html: string) => [...html.matchAll(/aria-label="([^"]*)"/g)].map((m) => decode(m[1]));

const strings = (v: unknown, out: string[] = []): string[] => {
  if (typeof v === "string") out.push(v);
  else if (Array.isArray(v)) v.forEach((x) => strings(x, out));
  else if (v && typeof v === "object") Object.values(v).forEach((x) => strings(x, out));
  return out;
};
const NUMBER = /\d+(?:[.,]\d+)*%?/g;

describe("/graficos: modo simples", () => {
  const html = render("");

  it("quatro figuras, cada uma com alternativa em tabela", () => {
    expect(count(html, "<figure")).toBe(4);
    expect(count(html, "<details")).toBe(4);
    expect(count(html, "<table")).toBe(4 + 1);
    expect(count(html, "<th scope=\"col\"")).toBeGreaterThan(8);
    expect(count(html, "Ver os números")).toBe(4);
    expect(count(html, 'role="img"')).toBeGreaterThanOrEqual(3);
  });

  it("sem whiskers, sem curva crua e sem colunas tecnicas", () => {
    expect(count(html, "data-ci")).toBe(0);
    expect(count(html, "data-cal-ci")).toBe(0);
    expect(count(html, "data-cal-raw")).toBe(0);
    expect(html).not.toContain("Intervalo de confiança");
  });

  it("textos de escopo, piso e notas vem do JSON", () => {
    const text = textOf(html);
    for (const t of [
      graficos.scope_text,
      graficos.floor_text,
      graficos.calibration.networks_note,
      graficos.error_by_g.missing_note,
      graficos.error_by_z.star_note,
    ]) {
      expect(text).toContain(t);
    }
  });

  it("nao ha secao de ECE nem de mapa cor-magnitude", () => {
    expect(graficos.ece).toBeNull();
    expect(graficos.cmd).toBeNull();
    expect(textOf(html)).not.toMatch(/\bECE\b|cor-magnitude/i);
  });

  it("nenhum 'empate' nem 'ADR'", () => {
    const text = textOf(html);
    expect(text).not.toMatch(/empate/i);
    expect(text).not.toContain("ADR");
  });

  it("figuras rotuladas e tabelas reais", () => {
    expect(count(html, "<figcaption")).toBe(3);
    expect(count(html, "aria-labelledby")).toBeGreaterThanOrEqual(7);
    expect(html).toContain("<caption>");
  });
});

describe("/graficos: modo tecnico", () => {
  const html = render("tecnico=1");

  it("acrescenta whiskers, curva crua e colunas de n e intervalo", () => {
    expect(count(html, "data-ci")).toBeGreaterThan(0);
    expect(count(html, "data-cal-ci")).toBeGreaterThan(0);
    expect(count(html, "data-cal-raw")).toBeGreaterThan(0);
    expect(html).toContain("Intervalo de confiança");
    expect(html).toContain("Objetos");
  });

  it("as mesmas quatro figuras", () => {
    expect(count(html, "<figure")).toBe(4);
    expect(count(html, "<details")).toBe(4);
  });
});

describe("/graficos: nenhum numero formatado no front", () => {
  const allowed = new Set<string>();
  for (const s of strings(graficos)) for (const m of s.matchAll(NUMBER)) allowed.add(m[0].replace(/%$/, ""));

  for (const search of ["", "tecnico=1"]) {
    it(`todo algarismo no DOM e texto do JSON ou tick de fmt.ts (${search || "simples"})`, () => {
      const html = render(search);
      const dom = [textOf(html), ...labelsOf(html)].join(" ");
      const unknown: string[] = [];
      for (const m of dom.matchAll(NUMBER)) {
        const token = m[0];
        if (allowed.has(token.replace(/%$/, ""))) continue;
        // tick de eixo: porcentagem que e a saida exata de ptPct
        const body = token.slice(0, -1);
        const digits = body.includes(",") ? body.split(",")[1].length : 0;
        if (token.endsWith("%") && ptPct(Number(body.replace(",", ".")) / 100, digits) === token) continue;
        unknown.push(token);
      }
      expect(unknown).toEqual([]);
    });
  }

  it("o codigo da view nao formata numero nem escreve algarismo em texto", () => {
    const src = readFileSync(fileURLToPath(new URL("../components/pages/GraficosView.tsx", import.meta.url)), "utf-8")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/(^|[^:])\/\/.*$/gm, "$1");
    expect(src).not.toMatch(/toFixed|toLocaleString|Intl\.|ptPct|ptNum|ptInt|ptG\b/);
    expect(src.match(/"[^"\n]*\d[^"\n]*"/g) ?? []).toEqual([]);
    expect(src).not.toMatch(/#[0-9a-fA-F]{3,8}\b|\brgba?\(/);
  });
});

describe("/graficos: abaixo do piso mostra o texto NA do JSON", () => {
  it("linhas por faixa abaixo do piso usam rate.text, nao um valor", () => {
    const below = [...graficos.error_by_g.series, ...graficos.error_by_z.series].flatMap((s) =>
      s.rows.filter((r) => r.below_floor),
    );
    expect(below.length).toBeGreaterThan(0);
    const html = render("");
    expect(count(html, "data-below-floor")).toBe(below.length);
    for (const r of below) expect(r.rate.point).toBeNull();
    expect(textOf(html)).toContain(below[0].rate.text);
  });

  it("celula da matriz abaixo do piso mostra share.text sem sombreado nem numero", () => {
    const data = structuredClone(graficos);
    const cell = data.confusion.rows[0].cells[0];
    cell.below_floor = true;
    cell.share = { point: null, lo: null, hi: null, text: "SEM-VALOR", ci_text: "SEM-VALOR" };
    const html = render("", data);
    const cellHtml = new RegExp(`data-cell="${data.confusion.rows[0].true}-${cell.pred}"[\\s\\S]*?</td>`).exec(html)![0];
    expect(cellHtml).toContain("SEM-VALOR");
    expect(cellHtml).not.toContain("data-shade");
  });
});
