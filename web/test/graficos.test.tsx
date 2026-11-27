import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { GraficosView } from "@/components/pages/GraficosView";
import { cmdRange } from "@/components/charts/CmdMap";
import { MIN_OPACITY } from "@/components/sky/scale";
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

  it("seis figuras (o by-z tem uma por classe), cada uma com alternativa em tabela", () => {
    expect(count(html, "<figure")).toBe(6);
    expect(count(html, "<details")).toBe(5);
    expect(count(html, "<table")).toBe(5 + 1);
    expect(count(html, "<th scope=\"col\"")).toBeGreaterThan(8);
    expect(count(html, "Ver os números")).toBe(5);
    expect(count(html, 'role="img"')).toBeGreaterThanOrEqual(3);
  });

  it("o by-z tem um grafico por classe, com os rotulos de faixa exatos do JSON da classe", () => {
    const section = html.slice(html.indexOf('id="graf-by-z"'), html.indexOf('id="graf-confusion"'));
    expect(count(section, "<figure")).toBe(graficos.error_by_z.series.length);
    expect(graficos.error_by_z.series.length).toBe(2);
    for (const s of graficos.error_by_z.series) {
      const start = section.indexOf(`id="graf-by-z-chart-${s.cls}"`);
      expect(start).toBeGreaterThan(-1);
      const fig = section.slice(start, section.indexOf("</figure>", start));
      const labels = [...fig.matchAll(/<text[^>]*data-bin-label[^>]*>([^<]*)<\/text>/g)].map((m) => decode(m[1]));
      expect(labels).toEqual(s.rows.map((r) => r.label));
    }
  });

  it("IC sempre visivel (Tarefa 34): whiskers na taxa, na matriz e na calibracao com tecnico=false", () => {
    const tech = render("tecnico=1");
    // o modo simples mostra os mesmos IC do modo tecnico
    expect(count(html, "data-ci")).toBeGreaterThan(0);
    expect(count(html, "data-ci")).toBe(count(tech, "data-ci"));
    expect(count(html, "data-cal-ci")).toBeGreaterThan(0);
    expect(count(html, "data-cal-ci")).toBe(count(tech, "data-cal-ci"));
    // a legenda do intervalo nao depende do modo tecnico
    expect(textOf(html)).toContain("Barras: intervalo de confiança (Wilson, por objeto)");
  });

  it("sem curva crua no modo simples (raw continua so tecnico)", () => {
    expect(count(html, "data-cal-raw")).toBe(0);
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

  it("nao ha secao de ECE; o mapa cor-magnitude aparece com o texto do JSON", () => {
    expect(graficos.ece).toBeNull();
    expect(textOf(html)).not.toMatch(/\bECE\b/i);
    expect(textOf(html)).toContain("Mapa cor-magnitude");
    expect(textOf(html)).toContain(graficos.cmd!.note);
  });

  it("nenhum 'empate' nem 'ADR'", () => {
    const text = textOf(html);
    expect(text).not.toMatch(/empate/i);
    expect(text).not.toContain("ADR");
  });

  it("figuras rotuladas e tabelas reais", () => {
    expect(count(html, "<figcaption")).toBe(5);
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

  it("as mesmas figuras, mais o mapa de contagem so no modo tecnico", () => {
    expect(count(html, "<figure")).toBe(7);
    expect(count(html, "<details")).toBe(5);
    expect(count(html, 'id="graf-cmd-count"')).toBe(1);
    expect(count(render(""), 'id="graf-cmd-count"')).toBe(0);
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
    const rows = html.slice(0, html.indexOf('id="graf-cmd"'));
    expect(count(rows, "data-below-floor")).toBe(below.length);
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

describe("/graficos: mapa cor-magnitude", () => {
  const cmd = graficos.cmd!;

  it("a fixture traz as 252 celulas, todas abaixo do piso: hachuradas e sem valor", () => {
    const html = render("");
    const section = html.slice(html.indexOf('id="graf-cmd"'));
    expect(cmd.cells).toHaveLength(252);
    expect(cmd.cells.every((c) => c.below_floor)).toBe(true);
    expect(count(section, "data-cell=")).toBe(252);
    expect(count(section, "data-below-floor")).toBe(cmd.cells.filter((c) => c.below_floor).length);
    expect(count(section, "data-point")).toBe(cmd.cells.length - count(section, "data-below-floor"));
    expect(textOf(section)).toContain("Célula hachurada");
  });

  const dense = (): Graficos => {
    const data = structuredClone(graficos);
    const c = data.cmd!;
    for (const [i, cell] of c.cells.slice(0, 3).entries()) {
      const rate = 0.1 * (i + 1);
      cell.below_floor = false;
      cell.n = { value: 50 * (i + 1), text: `N${i}` };
      cell.errors = { value: 5, text: `E${i}` };
      cell.rate = { point: rate, lo: rate / 2, hi: rate * 2, text: `R${i}`, ci_text: `CI${i}` };
    }
    c.cap = { ...c.cap, rate: 0.3, count: 150 };
    return data;
  };

  it("celulas com valor viram retangulos cinza com opacidade minima e linhas da tabela", () => {
    const html = render("", dense());
    const section = html.slice(html.indexOf('id="graf-cmd"'));
    expect(count(section, "data-point")).toBe(3);
    const ops = [...section.matchAll(/data-point=""[^>]*fill-opacity="([\d.]+)"/g)].map((m) => Number(m[1]));
    expect(ops).toHaveLength(3);
    expect(Math.min(...ops)).toBeGreaterThan(MIN_OPACITY);
    expect(Math.max(...ops)).toBe(1);
    const table = section.slice(section.indexOf("<tbody"), section.indexOf("</tbody>"));
    expect(count(table, "<tr")).toBe(3);
    for (const i of [0, 1, 2]) expect(table).toContain(`R${i}`);
    const range = cmdRange(cmd, cmd.cells[0]);
    expect(table).toContain(range.g);
    expect(table).toContain(range.c);
    expect(table).not.toContain("CI0");
  });

  it("no modo tecnico o mapa de contagem usa n e a tabela ganha n e intervalo", () => {
    const html = render("tecnico=1", dense());
    const from = html.slice(html.indexOf('id="graf-cmd-count"'));
    expect(count(from.slice(0, from.indexOf("</figure>")), "data-point")).toBe(3);
    expect(from).toContain("N2");
    expect(html.slice(html.indexOf("<tbody", html.indexOf('id="graf-cmd"')))).toContain("CI0");
  });

  it("sem cmd no JSON a secao nao existe", () => {
    const data = structuredClone(graficos);
    data.cmd = null;
    const html = render("", data);
    expect(html).not.toContain('id="graf-cmd');
    expect(textOf(html)).not.toContain("Mapa cor-magnitude");
    expect(count(html, "<figure")).toBe(5);
  });

  it("nenhuma chave por objeto no JSON versionado e o tamanho e pequeno", () => {
    const text = JSON.stringify(graficos);
    expect(text).not.toMatch(/"(source_id|ra|dec|l|b|healpix_l5|hp)":/);
    expect(Buffer.byteLength(text)).toBeLessThan(150 * 1024);
  });
});
