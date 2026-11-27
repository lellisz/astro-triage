import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CmdMap, cmdRange } from "@/components/charts/CmdMap";
import { MIN_OPACITY } from "@/components/sky/scale";
import type { Cmd, Est, Num } from "@/lib/schema";

const num = (value: number | null, text: string): Num => ({ value, text });
const est = (point: number | null, text: string, lo: number | null = null, hi: number | null = null): Est => ({
  point,
  lo,
  hi,
  text,
  ci_text: point === null ? "n/a" : `IC ${text}`,
});
const count = (html: string, needle: string) => html.split(needle).length - 1;

// Grade 3 x 3: so (0,0) e (1,1) tem valor; as outras sete estao abaixo do piso.
const capOf = (rate: number | null, count: number | null): Cmd["cap"] => ({
  rate,
  count,
  rate_text: rate === null ? null : "TXT-TAXA",
  count_text: count === null ? null : "TXT-N",
  rate_zero_text: "TXT-ZERO-TAXA",
  count_zero_text: "TXT-ZERO-N",
});

function cmdOf(cap: Cmd["cap"] = capOf(0.4, 100)): Cmd {
  const filled: Record<string, [number, number, string, string]> = {
    "0-0": [0.2, 100, "T0.2", "100"],
    "1-1": [0.4, 50, "T0.4", "50"],
  };
  const cells = [0, 1, 2].flatMap((gi) =>
    [0, 1, 2].map((ci) => {
      const f = filled[`${gi}-${ci}`];
      return f
        ? { gi, ci, n: num(f[1], f[3]), errors: num(5, "5"), rate: est(f[0], f[2], f[0] / 2, f[0] * 2), below_floor: false }
        : { gi, ci, n: num(null, "n/a"), errors: num(null, "n/a"), rate: est(null, "n/a"), below_floor: true };
    }),
  );
  return {
    g_edges: [12, 12.5, 13, 13.5],
    c_edges: [-0.5, -0.25, 0, 0.25],
    g_labels: ["GL0", "GL1", "GL2"],
    c_labels: ["CL0", "CL1", "CL2"],
    g_ticks: [
      { value: 12, text: "T12" },
      { value: 13, text: "T13" },
    ],
    c_ticks: [
      { value: -0.5, text: "T-0,5" },
      { value: 0, text: "T0,0" },
    ],
    cells,
    cap,
    n_missing: num(3, "3"),
    n_outside: num(4, "4"),
    note: "nota",
  };
}

const cmdHtml = (mode: "rate" | "count", cmd: Cmd = cmdOf()) =>
  renderToStaticMarkup(
    <CmdMap id="m" title="Mapa" cmd={cmd} mode={mode} xLabel="Cor" yLabel="Mag" scaleLabel="Escala" naText="sem dado" />,
  );
const opacities = (html: string) => [...html.matchAll(/data-point=""[^>]*fill-opacity="([\d.]+)"/g)].map((m) => Number(m[1]));

describe("CmdMap", () => {
  it("uma celula cheia por valor reportavel; as demais hachuradas e sem valor", () => {
    const html = cmdHtml("rate");
    expect(count(html, "data-cell=")).toBe(9);
    expect(count(html, "data-point")).toBe(2);
    expect(count(html, "data-below-floor")).toBe(7);
    const below = html.split('data-cell="0-1"')[1].split("</rect>")[0];
    expect(below).toContain("url(#");
    expect(below).not.toContain("fill-opacity");
    expect(below).toContain("n/a");
    expect(html).toContain("<pattern");
  });

  it("cinza com opacidade minima: o maior valor satura e o menor nunca some", () => {
    const [low, high] = opacities(cmdHtml("rate"));
    expect(high).toBe(1);
    expect(low).toBeGreaterThan(MIN_OPACITY);
    expect(low).toBeLessThan(1);
    const flat = opacities(cmdHtml("rate", cmdOf(capOf(null, null))));
    expect(flat.every((o) => o === MIN_OPACITY)).toBe(true);
  });

  it("title de cada celula traz as faixas e o texto do JSON", () => {
    const html = cmdHtml("rate");
    expect(count(html, "<title>")).toBe(9);
    expect(html).toContain("T0.2. IC T0.2");
    expect(html).toContain("Mag GL0; Cor CL0");
  });

  it("modo count pinta pelo n e imprime n.text, nao a taxa", () => {
    const html = cmdHtml("count");
    expect(opacities(html)).toEqual([1, MIN_OPACITY + (1 - MIN_OPACITY) * 0.5]);
    expect(html).toContain(": 100<");
    expect(html).not.toContain("T0.2");
  });

  it("eixos usam o texto dos ticks do JSON e a legenda mostra minimo e maximo", () => {
    const html = cmdHtml("rate");
    for (const t of ["T12", "T13", "T-0,5", "T0,0"]) expect(html).toContain(`>${t}<`);
    expect(html).toContain("TXT-ZERO-TAXA");
    expect(html).toContain("TXT-TAXA");
    expect(cmdHtml("count")).toContain("TXT-N");
    expect(cmdHtml("rate", cmdOf(capOf(null, null)))).toContain("sem dado");
  });

  it("cmdRange devolve os rotulos prontos do JSON", () => {
    expect(cmdRange(cmdOf(), { gi: 2, ci: 2 })).toEqual({ g: "GL2", c: "CL2" });
  });
});
