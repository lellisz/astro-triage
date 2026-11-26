import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CmdMap, cmdRange } from "@/components/charts/CmdMap";
import { ConfusionGrid } from "@/components/charts/ConfusionGrid";
import { RateByBin } from "@/components/charts/RateByBin";
import { ReliabilityPlot } from "@/components/charts/ReliabilityPlot";
import { MIN_OPACITY } from "@/components/sky/scale";
import type { Cls, Cmd, Est, Num } from "@/lib/schema";

const num = (value: number | null, text: string): Num => ({ value, text });
const est = (point: number | null, text: string, lo: number | null = null, hi: number | null = null): Est => ({
  point,
  lo,
  hi,
  text,
  ci_text: point === null ? "n/a" : `IC ${text}`,
});
const count = (html: string, needle: string) => html.split(needle).length - 1;

const row = (label: string, rate: number | null, below = false) => ({
  label,
  lo: null,
  hi: null,
  n: num(below ? 7 : 500, below ? "7" : "500"),
  errors: below ? num(null, "n/a") : num(50, "50"),
  rate: rate === null ? est(null, "n/a") : est(rate, `T${rate}`, rate / 2, rate * 2),
  below_floor: below,
});

const SERIES = [
  { key: "STAR", label: "Estrelas", rows: [row("a", 0.1), row("b", null, true), row("c", 0.2)] },
  { key: "QSO", label: "Quasares", rows: [row("a", 0.3), row("b", 0.4), row("c", null, true)] },
];

const render = (showCi: boolean) =>
  renderToStaticMarkup(
    <RateByBin id="r" title="Taxa" series={SERIES} showCi={showCi} yLabel="Erro" naText="sem dado" />,
  );

describe("RateByBin", () => {
  it("series com faixas diferentes lancam erro em vez de desenhar errado", () => {
    const mixed = [SERIES[0], { ...SERIES[1], rows: [row("x", 0.3), row("y", 0.4), row("z", 0.1)] }];
    expect(() =>
      renderToStaticMarkup(<RateByBin id="r" title="Taxa" series={mixed} showCi={false} yLabel="Erro" naText="sem dado" />),
    ).toThrow("RateByBin: todas as series devem ter as mesmas faixas");
  });

  it("um marcador cheio por linha reportavel; abaixo do piso e vazado com o texto do JSON", () => {
    const html = render(false);
    expect(count(html, "data-point")).toBe(4);
    expect(count(html, "data-hollow")).toBe(2);
    expect(count(html, "data-below-floor")).toBe(2);
    expect(html).toContain("n/a");
    expect(html).not.toContain("T0.1x");
  });

  it("sem whisker quando showCi e falso; com showCi, um por linha com lo e hi", () => {
    expect(count(render(false), "data-ci")).toBe(0);
    expect(count(render(true), "data-ci")).toBe(4);
  });

  it("whisker ausente quando lo ou hi e nulo", () => {
    const r = { ...row("a", 0.1), rate: est(0.1, "T0.1", null, null) };
    const html = renderToStaticMarkup(
      <RateByBin id="r" title="Taxa" series={[{ key: "STAR", label: "Estrelas", rows: [r] }]} showCi yLabel="Erro" naText="x" />,
    );
    expect(count(html, "data-ci")).toBe(0);
  });

  it("eixo x ordinal com o texto das linhas e rotulo direto da serie", () => {
    const html = render(false);
    expect(count(html, "data-bin-label")).toBe(3);
    expect(html).toContain("Estrelas");
    expect(html).toContain("Quasares");
  });
});

const CLASSES: { cls: Cls; label: string }[] = [
  { cls: "STAR", label: "Estrela" },
  { cls: "GALAXY", label: "Galaxia" },
  { cls: "QSO", label: "Quasar" },
];

function cells(trueCls: Cls, belowPred: Cls | null) {
  return CLASSES.map(({ cls }) => {
    const below = cls === belowPred;
    return {
      pred: cls,
      n: below ? num(null, "n/a") : num(300, "300"),
      share: below ? est(null, "n/a") : est(0.5, "S50"),
      diagonal: cls === trueCls,
      below_floor: below,
    };
  });
}

const MATRIX = CLASSES.map(({ cls }) => ({
  true: cls,
  n: num(900, "900"),
  cells: cells(cls, cls === "QSO" ? "STAR" : null),
}));

describe("ConfusionGrid", () => {
  const html = renderToStaticMarkup(
    <ConfusionGrid caption="Matriz" classes={CLASSES} rows={MATRIX} showCi={false} trueLabel="Verdadeira" predLabel="Prevista" />,
  );

  it("tabela real com th scope e 9 celulas", () => {
    expect(html).toContain("<table");
    expect(html).toContain('scope="row"');
    expect(html).toContain('scope="col"');
    expect(count(html, "data-cell=")).toBe(9);
  });

  it("imprime share.text e n.text; diagonal marcada", () => {
    expect(count(html, "S50")).toBe(8);
    expect(count(html, ">300<")).toBe(8);
    expect(count(html, "data-diagonal")).toBe(3);
  });

  it("celula abaixo do piso mostra n/a e nao tem sombreamento", () => {
    expect(count(html, "data-shade")).toBe(8);
    const below = html.split('data-cell="QSO-STAR"')[1].split("</td>")[0];
    expect(below).toContain("n/a");
    expect(below).not.toContain("data-shade");
  });
});

describe("ReliabilityPlot opcional", () => {
  const points = [
    { p_mean: 0.2, freq: 0.25, n: 100, freq_lo: 0.2, freq_hi: 0.3, raw_p_mean: 0.15, raw_freq: 0.25 },
    { p_mean: null, freq: null, n: 10, freq_lo: null, freq_hi: null, raw_p_mean: null, raw_freq: null },
    { p_mean: 0.8, freq: 0.7, n: 100, freq_lo: 0.6, freq_hi: 0.8, raw_p_mean: 0.9, raw_freq: 0.7 },
  ];
  const base = {
    id: "p",
    title: "Confiabilidade",
    series: [{ cls: "STAR" as const, label: "Estrelas", points }],
    xLabel: "x",
    yLabel: "y",
    diagonalLabel: "d",
  };

  it("pontos nulos sao ignorados", () => {
    expect(count(renderToStaticMarkup(<ReliabilityPlot {...base} />), "data-cal-point=")).toBe(2);
  });

  it("padrao nao desenha intervalo nem bruto", () => {
    const html = renderToStaticMarkup(<ReliabilityPlot {...base} />);
    expect(html).not.toContain("data-cal-ci");
    expect(html).not.toContain("data-cal-raw");
  });

  it("intervals e raw desenham um por ponto valido", () => {
    const html = renderToStaticMarkup(<ReliabilityPlot {...base} intervals raw rawLabel="bruto" intervalLabel="IC" />);
    expect(count(html, "data-cal-ci")).toBe(2);
    expect(count(html, "data-cal-raw=")).toBe(2);
    expect(html).toContain("bruto");
  });
});

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
