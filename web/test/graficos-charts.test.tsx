import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ConfusionGrid } from "@/components/charts/ConfusionGrid";
import { RateByBin } from "@/components/charts/RateByBin";
import { ReliabilityPlot } from "@/components/charts/ReliabilityPlot";
import type { Cls, Est, Num } from "@/lib/schema";

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
