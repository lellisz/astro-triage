// REV-02: abaixo das figuras quasar-pureza/quasar-completude, a Visao geral nomeia a linha
// escolhida (data-selected-row) com o mesmo rotulo do SVG; sem estado correspondente, nada.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { OverviewView } from "@/components/pages/OverviewView";
import { chartLabels, overviewLabels } from "@/content/labels";
import type { Meta, Overview } from "@/lib/schema";
import { FIGS } from "./fixture-figures";

const nav = vi.hoisted(() => ({ search: "" }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => "/",
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

const FIX = fileURLToPath(new URL("./fixtures", import.meta.url));
const load = <T,>(name: string): T =>
  JSON.parse(readFileSync(path.join(FIX, "fixture_v0", `${name}.json`), "utf-8")) as T;
const meta = load<Meta>("meta");
const overview = load<Overview>("overview");

const unescape = (s: string) =>
  s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&amp;/g, "&");

function render(search: string, m: Meta = meta): string {
  nav.search = search;
  return renderToStaticMarkup(
    <OverviewView
      figures={FIGS}
      meta={m}
      rows={overview.rows}
      accuracyNote={overview.accuracy_note}
      technical={overview.technical}
    />,
  );
}

const selectedParas = (html: string): string[] =>
  [...html.matchAll(/<p\b[^>]*data-selected-row=""[^>]*>([\s\S]*?)<\/p>/g)].map((m) =>
    unescape(m[1].replace(/<[^>]+>/g, "")),
  );

const svgTexts = (id: string): string[] =>
  [...FIGS[id].svg.matchAll(/<text\b[^>]*>([^<]*)<\/text>/g)].map((m) => unescape(m[1]));

// O estado da URL so vale no modo tecnico (o simples sempre usa o padrao).
const q = (s: { rule: string; prior: string }) => `regra=${s.rule}&prior=${s.prior}&tecnico=1`;

describe("REV-02 linha escolhida abaixo das figuras de quasar", () => {
  it("a fixture tem estados com rotulos distintos (pre-condicao do teste)", () => {
    expect(meta.states.length).toBeGreaterThanOrEqual(2);
    expect(new Set(meta.states.map((s) => s.label)).size).toBe(meta.states.length);
  });

  it.each(meta.states.map((s, i) => [i, s.label] as const))(
    "estado %i da URL: mostra exatamente uma vez o rotulo %s",
    (i) => {
      const s = meta.states[i];
      const paras = selectedParas(render(q(s)));
      expect(paras).toEqual([`${overviewLabels.selectedRow}: ${s.label}`]);
    },
  );

  it("o rotulo muda quando o estado da URL muda", () => {
    const [a, b] = [meta.states[1], meta.states[2]];
    const pa = selectedParas(render(q(a)));
    const pb = selectedParas(render(q(b)));
    expect(pa).not.toEqual(pb);
    expect(pa[0]).toContain(a.label);
    expect(pb[0]).toContain(b.label);
    expect(pb[0]).not.toContain(a.label);
  });

  it("URL sem estado usa o default_state", () => {
    const def = meta.states.find(
      (s) => s.rule === meta.default_state.rule && s.prior === meta.default_state.prior,
    )!;
    expect(selectedParas(render(""))).toEqual([`${overviewLabels.selectedRow}: ${def.label}`]);
  });

  it.each(["quasar-pureza", "quasar-completude"])(
    "o rotulo mostrado e o mesmo texto da linha no SVG %s",
    (fid) => {
      const texts = svgTexts(fid);
      for (const s of meta.states) {
        const shown = selectedParas(render(q(s)))[0].slice(`${overviewLabels.selectedRow}: `.length);
        expect(shown).toBe(s.label);
        // a linha parcial leva o sufixo "(parcial)" no SVG; o rotulo base e o mesmo
        expect(
          texts.some((t) => t === shown || t === `${shown} (${chartLabels.partialTag})`),
        ).toBe(true);
      }
    },
  );

  it("sem estado correspondente em meta.states, nada aparece", () => {
    const d = meta.default_state;
    const noDefault: Meta = {
      ...meta,
      states: meta.states.filter((s) => !(s.rule === d.rule && s.prior === d.prior)),
    };
    expect(noDefault.states.length).toBe(meta.states.length - 1);
    const html = render("", noDefault);
    expect(html).not.toContain("data-selected-row");
    expect(html).not.toContain(`${overviewLabels.selectedRow}:`);
    // contraprova: com o meta completo o paragrafo existe
    expect(render("")).toContain("data-selected-row");
  });
});
