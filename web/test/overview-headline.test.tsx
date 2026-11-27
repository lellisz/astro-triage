import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { GalaxiesView } from "@/components/pages/GalaxiesView";
import { OverviewView } from "@/components/pages/OverviewView";
import { overviewLabels as L } from "@/content/labels";
import type { Galaxies, Meta, Overview } from "@/lib/schema";
import { withState } from "@/lib/ui-state";
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
const galaxies = load<Galaxies>("galaxies");

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const render = (headline: Overview["headline"], search = "") => {
  nav.search = search;
  return renderToStaticMarkup(
    <OverviewView figures={FIGS}
      meta={meta}
      headline={headline}
      rows={overview.rows}
      accuracyNote={overview.accuracy_note}
      technical={overview.technical}
    />,
  );
};

const tie: Overview["headline"] = {
  ...overview.headline,
  key: "no_detectable_difference",
  title: "Titulo da chave de empate",
  caveat: "Aviso de falta de poder",
};
const unavailable: Overview["headline"] = {
  ...overview.headline,
  key: "unavailable",
  title: "Comparacao fora do ar",
  lead: "Texto sem a avaliacao",
  caveat: null,
};

describe("Visao geral: bloco da resposta", () => {
  const html = render(tie);
  const pos = (s: string) => html.indexOf(escapeHtml(s));

  it("mostra pergunta, titulo, lead, aviso, pares, contraste e linha em ingles", () => {
    for (const s of [
      tie.question,
      tie.title,
      tie.lead,
      tie.caveat!,
      tie.contrast.accuracy.text,
      tie.contrast.qso_purity.text,
      tie.summary_en,
    ]) {
      expect(pos(s), s).toBeGreaterThan(-1);
    }
    for (const p of tie.pairs) {
      expect(html).toContain(`<strong>${escapeHtml(p.net)}</strong>: ${escapeHtml(p.text)}`);
    }
    expect(html).toContain('lang="en"');
    expect(pos(tie.contrast.text), "contraste com jargao fora do modo simples").toBe(-1);
    expect(render(tie, "tecnico=1")).toContain(escapeHtml(tie.contrast.text));
  });

  it("o bloco vem antes dos KPIs", () => {
    expect(pos(tie.title)).toBeGreaterThan(-1);
    expect(pos(tie.title)).toBeLessThan(html.indexOf(`${L.purity}</p>`));
  });

  it("o link de detalhes tecnicos aponta para o mesmo caminho com tecnico=1", () => {
    const href = withState("/", { ...meta.default_state, tech: true }, meta.release);
    expect(href).toContain("tecnico=1");
    expect(html).toContain(`href="${escapeHtml(href)}"`);
    expect(html).toContain(L.techLink);
  });

  it("sem o link quando o modo tecnico ja esta ativo", () => {
    expect(render(tie, "tecnico=1")).not.toContain(`>${L.techLink}<`);
  });

  it("linha de protocolo em proposta logo abaixo do titulo (Tarefa 28, AUD-01)", () => {
    const proto = overview.headline.protocol_text;
    expect(proto).toContain("em proposta");
    const fixture = render(overview.headline);
    const at = fixture.indexOf(escapeHtml(proto!));
    expect(at).toBeGreaterThan(fixture.indexOf(escapeHtml(overview.headline.title)));
    expect(at).toBeLessThan(fixture.indexOf(escapeHtml(overview.headline.lead)));
    // nulo (protocolo aceito) nao deixa rastro
    expect(render({ ...overview.headline, protocol_text: null })).not.toContain("em proposta");
  });

  it("frase do par sem vitoria da fixture segue a D-1 (o titulo TIE e provado no pytest)", () => {
    const html = render(overview.headline);
    for (const p of overview.headline.pairs) {
      expect(p.text).not.toMatch(/sem diferença detectável/i);
      expect(html).toContain(escapeHtml(p.text));
    }
    expect(overview.headline.pairs.map((p) => p.text)).toContain("sem vitória do par, fica o LightGBM");
    expect(html).not.toMatch(/Sem diferença detectável/);
  });

  it("contraste cita o prior P2 e a mistura do teste (AUD-12)", () => {
    const t = overview.headline.contrast.text;
    expect(t).toContain("(prior P2)");
    expect(t).toContain(overview.rows[0].qso_prior!.text);
    expect(t).toContain("mistura do teste");
  });

  it("sem aviso quando caveat e nulo", () => {
    const wins = render({ ...tie, key: "network_wins", caveat: null });
    expect(wins).not.toContain(escapeHtml(tie.caveat!));
  });
});

describe("Visao geral: comparacao indisponivel", () => {
  const html = render(unavailable);

  it("mostra so titulo e texto, sem pares nem contraste", () => {
    expect(html).toContain(unavailable.title);
    expect(html).toContain(unavailable.lead);
    for (const p of unavailable.pairs) expect(html).not.toContain(escapeHtml(p.text));
    expect(html).not.toContain(escapeHtml(unavailable.contrast.text));
    // o bloco de contraste inteiro some (a palavra "Acuracia" continua no KPI)
    expect(html).not.toContain(`aria-label="${escapeHtml(L.contrastRegion)}"`);
    expect(render(tie)).toContain(`role="group" aria-label="${escapeHtml(L.contrastRegion)}"`);
  });

  it("os KPIs e os atalhos continuam", () => {
    expect(html).toContain(L.shortcutsTitle);
    expect(html).toContain(`${L.accuracy}</p>`);
  });
});

describe("Galaxias: nota da tabela de faixas", () => {
  it("renderiza error_rows_note", () => {
    nav.search = "";
    const html = renderToStaticMarkup(<GalaxiesView figures={FIGS} meta={meta} galaxies={galaxies} />);
    expect(galaxies.error_rows_note.length).toBeGreaterThan(0);
    expect(html).toContain(escapeHtml(galaxies.error_rows_note));
  });
});
