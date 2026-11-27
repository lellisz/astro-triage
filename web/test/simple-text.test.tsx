// Texto visivel do modo simples: sem jargao nas 4 paginas, aviso provisorio presente,
// e o modo tecnico (?tecnico=1) congelado por snapshot (estado de 09bf63f).
// Renderiza as paginas reais (app/**/page.tsx, com PageHeader e selo) dentro do Shell,
// lendo a fixture fixture_v0 pelo mesmo leitor do build (lib/data, lib/figures).
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import ComparePage from "@/app/comparacao-dl/page";
import GalaxiesPage from "@/app/galaxias/page";
import GraficosPage from "@/app/graficos/page";
import OverviewPage from "@/app/page";
import { Shell } from "@/components/shell/Shell";
import { footer } from "@/content/labels";
import { loadPage } from "@/lib/data";

const nav = vi.hoisted(() => ({ search: "", path: "/" }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => nav.path,
  useRouter: () => ({ push: () => {}, replace: () => {}, prefetch: () => {} }),
}));

const FIX = fileURLToPath(new URL("./fixtures", import.meta.url));
beforeAll(() => {
  vi.stubEnv("ASTRO_RELEASE", "fixture_v0");
  vi.stubEnv("ASTRO_WEB_DATA", FIX);
});
afterAll(() => vi.unstubAllEnvs());

const PAGES = {
  "/": OverviewPage,
  "/graficos": GraficosPage,
  "/galaxias": GalaxiesPage,
  "/comparacao-dl": ComparePage,
} as const;
type Route = keyof typeof PAGES;
const ROUTES = Object.keys(PAGES) as Route[];

const decode = (s: string) =>
  s
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&nbsp;|&#xa0;/gi, " ")
    .replace(/&amp;/g, "&");

// Texto visivel: tira figuras matplotlib (<figure data-figure>), svg, script e style,
// depois remove todas as tags (atributos somem junto) e decodifica entidades.
const visibleText = (html: string) =>
  decode(
    html
      .replace(/<figure\b[^>]*\bdata-figure=[\s\S]*?<\/figure>/g, " ")
      .replace(/<(svg|script|style)\b[\s\S]*?<\/\1>/g, " ")
      .replace(/<[^>]*>/g, " "),
  )
    .replace(/\s+/g, " ")
    .trim();

const renderHtml = (route: Route, search: string, withShell = true) => {
  nav.search = search;
  nav.path = route === "/" ? "/" : `${route}/`;
  try {
    const Page = PAGES[route];
    const body = <Page />;
    return renderToStaticMarkup(withShell ? <Shell meta={loadPage("meta")}>{body}</Shell> : body);
  } finally {
    nav.search = "";
    nav.path = "/";
  }
};

// Termos de jargao (sem caixa). Fronteiras unicode-aware para IC e prior.
const JARGON: [string, RegExp][] = [
  ["IC", /(?<![\p{L}\p{N}])IC(?![\p{L}\p{N}])/giu],
  ["intervalo", /intervalo/giu],
  ["bootstrap", /bootstrap/giu],
  ["p.p.", /p\.p\./giu],
  ["prior", /(?<![\p{L}\p{N}])prior/giu],
  ["regra", /regra/giu],
  ["F macro", /F macro/giu],
  ["piso", /piso/giu],
  ["calibra", /calibra/giu],
  ["temperatura", /temperatura/giu],
  ["redshift", /redshift/giu],
  ["δ", /δ/giu],
  ["semente", /semente/giu],
];

type Hit = { page: string; term: string; snippet: string };
const scanJargon = (page: string, text: string): Hit[] => {
  const hits: Hit[] = [];
  for (const [term, re] of JARGON) {
    for (const m of text.matchAll(re)) {
      const i = m.index ?? 0;
      const snippet = text.slice(Math.max(0, i - 50), i + m[0].length + 50);
      hits.push({ page, term, snippet: `...${snippet}...` });
    }
  }
  return hits;
};
const fmtHits = (hits: Hit[]) =>
  hits.map((h, n) => `${n + 1}. [${h.page}] "${h.term}": ${h.snippet}`).join("\n");

describe("criterio 1: modo simples sem jargao no texto visivel", () => {
  it("o extrator exclui figuras e atributos e o detector acha jargao plantado", () => {
    const html =
      '<div title="intervalo regra"><p>ok</p><figure data-figure="x"><figcaption>IC 95%</figcaption></figure>' +
      "<p>com bootstrap e IC</p></div>";
    const text = visibleText(html);
    expect(text).toBe("ok com bootstrap e IC");
    expect(scanJargon("t", text).map((h) => h.term).sort()).toEqual(["IC", "bootstrap"]);
    // fronteira: palavras que contem as letras mas nao o termo
    expect(scanJargon("t", "público lógica técnica anterior Pisos")).toEqual([
      { page: "t", term: "piso", snippet: "...público lógica técnica anterior Pisos..." },
    ]);
    // \bprior e prefixo: "prioridade" conta (como pedido no criterio)
    expect(scanJargon("t", "prioridade").map((h) => h.term)).toEqual(["prior"]);
    // sem caixa
    expect(scanJargon("t", "Calibração; PRIOR; Semente; Δ").map((h) => h.term)).toEqual([
      "prior",
      "calibra",
      "δ",
      "semente",
    ]);
  });

  it("shell (cabecalho, controles e rodape) no modo simples", () => {
    nav.search = "";
    const text = visibleText(renderToStaticMarkup(<Shell meta={loadPage("meta")}>{null}</Shell>));
    const hits = scanJargon("shell", text);
    expect(fmtHits(hits), "jargao no shell (pagina, termo, trecho)").toBe("");
  });

  for (const route of ROUTES) {
    for (const search of ["", "tecnico=0"]) {
      it(`${route}${search ? `?${search}` : ""}: conteudo da pagina sem jargao`, () => {
        const text = visibleText(renderHtml(route, search, false));
        expect(text.length).toBeGreaterThan(100);
        const hits = scanJargon(route, text);
        expect(fmtHits(hits), `jargao em ${route} (pagina, termo, trecho)`).toBe("");
      });
    }
  }
});

describe("criterio 2: aviso de resultados provisorios nas 4 paginas (modo simples)", () => {
  // lido sob demanda: o env da fixture so existe depois do beforeAll
  const meta = {
    get seal() {
      return loadPage("meta").seal_text;
    },
  };
  const hasNotice = (text: string, seal: string) =>
    text.includes(footer.note) && /provis[óo]ri/i.test(seal) && text.includes(seal);

  for (const route of ROUTES) {
    it(`${route}: selo do cabecalho e nota do rodape visiveis`, () => {
      const text = visibleText(renderHtml(route, ""));
      expect(meta.seal).toMatch(/provis[óo]ri/i);
      expect(text).toContain(meta.seal);
      expect(text).toContain(footer.note);
      expect(text).toMatch(/Resultados provisórios/);
      expect(hasNotice(text, meta.seal)).toBe(true);
    });
  }

  it("mutacao: texto sem o aviso (selo ou nota removidos) e detectado", () => {
    const text = visibleText(renderHtml("/graficos", ""));
    expect(hasNotice(text.split(footer.note).join(""), meta.seal)).toBe(false);
    expect(hasNotice(text.split(meta.seal).join(""), meta.seal)).toBe(false);
    // selo sem a palavra provisorio tambem e rejeitado
    expect(hasNotice(text, "fixture_v0")).toBe(false);
  });
});

describe("criterio 3: modo tecnico congelado (snapshot do texto visivel, sem figuras)", () => {
  for (const route of ROUTES) {
    it(`${route}?tecnico=1`, () => {
      const text = visibleText(renderHtml(route, "tecnico=1"));
      expect(text.length).toBeGreaterThan(200);
      expect(text).toMatchSnapshot();
    });
  }
});
