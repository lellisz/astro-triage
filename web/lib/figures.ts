// Leitor das figuras matplotlib (astro_triage.web_export.figures) em tempo de build, no mesmo
// padrao de lib/data.ts: le <ASTRO_WEB_DATA>/<release>/figures/<id>.svg e o figures.json da
// mesma pasta; qualquer problema lanca erro de uma linha e o build falha.
import fs from "node:fs";
import path from "node:path";
import { release } from "./data";

export type Figure = { id: string; title: string; caption: string; aria_label: string; svg: string };
export type Figures = Readonly<Record<string, Figure>>;

const FIG_ID = /^[a-z0-9][a-z0-9-]*$/;
// O SVG vai para o DOM sem sanitizador: qualquer vetor de execucao reprova o build.
const FORBIDDEN: readonly [RegExp, string][] = [
  [/<script/i, "<script"],
  [/on[a-z]+\s*=/i, "atributo on*="],
  [/javascript:/i, "javascript:"],
  [/<foreignObject/i, "<foreignObject"],
  [/href\s*=\s*(?!["']#)/i, "href que nao aponta para #id local"],
  [/&(?!(?:lt|gt|amp|quot|apos|#39);)/, "referencia de caractere fora de &lt; &gt; &amp; &quot; &#39; &apos;"],
];
// Lista de permissao: so os elementos que o matplotlib gera hoje nas figuras do site.
const ALLOWED = new Set(["svg", "g", "path", "text", "defs", "desc", "pattern", "rect", "title"]);
const TAG = /<\/?([A-Za-z][\w:.-]*)?/g;

type Entry = { title: string; caption: string; aria_label: string };

function figuresDir(rel: string): string {
  const root = process.env.ASTRO_WEB_DATA;
  return path.join(root ? root : path.join(process.cwd(), "data"), rel, "figures");
}

function readText(file: string): string {
  try {
    return fs.readFileSync(file, "utf8");
  } catch {
    throw new Error(`arquivo de figura ausente ou ilegivel: ${file}`);
  }
}

function entryOf(dir: string, rel: string, id: string): Entry {
  const file = path.join(dir, "figures.json");
  let parsed: unknown;
  try {
    parsed = JSON.parse(readText(file));
  } catch (err) {
    if (err instanceof SyntaxError) throw new Error(`JSON invalido em ${file}`);
    throw err;
  }
  const head = parsed as { release?: unknown; provisional?: unknown; figures?: Record<string, unknown> };
  if (head.release !== rel) {
    throw new Error(`${file}: release ${String(head.release)} difere de ASTRO_RELEASE ${rel}`);
  }
  if (head.provisional !== true) {
    throw new Error(`${file}: provisional deve ser true (toda metrica e provisoria)`);
  }
  const entry = head.figures?.[id] as Partial<Entry> | undefined;
  if (
    !entry ||
    typeof entry.title !== "string" ||
    typeof entry.caption !== "string" ||
    typeof entry.aria_label !== "string"
  ) {
    throw new Error(`${file}: figura ${id} sem title, caption ou aria_label`);
  }
  return { title: entry.title, caption: entry.caption, aria_label: entry.aria_label };
}

export function loadFigure(id: string): Figure {
  if (!FIG_ID.test(id)) throw new Error(`id de figura invalido: ${id}`);
  const rel = release();
  const dir = figuresDir(rel);
  const file = path.join(dir, `${id}.svg`);
  const svg = readText(file);
  if (!svg.startsWith("<svg")) throw new Error(`${file}: esperado um SVG que comeca com <svg`);
  for (const [re, what] of FORBIDDEN) {
    if (re.test(svg)) throw new Error(`${file}: SVG recusado, contem ${what}`);
  }
  for (const [, name] of svg.matchAll(TAG)) {
    // Sem nome: <!DOCTYPE, <!ENTITY, <![CDATA[, <!-- ou <? (instrucao de processamento).
    if (!name) throw new Error(`${file}: SVG recusado, contem marcacao <! ou <?`);
    if (!ALLOWED.has(name)) throw new Error(`${file}: SVG recusado, elemento <${name}> fora da lista de permissao`);
  }
  return { id, ...entryOf(dir, rel, id), svg };
}

export function loadFigures(ids: readonly string[]): Figures {
  return Object.fromEntries(ids.map((id) => [id, loadFigure(id)]));
}
