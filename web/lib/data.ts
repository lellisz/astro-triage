// Leitor de dados em tempo de build (componentes de servidor). Qualquer
// inconsistencia lanca um erro de uma linha, o que faz o build falhar.
import fs from "node:fs";
import path from "node:path";
import { SCHEMA_VERSION, type PageData, type PageKey } from "./schema";

export function release(): string {
  const value = process.env.ASTRO_RELEASE;
  if (!value) {
    throw new Error("ASTRO_RELEASE nao definido: informe a release a publicar (ex.: full_v0)");
  }
  return value;
}

function dataDir(rel: string): string {
  const root = process.env.ASTRO_WEB_DATA;
  return root ? path.join(root, rel) : path.join(process.cwd(), "data", rel);
}

export function loadPage<K extends PageKey>(page: K): PageData[K] {
  const rel = release();
  const file = path.join(dataDir(rel), `${page}.json`);

  let raw: string;
  try {
    raw = fs.readFileSync(file, "utf8");
  } catch {
    throw new Error(`arquivo de dados ausente ou ilegivel: ${file}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`JSON invalido em ${file}`);
  }

  // fmt_samples.json e uma lista sem cabecalho; a release vem do diretorio.
  if (page === "fmt_samples") {
    if (!Array.isArray(parsed)) throw new Error(`${file}: esperado uma lista`);
    return parsed as PageData[K];
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${file}: esperado um objeto JSON`);
  }
  const head = parsed as Record<string, unknown>;
  if (head.schema_version !== SCHEMA_VERSION) {
    throw new Error(
      `${file}: schema_version ${String(head.schema_version)} difere do esperado ${SCHEMA_VERSION}`,
    );
  }
  if (head.release !== rel) {
    throw new Error(`${file}: release ${String(head.release)} difere de ASTRO_RELEASE ${rel}`);
  }
  if (head.provisional !== true) {
    throw new Error(`${file}: provisional deve ser true (toda metrica e provisoria)`);
  }
  return parsed as PageData[K];
}
