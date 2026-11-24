import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadPage } from "../lib/data";
import type { PageKey } from "../lib/schema";

const FIXTURES = path.join(__dirname, "fixtures");
const PAGES: PageKey[] = [
  "compare",
  "fmt_samples",
  "galaxies",
  "meta",
  "method",
  "overview",
  "sky",
];
const saved = { d: process.env.ASTRO_WEB_DATA, r: process.env.ASTRO_RELEASE };
const tmps: string[] = [];

function restore(k: "ASTRO_WEB_DATA" | "ASTRO_RELEASE", v: string | undefined) {
  if (v === undefined) delete process.env[k];
  else process.env[k] = v;
}

beforeEach(() => {
  process.env.ASTRO_WEB_DATA = FIXTURES;
  process.env.ASTRO_RELEASE = "fixture_v0";
});
afterEach(() => {
  restore("ASTRO_WEB_DATA", saved.d);
  restore("ASTRO_RELEASE", saved.r);
  while (tmps.length) fs.rmSync(tmps.pop()!, { recursive: true, force: true });
});

// Copia fixture_v0 para o tmpdir do sistema e aplica a mutacao em meta.json.
function mutatedCopy(mutate: (o: Record<string, unknown>) => void): void {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "astro-web-test-"));
  tmps.push(root);
  const dst = path.join(root, "fixture_v0");
  fs.cpSync(path.join(FIXTURES, "fixture_v0"), dst, { recursive: true });
  const f = path.join(dst, "meta.json");
  const o = JSON.parse(fs.readFileSync(f, "utf8"));
  mutate(o);
  fs.writeFileSync(f, JSON.stringify(o));
  process.env.ASTRO_WEB_DATA = root;
}

function oneLineError(fn: () => unknown, re: RegExp): void {
  let err: unknown;
  try {
    fn();
  } catch (e) {
    err = e;
  }
  expect(err).toBeInstanceOf(Error);
  const msg = (err as Error).message;
  expect(msg).not.toContain("\n");
  expect(msg).toMatch(re);
}

describe("loadPage", () => {
  it.each(PAGES)("carrega a pagina %s", (p) => {
    expect(loadPage(p)).toBeTruthy();
  });

  it("meta carregada tem release e provisional corretos", () => {
    const m = loadPage("meta");
    expect(m.release).toBe("fixture_v0");
    expect(m.provisional).toBe(true);
  });

  it("ASTRO_RELEASE ausente lanca erro de uma linha", () => {
    delete process.env.ASTRO_RELEASE;
    oneLineError(() => loadPage("meta"), /ASTRO_RELEASE/);
  });

  it("arquivo ausente lanca erro de uma linha", () => {
    process.env.ASTRO_RELEASE = "nao_existe";
    oneLineError(() => loadPage("meta"), /ausente|ilegivel/);
  });

  it("schema_version != 1 lanca erro de uma linha", () => {
    mutatedCopy((o) => {
      o.schema_version = 2;
    });
    oneLineError(() => loadPage("meta"), /schema_version/);
  });

  it("release do arquivo != ASTRO_RELEASE lanca erro de uma linha", () => {
    mutatedCopy((o) => {
      o.release = "outra";
    });
    oneLineError(() => loadPage("meta"), /release/);
  });

  it.each([false, "true", undefined])("provisional=%s lanca erro de uma linha", (v) => {
    mutatedCopy((o) => {
      if (v === undefined) delete o.provisional;
      else o.provisional = v;
    });
    oneLineError(() => loadPage("meta"), /provisional/);
  });
});
