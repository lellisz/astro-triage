// Todas as figuras da fixture, lidas direto do disco (sem lib/data, que alguns testes mocam).
// A validacao do SVG e do figures.json fica coberta em mpl-figure.test.tsx.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Figure, Figures } from "@/lib/figures";

const DIR = fileURLToPath(new URL("./fixtures/fixture_v0/figures", import.meta.url));

type Entry = Omit<Figure, "id" | "svg">;
const manifest = JSON.parse(readFileSync(path.join(DIR, "figures.json"), "utf-8")) as {
  figures: Record<string, Entry>;
};

export const FIGS: Figures = Object.fromEntries(
  Object.entries(manifest.figures).map(([id, e]) => [
    id,
    { id, ...e, svg: readFileSync(path.join(DIR, `${id}.svg`), "utf-8") },
  ]),
);
