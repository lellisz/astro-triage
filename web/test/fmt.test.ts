import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ptCiLabel, ptG, ptInt, ptNum, ptPct } from "@/lib/fmt";

type Sample = { fn: string; args: unknown[]; text: string };

const samples: Sample[] = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("./fixtures/fixture_v0/fmt_samples.json", import.meta.url)),
    "utf-8",
  ),
);

const impls: Record<string, (...args: never[]) => string> = {
  pt_int: ptInt,
  pt_pct: ptPct,
  pt_num: ptNum,
  pt_g: ptG,
  pt_ci_label: ptCiLabel,
};

describe("paridade de formatacao com fmt_samples.json", () => {
  it("existem amostras", () => {
    expect(samples.length).toBeGreaterThan(0);
  });

  it.each(samples.map((s, i) => [i, s.fn, JSON.stringify(s.args), s] as const))(
    "#%i %s(%s)",
    (_i, fn, _args, s) => {
      const impl = impls[fn];
      expect(impl, `fn sem implementacao: ${fn}`).toBeTypeOf("function");
      expect((impl as (...a: unknown[]) => string)(...s.args)).toBe(s.text);
    },
  );
});
