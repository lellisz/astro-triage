import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { Meta } from "../lib/schema";
import { readState, withState } from "../lib/ui-state";

const meta = JSON.parse(
  fs.readFileSync(path.join(__dirname, "fixtures", "fixture_v0", "meta.json"), "utf8"),
) as Meta;
const P = (q: string) => new URLSearchParams(q);
const DEF = { rule: "as_shipped", prior: "P2", tech: false };

describe("readState", () => {
  it("regra desconhecida cai no default_state", () => {
    expect(readState(P("regra=xyz&prior=P1"), meta)).toEqual(DEF);
  });
  it("as_shipped com P1 (par fora de meta.states) cai no default", () => {
    expect(readState(P("regra=as_shipped&prior=P1"), meta)).toEqual(DEF);
  });
  it("params ausentes caem no default", () => {
    expect(readState(P(""), meta)).toEqual(DEF);
  });
  it("par valido e respeitado no modo tecnico", () => {
    expect(readState(P("regra=prior_matched&prior=P3&tecnico=1"), meta)).toEqual({
      rule: "prior_matched",
      prior: "P3",
      tech: true,
    });
  });
  it("modo simples ignora regra e prior da URL e usa o padrao (decisao do owner)", () => {
    const s = readState(P("regra=prior_matched&prior=P3&tecnico=0"), meta);
    expect(s).toEqual({ rule: meta.default_state.rule, prior: meta.default_state.prior, tech: false });
    expect(readState(P("regra=prior_matched&prior=P3"), meta).prior).toBe(meta.default_state.prior);
  });
  it("tecnico=1 liga tech; outros valores nao", () => {
    expect(readState(P("tecnico=1"), meta).tech).toBe(true);
    for (const v of ["0", "true", "2", ""]) {
      expect(readState(P(`tecnico=${v}`), meta).tech).toBe(false);
    }
  });
  it("release na URL e ignorada", () => {
    const a = readState(P("regra=prior_matched&prior=P1&release=outra"), meta);
    const b = readState(P("regra=prior_matched&prior=P1"), meta);
    expect(a).toEqual(b);
    expect(a).not.toHaveProperty("release");
  });
});

describe("withState", () => {
  const s = { rule: "prior_matched", prior: "P1", tech: true } as const;
  it("preserva path e escreve os quatro params", () => {
    const u = new URL(withState("/galaxias", s, "full_v0"), "http://x");
    expect(u.pathname).toBe("/galaxias");
    expect(u.searchParams.get("regra")).toBe("prior_matched");
    expect(u.searchParams.get("prior")).toBe("P1");
    expect(u.searchParams.get("tecnico")).toBe("1");
    expect(u.searchParams.get("release")).toBe("full_v0");
  });
  it("tech false escreve tecnico=0", () => {
    const u = new URL(withState("/a", { ...s, tech: false }, "r"), "http://x");
    expect(u.searchParams.get("tecnico")).toBe("0");
  });
  it("mantem outros params e o hash, e sobrescreve os antigos", () => {
    const out = withState("/a/b?foo=bar&regra=velha&x=1#sec", s, "r1");
    const u = new URL(out, "http://x");
    expect(u.pathname).toBe("/a/b");
    expect(u.searchParams.get("foo")).toBe("bar");
    expect(u.searchParams.get("x")).toBe("1");
    expect(u.searchParams.getAll("regra")).toEqual(["prior_matched"]);
    expect(u.hash).toBe("#sec");
  });
  it("hash sem query e preservado", () => {
    const out = withState("/a#h", s, "r");
    expect(out.startsWith("/a?")).toBe(true);
    expect(out.endsWith("#h")).toBe(true);
  });
  it("round-trip: readState le o que withState escreveu", () => {
    const out = withState("/a", s, "r");
    const q = new URL(out, "http://x").searchParams;
    expect(readState(q, meta)).toEqual(s);
  });
});
