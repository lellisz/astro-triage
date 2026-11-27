import type { Meta, Prior, Rule } from "./schema";

export type UiState = { rule: Rule; prior: Prior; tech: boolean };

export function readState(params: URLSearchParams, meta: Meta): UiState {
  const tech = params.get("tecnico") === "1";
  // Decisao do owner (2026-11-27): o modo simples nao tem seletor de regra e prior, entao sempre
  // mostra a condicao padrao; a escolha feita no modo tecnico nao vaza para o simples.
  const found = tech
    ? meta.states.find((s) => s.rule === params.get("regra") && s.prior === params.get("prior"))
    : undefined;
  const { rule, prior } = found ?? meta.default_state;
  return { rule, prior, tech };
}

// release e apenas eco de leitura: readState nunca o le.
export function withState(href: string, s: UiState, release: string): string {
  const hashAt = href.indexOf("#");
  const hash = hashAt === -1 ? "" : href.slice(hashAt);
  const beforeHash = hashAt === -1 ? href : href.slice(0, hashAt);
  const queryAt = beforeHash.indexOf("?");
  const base = queryAt === -1 ? beforeHash : beforeHash.slice(0, queryAt);
  const query = new URLSearchParams(queryAt === -1 ? "" : beforeHash.slice(queryAt + 1));
  query.set("regra", s.rule);
  query.set("prior", s.prior);
  query.set("tecnico", s.tech ? "1" : "0");
  query.set("release", release);
  return `${base}?${query.toString()}${hash}`;
}
