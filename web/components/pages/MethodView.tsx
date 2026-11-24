"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import styles from "@/app/metodo/page.module.css";
import type { Meta, Method } from "@/lib/schema";
import { readState } from "@/lib/ui-state";

export type TechnicalMethod = Pick<Method, "reading" | "usage" | "rule_lines" | "three_limits">;

type Props = { meta: Meta; technical: TechnicalMethod };

// Rotulos sem algarismos de proposito: o numero vem sempre do JSON.
const text = {
  reading: "Como ler os resultados",
  usage: "Uso do painel",
  rules: "Regras de decisão",
  limits: "Os três limites",
};

function Block({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <section className={styles.panel} aria-label={title}>
      <h2 className={styles.panelTitle}>{title}</h2>
      <ul className={styles.list}>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}

function Technical({ technical }: { technical: TechnicalMethod }) {
  return (
    <>
      <Block title={text.reading} items={technical.reading} />
      <Block title={text.usage} items={technical.usage} />
      <Block title={text.rules} items={technical.rule_lines} />
      <Block title={text.limits} items={technical.three_limits} />
    </>
  );
}

function WithState({ meta, technical }: Props) {
  const params = useSearchParams();
  return readState(params, meta).tech ? <Technical technical={technical} /> : null;
}

export function MethodView({ meta, technical }: Props) {
  return (
    <Suspense fallback={meta.default_state.tech ? <Technical technical={technical} /> : null}>
      <WithState meta={meta} technical={technical} />
    </Suspense>
  );
}
