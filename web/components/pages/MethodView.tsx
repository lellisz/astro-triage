"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, type ReactNode } from "react";
import { methodLabels as text } from "@/content/labels";
import type { Meta, Method } from "@/lib/schema";
import { readState } from "@/lib/ui-state";
import styles from "./method.module.css";

export type TechnicalMethod = Pick<Method, "reading" | "usage" | "rule_lines" | "limitations">;

type Props = { meta: Meta; technical: TechnicalMethod };

type DisclosureProps = { id: string; title: string; open: boolean; children: ReactNode };

function Disclosure({ id, title, open, children }: DisclosureProps) {
  return (
    <details className={styles.disclosure} open={open} id={id}>
      <summary className={styles.summary}>{title}</summary>
      <div className={styles.disclosureBody}>{children}</div>
    </details>
  );
}

function Items({ items }: { items: string[] }) {
  return (
    <ul className={styles.list}>
      {items.map((item, i) => (
        <li key={`${i}-${item}`}>{item}</li>
      ))}
    </ul>
  );
}

function Technical({ technical, open }: { technical: TechnicalMethod; open: boolean }) {
  const { reading, usage, rule_lines: rules, limitations } = technical;
  return (
    <section className={styles.panel} aria-labelledby="detalhes">
      <h2 id="detalhes" className={styles.panelTitle}>
        {text.details}
      </h2>
      {reading.length > 0 ? (
        <Disclosure id="leitura" title={text.reading} open={open}>
          <Items items={reading} />
        </Disclosure>
      ) : null}
      {usage.length > 0 ? (
        <Disclosure id="uso" title={text.usage} open={open}>
          <Items items={usage} />
        </Disclosure>
      ) : null}
      {rules.length > 0 ? (
        <Disclosure id="regras" title={text.rules} open={open}>
          <Items items={rules} />
        </Disclosure>
      ) : null}
      {limitations.length > 0 ? (
        <Disclosure id="limitacoes" title={text.limitations} open={open}>
          <ul className={styles.list}>
            {limitations.map((item, i) => (
              <li key={`${i}-${item.text}`}>
                {item.text}
                {item.original !== item.text ? (
                  <details className={styles.nested}>
                    <summary className={styles.summary}>{text.original}</summary>
                    <p className={styles.muted}>{item.original}</p>
                  </details>
                ) : null}
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}
    </section>
  );
}

// A frase do modo simples so vale fora do modo tecnico (como no Streamlit). O
// estado vem da URL, entao o filtro e feito aqui, no cliente.
function SimpleNoticeWithState({ meta, text: notice }: { meta: Meta; text: string }) {
  const params = useSearchParams();
  return readState(params, meta).tech ? null : <p className={styles.notice}>{notice}</p>;
}

export function SimpleNotice({ meta, text: notice }: { meta: Meta; text: string }) {
  return (
    <Suspense
      fallback={meta.default_state.tech ? null : <p className={styles.notice}>{notice}</p>}
    >
      <SimpleNoticeWithState meta={meta} text={notice} />
    </Suspense>
  );
}

function WithState({ meta, technical }: Props) {
  const params = useSearchParams();
  return <Technical technical={technical} open={readState(params, meta).tech} />;
}

export function MethodView({ meta, technical }: Props) {
  return (
    <Suspense fallback={<Technical technical={technical} open={meta.default_state.tech} />}>
      <WithState meta={meta} technical={technical} />
    </Suspense>
  );
}
