"use client";

import { useId, useRef, useState, useSyncExternalStore, type KeyboardEvent, type ReactNode } from "react";
import styles from "./kit.module.css";

export type TabItem = { key: string; label: string; content: ReactNode };

type TabsProps = { label: string; tabs: readonly TabItem[] };

const noopSubscribe = (): (() => void) => () => {};

// Sem JS (ou antes da hidratacao) todos os paineis ficam visiveis e a lista de abas
// fica oculta: o conteudo nunca depende da troca de aba.
export function Tabs({ label, tabs }: TabsProps) {
  const base = useId();
  const [active, setActive] = useState(0);
  const enhanced = useSyncExternalStore(noopSubscribe, () => true, () => false);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const go = (index: number): void => {
    const next = (index + tabs.length) % tabs.length;
    setActive(next);
    refs.current[next]?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number): void => {
    if (event.key === "ArrowRight") go(index + 1);
    else if (event.key === "ArrowLeft") go(index - 1);
    else if (event.key === "Home") go(0);
    else if (event.key === "End") go(tabs.length - 1);
    else return;
    event.preventDefault();
  };

  return (
    <div className={styles.tabs}>
      <div role="tablist" aria-label={label} className={styles.tabList} hidden={!enhanced}>
        {tabs.map((tab, i) => (
          <button
            key={tab.key}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={`${base}-tab-${tab.key}`}
            aria-selected={i === active}
            aria-controls={`${base}-panel-${tab.key}`}
            tabIndex={i === active ? 0 : -1}
            className={i === active ? styles.tabActive : styles.tab}
            onClick={() => setActive(i)}
            onKeyDown={(e) => onKeyDown(e, i)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab, i) => (
        <div
          key={tab.key}
          role="tabpanel"
          id={`${base}-panel-${tab.key}`}
          aria-labelledby={`${base}-tab-${tab.key}`}
          tabIndex={0}
          hidden={enhanced && i !== active}
          className={styles.tabPanel}
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}
