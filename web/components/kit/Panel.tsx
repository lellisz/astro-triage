import type { ReactNode } from "react";
import styles from "./kit.module.css";

type PanelProps = {
  id: string;
  title: string;
  subtitle?: string;
  eyebrow?: string;
  children?: ReactNode;
};

export function Panel({ id, title, subtitle, eyebrow, children }: PanelProps) {
  return (
    <section className={styles.panel} aria-labelledby={id}>
      {eyebrow ? <p className={styles.eyebrow}>{eyebrow}</p> : null}
      <h2 id={id} className={styles.panelTitle}>
        {title}
      </h2>
      {subtitle ? <p className={styles.panelSub}>{subtitle}</p> : null}
      {children}
    </section>
  );
}
