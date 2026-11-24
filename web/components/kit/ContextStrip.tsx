import styles from "./kit.module.css";

export type ContextItem = { label: string; text: string };

export function ContextStrip({ items }: { items: readonly ContextItem[] }) {
  return (
    <ul className={styles.strip}>
      {items.map((it) => (
        <li key={it.label} className={styles.chip}>
          <span className={styles.chipLabel}>{it.label}</span> <span className={styles.chipText}>{it.text}</span>
        </li>
      ))}
    </ul>
  );
}
