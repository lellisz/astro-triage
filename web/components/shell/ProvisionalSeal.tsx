import styles from "./ProvisionalSeal.module.css";

export function ProvisionalSeal({ text }: { text: string }) {
  return (
    <p className={styles.seal} role="status">
      <span className={styles.mark} aria-hidden="true" />
      {text}
    </p>
  );
}
