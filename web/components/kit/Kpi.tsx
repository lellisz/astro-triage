import styles from "./kit.module.css";

type KpiProps = {
  label: string;
  text: string;
  ci?: string;
  note?: string;
};

export function Kpi({ label, text, ci, note }: KpiProps) {
  return (
    <div className={styles.kpi}>
      <p className={styles.kpiLabel}>{label}</p>
      <p className={styles.kpiValue}>{text}</p>
      {ci ? <p className={styles.kpiCi}>{ci}</p> : null}
      {note ? <p className={styles.kpiNote}>{note}</p> : null}
    </div>
  );
}
