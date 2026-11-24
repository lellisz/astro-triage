import type { Num } from "@/lib/schema";
import styles from "./charts.module.css";

export type HBarRow = { key: string; label: string; value: Num };

type HBarProps = { id: string; title: string; rows: HBarRow[] };

const VIEW_W = 640;
const PLOT_X = 280;
const PLOT_W = 200;
const VALUE_X = 492;
const ROW_H = 32;
const BAR_H = 14;

export function HBar({ id, title, rows }: HBarProps) {
  const max = Math.max(0, ...rows.map((r) => r.value.value ?? 0));
  const viewH = rows.length * ROW_H;
  const w = (v: number | null): number => (max > 0 && v !== null ? (Math.max(v, 0) / max) * PLOT_W : 0);
  const summary = rows.map((r) => `${r.label}: ${r.value.text}`).join("; ");
  const titleId = `${id}-title`;

  return (
    <figure className={styles.figure} id={id} aria-labelledby={titleId}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <svg className={styles.svg} viewBox={`0 0 ${VIEW_W} ${viewH}`} role="img" aria-label={`${title}. ${summary}`}>
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X} y1={0} y2={viewH} />
        {rows.map((r, i) => {
          const cy = i * ROW_H + ROW_H / 2;
          return (
            <g key={r.key} data-feature={r.key}>
              <title>{r.value.text}</title>
              <text className={styles.label} x={0} y={cy} dominantBaseline="central">
                {r.label}
              </text>
              <rect className={styles.bar} x={PLOT_X} y={cy - BAR_H / 2} width={w(r.value.value)} height={BAR_H} />
              <text className={styles.value} x={VALUE_X} y={cy} dominantBaseline="central">
                {r.value.text}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
