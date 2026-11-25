import { ptPct } from "@/lib/fmt";
import styles from "./charts.module.css";

export type PurityRangeRow = {
  key: string;
  label: string;
  low: number;
  median: number;
  high: number;
  // Texto pronto do JSON (mediana, minimo e maximo); alimenta o resumo acessivel.
  detail: string;
  marker: "circle" | "square" | "diamond";
};

type PurityRangeProps = {
  id: string;
  title: string;
  rows: PurityRangeRow[];
};

const VIEW_W = 640;
const PLOT_X = 200;
const PLOT_W = 400;
const ROW_H = 44;
const CAP = 6;
const MARK = 7;
const AXIS_H = 32;
const PAD_FRACTION = 0.08;
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];
const FALLBACK_DOMAIN: [number, number] = [0, 1];

function domainOf(rows: PurityRangeRow[]): [number, number] {
  if (rows.length === 0) return FALLBACK_DOMAIN;
  const lo = Math.min(...rows.map((r) => r.low));
  const hi = Math.max(...rows.map((r) => r.high));
  const span = hi - lo > 0 ? hi - lo : 1;
  return [lo - span * PAD_FRACTION, hi + span * PAD_FRACTION];
}

function Mark({ kind, x, cy }: { kind: PurityRangeRow["marker"]; x: number; cy: number }) {
  if (kind === "circle") return <circle className={styles.diamond} data-point="" cx={x} cy={cy} r={MARK} />;
  if (kind === "square") {
    return <rect className={styles.diamond} data-point="" x={x - MARK} y={cy - MARK} width={MARK * 2} height={MARK * 2} />;
  }
  return (
    <polygon
      className={styles.diamond}
      data-point=""
      points={`${x},${cy - MARK} ${x + MARK},${cy} ${x},${cy + MARK} ${x - MARK},${cy}`}
    />
  );
}

export function PurityRange({ id, title, rows }: PurityRangeProps) {
  const plotH = rows.length * ROW_H;
  const viewH = plotH + AXIS_H;
  const [dMin, dMax] = domainOf(rows);
  const x = (v: number): number => PLOT_X + ((v - dMin) / (dMax - dMin)) * PLOT_W;
  const summary = rows.map((r) => `${r.label}: ${r.detail}`).join("; ");
  const titleId = `${id}-title`;

  return (
    <figure className={styles.figure} id={id} aria-labelledby={titleId}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <svg className={styles.svg} viewBox={`0 0 ${VIEW_W} ${viewH}`} role="img" aria-label={`${title}. ${summary}`}>
        {TICK_FRACTIONS.map((f) => {
          const v = dMin + f * (dMax - dMin);
          return (
            <g key={f}>
              <line className={styles.grid} x1={x(v)} x2={x(v)} y1={0} y2={plotH} />
              <text className={styles.tick} x={x(v)} y={plotH + 22} textAnchor="middle">
                {ptPct(v, 0)}
              </text>
            </g>
          );
        })}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={plotH} y2={plotH} />
        {rows.map((r, i) => {
          const cy = i * ROW_H + ROW_H / 2;
          return (
            <g key={r.key} data-row={r.key}>
              <title>{`${r.label}. ${r.detail}`}</title>
              <text className={styles.label} x={0} y={cy} dominantBaseline="central">
                {r.label}
              </text>
              <g className={styles.ciDashed} data-range="">
                <line x1={x(r.low)} x2={x(r.high)} y1={cy} y2={cy} />
                <line x1={x(r.low)} x2={x(r.low)} y1={cy - CAP} y2={cy + CAP} />
                <line x1={x(r.high)} x2={x(r.high)} y1={cy - CAP} y2={cy + CAP} />
              </g>
              <Mark kind={r.marker} x={x(r.median)} cy={cy} />
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
