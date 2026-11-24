import { ptInt, ptPct } from "@/lib/fmt";
import styles from "./charts.module.css";

type Bin = { lo: number; hi: number; n_right: number; n_wrong: number };

type HistogramPairProps = {
  id: string;
  title: string;
  bins: Bin[];
  rightLabel: string;
  wrongLabel: string;
  xLabel: string;
  yLabel: string;
};

const VIEW_W = 640;
const PLOT_X = 80;
const PLOT_Y = 8;
const PLOT_W = 520;
const PLOT_H = 220;
const VIEW_H = PLOT_Y + PLOT_H + 76;
const SWATCH = 14;
const Y_TICKS = [0, 0.5, 1];

export function HistogramPair({ id, title, bins, rightLabel, wrongLabel, xLabel, yLabel }: HistogramPairProps) {
  const titleId = `${id}-title`;
  const yMax = Math.max(1, ...bins.flatMap((b) => [b.n_right, b.n_wrong]));
  const slot = bins.length > 0 ? PLOT_W / bins.length : PLOT_W;
  const barW = (slot * 0.8) / 2;
  const y = (n: number): number => PLOT_Y + PLOT_H - (n / yMax) * PLOT_H;
  const range = (b: Bin): string => `${ptPct(b.lo, 0)}–${ptPct(b.hi, 0)}`;
  const summary = bins
    .map((b) => `${range(b)}: ${rightLabel} ${ptInt(b.n_right)}, ${wrongLabel} ${ptInt(b.n_wrong)}`)
    .join("; ");
  const legendY = PLOT_Y + PLOT_H + 56;

  return (
    <figure className={styles.figure} id={id} aria-labelledby={titleId}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label={`${title}. ${xLabel}; ${yLabel}. ${summary}`}
      >
        {Y_TICKS.map((t) => (
          <g key={t}>
            <line className={styles.grid} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={y(t * yMax)} y2={y(t * yMax)} />
            <text className={styles.tick} x={PLOT_X - 8} y={y(t * yMax)} textAnchor="end" dominantBaseline="central">
              {ptInt(t * yMax)}
            </text>
          </g>
        ))}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={PLOT_Y + PLOT_H} y2={PLOT_Y + PLOT_H} />
        <text
          className={styles.tick}
          transform={`translate(14 ${PLOT_Y + PLOT_H / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yLabel}
        </text>
        {bins.map((b, i) => {
          const x0 = PLOT_X + i * slot + slot * 0.1;
          return (
            <g key={`${b.lo}-${b.hi}`} data-bin={`${b.lo}-${b.hi}`}>
              <title>{`${range(b)}: ${rightLabel} ${ptInt(b.n_right)}, ${wrongLabel} ${ptInt(b.n_wrong)}`}</title>
              <rect
                className={styles.barRight}
                x={x0}
                y={y(b.n_right)}
                width={barW}
                height={PLOT_Y + PLOT_H - y(b.n_right)}
              />
              <rect
                className={styles.barWrong}
                x={x0 + barW}
                y={y(b.n_wrong)}
                width={barW}
                height={PLOT_Y + PLOT_H - y(b.n_wrong)}
              />
              <text className={styles.tick} x={PLOT_X + i * slot} y={PLOT_Y + PLOT_H + 18} textAnchor="middle">
                {ptPct(b.lo, 0)}
              </text>
              {i === bins.length - 1 ? (
                <text className={styles.tick} x={PLOT_X + (i + 1) * slot} y={PLOT_Y + PLOT_H + 18} textAnchor="middle">
                  {ptPct(b.hi, 0)}
                </text>
              ) : null}
            </g>
          );
        })}
        <text className={styles.tick} x={PLOT_X + PLOT_W / 2} y={PLOT_Y + PLOT_H + 38} textAnchor="middle">
          {xLabel}
        </text>
        <rect className={styles.barRight} x={PLOT_X} y={legendY - SWATCH / 2} width={SWATCH} height={SWATCH} />
        <text className={styles.label} x={PLOT_X + SWATCH + 8} y={legendY} dominantBaseline="central">
          {rightLabel}
        </text>
        <rect className={styles.barWrong} x={PLOT_X + 200} y={legendY - SWATCH / 2} width={SWATCH} height={SWATCH} />
        <text className={styles.label} x={PLOT_X + 200 + SWATCH + 8} y={legendY} dominantBaseline="central">
          {wrongLabel}
        </text>
      </svg>
    </figure>
  );
}
