import { ptPct } from "@/lib/fmt";
import type { Cls } from "@/lib/schema";
import styles from "./charts.module.css";

type Point = { p_mean: number; freq: number; n: number };

type ReliabilityPlotProps = {
  id: string;
  title: string;
  series: { cls: Cls; label: string; points: Point[] }[];
  xLabel: string;
  yLabel: string;
  diagonalLabel: string;
};

const VIEW_W = 620;
const PLOT_X = 72;
const PLOT_Y = 8;
const PLOT_S = 360;
const LEGEND_X = PLOT_X + PLOT_S + 16;
const VIEW_H = PLOT_Y + PLOT_S + 56;
const MARK = 6;
const TICKS = [0, 0.25, 0.5, 0.75, 1];

const px = (v: number): number => PLOT_X + Math.min(Math.max(v, 0), 1) * PLOT_S;
const py = (v: number): number => PLOT_Y + (1 - Math.min(Math.max(v, 0), 1)) * PLOT_S;

function Marker({ shape, cx, cy }: { shape: number; cx: number; cy: number }) {
  if (shape === 0) return <circle className={styles.mark} cx={cx} cy={cy} r={MARK} />;
  if (shape === 1) {
    return <rect className={styles.mark} x={cx - MARK} y={cy - MARK} width={MARK * 2} height={MARK * 2} />;
  }
  return (
    <polygon
      className={styles.mark}
      points={`${cx},${cy - MARK - 1} ${cx + MARK + 1},${cy + MARK} ${cx - MARK - 1},${cy + MARK}`}
    />
  );
}

export function ReliabilityPlot({ id, title, series, xLabel, yLabel, diagonalLabel }: ReliabilityPlotProps) {
  const titleId = `${id}-title`;
  const summary = series
    .map((s) => `${s.label}: ${s.points.map((p) => `${ptPct(p.p_mean)} → ${ptPct(p.freq)}`).join(", ")}`)
    .join("; ");

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
        {TICKS.map((t) => (
          <g key={t}>
            <line className={styles.grid} x1={px(t)} x2={px(t)} y1={PLOT_Y} y2={PLOT_Y + PLOT_S} />
            <line className={styles.grid} x1={PLOT_X} x2={PLOT_X + PLOT_S} y1={py(t)} y2={py(t)} />
            <text className={styles.tick} x={px(t)} y={PLOT_Y + PLOT_S + 20} textAnchor="middle">
              {ptPct(t, 0)}
            </text>
            <text className={styles.tick} x={PLOT_X - 8} y={py(t)} textAnchor="end" dominantBaseline="central">
              {ptPct(t, 0)}
            </text>
          </g>
        ))}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_S} y1={PLOT_Y + PLOT_S} y2={PLOT_Y + PLOT_S} />
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X} y1={PLOT_Y} y2={PLOT_Y + PLOT_S} />
        <line className={styles.zero} x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} />
        <text className={styles.tick} x={PLOT_X + PLOT_S / 2} y={VIEW_H - 8} textAnchor="middle">
          {xLabel}
        </text>
        <text
          className={styles.tick}
          transform={`translate(14 ${PLOT_Y + PLOT_S / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yLabel}
        </text>
        {series.map((s, si) =>
          s.points.map((p, i) => (
            <g key={`${s.cls}-${i}`} data-cal-point={`${s.cls}-${i}`}>
              <title>{`${s.label}: ${ptPct(p.p_mean)} → ${ptPct(p.freq)}`}</title>
              <Marker shape={si} cx={px(p.p_mean)} cy={py(p.freq)} />
            </g>
          )),
        )}
        <g>
          {series.map((s, si) => (
            <g key={s.cls} data-legend={s.cls}>
              <Marker shape={si} cx={LEGEND_X + MARK} cy={PLOT_Y + 12 + si * 26} />
              <text className={styles.label} x={LEGEND_X + 22} y={PLOT_Y + 12 + si * 26} dominantBaseline="central">
                {s.label}
              </text>
            </g>
          ))}
          <line
            className={styles.zero}
            x1={LEGEND_X}
            x2={LEGEND_X + 2 * MARK}
            y1={PLOT_Y + 12 + series.length * 26}
            y2={PLOT_Y + 12 + series.length * 26}
          />
          <text
            className={styles.label}
            x={LEGEND_X + 22}
            y={PLOT_Y + 12 + series.length * 26}
            dominantBaseline="central"
          >
            {diagonalLabel}
          </text>
        </g>
      </svg>
    </figure>
  );
}
