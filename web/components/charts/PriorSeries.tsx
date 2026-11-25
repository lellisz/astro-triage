import { NA } from "@/components/kit/NA";
import { chartLabels } from "@/content/labels";
import { ptPct } from "@/lib/fmt";
import type { Est } from "@/lib/schema";
import styles from "./charts.module.css";

export type PriorSeriesCategory = { key: string; label: string; partial: boolean; selected: boolean };

export type PriorSeriesPoint = { est: Est; partial: boolean; selected: boolean };

export type PriorSeriesLine = {
  key: string;
  label: string;
  // Contínuo e círculo, ou tracejado e quadrado: a série nunca depende de tom.
  style: "solid" | "dashed";
  points: PriorSeriesPoint[];
};

type PriorSeriesProps = {
  id: string;
  title: string;
  categories: PriorSeriesCategory[];
  series: PriorSeriesLine[];
  partialTag: string;
  naLabel?: string;
};

const VIEW_W = 640;
const PLOT_X = 56;
const PLOT_W = 560;
const LEGEND_H = 40;
const PLOT_H = 220;
const AXIS_H = 40;
const SERIES_DX = 24;
const CAP = 6;
const MARK = 7;
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];

const yOf = (v: number): number => LEGEND_H + PLOT_H - Math.min(Math.max(v, 0), 1) * PLOT_H;

function Marker({ kind, x, y, hollow }: { kind: "circle" | "square" | "triangle"; x: number; y: number; hollow: boolean }) {
  const cls = hollow ? styles.markHollow : styles.mark;
  if (kind === "circle") return <circle className={cls} cx={x} cy={y} r={MARK} />;
  if (kind === "square") return <rect className={cls} x={x - MARK} y={y - MARK} width={MARK * 2} height={MARK * 2} />;
  return <polygon className={cls} points={`${x},${y - MARK - 1} ${x + MARK + 1},${y + MARK} ${x - MARK - 1},${y + MARK}`} />;
}

export function PriorSeries({ id, title, categories, series, partialTag, naLabel }: PriorSeriesProps) {
  const viewH = LEGEND_H + PLOT_H + AXIS_H;
  const slot = PLOT_W / categories.length;
  const cx = (i: number): number => PLOT_X + slot * (i + 0.5);
  const side = (s: number): number => (s === 0 ? -1 : 1);
  const catLabel = (c: PriorSeriesCategory): string => (c.partial ? `${c.label} (${partialTag})` : c.label);
  const summary = series
    .map((s) =>
      `${s.label}: ${s.points
        .map((p, i) => `${categories[i].label} ${p.est.point === null ? (naLabel ?? p.est.text) : `${p.est.text}, ${p.est.ci_text}`}`)
        .join("; ")}`,
    )
    .join(". ");
  const titleId = `${id}-title`;

  return (
    <figure className={styles.figure} id={id} aria-labelledby={titleId}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <svg className={styles.svg} viewBox={`0 0 ${VIEW_W} ${viewH}`} role="img" aria-label={`${title}. ${summary}`}>
        {series.map((s, si) => {
          const lx = PLOT_X + si * (PLOT_W / series.length);
          return (
            <g key={s.key} data-legend={s.key}>
              <line
                className={s.style === "dashed" ? styles.seriesDashed : styles.seriesSolid}
                x1={lx}
                x2={lx + 36}
                y1={14}
                y2={14}
              />
              <Marker kind={s.style === "dashed" ? "square" : "circle"} x={lx + 18} y={14} hollow={false} />
              <text className={styles.value} x={lx + 46} y={14} dominantBaseline="central">
                {s.label}
              </text>
            </g>
          );
        })}
        {TICK_FRACTIONS.map((f) => (
          <g key={f}>
            <line className={styles.grid} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={yOf(f)} y2={yOf(f)} />
            <text className={styles.tick} x={PLOT_X - 8} y={yOf(f)} textAnchor="end" dominantBaseline="central">
              {ptPct(f, 0)}
            </text>
          </g>
        ))}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={yOf(0)} y2={yOf(0)} />
        {categories.map((c, i) => (
          <text
            key={c.key}
            className={c.selected ? styles.catSelected : styles.cat}
            x={cx(i)}
            y={LEGEND_H + PLOT_H + 26}
            textAnchor="middle"
          >
            {c.selected ? `${chartLabels.selectedGlyph} ${catLabel(c)}` : catLabel(c)}
          </text>
        ))}
        {series.map((s, si) => {
          const dx = side(si) * SERIES_DX;
          const known = s.points
            .map((p, i) => ({ p, i }))
            .filter(({ p }) => p.est.point !== null);
          const path = known.map(({ p, i }) => `${cx(i) + dx},${yOf(p.est.point as number)}`).join(" ");
          return (
            <g key={s.key} data-series={s.key}>
              {known.length > 1 ? (
                <polyline className={s.style === "dashed" ? styles.seriesDashed : styles.seriesSolid} points={path} />
              ) : null}
              {s.points.map((p, i) => {
                const { point, lo, hi } = p.est;
                const x = cx(i) + dx;
                const cls = [p.selected ? styles.selected : "", p.partial ? styles.partial : ""].filter(Boolean).join(" ");
                const rowKey = `${s.key}-${categories[i].key}`;
                if (point === null) {
                  return (
                    <g key={rowKey} data-row={rowKey} data-partial={p.partial ? "true" : undefined}>
                      <foreignObject x={x - 40} y={LEGEND_H + PLOT_H / 2 - 16} width={80} height={32}>
                        <div className={styles.naCenter}>
                          <NA text={p.est.text} />
                        </div>
                      </foreignObject>
                    </g>
                  );
                }
                return (
                  <g
                    key={rowKey}
                    className={cls || undefined}
                    data-row={rowKey}
                    data-partial={p.partial ? "true" : undefined}
                    aria-current={p.selected ? "true" : undefined}
                  >
                    <title>{`${s.label}, ${categories[i].label}: ${p.est.text}. ${p.est.ci_text}`}</title>
                    {lo !== null && hi !== null ? (
                      <g className={styles.ci} data-ci="">
                        <line x1={x} x2={x} y1={yOf(lo)} y2={yOf(hi)} />
                        <line x1={x - CAP} x2={x + CAP} y1={yOf(lo)} y2={yOf(lo)} />
                        <line x1={x - CAP} x2={x + CAP} y1={yOf(hi)} y2={yOf(hi)} />
                      </g>
                    ) : null}
                    {p.selected ? <circle className={styles.ring} cx={x} cy={yOf(point)} r={MARK + 6} /> : null}
                    <Marker
                      kind={p.partial ? "triangle" : s.style === "dashed" ? "square" : "circle"}
                      x={x}
                      y={yOf(point)}
                      hollow={p.partial}
                    />
                    <text
                      className={`${p.selected ? styles.valueSelected : styles.value} ${styles.halo}`}
                      x={x + side(si) * (MARK + 8)}
                      y={yOf(point)}
                      textAnchor={side(si) < 0 ? "end" : "start"}
                      dominantBaseline="central"
                    >
                      {p.est.text}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
