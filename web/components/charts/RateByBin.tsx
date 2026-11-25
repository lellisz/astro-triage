import { ptPct } from "@/lib/fmt";
import type { RateRow } from "@/lib/schema";
import styles from "./graficos.module.css";

type RateByBinProps = {
  id: string;
  title: string;
  series: { key: string; label: string; rows: RateRow[] }[];
  showCi: boolean;
  yLabel: string;
  naText: string;
};

// Nao se chama VIEW_W de proposito: a varredura de escala de texto em type-scale.test.ts
// conta os graficos por esse nome e a pagina que usar este grafico deve estende-la.
const VIEWBOX_W = 720;
const PLOT_X = 64;
const PLOT_W = 560;
const PLOT_Y = 12;
const PLOT_H = 260;
const LABEL_H = 76;
const VIEWBOX_H = PLOT_Y + PLOT_H + LABEL_H;
const LABEL_X = PLOT_X + PLOT_W + 12;
const MARK = 6;
const CAP = 5;
const DODGE = 14;
const LABEL_GAP = 20;
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];
const MIN_ROTATE_BINS = 7;
const FALLBACK_TOP = 1;
// Tetos redondos: com as quatro divisoes do eixo, cada tick cai em numero redondo.
// Taxa nunca passa de 100%.
const NICE_TOPS = [0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.5, 0.8, 1];

const reportable = (r: RateRow): boolean => !r.below_floor && r.rate.point !== null;

function Marker({
  shape,
  cx,
  cy,
  hollow,
  legend = false,
}: {
  shape: number;
  cx: number;
  cy: number;
  hollow: boolean;
  legend?: boolean;
}) {
  const cls = hollow ? styles.markHollow : styles.mark;
  const data = legend ? {} : hollow ? { "data-hollow": "" } : { "data-point": "" };
  if (shape === 0) return <circle className={cls} {...data} cx={cx} cy={cy} r={MARK} />;
  if (shape === 1) {
    return <rect className={cls} {...data} x={cx - MARK} y={cy - MARK} width={MARK * 2} height={MARK * 2} />;
  }
  return (
    <polygon
      className={cls}
      {...data}
      points={`${cx},${cy - MARK - 1} ${cx + MARK + 1},${cy + MARK} ${cx - MARK - 1},${cy + MARK}`}
    />
  );
}

export function RateByBin({ id, title, series, showCi, yLabel, naText }: RateByBinProps) {
  const titleId = `${id}-title`;
  const widest = series.reduce<RateRow[]>((acc, s) => (s.rows.length > acc.length ? s.rows : acc), []);
  const bins = widest.length;
  const band = bins > 0 ? PLOT_W / bins : PLOT_W;
  const baseY = PLOT_Y + PLOT_H;

  const tops: number[] = [];
  for (const s of series) {
    for (const r of s.rows) {
      if (!reportable(r)) continue;
      const v = showCi && r.rate.hi !== null ? r.rate.hi : r.rate.point;
      if (v !== null) tops.push(v);
    }
  }
  const peak = tops.length > 0 ? Math.max(...tops) : 0;
  const top = peak > 0 ? (NICE_TOPS.find((t) => t >= peak) ?? FALLBACK_TOP) : FALLBACK_TOP;
  const py = (v: number): number => baseY - (v / top) * PLOT_H;
  const cx = (i: number, si: number): number =>
    PLOT_X + band * (i + 0.5) + (si - (series.length - 1) / 2) * DODGE;

  const summary = series
    .map((s) => `${s.label}: ${s.rows.map((r) => `${r.label} ${r.rate.text}`).join(", ")}`)
    .join("; ");

  // Rotulo direto no fim de cada serie, no nivel do ultimo ponto com valor; sem colisao.
  const wanted = series.map((s, si) => {
    let y = PLOT_Y + 12 + si * LABEL_GAP;
    for (let i = s.rows.length - 1; i >= 0; i--) {
      const p = s.rows[i].rate.point;
      if (reportable(s.rows[i]) && p !== null) {
        y = py(p);
        break;
      }
    }
    return { si, y };
  });
  const placed = new Map<number, number>();
  let prev = Number.NEGATIVE_INFINITY;
  for (const w of [...wanted].sort((a, b) => a.y - b.y)) {
    const y = Math.max(w.y, prev + LABEL_GAP);
    placed.set(w.si, y);
    prev = y;
  }

  const rotate = bins >= MIN_ROTATE_BINS;

  return (
    <figure className={styles.figure} id={id} aria-labelledby={titleId}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${VIEWBOX_W} ${VIEWBOX_H}`}
        role="img"
        aria-label={`${title}. ${yLabel}. ${summary}`}
      >
        {TICK_FRACTIONS.map((f) => (
          <g key={f}>
            <line className={styles.grid} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={py(f * top)} y2={py(f * top)} />
            <text className={styles.tick} x={PLOT_X - 8} y={py(f * top)} textAnchor="end" dominantBaseline="central">
              {ptPct(f * top)}
            </text>
          </g>
        ))}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={baseY} y2={baseY} />
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X} y1={PLOT_Y} y2={baseY} />
        <text
          className={styles.tick}
          transform={`translate(14 ${PLOT_Y + PLOT_H / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yLabel}
        </text>
        {widest.map((r, i) => {
          const x = PLOT_X + band * (i + 0.5);
          return (
            <text
              key={`${r.label}-${i}`}
              className={styles.tick}
              data-bin-label=""
              textAnchor={rotate ? "end" : "middle"}
              transform={rotate ? `translate(${x} ${baseY + 16}) rotate(-40)` : undefined}
              x={rotate ? undefined : x}
              y={rotate ? undefined : baseY + 20}
            >
              {r.label}
            </text>
          );
        })}
        {series.map((s, si) => (
          <g key={s.key} data-series={s.key}>
            {s.rows.map((r, i) => {
              const x = cx(i, si);
              const { point, lo, hi } = r.rate;
              if (reportable(r) && point !== null) {
                return (
                  <g key={`${r.label}-${i}`} data-row={`${s.key}-${i}`}>
                    <title>{`${s.label}, ${r.label}: ${r.rate.text}. ${r.rate.ci_text}`}</title>
                    {showCi && lo !== null && hi !== null ? (
                      <g className={styles.whisker} data-ci="">
                        <line x1={x} x2={x} y1={py(lo)} y2={py(hi)} />
                        <line x1={x - CAP} x2={x + CAP} y1={py(lo)} y2={py(lo)} />
                        <line x1={x - CAP} x2={x + CAP} y1={py(hi)} y2={py(hi)} />
                      </g>
                    ) : null}
                    <Marker shape={si} cx={x} cy={py(point)} hollow={false} />
                  </g>
                );
              }
              return (
                <g key={`${r.label}-${i}`} data-row={`${s.key}-${i}`} data-below-floor="">
                  <title>{`${s.label}, ${r.label}: ${r.rate.text || naText}`}</title>
                  <Marker shape={si} cx={x} cy={baseY} hollow />
                </g>
              );
            })}
          </g>
        ))}
        {series.map((s, si) => {
          const y = placed.get(si) ?? PLOT_Y;
          return (
            <g key={s.key} data-legend={s.key}>
              <Marker shape={si} cx={LABEL_X + MARK} cy={y} hollow={false} legend />
              <text className={styles.label} x={LABEL_X + 22} y={y} dominantBaseline="central">
                {s.label}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
