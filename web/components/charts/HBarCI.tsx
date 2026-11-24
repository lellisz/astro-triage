import { NA } from "@/components/kit/NA";
import { chartLabels } from "@/content/labels";
import { ptPct } from "@/lib/fmt";
import type { Est } from "@/lib/schema";
import styles from "./charts.module.css";

export type HBarCIRow = {
  key: string;
  label: string;
  est: Est;
  selected: boolean;
  partial: boolean;
};

type HBarCIProps = {
  id: string;
  title: string;
  rows: HBarCIRow[];
  axisMax?: number;
  partialTag: string;
  naLabel?: string;
};

// Geometria fixa: duas instancias lado a lado ficam alinhadas independentemente do conteudo.
const VIEW_W = 640;
const PLOT_X = 280;
const PLOT_W = 200;
const VALUE_X = 492;
const ROW_H = 44;
const BAR_H = 14;
const CAP = 6;
const MARKER_R = 6;
const AXIS_H = 32;
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];

// Rotulo livre quebrado em ate 2 linhas dentro da coluna de rotulo (0..PLOT_X); excesso vira reticencias.
const LABEL_CHARS = 32;
const LABEL_LINE_H = 16;

function wrapLabel(label: string): { lines: string[]; truncated: boolean } {
  const words = label.split(" ");
  const lines: string[] = [];
  let cur = "";
  let i = 0;
  for (; i < words.length; i++) {
    const next = cur ? `${cur} ${words[i]}` : words[i];
    if (next.length > LABEL_CHARS && cur) {
      if (lines.length === 1) break;
      lines.push(cur);
      cur = words[i];
    } else {
      cur = next;
    }
  }
  const truncated = i < words.length;
  if (truncated || cur.length > LABEL_CHARS) cur = `${cur.slice(0, LABEL_CHARS - 1)}…`;
  lines.push(cur);
  return { lines, truncated: truncated || cur.endsWith("…") };
}

export function HBarCI({ id, title, rows, axisMax = 1, partialTag, naLabel }: HBarCIProps) {
  const plotH = rows.length * ROW_H;
  const viewH = plotH + AXIS_H;
  const x = (v: number): number => PLOT_X + (Math.min(Math.max(v, 0), axisMax) / axisMax) * PLOT_W;
  const rowLabel = (r: HBarCIRow): string => (r.partial ? `${r.label} (${partialTag})` : r.label);
  const summary = rows
    .map((r) => `${rowLabel(r)}: ${r.est.point === null ? (naLabel ?? r.est.text) : `${r.est.text}, ${r.est.ci_text}`}`)
    .join("; ");
  const titleId = `${id}-title`;

  return (
    <figure className={styles.figure} id={id} aria-labelledby={titleId}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${VIEW_W} ${viewH}`}
        role="img"
        aria-label={`${title}. ${summary}`}
      >
        {TICK_FRACTIONS.map((f) => (
          <g key={f}>
            <line className={styles.grid} x1={x(f * axisMax)} x2={x(f * axisMax)} y1={0} y2={plotH} />
            <text className={styles.tick} x={x(f * axisMax)} y={plotH + 22} textAnchor="middle">
              {ptPct(f * axisMax, 0)}
            </text>
          </g>
        ))}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={plotH} y2={plotH} />
        {rows.map((r, i) => {
          const cy = i * ROW_H + ROW_H / 2;
          const { point, lo, hi } = r.est;
          const hasPoint = point !== null;
          const fullLabel = `${r.selected ? `${chartLabels.selectedGlyph} ` : ""}${rowLabel(r)}`;
          const wrapped = wrapLabel(fullLabel);
          const classes = [r.selected ? styles.selected : "", r.partial ? styles.partial : ""]
            .filter(Boolean)
            .join(" ");
          return (
            <g
              key={r.key}
              className={classes || undefined}
              data-row={r.key}
              data-partial={r.partial ? "true" : undefined}
              aria-current={r.selected ? "true" : undefined}
            >
              <text className={styles.label} dominantBaseline="central">
                {wrapped.lines.map((ln, k) => (
                  <tspan key={k} x={0} y={cy + (k - (wrapped.lines.length - 1) / 2) * LABEL_LINE_H}>
                    {k < wrapped.lines.length - 1 ? `${ln} ` : ln}
                  </tspan>
                ))}
                {wrapped.truncated ? <title>{fullLabel}</title> : null}
              </text>
              {hasPoint ? (
                <>
                  <title>{`${r.est.text}. ${r.est.ci_text}`}</title>
                  <rect
                    className={styles.bar}
                    data-bar=""
                    x={PLOT_X}
                    y={cy - BAR_H / 2}
                    width={x(point) - PLOT_X}
                    height={BAR_H}
                  />
                  {lo !== null && hi !== null ? (
                    <g className={styles.ci} data-ci="">
                      <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} />
                      <line x1={x(lo)} x2={x(lo)} y1={cy - CAP} y2={cy + CAP} />
                      <line x1={x(hi)} x2={x(hi)} y1={cy - CAP} y2={cy + CAP} />
                    </g>
                  ) : null}
                  <circle className={styles.marker} cx={x(point)} cy={cy} r={MARKER_R} />
                  <text className={styles.value} x={VALUE_X} y={cy} dominantBaseline="central">
                    {r.est.text}
                  </text>
                </>
              ) : (
                <foreignObject x={VALUE_X} y={cy - ROW_H / 2} width={VIEW_W - VALUE_X} height={ROW_H}>
                  <div className={styles.na}>
                    <NA text={r.est.text} />
                  </div>
                </foreignObject>
              )}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
