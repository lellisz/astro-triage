import { NA } from "@/components/kit/NA";
import { ptNum } from "@/lib/fmt";
import type { Est } from "@/lib/schema";
import styles from "./charts.module.css";

export type DotCIRow = {
  key: string;
  label: string;
  est: Est;
  // Distingue series sem depender de tom: forma do marcador e traco do intervalo.
  marker?: "diamond" | "square" | "circle";
  dashed?: boolean;
};

type DotCIProps = {
  id: string;
  title: string;
  rows: DotCIRow[];
  zeroLine?: boolean;
  naLabel?: string;
};

// Geometria fixa, no mesmo espirito do HBarCI: o desenho nao depende do conteudo.
const VIEW_W = 640;
const PLOT_X = 280;
const PLOT_W = 200;
const VALUE_X = 492;
const ROW_H = 44;
const CAP = 6;
const DIAMOND = 7;
const AXIS_H = 32;
const PAD_FRACTION = 0.08;
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];
const FALLBACK_DOMAIN: [number, number] = [0, 1];

function domainOf(rows: DotCIRow[], zeroLine: boolean): [number, number] {
  const bounds: number[] = [];
  for (const r of rows) {
    if (r.est.lo !== null) bounds.push(r.est.lo);
    if (r.est.hi !== null) bounds.push(r.est.hi);
  }
  if (zeroLine) bounds.push(0);
  if (bounds.length === 0) return FALLBACK_DOMAIN;
  const lo = Math.min(...bounds);
  const hi = Math.max(...bounds);
  const span = hi - lo > 0 ? hi - lo : 1;
  return [lo - span * PAD_FRACTION, hi + span * PAD_FRACTION];
}

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
  if (truncated) cur = `${cur.slice(0, LABEL_CHARS - 1)}…`;
  else if (cur.length > LABEL_CHARS) cur = `${cur.slice(0, LABEL_CHARS - 1)}…`;
  lines.push(cur);
  return { lines, truncated: truncated || cur.endsWith("…") };
}

function PointMark({ kind, x, cy }: { kind: "diamond" | "square" | "circle"; x: number; cy: number }) {
  if (kind === "circle") return <circle className={styles.diamond} data-point="" cx={x} cy={cy} r={DIAMOND} />;
  if (kind === "square") {
    return <rect className={styles.diamond} data-point="" x={x - DIAMOND} y={cy - DIAMOND} width={DIAMOND * 2} height={DIAMOND * 2} />;
  }
  return (
    <polygon
      className={styles.diamond}
      data-point=""
      points={`${x},${cy - DIAMOND} ${x + DIAMOND},${cy} ${x},${cy + DIAMOND} ${x - DIAMOND},${cy}`}
    />
  );
}

export function DotCI({ id, title, rows, zeroLine = false, naLabel }: DotCIProps) {
  const plotH = rows.length * ROW_H;
  const viewH = plotH + AXIS_H;
  const [dMin, dMax] = domainOf(rows, zeroLine);
  const x = (v: number): number => PLOT_X + ((v - dMin) / (dMax - dMin)) * PLOT_W;
  const summary = rows
    .map((r) => `${r.label}: ${r.est.point === null ? (naLabel ?? r.est.text) : `${r.est.text}, ${r.est.ci_text}`}`)
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
        {TICK_FRACTIONS.map((f) => {
          const v = dMin + f * (dMax - dMin);
          return (
            <g key={f}>
              <line className={styles.grid} x1={x(v)} x2={x(v)} y1={0} y2={plotH} />
              <text className={styles.tick} x={x(v)} y={plotH + 22} textAnchor="middle">
                {ptNum(v, 2, zeroLine)}
              </text>
            </g>
          );
        })}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={plotH} y2={plotH} />
        {zeroLine ? (
          <line className={styles.zero} data-zero="" x1={x(0)} x2={x(0)} y1={0} y2={plotH} />
        ) : null}
        {rows.map((r, i) => {
          const cy = i * ROW_H + ROW_H / 2;
          const { point, lo, hi } = r.est;
          const wrapped = wrapLabel(r.label);
          return (
            <g key={r.key} data-row={r.key}>
              <text className={styles.label} dominantBaseline="central">
                {wrapped.lines.map((ln, k) => (
                  <tspan key={k} x={0} y={cy + (k - (wrapped.lines.length - 1) / 2) * LABEL_LINE_H}>
                    {k < wrapped.lines.length - 1 ? `${ln} ` : ln}
                  </tspan>
                ))}
                {wrapped.truncated ? <title>{r.label}</title> : null}
              </text>
              {point !== null ? (
                <>
                  <title>{`${r.est.text}. ${r.est.ci_text}`}</title>
                  {lo !== null && hi !== null ? (
                    <g className={r.dashed ? styles.ciDashed : styles.ci} data-ci="">
                      <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} />
                      <line x1={x(lo)} x2={x(lo)} y1={cy - CAP} y2={cy + CAP} />
                      <line x1={x(hi)} x2={x(hi)} y1={cy - CAP} y2={cy + CAP} />
                    </g>
                  ) : null}
                  <PointMark kind={r.marker ?? "diamond"} x={x(point)} cy={cy} />
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
