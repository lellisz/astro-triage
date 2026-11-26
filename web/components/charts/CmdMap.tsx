import { useId } from "react";
import { MIN_OPACITY } from "@/components/sky/scale";
import type { Cmd, CmdCell } from "@/lib/schema";
import styles from "./graficos.module.css";

export type CmdMode = "rate" | "count";

type CmdMapProps = {
  id: string;
  title: string;
  cmd: Cmd;
  mode: CmdMode;
  xLabel: string;
  yLabel: string;
  scaleLabel: string;
  naText: string;
};

// O nome VIEWBOX_W entra na varredura de escala de texto de type-scale.test.ts.
const VIEWBOX_W = 720;
const PLOT_X = 64;
const PLOT_W = 560;
const PLOT_Y = 12;
const CELL_H = 16;
const TICK_LEN = 5;
const X_AXIS_H = 64;
const HATCH_STEP = 6;

const valueOf = (c: CmdCell, mode: CmdMode): number | null => (mode === "rate" ? c.rate.point : c.n.value);
const reportable = (c: CmdCell, mode: CmdMode): boolean => !c.below_floor && valueOf(c, mode) !== null;

function opacityFor(value: number, cap: number | null): number {
  if (cap === null || cap <= 0) return MIN_OPACITY;
  const t = Math.min(Math.max(value / cap, 0), 1);
  return MIN_OPACITY + (1 - MIN_OPACITY) * t;
}

// Faixas da celula: textos prontos exportados pelo Python (g_labels, c_labels).
export function cmdRange(cmd: Cmd, cell: Pick<CmdCell, "gi" | "ci">): { g: string; c: string } {
  return { g: cmd.g_labels[cell.gi], c: cmd.c_labels[cell.ci] };
}

export function CmdMap({ id, title, cmd, mode, xLabel, yLabel, scaleLabel, naText }: CmdMapProps) {
  const uid = useId();
  const titleId = `${id}-title`;
  const hatchId = `${uid}-hatch`;
  const gradId = `${uid}-grad`;
  const rows = cmd.g_edges.length - 1;
  const cols = cmd.c_edges.length - 1;
  const cellW = PLOT_W / cols;
  const plotH = rows * CELL_H;
  const baseY = PLOT_Y + plotH;
  const viewH = baseY + X_AXIS_H;
  const [g0, gN] = [cmd.g_edges[0], cmd.g_edges[rows]];
  const [c0, cN] = [cmd.c_edges[0], cmd.c_edges[cols]];
  const cap = mode === "rate" ? cmd.cap.rate : cmd.cap.count;
  const px = (v: number): number => PLOT_X + ((v - c0) / (cN - c0)) * PLOT_W;
  const py = (v: number): number => PLOT_Y + ((v - g0) / (gN - g0)) * plotH;
  const min = mode === "rate" ? cmd.cap.rate_zero_text : cmd.cap.count_zero_text;
  const max = (mode === "rate" ? cmd.cap.rate_text : cmd.cap.count_text) ?? naText;

  return (
    <figure className={styles.figure} id={id} aria-labelledby={titleId}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${VIEWBOX_W} ${viewH}`}
        role="img"
        aria-label={`${title}. ${yLabel}, ${xLabel}. ${scaleLabel}: ${min} – ${max}`}
      >
        <defs>
          <pattern id={hatchId} width={HATCH_STEP} height={HATCH_STEP} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line className={styles.hatchLine} x1="0" y1="0" x2="0" y2={HATCH_STEP} />
          </pattern>
        </defs>
        {cmd.cells.map((c) => {
          const x = PLOT_X + c.ci * cellW;
          const y = PLOT_Y + c.gi * CELL_H;
          const range = cmdRange(cmd, c);
          const where = `${yLabel} ${range.g}; ${xLabel} ${range.c}`;
          const value = valueOf(c, mode);
          if (reportable(c, mode) && value !== null) {
            const text = mode === "rate" ? `${c.rate.text}. ${c.rate.ci_text}` : c.n.text;
            return (
              <rect
                key={`${c.gi}-${c.ci}`}
                className={styles.cmdCell}
                data-cell={`${c.gi}-${c.ci}`}
                data-point=""
                x={x}
                y={y}
                width={cellW}
                height={CELL_H}
                fillOpacity={opacityFor(value, cap)}
              >
                <title>{`${where}: ${text}`}</title>
              </rect>
            );
          }
          return (
            <rect
              key={`${c.gi}-${c.ci}`}
              className={styles.cmdBelow}
              data-cell={`${c.gi}-${c.ci}`}
              data-below-floor=""
              x={x}
              y={y}
              width={cellW}
              height={CELL_H}
              fill={`url(#${hatchId})`}
            >
              <title>{`${where}: ${c.rate.text || naText}`}</title>
            </rect>
          );
        })}
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={baseY} y2={baseY} />
        <line className={styles.axis} x1={PLOT_X} x2={PLOT_X} y1={PLOT_Y} y2={baseY} />
        {cmd.c_ticks.map((t) => (
          <g key={`c-${t.value}`}>
            <line className={styles.axis} x1={px(t.value)} x2={px(t.value)} y1={baseY} y2={baseY + TICK_LEN} />
            <text className={styles.tick} data-tick="c" x={px(t.value)} y={baseY + TICK_LEN + 16} textAnchor="middle">
              {t.text}
            </text>
          </g>
        ))}
        {cmd.g_ticks.map((t) => (
          <g key={`g-${t.value}`}>
            <line className={styles.axis} x1={PLOT_X - TICK_LEN} x2={PLOT_X} y1={py(t.value)} y2={py(t.value)} />
            <text className={styles.tick} data-tick="g" x={PLOT_X - TICK_LEN - 4} y={py(t.value)} textAnchor="end" dominantBaseline="central">
              {t.text}
            </text>
          </g>
        ))}
        <text className={styles.tick} x={PLOT_X + PLOT_W / 2} y={baseY + X_AXIS_H - 10} textAnchor="middle">
          {xLabel}
        </text>
        <text className={styles.tick} transform={`translate(14 ${PLOT_Y + plotH / 2}) rotate(-90)`} textAnchor="middle">
          {yLabel}
        </text>
      </svg>
      <div className={styles.cmdScale}>
        <span className={styles.cmdScaleTitle}>{scaleLabel}</span>
        <div className={styles.cmdScaleBar}>
          <span>{min}</span>
          <svg className={styles.cmdGradient} viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <linearGradient id={gradId} x1="0" x2="1" y1="0" y2="0">
                <stop className={styles.cmdStop} offset="0" stopOpacity={MIN_OPACITY} />
                <stop className={styles.cmdStop} offset="1" stopOpacity={1} />
              </linearGradient>
            </defs>
            <rect width="100" height="10" fill={`url(#${gradId})`} />
          </svg>
          <span>{max}</span>
        </div>
      </div>
    </figure>
  );
}
