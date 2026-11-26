"use client";

import { Suspense, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { CmdMap, cmdRange } from "@/components/charts/CmdMap";
import { ConfusionGrid } from "@/components/charts/ConfusionGrid";
import { RateByBin } from "@/components/charts/RateByBin";
import { ReliabilityPlot } from "@/components/charts/ReliabilityPlot";
import { DataTable, type DataColumn } from "@/components/kit/DataTable";
import { Panel } from "@/components/kit/Panel";
import { graficosLabels as L } from "@/content/labels";
import type { CalBin, Cmd, Graficos, Meta, RateRow } from "@/lib/schema";
import { readState, type UiState } from "@/lib/ui-state";
import graficosStyles from "./graficos.module.css";
import styles from "./page.module.css";

type Props = { meta: Meta; graficos: Graficos };
type Row = Record<string, string>;

function Scroller({ label, children, narrow = false }: { label: string; children: ReactNode; narrow?: boolean }) {
  return (
    <div className={narrow ? `${graficosStyles.scroll} ${graficosStyles.scrollCal}` : graficosStyles.scroll} role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  );
}

function Numbers({ children }: { children: ReactNode }) {
  return (
    <details className={graficosStyles.numbers}>
      <summary>{L.viewNumbers}</summary>
      {children}
    </details>
  );
}

function Legend({ items }: { items: string[] }) {
  return (
    <ul className={graficosStyles.legend} aria-label={L.legendTitle}>
      {items.map((t) => (
        <li key={t}>{t}</li>
      ))}
    </ul>
  );
}

function calPoint(b: CalBin) {
  return {
    p_mean: b.p_mean.value,
    freq: b.freq.point,
    n: b.n.value ?? 0,
    freq_lo: b.freq.lo,
    freq_hi: b.freq.hi,
    raw_p_mean: b.raw_p_mean,
    raw_freq: b.raw_freq,
  };
}

function columnsWith(base: readonly DataColumn[], tech: boolean, extra: readonly DataColumn[]) {
  return tech ? [...base, ...extra] : base;
}

function Calibration({ g, tech }: { g: Graficos; tech: boolean }) {
  const { calibration: cal } = g;
  const columns = columnsWith(
    [
      { key: "cls", label: L.colClass },
      { key: "p", label: L.colMeanProb, numeric: true },
      { key: "freq", label: L.colFreq, numeric: true },
    ],
    tech,
    [
      { key: "n", label: L.colObjects, numeric: true },
      { key: "ci", label: L.colFreqCi },
    ],
  );
  const rows: Row[] = cal.series.flatMap((s) =>
    s.bins.map((b) => ({
      cls: s.label,
      p: b.p_mean.text,
      freq: b.freq.text,
      n: b.n.text,
      ci: b.freq.ci_text,
    })),
  );
  return (
    <Panel id="graf-calibration" title={L.calibrationTitle}>
      <Scroller label={L.chartRegion} narrow>
        <ReliabilityPlot
          id="graf-calibration-chart"
          title={L.calibrationChart}
          series={cal.series.map((s) => ({ cls: s.cls, label: s.label, points: s.bins.map(calPoint) }))}
          xLabel={L.calibrationX}
          yLabel={L.calibrationY}
          diagonalLabel={L.legendDiagonal}
          intervals={tech}
          raw={tech}
          intervalLabel={L.legendInterval}
          rawLabel={L.legendRaw}
        />
      </Scroller>
      <p className={styles.callout}>{cal.networks_note}</p>
      <Numbers>
        <DataTable caption={L.calibrationChart} columns={columns} rows={rows} />
      </Numbers>
    </Panel>
  );
}

function rateRows(label: string, rows: readonly RateRow[]): Row[] {
  return rows.map((r) => ({
    cls: label,
    range: r.label,
    rate: r.rate.text,
    n: r.n.text,
    errors: r.errors.text,
    ci: r.rate.ci_text,
  }));
}

function RatePanel({
  id,
  title,
  chart,
  xLabel,
  note,
  series,
  tech,
  perClass = false,
}: {
  id: string;
  title: string;
  chart: string;
  xLabel: string;
  note: string;
  series: { key: string; label: string; rows: RateRow[] }[];
  tech: boolean;
  perClass?: boolean;
}) {
  const columns = columnsWith(
    [
      { key: "cls", label: L.colClass },
      { key: "range", label: xLabel },
      { key: "rate", label: L.colRate, numeric: true },
      { key: "n", label: L.colObjects, numeric: true },
    ],
    tech,
    [
      { key: "errors", label: L.colErrors, numeric: true },
      { key: "ci", label: L.colRateCi },
    ],
  );
  const rows = series.flatMap((s) => rateRows(s.label, s.rows));
  return (
    <Panel id={id} title={title}>
      {/* Faixas diferentes por classe (by-z): um grafico por classe, cada um com o seu eixo X. */}
      {(perClass ? series.map((s) => [s]) : [series]).map((group) => (
        <Scroller key={group.map((s) => s.key).join("-")} label={L.chartRegion}>
          <RateByBin
            id={perClass ? `${id}-chart-${group[0].key}` : `${id}-chart`}
            title={perClass ? group[0].label : chart}
            series={group}
            showCi={tech}
            yLabel={L.errorY}
            naText={L.notAvailable}
          />
        </Scroller>
      ))}
      <Legend items={[L.legendBelowFloor, ...(tech ? [L.legendInterval] : [])]} />
      {note ? <p className={styles.callout}>{note}</p> : null}
      <Numbers>
        <DataTable caption={chart} columns={columns} rows={rows} />
      </Numbers>
    </Panel>
  );
}

function Confusion({ g, tech }: { g: Graficos; tech: boolean }) {
  const { confusion: conf } = g;
  const labelOf = (cls: string): string => conf.classes.find((c) => c.cls === cls)?.label ?? cls;
  const columns = columnsWith(
    [
      { key: "true", label: L.confusionTrue },
      { key: "pred", label: L.confusionPred },
      { key: "share", label: L.colShare, numeric: true },
    ],
    tech,
    [
      { key: "n", label: L.colObjects, numeric: true },
      { key: "ci", label: L.colRateCi },
    ],
  );
  const rows: Row[] = conf.rows.flatMap((r) =>
    r.cells.map((c) => ({
      true: labelOf(r.true),
      pred: labelOf(c.pred),
      share: c.share.text,
      n: c.n.text,
      ci: c.share.ci_text,
    })),
  );
  return (
    <Panel id="graf-confusion" title={L.confusionTitle}>
      <figure aria-label={L.confusionTitle}>
        <ConfusionGrid
          caption={L.confusionCaption}
          classes={conf.classes}
          rows={conf.rows}
          showCi={tech}
          trueLabel={L.confusionTrue}
          predLabel={L.confusionPred}
        />
      </figure>
      <Numbers>
        <DataTable caption={L.confusionCaption} columns={columns} rows={rows} />
      </Numbers>
    </Panel>
  );
}

function CmdSection({ cmd, tech }: { cmd: Cmd; tech: boolean }) {
  const columns = columnsWith(
    [
      { key: "g", label: L.colGRange },
      { key: "c", label: L.colColorRange },
      { key: "rate", label: L.colRate, numeric: true },
    ],
    tech,
    [
      { key: "n", label: L.colObjects, numeric: true },
      { key: "ci", label: L.colRateCi },
    ],
  );
  const rows: Row[] = cmd.cells
    .filter((c) => !c.below_floor && c.rate.point !== null)
    .map((c) => {
      const range = cmdRange(cmd, c);
      return { g: range.g, c: range.c, rate: c.rate.text, n: c.n.text, ci: c.rate.ci_text };
    });
  const maps = [
    { mode: "rate" as const, title: L.cmdChartRate, scale: L.cmdScaleRate },
    ...(tech ? [{ mode: "count" as const, title: L.cmdChartCount, scale: L.cmdScaleCount }] : []),
  ];
  return (
    <Panel id="graf-cmd" title={L.cmdTitle}>
      {maps.map((m) => (
        <Scroller key={m.mode} label={L.chartRegion}>
          <CmdMap
            id={`graf-cmd-${m.mode}`}
            title={m.title}
            cmd={cmd}
            mode={m.mode}
            xLabel={L.cmdX}
            yLabel={L.cmdY}
            scaleLabel={m.scale}
            naText={L.notAvailable}
          />
        </Scroller>
      ))}
      <Legend items={[L.legendCmdBelowFloor]} />
      <p className={styles.callout}>{cmd.note}</p>
      <Numbers>
        <DataTable caption={L.cmdChartRate} columns={columns} rows={rows} />
      </Numbers>
    </Panel>
  );
}

function Body({ props, state }: { props: Props; state: UiState }) {
  const { graficos: g } = props;
  const toSeries = (s: { cls: string; label: string; rows: RateRow[] }) => ({
    key: s.cls,
    label: s.label,
    rows: s.rows,
  });
  return (
    <div className={styles.stack}>
      <div className={graficosStyles.context}>
        <p className={styles.answer}>{g.scope_text}</p>
        <p className={styles.muted}>{g.floor_text}</p>
      </div>
      <Calibration g={g} tech={state.tech} />
      <RatePanel
        id="graf-by-g"
        title={L.errorByGTitle}
        chart={L.errorByGChart}
        xLabel={L.errorByGX}
        note={g.error_by_g.missing_note}
        series={g.error_by_g.series.map(toSeries)}
        tech={state.tech}
      />
      <RatePanel
        id="graf-by-z"
        title={L.errorByZTitle}
        chart={L.errorByZChart}
        xLabel={L.errorByZX}
        note={g.error_by_z.star_note}
        series={g.error_by_z.series.map(toSeries)}
        tech={state.tech}
        perClass
      />
      <Confusion g={g} tech={state.tech} />
      {g.cmd === null ? null : <CmdSection cmd={g.cmd} tech={state.tech} />}
    </div>
  );
}

function WithUrlState(props: Props) {
  return <Body props={props} state={readState(useSearchParams(), props.meta)} />;
}

export function GraficosView(props: Props) {
  return (
    <Suspense fallback={<Body props={props} state={props.meta.default_state} />}>
      <WithUrlState {...props} />
    </Suspense>
  );
}
