"use client";

import { Suspense, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { CmdMap, cmdRange } from "@/components/charts/CmdMap";
import { MplFigure } from "@/components/charts/MplFigure";
import { DataTable, type DataColumn } from "@/components/kit/DataTable";
import { Panel } from "@/components/kit/Panel";
import { graficosLabels as L } from "@/content/labels";
import { zFigureId } from "@/lib/figure-ids";
import type { Figures } from "@/lib/figures";
import type { Cmd, Graficos, Meta, RateRow } from "@/lib/schema";
import { readState, type UiState } from "@/lib/ui-state";
import graficosStyles from "./graficos.module.css";
import styles from "./page.module.css";

type Props = { meta: Meta; graficos: Graficos; figures: Figures };
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

function columnsWith(base: readonly DataColumn[], tech: boolean, extra: readonly DataColumn[]) {
  return tech ? [...base, ...extra] : base;
}

// So existe no modo tecnico (o modo simples nao mostra a calibracao).
function Calibration({ g, figures }: { g: Graficos; figures: Figures }) {
  const { calibration: cal } = g;
  const columns: DataColumn[] = [
    { key: "cls", label: L.colClass },
    { key: "p", label: L.colMeanProb, numeric: true },
    { key: "freq", label: L.colFreq, numeric: true },
    { key: "n", label: L.colObjects, numeric: true },
    { key: "ci", label: L.colFreqCi },
  ];
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
        <MplFigure id="calibracao" figures={figures} />
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
  figureIds,
  figures,
}: {
  id: string;
  title: string;
  chart: string;
  xLabel: string;
  note: string;
  series: { key: string; label: string; rows: RateRow[] }[];
  tech: boolean;
  figureIds: string[];
  figures: Figures;
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
      {figureIds.map((fid) => (
        <Scroller key={fid} label={L.chartRegion}>
          <MplFigure id={fid} figures={figures} />
        </Scroller>
      ))}
      {note ? <p className={styles.callout}>{note}</p> : null}
      <Numbers>
        <DataTable caption={chart} columns={columns} rows={rows} />
      </Numbers>
    </Panel>
  );
}

function Confusion({ g, tech, figures }: { g: Graficos; tech: boolean; figures: Figures }) {
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
      <Scroller label={L.chartRegion}>
        <MplFigure id={tech ? "confusao" : "confusao-simples"} figures={figures} />
      </Scroller>
      <Numbers>
        <DataTable caption={L.confusionCaption} columns={columns} rows={rows} />
      </Numbers>
    </Panel>
  );
}

function CmdSection({ cmd, tech, figures }: { cmd: Cmd; tech: boolean; figures: Figures }) {
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
  return (
    <Panel id="graf-cmd" title={L.cmdTitle}>
      {/* Modo simples: sem as figuras do mapa (nem a legenda da hachura); fica a tabela. */}
      {tech ? (
        <Scroller label={L.chartRegion}>
          <MplFigure id="cmd-taxa" figures={figures} />
        </Scroller>
      ) : null}
      {/* A contagem por celula nao tem figura matplotlib: continua no componente React. */}
      {tech ? (
        <Scroller label={L.chartRegion}>
          <CmdMap
            id="graf-cmd-count"
            title={L.cmdChartCount}
            cmd={cmd}
            mode="count"
            xLabel={L.cmdX}
            yLabel={L.cmdY}
            scaleLabel={L.cmdScaleCount}
            naText={L.notAvailable}
          />
        </Scroller>
      ) : null}
      {tech ? <Legend items={[L.legendCmdBelowFloor]} /> : null}
      <p className={styles.callout}>{cmd.note}</p>
      <Numbers>
        <DataTable caption={L.cmdChartRate} columns={columns} rows={rows} />
      </Numbers>
    </Panel>
  );
}

function Body({ props, state }: { props: Props; state: UiState }) {
  const { graficos: g, figures } = props;
  const toSeries = (s: { cls: string; label: string; rows: RateRow[] }) => ({
    key: s.cls,
    label: s.label,
    rows: s.rows,
  });
  return (
    <div className={styles.stack}>
      {/* Modo simples: sem os textos e paineis de jargao (calibracao, redshift); ficam as figuras simples. */}
      {state.tech ? (
        <div className={graficosStyles.context}>
          <p className={styles.answer}>{g.scope_text}</p>
          <p className={styles.muted}>{g.floor_text}</p>
        </div>
      ) : null}
      {state.tech ? <Calibration g={g} figures={figures} /> : null}
      <RatePanel
        id="graf-by-g"
        title={L.errorByGTitle}
        chart={L.errorByGChart}
        xLabel={L.errorByGX}
        note={g.error_by_g.missing_note}
        series={g.error_by_g.series.map(toSeries)}
        tech={state.tech}
        figureIds={[state.tech ? "taxa-por-g" : "taxa-por-g-simples"]}
        figures={figures}
      />
      {state.tech ? (
        <RatePanel
          id="graf-by-z"
          title={L.errorByZTitle}
          chart={L.errorByZChart}
          xLabel={L.errorByZX}
          note={g.error_by_z.star_note}
          series={g.error_by_z.series.map(toSeries)}
          tech
          figureIds={g.error_by_z.series.map((s) => zFigureId(s.cls))}
          figures={figures}
        />
      ) : null}
      <Confusion g={g} tech={state.tech} figures={figures} />
      {g.cmd === null ? null : <CmdSection cmd={g.cmd} tech={state.tech} figures={figures} />}
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
