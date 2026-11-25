"use client";

import { Suspense, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { DotCI, type DotCIRow } from "@/components/charts/DotCI";
import { HBarCI, type HBarCIRow } from "@/components/charts/HBarCI";
import {
  PriorSeries,
  type PriorSeriesCategory,
  type PriorSeriesLine,
} from "@/components/charts/PriorSeries";
import { DataTable, type DataColumn } from "@/components/kit/DataTable";
import { Panel } from "@/components/kit/Panel";
import { chartLabels, galaxiesLabels as L } from "@/content/labels";
import type { Est, Galaxies, GalaxyErrorRow, Meta, Rule } from "@/lib/schema";
import { readState, type UiState } from "@/lib/ui-state";
import galaxiesStyles from "./galaxies.module.css";
import styles from "./page.module.css";

type Props = { meta: Meta; galaxies: Galaxies };

const RULES: readonly Rule[] = ["as_shipped", "prior_matched"];
const AXIS_STEPS = [0.2, 0.4, 0.8, 1];

const MISSING: Est = { point: null, lo: null, hi: null, text: L.pointMissing, ci_text: L.pointMissing };

const columns: readonly DataColumn[] = [
  { key: "label", label: L.colBand },
  { key: "n", label: L.colObjects, numeric: true },
  { key: "errors", label: L.colErrors, numeric: true },
  { key: "rate", label: L.colRate, numeric: true },
  { key: "ci", label: L.colRateCi },
];

// Só escala de eixo (layout): o menor degrau que contém o maior limite superior exportado.
function axisMaxOf(rows: readonly GalaxyErrorRow[]): number {
  const top = Math.max(0, ...rows.map((r) => r.rate.hi ?? 0));
  return AXIS_STEPS.find((s) => s >= top) ?? 1;
}

function errorBars(prefix: string, rows: readonly GalaxyErrorRow[]): HBarCIRow[] {
  return rows.map((r, i) => ({
    key: `${prefix}-${i}`,
    label: r.label,
    est: r.rate,
    selected: false,
    partial: false,
  }));
}

function tableRows(rows: readonly GalaxyErrorRow[]): Record<string, string>[] {
  return rows.map((r) => ({
    label: r.label,
    n: r.n.text,
    errors: r.errors.text,
    rate: r.rate.text,
    ci: r.rate.ci_text,
  }));
}

function Scroller({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className={galaxiesStyles.scroll} role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  );
}

function priorSeries(meta: Meta, g: Galaxies, state: UiState, pick: "purity" | "completeness") {
  const base = g.blocks.as_shipped;
  const categories: PriorSeriesCategory[] = base.map((b) => ({
    key: b.prior,
    label: b.prior,
    partial: b.partial,
    selected: b.prior === state.prior,
  }));
  const series: PriorSeriesLine[] = RULES.map((rule, i) => ({
    key: rule,
    label: meta.rule_labels[rule],
    style: i === 0 ? "solid" : "dashed",
    points: base.map((b) => {
      const block = g.blocks[rule].find((x) => x.prior === b.prior);
      return {
        est: block ? block[pick] : MISSING,
        partial: block ? block.partial : b.partial,
        selected: rule === state.rule && b.prior === state.prior,
      };
    }),
  }));
  return { categories, series };
}

function Networks({ g }: { g: Galaxies }) {
  const nets = Object.entries(g.networks).flatMap(([key, n]) => (n ? [{ key, n }] : []));
  const rows: DotCIRow[] = nets.map(({ key, n }) => ({ key, label: n.label, est: n.diff }));
  return (
    <>
      {g.dl_error ? <p className={styles.callout}>{g.dl_error}</p> : null}
      {g.networks_note ? <p className={styles.callout}>{g.networks_note}</p> : null}
      {nets.length > 0 ? (
        <>
          <Scroller label={L.networksRegion}>
            <DotCI id="gal-net" title={L.networkDiffChart} rows={rows} zeroLine />
          </Scroller>
          <ul className={galaxiesStyles.nets}>
            {nets.map(({ key, n }) => (
              <li key={key} className={galaxiesStyles.net} data-net={key} data-state={n.state}>
                <strong>{n.label}</strong>
                <span>
                  {L.networkState}: {n.word}
                </span>
                {n.underpowered ? (
                  <span className={galaxiesStyles.tag}>{L.networkUnderpowered}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : g.dl_error || g.networks_note ? null : (
        <p className={styles.callout}>{L.networksUnavailable}</p>
      )}
      {g.power_note ? <p className={styles.callout}>{g.power_note}</p> : null}
    </>
  );
}

function Body({ props, state }: { props: Props; state: UiState }) {
  const { meta, galaxies: g } = props;
  const purity = priorSeries(meta, g, state, "purity");
  const completeness = priorSeries(meta, g, state, "completeness");
  const overall = g.error_rows;
  return (
    <div className={styles.stack}>
      <div className={galaxiesStyles.context}>
        <p className={styles.answer}>{g.verdict.text}</p>
        {g.verdict.scope_text ? <p className={styles.muted}>{g.verdict.scope_text}</p> : null}
        <p className={styles.muted}>{g.fixed_point_text}</p>
      </div>

      <Panel id="gal-purity" title={L.purityTitle} subtitle={L.puritySub}>
        <Scroller label={L.chartsRegion}>
          <PriorSeries
            id="gal-purity-chart"
            title={L.purityChart}
            categories={purity.categories}
            series={purity.series}
            partialTag={chartLabels.partialTag}
          />
        </Scroller>
      </Panel>
      <Panel id="gal-completeness" title={L.completenessTitle} subtitle={L.completenessSub}>
        <Scroller label={L.chartsRegion}>
          <PriorSeries
            id="gal-completeness-chart"
            title={L.completenessChart}
            categories={completeness.categories}
            series={completeness.series}
            partialTag={chartLabels.partialTag}
          />
        </Scroller>
      </Panel>

      <Panel id="gal-networks" title={L.networksTitle} subtitle={L.networksSub}>
        <Networks g={g} />
      </Panel>

      <Panel id="gal-errors" title={L.errorsTitle} subtitle={L.errorsSub}>
        <Scroller label={L.errorsRegion}>
          <HBarCI
            id="gal-errors-chart"
            title={L.errorsChart}
            rows={errorBars("err", overall)}
            axisMax={axisMaxOf(overall)}
            partialTag={chartLabels.partialTag}
          />
        </Scroller>
        <p className={galaxiesStyles.errorsNote}>{g.error_rows_note}</p>
      </Panel>

      <Panel id="gal-limits" title={L.limitsTitle}>
        <div className={galaxiesStyles.notes}>
          <p>{g.floor_note}</p>
          {g.crossmatch_note ? <p>{g.crossmatch_note}</p> : null}
        </div>
      </Panel>

      <Panel id="gal-context" title={L.contextTitle} subtitle={L.contextSub}>
        <ul className={galaxiesStyles.contextList}>
          <li>{L.contextPlane}</li>
          <li>{L.contextClouds}</li>
        </ul>
      </Panel>

      {state.tech ? <Technical g={g} /> : null}
    </div>
  );
}

function Technical({ g }: { g: Galaxies }) {
  const { by_g: byG, by_z: byZ, missing_note: missing } = g.technical;
  return (
    <Panel id="gal-tech" title={L.techTitle} subtitle={missing}>
      <div className={galaxiesStyles.tech}>
        <Scroller label={L.errorsRegion}>
          <HBarCI
            id="gal-g-chart"
            title={L.techByG}
            rows={errorBars("g", byG)}
            axisMax={axisMaxOf(byG)}
            partialTag={chartLabels.partialTag}
          />
        </Scroller>
        <DataTable caption={L.techByG} columns={columns} rows={tableRows(byG)} />
        <Scroller label={L.errorsRegion}>
          <HBarCI
            id="gal-z-chart"
            title={L.techByZ}
            rows={errorBars("z", byZ)}
            axisMax={axisMaxOf(byZ)}
            partialTag={chartLabels.partialTag}
          />
        </Scroller>
        <DataTable caption={L.techByZ} columns={columns} rows={tableRows(byZ)} />
      </div>
    </Panel>
  );
}

function WithUrlState(props: Props) {
  return <Body props={props} state={readState(useSearchParams(), props.meta)} />;
}

export function GalaxiesView(props: Props) {
  return (
    <Suspense fallback={<Body props={props} state={props.meta.default_state} />}>
      <WithUrlState {...props} />
    </Suspense>
  );
}
