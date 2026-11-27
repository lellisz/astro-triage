"use client";

import Link from "next/link";
import { Suspense, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { HBarCI, type HBarCIRow } from "@/components/charts/HBarCI";
import { MplFigure } from "@/components/charts/MplFigure";
import { DataTable, type DataColumn } from "@/components/kit/DataTable";
import { Panel } from "@/components/kit/Panel";
import { chartLabels, compareHref, galaxiesLabels as L } from "@/content/labels";
import type { Figures } from "@/lib/figures";
import { ptNum } from "@/lib/fmt";
import type { Galaxies, GalaxyErrorRow, Meta, Rule } from "@/lib/schema";
import { readState, withState, type UiState } from "@/lib/ui-state";
import galaxiesStyles from "./galaxies.module.css";
import styles from "./page.module.css";

type Props = { meta: Meta; galaxies: Galaxies; figures: Figures };

const RULES: readonly Rule[] = ["as_shipped", "prior_matched"];

const columns: readonly DataColumn[] = [
  { key: "label", label: L.colBand },
  { key: "n", label: L.colObjects, numeric: true },
  { key: "errors", label: L.colErrors, numeric: true },
  { key: "rate", label: L.colRate, numeric: true },
  { key: "ci", label: L.colRateCi },
];

const netColumns: readonly DataColumn[] = [
  { key: "net", label: L.colNet },
  { key: "diff", label: L.colDiff, numeric: true },
  { key: "ci", label: L.colCi },
  { key: "delta", label: L.colDelta, numeric: true },
  { key: "verdict", label: L.colVerdict },
];

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

// As duas regras com cada prior do bloco, na mesma forma de barras da Visão geral.
function priorRows(meta: Meta, g: Galaxies, state: UiState, pick: "purity" | "completeness"): HBarCIRow[] {
  return RULES.flatMap((rule) =>
    g.blocks[rule].map((b) => ({
      key: `${rule}-${b.prior}`,
      label:
        meta.states.find((s) => s.rule === rule && s.prior === b.prior)?.label ??
        `${meta.rule_labels[rule]} · ${b.prior}`,
      est: b[pick],
      selected: rule === state.rule && b.prior === state.prior,
      partial: b.partial,
    })),
  );
}

function Networks({ g, meta, state }: { g: Galaxies; meta: Meta; state: UiState }) {
  const nets = Object.entries(g.networks).flatMap(([key, n]) => (n ? [{ key, n }] : []));
  const rows = nets.map(({ n }) => ({
    net: n.label,
    diff: n.diff.text,
    ci: n.diff.ci_text,
    // Mesma regra do kind "pp" de types.py (x100, 2 casas, sufixo " p.p."), unidade de D.
    delta: `${ptNum(n.delta * 100, 2)} p.p.`,
    verdict: n.word,
  }));
  return (
    <>
      {state.tech ? <p className={styles.muted}>{g.fixed_point_text}</p> : null}
      {g.dl_error ? <p className={styles.callout}>{g.dl_error}</p> : null}
      {g.networks_note ? <p className={styles.callout}>{g.networks_note}</p> : null}
      {nets.length > 0 ? (
        <>
          {state.tech ? <DataTable caption={L.netTableCaption} columns={netColumns} rows={rows} /> : null}
          <p>
            <Link href={withState(compareHref, state, meta.release)}>{L.compareLink}</Link>
          </p>
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
  const { meta, galaxies: g, figures } = props;
  const purity = priorRows(meta, g, state, "purity");
  const completeness = priorRows(meta, g, state, "completeness");
  return (
    <div className={styles.stack}>
      {/* Modo simples: o veredito e os graficos por prior saem; ficam a figura simples e as notas. */}
      {state.tech ? (
        <>
          <div className={galaxiesStyles.context}>
            <p className={styles.answer}>{g.verdict.text}</p>
            {g.verdict.scope_text ? <p className={styles.muted}>{g.verdict.scope_text}</p> : null}
          </div>

          <Panel id="gal-purity" title={L.purityTitle} subtitle={L.puritySub}>
            <Scroller label={L.chartsRegion}>
              <HBarCI
                id="gal-purity-chart"
                title={L.purityChart}
                rows={purity}
                partialTag={chartLabels.partialTag}
              />
            </Scroller>
          </Panel>
          <Panel id="gal-completeness" title={L.completenessTitle} subtitle={L.completenessSub}>
            <Scroller label={L.chartsRegion}>
              <HBarCI
                id="gal-completeness-chart"
                title={L.completenessChart}
                rows={completeness}
                partialTag={chartLabels.partialTag}
              />
            </Scroller>
          </Panel>
        </>
      ) : null}

      <Panel id="gal-networks" title={L.networksTitle} subtitle={L.networksSub}>
        <Networks g={g} meta={meta} state={state} />
      </Panel>

      <Panel id="gal-errors" title={L.errorsTitle} subtitle={state.tech ? L.errorsSub : L.errorsSubSimple}>
        <Scroller label={L.errorsRegion}>
          <MplFigure id={state.tech ? "galaxia-erros" : "galaxia-erros-simples"} figures={figures} />
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
        <DataTable caption={L.techByG} columns={columns} rows={tableRows(byG)} />
        <p className={styles.muted}>{L.galBandsJoinNote}</p>
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
