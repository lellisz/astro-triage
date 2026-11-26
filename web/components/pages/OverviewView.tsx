"use client";

import Link from "next/link";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { HBarCI, type HBarCIRow } from "@/components/charts/HBarCI";
import { ContextStrip } from "@/components/kit/ContextStrip";
import { Kpi } from "@/components/kit/Kpi";
import { Panel } from "@/components/kit/Panel";
import { SkyThumb } from "@/components/sky/SkyThumb";
import { OverviewTechnical } from "./OverviewTechnical";
import { chartLabels, overviewLabels as L, routes } from "@/content/labels";
import type { Meta, Overview, OverviewHeadline, Sky } from "@/lib/schema";
import { readState, withState, type UiState } from "@/lib/ui-state";
import overviewStyles from "./overview.module.css";
import styles from "./page.module.css";

type Props = {
  meta: Meta;
  headline?: OverviewHeadline;
  rows: Overview["rows"];
  accuracyNote: string;
  technical: Overview["technical"];
  sky?: Sky;
};

function Headline({ h, meta, state }: { h: OverviewHeadline; meta: Meta; state: UiState }) {
  const unavailable = h.key === "unavailable";
  const o = overviewStyles;
  return (
    <section className={o.headline} aria-labelledby="ov-headline-title" data-key={h.key}>
      <p className={o.question}>{h.question}</p>
      <h2 id="ov-headline-title" className={o.title}>
        {h.title}
      </h2>
      <p className={o.lead}>{h.lead}</p>
      {h.caveat ? <p className={o.caveat}>{h.caveat}</p> : null}
      {unavailable ? null : (
        <>
          <ul className={o.pairs}>
            {h.pairs.map((p) => (
              <li key={p.net}>
                <strong>{p.net}</strong>: {p.text}
              </li>
            ))}
          </ul>
          <div className={o.contrast} role="group" aria-label={L.contrastRegion}>
            <p className={o.figure}>
              <span className={o.figureLabel}>{L.contrastAccuracy}</span>
              <span className={o.figureValue}>{h.contrast.accuracy.text}</span>
            </p>
            <p className={o.figure}>
              <span className={o.figureLabel}>{L.contrastPurity}</span>
              <span className={o.figureValue}>{h.contrast.qso_purity.text}</span>
            </p>
          </div>
          <p className={o.contrastText}>{h.contrast.text}</p>
        </>
      )}
      <p className={o.summaryEn} lang="en">
        {h.summary_en}
      </p>
      {state.tech ? null : (
        <Link
          className={o.techLink}
          href={withState("/", { ...state, tech: true }, meta.release)}
          aria-label={L.techLinkAria}
        >
          {L.techLink}
        </Link>
      )}
    </section>
  );
}

function Body({ props, state }: { props: Props; state: UiState }) {
  const { meta, headline, rows, accuracyNote, technical, sky } = props;
  const row = rows.find((r) => r.rule === state.rule && r.prior === state.prior);
  const partialRow = rows.find((r) => r.partial && r.partial_notice);
  const chartRows = (pick: "purity" | "completeness"): HBarCIRow[] =>
    rows.map((r) => ({
      key: `${r.rule}-${r.prior}`,
      label:
        meta.states.find((s) => s.rule === r.rule && s.prior === r.prior)?.label ??
        `${meta.rule_labels[r.rule]} · ${r.prior}`,
      est: r[pick],
      selected: r.rule === state.rule && r.prior === state.prior,
      partial: r.partial,
    }));
  const shortcuts = routes.filter((r) => r.href !== "/");
  const box = technical.data_box;
  const contextItems = [
    { label: L.nTest, text: box.n_test.text },
    { label: L.nPixels, text: box.n_pixels.text },
    { label: L.nPairs, text: box.n_pairs.text },
  ];

  return (
    <div className={styles.stack}>
      {headline ? <Headline h={headline} meta={meta} state={state} /> : null}
      {row ? (
        <>
          <div className={overviewStyles.context}>
            <p className={overviewStyles.answer}>{row.answer_text}</p>
            <p className={overviewStyles.note}>{accuracyNote}</p>
            <ContextStrip items={contextItems} />
          </div>
          {row.partial_notice ? <p className={styles.callout}>{row.partial_notice}</p> : null}
        </>
      ) : (
        <p className={styles.callout}>{L.unavailable}</p>
      )}

      <div className={overviewStyles.grid}>
        <div className={overviewStyles.main}>
          {row ? (
            <div className={overviewStyles.kpis}>
              <Kpi label={L.purity} text={row.purity.text} ci={row.purity.ci_text || undefined} />
              <Kpi
                label={L.completeness}
                text={row.completeness.text}
                ci={row.completeness.ci_text || undefined}
              />
              <Kpi label={L.accuracy} text={row.accuracy.text} />
            </div>
          ) : null}
        <Panel id="ov-bars" title={L.barsTitle} subtitle={L.barsSub}>
          <div
            className={`${overviewStyles.charts} ${overviewStyles.scroll}`}
            role="region"
            aria-label={L.chartsRegion}
            tabIndex={0}
          >
            <HBarCI
              id="ov-purity"
              title={L.purityChart}
              rows={chartRows("purity")}
              partialTag={chartLabels.partialTag}
            />
            <HBarCI
              id="ov-completeness"
              title={L.completenessChart}
              rows={chartRows("completeness")}
              partialTag={chartLabels.partialTag}
            />
          </div>
          {partialRow?.partial_notice ? (
            <p className={overviewStyles.partialNote}>
              {chartLabels.partialTag}: {partialRow.partial_notice}
            </p>
          ) : null}
        </Panel>
        </div>
        <div className={overviewStyles.side}>
          {sky ? (
            <Panel id="ov-sky" title={L.skyTitle} subtitle={L.skySub}>
              <SkyThumb
                graticule={sky.graticule}
                regions={sky.regions}
                colorCap={sky.color_cap}
                aria={L.skyAria}
                caption={L.skyCaption}
              />
            </Panel>
          ) : null}
          <Panel id="ov-links" title={L.shortcutsTitle} subtitle={L.shortcutsSub}>
            <ul className={overviewStyles.shortcuts}>
              {shortcuts.map((r) => (
                <li key={r.href}>
                  <Link className={overviewStyles.shortcut} href={withState(r.href, state, meta.release)}>
                    {r.label}
                  </Link>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>

      {state.tech ? <OverviewTechnical technical={technical} /> : null}
    </div>
  );
}

function WithUrlState(props: Props) {
  return <Body props={props} state={readState(useSearchParams(), props.meta)} />;
}

export function OverviewView(props: Props) {
  return (
    <Suspense fallback={<Body props={props} state={props.meta.default_state} />}>
      <WithUrlState {...props} />
    </Suspense>
  );
}
