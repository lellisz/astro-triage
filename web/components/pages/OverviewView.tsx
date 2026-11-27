"use client";

import Link from "next/link";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { MplFigure } from "@/components/charts/MplFigure";
import { ContextStrip } from "@/components/kit/ContextStrip";
import { Kpi } from "@/components/kit/Kpi";
import { Panel } from "@/components/kit/Panel";
import { SkyThumb } from "@/components/sky/SkyThumb";
import { OverviewTechnical } from "./OverviewTechnical";
import { chartLabels, overviewLabels as L, routes } from "@/content/labels";
import { quasarSummaryFigureId } from "@/lib/figure-ids";
import type { Figures } from "@/lib/figures";
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
  figures: Figures;
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
      {h.protocol_text ? <p className={o.caveat}>{h.protocol_text}</p> : null}
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
  const { meta, headline, rows, accuracyNote, technical, sky, figures } = props;
  const row = rows.find((r) => r.rule === state.rule && r.prior === state.prior);
  const partialRow = rows.find((r) => r.partial && r.partial_notice);
  // The static figures draw every row alike; name the chosen one with the same label as the SVG.
  const selected = meta.states.find((s) => s.rule === state.rule && s.prior === state.prior);
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
              <Kpi
                label={L.accuracy}
                text={
                  row.accuracy.value === null && row.accuracy_na_reason === "not_test_mix"
                    ? L.accuracyNA
                    : row.accuracy.text
                }
              />
            </div>
          ) : null}
        <Panel
          id="ov-bars"
          title={state.tech ? L.barsTitle : L.barsTitleSimple}
          subtitle={state.tech ? L.barsSub : L.barsSubSimple}
        >
          <div
            className={`${overviewStyles.charts} ${overviewStyles.scroll}`}
            role="region"
            aria-label={L.chartsRegion}
            tabIndex={0}
          >
            {state.tech ? (
              <>
                <MplFigure id="quasar-pureza" figures={figures} />
                <MplFigure id="quasar-completude" figures={figures} />
              </>
            ) : selected ? (
              <MplFigure id={quasarSummaryFigureId(selected.rule, selected.prior)} figures={figures} />
            ) : null}
          </div>
          {selected ? (
            <p className={overviewStyles.partialNote} data-selected-row="">
              {L.selectedRow}: <strong>{selected.label}</strong>
            </p>
          ) : null}
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

      {state.tech ? <OverviewTechnical technical={technical} figures={figures} /> : null}
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
