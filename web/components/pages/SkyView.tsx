"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Kpi } from "@/components/kit/Kpi";
import { Panel } from "@/components/kit/Panel";
import { DataTable, type DataColumn } from "@/components/kit/DataTable";
import { MapLegend } from "@/components/sky/MapLegend";
import { MollweideMap, isSaturated, type MapMode } from "@/components/sky/MollweideMap";
import { skyLabels as L, skyMapLabels } from "@/content/labels";
import type { Meta, Sky } from "@/lib/schema";
import { ptInt } from "@/lib/fmt";
import { readState } from "@/lib/ui-state";
import skyStyles from "./sky.module.css";
import styles from "./page.module.css";

type Props = { meta: Meta; sky: Sky };

const topColumns: readonly DataColumn[] = [
  { key: "position", label: L.colPosition },
  { key: "n", label: L.colObjects, numeric: true },
  { key: "errors", label: L.colErrors, numeric: true },
  { key: "rate", label: L.colRate, numeric: true },
  { key: "ci", label: L.colRateCi },
];

const bandColumns: readonly DataColumn[] = [
  { key: "band", label: L.colBand },
  { key: "pixels", label: L.colPixels, numeric: true },
  { key: "n", label: L.colObjects, numeric: true },
  { key: "errors", label: L.colErrors, numeric: true },
  { key: "rate", label: L.colRate, numeric: true },
];

function Body({ props, tech }: { props: Props; tech: boolean }) {
  const { cards, texts, regions, graticule, color_cap: colorCap, bands, top_regions: top } = props.sky;
  const [mode, setMode] = useState<MapMode>("rate");
  const [hide, setHide] = useState(false);
  const hideBelow = tech && hide;
  const cap = mode === "rate" ? colorCap.rate : colorCap.count;
  const howText =
    mode === "count"
      ? hideBelow
        ? texts.how_count_hidden
        : texts.how_count
      : hideBelow
        ? texts.how_rate_hidden
        : texts.how_rate;
  const summary = `${texts.lead} ${L.below}: ${cards.n_below.text}. ${L.maxRate}: ${cards.max_rate.text}.`;

  const topRows = top.map((r) => ({
    position: r.position_text,
    n: ptInt(r.n),
    errors: ptInt(r.errors),
    rate: r.rate.text,
    ci: r.rate.ci_text,
  }));
  const bandRows = bands.map((b) => ({
    band: b.label,
    pixels: ptInt(b.n_pixels),
    n: ptInt(b.n),
    errors: ptInt(b.errors),
    rate: b.rate.text,
  }));

  return (
    <div className={styles.stack}>
      <div className={skyStyles.context}>
        <p className={skyStyles.note}>{texts.map_lead}</p>
      </div>
      <div className={skyStyles.kpis}>
        <Kpi label={L.regions} text={cards.n_regions.text} />
        <Kpi label={L.objects} text={cards.n_objects.text} />
        <Kpi label={L.below} text={cards.n_below.text} />
        <Kpi label={L.maxRate} text={cards.max_rate.text} ci={cards.max_rate.ci_text || undefined} />
      </div>
      {cards.diverge_note ? <p className={styles.callout}>{cards.diverge_note}</p> : null}

      <Panel id="sky-map" title={L.mapTitle}>
        <div className={skyStyles.toolbar} role="group" aria-label={L.toolbarAria}>
          <div className={skyStyles.modes} role="group" aria-label={L.modeGroupAria}>
            <button
              type="button"
              className={skyStyles.mode}
              aria-pressed={mode === "rate"}
              onClick={() => setMode("rate")}
            >
              {L.modeRateBtn}
            </button>
            <button
              type="button"
              className={skyStyles.mode}
              aria-pressed={mode === "count"}
              onClick={() => setMode("count")}
            >
              {L.modeCountBtn}
            </button>
          </div>
          {tech ? (
            <label className={skyStyles.check}>
              <input type="checkbox" checked={hide} onChange={(e) => setHide(e.target.checked)} />
              {L.hideBelow}
            </label>
          ) : null}
        </div>
        <MollweideMap
          regions={regions}
          graticule={graticule}
          colorCap={colorCap}
          bands={bands}
          mode={mode}
          hideBelow={hideBelow}
          labels={skyMapLabels}
          summary={summary}
        />
        <MapLegend
          mode={mode}
          colorCap={colorCap}
          saturated={isSaturated(regions, mode, cap, hideBelow)}
          howText={howText}
          hideBelowCount={hideBelow && mode === "count"}
          labels={skyMapLabels}
        />
        <div className={skyStyles.notes}>
          <p>{texts.position_note}</p>
          <p>{texts.neighbour_note}</p>
          <p>{texts.spread_note}</p>
        </div>
      </Panel>

      {tech ? (
        <div className={skyStyles.tables}>
          <Panel id="sky-top" title={L.topTitle}>
            <DataTable caption={L.topTitle} labelledBy="sky-top" columns={topColumns} rows={topRows} />
          </Panel>
          <Panel id="sky-bands" title={L.bandsTitle}>
            <DataTable caption={L.bandsTitle} labelledBy="sky-bands" columns={bandColumns} rows={bandRows} />
          </Panel>
        </div>
      ) : null}
    </div>
  );
}

function WithUrlState(props: Props) {
  return <Body props={props} tech={readState(useSearchParams(), props.meta).tech} />;
}

export function SkyView(props: Props) {
  return (
    <Suspense fallback={<Body props={props} tech={props.meta.default_state.tech} />}>
      <WithUrlState {...props} />
    </Suspense>
  );
}
