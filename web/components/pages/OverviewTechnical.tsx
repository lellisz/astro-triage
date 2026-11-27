import Link from "next/link";
import { ConfidenceTable } from "@/components/charts/ConfidenceTable";
import { HBar } from "@/components/charts/HBar";
import { DotCI, type DotCIRow } from "@/components/charts/DotCI";
import { DataTable, type DataColumn } from "@/components/kit/DataTable";
import { chartsHref, overviewLabels as L } from "@/content/labels";
import type { Cls, Num, Overview } from "@/lib/schema";
import overviewStyles from "./overview.module.css";
import styles from "./page.module.css";

const CLASS_ORDER: readonly Cls[] = ["STAR", "GALAXY", "QSO"];

const errorColumns: readonly DataColumn[] = [
  { key: "true", label: L.colTrue },
  { key: "pred", label: L.colPred },
  { key: "n", label: L.colN, numeric: true },
  { key: "share", label: L.colShare, numeric: true },
];

const f1Columns: readonly DataColumn[] = [
  { key: "model", label: L.colModel },
  { key: "score", label: L.colScore, numeric: true },
  { key: "ci", label: L.colCi },
];

function Facts({ items }: { items: [string, Num][] }) {
  return (
    <dl className={styles.dl}>
      {items.map(([label, n]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{n.text}</dd>
        </div>
      ))}
    </dl>
  );
}

export function OverviewTechnical({ technical }: { technical: Overview["technical"] }) {
  const { data_box: box, validation, error_directions: directions, triage, dsc, models, confidence_hist: hist } = technical;
  const boxItems: [string, Num][] = [
    [L.nPairs, box.n_pairs],
    [L.nTest, box.n_test],
    [L.nErrors, box.n_errors],
    [L.nPixels, box.n_pixels],
    [L.accuracyBox, box.accuracy],
    [L.qsoHits, box.qso_hits],
    [L.qsoTotal, box.qso_total],
  ];
  const mixItems: [string, Num][] = CLASS_ORDER.map((c) => [L.classNames[c], box.class_mix[c]]);
  const errorRows = directions.map((d) => ({
    true: L.classNames[d.true],
    pred: L.classNames[d.pred],
    n: d.n.text,
    share: d.share.text,
  }));
  const f1Rows = dsc.macro_f1.map((m) => ({ model: m.label, score: m.est.text, ci: m.est.ci_text }));
  const names = new Map<string, string>([...models, ...dsc.macro_f1].map((m) => [m.model, m.label]));
  const diffRows: DotCIRow[] = dsc.paired_diffs.map((d) => ({
    key: `${d.a}-${d.b}`,
    label: `${names.get(d.a) ?? d.a} − ${names.get(d.b) ?? d.b}`,
    est: d.est,
  }));

  return (
    <section className={styles.panel} aria-labelledby="ov-tech">
      <h2 id="ov-tech" className={styles.panelTitle}>{L.techTitle}</h2>
      <p className={styles.panelSub}>{L.techSub}</p>
      <div className={overviewStyles.tech}>
        <div className={overviewStyles.techBlock}>
          <h3>{L.dataBoxTitle}</h3>
          <Facts items={boxItems} />
          <h3>{L.classMixTitle}</h3>
          <Facts items={mixItems} />
        </div>

        {validation ? (
          <div className={overviewStyles.techBlock}>
            <h3>{L.validationTitle}</h3>
            <Facts
              items={[
                [L.validationPurity, validation.purity],
                [L.validationCompleteness, validation.completeness],
              ]}
            />
            <p className={overviewStyles.note}>
              {L.validationPrior}: {validation.prior}
            </p>
          </div>
        ) : null}

        <div className={overviewStyles.techBlock}>
          <h3>{L.errorsTitle}</h3>
          <DataTable caption={L.errorsCaption} columns={errorColumns} rows={errorRows} />
        </div>

        {triage ? (
          <div className={overviewStyles.techBlock}>
            <h3>{L.triageTitle}</h3>
            <Facts
              items={[
                [L.triageThreshold, triage.threshold],
                [L.triageCoverage, triage.coverage],
                [L.triageAccuracyKept, triage.accuracy_kept],
                [L.triageSetAside, triage.n_set_aside],
                [L.triageErrorsSetAside, triage.errors_set_aside],
                [L.triageErrorsTotal, triage.errors_total],
              ]}
            />
          </div>
        ) : null}

        <div className={overviewStyles.techBlock}>
          <h3>{L.dscTitle}</h3>
          <p className={overviewStyles.note}>
            {L.dscPopulation}: {dsc.population_text} {L.dscRows}: {dsc.n_rows.text}.
          </p>
          <p className={overviewStyles.note}>{dsc.prior_text}</p>
          <DataTable caption={L.dscScoreTable} columns={f1Columns} rows={f1Rows} />
          <div
            className={`${overviewStyles.charts} ${overviewStyles.scroll}`}
            role="region"
            aria-label={L.dscRegion}
            tabIndex={0}
          >
            <DotCI id="ov-dsc-diff" title={L.dscDiffChart} rows={diffRows} zeroLine />
          </div>
        </div>

        {models.length > 0 ? (
          <div className={overviewStyles.techBlock}>
            <h3>{L.importanceTitle}</h3>
            <div
              className={`${overviewStyles.charts} ${overviewStyles.scroll}`}
              role="region"
              aria-label={L.importanceRegion}
              tabIndex={0}
            >
              {models.map((m) => (
                <div key={m.model}>
                  <HBar
                    id={`ov-imp-${m.model}`}
                    title={`${L.importanceChart}: ${m.label}`}
                    rows={m.importance.map((f) => ({ key: f.feature, label: f.feature, value: f.gain }))}
                  />
                  <p className={overviewStyles.note}>
                    {L.temperature}: {m.temperature.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className={overviewStyles.techBlock}>
          <h3>{L.calibrationTitle}</h3>
          <p>
            <Link href={chartsHref}>{L.calibrationLink}</Link>
          </p>
          {models.map((m) => (
            <p key={m.model} className={overviewStyles.note}>
              {L.temperature} ({m.label}): {m.temperature.text}
            </p>
          ))}
        </div>

        <div className={overviewStyles.techBlock}>
          <h3>{L.confidenceTitle}</h3>
          <div
            className={`${overviewStyles.charts} ${overviewStyles.scroll}`}
            role="region"
            aria-label={L.confidenceRegion}
            tabIndex={0}
          >
            <ConfidenceTable
              id="ov-confidence"
              caption={L.confidenceChart}
              bins={hist}
              bandLabel={L.confidenceBand}
              rightLabel={L.confidenceRight}
              wrongLabel={L.confidenceWrong}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
