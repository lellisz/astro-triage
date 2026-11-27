import type { ReactNode } from "react";
import { DotCI, type DotCIRow } from "@/components/charts/DotCI";
import { PurityRange, type PurityRangeRow } from "@/components/charts/PurityRange";
import { DataTable, type DataColumn } from "@/components/kit/DataTable";
import { Panel } from "@/components/kit/Panel";
import { Tabs, type TabItem } from "@/components/kit/Tabs";
import { compareLabels as L } from "@/content/labels";
import type { Compare } from "@/lib/schema";
import cmp from "./compare.module.css";
import styles from "./page.module.css";

type Ok = Extract<Compare, { status: "ok" }>;
type Unavailable = Exclude<Compare, { status: "ok" }>;

const MARKERS = ["circle", "square", "diamond"] as const;
const markerAt = (i: number) => MARKERS[i % MARKERS.length];

const verdictColumns: readonly DataColumn[] = [
  { key: "net", label: L.colNet },
  { key: "word", label: L.colVerdict },
  { key: "diff", label: L.colDiff, numeric: true },
  { key: "ci", label: L.colCi },
  { key: "power", label: L.colPower },
];

const descriptiveColumns: readonly DataColumn[] = [
  { key: "pair", label: L.colPair },
  { key: "role", label: L.colRole },
  { key: "cls", label: L.colClass },
  { key: "diff", label: L.colDiff, numeric: true },
  { key: "ci", label: L.colCi },
];

function Scroller({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className={cmp.scroll} role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  );
}

function indexer(): (name: string) => number {
  const seen: string[] = [];
  return (name) => {
    const at = seen.indexOf(name);
    if (at >= 0) return at;
    seen.push(name);
    return seen.length - 1;
  };
}

function Models({ c }: { c: Ok }) {
  const ordered = [...c.purity].sort((a, b) => Number(b.reference) - Number(a.reference));
  return (
    <Panel id="cmp-models" title={L.modelsTitle} subtitle={L.modelsSub}>
      <ul className={cmp.cards}>
        {ordered.map((p) => {
          const pair = p.net === null ? undefined : c.pairs.find((x) => x.net === p.net);
          return (
            <li key={p.system} className={p.reference ? cmp.cardRef : cmp.card} data-system={p.system}>
              <h3 className={cmp.cardTitle}>{p.label}</h3>
              {p.reference ? <p className={cmp.tag}>{L.referenceTag}</p> : null}
              <p>{p.detail_text}</p>
              {p.reference ? (
                <p className={styles.muted}>{L.referenceNote}</p>
              ) : (
                <p className={cmp.verdictLine}>
                  <span className={styles.muted}>{L.pairVerdict}: </span>
                  <strong>{pair ? pair.text : L.pairMissing}</strong>
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}

function Verdicts({ c }: { c: Ok }) {
  const classes: { cls: string; label: string }[] = [];
  for (const g of c.grid) {
    if (!classes.some((k) => k.cls === g.cls)) classes.push({ cls: g.cls, label: g.cls_label });
  }
  return (
    <>
      <h3 className={cmp.subTitle}>{L.pairs}</h3>
      <ul className={cmp.pairs}>
        {c.pairs.map((pair) => (
          <li key={pair.net} className={cmp.pair}>
            <strong>{pair.net}</strong>
            <span>{pair.text}</span>
          </li>
        ))}
      </ul>
      <h3 className={cmp.subTitle}>{L.byClass}</h3>
      {classes.map((k) => (
        <DataTable
          key={k.cls}
          caption={k.label}
          columns={verdictColumns}
          rows={c.grid
            .filter((g) => g.cls === k.cls)
            .map((g) => ({
              net: g.net,
              word: g.word,
              diff: g.diff.text,
              ci: g.diff.ci_text,
              power: g.underpowered ? L.underpowered : L.powerOk,
            }))}
        />
      ))}
      {c.power_note ? <p className={styles.callout}>{c.power_note}</p> : null}
    </>
  );
}

function Purity({ c }: { c: Ok }) {
  const rows: PurityRangeRow[] = c.purity.map((p, i) => ({
    key: p.system,
    label: p.label,
    low: p.low,
    median: p.median,
    high: p.high,
    detail: p.detail_text,
    marker: markerAt(i),
  }));
  return (
    <>
      <h3 className={cmp.subTitle}>{L.absolutePurity}</h3>
      <p className={styles.muted}>{c.fixed_point.text}</p>
      <p className={styles.callout}>
        {L.purityRange} <strong>({L.notCi})</strong>
      </p>
      <Scroller label={L.purityRegion}>
        <PurityRange id="cmp-purity-chart" title={L.purityChart} rows={rows} />
      </Scroller>
      <ul className={cmp.details}>
        {c.purity.map((p) => (
          <li key={p.system}>
            <strong>{p.label}</strong>: {p.detail_text}
          </li>
        ))}
      </ul>
    </>
  );
}

function Diffs({ c }: { c: Ok }) {
  const netIndex = indexer();
  const decision: DotCIRow[] = c.grid.map((g) => {
    const i = netIndex(g.net);
    return {
      key: `g-${g.net}-${g.cls}`,
      label: `${g.net} · ${g.cls_label}`,
      est: g.diff,
      marker: markerAt(i),
      dashed: i % 2 === 1,
    };
  });
  return (
    <>
      <h3 className={cmp.subTitle}>{L.diffsTitle}</h3>
      <p className={styles.muted}>{L.diffsHint}</p>
      <Scroller label={L.diffsRegion}>
        <DotCI id="cmp-decision" title={L.decisionChart} rows={decision} zeroLine />
      </Scroller>
      {c.descriptive.length > 0 ? (
        <DataTable
          caption={L.descriptiveChart}
          columns={descriptiveColumns}
          rows={c.descriptive.map((d) => ({
            pair: d.pair,
            role: d.role,
            cls: d.cls,
            diff: d.diff.text,
            ci: d.diff.ci_text,
          }))}
        />
      ) : (
        <p className={styles.muted}>{L.noDescriptive}</p>
      )}
    </>
  );
}

function protocolText(c: Ok): string {
  if (c.protocol_accepted === null) return L.protocolUnknown;
  return c.protocol_accepted ? L.protocolAccepted : L.protocolProposed;
}

function Notes({ c }: { c: Ok }) {
  return (
    <>
      <h3 className={cmp.subTitle}>{L.methodTitle}</h3>
      <h4 className={cmp.minorTitle}>{L.rules}</h4>
      <ul className={cmp.list}>
        {c.rule_lines.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      <h4 className={cmp.minorTitle}>{L.reasons}</h4>
      <p>{c.reasons_text}</p>
      <ul className={cmp.list}>
        <li>{protocolText(c)}</li>
        <li>{c.freeze_exists ? L.freezeYes : L.freezeNo}</li>
      </ul>
      {c.limitations && c.limitations.length > 0 ? (
        <details className={cmp.disclosure}>
          <summary className={cmp.summary}>{L.limitations}</summary>
          <ul className={cmp.list}>
            {c.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </details>
      ) : (
        <p className={styles.muted}>{L.limitationsNone}</p>
      )}
    </>
  );
}

function Available({ c }: { c: Ok }) {
  const tabs: TabItem[] = [
    { key: "vereditos", label: L.tabVerdicts, content: <Verdicts c={c} /> },
    { key: "pureza", label: L.tabPurity, content: <Purity c={c} /> },
    { key: "diferencas", label: L.tabDiffs, content: <Diffs c={c} /> },
    { key: "metodo", label: L.tabMethod, content: <Notes c={c} /> },
  ];
  return (
    <div className={styles.stack}>
      <Panel id="cmp-result" title={L.result}>
        <p className={styles.answer}>{c.answer_text}</p>
        <p>{protocolText(c)}</p>
        <p className={styles.muted}>{c.fixed_point.text}</p>
      </Panel>
      <Models c={c} />
      <Tabs label={L.tabsLabel} tabs={tabs} />
    </div>
  );
}

function Missing({ c }: { c: Unavailable }) {
  return (
    <Panel id="cmp-unavailable" title={L.unavailable}>
      <p className={styles.muted}>{L.unavailableHint}</p>
      {c.error_text ? (
        <p className={styles.callout} role="status">
          {c.error_text}
        </p>
      ) : null}
    </Panel>
  );
}

export function CompareView({ compare }: { compare: Compare }) {
  return compare.status === "ok" ? <Available c={compare} /> : <Missing c={compare} />;
}
