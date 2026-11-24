"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import type { Galaxies, Meta } from "@/lib/schema";
import { galaxiesLabels as L } from "@/content/labels";
import { readState } from "@/lib/ui-state";
import styles from "./page.module.css";

type Props = {
  meta: Meta;
  verdict: Galaxies["verdict"];
  fixedPointText: string;
  crossmatchNote: string | null;
  missingNote: string;
};

function Panel({ id, title, sub }: { id: string; title: string; sub: string }) {
  return (
    <section className={styles.panel} aria-labelledby={id}>
      <h2 id={id} className={styles.panelTitle}>{title}</h2>
      <p className={styles.panelSub}>{sub}</p>
      <div className={styles.placeholder}>{L.chartPlaceholder}</div>
    </section>
  );
}

function Body({ props, tech }: { props: Props; tech: boolean }) {
  const { verdict, fixedPointText, crossmatchNote, missingNote } = props;
  return (
    <div className={styles.stack}>
      <p className={styles.answer}>{verdict.text}</p>
      <p className={styles.muted}>{verdict.scope_text}</p>
      <p className={styles.muted}>{fixedPointText}</p>
      {crossmatchNote ? <p className={styles.callout}>{crossmatchNote}</p> : null}

      <div className={styles.split}>
        <Panel id="gal-purity" title={L.purityTitle} sub={L.puritySub} />
        <Panel id="gal-completeness" title={L.completenessTitle} sub={L.completenessSub} />
      </div>
      <Panel id="gal-networks" title={L.networksTitle} sub={L.networksSub} />
      <Panel id="gal-errors" title={L.errorsTitle} sub={L.errorsSub} />

      {tech ? (
        <section className={styles.panel} aria-labelledby="gal-tech">
          <h2 id="gal-tech" className={styles.panelTitle}>{L.techTitle}</h2>
          <p className={styles.panelSub}>{missingNote}</p>
          <div className={styles.placeholder}>{L.chartPlaceholder}</div>
        </section>
      ) : null}
    </div>
  );
}

function WithUrlState(props: Props) {
  return <Body props={props} tech={readState(useSearchParams(), props.meta).tech} />;
}

export function GalaxiesView(props: Props) {
  return (
    <Suspense fallback={<Body props={props} tech={props.meta.default_state.tech} />}>
      <WithUrlState {...props} />
    </Suspense>
  );
}
