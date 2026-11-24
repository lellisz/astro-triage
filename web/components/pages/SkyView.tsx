"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import type { Meta, Sky } from "@/lib/schema";
import { readState } from "@/lib/ui-state";
import styles from "./page.module.css";

type Props = {
  meta: Meta;
  cards: Sky["cards"];
  texts: Pick<Sky["texts"], "map_lead" | "position_note" | "neighbour_note" | "spread_note">;
};

const L = {
  regions: "Regiões com dados",
  objects: "Objetos no teste",
  below: "Regiões abaixo do piso",
  maxRate: "Maior taxa de erro",
  mapTitle: "Mapa Mollweide do céu",
  mapPlaceholder: "Mapa interativo: será desenhado na próxima fase.",
  techTitle: "Notas técnicas do mapa",
  techPlaceholder: "Tabela de regiões e faixas de latitude: será montada na próxima fase.",
};

function Card({ label, text, ci }: { label: string; text: string; ci?: string }) {
  return (
    <div className={styles.kpi}>
      <div className={styles.kpiLabel}>{label}</div>
      <div className={styles.kpiValue}>{text}</div>
      {ci ? <div className={styles.kpiCi}>{ci}</div> : null}
    </div>
  );
}

function Body({ props, tech }: { props: Props; tech: boolean }) {
  const { cards, texts } = props;
  return (
    <div className={styles.stack}>
      <div className={styles.kpis}>
        <Card label={L.regions} text={cards.n_regions.text} />
        <Card label={L.objects} text={cards.n_objects.text} />
        <Card label={L.below} text={cards.n_below.text} />
        <Card label={L.maxRate} text={cards.max_rate.text} ci={cards.max_rate.ci_text || undefined} />
      </div>
      {cards.diverge_note ? <p className={styles.callout}>{cards.diverge_note}</p> : null}

      <section className={styles.panel} aria-labelledby="sky-map">
        <h2 id="sky-map" className={styles.panelTitle}>{L.mapTitle}</h2>
        <p className={styles.panelSub}>{texts.map_lead}</p>
        <div className={`${styles.placeholder} ${styles.mapPlaceholder}`}>{L.mapPlaceholder}</div>
      </section>

      <div className={styles.stack}>
        <p className={styles.muted}>{texts.position_note}</p>
        <p className={styles.muted}>{texts.neighbour_note}</p>
        <p className={styles.muted}>{texts.spread_note}</p>
      </div>

      {tech ? (
        <section className={styles.panel} aria-labelledby="sky-tech">
          <h2 id="sky-tech" className={styles.panelTitle}>{L.techTitle}</h2>
          <div className={styles.placeholder}>{L.techPlaceholder}</div>
        </section>
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
