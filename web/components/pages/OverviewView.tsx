"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import type { Est, Meta, Num, Overview } from "@/lib/schema";
import { readState, type UiState } from "@/lib/ui-state";
import styles from "./page.module.css";

type Props = {
  meta: Meta;
  rows: Overview["rows"];
  accuracyNote: string;
  dataBox: Overview["technical"]["data_box"];
};

const L = {
  purity: "Pureza de quasar",
  completeness: "Completude de quasar",
  accuracy: "Acurácia geral",
  unavailable: "Esta combinação de regra e prior não está disponível nesta versão dos dados.",
  barsTitle: "Pureza e completude nas condições avaliadas",
  barsSub: "Comparação da regra e dos priors de referência",
  barsPlaceholder: "Gráfico de barras com intervalos: será desenhado na próxima fase.",
  gaiaTitle: "Distribuição no céu",
  gaiaSub: "Referência astronômica real; não é saída do classificador",
  gaiaPlaceholder: "Mapa Gaia de referência: sem imagem nesta fase.",
  techTitle: "Detalhes técnicos",
  techSub: "Resumo do conjunto de teste",
  nPairs: "Pares cruzados",
  nTest: "Objetos no teste",
  nErrors: "Erros",
  nPixels: "Regiões do céu",
  accuracyBox: "Acurácia",
  qsoHits: "Quasares encontrados",
  qsoTotal: "Quasares de referência",
};

function Kpi({ label, text, ci }: { label: string; text: string; ci?: string }) {
  return (
    <div className={styles.kpi}>
      <div className={styles.kpiLabel}>{label}</div>
      <div className={styles.kpiValue}>{text}</div>
      {ci ? <div className={styles.kpiCi}>{ci}</div> : null}
    </div>
  );
}

const estText = (e: Est) => (e.point === null ? "n/a" : e.text);
const estCi = (e: Est) => (e.point === null ? undefined : e.ci_text || undefined);
const numText = (n: Num) => (n.value === null ? "n/a" : n.text);

function Body({ props, state }: { props: Props; state: UiState }) {
  const { rows, accuracyNote, dataBox } = props;
  const row = rows.find((r) => r.rule === state.rule && r.prior === state.prior);
  const box: [string, Num][] = [
    [L.nPairs, dataBox.n_pairs],
    [L.nTest, dataBox.n_test],
    [L.nErrors, dataBox.n_errors],
    [L.nPixels, dataBox.n_pixels],
    [L.accuracyBox, dataBox.accuracy],
    [L.qsoHits, dataBox.qso_hits],
    [L.qsoTotal, dataBox.qso_total],
  ];

  return (
    <div className={styles.stack}>
      {row ? (
        <>
          <p className={styles.answer}>{row.answer_text}</p>
          <div className={styles.kpis}>
            <Kpi label={L.purity} text={estText(row.purity)} ci={estCi(row.purity)} />
            <Kpi label={L.completeness} text={estText(row.completeness)} ci={estCi(row.completeness)} />
            <Kpi label={L.accuracy} text={numText(row.accuracy)} />
          </div>
          {row.partial_notice ? <p className={styles.callout}>{row.partial_notice}</p> : null}
        </>
      ) : (
        <p className={styles.callout}>{L.unavailable}</p>
      )}

      <div className={styles.split}>
        <section className={styles.panel} aria-labelledby="ov-bars">
          <h2 id="ov-bars" className={styles.panelTitle}>{L.barsTitle}</h2>
          <p className={styles.panelSub}>{L.barsSub}</p>
          <div className={styles.placeholder}>{L.barsPlaceholder}</div>
        </section>
        <section className={styles.panel} aria-labelledby="ov-gaia">
          <h2 id="ov-gaia" className={styles.panelTitle}>{L.gaiaTitle}</h2>
          <p className={styles.panelSub}>{L.gaiaSub}</p>
          <div className={styles.placeholder}>{L.gaiaPlaceholder}</div>
        </section>
      </div>

      <p className={styles.muted}>{accuracyNote}</p>

      {state.tech ? (
        <section className={styles.panel} aria-labelledby="ov-tech">
          <h2 id="ov-tech" className={styles.panelTitle}>{L.techTitle}</h2>
          <p className={styles.panelSub}>{L.techSub}</p>
          <dl className={styles.dl}>
            {box.map(([label, n]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{numText(n)}</dd>
              </div>
            ))}
          </dl>
        </section>
      ) : null}
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
