import { PageHeader } from "@/components/shell/PageHeader";
import { pages } from "@/content/labels";
import { loadPage } from "@/lib/data";
import type { Compare } from "@/lib/schema";
import styles from "./page.module.css";

// Rotulos que o labels.ts ainda nao tem; sem algarismos de proposito.
const text = {
  unavailable: "Comparação não disponível.",
  unavailableHint: "Não há resultado de aprendizado profundo para esta versão dos dados.",
  result: "Resultado",
  pairs: "Veredito por par",
  verdictGrid: "Grade de veredito",
  verdictGridSoon: "Em construção: a grade com o veredito por classe e por rede aparecerá aqui.",
  absolutePurity: "Pureza absoluta",
  absolutePuritySoon:
    "Em construção: a pureza de cada sistema aparecerá aqui. O intervalo de mínimo a máximo entre sementes não é IC.",
  notCi: "não é IC",
  rules: "Regra de decisão",
  descriptive: "Diferenças descritivas",
  limitations: "Limitações",
};

function Unavailable({ compare }: { compare: Exclude<Compare, { status: "ok" }> }) {
  return (
    <section className={styles.panel} aria-labelledby="indisponivel">
      <h2 id="indisponivel" className={styles.panelTitle}>
        {text.unavailable}
      </h2>
      <p className={styles.muted}>{text.unavailableHint}</p>
      {compare.status === "error" && compare.error_text ? (
        <p className={styles.notice} role="status">
          {compare.error_text}
        </p>
      ) : null}
    </section>
  );
}

function Available({ compare }: { compare: Extract<Compare, { status: "ok" }> }) {
  return (
    <>
      <section className={styles.panel} aria-labelledby="resultado">
        <h2 id="resultado" className={styles.panelTitle}>
          {text.result}
        </h2>
        <p className={styles.answer}>{compare.answer_text}</p>
        <p className={styles.muted}>{compare.fixed_point.text}</p>
        <p className={styles.notice}>{compare.reasons_text}</p>
      </section>

      <section className={styles.panel} aria-labelledby="pares">
        <h2 id="pares" className={styles.panelTitle}>
          {text.pairs}
        </h2>
        <ul className={styles.pairs}>
          {compare.pairs.map((pair) => (
            <li key={pair.net} className={styles.pair}>
              <span className={styles.pairNet}>{pair.net}</span>
              <span>{pair.text}</span>
            </li>
          ))}
        </ul>
      </section>

      <div className={styles.placeholders}>
        <section className={styles.placeholder} aria-labelledby="grade">
          <h2 id="grade" className={styles.placeholderTitle}>
            {text.verdictGrid}
          </h2>
          <p>{text.verdictGridSoon}</p>
        </section>
        <section className={styles.placeholder} aria-labelledby="pureza">
          <h2 id="pureza" className={styles.placeholderTitle}>
            {text.absolutePurity}
          </h2>
          <p>{text.absolutePuritySoon}</p>
          <p>
            <small>{text.notCi}</small>
          </p>
        </section>
      </div>

      <section className={styles.panel} aria-labelledby="regras">
        <h2 id="regras" className={styles.panelTitle}>
          {text.rules}
        </h2>
        <ul className={styles.list}>
          {compare.rule_lines.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </section>

      <details className={styles.details}>
        <summary className={styles.summary}>{text.descriptive}</summary>
        <div className={styles.detailsBody}>
          <ul className={styles.list}>
            {compare.descriptive.map((row) => (
              <li key={`${row.pair}|${row.cls}`}>
                {row.pair} · {row.role} · {row.cls}: {row.text}
              </li>
            ))}
          </ul>
        </div>
      </details>

      {compare.limitations && compare.limitations.length > 0 ? (
        <details className={styles.details}>
          <summary className={styles.summary}>{text.limitations}</summary>
          <div className={styles.detailsBody}>
            <ul className={styles.list}>
              {compare.limitations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </details>
      ) : null}
    </>
  );
}

export default function CompareDlPage() {
  const meta = loadPage("meta");
  const compare = loadPage("compare");
  const labels = pages.compare;

  return (
    <div className={styles.page}>
      <PageHeader
        eyebrow={labels.eyebrow}
        title={labels.title}
        lead={labels.lead}
        showStateChip={false}
        meta={meta}
      />
      {compare.status === "ok" ? <Available compare={compare} /> : <Unavailable compare={compare} />}
    </div>
  );
}
