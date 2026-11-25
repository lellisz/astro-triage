import { Kpi } from "@/components/kit/Kpi";
import { MethodView, SimpleNotice } from "@/components/pages/MethodView";
import { PageHeader } from "@/components/shell/PageHeader";
import { methodLabels as text, pages } from "@/content/labels";
import { loadPage } from "@/lib/data";
import type { Cls, Method, Num } from "@/lib/schema";
import styles from "@/components/pages/method.module.css";

const FACT_KEYS = [
  "n_test",
  "n_pixels",
  "prior_qso",
  "min_objects",
  "min_pixels",
  "n_boot",
  "completeness",
] as const satisfies readonly (keyof Method["facts"])[];

const CLASS_KEYS = ["STAR", "GALAXY", "QSO"] as const satisfies readonly Cls[];

function isHttps(url: string): boolean {
  try {
    return new URL(url).protocol === "https:";
  } catch {
    return false;
  }
}

const shown = (n: Num): string => (n.value === null ? text.notAvailable : n.text);

export default function MethodPage() {
  const meta = loadPage("meta");
  const method = loadPage("method");
  const labels = pages.method;
  const links = method.links.filter((link) => isHttps(link.url));

  return (
    <div className={styles.page}>
      <PageHeader
        eyebrow={labels.eyebrow}
        title={labels.title}
        lead={method.lead}
        showStateChip
        meta={meta}
      />

      <section className={styles.panel} aria-labelledby="como">
        <h2 id="como" className={styles.panelTitle}>
          {text.how}
        </h2>
        <p>{method.method_text}</p>
        <p className={styles.muted}>{method.mix_text}</p>
        <SimpleNotice meta={meta} text={method.simple} />
      </section>

      <section className={styles.panel} aria-labelledby="fatos">
        <h2 id="fatos" className={styles.panelTitle}>
          {text.factsTitle}
        </h2>
        <div className={styles.kpis}>
          {FACT_KEYS.map((key) => (
            <Kpi key={key} label={text.facts[key]} text={shown(method.facts[key])} />
          ))}
        </div>
      </section>

      <section className={styles.panel} aria-labelledby="mistura">
        <h2 id="mistura" className={styles.panelTitle}>
          {text.mixTitle}
        </h2>
        <div className={styles.kpis}>
          {CLASS_KEYS.map((cls) => (
            <Kpi key={cls} label={text.classes[cls]} text={shown(method.mix[cls])} />
          ))}
        </div>
      </section>

      {method.three_limits.length > 0 ? (
        <section className={styles.panel} aria-labelledby="tres-limites">
          <h2 id="tres-limites" className={styles.panelTitle}>
            {text.limits}
          </h2>
          <ul className={styles.list}>
            {method.three_limits.map((item, i) => (
              <li key={`${i}-${item}`}>{item}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <MethodView
        meta={meta}
        technical={{
          reading: method.reading,
          usage: method.usage,
          rule_lines: method.rule_lines,
          limitations: method.limitations,
        }}
      />

      <section className={styles.panel} aria-labelledby="glossario">
        <h2 id="glossario" className={styles.panelTitle}>
          {text.glossary}
        </h2>
        <dl className={styles.glossary}>
          {method.glossary.map((entry) => (
            <div key={entry.term} className={styles.term}>
              <dt className={styles.termName}>{entry.term}</dt>
              {entry.definition ? <dd className={styles.termDef}>{entry.definition}</dd> : null}
            </div>
          ))}
        </dl>
      </section>

      {links.length > 0 ? (
        <section className={styles.panel} aria-labelledby="links">
          <h2 id="links" className={styles.panelTitle}>
            {text.links}
          </h2>
          <ul className={styles.links}>
            {links.map((link) => (
              <li key={link.url} className={styles.linkItem}>
                <a className={styles.link} href={link.url} target="_blank" rel="noopener noreferrer">
                  {link.label}
                </a>
                <span className={styles.muted}>{link.note}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className={styles.panel} aria-labelledby="fontes">
        <h2 id="fontes" className={styles.panelTitle}>
          {text.sources}
        </h2>
        <ul className={styles.list}>
          {method.sources.map((source) => (
            <li key={source}>{source}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
