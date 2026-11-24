import { PageHeader } from "@/components/shell/PageHeader";
import { MethodView } from "@/components/pages/MethodView";
import { methodLabels as text, pages } from "@/content/labels";
import { loadPage } from "@/lib/data";
import styles from "@/components/pages/method.module.css";

function isHttps(url: string): boolean {
  try {
    return new URL(url).protocol === "https:";
  } catch {
    return false;
  }
}

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
        <p className={styles.notice}>{method.simple}</p>
      </section>

      <MethodView
        meta={meta}
        technical={{
          reading: method.reading,
          usage: method.usage,
          rule_lines: method.rule_lines,
          three_limits: method.three_limits,
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
