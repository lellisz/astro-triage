import { PageHeader } from "@/components/shell/PageHeader";
import { methodLabels as text, pages } from "@/content/labels";
import { loadPage } from "@/lib/data";
import styles from "@/components/pages/method.module.css";

// Página própria do glossário, para quem não conhece as siglas. Os termos vêm do mesmo
// export da página Método (method.glossary), então as duas listas nunca divergem.
export default function GlossaryPage() {
  const meta = loadPage("meta");
  const method = loadPage("method");
  const labels = pages.glossary;

  return (
    <div className={styles.page}>
      <PageHeader
        eyebrow={labels.eyebrow}
        title={labels.title}
        lead={labels.lead}
        showStateChip={false}
        meta={meta}
      />
      <section className={styles.panel} aria-label={labels.title}>
        <dl className={styles.glossary}>
          {method.glossary.map((entry) => (
            <div key={entry.term} id={slug(entry.term)} className={styles.term}>
              <dt className={styles.termName}>{entry.term}</dt>
              <dd className={styles.termDef}>{entry.definition ?? text.notAvailable}</dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
}

function slug(term: string): string {
  return term
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}
