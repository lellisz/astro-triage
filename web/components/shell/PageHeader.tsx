import type { Meta } from "@/lib/schema";
import { ProvisionalSeal } from "./ProvisionalSeal";
import { StateChip } from "./StateControls";
import styles from "./PageHeader.module.css";

type Props = {
  eyebrow: string;
  title: string;
  lead?: string;
  showStateChip: boolean;
  meta: Meta;
};

export function PageHeader({ eyebrow, title, lead, showStateChip, meta }: Props) {
  return (
    <header className={styles.head}>
      <div className={styles.text}>
        <p className={styles.eyebrow}>{eyebrow}</p>
        <h1 className={styles.title}>{title}</h1>
        {lead ? <p className={styles.lead}>{lead}</p> : null}
      </div>
      <div className={styles.meta}>
        <ProvisionalSeal text={meta.seal_text} />
        {showStateChip ? <StateChip meta={meta} /> : null}
      </div>
    </header>
  );
}
