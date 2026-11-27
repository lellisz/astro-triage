import type { Meta } from "@/lib/schema";
import { ModeText } from "./ModeText";
import { ProvisionalSeal } from "./ProvisionalSeal";
import { StateChip } from "./StateControls";
import styles from "./PageHeader.module.css";

type Props = {
  eyebrow: string;
  title: string;
  lead?: string;
  leadSimple?: string;
  showStateChip: boolean;
  meta: Meta;
};

export function PageHeader({ eyebrow, title, lead, leadSimple, showStateChip, meta }: Props) {
  return (
    <header className={styles.head}>
      <div className={styles.text}>
        <p className={styles.eyebrow}>{eyebrow}</p>
        <h1 className={styles.title}>{title}</h1>
        {lead ? (
          <p className={styles.lead}>
            {leadSimple ? <ModeText meta={meta} tech={lead} simple={leadSimple} /> : lead}
          </p>
        ) : null}
      </div>
      <div className={styles.meta}>
        <ProvisionalSeal text={meta.seal_text} />
        {showStateChip ? <StateChip meta={meta} /> : null}
      </div>
    </header>
  );
}
