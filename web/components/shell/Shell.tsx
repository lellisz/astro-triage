import type { ReactNode } from "react";
import { acessibilidade, footer, siteName, siteTagline } from "@/content/labels";
import type { Meta } from "@/lib/schema";
import { StateControls } from "./StateControls";
import { TopNav } from "./TopNav";
import styles from "./Shell.module.css";

type Props = { meta: Meta; children: ReactNode; hideRuleAndPrior?: boolean };

export function Shell({ meta, children, hideRuleAndPrior }: Props) {
  return (
    <div className={styles.app}>
      <a className={styles.skip} href="#conteudo">
        {acessibilidade.skipToContent}
      </a>
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <span className={styles.name}>{siteName}</span>
          <span className={styles.tagline}>{siteTagline}</span>
        </div>
        <TopNav meta={meta} />
      </header>
      <StateControls meta={meta} hideRuleAndPrior={hideRuleAndPrior} />
      <main id="conteudo" className={styles.main}>
        {children}
      </main>
      <footer className={styles.footer}>
        <span>
          {footer.release}: {meta.release}
        </span>
        <span>{footer.note}</span>
      </footer>
    </div>
  );
}
