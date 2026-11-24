import { kitLabels } from "@/content/labels";
import styles from "./kit.module.css";

type NAProps = { text: string };

export function NA({ text }: NAProps) {
  return (
    <span className={styles.na} aria-label={kitLabels.notAvailable} role="img">
      {text}
    </span>
  );
}
