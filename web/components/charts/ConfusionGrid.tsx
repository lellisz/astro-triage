import { NA } from "@/components/kit/NA";
import type { Cls, Est, Num } from "@/lib/schema";
import styles from "./graficos.module.css";

type ConfusionCell = { pred: Cls; n: Num; share: Est; diagonal: boolean; below_floor: boolean };
type ConfusionRow = { true: Cls; n: Num; cells: ConfusionCell[] };

type ConfusionGridProps = {
  caption: string;
  classes: { cls: Cls; label: string }[];
  rows: ConfusionRow[];
  showCi: boolean;
  trueLabel: string;
  predLabel: string;
};

// Teto da opacidade do cinza: o texto da celula precisa manter contraste.
const MAX_SHADE = 0.35;

const shadeOf = (share: number): number => Math.min(Math.max(share, 0), 1) * MAX_SHADE;

export function ConfusionGrid({ caption, classes, rows, showCi, trueLabel, predLabel }: ConfusionGridProps) {
  const labelOf = (cls: Cls): string => classes.find((c) => c.cls === cls)?.label ?? cls;

  return (
    <div className={styles.tableWrap}>
      <table className={styles.matrix}>
        <caption>{caption}</caption>
        <thead>
          <tr>
            <td />
            <th scope="colgroup" colSpan={classes.length}>
              {predLabel}
            </th>
          </tr>
          <tr>
            <th scope="col">{trueLabel}</th>
            {classes.map((c) => (
              <th key={c.cls} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.true}>
              <th scope="row">
                {labelOf(r.true)}
                <span className={styles.rowN}>{r.n.text}</span>
              </th>
              {r.cells.map((c) => {
                const reportable = !c.below_floor && c.share.point !== null;
                return (
                  <td
                    key={c.pred}
                    className={c.diagonal ? `${styles.cell} ${styles.diagonal}` : styles.cell}
                    data-cell={`${r.true}-${c.pred}`}
                    data-diagonal={c.diagonal ? "" : undefined}
                  >
                    {reportable ? (
                      <span className={styles.shade} data-shade="" style={{ opacity: shadeOf(c.share.point ?? 0) }} />
                    ) : null}
                    <span className={styles.cellBody}>
                      {reportable ? (
                        <>
                          <span className={styles.share}>{c.share.text}</span>
                          <span className={styles.count}>{c.n.text}</span>
                          {showCi ? <span className={styles.interval}>{c.share.ci_text}</span> : null}
                        </>
                      ) : (
                        <NA text={c.share.text} />
                      )}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
