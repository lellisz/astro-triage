import { useId } from "react";
import { kitLabels } from "@/content/labels";
import styles from "./kit.module.css";

export type DataColumn = { key: string; label: string; numeric?: boolean };

type DataTableProps = {
  caption: string;
  columns: readonly DataColumn[];
  rows: readonly Record<string, string>[];
  labelledBy?: string;
};

export function DataTable({ caption, columns, rows, labelledBy }: DataTableProps) {
  const captionId = useId();
  return (
    <div className={styles.tableBlock}>
      {labelledBy ? null : (
        <p id={captionId} className={styles.caption}>
          {caption}
        </p>
      )}
      <div
        className={styles.tableWrap}
        role="region"
        tabIndex={0}
        aria-label={`${kitLabels.tableRegion}: ${caption}`}
      >
        <table className={styles.table} aria-labelledby={labelledBy ?? captionId}>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.key} scope="col" className={c.numeric ? styles.num : undefined}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c.key} className={c.numeric ? styles.num : undefined}>
                    {row[c.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
