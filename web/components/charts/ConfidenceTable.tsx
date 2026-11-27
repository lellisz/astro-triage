import { ptInt, ptPct } from "@/lib/fmt";
import styles from "./charts.module.css";

export type ConfidenceBin = { lo: number; hi: number; n_right: number; n_wrong: number };

type ConfidenceTableProps = {
  id: string;
  caption: string;
  bins: readonly ConfidenceBin[];
  bandLabel: string;
  rightLabel: string;
  wrongLabel: string;
};

const BAR_W = 120;
const BAR_H = 10;

// Cada coluna tem a propria escala (so layout): com uma escala unica os erros, que sao o
// assunto da tabela, ficariam invisiveis ao lado dos acertos. A faixa vira rotulo de linha
// porque as faixas tem larguras diferentes.
function Bar({ value, max, className }: { value: number; max: number; className: string }) {
  const w = max > 0 ? (value / max) * BAR_W : 0;
  return (
    <svg width={BAR_W} height={BAR_H} aria-hidden="true" focusable="false">
      <rect className={className} x={0} y={0} width={w} height={BAR_H} />
    </svg>
  );
}

export function ConfidenceTable({ id, caption, bins, bandLabel, rightLabel, wrongLabel }: ConfidenceTableProps) {
  const maxRight = Math.max(0, ...bins.map((b) => b.n_right));
  const maxWrong = Math.max(0, ...bins.map((b) => b.n_wrong));
  return (
    <table id={id}>
      <caption>{caption}</caption>
      <thead>
        <tr>
          <th scope="col">{bandLabel}</th>
          <th scope="col">{rightLabel}</th>
          <th scope="col">{wrongLabel}</th>
        </tr>
      </thead>
      <tbody>
        {bins.map((b) => (
          <tr key={`${b.lo}-${b.hi}`}>
            <th scope="row">
              {ptPct(b.lo, 0)} – {ptPct(b.hi, 0)}
            </th>
            <td>
              <Bar value={b.n_right} max={maxRight} className={styles.barRight} /> {ptInt(b.n_right)}
            </td>
            <td>
              <Bar value={b.n_wrong} max={maxWrong} className={styles.barWrong} /> {ptInt(b.n_wrong)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
