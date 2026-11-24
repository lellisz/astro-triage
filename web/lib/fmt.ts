// Formatacao pt-BR espelhando astro_triage.dashboard.fmt (paridade garantida por
// test/fmt.test.ts contra fmt_samples.json). Uso restrito: ticks de eixo, minimo e
// maximo da legenda e valores do tooltip do mapa, sempre a partir de valores ja
// exportados. Nada e derivado no front; texto de dado vem pronto do JSON.

const CI_LEVEL_PERCENT = "95";

const toPt = (s: string): string => s.replace(".", ",");

export function ptInt(n: number): string {
  const t = Math.trunc(n);
  const digits = String(Math.abs(t)).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  return t < 0 ? `-${digits}` : digits;
}

export function ptPct(x: number, digits = 1): string {
  return `${toPt((x * 100).toFixed(digits))}%`;
}

export function ptNum(x: number, digits = 3, signed = false): string {
  const body = toPt(x.toFixed(digits));
  return signed && !body.startsWith("-") ? `+${body}` : body;
}

// Equivale ao formato "g" do Python: 6 algarismos significativos, sem zeros finais.
export function ptG(x: number): string {
  if (x === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(Number(x.toPrecision(6)))));
  if (exp < -4 || exp >= 6) {
    const [mant, e] = x.toExponential(5).split("e");
    const m = mant.includes(".") ? mant.replace(/0+$/, "").replace(/\.$/, "") : mant;
    const sign = e.startsWith("-") ? "-" : "+";
    return toPt(`${m}e${sign}${e.replace(/^[+-]/, "").padStart(2, "0")}`);
  }
  const fixed = x.toFixed(Math.max(0, 5 - exp));
  const m = fixed.includes(".") ? fixed.replace(/0+$/, "").replace(/\.$/, "") : fixed;
  return toPt(m);
}

export function ptCiLabel(): string {
  return `IC ${CI_LEVEL_PERCENT}%`;
}
