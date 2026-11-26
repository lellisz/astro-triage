// Opacidade minima do preenchimento: um marcador com valor zero continua visivel.
// Usada pelo mapa e pela legenda para que a escala desenhada seja a mesma.
export const MIN_OPACITY = 0.15;

// Moldura de tela em unidades de mapa, compartilhada pelo mapa e pela miniatura.
export const VIEW_X = -3.35;
export const VIEW_Y = -1.52;
export const VIEW_W = 6.7;
export const VIEW_H = 3.04;

export function opacityFor(value: number | null, cap: number | null): number {
  if (value === null || cap === null || cap <= 0) return MIN_OPACITY;
  const t = Math.min(Math.max(value / cap, 0), 1);
  return MIN_OPACITY + (1 - MIN_OPACITY) * t;
}

// O SVG cresce para baixo: y_svg = -y.
export const points = (line: [number, number][]): string =>
  line.map(([x, y]) => `${x},${-y}`).join(" ");
