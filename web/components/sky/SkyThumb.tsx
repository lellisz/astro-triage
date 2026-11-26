import type { Sky } from "@/lib/schema";
import { VIEW_H, VIEW_W, VIEW_X, VIEW_Y, opacityFor, points } from "./scale";
import styles from "./sky.module.css";

export type SkyThumbProps = {
  graticule: Sky["graticule"];
  regions: Sky["regions"];
  colorCap: Sky["color_cap"];
  aria: string;
  caption: string;
};

// Raio maior que o do mapa completo: na miniatura o ponto e o unico elemento legivel.
const DOT_RADIUS = 0.06;

// Figura estatica, desenhada so com as regioes reportaveis (acima do piso e com taxa).
export function SkyThumb({ graticule, regions, colorCap, aria, caption }: SkyThumbProps) {
  const drawn = regions.filter((r) => !r.below_floor && r.rate !== null);
  return (
    <figure className={styles.thumb}>
      <svg className={styles.svg} viewBox={`${VIEW_X} ${VIEW_Y} ${VIEW_W} ${VIEW_H}`} role="img" aria-label={aria}>
        <title>{aria}</title>
        {graticule.meridians.map((line, i) => (
          <polyline key={`m${i}`} className={styles.line} points={points(line)} />
        ))}
        {graticule.parallels.map((line, i) => (
          <polyline key={`p${i}`} className={styles.line} points={points(line)} />
        ))}
        {drawn.map((r) => (
          <circle
            key={`${r.x},${r.y}`}
            className={styles.dotStatic}
            cx={r.x}
            cy={-r.y}
            r={DOT_RADIUS}
            fillOpacity={opacityFor(r.rate, colorCap.rate)}
          />
        ))}
      </svg>
      <figcaption className={styles.thumbCaption}>{caption}</figcaption>
    </figure>
  );
}
