// Copia literal de web/components/sky/MollweideMap.tsx em main (antes da refatoracao),
// so com os imports ajustados. Serve de referencia para o teste de paridade; nao editar.
import { useState } from "react";
import type { Sky, SkyRegion } from "@/lib/schema";
import type { SkyMapLabels } from "@/content/labels";
import { RegionTooltip } from "@/components/sky/RegionTooltip";
import { MIN_OPACITY } from "@/components/sky/scale";
import styles from "@/components/sky/sky.module.css";

export type MapMode = "rate" | "count";

export type MollweideMapProps = {
  regions: Sky["regions"];
  graticule: Sky["graticule"];
  colorCap: Sky["color_cap"];
  bands: Sky["bands"];
  mode: MapMode;
  hideBelow: boolean;
  labels: SkyMapLabels;
  summary: string;
};

const VIEW_X = -3.35;
const VIEW_Y = -1.52;
const VIEW_W = 6.7;
const VIEW_H = 3.04;
const DOT_RADIUS = 0.035;
const LAT_LABEL_DX = -0.05;
const LON_LABEL_DY = 0.1;

function regionValue(region: SkyRegion, mode: MapMode): number | null {
  return mode === "rate" ? region.rate : region.n;
}

function capFor(colorCap: Sky["color_cap"], mode: MapMode): number | null {
  return mode === "rate" ? colorCap.rate : colorCap.count;
}

function opacityFor(value: number | null, cap: number | null): number {
  if (value === null || cap === null || cap <= 0) return MIN_OPACITY;
  const t = Math.min(Math.max(value / cap, 0), 1);
  return MIN_OPACITY + (1 - MIN_OPACITY) * t;
}

const points = (line: [number, number][]): string =>
  line.map(([x, y]) => `${x},${-y}`).join(" ");

export function MollweideMapBase({
  regions,
  graticule,
  colorCap,
  bands,
  mode,
  hideBelow,
  labels,
  summary,
}: MollweideMapProps) {
  const [active, setActive] = useState<SkyRegion | null>(null);
  const cap = capFor(colorCap, mode);
  const bandLabel = (b: SkyRegion["band"]): string =>
    bands.find((x) => x.band === b)?.label ?? b;
  const drawn = hideBelow ? regions.filter((r) => !r.below_floor) : regions;

  return (
    <div className={styles.scroll} tabIndex={0} role="region" aria-label={labels.mapScrollAria}>
    <div className={styles.wrap} onMouseLeave={() => setActive(null)}>
      <svg
        className={styles.svg}
        viewBox={`${VIEW_X} ${VIEW_Y} ${VIEW_W} ${VIEW_H}`}
        role="group"
        aria-label={labels.mapAria}
      >
        <title>{labels.mapAria}</title>
        <desc>{summary}</desc>
        <g aria-hidden="true">
          {graticule.meridians.map((line, i) => (
            <polyline key={`m${i}`} className={styles.line} points={points(line)} />
          ))}
          {graticule.parallels.map((line, i) => (
            <polyline key={`p${i}`} className={styles.line} points={points(line)} />
          ))}
          {graticule.labels.map((l, i) => (
            <text
              key={`l${i}`}
              className={l.kind === "lat" ? styles.gratLat : styles.gratLon}
              data-kind={l.kind}
              x={l.kind === "lat" ? l.x + LAT_LABEL_DX : l.x}
              y={l.kind === "lon" ? -l.y + LON_LABEL_DY : -l.y}
            >
              {l.text}
            </text>
          ))}
        </g>
        <g aria-hidden="true">
          {drawn.map((r, i) => {
            const ring = mode === "rate" && r.below_floor;
            const key = r.hp ?? `i${i}`;
            const common = {
              cx: r.x,
              cy: -r.y,
              r: DOT_RADIUS,
              "data-hp": r.hp ?? undefined,
              "data-below": r.below_floor ? "true" : undefined,
              onMouseEnter: () => setActive(r),
            };
            return ring ? (
              <circle key={key} {...common} className={styles.dotRing} fill="none" />
            ) : (
              <circle
                key={key}
                {...common}
                className={styles.dotFilled}
                fillOpacity={opacityFor(regionValue(r, mode), cap)}
              />
            );
          })}
        </g>
      </svg>
      {active ? (
        <RegionTooltip
          region={active}
          bandLabel={bandLabel(active.band)}
          labels={labels}
          leftPct={((active.x - VIEW_X) / VIEW_W) * 100}
          topPct={((-active.y - VIEW_Y) / VIEW_H) * 100}
        />
      ) : null}
    </div>
    </div>
  );
}
