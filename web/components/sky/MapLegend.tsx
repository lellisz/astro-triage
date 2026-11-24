import { useId } from "react";
import type { Sky } from "@/lib/schema";
import { ptInt, ptPct } from "@/lib/fmt";
import type { SkyMapLabels } from "@/content/labels";
import { MIN_OPACITY } from "./scale";
import styles from "./sky.module.css";

export type MapLegendProps = {
  mode: "rate" | "count";
  colorCap: Sky["color_cap"];
  howText: string;
  labels: SkyMapLabels;
  saturated: boolean;
  hideBelowCount?: boolean;
};

export function MapLegend({ mode, colorCap, howText, labels, saturated, hideBelowCount }: MapLegendProps) {
  const gid = useId();
  const cap = mode === "rate" ? colorCap.rate : colorCap.count;
  const min = mode === "rate" ? ptPct(0, 0) : ptInt(0);
  let max: string;
  if (cap === null) {
    max = labels.notAvailable;
  } else {
    const text = mode === "rate" ? ptPct(cap, 1) : ptInt(cap);
    max = saturated ? `${labels.atLeast} ${text}` : text;
  }
  return (
    <div className={styles.legend}>
      <span className={styles.legendTitle}>
        {labels.legendTitle}: {mode === "rate" ? labels.modeRate : labels.modeCount}
      </span>
      <div className={styles.legendBar}>
        <span>{min}</span>
        <svg className={styles.gradient} viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <linearGradient id={gid} x1="0" x2="1" y1="0" y2="0">
              <stop className={styles.stop} offset="0" stopOpacity={MIN_OPACITY} />
              <stop className={styles.stop} offset="1" stopOpacity={1} />
            </linearGradient>
          </defs>
          <rect width="100" height="10" fill={`url(#${gid})`} />
        </svg>
        <span>{max}</span>
      </div>
      {mode === "rate" ? (
        <span className={styles.legendRing}>
          <svg className={styles.legendRingIcon} width="16" height="16" viewBox="0 0 16 16" role="img" aria-label={labels.ringLegendAria}>
            <circle className={styles.ringIcon} cx="8" cy="8" r="6" fill="none" />
          </svg>
          {labels.legendRing}
        </span>
      ) : null}
      <span>{labels.legendCenter}</span>
      <p className={styles.howText}>
        <strong>{labels.howLabel}</strong> {howText}
      </p>
      {hideBelowCount ? <p className={styles.howText}>{labels.hiddenInCountNote}</p> : null}
    </div>
  );
}
