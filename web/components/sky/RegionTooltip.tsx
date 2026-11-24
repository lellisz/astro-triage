"use client";

import type { SkyRegion } from "@/lib/schema";
import { ptInt, ptPct } from "@/lib/fmt";
import type { SkyMapLabels } from "@/content/labels";
import styles from "./sky.module.css";

export type RegionTooltipProps = {
  region: SkyRegion;
  bandLabel: string;
  labels: SkyMapLabels;
  leftPct: number;
  topPct: number;
};

export function regionDescription(
  region: SkyRegion,
  bandLabel: string,
  labels: SkyMapLabels,
): string {
  const errors = region.errors === null ? labels.notAvailable : ptInt(region.errors);
  const rate =
    region.below_floor || region.rate === null
      ? `${labels.notAvailable} (${labels.belowFloor})`
      : ptPct(region.rate, 1);
  return `${labels.band}: ${bandLabel}. ${labels.objects}: ${ptInt(region.n)}. ${labels.errors}: ${errors}. ${labels.rate}: ${rate}.`;
}

export function RegionTooltip({ region, bandLabel, labels, leftPct, topPct }: RegionTooltipProps) {
  const rate =
    region.below_floor || region.rate === null
      ? `${labels.notAvailable} (${labels.belowFloor})`
      : ptPct(region.rate, 1);
  return (
    <div
      className={styles.tooltip}
      role="status"
      aria-label={labels.tooltipRegion}
      style={{ left: `${leftPct}%`, top: `${topPct}%` }}
    >
      <div className={styles.tooltipRow}>
        <span className={styles.tooltipKey}>{labels.band}</span>
        <span>{bandLabel}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span className={styles.tooltipKey}>{labels.objects}</span>
        <span>{ptInt(region.n)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span className={styles.tooltipKey}>{labels.errors}</span>
        <span>{region.errors === null ? labels.notAvailable : ptInt(region.errors)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span className={styles.tooltipKey}>{labels.rate}</span>
        <span>{rate}</span>
      </div>
    </div>
  );
}
