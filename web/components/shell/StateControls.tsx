"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { acessibilidade, chartsHref, compareHref, controls } from "@/content/labels";
import type { Meta, Prior, Rule } from "@/lib/schema";
import { readState, withState, type UiState } from "@/lib/ui-state";
import styles from "./StateControls.module.css";

type Props = { meta: Meta; hideRuleAndPrior?: boolean };

function trimSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

function priorsFor(meta: Meta, rule: Rule): Prior[] {
  return meta.states.filter((s) => s.rule === rule).map((s) => s.prior);
}

function priorAfterRuleChange(meta: Meta, current: Prior, rule: Rule): Prior {
  const allowed = priorsFor(meta, rule);
  if (allowed.includes(current)) return current;
  return allowed.includes(meta.shipped_prior) ? meta.shipped_prior : (allowed[0] ?? current);
}

function Controls({
  meta,
  state,
  hideRuleAndPrior,
  onChange,
}: {
  meta: Meta;
  state: UiState;
  hideRuleAndPrior: boolean;
  onChange?: (next: UiState) => void;
}) {
  const disabled = onChange === undefined;
  const rules = Object.keys(meta.rule_labels) as Rule[];
  const priors = priorsFor(meta, state.rule);
  const emit = (next: UiState) => onChange?.(next);

  return (
    <div className={styles.row}>
      {hideRuleAndPrior ? null : (
        <>
          <fieldset className={styles.group} disabled={disabled}>
            <legend className={styles.legend}>{controls.rule}</legend>
            {rules.map((rule) => (
              <label key={rule} className={styles.option}>
                <input
                  type="radio"
                  name="regra"
                  value={rule}
                  checked={state.rule === rule}
                  onChange={() =>
                    emit({
                      ...state,
                      rule,
                      prior: priorAfterRuleChange(meta, state.prior, rule),
                    })
                  }
                />
                <span>{meta.rule_labels[rule]}</span>
              </label>
            ))}
          </fieldset>
          <fieldset className={styles.group} disabled={disabled}>
            <legend className={styles.legend}>{controls.prior}</legend>
            {priors.map((prior) => (
              <label key={prior} className={styles.option}>
                <input
                  type="radio"
                  name="prior"
                  value={prior}
                  checked={state.prior === prior}
                  onChange={() => emit({ ...state, prior })}
                />
                <span>{prior}</span>
              </label>
            ))}
            {priors.length < 2 ? <span className={styles.hint}>{controls.priorFixed}</span> : null}
          </fieldset>
        </>
      )}
      <label className={styles.option}>
        <input
          type="checkbox"
          checked={state.tech}
          disabled={disabled}
          onChange={(event) => emit({ ...state, tech: event.target.checked })}
        />
        <span>{controls.tech}</span>
      </label>
    </div>
  );
}

function hidesRuleAndPrior(pathname: string, forced: boolean | undefined): boolean {
  if (forced !== undefined) return forced;
  const here = trimSlash(pathname);
  return here === trimSlash(compareHref) || here === trimSlash(chartsHref);
}

function ControlsWithState({ meta, hideRuleAndPrior }: Props) {
  const params = useSearchParams();
  const pathname = usePathname();
  const router = useRouter();
  return (
    <Controls
      meta={meta}
      state={readState(params, meta)}
      hideRuleAndPrior={hidesRuleAndPrior(pathname, hideRuleAndPrior)}
      onChange={(next) => router.replace(withState(pathname, next, meta.release), { scroll: false })}
    />
  );
}

export function StateControls({ meta, hideRuleAndPrior }: Props) {
  const pathname = usePathname();
  // Comparacao nao tem modo tecnico nem usa regra e prior (como no Streamlit):
  // sem controles com efeito, a barra inteira sai.
  if (trimSlash(pathname) === trimSlash(compareHref)) return null;
  return (
    <section className={styles.bar} aria-label={acessibilidade.settingsRegion}>
      <h2 className={styles.title}>{controls.settings}</h2>
      <Suspense
        fallback={
          <Controls
            meta={meta}
            state={meta.default_state}
            hideRuleAndPrior={hidesRuleAndPrior(pathname, hideRuleAndPrior)}
          />
        }
      >
        <ControlsWithState meta={meta} hideRuleAndPrior={hideRuleAndPrior} />
      </Suspense>
    </section>
  );
}

function chipLabel(meta: Meta, state: UiState): string {
  const found = meta.states.find((s) => s.rule === state.rule && s.prior === state.prior);
  const base = found?.label ?? "";
  return state.tech ? `${base} · ${controls.techShort}` : base;
}

function ChipWithState({ meta }: { meta: Meta }) {
  const params = useSearchParams();
  return <span className={styles.chip}>{chipLabel(meta, readState(params, meta))}</span>;
}

export function StateChip({ meta }: { meta: Meta }) {
  return (
    <span className={styles.chipWrap} aria-label={controls.currentState}>
      <Suspense fallback={<span className={styles.chip}>{chipLabel(meta, meta.default_state)}</span>}>
        <ChipWithState meta={meta} />
      </Suspense>
    </span>
  );
}
