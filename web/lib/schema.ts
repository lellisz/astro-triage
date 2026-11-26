// Espelho do Apendice A do ADR 0008, conferido contra o JSON gerado por
// astro_triage.web_export (release full_v0). Python e dono de todo numero e
// de toda frase que depende de dado; aqui so ha tipos.

export const SCHEMA_VERSION = 1;

export type Num = { value: number | null; text: string };
export type Est = {
  point: number | null;
  lo: number | null;
  hi: number | null;
  text: string;
  ci_text: string;
};
export type Rule = "as_shipped" | "prior_matched";
export type Prior = "P1" | "P2" | "P3";
export type Cls = "STAR" | "GALAXY" | "QSO";
export type CompareState = "win" | "tie" | "worse" | "n/a";

export type Head = {
  schema_version: typeof SCHEMA_VERSION;
  release: string;
  provisional: true;
  generated_at: string;
  page: string;
};

export type StateRow = { rule: Rule; prior: Prior; label: string };

export type Meta = Head & {
  states: StateRow[];
  default_state: { rule: Rule; prior: Prior; tech: boolean };
  shipped_prior: Prior;
  partial_priors: Prior[];
  rule_labels: Record<Rule, string>;
  seal_text: string;
  nside: number;
  band_edges_deg: number[];
  floor: { min_objects: number; min_pixels: number };
  n_boot: number;
};

export type OverviewRow = {
  rule: Rule;
  prior: Prior;
  partial: boolean;
  qso_prior: Num;
  purity: Est;
  completeness: Est;
  accuracy: Num;
  answer_text: string;
  partial_notice: string | null;
};

export type OverviewHeadline = {
  key: "no_detectable_difference" | "network_wins" | "unavailable";
  question: string;
  title: string;
  lead: string;
  caveat: string | null;
  pairs: { net: string; text: string }[];
  contrast: { accuracy: Num; qso_purity: Est; text: string };
  summary_en: string;
};

export type Overview = Head & {
  headline: OverviewHeadline;
  rows: OverviewRow[];
  accuracy_note: string;
  technical: {
    data_box: {
      n_pairs: Num;
      n_test: Num;
      n_errors: Num;
      n_pixels: Num;
      accuracy: Num;
      qso_hits: Num;
      qso_total: Num;
      class_mix: Record<Cls, Num>;
    };
    validation: { purity: Num; completeness: Num } | null;
    error_directions: { true: Cls; pred: Cls; n: Num; share: Num }[];
    triage: {
      threshold: Num;
      coverage: Num;
      accuracy_kept: Num;
      n_set_aside: Num;
      errors_set_aside: Num;
      errors_total: Num;
    } | null;
    dsc: {
      population: string;
      // No JSON real e um rotulo de populacao (ex.: "P0"), nao um Prior.
      prior: string;
      n_rows: Num;
      macro_f1: { model: string; label: string; est: Est }[];
      paired_diffs: { a: string; b: string; est: Est }[];
    };
    models: {
      model: string;
      label: string;
      n_features: number;
      best_iteration: number;
      temperature: Num;
      importance: { feature: string; gain: Num }[];
    }[];
    confidence_hist: { lo: number; hi: number; n_right: number; n_wrong: number }[];
    calibration: Record<Cls, { p_mean: number; freq: number; n: number }[]>;
  };
};

export type SkyRegion = {
  hp: number;
  x: number;
  y: number;
  band: "A" | "B" | "C";
  n: number;
  below_floor: boolean;
  errors: number | null;
  rate: number | null;
};

export type Sky = Head & {
  cards: {
    n_regions: Num;
    n_objects: Num;
    n_below: Num;
    max_rate: Est;
    diverges: boolean;
    diverge_note: string | null;
  };
  texts: {
    title: string;
    lead: string;
    how_rate: string;
    how_rate_hidden: string;
    how_count: string;
    map_lead: string;
    position_note: string;
    neighbour_note: string;
    spread_note: string;
  };
  color_cap: { rate: number | null; count: number };
  graticule: {
    meridians: [number, number][][];
    parallels: [number, number][][];
    labels: { x: number; y: number; text: string; kind: "lat" | "lon" }[];
  };
  regions: SkyRegion[];
  bands: {
    band: "A" | "B" | "C";
    label: string;
    n_pixels: number;
    n: number;
    errors: number;
    rate: Est;
  }[];
  top_regions: {
    hp: number;
    l: number;
    b: number;
    position_text: string;
    n: number;
    errors: number;
    rate: Est;
  }[];
};

export type GalaxyBlock = {
  prior: Prior;
  prior_source: string;
  n_rows: Num;
  purity: Est;
  completeness: Est;
  partial: boolean;
};

export type GalaxyBar = {
  rule: Rule;
  prior: Prior;
  label: string;
  est: Est;
  partial: boolean;
};

export type GalaxyNetwork = {
  label: string;
  state: CompareState;
  word: string;
  diff: Est;
  d: number;
  lo: number;
  hi: number;
  delta: number;
  underpowered: boolean;
};

export type GalaxyErrorRow = { label: string; n: Num; errors: Num; rate: Est };

export type Galaxies = Head & {
  blocks: Record<Rule, GalaxyBlock[]>;
  bars: { purity: GalaxyBar[]; completeness: GalaxyBar[] };
  verdict: {
    key: "depends_wide" | "depends" | "wide" | "no_difference" | "unavailable";
    text: string;
    scope_text: string | null;
  };
  networks: {
    mlp_full: GalaxyNetwork | null;
    ftt_full: GalaxyNetwork | null;
  };
  power_note: string | null;
  networks_note: string | null;
  dl_error: string | null;
  fixed_point_text: string;
  error_rows: GalaxyErrorRow[];
  error_rows_note: string;
  floor_note: string;
  crossmatch_note: string | null;
  technical: {
    by_g: (GalaxyErrorRow & { reportable: boolean })[];
    by_z: (GalaxyErrorRow & { reportable: boolean })[];
    missing_note: string;
  };
};

type CompareOk = Head & {
  status: "ok";
  error_text: null;
  fixed_point: { prior: string; level: string; text: string };
  answer_text: string;
  reasons_text: string;
  protocol_accepted: boolean | null;
  freeze_exists: boolean;
  pairs: { net: string; verdict: "tie" | "win"; text: string }[];
  grid: {
    cls: string;
    cls_label: string;
    net: string;
    state: CompareState;
    word: string;
    d: number;
    lo: number;
    hi: number;
    diff: Est;
    delta: number | null;
    underpowered: boolean;
  }[];
  power_note: string | null;
  purity: {
    system: string;
    reference: boolean;
    net: string | null;
    label: string;
    low: number;
    median: number;
    high: number;
    detail_text: string;
  }[];
  rule_lines: string[];
  descriptive: {
    pair: string;
    role: string;
    cls: string;
    d: number;
    lo: number;
    hi: number;
    diff: Est;
    text: string;
  }[];
  // No JSON real e uma lista de strings simples (o Apendice A nao fixa o tipo).
  limitations: string[] | null;
};

type CompareUnavailable = Head & {
  status: "unavailable" | "error";
  error_text: string | null;
  fixed_point: null;
  answer_text: null;
  reasons_text: null;
  protocol_accepted: null;
  freeze_exists: boolean;
  pairs: null;
  grid: null;
  power_note: null;
  purity: null;
  rule_lines: null;
  descriptive: null;
  limitations: null;
};

export type Compare = CompareOk | CompareUnavailable;

export type Method = Head & {
  facts: {
    n_test: Num;
    n_pixels: Num;
    prior_qso: Num;
    min_objects: Num;
    min_pixels: Num;
    n_boot: Num;
    completeness: Num;
  };
  mix: Record<Cls, Num>;
  mix_text: string;
  method_text: string;
  lead: string;
  simple: string;
  reading: string[];
  usage: string[];
  rule_lines: string[];
  three_limits: string[];
  limitations: { text: string; original: string }[];
  glossary: { term: string; definition: string | null }[];
  links: { label: string; note: string; url: string }[];
  sources: string[];
};

export type RateRow = {
  label: string;
  lo: number | null;
  hi: number | null;
  n: Num;
  errors: Num;
  rate: Est;
  below_floor: boolean;
};

export type CalBin = {
  p_mean: Num;
  freq: Est;
  n: Num;
  below_floor: boolean;
  raw_p_mean: number | null;
  raw_freq: number | null;
};

export type GraficosTick = { value: number; text: string };

export type CmdCell = {
  gi: number;
  ci: number;
  n: Num;
  errors: Num;
  rate: Est;
  below_floor: boolean;
};

export type Cmd = {
  g_edges: number[];
  c_edges: number[];
  g_ticks: GraficosTick[];
  c_ticks: GraficosTick[];
  cells: CmdCell[];
  g_labels: string[];
  c_labels: string[];
  cap: {
    rate: number | null;
    count: number | null;
    rate_text: string | null;
    count_text: string | null;
    rate_zero_text: string;
    count_zero_text: string;
  };
  n_missing: Num;
  n_outside: Num;
  note: string;
};

export type Graficos = Head & {
  floor: number;
  model_label: string;
  scope_text: string;
  floor_text: string;
  calibration: {
    n_bins: number;
    networks_note: string;
    series: { cls: Cls; label: string; bins: CalBin[] }[];
  };
  error_by_g: {
    missing_note: string;
    series: { cls: Cls; label: string; rows: RateRow[]; missing: Num }[];
  };
  error_by_z: {
    star_note: string;
    series: { cls: "GALAXY" | "QSO"; label: string; rows: RateRow[]; missing: Num }[];
  };
  confusion: {
    classes: { cls: Cls; label: string }[];
    rows: {
      true: Cls;
      n: Num;
      cells: { pred: Cls; n: Num; share: Est; diagonal: boolean; below_floor: boolean }[];
    }[];
  };
  ece: {
    status: "ok" | "unavailable";
    note: string;
    rows: {
      system: string;
      label: string;
      cls: Cls;
      mean: Num;
      low: number;
      high: number;
      detail_text: string;
    }[];
  } | null;
  cmd: Cmd | null;
};

export type FmtSample = { fn: string; args: (number | boolean)[]; text: string };

export type PageData = {
  meta: Meta;
  overview: Overview;
  sky: Sky;
  galaxies: Galaxies;
  compare: Compare;
  method: Method;
  graficos: Graficos;
  fmt_samples: FmtSample[];
};

export type PageKey = keyof PageData;
