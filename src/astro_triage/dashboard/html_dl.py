"""Comparação page blocks as HTML (design kit). Pure: no streamlit; no number is written here.

Every figure is formatted at run time from ``results_dl.json``, from the constants of
``dl_compare`` (the rule) or from ``runner.GRID`` (the priors). Text is escaped by the kit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import TYPE_CHECKING, Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard.constants import ADR_NUMBER
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.html_kit import box, section, stat, table
from astro_triage.dashboard.html_system import PAIR_ALIAS

if TYPE_CHECKING:
    from astro_triage.dashboard.views_dl import Cell, PriorRow, QsoPurity, RuleFacts

TITLE: Final = "Comparação: LightGBM × deep learning"
SENSITIVITY_TITLE: Final = "Sensibilidade: outros priors e outra completude"
LEAD: Final = (
    "Duas redes neurais (MLP e FT-Transformer) foram postas contra o LightGBM, no teste, "
    "com uma regra escrita antes de olhar o resultado."
)
HINT: Final = (
    'Ligue o "Modo técnico" no topo para ver a regra, as tabelas por classe, a pureza '
    "absoluta e as limitações."
)
UNAVAILABLE: Final = "não disponível"
UNAVAILABLE_NOTE: Final = (
    "Esta versão dos dados não tem a avaliação do aprendizado profundo. Escolha outra "
    "versão na barra lateral."
)
CHART_CAPTION: Final = (
    "Cada ponto é a diferença de pureza (primeiro menos segundo) com o intervalo de "
    "confiança; a faixa clara é a margem δ. Cada painel tem a própria escala."
)
RISK_TITLE: Final = "Acerto contra cobertura (descritivo)"
RISK_NOTE: Final = (
    "Curva bruta, sem reponderação, média das sementes; não entra no veredito."
)
READING: Final = (
    (
        "O par só vence se vencer nas duas classes (quasar e galáxia). Em cada classe, "
        "vence quando D fica acima da margem δ e o limite inferior do intervalo fica acima "
        "de zero; fica pior no caso espelhado. Qualquer outro resultado é empate, e o "
        "empate fica com o LightGBM."
    ),
    (
        "Em galáxia o intervalo costuma ser mais largo que a margem δ. Isso é falta de poder "
        "estatístico para decidir, não indica que os dois modelos sejam iguais."
    ),
    "Quasar sozinho pode ganhar e o par mesmo assim empatar: a regra olha as duas classes.",
)
PRIOR_LEGEND: Final = (
    "Priors de classificador, não frações medidas do céu; pi_src não é P0."
)
PRIOR_SOURCES: Final = {
    "P0": "uniforme (regime balanceado)",
    "P1": "Bailer-Jones+2019",
    "P2": "Delchambre+2023, céu inteiro (estrela derivada)",
    "P3": "Delchambre+2023, alta latitude galáctica (estrela derivada)",
}
LIMITS: Final = (
    (
        "Em outros cortes do projeto, algumas células ficam abaixo do piso de relato; por "
        "isso nenhum número é definitivo."
    ),
    "A regra de redshift (GCUT-RC3) não está implementada.",
    (
        "A amostra é a interseção SDSS × Gaia: só entram objetos com espectro do SDSS que "
        "casam com o Gaia, então ela não representa o céu."
    ),
    "O protocolo da ADR {adr} foi executado como registrado, e a ADR segue em rascunho.",
    "O teste foi aberto uma vez; não há segunda leitura para ajustar a regra.",
)
SEAL_REASONS: Final = (
    "células abaixo do piso de relato",
    "regra de redshift (GCUT-RC3) não implementada",
    "seleção SDSS × Gaia",
    "protocolo ainda em rascunho",
)
_FAMILY: Final = {"lgbm": "LightGBM", "mlp": "MLP", "ftt": "FT-Transformer"}
_FEATURES: Final = {"full": "completo", "noind": "sem indicadores"}
_ENSEMBLE: Final = "ens_"
_CLASS_VERDICT: Final = {
    "win": "vitória na classe",
    "tie": "empate na classe",
    "worse": "pior na classe (descritivo; não decide o par)",
}
_PAIR_VERDICT: Final = {
    "tie": "empate: fica o LightGBM",
    "win": "vitória do par",
}
_ROLE: Final = {
    "headline": "destaque",
    "sensitivity": "sensibilidade",
    "descriptive": "descritivo",
}
_FRONT: Final = "front"


def system_label(name: str) -> str:
    """``ens_mlp_full`` -> 'conjunto MLP completo'; unknown parts fall back to 'n/d'."""
    ensemble = name.startswith(_ENSEMBLE)
    family, _, feats = name.removeprefix(_ENSEMBLE).partition("_")
    text = f"{_FAMILY.get(family, 'n/d')} {_FEATURES.get(feats, 'n/d')}"
    return f"conjunto {text}" if ensemble else text


def pair_label(pair: str) -> str:
    first, _, second = pair.partition("-")
    return f"{system_label(first)} × {system_label(second)}"


def role_label(role: str) -> str:
    if role.startswith(_FRONT):
        return f"frente {role.removeprefix(_FRONT)}"
    return _ROLE.get(role, "n/d")


def class_label(cls: str) -> str:
    return PAIR_NAMES.get(cls, cls).capitalize()


def class_verdict(key: str | None) -> str:
    return "descritivo" if key is None else _CLASS_VERDICT.get(key, "n/d")


def pair_verdict(text: str | None) -> str:
    if not text:
        return UNAVAILABLE
    head = text.partition(":")[0].strip().lower()
    return _PAIR_VERDICT.get(PAIR_ALIAS.get(head, ""), UNAVAILABLE)


def _p(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def _num(x: float | None, signed: bool = False) -> str:
    return UNAVAILABLE if x is None else fmt.pt_num(x, 4, signed)


def _interval(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return UNAVAILABLE
    return f"[{fmt.pt_num(lo, 4)}; {fmt.pt_num(hi, 4)}]"


def _level(prior: str, completeness: float) -> str:
    return f"{prior}, C* = {fmt.pt_num(completeness, 2)}"


def lead_html() -> str:
    return section(TITLE, LEAD)


def hint_html() -> str:
    return f"<p class='at-cap'>{escape(HINT)}</p>"


def unavailable_html() -> str:
    return section(TITLE, f"{UNAVAILABLE_NOTE} ({UNAVAILABLE})")


def seal_html() -> str:
    body = _p("Resultado provisório. Razões: " + "; ".join(SEAL_REASONS) + ".")
    return box("Provisório", body)


def chart_caption_html() -> str:
    return f"<p class='at-cap'>{escape(CHART_CAPTION)}</p>"


def verdict_html(
    verdicts: Mapping[str, str | None], prior: str, completeness: float
) -> str:
    if not verdicts:
        return _p(UNAVAILABLE)
    return "".join(
        stat(
            pair_verdict(text),
            pair_label(pair),
            _level(prior, completeness),
            accent=True,
        )
        for pair, text in verdicts.items()
    )


def rule_lines(rule: RuleFacts) -> list[str]:
    levels = " e ".join(fmt.pt_num(c, 2) for c in rule.levels)
    priors = ", ".join((rule.primary_prior, *rule.sensitivity_priors))
    return [
        (
            f"Métrica principal: pureza de quasar e de galáxia com completude fixa "
            f"C* = {fmt.pt_num(rule.primary_completeness, 2)} e prior {rule.primary_prior}; "
            f"completudes lidas: {levels}; priors: {priors}."
        ),
        (
            f"Margem δ = {fmt.pt_g(rule.multiplier)} × o desvio-padrão (ddof = {rule.ddof}) "
            f"de {fmt.pt_int(rule.n_seeds)} sementes do LightGBM, medido na validação."
        ),
        (
            f"Intervalo por bootstrap em blocos do céu (bloco = pixel nível {rule.block_level} "
            f"da divisão por {fmt.pt_int(rule.block_divisor)}), com ao menos "
            f"{fmt.pt_int(rule.min_blocks)} blocos."
        ),
        (
            "Veredito da classe: vitória quando D fica acima de δ e o limite inferior do "
            "intervalo fica acima de zero; pior quando D fica abaixo de -δ e o limite "
            "superior fica abaixo de zero; empate nos demais casos."
        ),
        "Em empate fica o LightGBM; vitória do par exige as duas classes.",
    ]


def rule_html(rule: RuleFacts) -> str:
    items = "".join(f"<li>{escape(t)}</li>" for t in rule_lines(rule))
    return box(
        f"Regra pré-registrada (ADR {ADR_NUMBER:04d}, rascunho)", f"<ul>{items}</ul>"
    )


def reading_html() -> str:
    return section("Como ler", READING[0]) + "".join(_p(t) for t in READING[1:])


def _cell_row(c: Cell) -> list[str]:
    ci = UNAVAILABLE if c.na else _interval(c.lo, c.hi)
    d = UNAVAILABLE if c.na else _num(c.d, True)
    return [
        pair_label(c.pair),
        role_label(c.role),
        class_label(c.cls),
        d,
        ci,
        _num(c.delta),
        class_verdict(c.verdict),
    ]


_CELL_HEADERS: Final = (
    "Par",
    "Papel",
    "Classe",
    "D",
    fmt.pt_ci_label(),
    "δ",
    "Veredito da classe",
)


def headline_table_html(rows: Sequence[Cell], prior: str, completeness: float) -> str:
    title = f"Destaque: {_level(prior, completeness)}"
    note = (
        "D é a diferença de pureza, primeiro menos segundo. Pares marcados como descritivos "
        "ou frentes não entram no veredito."
    )
    if not rows:
        return section(title, UNAVAILABLE)
    body = [_cell_row(c) for c in rows]
    return section(title, note) + table(_CELL_HEADERS, body)


def sensitivity_html(groups: Sequence[tuple[str, str, Sequence[Cell]]]) -> str:
    rows = [
        [f"{prior}, C* = {level}".replace(".", ","), *_cell_row(c)]
        for prior, level, cs in groups
        for c in cs
        if c.role in ("headline", "sensitivity")
    ]
    if not rows:
        return _p(UNAVAILABLE)
    return table(("Prior e completude", *_CELL_HEADERS), rows)


def qso_purity_html(
    rows: Sequence[QsoPurity], prior: str, completeness: float, qso_prior: float
) -> str:
    title = "Pureza absoluta de quasar"
    note = (
        f"No ponto de operação {_level(prior, completeness)}, em que o prior de quasar é "
        f"{fmt.pt_pct(qso_prior, 3)}. A pureza é baixa em todos os sistemas porque esse "
        "prior é pequeno; a comparação entre eles é a diferença D, não o nível."
    )
    if not rows:
        return section(title, UNAVAILABLE)
    body = [
        [
            system_label(r.system),
            fmt.pt_pct(r.median, 2),
            f"{fmt.pt_pct(r.low, 2)} a {fmt.pt_pct(r.high, 2)}",
        ]
        for r in rows
    ]
    return section(title, note) + table(
        ("Sistema", "Mediana das sementes", "Mínimo a máximo"), body
    )


def fronts_html(rows: Sequence[Cell], prior: str, completeness: float) -> str:
    title = "Frentes de remoção dos indicadores (descritivo)"
    note = (
        "D é a pureza de quasar com indicadores menos sem indicadores, em "
        f"{_level(prior, completeness)}."
    )
    if not rows:
        return section(title, UNAVAILABLE)
    if all(c.lo is not None and c.lo > 0 for c in rows):
        note += (
            " Em todas as famílias o intervalo fica acima de zero: remover os "
            "indicadores custa pureza de quasar."
        )
    body = [
        [pair_label(c.pair), role_label(c.role), _num(c.d, True), _interval(c.lo, c.hi)]
        for c in rows
    ]
    return section(title, note) + table(("Par", "Frente", "D", fmt.pt_ci_label()), body)


def priors_html(rows: Sequence[PriorRow], p3_bands: Sequence[str]) -> str:
    classes = tuple(PAIR_NAMES)
    body = []
    for r in rows:
        source = PRIOR_SOURCES.get(r.prior, UNAVAILABLE)
        if r.prior == "P3":
            source += f"; só nas faixas {' e '.join(p3_bands)}"
        body.append(
            [
                r.prior,
                source,
                *(
                    fmt.pt_pct(r.fractions[c], 4) if c in r.fractions else UNAVAILABLE
                    for c in classes
                ),
            ]
        )
    headers = ("Prior", "Origem", *(class_label(c) for c in classes))
    return section("Priors usados", PRIOR_LEGEND) + table(headers, body)


def risk_title_html() -> str:
    return section(RISK_TITLE, RISK_NOTE)


def limits_html() -> str:
    items = "".join(
        f"<li>{escape(t.format(adr=f'{ADR_NUMBER:04d}'))}</li>" for t in LIMITS
    )
    return box("Limitações", f"<ul>{items}</ul>")
