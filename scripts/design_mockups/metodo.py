"""Screen 4, 'Método': pipeline in five wire boxes, ADR status verbatim, limitations, links."""

from __future__ import annotations

from pathlib import Path

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.limits_pt import translate

from . import loaders
from .kit import NA, TABLE_CSS, Ledger, e
from .loaders import Cell

SCREEN = "metodo"
TITLE = "Método"
ADR_DIR = Path(__file__).resolve().parents[2] / "docs" / "adr"
ADR_IDS = ("0001", "0002", "0003", "0004", "0005", "0006")
ADR_TOPIC = {
    "0001": "Armazenamento",
    "0002": "Rótulos e circularidade",
    "0003": "Divisão por regiões e priors",
    "0004": "Caminho de dados sem login",
    "0005": "Extensão opcional (fechada)",
    "0006": "Aprendizado profundo × LightGBM",
}
STATUS_PREFIX = "- Status:"
EMPHASIS = "**"  # Markdown emphasis marks, dropped from the display only (the ledger keeps the text)
HEAD_LINES = 60
LINKS = ("Repositório do código", "Documentação das ADRs", "Painel completo")
CLASS_PT = {"STAR": "estrelas", "GALAXY": "galáxias", "QSO": "quasares"}

CSS = (
    TABLE_CSS
    + f"""
.flow{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:24px 0}}
.box{{border:1px solid {theme.DASH};padding:16px;min-width:0}}
.box .k{{font:{theme.MIN_SECONDARY_PX}px {theme.FONT_SANS};color:{theme.TEXT_SECONDARY}}}
.box h3{{font:600 {theme.CONTROL_PX}px/1.3 {theme.FONT_SANS};margin:4px 0 12px}}
.box .v{{font:400 32px/1.1 {theme.FONT_SERIF};color:{theme.ACCENT}}}
.box .s{{font-size:{theme.MIN_SECONDARY_PX}px;color:{theme.TEXT_SECONDARY};margin-top:6px}}
pre.st{{margin:0;white-space:pre-wrap;word-break:break-word;font:{theme.MIN_SECONDARY_PX}px/1.5 {theme.FONT_SANS};color:{theme.TEXT}}}
td.nw{{white-space:nowrap;min-width:96px}}
ul.lim{{margin:0;padding-left:20px;max-width:900px}}
ul.lim li{{margin-bottom:8px}}
@media(max-width:1100px){{.flow{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:720px){{.flow{{grid-template-columns:1fr}}}}
"""
)


def adr_status(adr_id: str) -> tuple[str, str]:
    """(file name, status text verbatim: the '- Status:' bullet and its indented continuation)."""
    path = next(ADR_DIR.glob(f"{adr_id}-*.md"), None)
    if path is None:
        return f"{adr_id}-*.md", ""
    lines = path.read_text(encoding="utf-8").splitlines()[:HEAD_LINES]
    start = next((i for i, ln in enumerate(lines) if ln.startswith(STATUS_PREFIX)), None)
    if start is None:
        return path.name, ""
    text = [lines[start][len(STATUS_PREFIX) :].strip()]
    for ln in lines[start + 1 :]:
        if not ln.startswith("  ") or not ln.strip():
            break
        text.append(ln.strip())
    return path.name, "\n".join(text)


def _box(step: str, title: str, big: str, lines: list[str]) -> str:
    sub = "".join(f"<div class='s'>{e(t)}</div>" for t in lines)
    return (
        f"<div class='box'><div class='k'>{e(step)}</div><h3>{e(title)}</h3>"
        f"<div class='v'>{e(big)}</div>{sub}</div>"
    )


def _pipeline(L: Ledger, rel) -> str:
    c = loaders.dataset_cells(rel)
    cen = loaders.census(rel)
    acc = loaders.accuracy(rel)
    i = L.integer
    classes = " · ".join(
        f"{i(c[f'class_{k}'], f'caixa 2 · {CLASS_PT[k]}')} {CLASS_PT[k]}" for k in CLASS_PT
    )
    folds = (
        f"treino {i(c['fold_train'], 'caixa 3 · objetos de treino')} · "
        f"validação {i(c['fold_val'], 'caixa 3 · objetos de validação')}"
    )
    boxes = [
        _box(
            "1 · Cruzamento",
            "Gaia DR3 × SDSS",
            i(c["pairs_after_r_match"], "caixa 1 · pares dentro do raio"),
            [
                f"de {i(c['input_pairs'], 'caixa 1 · pares candidatos')} pares candidatos",
                f"raio de {L.num(c['r_match'], 'caixa 1 · raio de cruzamento (arcsec)', 2)} arcsec",
            ],
        ),
        _box(
            "2 · Rótulos",
            "Classe do espectro SDSS",
            i(c["final_rows"], "caixa 2 · linhas totais"),
            ["objetos rotulados", classes],
        ),
        _box(
            "3 · Divisão",
            "Por região do céu (nível 5)",
            i(cen["objects"], "caixa 3 · objetos de teste"),
            [
                f"objetos de teste em {i(cen['regions'], 'caixa 3 · regiões de teste')} regiões",
                folds,
            ],
        ),
        _box(
            "4 · Modelo",
            "LightGBM calibrado",
            i(c["n_features"], "caixa 4 · atributos"),
            ["atributos por objeto", "temperatura ajustada só na validação"],
        ),
        _box(
            "5 · Teste fechado",
            "Avaliação e comparação",
            i(acc["n_seeds"], "caixa 5 · sementes"),
            [
                "sementes por sistema",
                f"intervalos por bootstrap de regiões (n_boot {i(cen['n_boot'], 'caixa 5 · n_boot')})",
            ],
        ),
    ]
    return f"<div class='flow'>{''.join(boxes)}</div>"


def _adr_table(L: Ledger) -> str:
    rows, found = [], {}
    for adr_id in ADR_IDS:
        name, status = adr_status(adr_id)
        found[adr_id] = bool(status)
        shown = L.text(
            Cell(status or None, f"docs/adr/{name}", "cabeçalho · Status"),
            f"ADR {adr_id} · status (verbatim)",
        )
        rows.append(
            f"<tr><td class='mono nw'>ADR {adr_id}</td><td>{e(ADR_TOPIC[adr_id])}</td>"
            f"<td><pre class='st'>{e(shown).replace(EMPHASIS, "")}</pre></td></tr>"
        )
    missing = [k for k, ok in found.items() if not ok]
    if missing:
        raise ValueError(f"método: status não encontrado nas ADRs {missing}")
    return (
        "<div class='tw'><table><tr><th>ADR</th><th>Tema</th>"
        "<th>Status (texto do cabeçalho, no idioma original, sem os marcadores de ênfase do Markdown)</th></tr>"
        f"{''.join(rows)}</table></div>"
    )


def _limits(L: Ledger, rel) -> str:
    items = []
    for n, original in enumerate(rel.results.get("limitations", [])):
        shown, translated = translate(original)
        text = shown if translated else f"{shown} (sem tradução)"
        L.text(Cell(text, loaders.RESULTS_FILE, f"limitations[{n}]"), f"limitação {n + 1}")
        items.append(f"<li>{e(text)}</li>")
    return f"<ul class='lim'>{''.join(items)}</ul>" if items else f"<p>{NA}</p>"


def render(rel, L: Ledger) -> tuple[str, str, dict]:
    links = "".join(f"<li>{e(t)} (URL a definir)</li>" for t in LINKS)
    body = (
        "<p class='lead'>O que foi construído, em cinco passos.</p>"
        f"{_pipeline(L, rel)}"
        "<section class='sec'><h2>Status das decisões (ADRs)</h2>"
        f"{_adr_table(L)}</section>"
        "<section class='sec'><h2>Limitações</h2>"
        f"{_limits(L, rel)}</section>"
        "<section class='sec'><h2>Links</h2>"
        "<p class='sm'>O repositório é privado; os endereços ainda não existem.</p>"
        f"<ul class='lim'>{links}</ul></section>"
    )
    footer = f"Release {e(rel.name)} · {e(fmt.pt_ci_label())} por bootstrap de regiões"
    return body, footer, {"adrs": list(ADR_IDS), "limitations": len(rel.results["limitations"])}
