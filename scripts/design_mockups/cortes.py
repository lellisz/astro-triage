"""Cuts report: candidates to remove from the real dashboard, each with file:line evidence.

The evidence is found by regex at build time (nothing is typed as a line number) and nothing is
deleted. Recommendations are prose written from that evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

DASH = "src/astro_triage/dashboard"
TESTS = "tests"
GAL = f"{DASH}/*galaxies*.py"
LAYER2_OWN = re.compile(r"(views|charts|data)_layer2\.py:")


def _files(hits: list[str], skip: re.Pattern | None = None) -> list[str]:
    """File names (sorted, unique) cited by the evidence lines, minus the ones matching skip."""
    names = {
        m.group(1).rsplit("/", 1)[-1]
        for h in hits
        if (m := re.match(r"- `([^`:]+:\d+)`", h)) and not (skip and skip.search(m.group(1).rsplit("/", 1)[-1] + ":"))
    }
    return sorted(n.split(":")[0] for n in names)


def _layer2_advice(hits: list[list[str]]) -> str:
    importers = _files(hits[0], LAYER2_OWN)
    callers = _files(hits[2], LAYER2_OWN)
    if not (importers and callers):
        return ("REVISAR: não há importação e chamada fora dos próprios módulos "
                f"(importam: {', '.join(importers) or 'ninguém'}; chamam: {', '.join(callers) or 'ninguém'}). "
                "Podem estar mortos; confirmar antes de decidir.")
    return ("MANTER no código por ora: NÃO estão mortos (importados por "
            f"{', '.join(importers)} e chamados por {', '.join(callers)}). Tirar da vitrine exige "
            "editar esses arquivos e os testes do painel, fora deste lote. A aba só aparece se "
            "existir o arquivo de avaliação; o mockup simplesmente não a mostra.")


def _galaxies_advice(hits: list[list[str]]) -> str:
    if hits[0]:
        return (f"REVISAR. Há {len(hits[0])} ocorrência(s) de região, pixel ou latitude nas "
                "Galáxias (linhas acima): a hipótese de repetição do Mapa não se descarta sem ler cada uma.")
    return ("MANTER. Sem ocorrência de região, pixel ou latitude nas Galáxias: os gráficos delas "
            "são por magnitude G e redshift, não repetem o Mapa. A hipótese não se confirma.")


# (title, [(what the search shows, glob, regex)], recommendation: text or f(hits per search))
CANDIDATES = (
    (
        'Cartão "Próximo passo: comparar com redes neurais"',
        [
            ("o cartão", f"{DASH}/html_overview.py", r"Próximo passo|Comparar com redes neurais"),
            ("a comparação que ele anuncia já tem página", f"{DASH}/app.py", r"DL_PAGE|dl_page"),
        ],
        ("CORTAR. O cartão anuncia como futuro algo que já existe (a página DL × LightGBM, "
         "linhas acima) e repete o texto de resultado provisório da própria página."),
    ),
    (
        "Cartões de veredito duplicados",
        [
            ("cartões por par", f"{DASH}/html_dl.py", r"def verdict_html|def pair_verdict"),
            ("onde os cartões são montados", f"{DASH}/views_dl.py", r"verdict_html|pair_verdicts"),
            ("o mesmo veredito na tabela de destaque", f"{DASH}/html_dl.py", r"Veredito da classe|def headline_table_html"),
            ("e no gráfico", f"{DASH}/charts_dl.py", r"verdict"),
        ],
        ("CORTAR os cartões por par (verdict_html). O veredito aparece três vezes (cartão, "
         "tabela e gráfico); fica a linha única derivada do mockup e a tabela no modo técnico."),
    ),
    (
        "Plotly na Comparação",
        [
            ("importações", f"{DASH}/charts_dl.py", r"plotly"),
            ("uso na página", f"{DASH}/views_dl.py", r"plotly_chart|charts_dl\."),
        ],
        ("CORTAR a figura de intervalos por par (substituída pela régua de diferença em HTML). "
         "MANTER risk_coverage_figure (curva de acerto contra cobertura) até existir substituto."),
    ),
    (
        "Módulos do Layer 2 (views_layer2, charts_layer2, data_layer2)",
        [
            ("quem importa (src)", DASH + "/*.py", r"(views|charts|data)_layer2"),
            ("quem importa (testes)", f"{TESTS}/*.py", r"dashboard\.(views|charts|data)_layer2"),
            ("quem usa o que foi importado", DASH + "/*.py", r"layer2_page\(|layer2_f1\(|layer2_units_frame\(|layer2_diffs_frame\(|layer2_summary\("),
            ("a rota é condicional ao arquivo da avaliação", f"{DASH}/app.py", r"LAYER2_PAGE|load_layer2"),
        ],
        _layer2_advice,
    ),
    (
        "Gráficos de Galáxias que repetem o Mapa",
        [
            ("referências a céu, região ou pixel nas Galáxias", GAL, r"healpix|pixel|lat_band|sky_|mollweide|l_centre"),
            ("o que as Galáxias desenham (magnitude G e redshift)", GAL, r"def (rate_dots|galaxy_scatter|simple_html|tech_html)|error_rate_table"),
            ("o que o Mapa desenha (região e latitude)", f"{DASH}/views_sky.py", r"test_pixel_stats|lat_band_table|top_error_regions"),
        ],
        _galaxies_advice,
    ),
    (
        "Barra lateral",
        [
            ("construção da barra", f"{DASH}/app.py", r"st\.sidebar|sidebar_counts"),
            ("conteúdo da barra", f"{DASH}/sidebar_info.py", r"st\.sidebar|def sidebar_counts"),
        ],
        ("CORTAR a barra como navegação e levar para o rodapé: ela só tem seletor de versão e "
         "contagens, que o selo e o rodapé de cada tela já cobrem. Exige editar app.py e "
         "sidebar_info.py (fora deste lote)."),
    ),
)


def _search(root: Path, glob: str, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    out = []
    for path in sorted(root.glob(glob)):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                out.append(f"- `{path.relative_to(root).as_posix()}:{n}`: `{line.strip()[:110]}`")
    return out


def render(root: Path, release: str) -> str:
    parts = [
        f"# Relatório de cortes ({release})",
        "",
        "Provisório. Nada foi apagado; as linhas abaixo são evidência lida do código.",
    ]
    for title, searches, advice in CANDIDATES:
        parts += ["", f"## {title}", ""]
        found = []
        for what, glob, pattern in searches:
            hits = _search(root, glob, pattern)
            found.append(hits)
            parts.append(f"{what}:")
            parts += hits or ["- nenhuma ocorrência"]
        text = advice(found) if callable(advice) else advice
        parts += ["", f"Recomendação: {text}"]
    return "\n".join(parts) + "\n"
