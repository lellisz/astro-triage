"""Methodological caveats kept out of the pages, for the "Método" page.

Pure strings and data: no Streamlit, and no number is written here; every figure comes in as an
argument already formatted (pt-BR) by the caller.
"""

from __future__ import annotations

PRIOR_NOTE = (
    "As purezas de quasar valem num céu com o prior indicado (Delchambre+2023), uma premissa "
    "do classificador e não uma fração medida do céu. O acerto vale para a mistura do conjunto "
    "de teste, não para as proporções do céu."
)

INTERVAL_NOTE = "Compare o tamanho do intervalo antes de concluir: intervalos largos indicam poucos objetos."

MAP_NOTE = (
    "O mapa mostra onde estão os erros, mas não permite comparar regiões individuais."
)

NEIGHBOUR_NOTE = (
    "Regiões vizinhas não são independentes: não há margem de exclusão entre treino e teste "
    "(decisão da divisão do céu), então a correlação em escala de graus é medida, não removida."
)

COUNT_UNDER_BAND_NOTE = "O número sob cada faixa é o total de objetos."

SCATTER_OFF_NOTE = (
    "A dispersão por objeto fica desligada por padrão: ligá-la envia ao navegador um ponto "
    "por objeto desenhado."
)


def sample_note(n_test: str, n_pixels: str, prior: str) -> str:
    """Test-set size and the prior behind the quasar purities (arguments pre-formatted)."""
    return (
        f"Medido em {n_test} objetos de teste que ficaram de fora do treino, em {n_pixels} "
        "regiões do céu que o modelo também nunca viu. O acerto vale para a mistura do conjunto "
        f"de teste; as purezas de quasar valem num céu com prior {prior} (Delchambre+2023), "
        "não nas proporções do teste."
    )


def floor_note(min_objects: str, min_pixels: str) -> str:
    """Reporting floor of the evaluation, which is also the minimum shown on the sky map."""
    return (
        f"O piso de relato vale por célula ({min_objects} objetos em pelo menos {min_pixels} "
        "regiões) e abaixo dele a célula vira n/a; o mapa do céu usa o mesmo mínimo de objetos."
    )


def sky_scale_note(n_floor: str, by_rate: bool) -> str:
    """How the colour scale of the sky map is computed."""
    ends = (
        "do tom mais escuro (menor taxa de erro) ao mais claro (maior)"
        if by_rate
        else "do tom mais escuro (menos objetos) ao mais claro (mais objetos)"
    )
    return (
        f"Regiões de teste com ao menos {n_floor} objetos: {ends}, escala calculada só entre "
        "elas. O centro da Galáxia fica no meio da projeção."
    )


def bootstrap_note(n_boot: str) -> str:
    """Resampling used for the intervals of the Gaia DSC comparison."""
    return f"Bootstrap em blocos de pixel do céu, {n_boot} repetições."


def scatter_note(max_points: str, z_label: str) -> str:
    """How the per-galaxy scatter is sampled."""
    return (
        "Os acertos são sorteados com semente fixa e estratificados por faixa de redshift "
        f"(limite de {max_points} acertos, com a faixa {z_label} sempre inteira); "
        "todos os erros aparecem."
    )
