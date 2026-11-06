"""Layer 2 tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from astro_triage.dashboard import charts as C
from astro_triage.dashboard import data as D
from astro_triage.dashboard import fmt
from astro_triage.dashboard.constants import NO_BAR


def layer2_page(layer2: dict) -> None:
    """Layer 2 section. Only fixed text and numbers go through markdown; model output never."""
    s = D.layer2_summary(layer2)
    st.title("Camada 2: o modelo de linguagem ajuda?")
    st.markdown(
        "<span class='small'>Regra fixada antes da avaliação: há ganho se o intervalo pareado "
        f"({fmt.pt_ci_label()}) da diferença de F1 macro fica inteiro acima de zero em algum dos "
        "dois pares principais (S2 contra S0 e S3 contra S1*).</span>",
        unsafe_allow_html=True,
    )
    pairs = s["pairs"]
    if s["outcome"] == "gain":
        n_gain = int(pairs["gain"].sum())
        st.subheader(
            f"Resultado: diferença positiva em F1 macro em {n_gain} de "
            f"{s['n_pairs']} comparações"
        )
    else:
        st.subheader("Resultado: a Camada 2 não agregou valor mensurável")
    for r in pairs.itertuples():
        st.markdown(
            f"- **{r.pair}**: {fmt.pt_interval(r.point, r.lo, r.hi, r.mark, signed=True)}"
        )
    usable = pairs[pairs["usable"]]
    if len(usable):
        st.plotly_chart(C.paired_diffs(usable), width="stretch", config=NO_BAR)
    if s["n_evaluable"] < s["n_pairs"]:
        st.markdown(
            f"<span class='small'>{s['n_pairs'] - s['n_evaluable']} de {s['n_pairs']} pares ficam "
            f"como n/a: menos de {s['floor']['min_objects']} objetos em "
            f"{s['floor']['min_pixels']} regiões por classe.</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader("Sistemas comparados")
    st.markdown(
        "<span class='small'>S0: só a Camada 1. S1: a Camada 1, abstendo-se nos objetos "
        "encaminhados. S1*: o mesmo com limiar ajustado na validação. S2: Camada 1 mais "
        "Camada 2, com a Camada 1 de reserva. S3: Camada 1 mais Camada 2, com abstenção.</span>",
        unsafe_allow_html=True,
    )
    sy = s["systems"]
    table = pd.DataFrame(
        {
            "sistema": sy["unit"],
            "acerto": [
                fmt.pt_interval(
                    r.accuracy, r.accuracy_lo, r.accuracy_hi, r.accuracy_mark
                )
                for r in sy.itertuples()
            ],
            "F1 macro": [
                fmt.pt_interval(
                    r.macro_f1, r.macro_f1_lo, r.macro_f1_hi, r.macro_f1_mark
                )
                for r in sy.itertuples()
            ],
            "cobertura": [
                fmt.pt_interval(
                    r.coverage, r.coverage_lo, r.coverage_hi, r.coverage_mark
                )
                for r in sy.itertuples()
            ],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(C.layer2_f1(sy), width="stretch", config=NO_BAR)
    st.markdown(
        f"<span class='small'>Intervalos: bootstrap em blocos de pixel, {s['n_boot']} repetições, "
        "as mesmas repetições para todos os sistemas.</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader("Só os objetos encaminhados à Camada 2")
    mix = " · ".join(
        f"{C.GLYPH[c]} {C.CLASS_PT[c].lower()} {fmt.pt_int(s['class_mix'][c])}"
        for c in D.CLASSES
    )
    st.markdown(
        f"{fmt.pt_int(s['n_routed'])} objetos avaliados de baixa confiança da Camada 1: {mix}. "
        "É a cauda difícil do conjunto avaliado, e nada aqui vale fora dela."
    )
    ro = s["routed"]
    st.dataframe(
        pd.DataFrame(
            {
                "visão": ro["unit"],
                "objetos": ro["n"],
                "respondidos": ro["n_answered"],
                "acerto": [
                    fmt.pt_interval(
                        r.accuracy, r.accuracy_lo, r.accuracy_hi, r.accuracy_mark
                    )
                    for r in ro.itertuples()
                ],
                "F1 macro": [
                    fmt.pt_interval(
                        r.macro_f1, r.macro_f1_lo, r.macro_f1_hi, r.macro_f1_mark
                    )
                    for r in ro.itertuples()
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader("Quando a Camada 2 não responde")
    st.markdown(
        f"Sem resposta (UNKNOWN): **{fmt.pt_pct(s['unknown_rate'])}**. Taxa de falha (sem resposta válida e "
        f"não UNKNOWN): **{fmt.pt_pct(s['failure_rate'])}**. Todo objeto encaminhado conta no denominador."
    )
    stt = s["status"]
    st.dataframe(
        pd.DataFrame(
            {
                "status": stt["status"],
                "tipo": stt["kind"],
                "objetos": stt["n"],
                "taxa": [fmt.pt_pct(x) for x in stt["rate"]],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Custo por objeto encaminhado")
    cost = s["cost"]
    st.dataframe(
        pd.DataFrame(
            {
                "item": cost["item"],
                "objetos": cost["n"],
                "média": [fmt.pt_num(x, 1) for x in cost["mean"]],
                "mediana": [fmt.pt_num(x, 1) for x in cost["p50"]],
                "p95": [fmt.pt_num(x, 1) for x in cost["p95"]],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    with st.expander("Origem da resposta", expanded=False):
        st.text(
            f"modelo pedido: {s['model_requested']}\n"
            f"modelo informado: {', '.join(f'{k} ({v})' for k, v in s['models_reported'].items())}\n"
            f"versão do prompt: {s['prompt_version']}"
        )
