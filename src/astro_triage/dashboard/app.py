"""Streamlit dashboard. Run with:

    .venv\\Scripts\\python.exe -m streamlit run src/astro_triage/dashboard/app.py

Every number and sentence below is computed from one release in ``data/interim/<name>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from astro_triage.dashboard import charts as C
from astro_triage.dashboard import data as D

MODEL = "lgbm_full"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #0E1013; color: #D2D8E3; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2.2rem; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; color: #E8EAEE; letter-spacing: -0.01em; }
h1 { font-size: 2.6rem !important; line-height: 1.15 !important; }
h2 { font-size: 1.5rem !important; margin-top: 2rem; }
.small { color: #9AA3B5; font-size: 0.85rem; }
.err { color: #8EC5FF; }
.big { font-family: 'Fraunces', serif; font-weight: 300; font-size: 4.2rem; line-height: 1; color: #E8EAEE; }
.rule { border-top: 1px solid #232831; margin: 1.6rem 0; }
.notice { border-left: 2px solid #8EC5FF; padding: 0.2rem 0 0.2rem 0.9rem; color: #D2D8E3; }
button[data-baseweb="tab"] { font-size: 1rem; }
[data-testid="stExpander"] { border: 1px solid #232831; background: transparent; }
</style>
"""


def pt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def pt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}".replace(".", ",") + "%"


@st.cache_resource(show_spinner="Lendo a versão dos dados…")
def get_release(name: str) -> D.Release:
    return D.load_release(name)


@st.cache_data(show_spinner=False)
def get_pixels(name: str) -> pd.DataFrame:
    return D.pixel_table(get_release(name))


def available_releases() -> list[str]:
    found = sorted(
        p.parent.parent.name for p in D.INTERIM_DIR.glob("*/eval/results.json")
    )
    return found or [D.DEFAULT_RELEASE]


def provisional_notice(rel: D.Release) -> None:
    if rel.is_provisional:
        # Built from the release's own counts: the header text stored in older results.json
        # files (e.g. "100k sample") can be stale for a different release.
        floor = rel.meta["floor"]
        st.markdown(
            '<div class="notice small">Resultado provisório: '
            f"{pt_int(rel.meta['n_test'])} objetos de teste em {pt_int(rel.meta['n_test_pixels'])} "
            "regiões do céu, ainda sem aprovação humana das ADRs. Células com menos de "
            f"{floor['min_objects']} objetos em {floor['min_pixels']} regiões ficam como n/a.</div>",
            unsafe_allow_html=True,
        )


def overview(rel: D.Release) -> None:
    h = D.headline(rel, MODEL)
    st.title(f"Em cada 100 objetos, o modelo acerta {h.hits_per_100}.")
    st.markdown(
        f"<span class='small'>Medido em {pt_int(h.n_test)} objetos de teste que ficaram de fora do "
        f"treino, em {pt_int(h.n_test_pixels)} regiões do céu que o modelo também nunca viu.</span>",
        unsafe_allow_html=True,
    )
    provisional_notice(rel)

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.plotly_chart(
            C.waffle(h.hits_per_100),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            f"<span class='small'>● cinza: acerto &nbsp; <span class='err'>● azul: erro</span> "
            f"&nbsp; (arredondado de {pt_pct(h.accuracy, 2)})</span>",
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Quanto acerta cada tipo de objeto")
        conf = D.confusion(rel, MODEL)
        rows = []
        for cls in D.CLASSES:
            n, k = int(conf.loc[cls].sum()), int(conf.loc[cls, cls])
            lo, hi = D.wilson(k, n)
            rows.append({"cls": cls, "rate": k / n, "ci_lo": lo, "ci_hi": hi, "n": n})
        st.plotly_chart(
            C.recall_dots(pd.DataFrame(rows)),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            "<span class='small'>Cada ponto é a fração de objetos daquela classe que o modelo "
            "acertou. A linha é o intervalo de 95%.</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader("Onde o modelo erra")
    errs = D.errors_by_type(rel, MODEL)
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.plotly_chart(
            C.error_types(errs),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            "<span class='small'>● estrela · ■ galáxia · ▲ quasar. "
            "A seta vai da classe verdadeira para a classe que o modelo disse.</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"Dos {pt_int(h.n_errors)} erros, **{pt_pct(h.star_qso_error_share, 0)}** são confusão entre "
            "estrela e quasar. Isso não é falha do modelo só: no brilho e nas cores do Gaia, "
            "os dois objetos aparecem como um ponto. Separá-los de forma confiável exige espectro."
        )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader("O que este número não diz")
    mix = ", ".join(
        f"{C.CLASS_PT[c].lower()} {pt_pct(h.class_mix_train[c], 0)}" for c in D.CLASSES
    )
    st.markdown(
        f"- Os objetos têm rótulo espectroscópico do SDSS. Só entram os que o SDSS observou **e** "
        f"que casam com o Gaia, então a mistura ({mix}) não é a do céu.\n"
        f"- A acurácia acima vale para essa mistura. Num céu realista, com muito mais estrelas, o "
        f"resultado por classe muda. Por isso o painel de modelos mostra dois regimes.\n"
        f"- Só {pt_pct(h.dsc_coverage, 0)} dos objetos de teste têm classificação do Gaia DSC, o "
        f"baseline. A comparação com ele usa essa parte."
    )

    with st.expander("Ver detalhes técnicos"):
        technical(rel)


def technical(rel: D.Release) -> None:
    st.markdown("### Comparação com o Gaia DSC")
    pop, prior = "c_sdsc", "P0"
    st.markdown(
        f"<span class='small'>População (c): {pt_int(D.population_rows(rel, pop))} objetos de teste com "
        f"as três probabilidades do DSC. Regime equilibrado (as três classes com o mesmo peso). "
        f"Bootstrap em blocos de pixel, {rel.meta['n_boot']} repetições.</span>",
        unsafe_allow_html=True,
    )
    m = D.metrics_frame(rel, pop, prior, "prior_matched")
    f1 = m[m["metric"] == "macro_f1"]
    st.plotly_chart(
        C.f1_with_ci(f1, list(D.MODELS) + ["dsc_allosmod"]),
        width="stretch",
        config={"displayModeBar": False},
    )
    d = D.paired_diffs_frame(rel, pop, prior, "prior_matched")
    d = d[(d["metric"] == "macro_f1") & d["usable"]]
    st.markdown(
        "**Diferença pareada de F1 macro** (quando o intervalo não cruza o zero, a diferença é real):"
    )
    st.plotly_chart(
        C.paired_diffs(d), width="stretch", config={"displayModeBar": False}
    )

    st.markdown("### O que cada modelo usa")
    a, b = st.columns(2, gap="large")
    for col, model in zip((a, b), D.MODELS, strict=True):
        s = D.model_summary(rel, model)
        with col:
            st.markdown(f"**{C.MODEL_PT[model]}**")
            st.markdown(
                f"<span class='small'>{s['n_features']} variáveis · {s['best_iteration']} árvores · "
                f"temperatura {s['temperature']:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                C.importance(D.feature_importance(rel, model)),
                width="stretch",
                config={"displayModeBar": False},
            )

    st.markdown("### Confiança e calibração")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Quanto o modelo confia, e quanto erra**")
        st.plotly_chart(
            C.confidence_hist(D.confidence_histogram(rel, MODEL)),
            width="stretch",
            config={"displayModeBar": False},
        )
    with c2:
        st.markdown("**A probabilidade bate com a frequência?**")
        st.plotly_chart(
            C.calibration(D.calibration_points(rel, MODEL)),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.markdown("### Abstenção: deixar para o espectro o que o modelo não sabe")
    thr = st.slider("Limiar de confiança", 0.50, 0.99, 0.90, 0.01)
    curve = D.abstention_curve(rel, MODEL)
    row = curve.iloc[(curve["threshold"] - thr).abs().argmin()]
    st.markdown(
        f"Com limiar **{thr:.2f}**, o sistema decide {pt_pct(row.coverage, 1)} dos objetos, e acerta "
        f"**{pt_pct(row.accuracy_kept, 1)}** deles. Os {pt_int(row.n_set_aside)} objetos separados "
        f"contêm {pt_int(row.errors_set_aside)} dos {pt_int(row.errors_total)} erros."
    )
    st.plotly_chart(
        C.abstention(curve, thr),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.markdown("### Limitações registradas pela avaliação")
    for text in rel.results["limitations"]:
        st.markdown(f"- {text}")


def sky_page(rel: D.Release) -> None:
    px = get_pixels(rel.name)
    h = D.headline(rel, MODEL)
    stats = D.test_pixel_stats(rel, MODEL, px)
    st.title("Onde no céu o modelo erra")
    provisional_notice(rel)
    st.plotly_chart(
        C.sky_map(
            px, D.sky_points(rel, MODEL), stats, int(rel.meta["floor"]["min_objects"])
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.markdown(
        "<span class='small'>Projeção de Mollweide em coordenadas galácticas, centro da Galáxia no meio. "
        "Pontos escuros: regiões de treino; médios: validação; claros: teste. Em azul, os erros.</span>",
        unsafe_allow_html=True,
    )
    chance = D.chance_share_of_pixels_with_error(stats, h.n_errors / h.n_test)
    st.markdown(
        f"Das {pt_int(h.n_test_pixels)} regiões de teste, **{pt_int(h.n_pixels_with_error)}** "
        f"({pt_pct(h.n_pixels_with_error / h.n_test_pixels, 0)}) têm pelo menos um erro. Se os erros "
        f"caíssem ao acaso, esperaríamos cerca de {pt_pct(chance, 0)}. "
        f"A mediana é de {int(stats['n'].median())} objetos por região, abaixo do piso de "
        f"{rel.meta['floor']['min_objects']} que o projeto exige para afirmar algo por região, "
        "então o mapa mostra onde estão os erros, mas não permite concluir que uma região é pior que outra."
    )
    st.subheader("Por faixa de latitude galáctica")
    bands = D.lat_band_table(rel, MODEL, px)
    show = bands.assign(
        faixa=bands["band"].map(
            {"A": "A: perto do plano", "B": "B: intermediária", "C": "C: alta latitude"}
        ),
        taxa=[pt_pct(r) for r in bands["rate"]],
        intervalo=[
            f"{pt_pct(lo)} a {pt_pct(hi)}"
            for lo, hi in zip(bands["ci_lo"], bands["ci_hi"], strict=True)
        ],
    )[["faixa", "n_pixels", "n", "errors", "taxa", "intervalo"]]
    show.columns = ["faixa", "regiões", "objetos", "erros", "taxa de erro", "IC 95%"]
    st.dataframe(show, hide_index=True, width="stretch")
    st.markdown(
        "<span class='small'>A faixa A tem poucos objetos de teste: o intervalo largo diz que "
        "não dá para comparar.</span>",
        unsafe_allow_html=True,
    )


def galaxies_page(rel: D.Release) -> None:
    st.title("Galáxias: onde o modelo as confunde")
    provisional_notice(rel)
    t = rel.test[rel.test["true"] == "GALAXY"]
    n, k = len(t), int(t[f"err_{MODEL}"].sum())
    to_qso = int(((t[f"err_{MODEL}"]) & (t[f"pred_{MODEL}"] == "QSO")).sum())
    st.markdown(
        f"Das {pt_int(n)} galáxias de teste, o modelo errou **{pt_int(k)}** ({pt_pct(k / n)}): "
        f"{pt_int(to_qso)} viraram quasar e {pt_int(k - to_qso)} viraram estrela. "
        "O azul abaixo marca esses erros."
    )
    st.plotly_chart(
        C.galaxy_scatter(D.galaxy_scatter(rel, MODEL)),
        width="stretch",
        config={"displayModeBar": False},
    )
    c1, c2, c3 = st.columns(3, gap="large")
    for col, title, by, edges in (
        (c1, "Por magnitude G", "G", D.GALAXY_G_EDGES),
        (c2, "Por redshift z", "z", D.GALAXY_Z_EDGES),
        (c3, "Por solução astrométrica", "solved", None),
    ):
        with col:
            st.markdown(f"**{title}**")
            table = D.error_rate_table(rel, MODEL, "GALAXY", by, edges)
            st.plotly_chart(
                C.rate_bars(table),
                width="stretch",
                config={"displayModeBar": False},
            )
    st.markdown(
        "<span class='small'>Barras: taxa de erro. Linhas: intervalo de 95% (Wilson). "
        "Galáxias fracas (G alto) são as que o modelo mais acerta; as brilhantes e as de redshift "
        "alto são poucas e erradas com mais frequência. Compare o tamanho do intervalo antes de concluir.</span>",
        unsafe_allow_html=True,
    )

    st.subheader("Como são, a olho")
    if st.toggle(
        "Mostrar imagens do SDSS (o navegador consulta o site do SDSS)", value=False
    ):
        g = D.gallery_sample(rel, MODEL)
        cols = st.columns(4)
        for i, r in enumerate(g.itertuples()):
            with cols[i % 4]:
                st.image(r.url, width="stretch")
                state = (
                    "acerto"
                    if r.correct
                    else f"erro: viu {C.CLASS_PT.get(r.pred, r.pred).lower()}"
                )
                st.markdown(
                    f"<span class='small {'' if r.correct else 'err'}'>{state} · confiança "
                    f"{r.confidence:.2f} · z={r.sdss_z:.3f}</span>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            "<span class='small'>Sorteio com semente fixa, quatro acertos e quatro erros. "
            "Imagens: SDSS SkyServer; confira os termos de uso antes de publicar.</span>",
            unsafe_allow_html=True,
        )


def pt_num(x: float, digits: int = 3, signed: bool = False) -> str:
    return f"{x:{'+' if signed else ''}.{digits}f}".replace(".", ",")


def pt_interval(
    point: float, lo: float, hi: float, mark: str, signed: bool = False
) -> str:
    if mark:
        return mark
    return (
        f"{pt_num(point, 3, signed)} (IC 95%: {pt_num(lo, 3, signed)} a "
        f"{pt_num(hi, 3, signed)})"
    )


def layer2_page(layer2: dict) -> None:
    """Layer 2 section. Only fixed text and numbers go through markdown; model output never."""
    s = D.layer2_summary(layer2)
    st.title("Camada 2: o modelo de linguagem ajuda?")
    st.markdown(
        "<span class='small'>Regra fixada antes de ver o teste: há ganho se o intervalo pareado de "
        "95% da diferença de F1 macro fica inteiro acima de zero em algum dos dois pares "
        "principais (S2 contra S0 e S3 contra S1*).</span>",
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
            f"- **{r.pair}**: {pt_interval(r.point, r.lo, r.hi, r.mark, signed=True)}"
            f"{'' if r.evaluable else ' (célula abaixo do piso ou indefinida: nada a afirmar)'}"
        )
    usable = pairs[pairs["usable"]]
    if len(usable):
        st.plotly_chart(
            C.paired_diffs(usable), width="stretch", config={"displayModeBar": False}
        )
    if s["n_evaluable"] < s["n_pairs"]:
        st.markdown(
            f"<span class='small'>{s['n_pairs'] - s['n_evaluable']} de {s['n_pairs']} pares ficam "
            f"como n/a: menos de {s['floor']['min_objects']} objetos em "
            f"{s['floor']['min_pixels']} regiões por classe.</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader("Os cinco sistemas, no teste inteiro")
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
                pt_interval(r.accuracy, r.accuracy_lo, r.accuracy_hi, r.accuracy_mark)
                for r in sy.itertuples()
            ],
            "F1 macro": [
                pt_interval(r.macro_f1, r.macro_f1_lo, r.macro_f1_hi, r.macro_f1_mark)
                for r in sy.itertuples()
            ],
            "cobertura": [
                pt_interval(r.coverage, r.coverage_lo, r.coverage_hi, r.coverage_mark)
                for r in sy.itertuples()
            ],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(C.layer2_f1(sy), width="stretch", config={"displayModeBar": False})
    st.markdown(
        f"<span class='small'>Intervalos: bootstrap em blocos de pixel, {s['n_boot']} repetições, "
        "as mesmas repetições para todos os sistemas. Células abaixo do piso aparecem como n/a.</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader("Só os objetos encaminhados à Camada 2")
    mix = " · ".join(
        f"{C.GLYPH[c]} {C.CLASS_PT[c].lower()} {pt_int(s['class_mix'][c])}"
        for c in D.CLASSES
    )
    st.markdown(
        f"{pt_int(s['n_routed'])} objetos de teste de baixa confiança da Camada 1: {mix}. "
        "É a cauda difícil do teste, e nada aqui vale fora dela."
    )
    ro = s["routed"]
    st.dataframe(
        pd.DataFrame(
            {
                "visão": ro["unit"],
                "objetos": ro["n"],
                "respondidos": ro["n_answered"],
                "acerto": [
                    pt_interval(
                        r.accuracy, r.accuracy_lo, r.accuracy_hi, r.accuracy_mark
                    )
                    for r in ro.itertuples()
                ],
                "F1 macro": [
                    pt_interval(
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
        f"Taxa de UNKNOWN: **{pt_pct(s['unknown_rate'])}**. Taxa de falha (sem resposta válida e "
        f"não UNKNOWN): **{pt_pct(s['failure_rate'])}**. Todo objeto encaminhado conta no denominador."
    )
    stt = s["status"]
    st.dataframe(
        pd.DataFrame(
            {
                "status": stt["status"],
                "tipo": stt["kind"],
                "objetos": stt["n"],
                "taxa": [pt_pct(x) for x in stt["rate"]],
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
                "média": [pt_num(x, 1) for x in cost["mean"]],
                "mediana": [pt_num(x, 1) for x in cost["p50"]],
                "p95": [pt_num(x, 1) for x in cost["p95"]],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Origem da resposta")
    st.text(
        f"modelo pedido: {s['model_requested']}\n"
        f"modelo informado: {', '.join(f'{k} ({v})' for k, v in s['models_reported'].items())}\n"
        f"versão do prompt: {s['prompt_version']}"
    )


def main() -> None:
    st.set_page_config(page_title="astro-triage", page_icon="✦", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### astro-triage")
        releases = available_releases()
        default = releases.index("full_v0") if "full_v0" in releases else 0
        name = st.selectbox("Versão dos dados", releases, index=default)
    try:
        rel = get_release(name)
    except FileNotFoundError as exc:
        st.error(f"Arquivos da versão não encontrados: {exc}")
        st.stop()
    with st.sidebar:
        st.markdown(
            f"<span class='small'>{pt_int(len(rel.test))} objetos de teste"
            f"{' · provisório' if rel.is_provisional else ''}</span>",
            unsafe_allow_html=True,
        )
    try:
        layer2 = D.load_layer2(rel)
    except (OSError, ValueError, KeyError) as exc:
        layer2 = None
        st.sidebar.error(f"Avaliação da Camada 2 ilegível: {exc!r}")
    titles = ["Visão geral", "Mapa do céu", "Galáxias"]
    tabs = st.tabs([*titles, "Camada 2"] if layer2 is not None else titles)
    with tabs[0]:
        overview(rel)
    with tabs[1]:
        sky_page(rel)
    with tabs[2]:
        galaxies_page(rel)
    if layer2 is not None:
        with tabs[3]:
            layer2_page(layer2)


main()
