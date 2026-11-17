"""Pure HTML kit of the 'prior ruler' redesign: dot field, rulers, seal, verdict line, CSS.

No UI-framework and no plotting import. Callers pass plain values (float or None); every text
argument of an HTML-returning function is escaped. ``verdict_line`` is the exception: it
returns plain, unescaped text; ``verdict_html`` wraps it escaped. HTML output is a single
line so Markdown keeps it as HTML.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape
from itertools import pairwise

from astro_triage.dashboard import fmt, theme

NA = "n/a"
MONO = "ui-monospace, Menlo, Consolas, monospace"
SEAL_PREFIX = "PROVISÓRIO"
FIELD_N = 1000
FIELD_COLS = 40
FIELD_LABEL = "pontos são quasares"
DOT_GAP_PX = 3
PERCENT = 100.0
HALF = 50.0
DECADE = 1
EPS = 1e-9
SCALE_PAD = 1.15
BAND_ALPHA = 0.16
CI_ALPHA = 0.55
TICK_DIGITS = 4
CHAR_PX = 9.0
MIN_TRACK_PX = 280
PAIR_TEXT = {"tie": "empate", "win": "vitória do par"}
# Vocabulary of dl_compare.pair_verdict: "X wins" or "tie: LightGBM stays". "win" is kept
# as a compatibility alias; anything else is unknown (n/a).
PAIR_ALIAS = {"x wins": "win", "win": "win", "tie": "tie"}
CLASS_JOIN = " e "
WIN_PHRASE = "vence"
WORSE_PHRASE = "fica atrás"
ONLY = " só"
WIN_TAG = "(classe)"
WORSE_TAG = "(classe, descritivo)"
PAIR_NEEDS = "o par exige as duas classes"
SEAL_TRACK = 0.04
HEAD_WEIGHT = 600
ONE = 1
MEDIA_PX = 1000
PAGE_LANG = "pt-BR"
NARROW_VIEWPORT_PX = 390
WIDE_VIEWPORT_PX = 1440
PAGE_PAD_PX = 16
PANEL_GAP_PX = 32
NAME_COL_PX = 72
ROW_GAP_PX = 12
POINT_PX = 18
BAND_MIN_PX = 6
CI_MIN_PX = 2
LABEL_GAP_PX = 4
VLINE_MIN_PX = 26
VLINE_VW = 3.6
VLINE_MAX_PX = 44
DELTA = "δ"
CARD_LEVEL = 3


def _t(text: object) -> str:
    return escape(str(text), quote=True)


def _valid(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _px(n: float) -> str:
    return f"{n}px"


def _pc(n: float) -> str:
    return f"{n}%"


def _r(selector: str, **decls: object) -> str:
    body = ";".join(f"{k.replace('_', '-')}:{v}" for k, v in decls.items())
    return f"{selector}{{{body}}}"


def seal(release: str) -> str:
    return f"<div class='hs-seal'>{SEAL_PREFIX} · {_t(release)}</div>"


def _na_block() -> str:
    return f"<div class='hs-na'>{NA}</div>"


def blue_count(purity: float | None, n: int = FIELD_N) -> int | None:
    if not _valid(purity):
        return None
    return round(n * min(1.0, max(0.0, float(purity))))


def dot_field(
    purity: float | None,
    n: int = FIELD_N,
    cols: int = FIELD_COLS,
    label: str = FIELD_LABEL,
) -> str:
    blue = blue_count(purity, n)
    if blue is None:
        return _na_block()
    dots = "<i class='hs-q'></i>" * blue + "<i class='hs-n'></i>" * (n - blue)
    aria = f"{fmt.pt_int(blue)} de {fmt.pt_int(n)} {label}"
    return (
        f"<div class='hs-field' style='--hs-cols:{int(cols)}' role='img' "
        f"aria-label='{_t(aria)}'>{dots}</div>"
    )


def log_position(q: float | None, lo: float, hi: float) -> float | None:
    """Position of ``q`` on a log scale from ``lo`` to ``hi``, in percent, clamped."""
    if not _valid(q) or q <= 0 or lo <= 0 or hi <= lo:
        return None
    span = math.log10(hi) - math.log10(lo)
    pos = (math.log10(q) - math.log10(lo)) / span * PERCENT
    return max(0.0, min(PERCENT, pos))


def _scale(valid: Sequence[float], lo: float | None, hi: float | None):
    if lo is None or hi is None:
        if not valid:
            return None
        lo = 10.0 ** (math.floor(math.log10(min(valid)) + EPS) - DECADE)
        hi = 10.0 ** (math.ceil(math.log10(max(valid)) - EPS) + DECADE)
    return (lo, hi) if 0 < lo < hi else None


def prior_ruler(
    priors: Mapping[str, float | None],
    selected: str | None = None,
    lo: float | None = None,
    hi: float | None = None,
    callouts: Mapping[str, str] | None = None,
) -> str:
    marks = {k: v for k, v in priors.items() if k != "P0"}
    valid = [v for v in marks.values() if _valid(v) and v > 0]
    scale = _scale(valid, lo, hi)
    if scale is None:
        return _na_block()
    lo, hi = scale
    first = math.ceil(math.log10(lo) - EPS)
    last = math.floor(math.log10(hi) + EPS)
    ticks = "".join(
        f"<span class='t' style='left:{log_position(10.0**k, lo, hi):.2f}%'>"
        f"{_t(fmt.pt_g(10.0**k * PERCENT) + '%')}</span>"
        for k in range(first, last + 1)
    )
    drawn, off = [], []
    for name, q in marks.items():
        x = log_position(q, lo, hi) if _valid(q) else None
        if x is None:
            off.append(
                f"<span class='hs-off'><i class='hs-mk off'></i>{_t(name)} · {NA}</span>"
            )
            continue
        on = " on" if name == selected else ""
        cur = " aria-current='true'" if on else ""
        drawn.append(
            f"<div class='mkl' style='left:{x:.2f}%'>{_t(name)}</div>"
            f"<div class='mk{on}' style='left:{x:.2f}%'{cur}></div>"
        )
        if callouts and name in callouts:
            drawn.append(
                f"<div class='cal' style='left:{x:.2f}%'>{_t(callouts[name])}</div>"
            )
    legend = f"<div class='hs-offs'>{''.join(off)}</div>" if off else ""
    return (
        "<div class='hs-rl-scroll'><div class='hs-rl'><div class='axis'></div>"
        f"{ticks}{''.join(drawn)}</div></div>{legend}"
    )


def tick_boxes(labels: Sequence[str], track_px: float) -> list[tuple[float, float]]:
    """Estimated (start, end) of each label; first anchored left, last right, others centred."""
    count = len(labels)
    boxes = []
    for i, text in enumerate(labels):
        w = len(text) * CHAR_PX
        frac = i / (count - 1) if count > 1 else 0.5
        if i == 0 and count > 1:
            start = 0.0
        elif i == count - 1 and count > 1:
            start = track_px - w
        else:
            start = track_px * frac - w / 2
        boxes.append((start, start + w))
    return boxes


def _overlap(labels: Sequence[str]) -> bool:
    boxes = tick_boxes(labels, MIN_TRACK_PX)
    return any(a[1] >= b[0] for a, b in pairwise(boxes))


def _tick_row(labels: list[str]) -> str:
    keep = labels if not _overlap(labels) else [labels[0], labels[-1]]
    if _overlap(keep):
        return ""
    n = len(keep)
    spans = "".join(
        f"<span class='{'l' if i == 0 else 'r' if i == n - 1 else 'c'}' "
        f"style='left:{PERCENT * i / (n - 1):.2f}%'>{_t(t)}</span>"
        for i, t in enumerate(keep)
    )
    return f"<div class='hs-dr'><div></div><div class='hs-ax' aria-hidden='true'>{spans}</div></div>"


def _pos(value: float, half: float) -> float:
    return max(0.0, min(PERCENT, HALF + value / half * HALF))


def _diff_row(row: Mapping, half: float) -> str:
    d, lo, hi, delta = (row.get(k) for k in ("d", "lo", "hi", "delta"))
    marks = ""
    if _valid(delta):
        a, b = _pos(-delta, half), _pos(delta, half)
        marks += f"<div class='band' style='left:{a:.2f}%;width:{b - a:.2f}%'></div>"
    marks += "<div class='lo'></div><div class='zero'></div>"
    if _valid(lo) and _valid(hi):
        a, b = _pos(lo, half), _pos(hi, half)
        marks += f"<div class='ci' style='left:{a:.2f}%;width:{b - a:.2f}%'></div>"
    if _valid(d):
        marks += f"<div class='pt' style='left:{_pos(d, half):.2f}%'></div>"
    return (
        f"<div class='hs-dr'><div class='dn'>{_t(row.get('name', ''))}</div>"
        f"<div class='trk'>{marks}</div>"
        f"<div class='dv'>{_t(row.get('detail', ''))}</div></div>"
    )


def _diff_panel(panel: Mapping) -> str:
    rows = list(panel.get("rows") or ())
    vals = [
        abs(v)
        for r in rows
        for v in (r.get(k) for k in ("d", "lo", "hi", "delta"))
        if _valid(v)
    ]
    head = f"<div class='hs-pt'>{_t(panel.get('title') or '')}</div>"
    half = max(vals) * SCALE_PAD if vals else 0.0
    if half <= 0:
        return f"<section class='hs-panel'>{head}{_na_block()}</section>"
    shown = fmt.pt_num(half, TICK_DIGITS)
    zero = fmt.pt_num(0.0, 0)
    ticks = _tick_row([f"−{shown}", zero, f"+{shown}"])
    body = "".join(_diff_row(r, half) for r in rows)
    return f"<section class='hs-panel'>{head}{body}{ticks}</section>"


def diff_ruler(panels: Sequence[Mapping]) -> str:
    return f"<div class='hs-panels'>{''.join(_diff_panel(p) for p in panels)}</div>"


def _pair_key(raw: object) -> str:
    head = str(raw or "").partition(":")[0].strip().lower()
    return PAIR_ALIAS.get(head, "")


def verdict_line(pairs: Sequence[Mapping]) -> str:
    """Plain, UNESCAPED text (for aria labels and tests). Use ``verdict_html`` for markup."""
    if not pairs:
        return NA
    heads, extra = [], []
    wins_cited = False
    for item in pairs:
        fam = str(item.get("family", ""))
        key = _pair_key(item.get("pair_verdict"))
        heads.append(f"{fam}: {PAIR_TEXT.get(key, NA)}")
        classes = item.get("classes") or {}
        for flag, phrase, tag in (
            ("win", WIN_PHRASE, WIN_TAG),
            ("worse", WORSE_PHRASE, WORSE_TAG),
        ):
            names = [str(c) for c, v in classes.items() if v == flag]
            if not names or (flag == "win" and key == "win"):
                continue
            only = "" if len(names) == len(classes) else ONLY
            extra.append(f"{fam} {phrase}{only} em {CLASS_JOIN.join(names)} {tag}")
            wins_cited = wins_cited or flag == "win"
    if wins_cited:
        extra.append(PAIR_NEEDS)
    line = " · ".join(heads) + "."
    if extra:
        line += f" {'; '.join(extra)}."
    return line


def verdict_html(pairs: Sequence[Mapping]) -> str:
    """The verdict line escaped inside a single element; family and class names are text."""
    return f"<div class='hs-verdict'>{_t(verdict_line(pairs))}</div>"


def lang_root(body_html: str) -> str:
    """Wrap trusted markup in a container that declares the page language."""
    return f"<div class='hs-page' lang='{PAGE_LANG}'>{body_html}</div>"


def verdict_oneline(text: str) -> str:
    """The one-line verdict in italic serif; ``text`` is escaped."""
    return f"<p class='hs-vline'>{_t(text)}</p>"


def track_px(viewport_px: float) -> float:
    """Width of a ruler track at a viewport width (the layout model of ``system_css``)."""
    page = min(viewport_px, theme.WIDE_WIDTH_PX) - 2 * PAGE_PAD_PX
    panel = page if viewport_px <= MEDIA_PX else (page - PANEL_GAP_PX) / 2
    return panel - NAME_COL_PX - ROW_GAP_PX


def pair_card(title: str, verdict: str, rows: Sequence[Mapping], warn: str = "") -> str:
    """Card of one pair. The DOM is the same for every pair: title, verdict, one block per
    class (name, verdict, numbers, note) and a warning line that is empty when not needed."""
    blocks = "".join(
        "<div class='hs-card-r'>"
        f"<span class='k'>{_t(r.get('name', ''))}</span>"
        f"<span class='v'>{_t(r.get('verdict', ''))}</span>"
        f"<span class='n'>{_t(r.get('numbers', ''))}</span>"
        f"<span class='w'>{_t(r.get('note', ''))}</span></div>"
        for r in rows
    )
    return (
        f"<article class='hs-card'><div class='hs-card-t' role='heading' aria-level='{CARD_LEVEL}'>"
        f"{_t(title)}</div><p class='hs-card-v'>{_t(verdict)}</p>{blocks}"
        f"<p class='hs-card-x' role='note'>{_t(warn)}</p></article>"
    )


def cards(items_html: Sequence[str]) -> str:
    return f"<div class='hs-cards'>{''.join(items_html)}</div>"


def _delta_row(row: Mapping, half: float, delta: float, two_marks: bool) -> str:
    d, lo, hi = (row.get(k) for k in ("d", "lo", "hi"))
    a, b = _pos(-delta, half), _pos(delta, half)
    if two_marks:
        marks = f"<div class='dm' style='left:{a:.2f}%'></div>"
        marks += f"<div class='dm' style='left:{b:.2f}%'></div>"
    else:
        marks = f"<div class='band' style='left:{a:.2f}%;width:{b - a:.2f}%'></div>"
    marks += "<div class='lo'></div><div class='zero'></div>"
    if _valid(lo) and _valid(hi):
        x, y = _pos(lo, half), _pos(hi, half)
        marks += f"<div class='ci' style='left:{x:.2f}%;width:{y - x:.2f}%'></div>"
    if _valid(d):
        marks += f"<div class='pt' style='left:{_pos(d, half):.2f}%'></div>"
    detail = _t(row.get("detail", ""))
    return (
        f"<div class='hs-dr hs-dx'><div class='dn'>{_t(row.get('name', ''))}</div>"
        f"<div class='trk' role='img' aria-label='{detail}'>{marks}</div>"
        f"<div class='dv'>{detail}</div></div>"
    )


def delta_ruler(panel: Mapping) -> str:
    """One ruler: own symmetric scale, band of +-delta, interval and point D.

    The axis carries only -delta, 0 and +delta. When the band would be narrower than
    ``BAND_MIN_PX`` on the narrowest layout, delta is drawn as two labelled marks instead.
    """
    rows = list(panel.get("rows") or ())
    delta = panel.get("delta")
    head = f"<div class='hs-pt'>{_t(panel.get('title') or '')}</div>"
    vals = [
        abs(v) for r in rows for v in (r.get(k) for k in ("d", "lo", "hi")) if _valid(v)
    ]
    if not _valid(delta) or delta <= 0 or not vals:
        return f"<section class='hs-panel'>{head}{_na_block()}</section>"
    half = max(*vals, delta) * SCALE_PAD
    a, b = _pos(-delta, half), _pos(delta, half)
    narrow = track_px(NARROW_VIEWPORT_PX)
    two_marks = (b - a) / PERCENT * narrow < BAND_MIN_PX
    low, high = f"−{DELTA}", f"+{DELTA}"
    stacked = (b - HALF) / PERCENT * narrow < len(high) * CHAR_PX + LABEL_GAP_PX
    cls = " class='u'" if stacked else ""
    spans = (
        f"<span{cls} style='left:{a:.2f}%'>{low}</span>"
        f"<span style='left:{HALF:.2f}%'>{fmt.pt_num(0.0, 0)}</span>"
        f"<span{cls} style='left:{b:.2f}%'>{high}</span>"
    )
    two = " two" if stacked else ""
    axis = (
        f"<div class='hs-dr'><div></div><div class='hs-ax{two}' aria-hidden='true'>"
        f"{spans}</div></div>"
    )
    body = "".join(_delta_row(r, half, delta, two_marks) for r in rows)
    return f"<section class='hs-panel'>{head}{body}{axis}</section>"


def delta_rulers(panels: Sequence[Mapping]) -> str:
    return f"<div class='hs-panels'>{''.join(delta_ruler(p) for p in panels)}</div>"


def abs_bars(rows: Sequence[Mapping]) -> str:
    """Bars from zero to the median with a min-max whisker; one scale, only '0' labelled."""
    highs = [r["high"] for r in rows if _valid(r.get("high"))]
    if not highs or max(highs) <= 0:
        return _na_block()
    top = max(highs) * SCALE_PAD
    out = []
    for r in rows:
        low, mid, high = (r.get(k) for k in ("low", "median", "high"))
        marks = "<div class='lo'></div>"
        if all(_valid(v) for v in (low, mid, high)):
            x, y = low / top * PERCENT, high / top * PERCENT
            marks += f"<div class='bar' style='width:{mid / top * PERCENT:.2f}%'></div>"
            marks += f"<div class='ci' style='left:{x:.2f}%;width:{y - x:.2f}%'></div>"
        detail = _t(r.get("detail", ""))
        out.append(
            f"<div class='hs-dr hs-ab'><div class='dn'>{_t(r.get('name', ''))}</div>"
            f"<div class='trk' role='img' aria-label='{detail}'>{marks}</div>"
            f"<div class='dv'>{detail}</div></div>"
        )
    axis = (
        "<div class='hs-dr'><div></div><div class='hs-ax' aria-hidden='true'>"
        f"<span class='l' style='left:{0.0:.2f}%'>{fmt.pt_num(0.0, 0)}</span></div></div>"
    )
    return f"<section class='hs-panel'>{''.join(out)}{axis}</section>"


def system_css() -> str:
    t = theme
    hair = f"{_px(1)} solid"
    ring = f"inset {_px(0)} {_px(0)} {_px(0)}"
    small = f"{_px(t.MIN_SECONDARY_PX)} {MONO}"
    rules = [
        _r(
            ".hs-seal",
            border=f"{hair} {t.ACCENT}",
            padding=f"{_px(2)} {_px(10)}",
            font=small,
            color=t.TEXT,
            white_space="nowrap",
            letter_spacing=f"{SEAL_TRACK}em",
            display="inline-block",
        ),
        _r(".hs-na", color=t.TEXT_SECONDARY, font=small),
        _r(
            ".hs-field",
            display="grid",
            grid_template_columns=f"repeat(var(--hs-cols,{FIELD_COLS}),{ONE}fr)",
            gap=_px(DOT_GAP_PX),
            min_width=_px(0),
        ),
        _r(
            ".hs-field i",
            aspect_ratio=1,
            border_radius=_pc(HALF),
            box_shadow=f"{ring} {_px(1)} {t.DASH}",
        ),
        _r(".hs-field i.hs-q", background=t.ACCENT, box_shadow="none"),
        _r(".hs-rl-scroll", overflow_x="auto", margin=f"{_px(16)} {_px(0)}"),
        _r(
            ".hs-rl",
            position="relative",
            min_width=_px(640),
            height=_px(150),
            margin=f"{_px(0)} {_px(44)}",
        ),
        _r(
            ".hs-rl .axis",
            position="absolute",
            left=_px(0),
            right=_px(0),
            top=_px(80),
            border_top=f"{hair} {t.SOFT}",
        ),
        _r(
            ".hs-rl .t",
            position="absolute",
            top=_px(92),
            transform=f"translateX(-{_pc(HALF)})",
            font=small,
            color=t.TEXT_SECONDARY,
        ),
        _r(
            ".hs-rl .t::before",
            content='""',
            position="absolute",
            left=_pc(HALF),
            top=_px(-12),
            height=_px(8),
            border_left=f"{hair} {t.SOFT}",
        ),
        _r(
            ".hs-rl .mk,.hs-mk",
            position="absolute",
            top=_px(73),
            width=_px(14),
            height=_px(14),
            margin_left=_px(-7),
            border_radius=_pc(HALF),
            background="transparent",
            box_shadow=f"{ring} {_px(2)} {t.ACCENT}",
        ),
        _r(".hs-rl .mk.on", background=t.ACCENT),
        _r(".hs-rl .mk.off,.hs-mk.off", box_shadow=f"{ring} {_px(1)} {t.DASH}"),
        _r(
            ".hs-rl .mkl",
            position="absolute",
            top=_px(46),
            transform=f"translateX(-{_pc(HALF)})",
            font=small,
            color=t.TEXT,
        ),
        _r(
            ".hs-rl .cal",
            position="absolute",
            top=_px(0),
            transform=f"translateX(-{_pc(HALF)})",
            white_space="nowrap",
            font=small,
            color=t.ACCENT,
        ),
        _r(
            ".hs-rl .cal::after",
            content='""',
            position="absolute",
            left=_pc(HALF),
            top=_pc(PERCENT),
            height=_px(24),
            border_left=f"{hair} {t.ACCENT}",
        ),
        _r(
            ".hs-offs",
            display="flex",
            flex_wrap="wrap",
            gap=_px(16),
            font=small,
            color=t.TEXT_SECONDARY,
        ),
        _r(".hs-off", display="inline-flex", align_items="center", gap=_px(8)),
        _r(".hs-off .hs-mk", position="static", margin=_px(0), display="inline-block"),
        _r(
            ".hs-panels",
            display="grid",
            grid_template_columns=f"{ONE}fr {ONE}fr",
            gap=_px(32),
        ),
        _r(
            ".hs-panel",
            border_top=f"{hair} {t.HAIRLINE}",
            padding_top=_px(16),
            min_width=_px(0),
        ),
        _r(
            ".hs-pt",
            font=f"{HEAD_WEIGHT} {_px(t.HEADING_PX)} {t.FONT_SANS}",
            color=t.TEXT,
        ),
        _r(
            ".hs-dr",
            display="grid",
            grid_template_columns=f"{_px(NAME_COL_PX)} minmax({_px(0)},{ONE}fr)",
            gap=f"{_px(4)} {_px(ROW_GAP_PX)}",
            margin_top=_px(20),
            align_items="center",
        ),
        _r(".hs-dr .dn", font_size=_px(t.CONTROL_PX), color=t.SOFT),
        _r(".hs-dr .dv", grid_column=2, font=small, color=t.TEXT_SECONDARY),
        _r(".hs-dr .trk", position="relative", height=_px(36)),
        _r(
            ".hs-dr .zero",
            position="absolute",
            left=_pc(HALF),
            top=_px(0),
            bottom=_px(0),
            border_left=f"{hair} {t.TEXT_SECONDARY}",
        ),
        _r(
            ".hs-dr .band",
            position="absolute",
            top=_px(0),
            bottom=_px(0),
            background=t.ACCENT,
            opacity=BAND_ALPHA,
        ),
        _r(
            ".hs-dr .ci",
            position="absolute",
            top=_px(13),
            height=_px(10),
            background=t.ACCENT,
            opacity=CI_ALPHA,
        ),
        _r(
            ".hs-dr .pt",
            position="absolute",
            top=_px(9),
            width=_px(POINT_PX),
            height=_px(POINT_PX),
            margin_left=_px(-POINT_PX // 2),
            border_radius=_pc(HALF),
            background=t.ACCENT,
            box_shadow=f"{_px(0)} {_px(0)} {_px(0)} {_px(2)} {t.BG}",
        ),
        _r(
            ".hs-dr .lo",
            position="absolute",
            left=_px(0),
            right=_px(0),
            top=_pc(HALF),
            border_top=f"{hair} {t.HAIRLINE}",
        ),
        _r(
            ".hs-ax",
            position="relative",
            height=_px(24),
            font=small,
            color=t.TEXT_SECONDARY,
        ),
        _r(
            ".hs-ax span",
            position="absolute",
            top=_px(2),
            white_space="nowrap",
            transform=f"translateX(-{_pc(HALF)})",
        ),
        _r(".hs-ax span.l", transform="none"),
        _r(".hs-ax span.r", transform=f"translateX(-{_pc(PERCENT)})"),
        _r(".hs-dx .band", min_width=_px(BAND_MIN_PX)),
        _r(".hs-dx .ci", min_width=_px(CI_MIN_PX)),
        _r(
            ".hs-dr .dm",
            position="absolute",
            top=_px(0),
            bottom=_px(0),
            border_left=f"{_px(2)} solid {t.ACCENT}",
        ),
        _r(".hs-ax.two", height=_px(48)),
        _r(".hs-ax span.u", top=_px(24)),
        _r(
            ".hs-ab .bar",
            position="absolute",
            left=_px(0),
            top=_px(11),
            height=_px(14),
            background=t.DASH,
        ),
        _r(".hs-ab .ci", top=_px(7), height=_px(22), min_width=_px(CI_MIN_PX)),
        _r(
            ".hs-vline",
            font_family=t.FONT_SERIF,
            font_style="italic",
            font_weight=400,
            font_size=f"clamp({_px(VLINE_MIN_PX)},{VLINE_VW}vw,{_px(VLINE_MAX_PX)})",
            line_height=1.2,
            margin=f"{_px(24)} {_px(0)}",
            color=t.TEXT,
        ),
        _r(
            ".hs-cards",
            display="grid",
            grid_template_columns=f"{ONE}fr {ONE}fr",
            gap=_px(PANEL_GAP_PX),
        ),
        _r(
            ".hs-card",
            border_top=f"{hair} {t.HAIRLINE}",
            padding_top=_px(16),
            min_width=_px(0),
        ),
        _r(
            ".hs-card-t",
            font=f"{HEAD_WEIGHT} {_px(t.HEADING_PX)} {t.FONT_SANS}",
            color=t.TEXT,
        ),
        _r(
            ".hs-card-v",
            font_weight=HEAD_WEIGHT,
            color=t.TEXT,
            margin=f"{_px(8)} {_px(0)}",
        ),
        _r(".hs-card-r", margin_top=_px(12), border_top=f"{hair} {t.HAIRLINE}"),
        _r(".hs-card-r span", display="block"),
        _r(".hs-card-r .k", color=t.SOFT, font_weight=HEAD_WEIGHT),
        _r(".hs-card-r .v", color=t.TEXT),
        _r(".hs-card-r .n", font=small, color=t.TEXT_SECONDARY),
        _r(
            ".hs-card-r .w,.hs-card-x",
            font_size=_px(t.MIN_SECONDARY_PX),
            color=t.TEXT_SECONDARY,
        ),
        _r(".hs-page", min_width=_px(0), overflow_wrap="anywhere"),
        f"@media(max-width:{_px(MEDIA_PX)}){{{_r('.hs-panels,.hs-cards', grid_template_columns=f'{ONE}fr')}}}",
    ]
    return f"<style>{''.join(rules)}</style>"
