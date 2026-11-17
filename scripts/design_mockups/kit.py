"""Shared kit of the mockups: page shell, navigation, seal, escape, number ledger and audits."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from astro_triage.dashboard import fmt, theme

from .loaders import Cell

NA = "n/a"
MONO = "ui-monospace, Menlo, Consolas, monospace"
FONT_DIR = Path(theme.__file__).parent / "static" / "fonts"
FONTS = (
    ("Instrument Serif", "instrument-serif-latin-400.woff2", "normal", "400"),
    ("Instrument Serif", "instrument-serif-latin-400-italic.woff2", "italic", "400"),
    ("Inter", "inter-latin-400-600.woff2", "normal", "400 600"),
)
NAV = (
    ("resultado", "Resultado"),
    ("ceu", "Céu"),
    ("galaxias", "Galáxias"),
    ("comparacao", "DL × LightGBM"),
    ("metodo", "Método"),
)
NAV_STUBS = {"galaxias": "fora deste lote"}
PALETTE = {
    n: getattr(theme, n)
    for n in (
        "BG", "TEXT", "TEXT_SECONDARY", "SOFT", "HAIRLINE", "GRID", "DASH", "ACCENT", "RAMP_LOW",
    )
}  # fmt: skip
TEXT_PAIRS = (
    ("TEXT", "BG"),
    ("TEXT_SECONDARY", "BG"),
    ("SOFT", "BG"),
    ("ACCENT", "BG"),
)
GRAPHIC_PAIRS = (("ACCENT", "BG"), ("DASH", "BG"))
FORBIDDEN = ("melhor", "ganho", "equivalente", "deploy", "trace", "camada 2", "llm")
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def e(value: object) -> str:
    """HTML escape for every text that did not come from a literal of this kit."""
    return escape(str(value), quote=True)


@dataclass
class Ledger:
    """Records 'displayed number -> artifact file/key'; formatting goes through it."""

    rows: list[dict] = field(default_factory=list)

    def _rec(self, shown: str, what: str, cell: Cell, note: str | None = None) -> str:
        row = {"shown": shown, "what": what, "file": cell.file, "key": cell.key}
        self.rows.append(row | ({"derived": note} if note else {}))
        return shown

    def pct(self, cell: Cell, what: str, digits: int = 1) -> str:
        return self._rec(fmt.pt_pct(cell.value, digits) if cell.ok else NA, what, cell)

    def num(
        self, cell: Cell, what: str, digits: int = 4, signed: bool = False
    ) -> str:
        shown = fmt.pt_num(cell.value, digits, signed) if cell.ok else NA
        return self._rec(shown, what, cell)

    def integer(self, cell: Cell, what: str) -> str:
        return self._rec(fmt.pt_int(cell.value) if cell.ok else NA, what, cell)

    def text(self, cell: Cell, what: str) -> str:
        return self._rec(str(cell.value) if cell.ok else NA, what, cell)

    def interval(self, lo: Cell, hi: Cell, what: str, digits: int = 1) -> str:
        """'IC 95%: lo a hi', n/a when either end is missing."""
        a, b = (
            self.pct(lo, f"{what} (IC, limite inferior)", digits),
            self.pct(hi, f"{what} (IC, limite superior)", digits),
        )
        return (
            f"{fmt.pt_ci_label()}: {a} a {b}"
            if NA not in (a, b)
            else f"{fmt.pt_ci_label()}: {NA}"
        )

    def derived(self, shown: str, what: str, cell: Cell, note: str) -> str:
        return self._rec(shown, what, cell, note)


def font_css() -> str:
    faces = []
    for family, name, style, weight in FONTS:
        data = base64.b64encode((FONT_DIR / name).read_bytes()).decode("ascii")
        faces.append(
            f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};"
            f"src:url(data:font/woff2;base64,{data}) format('woff2');font-display:swap}}"
        )
    return "".join(faces)


def base_css() -> str:
    t = theme
    return f"""
*{{box-sizing:border-box}}
body{{margin:0;background:{t.BG};color:{t.TEXT};font:{t.BODY_PX}px/1.55 {t.FONT_SANS}}}
.mono{{font-family:{MONO}}}
.sm{{color:{t.TEXT_SECONDARY};font-size:{t.MIN_SECONDARY_PX}px;line-height:1.5}}
.top{{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:12px 32px;border-bottom:1px solid {t.HAIRLINE}}}
.top nav{{min-width:0;overflow-x:auto;white-space:nowrap;scrollbar-width:none}}
.top nav a,.top nav span{{display:inline-block;padding:10px 14px;min-height:{t.MIN_TARGET_PX}px;font-size:{t.CONTROL_PX}px;color:{t.SOFT};text-decoration:none;border-bottom:2px solid transparent}}
.top nav a.on{{color:{t.ACCENT};border-bottom-color:{t.ACCENT}}}
.top nav span{{color:{t.TEXT_SECONDARY}}}
.seal{{border:1px solid {t.ACCENT};padding:2px 10px;font:{t.MIN_SECONDARY_PX}px {MONO};white-space:nowrap;letter-spacing:.04em}}
main{{max-width:1200px;margin:0 auto;padding:{t.PAGE_TOP_PX}px 32px {t.PAGE_BOTTOM_PX}px}}
footer{{border-top:1px solid {t.HAIRLINE};padding:24px 32px;max-width:1200px;margin:0 auto}}
h2{{font:600 {t.HEADING_PX}px/1.25 {t.FONT_SANS};margin:0 0 {t.SPACE_MD}px}}
@media(max-width:720px){{
.top{{flex-wrap:wrap;gap:4px;padding:12px 16px 0}}
.top .seal{{order:1}}
.top nav{{order:2;width:100%;-webkit-mask-image:linear-gradient(to right,transparent,black 12px,black calc(100% - 28px),transparent);mask-image:linear-gradient(to right,transparent,black 12px,black calc(100% - 28px),transparent)}}
main{{padding:24px 16px 40px}}footer{{padding:16px}}}}
"""


TABLE_CSS = f"""
.tw{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;min-width:640px;font-size:{theme.MIN_SECONDARY_PX}px}}
th,td{{text-align:left;padding:8px 12px 8px 0;border-bottom:1px solid {theme.HAIRLINE};vertical-align:top}}
th{{color:{theme.TEXT_SECONDARY};font-weight:400}}
.sec{{margin-top:56px}}
.lead{{font:italic 400 clamp(26px,3.6vw,44px)/1.2 {theme.FONT_SERIF};margin:0 0 32px;max-width:960px}}
"""


def nav_html(active: str, built: set[str]) -> str:
    items = []
    for key, label in NAV:
        if key in NAV_STUBS:
            items.append(f"<span title='{e(NAV_STUBS[key])}'>{e(label)}</span>")
        elif key == active:
            items.append(
                f"<a class='on' href='{key}.html' aria-current='page'>{e(label)}</a>"
            )
        else:
            items.append(f"<a href='{key}.html'>{e(label)}</a>")
    return f"<nav aria-label='Navegação'>{''.join(items)}</nav>"


def seal_html(release: str) -> str:
    return f"<div class='seal'>PROVISÓRIO · {e(release)}</div>"


def page(
    screen: str, title: str, body: str, release: str, footer: str, extra_css: str = ""
) -> str:
    return (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{e(title)} · {e(release)}</title>"
        f"<style>{font_css()}{base_css()}{extra_css}</style></head><body>"
        f"<header class='top'>{nav_html(screen, set())}{seal_html(release)}</header>"
        f"<main>{body}</main><footer class='sm'>{footer}</footer></body></html>"
    )


def contrast_table() -> list[dict]:
    """Ratios of every text and graphic pair in use; raises when one is below its floor."""
    rows = []
    for pairs, floor, kind in (
        (TEXT_PAIRS, theme.MIN_TEXT_CONTRAST, "texto"),
        (GRAPHIC_PAIRS, theme.MIN_GRAPHIC_CONTRAST, "gráfico"),
    ):
        for fg, bg in pairs:
            ratio = theme.contrast_ratio(PALETTE[fg], PALETTE[bg])
            rows.append(
                {
                    "kind": kind,
                    "fg": fg,
                    "bg": bg,
                    "ratio": round(ratio, 2),
                    "min": floor,
                }
            )
            if ratio < floor:
                raise ValueError(f"contrast {fg} on {bg} = {ratio:.2f} < {floor}")
    return rows


def audit(html: str) -> None:
    """Fails on a colour outside the palette, a forbidden word, or an external request."""
    colours = {c.lower() for c in _HEX.findall(html)}
    extra = colours - {v.lower() for v in PALETTE.values()}
    if extra:
        raise ValueError(f"colours outside the palette: {sorted(extra)}")
    prose = re.sub(r"<style>.*?</style>", "", html, flags=re.DOTALL).lower()
    hits = [w for w in FORBIDDEN if w in prose]
    if hits:
        raise ValueError(f"forbidden words: {hits}")
    if re.search(r"(?:src|href)\s*=\s*['\"]?https?:|url\(\s*['\"]?https?:", html):
        raise ValueError("external request found")
