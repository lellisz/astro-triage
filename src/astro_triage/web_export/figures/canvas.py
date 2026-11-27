"""Pixel-space drawing (y grows downwards) rendered by matplotlib into a deterministic SVG.

Operations are recorded first and replayed at the end, because the figure height is only known
after the notes and the source line have been laid out. Text stays as ``<text>``
(``svg.fonttype = none``), so the browser draws it in the site's Inter; widths are measured with
DejaVu Sans, which ships with matplotlib and is wider than Inter, so wrapping errs on the safe side.
"""

from __future__ import annotations

import io
import logging
import math
import re
from collections.abc import Callable
from functools import lru_cache
from html import escape
from typing import Final

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath

W: Final = 640
# Rendered between W (text at --text-sm, 15px) and 1.25 W (the site's growth ceiling).
MAX_W: Final = round(W * 1.25)
M: Final = 16
FS: Final = 15
LINE: Final = 20

SURFACE: Final = "#141414"
TEXT: Final = "#eeeeee"
TEXT2: Final = "#b4b4b4"
TEXT3: Final = "#8c8c8c"
AXIS: Final = "#8a8a8a"
GRID: Final = "#2a2a2a"
HATCH: Final = "#3a3a3a"
ACCENT: Final = "#6db3ff"
SERIES: Final = ("#d6d6d6", "#a0a0a0", "#7a7a7a")
SERIES_DASH: Final = (None, (6, 4), (8, 3, 2, 3))

MEASURE_FAMILY: Final = "DejaVu Sans"
# The site loads Inter through next/font under a hashed family name exposed as --font-sans;
# the fallback list keeps the standalone file readable.
FONT_CSS: Final = "font-family: var(--font-sans, 'Inter', 'DejaVu Sans', sans-serif)"

RC: Final = {
    "svg.fonttype": "none",
    "svg.hashsalt": "astro-triage",
    "font.family": [MEASURE_FAMILY],
    "font.size": FS,
    "lines.scale_dashes": False,
    "hatch.color": HATCH,
    "hatch.linewidth": 1.5,
}

# class -> (color, size, weight)
TEXT_STYLE: Final = {
    "t": (TEXT, 17, 600),
    "s": (TEXT2, FS, 400),
    "n": (TEXT3, FS, 400),
    "tk": (TEXT3, FS, 400),
    "nm": (TEXT, FS, 600),
    "val": (ACCENT, FS, 600),
    "big": (ACCENT, 17, 600),
}

# class -> (color, width, dashes)
LINE_STYLE: dict[str, tuple[str, float, tuple[float, ...] | None]] = {
    "grid": (GRID, 1, None),
    "axis": (AXIS, 1, None),
    "ref": (TEXT3, 1, (4, 4)),
    "lead": (TEXT3, 1, None),
    "cil": (TEXT, 1.5, None),
    "rng": (SERIES[1], 3, None),
    "est": (TEXT, 2.5, None),
}
for _i, (_c, _d) in enumerate(zip(SERIES, SERIES_DASH, strict=True), start=1):
    LINE_STYLE[f"ln{_i}"] = (_c, 2, _d)
    LINE_STYLE[f"ci{_i}"] = (_c, 1.5, None)
    LINE_STYLE[f"raw{_i}"] = (_c, 1, (2, 3))

# DejaVu has no 600 weight; the fallback to 700 only affects measurement, the SVG keeps 600.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_TTP: Final = TextToPath()
_ANCHOR: Final = {"start": "left", "middle": "center", "end": "right"}


@lru_cache(maxsize=4096)
def tw(s: str, size: float = FS, bold: bool = False) -> float:
    prop = FontProperties(family=MEASURE_FAMILY, size=size, weight=600 if bold else 400)
    width, _, _ = _TTP.get_text_width_height_descent(s, prop, ismath=False)
    return float(width)


def wrap(s: str, maxw: float, size: float = FS, bold: bool = False) -> list[str]:
    out: list[str] = []
    cur = ""
    for word in s.split():
        nxt = f"{cur} {word}" if cur else word
        if cur and tw(nxt, size, bold) > maxw:
            out.append(cur)
            cur = word
        else:
            cur = nxt
    if cur:
        out.append(cur)
    return out


# ---- pt-BR formatting of numbers read from the JSON ----
def pt_int(n: float) -> str:
    return f"{round(n):,}".replace(",", ".")


def pt_dec(x: float, d: int, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("+-")
    return s.replace(".", ",")


def pt_pct(x: float, d: int = 0) -> str:
    return pt_dec(x * 100, d) + "%"


def nice_ticks(lo: float, hi: float, n: int = 4) -> tuple[list[float], float]:
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5) if m * mag >= raw - 1e-12), 10 * mag)
    a = math.floor(lo / step + 1e-9) * step
    b = math.ceil(hi / step - 1e-9) * step
    k = round((b - a) / step)
    return [a + i * step for i in range(k + 1)], step


def decimals_for(step: float) -> int:
    d = max(0, -math.floor(math.log10(step) + 1e-9))
    if abs(step * 10**d - round(step * 10**d)) > 1e-9:
        d += 1
    return d


Y_TITLE_X: Final = 18


def left_pad(labels: list[str]) -> float:
    """Plot left edge that keeps right-aligned tick labels clear of the rotated axis title at Y_TITLE_X."""
    return Y_TITLE_X + 8 + 12 + max((tw(t) for t in labels), default=0.0) + 8


def place_labels(wanted: list[tuple[int, float]], gap: float, lo: float, hi: float) -> dict[int, float]:
    """Stack direct labels without overlap, keeping their vertical order."""
    order = sorted(wanted, key=lambda t: t[1])
    ys: list[float] = []
    prev = -1e9
    for _, y in order:
        y = max(y, prev + gap, lo)
        ys.append(y)
        prev = y
    over = ys[-1] - hi if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    return {k: y for (k, _), y in zip(order, ys, strict=True)}


Op = Callable[[Axes], None]


class Canvas:
    """Records drawing operations in viewBox pixels; ``render`` returns the final SVG text."""

    def __init__(self, fid: str, title: str, subtitle: str, simple: bool = False) -> None:
        self.fid = fid
        self.title = title
        # Simple mode (lay reader): no notes and no source line; an empty subtitle draws no line.
        self.simple = simple
        self.ops: list[Op] = []
        self.y: float = 2
        for ln in wrap(title, W - 2 * M, size=17, bold=True):
            self.y += LINE
            self.text(M, self.y, ln, "t")
        for ln in wrap(subtitle, W - 2 * M):
            self.y += LINE
            self.text(M, self.y, ln, "s")

    def text(self, x: float, y: float, s: str, cls: str, anchor: str = "start") -> None:
        color, size, weight = TEXT_STYLE[cls]
        self.ops.append(
            lambda ax: ax.text(
                x, y, s, color=color, fontsize=size, fontweight=weight,
                ha=_ANCHOR[anchor], va="baseline", clip_on=False,
            )
        )

    def vtext(self, x: float, y: float, s: str, cls: str, anchor: str = "middle") -> None:
        """Text rotated to read bottom-up; ``anchor`` refers to the text's own direction."""
        color, size, weight = TEXT_STYLE[cls]
        self.ops.append(
            lambda ax: ax.text(
                x, y, s, color=color, fontsize=size, fontweight=weight, rotation=90,
                rotation_mode="anchor", ha=_ANCHOR[anchor], va="center", clip_on=False,
            )
        )

    def poly(self, pts: list[tuple[float, float]], cls: str) -> None:
        color, width, dashes = LINE_STYLE[cls]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        def draw(ax: Axes) -> None:
            (ln,) = ax.plot(xs, ys, color=color, linewidth=width, marker="", solid_capstyle="butt",
                            solid_joinstyle="round", clip_on=False)
            if dashes:
                ln.set_dashes(dashes)
                ln.set_dash_capstyle("butt")

        self.ops.append(draw)

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str) -> None:
        self.poly([(x1, y1), (x2, y2)], cls)

    def rect(
        self, x: float, y: float, w: float, h: float, *, fill: str | None = None, alpha: float = 1.0,
        edge: str | None = None, lw: float = 0, dashes: tuple[float, ...] | None = None,
        hatch: bool = False,
    ) -> None:
        def draw(ax: Axes) -> None:
            r = Rectangle(
                (x, y), max(w, 0), h, facecolor=to_rgba(fill, alpha) if fill else "none",
                edgecolor=edge or "none", linewidth=lw, hatch="///" if hatch else None, clip_on=False,
            )
            if dashes:
                r.set_linestyle((0, dashes))
            ax.add_patch(r)

        self.ops.append(draw)

    def notes(self, lines: list[str], gap: float = 28) -> None:
        if self.simple:
            return
        self.y += gap
        first = True
        for s in lines:
            for ln in wrap(s, W - 2 * M):
                if not first:
                    self.y += LINE
                self.text(M, self.y, ln, "n")
                first = False

    def source(self, source: str, release: str, floor_line: str | None) -> None:
        if self.simple:
            return
        self.y += 8
        for s in [f"{source} · {release}"] + ([floor_line] if floor_line else []):
            for ln in wrap(s, W - 2 * M):
                self.y += LINE
                self.text(M, self.y, ln, "n")

    def render(self, desc: str) -> str:
        h = math.ceil(self.y + 16)
        with matplotlib.rc_context(RC):
            fig = Figure(figsize=(W / 72, h / 72), dpi=72, facecolor="none")
            ax = fig.add_axes((0, 0, 1, 1))
            ax.set_xlim(0, W)
            ax.set_ylim(h, 0)
            ax.set_axis_off()
            for op in self.ops:
                op(ax)
            buf = io.StringIO()
            fig.savefig(buf, format="svg", transparent=True, metadata={"Date": None})
        return _finish(buf.getvalue(), self.fid, self.title, desc, h)


def _finish(svg: str, fid: str, title: str, desc: str, h: int) -> str:
    """Make the file safe to inline: no prolog, ids prefixed by the figure id, accessible name."""
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r"\s*<metadata>.*?</metadata>", "", svg, flags=re.DOTALL)
    # Inline, a <style> would leak its selectors into the whole page; the root attributes inherit instead.
    svg = re.sub(r"\s*<defs>\s*<style[^>]*>.*?</style>\s*</defs>", "", svg, flags=re.DOTALL)
    svg = re.sub(r'\bid="([^"]+)"', rf'id="{fid}-\1"', svg)
    svg = re.sub(r"url\(#([^)]+)\)", rf"url(#{fid}-\1)", svg)
    svg = re.sub(r'href="#([^"]+)"', rf'href="#{fid}-\1"', svg)
    svg = re.sub(r"font-family: [^;\"]+", FONT_CSS, svg)
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        f'role="img" aria-labelledby="{fid}-title" aria-describedby="{fid}-desc" '
        'stroke-linejoin="round" stroke-linecap="butt" '
        f'style="width:100%;min-width:{W}px;max-width:{MAX_W}px;height:auto">'
        f'<title id="{fid}-title">{escape(title)}</title><desc id="{fid}-desc">{escape(desc)}</desc>'
    )
    return head + svg[svg.index(">") + 1:]
