"""Shared figure style for both editions.

Colour semantics (identical to the LaTeX preamble and used in every figure):
    electric quantities  -> ELEC  (blue)
    magnetic quantities  -> MAG   (vermilion)
    currents / charge flow -> CURR (bluish green)
    potentials           -> POT   (reddish purple)
    materials / regions  -> MATL  (warm grey) with light fills
The palette is based on the Okabe-Ito colour-blind-safe set.

Language is selected with the environment variable EM_LANG = 'zh' | 'en'.
No font files are distributed: the Simplified-Chinese face is extracted from
an installed Noto Sans CJK collection into figsrc/.fontcache when needed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LANG = os.environ.get("EM_LANG", "zh")
ZH = LANG == "zh"
OUT = ROOT / ("figures_zh" if ZH else "figures")
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------------
ELEC = "#0072B2"
MAG = "#D55E00"
CURR = "#009E73"
POT = "#A8508A"
MATL = "#8C8279"
INK = "#1E2328"
MUTED = "#646B73"
ACCENT = "#1F4E79"
AMBER = "#B7791F"
WARN = "#A33A3A"
LIGHT = "#E9ECEF"


def tint(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour with white; amount=0 -> white, 1 -> the colour."""
    c = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(1 - amount * (1 - c))


ELEC_T, MAG_T, CURR_T, POT_T = (tint(c, 0.13) for c in (ELEC, MAG, CURR, POT))
MATL_T = tint("#C9B79C", 0.35)  # tissue-like warm sand
IRON_T = tint("#7A8691", 0.30)  # soft iron / yoke
NEUTRAL_T = tint("#6B7785", 0.10)


def T(zh: str, en: str) -> str:
    """Return the label for the active language."""
    return zh if ZH else en


# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------
CACHE = HERE / ".fontcache"
CJK_FAMILY = "EM Noto Sans SC"


def _extract_cjk_faces() -> None:
    """Extract the Simplified-Chinese face (index 2) from Noto Sans CJK .ttc files."""
    CACHE.mkdir(exist_ok=True)
    needed = [w for w in ("Regular", "Medium", "Bold") if not (CACHE / f"EMNotoSansSC-{w}.otf").exists()]
    if not needed:
        return
    try:
        from fontTools.ttLib import TTCollection
    except ImportError:  # pragma: no cover - documented requirement
        return
    candidates = [Path(p) for p in fm.findSystemFonts() if Path(p).name.startswith("NotoSansCJK-")]
    for weight in needed:
        src = next((p for p in candidates if p.name == f"NotoSansCJK-{weight}.ttc"), None)
        if src is None:
            continue
        coll = TTCollection(str(src))
        face = next((f for f in coll.fonts if f["name"].getDebugName(1) == "Noto Sans CJK SC"), None)
        if face is None:
            continue
        for rec in face["name"].names:
            if rec.nameID in (1, 16):
                rec.string = CJK_FAMILY
            elif rec.nameID == 4:
                rec.string = f"{CJK_FAMILY} {weight}"
            elif rec.nameID == 6:
                rec.string = f"EMNotoSansSC-{weight}"
        face.save(str(CACHE / f"EMNotoSansSC-{weight}.otf"))


def configure_fonts() -> str:
    _extract_cjk_faces()
    for f in CACHE.glob("EMNotoSansSC-*.otf"):
        fm.fontManager.addfont(str(f))
    names = {f.name for f in fm.fontManager.ttflist}
    cjk = next((n for n in (CJK_FAMILY, "Noto Sans CJK SC", "Source Han Sans SC", "Noto Sans SC") if n in names), "DejaVu Sans")
    latin = next((n for n in ("Inter", "TeX Gyre Heros", "Liberation Sans", "DejaVu Sans") if n in names), "DejaVu Sans")
    # Mixed CJK + mathtext strings render their plain text in the *first* family,
    # so the Chinese edition uses the CJK face first (it contains Latin glyphs).
    family = [cjk, latin] if ZH else [latin, cjk]
    plt.rcParams.update({
        "font.family": family,
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.8,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,     # true minus sign (U+2212) on axes; both figure fonts have it
        "pdf.fonttype": 3,
        "svg.fonttype": "path",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 220,
        "hatch.linewidth": 0.6,
    })
    return family[0]


FONT = configure_fonts()

# Width of the text block (152 mm) in inches; figures are drawn at this size
# so that 9.5 pt figure text matches the body text after inclusion.
FIG_W = 6.0

# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
REGISTRY: dict[str, Callable[[], plt.Figure]] = {}


def figure(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        fn.figname = name
        return fn
    return deco


def save(fig: plt.Figure, name: str, formats=("pdf", "png", "svg")) -> None:
    for ext in formats:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04,
                    transparent=False, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def canvas(w: float = FIG_W, h: float = 2.6, xlim=(0, 10), ylim=(0, 4.3)):
    """Blank diagram canvas with data units roughly matching inches."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def clean_axes(ax, grid: bool = False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    return ax


def arrow(ax, p, q, color=INK, lw=1.4, style="-|>", ms=10, ls="-", z=5, shrink=0.0, alpha=1.0, **kw):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                        linestyle=ls, shrinkA=shrink, shrinkB=shrink, zorder=z,
                        capstyle="round", joinstyle="round", alpha=alpha, **kw)
    ax.add_patch(a)
    return a


def curved_arrow(ax, p, q, rad=0.25, **kw):
    return arrow(ax, p, q, connectionstyle=f"arc3,rad={rad}", **kw)


def label(ax, x, y, s, color=INK, size=9.5, weight="normal", ha="center", va="center", z=8, **kw):
    return ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va,
                   zorder=z, linespacing=1.3, **kw)


def rbox(ax, x, y, w, h, fc=NEUTRAL_T, ec=None, lw=1.0, r=0.12, z=2, ls="-", alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                       ec=ec if ec is not None else "none", lw=lw, zorder=z, linestyle=ls, alpha=alpha)
    ax.add_patch(p)
    return p


def card(ax, x, y, w, h, title, lines=(), color=ACCENT, fill=None, title_size=9.5,
         body_size=9.2, body_color=INK, band=0.36, z=2, align="center"):
    """Rounded card with a coloured title band and centred body lines."""
    fill = fill if fill is not None else tint(color, 0.08)
    rbox(ax, x, y, w, h, fc=fill, ec=tint(color, 0.55), lw=0.9, z=z)
    rbox(ax, x, y + h - band, w, band, fc=tint(color, 0.85), ec="none", z=z + 0.1)
    # square-off the band's lower corners
    ax.add_patch(Rectangle((x, y + h - band), w, band * 0.5, fc=tint(color, 0.85), ec="none", zorder=z + 0.1))
    label(ax, x + w / 2, y + h - band / 2, title, color="white", size=title_size, weight="bold", z=z + 1)
    if lines:
        n = len(lines)
        top = y + h - band
        for i, s in enumerate(lines):
            yy = top - (i + 0.5) * (top - y) / n
            if align == "left":
                label(ax, x + 0.12, yy, s, size=body_size, color=body_color, ha="left", z=z + 1)
            else:
                label(ax, x + w / 2, yy, s, size=body_size, color=body_color, z=z + 1)


def pill(ax, x, y, s, color=ACCENT, size=8.6, pad=0.25, z=9, text_color="white"):
    return ax.text(x, y, s, fontsize=size, color=text_color, ha="center", va="center", zorder=z,
                   bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.5", fc=color, ec="none"))


def dot(ax, x, y, color=INK, r=0.05, z=6):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=z))


def out_of_page(ax, x, y, color=INK, r=0.11, lw=1.0, z=6):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=lw, zorder=z))
    ax.add_patch(Circle((x, y), r * 0.28, fc=color, ec="none", zorder=z + 0.1))


def into_page(ax, x, y, color=INK, r=0.11, lw=1.0, z=6):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=lw, zorder=z))
    d = r * 0.62
    ax.plot([x - d, x + d], [y - d, y + d], color=color, lw=lw, zorder=z + 0.1)
    ax.plot([x - d, x + d], [y + d, y - d], color=color, lw=lw, zorder=z + 0.1)


def charge(ax, x, y, sign="+", color=None, r=0.13, size=9, z=7):
    color = color or (ELEC if sign == "+" else ELEC)
    fc = tint(MAG, 0.18) if sign == "+" else tint(ELEC, 0.18)
    ec = MAG if sign == "+" else ELEC
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.9, zorder=z))
    label(ax, x, y - 0.005, "+" if sign == "+" else "−", color=ec, size=size, weight="bold", z=z + 1)


def polyline_arrow(ax, pts, color=INK, lw=1.4, ms=9, frac=0.55, z=5):
    """Draw a polyline and put an arrow head at fraction `frac` of its length."""
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, solid_capstyle="round")
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    s = frac * L.sum()
    k = np.searchsorted(np.cumsum(L), s)
    k = min(k, len(L) - 1)
    s0 = s - (np.cumsum(L)[k] - L[k])
    p = pts[k] + seg[k] / L[k] * s0
    d = seg[k] / L[k] * 0.02
    arrow(ax, p - d, p + d, color=color, lw=lw, ms=ms, z=z + 0.5)


def stream(ax, X, Y, U, V, color=ELEC, density=1.0, lw=1.1, arrowsize=0.9, z=3, **kw):
    return ax.streamplot(X, Y, U, V, color=color, density=density, linewidth=lw,
                         arrowsize=arrowsize, arrowstyle="-|>", zorder=z, **kw)


def unit_quiver(ax, X, Y, U, V, color=ELEC, scale=None, width=0.006, z=4, normalize=False, **kw):
    if normalize:
        n = np.hypot(U, V)
        n[n == 0] = 1
        U, V = U / n, V / n
    return ax.quiver(X, Y, U, V, color=color, angles="xy", scale_units="xy", scale=scale,
                     width=width, headwidth=3.6, headlength=4.2, headaxislength=3.8, zorder=z, **kw)


def panel_title(ax, s, color=INK, size=9.8, y=1.02, weight="bold"):
    ax.set_title(s, color=color, fontsize=size, fontweight=weight, y=y, pad=4)


def colorbar(fig, mappable, ax, text, **kw):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03, **kw)
    cb.set_label(text, fontsize=8.8)
    cb.ax.tick_params(labelsize=8, length=2.5)
    cb.outline.set_linewidth(0.6)
    return cb


def seq_cmap(color: str, name: str = "c"):
    """Light-to-colour sequential colormap for magnitudes."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, [tint(color, 0.02), tint(color, 0.35), color, matplotlib.colors.to_hex(np.array(matplotlib.colors.to_rgb(color)) * 0.55)])


def flux_lines(ax, X, Z, Psi, levels, Fx, Fz, color=MAG, lw=1.0, arrows=True, ms=7,
               arrow_pos=0.5, mask=None, z=3, min_len=0.15):
    """Field lines of an axisymmetric (or 2-D solenoidal) field as contours of its
    flux / stream function.  Arrow heads follow the local field direction.

    For a field written as F = curl(Psi e_phi / rho) the lines are Psi = const,
    and equal Psi steps carry equal flux, so line density shows field strength.
    """
    from scipy.interpolate import RegularGridInterpolator
    P = np.ma.array(Psi, mask=mask) if mask is not None else Psi
    cs = ax.contour(X, Z, P, levels=levels, colors=[color], linewidths=lw, zorder=z,
                    negative_linestyles="solid")
    if not arrows:
        return cs
    gx, gz = X[0, :], Z[:, 0]
    fx = RegularGridInterpolator((gz, gx), Fx, bounds_error=False, fill_value=0.0)
    fz = RegularGridInterpolator((gz, gx), Fz, bounds_error=False, fill_value=0.0)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for seg in cs.allsegs:
        for path in seg:
            if len(path) < 4:
                continue
            inside = (path[:, 0] > xl[0]) & (path[:, 0] < xl[1]) & (path[:, 1] > yl[0]) & (path[:, 1] < yl[1])
            pts = path[inside]
            if len(pts) < 4:
                continue
            d = np.hypot(*np.diff(pts, axis=0).T)
            if d.sum() < min_len:
                continue
            if arrow_pos == "xmax":
                k = int(np.argmax(np.abs(pts[:, 0])))
            else:
                s = np.concatenate(([0], np.cumsum(d)))
                k = int(np.searchsorted(s, arrow_pos * s[-1]))
            k = min(max(k, 1), len(pts) - 2)
            p = pts[k]
            v = np.array([fx((p[1], p[0])), fz((p[1], p[0]))]).ravel()
            n = np.hypot(*v)
            if n == 0:
                continue
            v = v / n * 0.02
            arrow(ax, p - v, p + v, color=color, lw=lw, ms=ms, z=z + 0.5)
    return cs


def halo(artist, width=3.5, color="white"):
    """White outline so an arrow or label stays legible over field lines."""
    import matplotlib.patheffects as pe
    artist.set_path_effects([pe.withStroke(linewidth=width, foreground=color)])
    return artist
