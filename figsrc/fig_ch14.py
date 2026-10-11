"""Figures for Ch 14 (lumped magnetic circuits).

All computed quantities come from checks/check_ch14.py (which uses numerics.py), so the
figures, the text and the assertions share one implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import matplotlib.tri as mtri

from style import (ACCENT, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, IRON_T, LIGHT, MAG, MAG_T, MATL, MUTED,
                   NEUTRAL_T, T, arrow, canvas, clean_axes, figure, halo, into_page, label, out_of_page,
                   panel_title, pill, plt, rbox, seq_cmap, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch14 as LM  # noqa: E402
from numerics import MU0, synthetic_B  # noqa: E402

IRON_E = "#8E989F"          # outline of iron parts
POLE_T = tint("#7A8691", 0.55)
WIRE = INK


# ---------------------------------------------------------------------------
# Small drawing helpers (network symbols)
# ---------------------------------------------------------------------------
def wire(ax, pts, color=WIRE, lw=1.15, z=3, ls="-"):
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, ls=ls,
            solid_capstyle="round", solid_joinstyle="round")


def rel(ax, c, vertical=True, text=None, side="right", L=0.6, Wd=0.24, color=MAG, size=9.5, dashed=False,
        fill=None, dx=0.12, z=5, text_color=None):
    """Reluctance (IEC resistor-like rectangle) centred at c."""
    x, y = c
    w_, h_ = (Wd, L) if vertical else (L, Wd)
    ax.add_patch(Rectangle((x - w_ / 2, y - h_ / 2), w_, h_, fc=fill if fill is not None else tint(color, 0.16),
                           ec=color, lw=1.0, zorder=z, ls=(0, (3, 2)) if dashed else "-"))
    if text:
        tc = text_color or INK
        if vertical:
            if side == "right":
                label(ax, x + w_ / 2 + dx, y, text, size=size, ha="left", color=tc)
            else:
                label(ax, x - w_ / 2 - dx, y, text, size=size, ha="right", color=tc)
        else:
            if side == "top":
                label(ax, x, y + h_ / 2 + dx + 0.05, text, size=size, color=tc)
            else:
                label(ax, x, y - h_ / 2 - dx - 0.05, text, size=size, color=tc)


def mmf_src(ax, c, text=None, side="right", r=0.24, up=True, size=9.5, dx=0.12, color=MAG):
    """MMF source: circle with + / - (the + side is where the source raises the magnetic potential)."""
    x, y = c
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=1.1, zorder=6))
    s = 1 if up else -1
    label(ax, x, y + s * r * 0.45, "+", size=8.5, color=color, weight="bold", z=7)
    label(ax, x, y - s * r * 0.48, "−", size=9, color=color, weight="bold", z=7)
    if text:
        if side == "right":
            label(ax, x + r + dx, y, text, size=size, ha="left")
        else:
            label(ax, x - r - dx, y, text, size=size, ha="right")


def flux_src(ax, c, text=None, side="right", r=0.24, up=True, size=9.5, dx=0.12, color=MAG):
    """Flux source: circle with an arrow (like a current source)."""
    x, y = c
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=1.1, zorder=6))
    s = 1 if up else -1
    arrow(ax, (x, y - s * r * 0.6), (x, y + s * r * 0.62), color=color, lw=1.2, ms=8, z=7)
    if text:
        if side == "right":
            label(ax, x + r + dx, y, text, size=size, ha="left")
        else:
            label(ax, x - r - dx, y, text, size=size, ha="right")


def node(ax, x, y, r=0.055, color=INK):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=8))


def flux_arrow(ax, p, q, text=None, tpos=None, size=9.2, color=MAG, lw=1.5, ha="center"):
    arrow(ax, p, q, color=color, lw=lw, ms=10, z=7)
    if text:
        tp = tpos if tpos is not None else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        label(ax, tp[0], tp[1], text, size=size, color=color, ha=ha)


def iron_rect(ax, x, y, w, h, fc=IRON_T, ec=IRON_E, z=2, lw=0.8, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, hatch=hatch))


def iron_frame(ax, x0, y0, W, H, w, gap=None, z=2):
    """Rectangular C-core (outer W x H, limb width w) drawn as one polygon with an optional gap
    (y-interval) cut through the right limb."""
    if gap is None:
        outer = Rectangle((x0, y0), W, H, fc=IRON_T, ec=IRON_E, lw=0.9, zorder=z)
        ax.add_patch(outer)
    else:
        g0, g1 = gap
        lower = [(x0, y0), (x0 + W, y0), (x0 + W, g0), (x0 + W - w, g0), (x0 + W - w, y0 + w), (x0 + w, y0 + w),
                 (x0 + w, y0 + H - w), (x0 + W - w, y0 + H - w), (x0 + W - w, g1), (x0 + W, g1), (x0 + W, y0 + H),
                 (x0, y0 + H)]
        ax.add_patch(Polygon(lower, closed=True, fc=IRON_T, ec=IRON_E, lw=0.9, zorder=z))
        return
    ax.add_patch(Rectangle((x0 + w, y0 + w), W - 2 * w, H - 2 * w, fc="white", ec=IRON_E, lw=0.9, zorder=z + 0.1))


def coil(ax, xl, xr, y0, y1, n=6, color=CURR, lw=1.5, z=6):
    """Front half-turns of a winding around a vertical limb between xl and xr."""
    ys = np.linspace(y0, y1, n)
    for y in ys:
        ax.plot([xl - 0.1, xr + 0.1], [y, y + 0.13], color=color, lw=lw, zorder=z, solid_capstyle="round")
    return ys


def magnet_block(ax, x, y, w, h, north_up=True, z=4, size=9):
    """Permanent magnet: magnetic tint, light hatching, N/S labels and magnetisation arrow."""
    ax.add_patch(Rectangle((x, y), w, h, fc=tint(MAG, 0.22), ec=MAG, lw=1.0, zorder=z, hatch="////"))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=tint(MAG, 0.4), lw=0, zorder=z, hatch=None))
    top, bot = ("N", "S") if north_up else ("S", "N")
    sgn = 1 if north_up else -1
    xc = x + w / 2
    halo(arrow(ax, (xc, y + h / 2 - sgn * h * 0.32), (xc, y + h / 2 + sgn * h * 0.32), color=MAG, lw=1.6, ms=9, z=z + 2), 3)
    label(ax, x - 0.13, y + h - 0.1, top, size=size - 1.5, color=MAG, weight="bold", z=z + 2)
    label(ax, x - 0.13, y + 0.1, bot, size=size - 1.5, color=MAG, weight="bold", z=z + 2)


# ---------------------------------------------------------------------------
@figure("circuit_analogy")
def circuit_analogy():
    fig, ax = canvas(FIG_W, 3.45, xlim=(0, 10.4), ylim=(0, 5.98))
    xL, xR, wp = 0.15, 6.35, 3.9
    # titles
    label(ax, xL + wp / 2, 5.72, T("电路：导体约束电流", "Electric circuit: a conductor confines current"),
          weight="bold", color=ELEC, size=9.8)
    label(ax, xR + wp / 2, 5.72, T("磁路：铁芯约束磁通", "Magnetic circuit: iron confines flux"),
          weight="bold", color=MAG, size=9.8)
    # --- row 1: a uniform bar -> element
    def bar(x0, fc, ec, flow_col, field_col, flow_sym, field_sym, mat_sym):
        bx, by, bw, bh = x0 + 0.55, 4.35, 2.8, 0.62
        ax.add_patch(Rectangle((bx, by), bw, bh, fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.add_patch(Polygon([(bx + bw, by), (bx + bw + 0.22, by + 0.16), (bx + bw + 0.22, by + bh + 0.16),
                              (bx + bw, by + bh)], fc=tint(ec, 0.25), ec=ec, lw=0.8, zorder=2))
        ax.add_patch(Polygon([(bx, by + bh), (bx + 0.22, by + bh + 0.16), (bx + bw + 0.22, by + bh + 0.16),
                              (bx + bw, by + bh)], fc=tint(ec, 0.12), ec=ec, lw=0.8, zorder=2))
        for yy in (by + 0.17, by + 0.45):
            arrow(ax, (bx + 0.25, yy), (bx + bw - 0.3, yy), color=flow_col, lw=1.3, ms=9, z=4)
        label(ax, bx + bw / 2, by + bh + 0.38, flow_sym, color=flow_col, size=9.5)
        # length & area
        arrow(ax, (bx, by - 0.18), (bx + bw, by - 0.18), color=MUTED, lw=0.8, ms=6, style="<|-|>")
        label(ax, bx + bw / 2, by - 0.38, r"$\ell$", color=MUTED, size=9)
        label(ax, bx - 0.12, by + bh / 2, r"$A$", color=MUTED, size=9, ha="right")
        label(ax, bx + bw + 0.42, by + bh / 2 + 0.05, mat_sym, color=INK, size=9.5, ha="left")
        return bx, by, bw, bh

    bar(xL, ELEC_T, ELEC, CURR, ELEC, T(r"$\mathbf{J}=\sigma\mathbf{E}$，总量 $I=JA$", r"$\mathbf{J}=\sigma\mathbf{E}$, total $I=JA$"), "", r"$\sigma$")
    bar(xR, IRON_T, IRON_E, MAG, MAG, T(r"$\mathbf{B}=\mu\mathbf{H}$，总量 $\Phi=BA$", r"$\mathbf{B}=\mu\mathbf{H}$, total $\Phi=BA$"), "", r"$\mu$")
    label(ax, xL + wp / 2 + 0.1, 3.5, r"$V=\int\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}=I\,\frac{\ell}{\sigma A}=RI$",
          size=10, color=ELEC)
    label(ax, xR + wp / 2 + 0.1, 3.5, r"$\int\mathbf{H}\cdot\mathrm{d}\boldsymbol{\ell}=\Phi\,\frac{\ell}{\mu A}=\mathcal{R}\,\Phi$",
          size=10, color=MAG)
    # --- row 2: network laws
    def lawcard(x0, col, l1, l2):
        rbox(ax, x0 + 0.1, 1.6, wp - 0.2, 1.3, fc=tint(col, 0.07), ec=tint(col, 0.45), lw=0.8)
        label(ax, x0 + 0.22, 2.55, l1, size=8.8, ha="left")
        label(ax, x0 + 0.22, 1.95, l2, size=8.8, ha="left")
    lawcard(xL, ELEC, T(r"$\nabla\!\cdot\mathbf{J}=0\ \Rightarrow\ \sum I_k=0$（节点）", r"$\nabla\!\cdot\mathbf{J}=0\ \Rightarrow\ \sum I_k=0$ (node)"),
            T(r"$\oint\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}\ \Rightarrow\ \sum R_kI_k=\sum V_s$（回路）", r"$\oint\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}\ \Rightarrow\ \sum R_kI_k=\sum V_s$ (loop)"))
    lawcard(xR, MAG, T(r"$\nabla\!\cdot\mathbf{B}=0\ \Rightarrow\ \sum \Phi_k=0$（节点）", r"$\nabla\!\cdot\mathbf{B}=0\ \Rightarrow\ \sum \Phi_k=0$ (node)"),
            T(r"$\oint\mathbf{H}\cdot\mathrm{d}\boldsymbol{\ell}=I_f\ \Rightarrow\ \sum \mathcal{R}_k\Phi_k=\sum\mathcal{F}_j$（回路）",
              r"$\oint\mathbf{H}\cdot\mathrm{d}\boldsymbol{\ell}=I_f\ \Rightarrow\ \sum \mathcal{R}_k\Phi_k=\sum\mathcal{F}_j$ (loop)"))
    # --- centre strip: correspondences
    pairs = [(r"$\mathbf{E}$", r"$\mathbf{H}$"), (r"$\mathbf{J}$", r"$\mathbf{B}$"), (r"$\sigma$", r"$\mu$"),
             (r"$V$", r"$\mathcal{F}$"), (r"$I$", r"$\Phi$"), (r"$R$", r"$\mathcal{R}$"), (r"$G$", r"$\mathcal{P}$")]
    ys = np.linspace(5.2, 1.75, len(pairs))
    xc = 5.2
    rbox(ax, xc - 0.78, 1.45, 1.56, 4.05, fc=tint(ACCENT, 0.05), ec=tint(ACCENT, 0.3), lw=0.7, r=0.15)
    for (a_, b_), yy in zip(pairs, ys):
        label(ax, xc - 0.42, yy, a_, color=ELEC, size=10.5)
        label(ax, xc, yy, "↔", color=MUTED, size=9.5)
        label(ax, xc + 0.42, yy, b_, color=MAG, size=10.5)
    # --- row 3: insulation contrast
    rbox(ax, xL + 0.1, 0.18, wp - 0.2, 0.95, fc=tint(CURR, 0.08), ec="none")
    label(ax, xL + wp / 2, 0.82, T("空气是很好的“绝缘体”", "air is an excellent insulator"), size=9, weight="bold", color=INK)
    label(ax, xL + wp / 2, 0.42, T(r"$\sigma_{\rm Cu}/\sigma_{\rm air}\gtrsim10^{21}$：漏电流可以忽略", r"$\sigma_{\rm Cu}/\sigma_{\rm air}\gtrsim10^{21}$: leakage current negligible"), size=8.8)
    rbox(ax, xR + 0.1, 0.18, wp - 0.2, 0.95, fc=tint(MAG, 0.10), ec="none")
    label(ax, xR + wp / 2, 0.82, T("磁通没有“绝缘体”", "flux has no insulator"), size=9, weight="bold", color=MAG)
    label(ax, xR + wp / 2, 0.42, T(r"$\mu_{\rm Fe}/\mu_0\sim10^{3}$–$10^{5}$：漏磁常常不可忽略", r"$\mu_{\rm Fe}/\mu_0\sim10^{3}$–$10^{5}$: leakage often matters"), size=8.8)
    label(ax, xc, 0.65, T("同形\n不同量级", "same form,\nnot same\nscale"), size=8.2, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
def draw_c_core(ax, x0, y0, s, gap_draw=0.16, n_turns=6, flux=True, path=True):
    """C-core of Example 14.1 (outer 0.10 x 0.09 m, limbs 0.02 m) at scale s (units per metre).
    The gap is exaggerated to `gap_draw` drawing units."""
    W, H, w = 0.10 * s, 0.09 * s, 0.02 * s
    yc = y0 + 0.045 * s
    g0, g1 = yc - gap_draw / 2, yc + gap_draw / 2
    iron_frame(ax, x0, y0, W, H, w, gap=(g0, g1))
    # gap field
    for xx in np.linspace(x0 + W - w + 0.1, x0 + W - 0.1, 3):
        arrow(ax, (xx, g1 + 0.02), (xx, g0 - 0.02), color=MAG, lw=1.0, ms=6, z=5)
    # coil on the left limb
    ys = coil(ax, x0, x0 + w, y0 + 0.028 * s, y0 + 0.06 * s, n=n_turns)
    wire(ax, [(x0 - 0.1, ys[0]), (x0 - 0.45, ys[0] - 0.25)], color=CURR, lw=1.5)
    wire(ax, [(x0 + w + 0.1, ys[-1] + 0.13), (x0 - 0.45, ys[-1] + 0.45)], color=CURR, lw=1.5)
    arrow(ax, (x0 - 0.62, ys[0] - 0.37), (x0 - 0.3, ys[0] - 0.15), color=CURR, lw=1.3, ms=9)
    if path:
        m = w / 2
        pts = [(x0 + m, y0 + m), (x0 + W - m, y0 + m), (x0 + W - m, y0 + H - m), (x0 + m, y0 + H - m), (x0 + m, y0 + m)]
        wire(ax, pts, color=MAG, lw=1.0, ls=(0, (4, 2.5)), z=4)
        if flux:
            arrow(ax, (x0 + m, yc - 0.25), (x0 + m, yc + 0.15), color=MAG, lw=1.6, ms=11, z=7)
            arrow(ax, (x0 + W / 2 - 0.2, y0 + H - m), (x0 + W / 2 + 0.2, y0 + H - m), color=MAG, lw=1.6, ms=11, z=7)
            arrow(ax, (x0 + W / 2 + 0.2, y0 + m), (x0 + W / 2 - 0.2, y0 + m), color=MAG, lw=1.6, ms=11, z=7)
    return dict(W=W, H=H, w=w, yc=yc, g0=g0, g1=g1, coil_y=ys)


@figure("gapped_core_network")
def gapped_core_network():
    a_ = LM.gapped_core()
    fig, ax = canvas(FIG_W, 2.85, xlim=(0, 10.5), ylim=(0, 4.99))
    s = 34.0
    x0, y0 = 1.05, 0.95
    d = draw_c_core(ax, x0, y0, s)
    W, H, w = d["W"], d["H"], d["w"]
    label(ax, x0 - 0.75, d["coil_y"][0] - 0.55, r"$I$", color=CURR, size=10)
    label(ax, x0 + w + 0.62, d["coil_y"][2] + 0.1, r"$N$" + T(" 匝", " turns"), color=CURR, size=9, ha="left")
    label(ax, x0 + W + 0.12, d["yc"], T(r"气隙 $g$", r"gap $g$"), size=9, ha="left", color=INK)
    label(ax, x0 + W / 2, y0 - 0.32, T(r"铁芯平均路径 $\ell_{\rm Fe}$（虚线），截面 $A$", r"mean iron path $\ell_{\rm Fe}$ (dashed), area $A$"), size=8.6, color=MUTED)
    label(ax, x0 + W / 2, y0 + H + 0.28, T("几何与场", "Geometry and field"), weight="bold", size=9.6)
    # compression arrow (above the gap label)
    arrow(ax, (4.85, 3.55), (5.6, 3.55), color=ACCENT, lw=2.0, ms=13)
    label(ax, 5.22, 3.95, T("压缩", "compress"), color=ACCENT, size=8.8)
    # network: source left, R_c on top, R_g on the right
    xa, xb, yb, yt = 6.25, 8.45, 0.95, 3.55
    wire(ax, [(xa, yb), (xa, yt), (xb, yt), (xb, yb), (xa, yb)])
    mmf_src(ax, (xa, 1.65), r"$\mathcal{F}=NI$", side="left", size=9.2)
    rel(ax, ((xa + xb) / 2, yt), vertical=False, text=None)
    label(ax, (xa + xb) / 2, yt + 0.62, r"$\mathcal{R}_{\rm Fe}=\dfrac{\ell_{\rm Fe}}{\mu A}$", size=9.2)
    rel(ax, (xb, 2.25), vertical=True, text=None)
    label(ax, xb - 0.25, 2.25, r"$\mathcal{R}_g=\dfrac{g}{\mu_0 A}$", size=9.2, ha="right")
    flux_arrow(ax, ((xa + xb) / 2 + 0.25, yb + 0.3), ((xa + xb) / 2 - 0.25, yb + 0.3))
    label(ax, (xa + xb) / 2, yb + 0.6, r"$\Phi$", color=MAG, size=10)
    label(ax, (xa + xb) / 2, 4.72, T("磁路网络", "Network"), weight="bold", size=9.6)
    # MMF split bar (computed from Example 14.1)
    bx, bw, by0, bh = 9.85, 0.42, 0.95, 2.6
    fg = a_["frac_gap"]
    ax.add_patch(Rectangle((bx, by0), bw, bh * fg, fc=tint(MAG, 0.55), ec=MAG, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((bx, by0 + bh * fg), bw, bh * (1 - fg), fc=IRON_T, ec=IRON_E, lw=0.8, zorder=3))
    label(ax, bx + bw / 2, by0 + bh * fg / 2, f"{100*fg:.0f}%", size=8.6, color="white", weight="bold", rotation=90)
    label(ax, bx + bw / 2, by0 + bh * (1 + fg) / 2, f"{100*(1-fg):.0f}%", size=7.6, color=INK, rotation=90)
    label(ax, bx + bw / 2, by0 + bh + 0.45, T("磁动势\n分配", "MMF\nsplit"), size=8.4, color=INK)
    label(ax, bx - 0.1, by0 + bh * fg / 2, T("气隙", "gap"), size=8.4, ha="right", color=MAG)
    label(ax, bx - 0.1, by0 + bh * (1 + fg) / 2, T("铁", "iron"), size=8.4, ha="right", color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("three_limb_network")
def three_limb_network():
    e = LM.ecore()
    p = LM.ECORE
    fig, ax = canvas(FIG_W, 3.0, xlim=(0, 10.5), ylim=(0, 5.25))
    s = 33.0
    x0, y0 = 0.75, 0.75
    Wc, Hc, wo, wc_, t = 0.12 * s, 0.10 * s, 0.02 * s, 0.04 * s, 0.02 * s
    gd = 0.16
    ymid = y0 + Hc / 2
    # outline: outer rectangle minus two windows, gaps cut in centre and right limbs
    iron_rect(ax, x0, y0, Wc, Hc)
    for wx in (x0 + wo, x0 + wo + 0.02 * s + wc_):
        ax.add_patch(Rectangle((wx, y0 + t), 0.02 * s, Hc - 2 * t, fc="white", ec=IRON_E, lw=0.8, zorder=2.1))
    ax.add_patch(Rectangle((x0 + wo + 0.02 * s - 0.01, ymid - gd / 2), wc_ + 0.02, gd, fc="white", ec="none", zorder=2.2))
    ax.plot([x0 + wo + 0.02 * s, x0 + wo + 0.02 * s + wc_], [ymid - gd / 2] * 2, color=IRON_E, lw=0.8, zorder=2.3)
    ax.plot([x0 + wo + 0.02 * s, x0 + wo + 0.02 * s + wc_], [ymid + gd / 2] * 2, color=IRON_E, lw=0.8, zorder=2.3)
    gr = gd * 0.45
    ax.add_patch(Rectangle((x0 + Wc - wo - 0.01, ymid - gr / 2), wo + 0.02, gr, fc="white", ec="none", zorder=2.2))
    ax.plot([x0 + Wc - wo, x0 + Wc], [ymid - gr / 2] * 2, color=IRON_E, lw=0.8, zorder=2.3)
    ax.plot([x0 + Wc - wo, x0 + Wc], [ymid + gr / 2] * 2, color=IRON_E, lw=0.8, zorder=2.3)
    # coil on left limb
    ys = coil(ax, x0, x0 + wo, y0 + 0.9, y0 + Hc - 1.0, n=5)
    label(ax, x0 - 0.2, ys[0] - 0.35, r"$N_1I_1$", color=CURR, size=9.5, ha="right")
    # flux arrows with width ~ flux
    PhiL, PhiC, PhiR = e["Phi"]["L"], -e["Phi"]["C"], -e["Phi"]["R"]
    def lw_of(phi):
        return 1.0 + 3.2 * phi / PhiL
    xL, xC, xR = x0 + wo / 2, x0 + Wc / 2, x0 + Wc - wo / 2
    yT, yB = y0 + Hc - t / 2, y0 + t / 2
    for pts, phi in (([(xL, yB + 0.3), (xL, yT), (xC - 0.1, yT)], PhiL),):
        pass
    arrow(ax, (xL, ymid - 0.2), (xL, ymid + 0.45), color=MAG, lw=lw_of(PhiL), ms=11 + 2 * lw_of(PhiL), z=7)
    arrow(ax, (xC, ymid + 0.7), (xC, ymid + 0.25), color=MAG, lw=lw_of(PhiC), ms=9 + 2 * lw_of(PhiC), z=7)
    arrow(ax, (xR, ymid + 0.7), (xR, ymid + 0.25), color=MAG, lw=lw_of(PhiR), ms=9 + 2 * lw_of(PhiR), z=7)
    arrow(ax, (xL + 0.35, yT), (xL + 0.95, yT), color=MAG, lw=lw_of(PhiL), ms=11 + 2 * lw_of(PhiL), z=7)
    sc = 1e4
    halo(label(ax, xL + 1.05, yT, f"{PhiL*sc:.2f}", color=MAG, size=8.6, ha="left", weight="bold"), 2.5)   # beside the top-yoke arrow, which carries Phi_L
    halo(label(ax, xC + 0.08, ymid - 0.45, f"{PhiC*sc:.2f}", color=MAG, size=8.6, ha="left", weight="bold"), 2.5)
    halo(label(ax, xR, ymid - 0.45, f"{PhiR*sc:.2f}", color=MAG, size=8.6, ha="center", weight="bold"), 2.5)
    label(ax, x0 + Wc / 2, y0 - 0.3, T(r"磁通（$10^{-4}$ Wb），箭头粗细正比于磁通", r"flux ($10^{-4}$ Wb); arrow width proportional to flux"),
          size=8.4, color=MUTED)
    label(ax, xC, y0 + Hc + 0.28, T(r"气隙：中柱 1.0 mm，右柱 0.25 mm", r"gaps: center 1.0 mm, right 0.25 mm"), size=8.6)
    # node markers on the core
    for yy, nm in ((yT, "T"), (yB, "B")):
        ax.add_patch(Circle((xC, yy), 0.08, fc="white", ec=INK, lw=0.9, zorder=8))
        label(ax, xC + 0.2, yy + (0.22 if nm == "T" else -0.22), nm, size=8.5, weight="bold", color=INK)
    # ---- network
    nx0 = 6.0
    xs_ = [nx0, nx0 + 1.75, nx0 + 3.5]
    ytn, ybn = 4.2, 0.9
    wire(ax, [(xs_[0], ytn), (xs_[2], ytn)])
    wire(ax, [(xs_[0], ybn), (xs_[2], ybn)])
    for x in xs_:
        wire(ax, [(x, ybn), (x, ytn)])
    node(ax, xs_[1], ytn)
    node(ax, xs_[1], ybn)
    label(ax, xs_[1], ytn + 0.3, T(rf"T：$\psi_T={e['psi']:.0f}$ A", rf"T: $\psi_T={e['psi']:.0f}$ A"), size=9)
    label(ax, xs_[1], ybn - 0.3, T(r"B：参考 $\psi_B=0$", r"B: reference $\psi_B=0$"), size=9)
    mmf_src(ax, (xs_[0], 1.75), r"$N_1I_1$", side="left", size=9.2)
    R = e["R"]
    rel(ax, (xs_[0], 3.15), text=r"$\mathcal{R}_L$", side="left")
    rel(ax, (xs_[1], 2.55), text=r"$\mathcal{R}_C$", side="right")
    rel(ax, (xs_[2], 2.55), text=r"$\mathcal{R}_R$", side="right")
    label(ax, xs_[0] - 0.42, 2.55, f"{R['L']/1e5:.2f}", size=8, color=MUTED, ha="right")
    label(ax, xs_[1] + 0.42, 2.12, f"{R['C']/1e5:.1f}", size=8, color=MUTED, ha="left")
    label(ax, xs_[2] + 0.42, 2.12, f"{R['R']/1e5:.2f}", size=8, color=MUTED, ha="left")
    label(ax, nx0 + 1.75, 0.3, T(r"磁阻单位：$10^{5}$ A/Wb", r"reluctance in $10^{5}$ A/Wb"), size=8.2, color=MUTED)
    flux_arrow(ax, (xs_[0] + 0.22, 2.2), (xs_[0] + 0.22, 2.75), r"$\Phi_L$", tpos=(xs_[0] + 0.48, 2.48), ha="left")
    flux_arrow(ax, (xs_[1] - 0.22, 3.55), (xs_[1] - 0.22, 3.0), r"$\Phi_C$", tpos=(xs_[1] - 0.5, 3.28), ha="right")
    flux_arrow(ax, (xs_[2] - 0.22, 3.55), (xs_[2] - 0.22, 3.0), r"$\Phi_R$", tpos=(xs_[2] - 0.5, 3.28), ha="right")
    return fig


# ---------------------------------------------------------------------------
_FEM_CACHE: dict = {}


def fem_solution(g, **kw):
    key = (g, tuple(sorted(kw.items())))
    if key not in _FEM_CACHE:
        _FEM_CACHE[key] = LM.c_core_fem(g, keep=True, **kw)
    return _FEM_CACHE[key]


def draw_fem_field(ax, sol, xlim, ylim, n_levels, lw=0.75, shade=True, vmax=None):
    tri = mtri.Triangulation(sol["p"][:, 0], sol["p"][:, 1], sol["cells"])
    c = sol["p"][sol["cells"]].mean(axis=1)
    inview = (c[:, 0] > xlim[0] - 0.01) & (c[:, 0] < xlim[1] + 0.01) & (c[:, 1] > ylim[0] - 0.01) & (c[:, 1] < ylim[1] + 0.01)
    tri.set_mask(~inview)
    if shade:
        vm = vmax or np.percentile(sol["Bmag"][inview & sol["iron"]], 95)
        ax.tripcolor(tri, facecolors=np.where(inview, sol["Bmag"], 0), cmap=seq_cmap(MAG), vmin=0, vmax=vm * 1.6,
                     shading="flat", zorder=1, alpha=0.55, rasterized=True)
    Az = sol["Az"]
    lo, hi = np.percentile(Az, 0.01), Az.max()
    levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
    ax.tricontour(tri, Az, levels=levels, colors=[MAG], linewidths=lw, zorder=3, negative_linestyles="solid")


def outline_core(ax, g, lw=0.9):
    W, Hh, w, yc = 0.10, 0.09, 0.02, 0.045
    pts = [(0, 0), (W, 0), (W, yc - g / 2), (W - w, yc - g / 2), (W - w, w), (w, w), (w, Hh - w), (W - w, Hh - w),
           (W - w, yc + g / 2), (W, yc + g / 2), (W, Hh), (0, Hh), (0, 0)]
    ax.plot(*zip(*pts), color=IRON_E, lw=lw, zorder=4)


@figure("fringing_leakage")
def fringing_leakage():
    g = 4.0e-3
    sol = fem_solution(g)
    fig = plt.figure(figsize=(FIG_W, 3.05))
    axA = fig.add_axes([0.0, 0.03, 0.40, 0.88])
    axB = fig.add_axes([0.405, 0.03, 0.24, 0.88])
    axN = fig.add_axes([0.665, 0.0, 0.335, 1.0])
    # --- panel A: whole core
    xl, yl = (-0.026, 0.122), (-0.024, 0.114)
    axA.set_xlim(*xl); axA.set_ylim(*yl); axA.set_aspect("equal"); axA.axis("off")
    draw_fem_field(axA, sol, xl, yl, 22, lw=0.7)
    outline_core(axA, g)
    for (x0_, x1_), sym in (((-0.01, 0.0), "out"), ((0.02, 0.03), "in")):
        axA.add_patch(Rectangle((x0_, 0.025), x1_ - x0_, 0.04, fc=tint(CURR, 0.25), ec=CURR, lw=0.7, zorder=4))
        for yy in (0.035, 0.055):
            (out_of_page if sym == "out" else into_page)(axA, (x0_ + x1_) / 2, yy, color=CURR, r=0.0028, lw=0.7)
    axA.add_patch(Rectangle((0.068, 0.026), 0.044, 0.038, fc="none", ec=ACCENT, lw=0.9, ls=(0, (3, 2)), zorder=6))
    def ann(ax, xy, xyt, s, ha="center", color=INK):
        t = ax.annotate(s, xy=xy, xytext=xyt, fontsize=8.4, color=color, ha=ha, va="center", zorder=9,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=1, shrinkB=1))
        halo(t, 2.5)
    ann(axA, (0.05, 0.081), (0.05, 0.103), T("主磁通：沿铁芯", "main flux: along the iron"))
    ann(axA, (0.05, 0.045), (0.05, 0.045), T("漏磁：\n穿过窗口", "leakage\nacross\nwindow"), color=MAG)
    ann(axA, (-0.006, 0.075), (-0.012, 0.098), T("绕组外侧\n的漏磁", "leakage\noutside\ncoil"), color=MAG)
    panel_title(axA, T("二维场解（单位深度，g = 4 mm）", "2-D field solution (per depth, g = 4 mm)"), size=9.0)
    # --- panel B: zoom on the gap
    xl2, yl2 = (0.068, 0.112), (0.026, 0.064)
    axB.set_xlim(*xl2); axB.set_ylim(*yl2); axB.set_aspect("equal"); axB.axis("off")
    draw_fem_field(axB, sol, xl2, yl2, 64, lw=0.65)
    outline_core(axB, g, lw=1.0)
    axB.add_patch(Rectangle((xl2[0], yl2[0]), xl2[1] - xl2[0], yl2[1] - yl2[0], fc="none", ec=ACCENT, lw=0.9,
                            ls=(0, (3, 2)), zorder=6))
    ann(axB, (0.1018, 0.0485), (0.1045, 0.0595), T("边缘\n磁通", "fringing"), color=MAG)
    ann(axB, (0.09, 0.045), (0.09, 0.032), T("气隙中部近似均匀", "near-uniform mid-gap"), color=INK)
    panel_title(axB, T("气隙附近放大", "zoom at the gap"), size=9.0)
    # --- panel N: network with calibrated permeances
    axN.set_xlim(0, 4.2); axN.set_ylim(0, 5.1); axN.set_aspect("equal"); axN.axis("off")
    xa, xb, xc_, yb, yt = 0.95, 2.45, 3.6, 1.3, 4.1
    wire(axN, [(xa, yb), (xa, yt), (xc_, yt), (xc_, yb), (xa, yb)])
    wire(axN, [(xb, yb), (xb, yt)])
    mmf_src(axN, (xa, 2.2), r"$\mathcal{F}$", side="left", size=9.5)
    rel(axN, (xa, 3.3), text=r"$\mathcal{R}_{\rm Fe}$", side="left")
    rel(axN, (xb, 2.7), text=None)
    label(axN, xb - 0.2, 2.7, r"$k\mathcal{P}_g$", size=9.5, ha="right")
    rel(axN, (xc_, 2.7), text=r"$\mathcal{P}_{\rm leak}$", side="right", dashed=True)
    node(axN, xb, yt); node(axN, xb, yb)
    label(axN, 2.1, 0.75, T("气隙磁导乘以 $k$（边缘），\n并联漏磁导 $\\mathcal{P}_{\\rm leak}$", "gap permeance times $k$ (fringing),\nleakage $\\mathcal{P}_{\\rm leak}$ in parallel"), size=8.3)
    label(axN, 2.1, 0.12, T("$k$ 与 $\\mathcal{P}_{\\rm leak}$ 只能由场解或测量标定", "$k$, $\\mathcal{P}_{\\rm leak}$: calibrate by field solution or measurement"),
          size=8.1, color=WARN_C)
    label(axN, 2.1, 4.65, T("带边缘与漏磁的网络", "Network with fringing, leakage"), weight="bold", size=9.0)
    return fig


WARN_C = "#A33A3A"


# ---------------------------------------------------------------------------
@figure("nonlinear_operating_point")
def nonlinear_operating_point():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 2.75), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    l, g = LM.CORE["lc"], LM.CORE["g"]
    H = np.linspace(0, 9000, 1200)
    a1.plot(H, synthetic_B(H), color=MAG, lw=2.0, zorder=3, label=T("铁芯 B–H 曲线", "iron B–H curve"))
    cols = [tint(ACCENT, 0.45), tint(ACCENT, 0.7), ACCENT]
    for NI, c in zip((400.0, 1000.0, 2000.0), cols):
        o = LM.core_operating_point(NI)
        Hl = np.linspace(0, NI / l, 50)
        a1.plot(Hl, MU0 * (NI - l * Hl) / g, color=c, lw=1.2, ls=(0, (4, 2)), zorder=2)
        a1.plot(o["H"], o["B"], "o", ms=5.5, mfc="white", mec=MAG, mew=1.4, zorder=5)
        a1.text(o["H"] + 180, o["B"] + (0.05 if NI < 2000 else 0.09), f"{NI:.0f} A", fontsize=8.3, color=ACCENT,
                ha="left", va="bottom")
    a1.set_xlim(0, 7000); a1.set_ylim(0, 2.3)
    a1.set_xlabel(T(r"铁中磁场强度 $H_{\rm Fe}$（A/m）", r"iron intensity $H_{\rm Fe}$ (A/m)"))
    a1.set_ylabel(r"$B$ (T)")
    a1.text(4250, 1.02, T("气隙“负载线”\n$B=\\mu_0(NI-\\ell_{\\rm Fe}H_{\\rm Fe})/g$", "gap “load line”\n$B=\\mu_0(NI-\\ell_{\\rm Fe}H_{\\rm Fe})/g$"),
            fontsize=8.4, color=ACCENT, ha="left", va="bottom")
    clean_axes(a1)
    panel_title(a1, T("(a) 图解：直线与曲线的交点", "(a) graphical: line meets curve"), size=9.2)
    # (b) B_g vs NI
    NIs = np.linspace(1, 3000, 160)
    Bn = np.array([LM.core_operating_point(x)["B"] for x in NIs])
    a2.plot(NIs, MU0 * NIs / g, color=MUTED, lw=1.2, ls=(0, (1.5, 1.5)), label=T(r"理想铁 $\mu\to\infty$", r"ideal iron $\mu\to\infty$"))
    a2.plot(NIs, MU0 * NIs / (g + l / LM.CORE["mur"]), color=ACCENT, lw=1.3, ls=(0, (4, 2)), label=T(r"线性铁 $\mu_r=2000$", r"linear iron $\mu_r=2000$"))
    a2.plot(NIs, Bn, color=MAG, lw=2.0, label=T("合成 B–H 曲线", "synthetic B–H curve"))
    for NI in (400.0, 1000.0, 2000.0):
        o = LM.core_operating_point(NI)
        a2.plot(NI, o["B"], "o", ms=5, mfc="white", mec=MAG, mew=1.3, zorder=5)
    a2.set_xlim(0, 3000); a2.set_ylim(0, 2.3)
    a2.set_xlabel(T(r"磁动势 $NI$（A）", r"MMF $NI$ (A)"))
    a2.set_ylabel(T(r"气隙 $B_g$（T）", r"gap $B_g$ (T)"))
    a2.legend(loc="upper left", fontsize=8.2, handlelength=1.8)
    clean_axes(a2)
    panel_title(a2, T("(b) 气隙磁场随磁动势", "(b) gap field versus MMF"), size=9.2)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.9, wspace=0.3)
    return fig


# ---------------------------------------------------------------------------
@figure("pm_equivalents")
def pm_equivalents():
    pm = LM.pm_single_loop()
    fig = plt.figure(figsize=(FIG_W, 2.7))
    a1 = fig.add_axes([0.075, 0.2, 0.27, 0.66])
    ax = fig.add_axes([0.37, 0.0, 0.63, 1.0])
    # (a) recoil line
    Hc = pm["Hc"]
    Hm = np.linspace(-Hc * 1.08, 0, 50)
    a1.plot(Hm / 1e3, 1.2 + MU0 * 1.05 * Hm, color=MAG, lw=2.0)
    a1.plot([-Hc / 1e3], [0], "o", ms=5.5, color=MAG)
    a1.plot([0], [1.2], "o", ms=5.5, color=MAG)
    a1.set_xlim(-1000, 60); a1.set_ylim(0, 1.45)
    a1.axvline(0, color=MUTED, lw=0.7); a1.axhline(0, color=MUTED, lw=0.7)
    a1.set_xlabel(T(r"$H_m$（kA/m）", r"$H_m$ (kA/m)"))
    a1.set_ylabel(r"$B_m$ (T)")
    a1.text(-40, 1.27, T(r"$B_r$：外部短路", r"$B_r$: external short"), fontsize=8.2, ha="right", color=MAG)
    a1.text(-760, 0.07, T(r"$-H_c$：外部开路", r"$-H_c$: external open"), fontsize=8.2, ha="left", color=MAG)
    a1.text(-560, 0.78, T("回复线", "recoil line"), fontsize=8.4, color=MAG, rotation=30)
    clean_axes(a1)
    a1.set_title(T("(a) 同一条回复线", "(a) one recoil line"), fontsize=9.2, fontweight="bold", pad=4)
    # (b)/(c) equivalent circuits
    ax.set_xlim(0, 7.1); ax.set_ylim(0, 3.2); ax.set_aspect("equal"); ax.axis("off")
    yb, yt = 0.6, 2.35
    # Thevenin
    x1, x2 = 1.05, 2.75
    wire(ax, [(x1, yb), (x1, yt), (x2, yt)])
    wire(ax, [(x1, yb), (x2, yb)])
    mmf_src(ax, (x1, 1.05), None)
    label(ax, x1 - 0.32, 1.05, r"$\mathcal{F}_c$", size=9.5, ha="right")
    rel(ax, (x1, 1.88), text=r"$\mathcal{R}_m$", side="left")
    for yy in (yb, yt):
        ax.add_patch(Circle((x2, yy), 0.06, fc="white", ec=INK, lw=1.0, zorder=8))
    rbox(ax, 0.2, 0.38, 1.55, 2.2, fc="none", ec=tint(MAG, 0.5), lw=0.8, ls=(0, (3, 2)), z=1)
    label(ax, 1.6, 2.98, T("(b) 磁动势源 + 串联磁阻", r"(b) MMF source + series $\mathcal{R}_m$"), size=9.0, weight="bold")
    label(ax, 1.6, 0.06, r"$\mathcal{F}_c=H_c\ell_m,\ \ \mathcal{R}_m=\dfrac{\ell_m}{\mu_0\mu_{\rm rec}A_m}$", size=8.8)
    # Norton
    x3, x4, x5 = 4.4, 5.4, 6.75
    wire(ax, [(x3, yb), (x3, yt), (x5, yt)])
    wire(ax, [(x3, yb), (x5, yb)])
    wire(ax, [(x4, yb), (x4, yt)])
    flux_src(ax, (x3, 1.48), None)
    label(ax, x3 - 0.32, 1.48, r"$\Phi_r$", size=9.5, ha="right")
    rel(ax, (x4, 1.48), text=r"$\mathcal{P}_m$", side="right")
    node(ax, x4, yt); node(ax, x4, yb)
    for yy in (yb, yt):
        ax.add_patch(Circle((x5, yy), 0.06, fc="white", ec=INK, lw=1.0, zorder=8))
    rbox(ax, 3.6, 0.38, 2.45, 2.2, fc="none", ec=tint(MAG, 0.5), lw=0.8, ls=(0, (3, 2)), z=1)
    label(ax, 5.3, 2.98, T("(c) 磁通源 + 并联磁导", r"(c) flux source + parallel $\mathcal{P}_m$"), size=9.0, weight="bold")
    label(ax, 5.3, 0.06, r"$\Phi_r=B_rA_m,\ \ \mathcal{P}_m=1/\mathcal{R}_m$", size=8.8)
    label(ax, 3.2, 1.48, "≡", size=16, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("load_line")
def load_line():
    pm = LM.pm_single_loop()
    r = pm["r"]
    Br, mur, lm = 1.2, 1.05, 0.05
    Hc = pm["Hc"]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.25))
    Hm = np.linspace(-1.0e6, 0, 400)
    ax.plot(Hm / 1e3, Br + MU0 * mur * Hm, color=MAG, lw=2.2, zorder=4,
            label=T(r"回复线 $B_m=B_r+\mu_0\mu_{\rm rec}H_m$", r"recoil line $B_m=B_r+\mu_0\mu_{\rm rec}H_m$"))
    ax.plot([], [], color=ACCENT, lw=1.2, ls=(0, (4, 2)), label=T(r"负载线 $B_m=-\mu_0(\ell_m/g)H_m$", r"load line $B_m=-\mu_0(\ell_m/g)H_m$"))
    Hh = np.linspace(-1.0e6, -1.9e5, 300)
    ax.plot(Hh / 1e3, pm["BHmax"] / np.abs(Hh), color=MUTED, lw=1.0, ls=(0, (1.5, 1.5)), zorder=2,
            label=T(r"等磁能积曲线 $|B_mH_m|=(BH)_{\max}$", r"$|B_mH_m|=(BH)_{\max}$"))
    # nonlinear-yoke load curve of the worked example (A_y = A_g / 2)
    Hy = np.linspace(0, 6.5e4, 2000)
    Bm_c = synthetic_B(Hy) * r["Ay_m2"] / r["Am_m2"]
    Hm_c = -(r["gap_m"] * Bm_c / MU0 + r["yoke_length_m"] * Hy) / r["magnet_length_m"]
    ax.plot(Hm_c / 1e3, Bm_c, color=MAG, lw=1.3, ls=(0, (5, 1.5, 1, 1.5)), zorder=3,
            label=T("计入非线性磁轭的负载曲线（g = 20 mm）", "load curve with nonlinear yoke (g = 20 mm)"))
    for gg, txt in ((0.01, "10"), (0.02, "20"), (lm / mur, "47.6")):
        Hl = np.linspace(-1.0e6, 0, 10)
        ax.plot(Hl / 1e3, -MU0 * lm / gg * Hl, color=ACCENT, lw=1.5 if txt == "20" else 1.0, ls=(0, (4, 2)), zorder=3)
        Hop, Bop = LM.recoil_operating_point(gg)
        ax.plot(Hop / 1e3, Bop, "o", ms=5, mfc="white", mec=ACCENT, mew=1.3, zorder=6)
        Btop = 1.36
        Hlab = -Btop * gg / (MU0 * lm)
        if Hlab < -0.97e6:
            Hlab = -0.97e6
            Btop = MU0 * lm / gg * 0.97e6
            ax.text(Hlab / 1e3 + 6, Btop + 0.04, f"g = {txt} mm", fontsize=8.2, color=ACCENT, ha="left", va="bottom")
        else:
            ax.text(Hlab / 1e3 + 10, Btop, f"g = {txt} mm", fontsize=8.2, color=ACCENT, ha="left", va="center")
    ax.plot(-Hc / 2 / 1e3, Br / 2, "s", ms=5.5, mfc=MUTED, mec=MUTED, zorder=6)
    ax.annotate(T(r"$(BH)_{\max}$ 点", r"$(BH)_{\max}$ point"), xy=(-Hc / 2 / 1e3, Br / 2), xytext=(-770, 0.72),
                fontsize=8.4, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.plot(r["Hm_A_per_m"] / 1e3, r["Bm_T"], "D", ms=5.5, mfc=MAG, mec="white", mew=0.8, zorder=7)
    ax.annotate(T("非线性磁轭：0.8035 T", "nonlinear yoke: 0.8035 T"), xy=(r["Hm_A_per_m"] / 1e3, r["Bm_T"]),
                xytext=(-680, 1.16), fontsize=8.3, color=MAG, arrowprops=dict(arrowstyle="-", color=MAG, lw=0.7))
    Hi, Bi = LM.recoil_operating_point(0.02)
    ax.annotate(T("理想磁轭：0.8451 T", "ideal yoke: 0.8451 T"), xy=(Hi / 1e3, Bi), xytext=(-680, 1.01),
                fontsize=8.3, color=ACCENT, ha="left", arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))
    ax.set_xlim(-1000, 0); ax.set_ylim(0, 1.45)
    ax.set_xlabel(T(r"磁体内部 $H_m$（kA/m）", r"internal magnet $H_m$ (kA/m)"))
    ax.set_ylabel(r"$B_m$ (T)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.0, handlelength=2.2,
              columnspacing=1.2)
    clean_axes(ax)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.84)
    return fig


# ---------------------------------------------------------------------------
@figure("multipillar_yoke")
def multipillar_yoke():
    fig = plt.figure(figsize=(FIG_W, 4.35))
    axS = fig.add_axes([0.0, 0.5, 0.52, 0.5])
    axN = fig.add_axes([0.53, 0.5, 0.47, 0.5])
    a1 = fig.add_axes([0.085, 0.085, 0.38, 0.33])
    a2 = fig.add_axes([0.6, 0.085, 0.33, 0.33])
    # ---- sketch (side view, not to scale)
    ax = axS
    ax.set_xlim(0, 6.2); ax.set_ylim(0, 3.6); ax.set_aspect("equal"); ax.axis("off")
    X0, X1 = 0.55, 4.45
    iron_rect(ax, X0, 2.85, X1 - X0, 0.32)      # top plate
    iron_rect(ax, X0, 0.43, X1 - X0, 0.32)      # bottom plate
    for xp in (X0, X1 - 0.38):
        iron_rect(ax, xp, 0.75, 0.38, 2.1)       # pillars
    xm0, xm1 = 1.55, 3.45
    magnet_block(ax, xm0, 2.43, xm1 - xm0, 0.42, north_up=True)
    magnet_block(ax, xm0, 0.75, xm1 - xm0, 0.42, north_up=True)
    iron_rect(ax, xm0 - 0.08, 2.25, xm1 - xm0 + 0.16, 0.18, fc=POLE_T)      # pole pieces
    iron_rect(ax, xm0 - 0.08, 1.17, xm1 - xm0 + 0.16, 0.18, fc=POLE_T)
    for xx in np.linspace(xm0 + 0.25, xm1 - 0.25, 5):
        arrow(ax, (xx, 1.42), (xx, 2.18), color=MAG, lw=1.1, ms=7, z=5)
    label(ax, (xm0 + xm1) / 2, 1.8, r"$B_g$", size=9.5, color=MAG, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    # return flux
    for xs_, d in ((X0 + 0.19, -1), (X1 - 0.19, -1)):
        arrow(ax, (xs_, 2.2), (xs_, 1.4), color=MAG, lw=1.5, ms=10, z=6)
    arrow(ax, (2.9, 3.01), (3.8, 3.01), color=MAG, lw=1.3, ms=9, z=6)
    arrow(ax, (2.1, 3.01), (1.2, 3.01), color=MAG, lw=1.3, ms=9, z=6)
    arrow(ax, (1.2, 0.59), (2.1, 0.59), color=MAG, lw=1.3, ms=9, z=6)
    arrow(ax, (3.8, 0.59), (2.9, 0.59), color=MAG, lw=1.3, ms=9, z=6)
    label(ax, (X0 + X1) / 2, 3.42, T("侧视示意（不按比例）", "side view (not to scale)"), size=8.6, color=MUTED)
    label(ax, X0 + 0.19, 0.2, T("柱", "pillar"), size=8.4, color=MUTED)
    label(ax, X1 - 0.19, 0.2, T("柱", "pillar"), size=8.4, color=MUTED)
    label(ax, (X0 + X1) / 2, 0.2, T("轭板", "plate"), size=8.4, color=MUTED)
    label(ax, xm1 + 0.12, 2.34, T("极靴", "pole"), size=8.2, color=MUTED, ha="left")
    label(ax, xm1 + 0.12, 2.64, T("磁体", "magnet"), size=8.2, color=MAG, ha="left")
    # top view inset: N = 4 pillars
    cx, cy, rr = 5.35, 1.75, 0.5
    ax.add_patch(Circle((cx, cy), rr, fc=tint(MAG, 0.18), ec=MAG, lw=0.9, zorder=3, hatch="////"))
    for ang in np.deg2rad([45, 135, 225, 315]):
        ax.add_patch(Rectangle((cx + 0.82 * np.cos(ang) - 0.12, cy + 0.82 * np.sin(ang) - 0.12), 0.24, 0.24,
                               fc=IRON_T, ec=IRON_E, lw=0.8, zorder=3))
    label(ax, cx, cy + 1.25, T("俯视：N 根柱", "top view"), size=8.4, color=MUTED)
    # ---- network
    ax = axN
    ax.set_xlim(0, 5.5); ax.set_ylim(0, 3.9); ax.set_aspect("equal"); ax.axis("off")
    xa, yb, yt = 0.7, 0.55, 3.25
    xp = [2.75, 3.45, 4.15, 4.85]
    wire(ax, [(xa, yb), (xa, yt), (xp[-1], yt)])
    wire(ax, [(xa, yb), (xp[-1], yb)])
    mmf_src(ax, (xa, 1.0), None)
    label(ax, xa - 0.32, 1.0, r"$2\mathcal{F}_c$", size=9.2, ha="right")
    rel(ax, (xa, 1.85), text=r"$2\mathcal{R}_m$", side="left", L=0.5)
    rel(ax, (xa, 2.7), text=r"$\mathcal{R}_g$", side="left", L=0.5)
    # leakage permeance (dashed, not computed)
    xlk = 1.85
    wire(ax, [(xlk, yb), (xlk, yt)], ls=(0, (3, 2)), color=MUTED)
    rel(ax, (xlk, 1.9), text=None, dashed=True, color=MUTED, fill="white")
    label(ax, xlk, 3.55, T(r"$\mathcal{P}_{\rm leak}$ ?", r"$\mathcal{P}_{\rm leak}$ ?"), size=8.8, color=MUTED)
    for x in xp:
        wire(ax, [(x, yb), (x, yt)])
        rel(ax, (x, 1.9), text=None, L=0.55, fill=IRON_T, color=IRON_E)
        node(ax, x, yt, r=0.045); node(ax, x, yb, r=0.045)
    label(ax, (xp[0] + xp[-1]) / 2, 3.55, T(r"$N$ 根柱并联（非线性）", r"$N$ nonlinear pillars"), size=8.8)
    flux_arrow(ax, (xa + 0.22, 1.35), (xa + 0.22, 1.6))
    label(ax, xa + 0.42, 1.48, r"$\Phi$", color=MAG, size=9.5, ha="left")
    label(ax, 2.95, 0.12, T("漏磁（虚线）不在本例计算之内，须由场解标定", "leakage (dashed) is not computed; calibrate by field solution"),
          size=8.1, color=MUTED)
    # ---- curves
    ideal = LM.multipillar(4, 0.01)["ideal"]
    Ns = np.arange(1, 11)
    for Ap, c, mk in ((0.004, tint(MAG, 0.5), "s"), (0.005, MAG, "o"), (0.007, "#7a2d00", "^")):
        Bg = [LM.multipillar(int(n), Ap)["Bg"] for n in Ns]
        a1.plot(Ns, Bg, color=c, lw=1.4, marker=mk, ms=4.2, label=rf"$A_p={Ap*1e4:.0f}$ cm$^2$" if False else
                T(rf"$A_p={Ap:.3f}$ m²", rf"$A_p={Ap:.3f}$ m²"))
    a1.axhline(ideal, color=MUTED, lw=1.0, ls=(0, (1.5, 1.5)))
    a1.text(10.2, ideal + 0.006, T("理想磁轭", "ideal yoke"), fontsize=8.2, color=MUTED, ha="right", va="bottom")
    a1.set_xlim(0.5, 10.5); a1.set_ylim(0, 0.215)
    a1.set_xticks([1, 2, 3, 4, 6, 8, 10])
    a1.set_xlabel(T(r"柱数 $N$", r"number of pillars $N$"))
    a1.set_ylabel(T(r"$B_g$（T）", r"$B_g$ (T)"))
    a1.legend(loc="lower right", fontsize=7.8, handlelength=1.6)
    clean_axes(a1)
    a1.set_title(T("(c) 气隙磁场随柱数", "(c) gap field versus N"), fontsize=9.0, fontweight="bold", pad=3)
    # collapse onto total area
    tot = np.linspace(0.004, 0.06, 220)
    ratio = np.array([LM.multipillar(4, t / 4)["ratio"] for t in tot])
    Bp = np.array([LM.multipillar(4, t / 4)["Bp"] for t in tot])
    a2.plot(tot, ratio, color=MAG, lw=1.8)
    for n, mk in ((2, "s"), (4, "o"), (6, "^")):
        sel = [0.008, 0.016, 0.024, 0.032, 0.048]
        a2.plot(sel, [LM.multipillar(n, t / n)["ratio"] for t in sel], mk, ms=3.8, mfc="white", mec=MAG, mew=1.0)
    a2.set_ylim(0, 1.1); a2.set_xlim(0, 0.06)
    a2.set_xlabel(T(r"柱总截面 $NA_p$（m²）", r"total pillar area $NA_p$ (m²)"))
    a2.set_ylabel(T(r"$B_g/B_{g,\rm ideal}$", r"$B_g/B_{g,\rm ideal}$"), color=MAG)
    a2.tick_params(axis="y", colors=MAG)
    b2 = a2.twinx()
    b2.plot(tot, Bp, color=MUTED, lw=1.3, ls=(0, (4, 2)))
    b2.set_ylim(0, 2.2)
    b2.set_ylabel(T(r"柱中 $B_p$（T）", r"pillar $B_p$ (T)"), color=MUTED)
    b2.tick_params(axis="y", colors=MUTED, labelsize=8)
    b2.spines[["top"]].set_visible(False)
    a2.spines[["top"]].set_visible(False)
    a2.text(0.059, 0.64, T("实线：比值\n虚线：柱中 B\n点：N=2,4,6", "solid: ratio\ndashed: pillar B\nmarkers: N=2,4,6"),
            fontsize=7.8, color=INK, ha="right", va="bottom")
    a2.set_title(T("(d) 只依赖柱总截面", "(d) depends on total area only"), fontsize=9.0, fontweight="bold", pad=3)
    return fig


# ---------------------------------------------------------------------------
@figure("lumped_validation")
def lumped_validation():
    gaps = LM.FEM_GAPS
    res = [LM.c_core_fem(g) for g in gaps]
    gw = np.array([r_["gw"] for r_ in res])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 2.55))
    a1.plot(gw, [r_["limb_ratio"] for r_ in res], "o-", color=MAG, lw=1.6, ms=4.5,
            label=T("绕组铁柱中的磁通（决定电感）", "flux in the wound limb (sets L)"))
    a1.plot(gw, [r_["Bc_ratio"] for r_ in res], "s-", color=ACCENT, lw=1.6, ms=4.2,
            label=T("气隙中心的 $B$", "$B$ at the gap center"))
    a1.axhline(1, color=MUTED, lw=0.8)
    a1.set_xlim(0, 0.42); a1.set_ylim(0.8, 2.7)
    a1.set_xlabel(T(r"气隙/柱宽 $g/w$", r"gap / limb width $g/w$"))
    a1.set_ylabel(T("场解 / 集总模型", "field solution / lumped"))
    a1.legend(loc="upper left", fontsize=8.0, handlelength=1.8)
    clean_axes(a1)
    a1.set_title(T("(a) 同一几何，两种模型", "(a) one geometry, two models"), fontsize=9.0, fontweight="bold", pad=3)
    k = np.array([r_["k_eff"] for r_ in res])
    a2.plot(gw, k, "o-", color=MAG, lw=1.6, ms=4.5, label=T(r"由场解反推的 $k$", r"$k$ inferred from field solution"))
    k1 = res[1]["k_eff"]
    a2.axhline(k1, color=ACCENT, lw=1.1, ls=(0, (4, 2)), label=T(r"在 $g/w=0.05$ 处标定后固定", r"calibrated at $g/w=0.05$, then frozen"))
    a2.plot(res[1]["gw"], k1, "o", ms=7, mfc="none", mec=ACCENT, mew=1.4)
    a2.annotate(T("外推到 g/w=0.4：\n磁通低估约一半", "extrapolated to g/w=0.4:\nflux about half too low"),
                xy=(0.4, k1), xytext=(0.2, 1.95), fontsize=8.0, color=ACCENT, ha="left",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))
    a2.set_xlim(0, 0.42); a2.set_ylim(1.0, 2.8)
    a2.set_xlabel(T(r"气隙/柱宽 $g/w$", r"gap / limb width $g/w$"))
    a2.set_ylabel(T(r"等效面积系数 $k$", r"effective-area factor $k$"))
    a2.legend(loc="upper left", fontsize=8.0, handlelength=1.8)
    clean_axes(a2)
    a2.set_title(T("(b) 修正系数随几何变化", "(b) the correction changes with geometry"), fontsize=9.0, fontweight="bold", pad=3)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.9, wspace=0.28)
    return fig
