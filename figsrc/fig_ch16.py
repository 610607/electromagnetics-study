"""Figures for Ch 16 (scalar finite elements: from conservation to a matrix).

All computed quantities come from checks/check_ch16.py (which uses numerics.py), so the figures,
the text and the assertions share one implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib import gridspec

from style import (ACCENT, CURR, ELEC, FIG_W, INK, LIGHT, MATL, MATL_T, MUTED, POT, T, arrow, canvas,
                   clean_axes, figure, halo, label, panel_title, plt, rbox, seq_cmap, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch16 as C16  # noqa: E402


# ---------------------------------------------------------------------------
def _bracket(ax, x0, x1, y0, y1, color=INK, lw=1.0, tick=0.09):
    """Square matrix brackets around [x0,x1] x [y0,y1]."""
    for x, s in ((x0, 1), (x1, -1)):
        ax.plot([x + s * tick, x, x, x + s * tick], [y0, y0, y1, y1], color=color, lw=lw,
                solid_capstyle="butt", zorder=5)


@figure("two_element_bar")
def two_element_bar():
    b = C16.bar_dirichlet()
    fig, ax = canvas(FIG_W, 3.55, xlim=(0, 10.6), ylim=(0, 6.27))
    # ------------------------------------------------------------------ left: geometry
    xs = [0.75, 3.05, 5.35]                       # node positions (20 mm bar)
    yb0, yb1 = 2.55, 3.15                         # bar body
    dz = 0.22
    ax.add_patch(Rectangle((xs[0], yb0), xs[2] - xs[0], yb1 - yb0, fc=MATL_T, ec=MATL, lw=0.9, zorder=2))
    ax.add_patch(Polygon([(xs[0], yb1), (xs[0] + dz, yb1 + dz * 0.75), (xs[2] + dz, yb1 + dz * 0.75),
                          (xs[2], yb1)], fc=tint("#C9B79C", 0.5), ec=MATL, lw=0.8, zorder=2))
    ax.add_patch(Polygon([(xs[2], yb0), (xs[2] + dz, yb0 + dz * 0.75), (xs[2] + dz, yb1 + dz * 0.75),
                          (xs[2], yb1)], fc=tint("#C9B79C", 0.65), ec=MATL, lw=0.8, zorder=2))
    # node cross-sections and dots
    for k, x in enumerate(xs):
        ax.plot([x, x], [yb0, yb1], color=INK, lw=0.8, ls=(0, (2, 1.6)), zorder=3)
        ax.add_patch(plt.Circle((x, (yb0 + yb1) / 2), 0.075, fc=INK, ec="white", lw=0.8, zorder=6))
        label(ax, x, yb0 - 0.27, T(f"节点 {k + 1}", f"node {k + 1}"), size=8.6, color=INK)
    # current arrows inside (green) and E (blue) above the bar
    for x0 in (xs[0], xs[1]):
        halo(arrow(ax, (x0 + 0.45, (yb0 + yb1) / 2), (x0 + 1.85, (yb0 + yb1) / 2), color=CURR, lw=1.5, ms=9, z=5), 2.5)
    # element labels
    for k, (xa, xb) in enumerate(((xs[0], xs[1]), (xs[1], xs[2]))):
        label(ax, (xa + xb) / 2, yb0 - 0.72, T(f"单元 {k + 1}", f"element {k + 1}"), size=8.8, weight="bold", color=ACCENT)
        label(ax, (xa + xb) / 2, yb0 - 1.07, T(r"$G=\sigma A/h=5\,$mS", r"$G=\sigma A/h=5\,$mS"), size=8.6, color=INK)
        arrow(ax, (xa + 0.04, yb0 - 1.42), (xb - 0.04, yb0 - 1.42), color=MUTED, lw=0.8, ms=6, style="<|-|>")
        label(ax, (xa + xb) / 2, yb0 - 1.65, r"$h=10\,$mm", size=8.3, color=MUTED)
    # terminals
    ax.add_patch(Rectangle((xs[0] - 0.16, yb0 - 0.12), 0.16, yb1 - yb0 + 0.24, fc="#5B6670", ec="none", zorder=4))
    ax.add_patch(Rectangle((xs[2], yb0 - 0.12), 0.16, yb1 - yb0 + 0.24, fc="#5B6670", ec="none", zorder=4))
    # injected / extracted current; the right terminal returns the current to ground
    yc = (yb0 + yb1) / 2
    arrow(ax, (0.02, yc), (xs[0] - 0.2, yc), color=CURR, lw=1.6, ms=9)
    label(ax, -0.02, yc + 0.36, r"$1\,$mA", color=CURR, size=8.4, ha="left")
    gx = xs[2] + 0.95
    arrow(ax, (xs[2] + 0.2, yc), (gx - 0.06, yc), color=CURR, lw=1.6, ms=9)
    label(ax, xs[2] + 0.62, yc + 0.42, r"$1\,$mA", color=CURR, size=8.6, ha="left")
    ax.plot([gx, gx], [yc, yb0 - 0.55], color=INK, lw=0.9)
    for k, w in enumerate((0.2, 0.13, 0.06)):
        ax.plot([gx - w, gx + w], [yb0 - 0.55 - 0.08 * k] * 2, color=INK, lw=0.9)
    label(ax, 2.2, yb1 + 0.43, T(r"$E_x=20\,$V/m，$J_x=10\,$A/m$^2$", r"$E_x=20\,$V/m, $J_x=10\,$A/m$^2$"),
          size=8.6, color=INK)
    # ------------------------------------------------------------------ left top: potential profile
    y0, y1 = 4.15, 5.65                         # phi = 0 .. 0.4 V
    phis = b["phi"]
    yy = y0 + (y1 - y0) * phis / 0.4
    ax.plot([xs[0], xs[2] + 0.25], [y0, y0], color=MUTED, lw=0.7)
    ax.plot([xs[0], xs[0]], [y0, y1 + 0.15], color=MUTED, lw=0.7)
    ax.plot(xs, yy, color=POT, lw=1.8, zorder=4)
    for k, (x, y, p) in enumerate(zip(xs, yy, phis)):
        ax.add_patch(plt.Circle((x, y), 0.07, fc=POT if k == 1 else "white", ec=POT, lw=1.2, zorder=6))
        ax.plot([x, x], [y0, y], color=tint(POT, 0.4), lw=0.6, ls=":", zorder=3)
    label(ax, xs[0] + 0.2, yy[0] + 0.02, r"$\phi_1=0.4\,$V", color=POT, size=8.8, ha="left")
    label(ax, xs[1] + 0.12, yy[1] + 0.22, r"$\phi_2=0.2\,$V", color=POT, size=8.8, ha="left")
    label(ax, xs[2] + 0.05, yy[2] + 0.33, r"$\phi_3=0$", color=POT, size=8.8, ha="center")
    label(ax, xs[0] - 0.1, y1 + 0.12, r"$\phi$", color=POT, size=9.5, ha="right")
    label(ax, 3.05, 6.08, T("给定两端电势，只剩一个未知数", "two potentials given, one unknown left"),
          size=9.2, weight="bold", color=INK)
    # ------------------------------------------------------------------ right: matrix equation
    mx0 = 6.55
    label(ax, 8.45, 6.08, T(r"$\mathbf{K}\boldsymbol{\phi}=$ 节点注入电流", r"$\mathbf{K}\boldsymbol{\phi}=$ injected node currents"),
          size=9.2, weight="bold", color=INK)
    rows_y = [5.05, 4.45, 3.85]
    # row highlights
    bx0 = mx0 + 0.66
    rbox(ax, bx0, rows_y[1] - 0.25, 3.3 - 0.0, 0.5, fc=tint(ACCENT, 0.12), ec="none", r=0.06, z=1)
    for yy_ in (rows_y[0], rows_y[2]):
        rbox(ax, bx0, yy_ - 0.25, 3.3, 0.5, fc=tint(CURR, 0.12), ec="none", r=0.06, z=1)
    label(ax, bx0 - 0.08, rows_y[1], r"$5\,$mS$\,\times$", size=8.6, color=INK, ha="right")
    # K
    kx = [mx0 + 1.05, mx0 + 1.5, mx0 + 1.95]
    Kint = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    for i, yr in enumerate(rows_y):
        for j, xc in enumerate(kx):
            v = Kint[i, j]
            s = f"{v:d}" if v >= 0 else f"−{-v:d}"
            label(ax, xc, yr, s, size=9.4, color=INK if v != 0 else MUTED)
    _bracket(ax, kx[0] - 0.27, kx[2] + 0.27, rows_y[2] - 0.27, rows_y[0] + 0.27)
    # phi vector
    px = mx0 + 2.62
    for i, (yr, s) in enumerate(zip(rows_y, ("0.4", "0.2", "0"))):
        label(ax, px, yr, s, size=9.4, color=POT, weight="bold" if i == 1 else "normal")
    _bracket(ax, px - 0.27, px + 0.27, rows_y[2] - 0.27, rows_y[0] + 0.27, color=POT)
    label(ax, px, rows_y[2] - 0.5, "V", size=8.4, color=POT)
    label(ax, px + 0.47, rows_y[1], "=", size=10)
    # rhs vector
    rx = mx0 + 3.42
    for yr, s in zip(rows_y, ("+1", "0", "−1")):
        label(ax, rx, yr, s, size=9.4, color=CURR if s != "0" else INK, weight="bold")
    _bracket(ax, rx - 0.27, rx + 0.27, rows_y[2] - 0.27, rows_y[0] + 0.27, color=CURR)
    label(ax, rx, rows_y[2] - 0.5, "mA", size=8.4, color=CURR)
    # annotations below the matrix
    rbox(ax, mx0 + 0.22, 2.3, 0.28, 0.24, fc=tint(ACCENT, 0.12), ec="none", r=0.05)
    label(ax, mx0 + 0.62, 2.42, T("第 2 行：未知数 $\\phi_2$ 的方程（节点 2 的电流守恒）",
                                  "row 2: the equation for $\\phi_2$ (current balance at node 2)"),
          size=T(8.5, 8.1), ha="left")
    rbox(ax, mx0 + 0.22, 1.75, 0.28, 0.24, fc=tint(CURR, 0.12), ec="none", r=0.05)
    label(ax, mx0 + 0.62, 1.87, T("第 1、3 行：解出后计算，得到端子电流", "rows 1, 3: evaluated afterwards,"),
          size=T(8.5, 8.1), ha="left")
    label(ax, mx0 + 0.62, 1.5, T("（边界反力），不要求它们为零", "giving the terminal currents (reactions)"),
          size=T(8.5, 8.1), ha="left", color=INK)
    # ------------------------------------------------------------------ bottom: power three ways
    rbox(ax, 6.47, 0.25, 4.05, 0.82, fc=tint(ACCENT, 0.06), ec=tint(ACCENT, 0.35), lw=0.7, r=0.1)
    label(ax, 8.5, 0.83, T("同一个功率，三种算法", "one power, three ways"), size=8.6, color=ACCENT, weight="bold")
    label(ax, 8.5, 0.47, r"$VI=\boldsymbol{\phi}^{\rm T}\mathbf{K}\boldsymbol{\phi}=\int\sigma A\,(\mathrm{d}\phi/\mathrm{d}x)^2\,\mathrm{d}x=0.4\,$mW",
          size=9.0, color=INK)
    return fig


# ---------------------------------------------------------------------------
XN = np.array([0.0, 0.16, 0.38, 0.55, 0.78, 1.0])      # 1-D mesh nodes (non-uniform on purpose)


def hat(x, i, xn=XN):
    y = np.zeros_like(x)
    if i > 0:
        m = (x >= xn[i - 1]) & (x <= xn[i])
        y[m] = (x[m] - xn[i - 1]) / (xn[i] - xn[i - 1])
    if i < len(xn) - 1:
        m = (x >= xn[i]) & (x <= xn[i + 1])
        y[m] = (xn[i + 1] - x[m]) / (xn[i + 1] - xn[i])
    return y


def dhat(x, i, xn=XN):
    y = np.zeros_like(x)
    if i > 0:
        m = (x > xn[i - 1]) & (x < xn[i])
        y[m] = 1 / (xn[i] - xn[i - 1])
    if i < len(xn) - 1:
        m = (x > xn[i]) & (x < xn[i + 1])
        y[m] = -1 / (xn[i + 1] - xn[i])
    return y


@figure("hat_functions_1d")
def hat_functions_1d():
    fig = plt.figure(figsize=(FIG_W, 4.05))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.18], width_ratios=[1, 1],
                           left=0.065, right=0.985, top=0.93, bottom=0.075, hspace=0.45, wspace=0.26)
    x = np.linspace(0, 1, 2001)
    # (a) hats
    ax = fig.add_subplot(gs[0, 0])
    for i in range(len(XN)):
        ax.plot(x, hat(x, i), color=tint(MUTED, 0.45), lw=1.0, zorder=2)
    ax.fill_between(x, hat(x, 2), color=tint(POT, 0.18), lw=0, zorder=1)
    ax.plot(x, hat(x, 2), color=POT, lw=2.0, zorder=3)
    ax.plot(x, sum(hat(x, i) for i in range(len(XN))), color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=2)
    label(ax, 0.8, 1.13, r"$N_0+N_1+\cdots+N_5=1$", size=8.4, color=MUTED)
    halo(label(ax, XN[2], 1.13, r"$N_2$", size=9.5, color=POT, weight="bold"), 2.5)
    ax.plot(XN, np.zeros_like(XN), "o", ms=4.5, color=INK, zorder=5, clip_on=False)
    for i, xv in enumerate(XN):
        label(ax, xv, -0.2, f"$x_{i}$", size=8.3, color=MUTED)
    ax.set_xlim(-0.08, 1.03)
    ax.set_ylim(-0.32, 1.3)
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    clean_axes(ax)
    ax.spines["bottom"].set_visible(False)
    panel_title(ax, T("(a) 帽函数：自己的节点上为 1，别处为 0", "(a) hats: 1 at their own node, 0 elsewhere"), size=T(9.2, 8.8))
    # (b) interpolant
    ax = fig.add_subplot(gs[0, 1])
    u = lambda s: 0.45 + 0.42 * np.sin(5.2 * s + 0.4) * np.exp(-0.6 * s)   # noqa: E731 (illustrative)
    ui = u(XN)
    for i in range(len(XN)):
        ax.fill_between(x, ui[i] * hat(x, i), color=tint(POT, 0.10), lw=0, zorder=1)
        ax.plot(x, ui[i] * hat(x, i), color=tint(POT, 0.35), lw=0.8, zorder=2)
    ax.plot(x, u(x), color=MUTED, lw=1.2, ls=(0, (4, 2)), zorder=3, label=T("光滑函数 $u$", "smooth $u$"))
    ax.plot(XN, ui, color=POT, lw=2.0, zorder=4, label=r"$u_h=\Sigma_i\,u_iN_i$")
    ax.plot(XN, ui, "o", ms=4.5, color=POT, zorder=5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])
    clean_axes(ax)
    ax.legend(loc="upper right", fontsize=8.2, handlelength=1.8, borderaxespad=0.1)
    panel_title(ax, T("(b) 节点值 × 帽函数：连点成线", "(b) nodal values × hats: connect the dots"), size=T(9.2, 8.8))
    # (c) overlap of N_2 and N_3, derivatives
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.12, height_ratios=[1, 1.05])
    ax1 = fig.add_subplot(sub[0])
    ax2 = fig.add_subplot(sub[1], sharex=ax1)
    for a_ in (ax1, ax2):
        a_.axvspan(XN[2], XN[3], color=tint(ACCENT, 0.10), lw=0, zorder=0)
    ax1.plot(x, hat(x, 2), color=POT, lw=1.8)
    ax1.plot(x, hat(x, 3), color=ELEC, lw=1.8)
    ax1.plot(x, hat(x, 4), color=tint(MUTED, 0.6), lw=1.1, ls=(0, (3, 2)))
    halo(label(ax1, XN[2] - 0.05, 0.98, r"$N_2$", size=9, color=POT), 2)
    halo(label(ax1, XN[3] + 0.05, 0.98, r"$N_3$", size=9, color=ELEC), 2)
    halo(label(ax1, XN[4] + 0.06, 0.92, r"$N_4$", size=9, color=MUTED), 2)
    ax1.set_ylim(-0.05, 1.22)
    ax1.set_yticks([])
    ax1.tick_params(labelbottom=False, length=0)
    clean_axes(ax1)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    d2, d3 = dhat(x, 2), dhat(x, 3)
    ax2.axhline(0, color=MUTED, lw=0.6)
    ax2.plot(x, d2, color=POT, lw=1.6, drawstyle="default")
    ax2.plot(x, d3, color=ELEC, lw=1.6)
    ax2.set_ylim(-7.5, 7.5)
    ax2.set_yticks([])
    ax2.set_xticks(XN)
    ax2.set_xticklabels([f"$x_{i}$" for i in range(len(XN))], fontsize=8.2)
    ax2.set_xlim(-0.03, 1.03)
    clean_axes(ax2)
    ax2.spines["left"].set_visible(False)
    halo(label(ax2, 0.255, 6.0, r"$\mathrm{d}N_2/\mathrm{d}x$", size=8.8, color=POT), 2)
    halo(label(ax2, 0.68, 6.0, r"$\mathrm{d}N_3/\mathrm{d}x$", size=8.8, color=ELEC), 2)
    ax2.annotate(T("只在这里\n同时非零", "both nonzero\nonly here"), xy=(XN[2] + 0.03, -3.2), xytext=(0.07, -4.6),
                 fontsize=7.9, color=ACCENT, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8, mutation_scale=7, shrinkA=2, shrinkB=1))
    panel_title(ax1, T("(c) 导数只在重叠的单元上相乘", "(c) derivatives multiply only where hats overlap"), size=T(9.2, 8.8))
    # (d) matrix pattern
    ax = fig.add_subplot(gs[1, 1])
    n = len(XN)
    hs = np.diff(XN)
    for i in range(n):
        for j in range(n):
            if i == j:
                fc, s = tint(POT, 0.22), r"$+$"
            elif abs(i - j) == 1:
                fc, s = tint(ELEC, 0.18), r"$-$"
            else:
                fc, s = "white", "0"
            ax.add_patch(Rectangle((j, n - 1 - i), 0.94, 0.94, fc=fc, ec=tint(MUTED, 0.35), lw=0.6))
            label(ax, j + 0.47, n - 1 - i + 0.47, s, size=8.4, color=INK if s != "0" else tint(MUTED, 0.6))
    for (i, j), col in (((2, 3), ELEC), ((2, 2), POT), ((2, 4), MUTED)):
        ax.add_patch(Rectangle((j - 0.03, n - 1 - i - 0.03), 1.0, 1.0, fc="none", ec=col, lw=1.6, zorder=5))
    for k in range(n):
        label(ax, k + 0.47, n + 0.25, f"{k}", size=7.8, color=MUTED)
        label(ax, -0.28, n - 1 - k + 0.47, f"{k}", size=7.8, color=MUTED)
    ax.set_xlim(-0.5, 10.6)
    ax.set_ylim(-0.25, n + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")
    tx = n + 0.35
    label(ax, tx, 4.75, T(r"$K_{22}=\sigma A\left(\frac{1}{h_2}+\frac{1}{h_3}\right)$", r"$K_{22}=\sigma A\left(\frac{1}{h_2}+\frac{1}{h_3}\right)$"),
          size=8.8, color=POT, ha="left")
    label(ax, tx, 3.3, T(r"$K_{23}=-\sigma A/h_3$", r"$K_{23}=-\sigma A/h_3$"), size=8.8, color=ELEC, ha="left")
    label(ax, tx, 1.85, T(r"$K_{24}=0$：不重叠", r"$K_{24}=0$: no overlap"), size=8.8, color=MUTED, ha="left")
    label(ax, tx, 0.6, T("每行之和 = 0", "each row sums to 0"), size=8.6, color=INK, ha="left")
    panel_title(ax, T("(d) 刚度矩阵：三对角、稀疏", "(d) stiffness matrix: tridiagonal, sparse"), size=T(9.2, 8.8))
    return fig


# ---------------------------------------------------------------------------
def _patch_mesh():
    """Small unstructured patch: centre node, a ring of 6, a ring of 12 (deterministic jitter)."""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(4)
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = 2 * np.pi * k / 6 + 0.25 + rng.uniform(-0.12, 0.12)
        r = 1.0 + rng.uniform(-0.1, 0.1)
        pts.append((r * np.cos(a), r * np.sin(a)))
    for k in range(12):
        a = 2 * np.pi * k / 12 + 0.05 + rng.uniform(-0.08, 0.08)
        r = 2.0 + rng.uniform(-0.12, 0.12)
        pts.append((r * np.cos(a), 0.92 * r * np.sin(a)))
    P = np.array(pts)
    tri = Delaunay(P).simplices
    # orient counter-clockwise
    for t in tri:
        a, b, c = P[t]
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
            t[[1, 2]] = t[[2, 1]]
    return P, tri


def _tent(P, tri, i):
    """Vertices / faces of the graph of N_i over its star."""
    star = [t for t in tri if i in t]
    V, F = [], []
    for t in star:
        base = len(V)
        for v in t:
            V.append((P[v, 0], P[v, 1], 1.0 if v == i else 0.0))
        F.append([base, base + 1, base + 2])
    return np.array(V), np.array(F), star


@figure("nodal_basis")
def nodal_basis():
    from render3d import Scene, put, render, trim
    P, tri = _patch_mesh()
    i = 0
    nbrs = sorted({v for t in tri if i in t for v in t} - {i})
    j = nbrs[1]
    ring2 = [v for v in range(len(P)) if v not in nbrs and v != i]
    k = min(ring2, key=lambda v: np.hypot(*(P[v] - np.array([-1.6, -1.2]))))
    s = Scene(760, 560)
    s.camera(position=(3.4, -5.6, 3.9), target=(0.0, 0.1, 0.25), fov=29)
    H = 1.25                                     # vertical exaggeration of the tents
    cm = seq_cmap(POT)
    # base mesh (z = 0)
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            s.polyline([[*P[a], 0.0], [*P[b], 0.0]], color=tint(MUTED, 0.75))
    s.mesh(np.column_stack([P, np.zeros(len(P))]), tri, color=tint(MUTED, 0.06), opacity=1.0, unlit=True)
    # tent N_i (shaded by height) and tent N_j (translucent blue)
    for idx, col, op in ((i, None, 1.0), (j, ELEC, 0.42)):
        V, F, star = _tent(P, tri, idx)
        V = V.copy()
        V[:, 2] *= H
        if col is None:
            # subdivide each face so the colour follows the height
            n = 10
            verts, cols, faces = [], [], []
            for f in F:
                A, B, Cc = V[f]
                base = len(verts)
                ids = {}
                for a_ in range(n + 1):
                    for b_ in range(n + 1 - a_):
                        p = A + (B - A) * a_ / n + (Cc - A) * b_ / n
                        ids[(a_, b_)] = len(verts)
                        verts.append(p)
                        cols.append(cm(0.28 + 0.62 * p[2] / H))
                for a_ in range(n):
                    for b_ in range(n - a_):
                        faces.append([ids[(a_, b_)], ids[(a_ + 1, b_)], ids[(a_, b_ + 1)]])
                        if b_ < n - a_ - 1:
                            faces.append([ids[(a_ + 1, b_)], ids[(a_ + 1, b_ + 1)], ids[(a_, b_ + 1)]])
            s.mesh(np.array(verts) + [0, 0, 0.002], np.array(faces), vertex_colors=np.array(cols), opacity=1.0,
                   roughness=0.75, render_order=2)
            for f in F:
                for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                    s.tube([V[f[a_]], V[f[b_]]], radius=0.012, color=tint(POT, 0.9))
        else:
            s.mesh(V + [0, 0, 0.004], F, color=tint(ELEC, 0.6), opacity=op, render_order=3)
            for f in F:
                for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                    if V[f[a_], 2] > 0 or V[f[b_], 2] > 0:
                        s.tube([V[f[a_]], V[f[b_]]], radius=0.009, color=ELEC)
    s.sphere([*P[i], 0.0], 0.055, color=POT)
    s.sphere([*P[j], 0.0], 0.05, color=ELEC)
    s.sphere([*P[k], 0.0], 0.05, color=MUTED)
    s.label("peak_i", (P[i, 0], P[i, 1], H))
    s.label("peak_j", (P[j, 0], P[j, 1], H))
    s.label("node_k", (P[k, 0], P[k, 1], 0.0))
    s.label("node_i", (P[i, 0], P[i, 1], 0.0))
    img, px = trim(*render(s))
    # ---------------------------------------------------------------- compose: 3-D left, top view right
    h_px, w_px = img.shape[:2]
    W = FIG_W
    left_w = 3.45
    Hfig = left_w * h_px / w_px + 0.3
    fig = plt.figure(figsize=(W, Hfig))
    ax = fig.add_axes([0, 0, left_w / W, (Hfig - 0.3) / Hfig])
    ax.imshow(img, interpolation="none")
    ax.set_xlim(0, w_px); ax.set_ylim(h_px, 0); ax.axis("off")
    put(ax, px, "peak_i", r"$N_i$", dx=-60, dy=-10, color=POT, size=10.5)
    put(ax, px, "peak_j", r"$N_j$", dx=55, dy=-18, color=ELEC, size=10.5)
    put(ax, px, "node_k", T("节点 $k$", "node $k$"), dx=-10, dy=48, color=MUTED, size=9)
    fig.text(left_w / 2 / W, (Hfig - 0.24) / Hfig, T("节点基函数：一顶“帐篷”", "a nodal function: one ‘tent’"),
             ha="center", va="bottom", fontsize=9.6, fontweight="bold", color=POT)
    # top view
    ax2 = fig.add_axes([(left_w + 0.12) / W, 0.04, (W - left_w - 0.14) / W, (Hfig - 0.38) / Hfig])
    star_i = [t for t in tri if i in t]
    star_j = [t for t in tri if j in t]
    star_k = [t for t in tri if k in t]
    for t in tri:
        ax2.add_patch(Polygon(P[t], closed=True, fc="white", ec=tint(MUTED, 0.6), lw=0.6, zorder=1))
    for t in star_i:
        ax2.add_patch(Polygon(P[t], closed=True, fc=tint(POT, 0.2), ec=tint(MUTED, 0.6), lw=0.6, zorder=2))
    for t in star_j:
        both = any(set(t) == set(u) for u in star_i)
        ax2.add_patch(Polygon(P[t], closed=True, fc=tint(ELEC, 0.35) if both else tint(ELEC, 0.14),
                              ec=tint(MUTED, 0.6), lw=0.6, zorder=3, hatch="////" if both else None))
    for t in star_k:
        ax2.add_patch(Polygon(P[t], closed=True, fc="none", ec=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=4))
    # gradient arrows of N_i (constant per triangle, pointing towards node i)
    G_all = []
    for t in star_i:
        Gm, _ = C16.simplex_gradients(P[t])
        g = Gm[list(t).index(i)]
        c = P[t].mean(axis=0)
        d = g / np.linalg.norm(g) * 0.36
        halo(arrow(ax2, c - d / 2, c + d / 2, color=POT, lw=1.2, ms=7, z=6), 2.2)
        G_all.append(g)
    ax2.plot(*P[i], "o", ms=5, color=POT, zorder=7)
    ax2.plot(*P[j], "o", ms=4.5, color=ELEC, zorder=7)
    ax2.plot(*P[k], "o", ms=4.5, color=MUTED, zorder=7)
    halo(label(ax2, P[i, 0] + 0.02, P[i, 1] - 0.3, "$i$", size=9.5, color=POT), 2.5)
    halo(label(ax2, P[j, 0] + 0.22, P[j, 1] + 0.12, "$j$", size=9.5, color=ELEC), 2.5)
    halo(label(ax2, P[k, 0] - 0.2, P[k, 1] - 0.15, "$k$", size=9.5, color=MUTED), 2.5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    lo, hi = P.min(axis=0), P.max(axis=0)
    ax2.set_xlim(lo[0] - 0.1, hi[0] + 0.1)
    ax2.set_ylim(lo[1] - 1.05, hi[1] + 0.1)
    yb = lo[1] - 0.42
    label(ax2, lo[0] + 0.0, yb, T("■ 斜线：$i$、$j$ 支集重叠 → $K_{ij}\\neq0$", "■ hatched: supports of $i$, $j$ overlap → $K_{ij}\\neq0$"),
          size=T(8.2, 7.8), ha="left", color=ELEC)
    label(ax2, lo[0] + 0.0, yb - 0.45, T("虚线：$k$ 的支集，与 $i$ 不重叠 → $K_{ik}=0$", "dashed: support of $k$, no overlap → $K_{ik}=0$"),
          size=T(8.2, 7.8), ha="left", color=MUTED)
    fig.text((left_w + 0.12 + (W - left_w - 0.14) / 2) / W, (Hfig - 0.24) / Hfig,
             T("俯视：支集与梯度 $\\nabla N_i$", "top view: supports and $\\nabla N_i$"),
             ha="center", va="bottom", fontsize=9.6, fontweight="bold", color=INK)
    return fig


# ---------------------------------------------------------------------------
@figure("convergence")
def convergence():
    conv = C16.convergence_table()
    h = np.array([r["h"] for r in conv])
    n = np.array([r["n"] for r in conv])
    nodes = np.array([r["nodes"] for r in conv])
    l2 = np.array([r["L2_error"] for r in conv])
    h1 = np.array([r["H1_seminorm_error"] for r in conv])
    res = np.array([r["relative_algebraic_residual"] for r in conv])
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 2.85), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    ax.loglog(h, h1, "s-", color=ELEC, ms=5, lw=1.6, label=T(r"梯度（电场）误差 $|u-u_h|_{H^1}$", r"gradient (field) error $|u-u_h|_{H^1}$"))
    ax.loglog(h, l2, "o-", color=POT, ms=5, lw=1.6, label=T(r"电势误差 $\|u-u_h\|_{L^2}$", r"potential error $\|u-u_h\|_{L^2}$"))
    for k in range(1, len(h)):
        xm = np.sqrt(h[k - 1] * h[k])
        for e, col, dy in ((h1, ELEC, 1.75), (l2, POT, 2.1)):
            ym = np.sqrt(e[k - 1] * e[k]) * dy
            p = np.log2(e[k - 1] / e[k])
            halo(label(ax, xm, ym, f"{p:.2f}", size=8.0, color=col), 2.5)
    ax.invert_xaxis()
    ax.set_xlabel(T("网格间距 $h$（向右加密）", "mesh size $h$ (refining to the right)"))
    ax.set_ylabel(T("误差", "error"))
    ax.set_ylim(1.2e-4, 2.0)
    ax.set_xticks(h)
    ax.set_xticklabels([f"1/{v}" for v in n])
    ax.minorticks_off()
    clean_axes(ax)
    ax.legend(loc="lower left", fontsize=7.8, handlelength=1.6, borderaxespad=0.2)
    panel_title(ax, T("(a) 测得的误差与阶数", "(a) measured errors and orders"), size=9.4)
    ax = axs[1]
    ax.loglog(nodes, h1 / C16.H1_NORM_EXACT, "s-", color=ELEC, ms=5, lw=1.4, label=T("相对梯度误差", "relative gradient error"))
    ax.loglog(nodes, l2 / C16.L2_NORM_EXACT, "o-", color=POT, ms=5, lw=1.4, label=T("相对电势误差", "relative potential error"))
    ax.loglog(nodes, res, "D-", color=INK, ms=4.5, lw=1.2, label=T("相对代数残差", "relative algebraic residual"))
    ax.annotate("", xy=(nodes[-1] * 1.25, res[-1]), xytext=(nodes[-1] * 1.25, h1[-1] / C16.H1_NORM_EXACT),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=0.9, mutation_scale=7))
    label(ax, nodes[-1] * 1.42, 3e-8, T("约 11 个\n数量级", "≈ 11\ndecades"), size=8.0, color=MUTED, ha="left")
    ax.set_xlim(55, 1.6e4)
    ax.set_ylim(3e-16, 3)
    ax.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1])
    ax.set_xlabel(T("节点数", "number of nodes"))
    clean_axes(ax)
    ax.legend(loc="center left", fontsize=7.8, handlelength=1.6, borderaxespad=0.1, bbox_to_anchor=(0.0, 0.55))
    panel_title(ax, T("(b) 两个不同的“小”", "(b) two different kinds of ‘small’"), size=9.4)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.17, wspace=0.3)
    return fig
