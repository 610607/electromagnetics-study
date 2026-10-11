"""Figures for Ch 17 (vector finite elements and nonlinear solution).

All computed quantities come from checks/check_ch17.py (which uses numerics.py), so the figures,
the text and the assertions share one implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Polygon

from style import (ACCENT, AMBER, CURR, ELEC, FIG_W, INK, LIGHT, MAG, MATL, MUTED, POT, WARN, T, arrow,
                   canvas, clean_axes, figure, halo, label, panel_title, plt, rbox, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch17 as C17  # noqa: E402


# ---------------------------------------------------------------------------
def _lattice(V, n=7):
    """Interior barycentric lattice points of a triangle."""
    pts = []
    for i in range(1, n):
        for j in range(1, n - i):
            k = n - i - j
            if k >= 1:
                pts.append((i * V[0] + j * V[1] + k * V[2]) / n)
    return np.array(pts)


def _edge_marker(ax, a, b, color, lw, z=4, head=True, ms=9):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z, solid_capstyle="round")
    if head:
        m = 0.5 * (a + b)
        d = (b - a) / np.linalg.norm(b - a) * 0.02
        arrow(ax, m - d, m + d, color=color, lw=lw, ms=ms, z=z + 0.5)


@figure("edge_vs_nodal")
def edge_vs_nodal():
    V = C17.TRI
    fig = plt.figure(figsize=(FIG_W, 4.3))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.25], hspace=0.28,
                              left=0.01, right=0.99, top=0.905, bottom=0.075)
    top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.08)
    # ------------------------------------------------------------- (a) three Whitney functions
    pts = _lattice(V, 7)
    allW = [C17.whitney(V, i, j, pts) for i, j in C17.TRI_EDGES]
    smax = max(np.hypot(*W.T).max() for W in allW)
    names = [r"$\mathbf{W}_{01}$", r"$\mathbf{W}_{12}$", r"$\mathbf{W}_{20}$"]
    cen = V.mean(axis=0)
    for k, ((i, j), W) in enumerate(zip(C17.TRI_EDGES, allW)):
        ax = fig.add_subplot(top[k])
        ax.add_patch(Polygon(V, closed=True, fc=tint(ACCENT, 0.05), ec="none", zorder=0))
        for e, (a, b) in enumerate(C17.TRI_EDGES):
            own = e == k
            _edge_marker(ax, V[a], V[b], ACCENT if own else tint(MUTED, 0.55), 2.2 if own else 0.9,
                         head=own, ms=10)
            m = 0.5 * (V[a] + V[b])
            out = m - cen
            out = out / np.linalg.norm(out)
            s = "1" if own else "0"
            halo(label(ax, *(m + 0.13 * out), s, size=9.2 if own else 8.6,
                       color=ACCENT if own else MUTED, weight="bold" if own else "normal"), 2.5)
        sc = 0.17 / smax
        ax.quiver(pts[:, 0], pts[:, 1], W[:, 0] * sc, W[:, 1] * sc, color=ELEC, angles="xy",
                  scale_units="xy", scale=1, width=0.011, headwidth=3.6, headlength=3.8, headaxislength=3.4,
                  pivot="middle", zorder=5)
        for v in range(3):
            ax.plot(*V[v], "o", ms=3.6, color=INK, zorder=6)
            off = (V[v] - cen) / np.linalg.norm(V[v] - cen) * 0.11
            label(ax, *(V[v] + off), f"{v}", size=8.2, color=MUTED)
        label(ax, 0.86, 0.73, names[k], size=10.5, color=ELEC)
        ax.set_xlim(-0.16, 1.16)
        ax.set_ylim(-0.17, 0.98)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, 0.955, T("(a) 一个三角形上的三个棱基函数：沿自己的棱环量为 1，沿另两条棱为 0",
                           "(a) the three edge functions of one triangle: circulation 1 along their own edge, 0 along the others"),
             ha="center", va="center", fontsize=T(9.4, 8.9), fontweight="bold", color=INK)
    # ------------------------------------------------------------- (b)(c)(d) interface field
    bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.16, width_ratios=[1, 1, 0.86])
    P2 = C17.T2
    up, lo = [list(t) for t in C17.T2_TRIS]
    grid_pts = np.vstack([_lattice(P2[up], 6), _lattice(P2[lo], 6)])
    titles = [T("(b) 节点矢量插值", "(b) nodal vector interpolant"), T("(c) 棱单元表示", "(c) edge representation")]
    notes = [T("每个共享节点只有一个矢量", "one vector per shared node"), T("切向分量相同，法向分量跳变", "tangential equal, normal jumps")]
    fields = [C17.nodal_interface, C17.edge_interface]
    xl = 0.62                                     # the dashed sampling line of panel (d)
    for k in range(2):
        ax = fig.add_subplot(bot[k])
        ax.add_patch(Polygon(P2[up], closed=True, fc=tint("#C9B79C", 0.22), ec=tint(MATL, 0.7), lw=0.7, zorder=0))
        ax.add_patch(Polygon(P2[lo], closed=True, fc=tint("#C9B79C", 0.55), ec=tint(MATL, 0.7), lw=0.7, zorder=0))
        ax.plot([P2[0, 0], P2[1, 0]], [P2[0, 1], P2[1, 1]], color=INK, lw=1.3, zorder=2)
        F = fields[k](grid_pts)
        sc = 0.15
        ax.quiver(grid_pts[:, 0], grid_pts[:, 1], F[:, 0] * sc, F[:, 1] * sc, color=ELEC, angles="xy",
                  scale_units="xy", scale=1, width=0.012, headwidth=3.4, headlength=3.6, headaxislength=3.2,
                  pivot="middle", zorder=5)
        if k == 0:
            vals = {0: 0.5 * (C17.E_UP + C17.E_LO), 1: 0.5 * (C17.E_UP + C17.E_LO), 2: C17.E_UP, 3: C17.E_LO}
            for v, vec in vals.items():
                ax.plot(*P2[v], "o", ms=3.8, color=INK, zorder=6)
                arrow(ax, P2[v], P2[v] + 0.19 * vec, color=AMBER, lw=1.7, ms=8, z=7)
            col = AMBER
        else:
            for v in range(4):
                ax.plot(*P2[v], "o", ms=2.6, color=INK, zorder=6)
            for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (3, 1)):
                pa, pb = P2[a], P2[b]
                _edge_marker(ax, pa, pb, ACCENT, 1.5 if (a, b) != (0, 1) else 1.8, head=True, ms=8, z=3)
            col = ACCENT
        label(ax, 0.6, -0.86, notes[k], size=T(7.9, 7.4), color=col)
        label(ax, 0.02, 0.52, r"$\sigma_1$", size=9.5, color=INK, ha="right")
        label(ax, 0.30, -0.47, r"$\sigma_2=4\sigma_1$", size=8.8, color=INK, ha="right")
        ax.plot([xl, xl], [-0.40, 0.52], color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(-0.42, 1.3)
        ax.set_ylim(-0.98, 0.98)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titles[k], fontsize=T(9.2, 8.8), fontweight="bold", color=INK, pad=2)
    # (d) normal component along the dashed line
    ax = fig.add_subplot(bot[2])
    y = np.linspace(-0.40, 0.52, 400)
    line = np.column_stack([np.full_like(y, xl), y])
    En_nod = C17.nodal_interface(line)[:, 1]
    En_ex = C17.interface_field(line)[:, 1]
    ax.axvspan(-0.45, 0, color=tint("#C9B79C", 0.4), lw=0, zorder=0)
    ax.axvspan(0, 0.6, color=tint("#C9B79C", 0.15), lw=0, zorder=0)
    ax.plot(y, En_ex, color=ELEC, lw=2.0, label=T("精确 = 棱单元", "exact = edge"), zorder=3)
    ax.plot(y, En_nod, color=AMBER, lw=1.7, ls=(0, (4, 2)), label=T("节点插值", "nodal"), zorder=4)
    ax.axvline(0, color=INK, lw=0.9, zorder=2)
    ax.set_xlim(-0.42, 0.54)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.25, 0.625, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.625", "1"])
    ax.set_xticks([])
    ax.set_xlabel(T("沿虚线（下 → 上）", "along the dashed line (up)"), fontsize=8.4)
    clean_axes(ax)
    leg = ax.legend(loc="upper left", fontsize=T(7.8, 7.4), handlelength=1.8, borderaxespad=0.1, frameon=True,
                    framealpha=1.0, facecolor="white", edgecolor="none", borderpad=0.25)
    leg.set_zorder(10)
    label(ax, -0.21, 0.12, r"$\sigma_2$", size=8.8, color=INK)
    label(ax, 0.27, 0.12, r"$\sigma_1$", size=8.8, color=INK)
    ax.set_title(T("(d) 法向分量 $E_n$", "(d) normal component $E_n$"), fontsize=T(9.2, 8.8),
                 fontweight="bold", color=INK, pad=2)
    return fig


# ---------------------------------------------------------------------------
def _tet_icon(ax, x0, y0, s, mode, color):
    """Small projected tetrahedron; mode: 'node', 'edge', 'face', 'cell'."""
    P = np.array([[0.0, 0.0], [1.0, -0.22], [0.78, 0.42], [0.36, 0.92]]) * s + [x0, y0]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    hidden = {(0, 2)}
    if mode == "cell":
        for f in ((0, 1, 3), (1, 2, 3), (0, 1, 2)):
            ax.add_patch(Polygon(P[list(f)], closed=True, fc=tint(color, 0.28), ec="none", zorder=2))
    if mode == "face":
        ax.add_patch(Polygon(P[[0, 1, 3]], closed=True, fc=tint(color, 0.30), ec="none", zorder=2))
        ax.add_patch(Polygon(P[[1, 2, 3]], closed=True, fc=tint(color, 0.15), ec="none", zorder=2))
        c = P[[0, 1, 3]].mean(axis=0)
        arrow(ax, c, c + np.array([-0.10, -0.42]) * s, color=color, lw=1.5, ms=8, z=6)
        c2 = P[[1, 2, 3]].mean(axis=0)
        arrow(ax, c2, c2 + np.array([0.42, 0.05]) * s, color=color, lw=1.5, ms=8, z=6)
    for a, b in edges:
        ls = (0, (2, 2)) if (a, b) in hidden else "-"
        if mode == "edge":
            col, lw = color, 1.5
        else:
            col, lw = tint(MUTED, 0.7), 0.9
        ax.plot(*P[[a, b]].T, color=col, lw=lw, ls=ls, zorder=3, solid_capstyle="round")
        if mode == "edge" and (a, b) not in hidden:
            m = 0.5 * (P[a] + P[b])
            d = (P[b] - P[a]) / np.linalg.norm(P[b] - P[a]) * 0.02
            arrow(ax, m - d, m + d, color=color, lw=1.4, ms=8, z=5)
    for v in range(4):
        if mode == "node":
            ax.plot(*P[v], "o", ms=6.2, color=color, zorder=6, mec="white", mew=0.8)
        else:
            ax.plot(*P[v], "o", ms=2.6, color=tint(MUTED, 0.8), zorder=4)


@figure("exact_sequence")
def exact_sequence():
    fig, ax = canvas(FIG_W, 3.25, xlim=(-0.05, 12.05), ylim=(-0.12, 6.4))
    W, H, gap = 2.35, 4.4, 0.88
    y0 = 0.92
    xs = [k * (W + gap) for k in range(4)]
    spaces = [r"$H^1$", r"$H(\mathrm{curl})$", r"$H(\mathrm{div})$", r"$L^2$"]
    sub = [T("节点空间", "nodal"), T("棱空间", "edge"), T("面空间", "face"), T("单元空间", "cell")]
    modes = ["node", "edge", "face", "cell"]
    icol = [POT, ELEC, MAG, CURR]
    dof = [T("节点上的值", "nodal values"), T("棱上的线积分", "line integrals"),
           T("面上的通量", "face fluxes"), T("单元上的积分", "cell integrals")]
    cont = [T("值连续", "value continuous"), T("切向分量连续", "tangential cont."),
            T("法向分量连续", "normal cont."), T("不要求连续", "no continuity")]
    phys = [[(r"$\phi$", POT), (r"$\psi$", POT), (r"$A_z$", POT)],
            [(r"$\mathbf{A}$", POT), (r"$\mathbf{E}$", ELEC), (r"$\mathbf{H}$", MAG)],
            [(r"$\mathbf{B}$", MAG), (r"$\mathbf{J}$", CURR), (r"$\mathbf{D}$", ELEC)],
            [(r"$\rho$", ELEC), (r"$\nabla\!\cdot\!\mathbf{J}$", CURR)]]
    band = 0.62
    top = y0 + H
    for k, x in enumerate(xs):
        rbox(ax, x, y0, W, H, fc=tint(ACCENT, 0.05), ec=tint(ACCENT, 0.45), lw=0.8, r=0.12)
        rbox(ax, x, top - band, W, band, fc=tint(ACCENT, 0.85), ec="none", r=0.12, z=2.1)
        ax.add_patch(plt.Rectangle((x, top - band), W, band / 2, fc=tint(ACCENT, 0.85), ec="none", zorder=2.1))
        label(ax, x + W / 2, top - band / 2, spaces[k] + "  " + sub[k], color="white", size=T(9.6, 9.2), weight="bold")
        _tet_icon(ax, x + 0.66, top - 1.8, 1.0, modes[k], icol[k])
        ly = y0 + 1.86
        label(ax, x + W / 2, ly, dof[k], size=T(8.6, 8.1), color=INK)
        label(ax, x + W / 2, ly - 0.5, cont[k], size=T(8.6, 8.1), color=MUTED)
        ax.plot([x + 0.25, x + W - 0.25], [ly - 0.86, ly - 0.86], color=tint(ACCENT, 0.3), lw=0.6)
        items = phys[k]
        span = 0.62
        cx = x + W / 2 - span * (len(items) - 1) / 2
        for i, (s_, c) in enumerate(items):
            label(ax, cx + i * span, ly - 1.28, s_, size=10.5 if k < 3 else 9.6, color=c)
    # arrows with operators
    ops = [(r"$\nabla$", r"$\mathbf{G}$"), (r"$\nabla\times$", r"$\mathbf{C}$"), (r"$\nabla\cdot$", r"$\mathbf{D}$")]
    ya = top - 1.45
    for k in range(3):
        xa, xb = xs[k] + W + 0.06, xs[k + 1] - 0.06
        arrow(ax, (xa, ya), (xb, ya), color=INK, lw=1.4, ms=11)
        label(ax, (xa + xb) / 2, ya + 0.4, ops[k][0], size=11, color=INK)
        label(ax, (xa + xb) / 2, ya - 0.4, ops[k][1], size=10, color=MUTED)
    # the two identities: one bracket above the cards, one below (they do not cross)
    for (ka, kb, yb, s1, s2, up_) in ((0, 2, top + 0.55, r"$\nabla\times\nabla=0$", r"$\mathbf{C}\mathbf{G}=0$", True),
                                       (1, 3, y0 - 0.55, r"$\nabla\cdot\nabla\times=0$", r"$\mathbf{D}\mathbf{C}=0$", False)):
        xa, xb = xs[ka] + W / 2, xs[kb] + W / 2
        yc = top + 0.04 if up_ else y0 - 0.04
        ax.plot([xa, xa, xb, xb], [yc, yb, yb, yc], color=tint(ACCENT, 0.6), lw=0.9, zorder=1)
        arrow(ax, (xb, yb), (xb, yc), color=tint(ACCENT, 0.6), lw=0.9, ms=7, z=1)
        label(ax, 0.5 * (xa + xb), yb, s1 + T("，即 ", ", i.e. ") + s2, size=T(9.0, 8.7), color=ACCENT,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    return fig


# ---------------------------------------------------------------------------
@figure("picard_newton")
def picard_newton():
    Fc = C17.pm_parts()[2]
    Ay = C17.PM["Ay"]
    _, hN = C17.newton(iters=6)
    _, hP = C17.picard(1.0, iters=130)
    _, h05 = C17.picard(0.05, iters=130)
    _, h02 = C17.picard(0.02, iters=130)
    Bsol = C17.newton_solve() / Ay
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 2.75), gridspec_kw=dict(width_ratios=[1, 1.15]))
    cN, cP, c05, c02 = ACCENT, WARN, AMBER, MUTED
    # (a) iterates of the yoke flux density
    ax = axs[0]
    kmax = 14
    ax.axhline(Bsol, color=INK, lw=0.8, ls=(0, (1, 1.6)), zorder=1)
    ax.plot(range(kmax + 1), [p / Ay for p, _, _ in hP[:kmax + 1]], "s-", color=cP, ms=3.4, lw=1.0, zorder=3)
    ax.plot(range(kmax + 1), [p / Ay for p, _, _ in h05[:kmax + 1]], "^-", color=c05, ms=3.6, lw=1.1, zorder=3)
    ax.plot(range(len(hN)), [p / Ay for p, _ in hN], "o-", color=cN, ms=4.4, lw=1.8, zorder=5)
    halo(label(ax, kmax, 1.93, T(f"解 $B_y={Bsol:.3f}$ T", f"solution $B_y={Bsol:.3f}$ T"), size=8.2, color=INK,
               ha="right"), 2.5)
    ax.annotate("", xy=(kmax - 0.6, Bsol + 0.02), xytext=(kmax - 0.6, 1.86),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6))
    halo(label(ax, 0.95, 0.72, T("牛顿", "Newton"), size=8.6, color=cN, ha="left", weight="bold"), 2.5)
    halo(label(ax, 8.6, 0.94, T(r"Picard，$\omega=1$：两点循环", r"Picard, $\omega=1$: 2-cycle"), size=8.2, color=cP), 2.5)
    halo(label(ax, 8.8, 0.36, T(r"Picard，$\omega=0.05$", r"Picard, $\omega=0.05$"), size=8.2, color=c05, ha="left"), 2.5)
    ax.set_xlim(-0.4, kmax + 0.4)
    ax.set_ylim(0, 2.05)
    ax.set_xticks(range(0, kmax + 1, 2))
    ax.set_xlabel(T("迭代步 $k$", "iteration $k$"))
    ax.set_ylabel(T("磁轭 $B_y$（T）", "yoke $B_y$ (T)"))
    clean_axes(ax)
    panel_title(ax, T("(a) 迭代值", "(a) iterates"), size=9.4)
    # (b) residual histories
    ax = axs[1]
    rN = [max(abs(r) / Fc, 1e-17) for _, r in hN]
    ax.semilogy([abs(r) / Fc for _, r, _ in hP], color=cP, lw=0.9, label=T(r"Picard，$\omega=1$", r"Picard, $\omega=1$"))
    ax.semilogy([abs(r) / Fc for _, r, _ in h05], color=c05, lw=1.5, label=r"$\omega=0.05$")
    ax.semilogy([abs(r) / Fc for _, r, _ in h02], color=c02, lw=1.5, label=r"$\omega=0.02$")
    ax.semilogy([u for _, _, u in h02], color=c02, lw=1.1, ls=(0, (3, 2)),
                label=T(r"$\omega=0.02$ 的相对更新", r"$\omega=0.02$: relative update"))
    ax.semilogy(range(len(rN)), rN, "o-", color=cN, ms=4, lw=1.6, label=T("牛顿", "Newton"), zorder=5)
    k = 78
    ax.plot([k, k], [h02[k][2], abs(h02[k][1]) / Fc], color=c02, lw=0.8, zorder=2)
    ax.plot([k], [abs(h02[k][1]) / Fc], "o", color=c02, ms=3.5)
    ax.plot([k], [h02[k][2]], "o", color=c02, ms=3.5, mfc="white")
    ax.set_xlim(-2, 125)
    ax.set_ylim(1e-16, 8)
    ax.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1])
    ax.set_xlabel(T("迭代步 $k$", "iteration $k$"))
    ax.set_ylabel(r"$|f|/\mathcal{F}_c$")
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.13, 0.02), fontsize=T(7.6, 7.3), handlelength=1.8,
              borderaxespad=0.1)
    panel_title(ax, T("(b) 残差（实线）与更新量（虚线）", "(b) residual (solid) and update (dashed)"), size=9.4)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.16, wspace=0.27)
    return fig
