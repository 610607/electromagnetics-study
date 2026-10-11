"""Figures for Chapter 6: interfaces — what is continuous, what jumps."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Ellipse, Polygon, Rectangle

from style import (ACCENT, AMBER, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, IRON_T, LIGHT, MAG, MAG_T, MATL, MATL_T,
                   MUTED, POT, T, arrow, canvas, clean_axes, dot, figure, halo, label, out_of_page, panel_title, pill,
                   plt, rbox, tint)

REG1 = tint("#C9B79C", 0.45)   # material 1 (below the interface)
REG2 = tint("#9FB4C7", 0.30)   # material 2 (above the interface)


def _regions(ax, x0, x1, y0, y1, yi=0.0, labels=None, size=9.0):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, yi - y0, fc=REG1, ec="none", zorder=0))
    ax.add_patch(Rectangle((x0, yi), x1 - x0, y1 - yi, fc=REG2, ec="none", zorder=0))
    ax.plot([x0, x1], [yi, yi], color=INK, lw=1.1, zorder=1)
    if labels:
        label(ax, x0 + 0.12, y0 + 0.2, labels[0], size=size, ha="left", color=INK)
        label(ax, x0 + 0.12, y1 - 0.2, labels[1], size=size, ha="left", color=INK)


# ---------------------------------------------------------------------------
def _ray(ax, P, v1, v2, color, scale, lw=2.0, comp=True, ls_comp=(0, (2.5, 2))):
    """Incoming vector v1 (in region 1, below) ending at P and outgoing v2 starting at P.
    Vectors are (t, n) components; dashed projections show the two parts."""
    P = np.asarray(P, float)
    a1 = P - scale * np.asarray(v1)
    halo(arrow(ax, tuple(a1), tuple(P), color=color, lw=lw, ms=12, z=6), 3)
    b2 = P + scale * np.asarray(v2)
    halo(arrow(ax, tuple(P), tuple(b2), color=color, lw=lw, ms=12, z=6), 3)
    if comp:
        # region 1: tangential projection drawn along the bottom of the triangle, normal along its side
        ax.plot([a1[0], P[0]], [a1[1], a1[1]], color=color, lw=0.9, ls=ls_comp, zorder=5)
        ax.plot([P[0], P[0]], [a1[1], P[1]], color=color, lw=0.9, ls=ls_comp, zorder=5)
        ax.plot([P[0], b2[0]], [b2[1], b2[1]], color=color, lw=0.9, ls=ls_comp, zorder=5)
        ax.plot([P[0], P[0]], [P[1], b2[1]], color=color, lw=0.9, ls=ls_comp, zorder=5)
    return a1, b2


def _lines(ax, v1, v2, color, xl, yl, n=9):
    """Parallel refracted lines: direction v1 below y=0 and v2 above, clipped to the panel."""
    d1, d2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    for x0 in np.linspace(xl[0] - 1.5, xl[1] + 0.3, n):
        s1 = -yl[0] / d1[1]
        s2 = yl[1] / d2[1]
        ax.plot([x0 - s1 * d1[0], x0], [yl[0], 0.0], color=color, lw=0.9, zorder=1, clip_on=True)
        ax.plot([x0, x0 + s2 * d2[0]], [0.0, yl[1]], color=color, lw=0.9, zorder=1, clip_on=True)


@figure("interface")
def interface():
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.35))
    xl, yl = (-1.75, 1.75), (-1.55, 1.75)
    # ---------------- left: conductivity interface sigma2 = sigma1/4
    ax = axs[0]
    _regions(ax, xl[0], xl[1], yl[0], yl[1], 0.0)
    th1 = np.deg2rad(60.0)
    E1 = np.array([np.sin(th1), np.cos(th1)])       # (t, n)
    E2 = np.array([E1[0], 4 * E1[1]])
    J1, J2 = E1.copy(), E2 / 4
    # background current lines (green, light) — same direction as E in each region
    _lines(ax, E1, E2, tint(CURR, 0.35), xl, yl)
    sc = 0.62
    _ray(ax, (-0.85, 0.0), E1, E2, ELEC, sc)
    _ray(ax, (0.75, 0.0), J1, J2, CURR, sc)
    lab = dict(size=8.4, z=8)
    halo(label(ax, -0.12, 1.42, r"$\mathbf{E}_2$", color=ELEC, size=10), 3)
    halo(label(ax, -1.5, -0.62, r"$\mathbf{E}_1$", color=ELEC, size=10), 3)
    halo(label(ax, 1.25, 0.55, r"$\mathbf{J}_2$", color=CURR, size=10), 3)
    halo(label(ax, 0.1, -0.62, r"$\mathbf{J}_1$", color=CURR, size=10), 3)
    halo(label(ax, -1.68, 0.62, T(r"$E_{2n}=4E_{1n}$", r"$E_{2n}=4E_{1n}$"), color=ELEC, ha="left", **lab), 3)
    halo(label(ax, -1.62, -0.92, T(r"$E_t$ 相同", r"same $E_t$"), color=ELEC, ha="left", **lab), 3)
    halo(label(ax, 0.98, -0.47, T(r"$J_n$ 相同", r"same $J_n$"), color=CURR, ha="left", **lab), 3)
    halo(label(ax, 0.62, 1.02, T(r"$J_{2t}=J_{1t}/4$", r"$J_{2t}=J_{1t}/4$"), color=CURR, ha="center", **lab), 3)
    label(ax, xl[0] + 0.08, yl[1] - 0.17, r"$\sigma_2=\sigma_1/4$", size=9, ha="left")
    label(ax, xl[0] + 0.08, yl[0] + 0.17, r"$\sigma_1$", size=9, ha="left")
    panel_title(ax, T("电导率界面：E 与 J", "Conductivity interface: E and J"), color=INK)
    label(ax, 0, yl[0] - 0.22, T(r"连续：$E_t$、$J_n$　　跳变：$E_n$、$J_t$", r"continuous: $E_t$, $J_n$    jump: $E_n$, $J_t$"), size=8.8)
    # ---------------- right: permeability interface mu2 = 4 mu1
    ax = axs[1]
    _regions(ax, xl[0], xl[1], yl[0], yl[1], 0.0)
    th1 = np.deg2rad(30.0)
    B1 = np.array([np.sin(th1), np.cos(th1)])
    B2 = np.array([4 * B1[0], B1[1]])
    H1, H2 = B1.copy(), B2 / 4
    _lines(ax, B1, B2, tint(MAG, 0.35), xl, yl)
    _ray(ax, (-1.05, 0.0), B1, B2, MAG, sc)
    _ray(ax, (0.85, 0.0), H1, H2, tint(MAG, 0.6), sc, lw=1.7)
    halo(label(ax, -0.15, 0.78, r"$\mathbf{B}_2$", color=MAG, size=10), 3)
    halo(label(ax, -1.62, -0.38, r"$\mathbf{B}_1$", color=MAG, size=10), 3)
    halo(label(ax, 1.38, 0.38, r"$\mathbf{H}_2$", color=tint(MAG, 0.75), size=10), 3)
    halo(label(ax, 1.05, 0.72, T(r"$H_{2n}=H_{1n}/4$", r"$H_{2n}=H_{1n}/4$"), color=tint(MAG, 0.75), ha="center", **lab), 3)
    halo(label(ax, 0.36, -0.75, r"$\mathbf{H}_1$", color=tint(MAG, 0.75), size=10), 3)
    halo(label(ax, -1.25, 0.25, T(r"$B_n$ 相同", r"same $B_n$"), color=MAG, ha="left", **lab), 3)
    halo(label(ax, 0.62, -1.05, T(r"$H_t$ 相同", r"same $H_t$"), color=tint(MAG, 0.75), ha="left", **lab), 3)
    halo(label(ax, -0.95, 1.05, T(r"$B_{2t}=4B_{1t}$", r"$B_{2t}=4B_{1t}$"), color=MAG, ha="left", **lab), 3)
    label(ax, xl[0] + 0.08, yl[1] - 0.17, r"$\mu_2=4\mu_1$", size=9, ha="left")
    label(ax, xl[0] + 0.08, yl[0] + 0.17, r"$\mu_1$", size=9, ha="left")
    panel_title(ax, T("磁导率界面：B 与 H", "Permeability interface: B and H"), color=INK)
    label(ax, 0, yl[0] - 0.22, T(r"连续：$H_t$、$B_n$　　跳变：$H_n$、$B_t$", r"continuous: $H_t$, $B_n$    jump: $H_n$, $B_t$"), size=8.8)
    for ax in axs:
        ax.set_xlim(*xl)
        ax.set_ylim(yl[0] - 0.4, yl[1])
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
@figure("conduction_profiles")
def conduction_profiles():
    d1, d2 = 8.0, 2.0                       # mm
    s1, s2 = 0.2, 0.02                      # S/m
    I, A = 1e-3, 1e-4
    J = I / A
    E1, E2 = J / s1, J / s2
    x = np.linspace(0, d1 + d2, 1001)
    left = x <= d1
    Jx = np.full_like(x, J)
    Ex = np.where(left, E1, E2)
    V2 = E2 * d2 * 1e-3
    phi = np.where(left, V2 + E1 * (d1 - x) * 1e-3, E2 * (d1 + d2 - x) * 1e-3)
    p = np.where(left, s1 * E1**2, s2 * E2**2)

    fig = plt.figure(figsize=(FIG_W, 3.75))
    L, W = 0.13, 0.62
    # schematic
    sx = fig.add_axes([L, 0.84, W, 0.14])
    sx.set_xlim(0, 10)
    sx.set_ylim(0, 1)
    sx.axis("off")
    sx.add_patch(Rectangle((0, 0.15), d1, 0.7, fc=REG1, ec=MATL, lw=0.8))
    sx.add_patch(Rectangle((d1, 0.15), d2, 0.7, fc=REG2, ec=MATL, lw=0.8))
    sx.add_patch(Rectangle((-0.22, 0.05), 0.22, 0.9, fc="#9AA3AB", ec="none", clip_on=False))
    sx.add_patch(Rectangle((10.0, 0.05), 0.22, 0.9, fc="#9AA3AB", ec="none", clip_on=False))
    sx.text(d1 / 2, 0.5, T(r"层 1：$8\,$mm，$\sigma_1=0.2\,$S/m", r"layer 1: $8\,$mm, $\sigma_1=0.2\,$S/m"), ha="center", va="center", fontsize=8.8)
    sx.text(d1 + d2 / 2, 0.5, T(r"层 2" + "\n" + r"$0.02\,$S/m", r"layer 2" + "\n" + r"$0.02\,$S/m"), ha="center", va="center", fontsize=8.0, linespacing=1.15)
    for yy in (0.3, 0.5, 0.7):
        sx.text(d1, yy, "+", color=MAG, fontsize=8.5, ha="center", va="center", fontweight="bold")
    sx.annotate("", xy=(1.6, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="-|>", color=CURR, lw=1.6, mutation_scale=11))
    sx.text(-0.4, 0.5, r"$I=1\,$mA", ha="right", va="center", fontsize=8.8, color=CURR)
    # stacked profiles
    axes = []
    h, gap, top = 0.152, 0.038, 0.80
    specs = [(Jx, CURR, T(r"$J$ (A/m$^2$)", r"$J$ (A/m$^2$)"), (0, 13), [0, 10]),
             (Ex, ELEC, T(r"$E$ (V/m)", r"$E$ (V/m)"), (0, 600), [50, 500]),
             (phi, POT, T(r"$\phi$ (V)", r"$\phi$ (V)"), (0, 1.6), [0, 1.0, 1.4]),
             (p, AMBER, T(r"$p=\mathbf{E}\cdot\mathbf{J}$ (W/m$^3$)", r"$p=\mathbf{E}\cdot\mathbf{J}$ (W/m$^3$)"), (0, 6000), [500, 5000])]
    for k, (y, col, yl, lim, ticks) in enumerate(specs):
        bx = fig.add_axes([L, top - (k + 1) * h - k * gap, W, h], sharex=axes[0] if axes else None)
        bx.axvspan(0, d1, color=REG1, lw=0, zorder=0, alpha=0.55)
        bx.axvspan(d1, d1 + d2, color=REG2, lw=0, zorder=0, alpha=0.7)
        bx.axvline(d1, color=INK, lw=0.8, zorder=1)
        bx.plot(x, y, color=col, lw=2.0, zorder=3)
        bx.set_ylim(*lim)
        bx.set_yticks(ticks)
        bx.set_ylabel(yl, fontsize=8.6, rotation=0, ha="right", va="center", labelpad=6)
        clean_axes(bx)
        if k < 3:
            plt.setp(bx.get_xticklabels(), visible=False)
        axes.append(bx)
    axes[-1].set_xlim(0, 10)
    axes[-1].set_xlabel(T("位置 $x$ (mm)", "position $x$ (mm)"))
    # notes on the right
    R = L + W + 0.02
    notes = [T("连续：两层相同", "continuous:\nsame in both"),
             T("跳 10 倍：$E=J/\\sigma$", "jumps 10×:\n$E=J/\\sigma$"),
             T("连续，斜率变；\n压降 0.4 V + 1.0 V", "continuous, new\nslope; drops\n0.4 V + 1.0 V"),
             T("低 σ 层发热\n密度大 10 倍", "low-σ layer:\n10× power\ndensity")]
    for k, s in enumerate(notes):
        yc = top - (k + 0.5) * h - k * gap
        fig.text(R, yc, s, fontsize=8.4, va="center", ha="left", color=specs[k][1] if k != 3 else INK, linespacing=1.25)
    axes[2].annotate("", xy=(d1, 1.0), xytext=(d1 - 2.2, 1.28), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    axes[2].text(d1 - 2.3, 1.3, "1.0 V", fontsize=8.0, color=MUTED, ha="right", va="center")
    return fig
