"""Figures for Ch 19 (magnetic observables of MRI).

gradient_maxwell  x gradient, z gradient, and the target b = (0, 0, Gx) that Maxwell's equations forbid.
concomitant       the concomitant field (b_x^2 + b_y^2)/(2 B0) of linear x and z gradients (computed).

Field models and teaching numbers come from checks/check_ch19.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from style import (ACCENT, ELEC, FIG_W, INK, MAG, MUTED, T, arrow, clean_axes, figure, flux_lines, halo,
                   panel_title, plt, seq_cmap, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch19 as MR  # noqa: E402

BZ_CMAP = LinearSegmentedColormap.from_list("bz", [tint("#5C6BC0", 0.55), "#FFFFFF", tint(MAG, 0.55)])


def _frame(ax, lim=1.0):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$x/L$", labelpad=1); ax.set_ylabel(r"$z/L$", labelpad=0)
    ax.tick_params(labelsize=7.8, length=2.5)
    for s in ax.spines.values():
        s.set_color(tint(MUTED, 0.6))


@figure("gradient_maxwell")
def gradient_maxwell():
    fig, axs = plt.subplots(1, 3, figsize=(FIG_W, 2.45))
    n = 241
    g = np.linspace(-1.0, 1.0, n)
    X, Z = np.meshgrid(g, g)
    # (a) x gradient: b = (z, 0, x); flux function A = (z^2 - x^2)/2 with b_x = dA/dz, b_z = -dA/dx
    a = axs[0]
    a.imshow(X, extent=(-1, 1, -1, 1), origin="lower", cmap=BZ_CMAP, vmin=-1, vmax=1, zorder=0)
    A = 0.5 * (Z**2 - X**2)
    lev = np.array([-0.42, -0.24, -0.1, -0.02, 0.02, 0.1, 0.24, 0.42])
    flux_lines(a, X, Z, A, lev, Z, X, color=MAG, lw=1.0, ms=7, arrow_pos=0.5)
    _frame(a)
    panel_title(a, T(r"(a) $x$ 梯度：$\mathbf{b}=G(z,\,0,\,x)$", r"(a) $x$ gradient: $\mathbf{b}=G(z,0,x)$"), size=8.6)
    # (b) z gradient, meridian plane: Psi = x^2 z / 2, b_x = -x/2, b_z = z
    a = axs[1]
    a.imshow(Z, extent=(-1, 1, -1, 1), origin="lower", cmap=BZ_CMAP, vmin=-1, vmax=1, zorder=0)
    Psi = 0.5 * X**2 * Z
    lev = np.array([-0.3, -0.15, -0.06, -0.015, 0.015, 0.06, 0.15, 0.3])
    flux_lines(a, X, Z, Psi, lev, -X / 2, Z, color=MAG, lw=1.0, ms=7, arrow_pos=0.5)
    a.plot([0, 0], [-1, 1], color=MAG, lw=1.0, zorder=3)
    arrow(a, (0, 0.55), (0, 0.62), color=MAG, lw=1.0, ms=7)
    arrow(a, (0, -0.55), (0, -0.62), color=MAG, lw=1.0, ms=7)   # b_z = G z < 0 below the centre
    _frame(a)
    panel_title(a, T(r"(b) $z$ 梯度：$\mathbf{b}=G(-\frac{x}{2},-\frac{y}{2},z)$",
                     r"(b) $z$ gradient: $\mathbf{b}=G(-\frac{x}{2},-\frac{y}{2},z)$"), size=8.6)
    # (c) forbidden target b = (0, 0, x)
    a = axs[2]
    a.imshow(X, extent=(-1, 1, -1, 1), origin="lower", cmap=BZ_CMAP, vmin=-1, vmax=1, zorder=0)
    q = np.linspace(-0.8, 0.8, 7)
    QX, QZ = np.meshgrid(q, q)
    a.quiver(QX, QZ, 0 * QX, 0.22 * QX, color=MAG, angles="xy", scale_units="xy", scale=1, width=0.012,
             headwidth=3.4, headlength=3.6, headaxislength=3.2, zorder=4, pivot="middle")
    # circulation loop
    x0, x1, z0, z1 = -0.48, 0.48, -0.5, 0.5
    a.plot([x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0], color=INK, lw=1.1, ls=(0, (3, 2)), zorder=5)
    for p, qv in (((0.0, z0), (0.08, z0)), ((x1, 0.0), (x1, 0.08)), ((0.0, z1), (-0.08, z1)), ((x0, 0.0), (x0, -0.08))):
        arrow(a, p, qv, color=INK, lw=1.1, ms=8, z=6)
    t = a.text(0, 0.0, T("$\\oint\\mathbf{b}\\cdot\\mathrm{d}\\boldsymbol{\\ell}\\neq0$\n需要电流", "$\\oint\\mathbf{b}\\cdot\\mathrm{d}\\boldsymbol{\\ell}\\neq0$\nneeds current"),
               fontsize=8.2, color=INK, ha="center", va="center", zorder=7)
    halo(t, 3.0)
    _frame(a)
    panel_title(a, T(r"(c) 只要 $b_z=Gx$：不可实现", r"(c) only $b_z=Gx$: impossible"), size=8.6)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.17, top=0.88, wspace=0.32)
    return fig


@figure("concomitant")
def concomitant():
    G, B0 = MR.G_T, MR.B0_LOW
    fig = plt.figure(figsize=(FIG_W, 2.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.055, 0.42, 1.2], wspace=0.3, left=0.07, right=0.99,
                          bottom=0.18, top=0.86)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1]); ca = fig.add_subplot(gs[2]); a3 = fig.add_subplot(gs[4])
    L = 0.2
    g = np.linspace(-L, L, 201)
    X, Z = np.meshgrid(g, g)
    cmap = seq_cmap(MAG)
    vmax = 20.0
    bcx = MR.bc_xgrad(Z, B0) * 1e6                  # x gradient, depends on z only (uT)
    bcz = MR.bc_zgrad(np.abs(X), B0) * 1e6          # z gradient, y = 0 plane: depends on x only
    for a, bc, lines_vertical, title in (
            (a1, bcx, True, T(r"(a) $x$ 梯度的伴随场", r"(a) $x$ gradient")),
            (a2, bcz, False, T(r"(b) $z$ 梯度的伴随场", r"(b) $z$ gradient"))):
        im = a.imshow(bc, extent=(-L, L, -L, L), origin="lower", cmap=cmap, vmin=0, vmax=vmax, zorder=0)
        a.contour(X, Z, bc, levels=[1, 5, 10], colors=[tint(MAG, 0.9)], linewidths=0.8, zorder=2)
        for v in (-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15):
            if lines_vertical:
                a.plot([v, v], [-L, L], color=ELEC, lw=0.7, ls=(0, (3, 2)), zorder=3)
            else:
                a.plot([-L, L], [v, v], color=ELEC, lw=0.7, ls=(0, (3, 2)), zorder=3)
        a.add_patch(plt.Circle((0, 0), 0.1, fc="none", ec=INK, lw=1.0, zorder=4))
        a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect("equal")
        a.set_xticks([-0.1, 0, 0.1]); a.set_yticks([-0.1, 0, 0.1])
        a.tick_params(labelsize=7.6, length=2.5)
        a.set_xlabel(r"$x$ (m)", labelpad=1)
        panel_title(a, title, size=8.6)
    a1.set_ylabel(r"$z$ (m)", labelpad=0)
    a2.set_yticklabels([])
    a1.plot(0, 0.1, "o", ms=4, mfc="white", mec=INK, mew=1.1, zorder=6)
    t = a1.text(0.012, 0.118, r"5 $\mu$T", fontsize=7.8, color=INK, ha="left", va="bottom", zorder=7); halo(t, 2.5)
    a2.plot(0.1, 0, "o", ms=4, mfc="white", mec=INK, mew=1.1, zorder=6)
    t = a2.text(0.153, 0.03, r"1.25 $\mu$T", fontsize=7.8, color=INK, ha="center", va="bottom", zorder=7); halo(t, 2.5)
    cb = fig.colorbar(im, cax=ca)
    cb.set_label(T(r"$b_c$（$\mu$T）", r"$b_c$ ($\mu$T)"), fontsize=8.2, labelpad=2)
    cb.ax.tick_params(labelsize=7.4, length=2)
    cb.outline.set_linewidth(0.6)
    # (c) dependence on B0
    B = np.logspace(np.log10(0.05), np.log10(3.0), 100)
    a3.loglog(B, MR.bc_xgrad(0.1, B) * 1e6, color=MAG, lw=1.9, label=T(r"$x$ 梯度，$z=10$ cm", r"$x$ gradient, $z=10$ cm"))
    a3.loglog(B, MR.bc_zgrad(0.1, B) * 1e6, color=MAG, lw=1.5, ls=(0, (4, 2)),
              label=T(r"$z$ 梯度，$r=10$ cm", r"$z$ gradient, $r=10$ cm"))
    for b0 in (0.1, 1.5):
        v = MR.bc_xgrad(0.1, b0) * 1e6
        a3.plot(b0, v, "o", ms=4.5, mfc="white", mec=MAG, mew=1.3, zorder=5)
    a3.text(0.115, 6.5, T("5 µT\n= 50 ppm", "5 µT\n= 50 ppm"), fontsize=7.8, color=INK, ha="left", va="bottom")
    a3.text(1.6, 0.42, T("0.33 µT\n= 0.22 ppm", "0.33 µT\n= 0.22 ppm"), fontsize=7.8, color=INK, ha="left", va="bottom")
    a3.set_xlim(0.05, 3.0); a3.set_ylim(0.008, 30)
    a3.set_xlabel(T(r"主磁场 $B_0$（T）", r"main field $B_0$ (T)"), labelpad=1)
    a3.set_ylabel(T(r"$b_c$（$\mu$T）", r"$b_c$ ($\mu$T)"), labelpad=0)
    a3.legend(loc="lower left", fontsize=7.4, handlelength=2.0, borderaxespad=0.2)
    a3.tick_params(labelsize=7.6, length=2.5)
    clean_axes(a3)
    panel_title(a3, T(r"(c) 随 $B_0$ 的变化", r"(c) versus $B_0$"), size=8.6)
    return fig
