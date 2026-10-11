"""Figures for Chapter 3 (charge, electric field and potential; current and Ohm's law).

charge_current       three control volumes: net charge without current, current without net
                     charge, and charge accumulation when inflow exceeds outflow.
potential_vs_field   phi = 1e6 V everywhere (E = 0) versus phi = 1 V - (100 V/m) x (E = 100 V/m).
anisotropic_current  isotropic sigma = 1 S/m versus sigma = diag(1, 4) S/m for E = (1, 1) V/m.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, Rectangle

from style import (ACCENT, CURR, ELEC, FIG_W, INK, MAG, MATL, MATL_T, MUTED, POT, T, arrow, canvas, charge,
                   clean_axes, figure, halo, label, panel_title, plt, rbox, tint)


# ---------------------------------------------------------------------------
@figure("charge_current")
def charge_current():
    fig, ax = canvas(FIG_W, 2.55, xlim=(0, 10.4), ylim=(0, 4.42))
    w, h, y0 = 2.75, 2.0, 1.05
    xs = (0.1, 3.65, 7.1)
    titles = (T("(a) 有净电荷，无电流", "(a) Net charge, no current"),
              T("(b) 有电流，无净电荷", "(b) Current, no net charge"),
              T("(c) 流入多于流出", "(c) More in than out"))
    for x, t in zip(xs, titles):
        rbox(ax, x, y0, w, h, fc=MATL_T, ec="none", r=0.08, z=1)
        ax.add_patch(Rectangle((x, y0), w, h, fc="none", ec=INK, lw=1.0, ls=(0, (4, 2.5)), zorder=3))
        label(ax, x + w / 2, y0 + h + 0.95, t, weight="bold", size=9.2)
    # (a) static positive charge, no motion
    rng = np.random.default_rng(3)
    pts = [(0.4, 0.45), (1.1, 0.35), (1.85, 0.5), (2.4, 0.35), (0.65, 1.05), (1.45, 1.0), (2.2, 1.15),
           (0.38, 1.6), (1.1, 1.62), (1.85, 1.6), (2.42, 1.7)]
    for px, py in pts:
        charge(ax, xs[0] + px, y0 + py, "+", r=0.13, size=8)
    label(ax, xs[0] + w / 2, y0 - 0.35, r"$\rho>0,\quad \mathbf{J}=0$", size=9.5)
    label(ax, xs[0] + w / 2, y0 - 0.75, T("例：静电平衡时导体表面的电荷", "e.g. charge on a conductor at rest"), size=8.2, color=MUTED)
    # (b) +/- carriers drifting oppositely, driven by E
    x = xs[1]
    for k, px in enumerate((0.3, 1.15, 2.0)):
        charge(ax, x + px, y0 + 1.45, "+", r=0.13, size=8)
        arrow(ax, (x + px + 0.18, y0 + 1.45), (x + px + 0.62, y0 + 1.45), color=INK, lw=1.1, ms=7)
        charge(ax, x + px + 0.55, y0 + 0.62, "-", r=0.13, size=8)
        arrow(ax, (x + px + 0.37, y0 + 0.62), (x + px - 0.07, y0 + 0.62), color=INK, lw=1.1, ms=7)
    arrow(ax, (x + 0.15, y0 + h + 0.32), (x + 0.95, y0 + h + 0.32), color=ELEC, lw=1.8, ms=11)
    label(ax, x + 1.1, y0 + h + 0.33, r"$\mathbf{E}$", color=ELEC, size=10.5, ha="left")
    arrow(ax, (x + 1.6, y0 + h + 0.32), (x + 2.4, y0 + h + 0.32), color=CURR, lw=2.6, ms=13, z=6)
    label(ax, x + 2.55, y0 + h + 0.33, r"$\mathbf{J}$", color=CURR, size=10.5, ha="left")
    label(ax, x + w / 2, y0 - 0.35, r"$\rho\approx0,\quad \mathbf{J}\neq0$", size=9.5)
    label(ax, x + w / 2, y0 - 0.75, T("两种载流子对 $\\mathbf{J}$ 的贡献同向", "both carriers add to $\\mathbf{J}$"), size=8.2, color=MUTED)
    # (c) continuity: more current in than out
    x = xs[2]
    arrow(ax, (x - 0.5, y0 + h / 2), (x + 0.05, y0 + h / 2), color=CURR, lw=3.4, ms=15, z=6)
    label(ax, x - 0.25, y0 + h / 2 + 0.38, r"$I_{\rm in}$", color=CURR, size=9.5)
    arrow(ax, (x + w - 0.05, y0 + h / 2), (x + w + 0.42, y0 + h / 2), color=CURR, lw=1.3, ms=9, z=6)
    label(ax, x + w + 0.22, y0 + h / 2 + 0.33, r"$I_{\rm out}$", color=CURR, size=9.5)
    for px, py in ((0.55, 0.5), (1.05, 1.35), (1.5, 0.7), (1.95, 1.45), (2.25, 0.55), (0.75, 1.6)):
        charge(ax, x + px, y0 + py, "+", r=0.13, size=8)
    arrow(ax, (x + w + 0.02, y0 + 0.28), (x + w + 0.42, y0 + 0.28), color=INK, lw=1.0, ms=7, z=6)
    label(ax, x + w + 0.25, y0 + 0.02, r"$\mathbf{n}$", size=9)
    label(ax, x + w / 2, y0 - 0.35, r"$\frac{dQ}{dt}=I_{\rm in}-I_{\rm out}>0$", size=9.5)
    label(ax, x + w / 2, y0 - 0.75, T("虚线是记账边界，不是墙", "dashed: a bookkeeping surface, not a wall"), size=8.2, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("potential_vs_field")
def potential_vs_field():
    fig = plt.figure(figsize=(FIG_W, 3.0))
    x_mm = np.linspace(0, 10, 201)
    cases = [
        (np.full_like(x_mm, 1.0), T("$\\phi$（$10^6$ V）", "$\\phi$ ($10^6$ V)"),
         T("(a) 电势很高，处处相同", "(a) High potential, everywhere equal"), r"$\phi_1=10^6\,\mathrm{V}$"),
        (1.0 - 0.1 * x_mm, T("$\\phi$（V）", "$\\phi$ (V)"),
         T("(b) 电势很低，但随位置变化", "(b) Low potential, but varying"), r"$\phi_2=1\,\mathrm{V}-(100\,\mathrm{V/m})\,x$"),
    ]
    for i, (phi, ylab, title, formula) in enumerate(cases):
        left = 0.085 + i * 0.5
        ax = fig.add_axes([left, 0.5, 0.39, 0.36])
        ax.plot(x_mm, phi, color=POT, lw=2.2)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(ylab, fontsize=8.8)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.tick_params(labelbottom=False)
        clean_axes(ax, grid=True)
        ax.text(5, 1.12 if i == 0 else 0.98, formula, ha="center", fontsize=9, color=POT,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.set_title(title, fontsize=9.4, fontweight="bold", pad=6)
        # strip with equipotentials and field arrows
        sx = fig.add_axes([left, 0.17, 0.39, 0.3])
        sx.set_xlim(0, 10)
        sx.set_ylim(0, 1)
        sx.set_yticks([])
        sx.set_xticks([0, 2, 4, 6, 8, 10])
        sx.set_xlabel(T("$x$（mm）", "$x$ (mm)"), fontsize=8.8)
        clean_axes(sx)
        sx.spines["left"].set_visible(False)
        if i == 0:
            sx.add_patch(Rectangle((0, 0.05), 10, 0.9, fc=tint(POT, 0.18), ec="none"))
            sx.text(5, 0.5, T("整个区域是同一个等势区：$\\mathbf{E}=0$", "one equipotential region: $\\mathbf{E}=0$"),
                    ha="center", va="center", fontsize=9, color=INK)
        else:
            sx.add_patch(Rectangle((0, 0.05), 10, 0.9, fc=tint(POT, 0.05), ec="none"))
            for xx in range(0, 11, 2):
                sx.plot([xx, xx], [0.05, 0.95], color=POT, lw=1.2)
                txt = "1.0 V" if xx == 0 else ("0 V" if xx == 10 else f"{1 - 0.1 * xx:.1f}")
                sx.text(xx + (0.12 if xx < 10 else -0.12), 0.9, txt, ha="left" if xx < 10 else "right",
                        va="top", fontsize=7.8, color=POT)
            for xx in (1, 3, 5, 7, 9):
                arrow(sx, (xx - 0.65, 0.47), (xx + 0.65, 0.47), color=ELEC, lw=1.6, ms=9)
            sx.text(5, 0.2, T("$\\mathbf{E}=100\\,\\mathrm{V/m}$，指向电势降低的方向",
                              "$\\mathbf{E}=100\\,\\mathrm{V/m}$, pointing downhill"),
                    ha="center", va="center", fontsize=8.6, color=ELEC,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    return fig


# ---------------------------------------------------------------------------
@figure("anisotropic_current")
def anisotropic_current():
    fig = plt.figure(figsize=(FIG_W, 3.05))
    E = np.array([1.0, 1.0])
    cases = [(np.diag([1.0, 1.0]), T("(a) 各向同性：$\\sigma=1\\,\\mathrm{S/m}$", "(a) Isotropic: $\\sigma=1\\,\\mathrm{S/m}$")),
             (np.diag([1.0, 4.0]), T("(b) 各向异性：$\\boldsymbol{\\sigma}=\\mathrm{diag}(1,4)\\,\\mathrm{S/m}$",
                                     "(b) Anisotropic: $\\boldsymbol{\\sigma}=\\mathrm{diag}(1,4)\\,\\mathrm{S/m}$"))]
    for i, (S, title) in enumerate(cases):
        J = S @ E
        ax = fig.add_axes([0.02 + i * 0.5, 0.03, 0.46, 0.86])
        ax.set_xlim(-0.55, 3.35)
        ax.set_ylim(-0.45, 4.55)
        ax.set_aspect("equal")
        ax.axis("off")
        # principal axes with their conductivities
        arrow(ax, (-0.3, 0), (2.9, 0), color=MUTED, lw=0.9, ms=8)
        arrow(ax, (0, -0.3), (0, 4.4), color=MUTED, lw=0.9, ms=8)
        label(ax, 2.95, -0.25, "$x$", size=9, color=MUTED)
        label(ax, -0.25, 4.35, "$y$", size=9, color=MUTED)
        label(ax, 2.05, -0.3, f"$\\sigma_x={S[0, 0]:g}$", size=8.6, color=MUTED)
        label(ax, -0.32, 2.6, f"$\\sigma_y={S[1, 1]:g}$", size=8.6, color=MUTED, rotation=90)
        # component bookkeeping (dotted)
        ax.plot([E[0], E[0]], [0, J[1]], color=tint(CURR, 0.6), lw=0.9, ls=(0, (2, 2)), zorder=2)
        ax.plot([0, J[0]], [J[1], J[1]], color=tint(CURR, 0.6), lw=0.9, ls=(0, (2, 2)), zorder=2)
        ax.plot([0, E[0]], [E[1], E[1]], color=tint(ELEC, 0.6), lw=0.9, ls=(0, (2, 2)), zorder=2)
        arrow(ax, (0, 0), tuple(E), color=ELEC, lw=4.2 if i == 0 else 2.4, ms=15 if i == 0 else 13, z=5)
        arrow(ax, (0, 0), tuple(J), color=CURR, lw=1.8 if i == 0 else 2.4, ms=10 if i == 0 else 13, z=6)
        dot = float(E @ J)
        ang = np.degrees(np.arctan2(J[1], J[0]) - np.arctan2(E[1], E[0]))
        if i == 0:
            halo(label(ax, 1.12, 0.78, r"$\mathbf{E}=(1,1)\,\mathrm{V/m}$", color=ELEC, size=9, ha="left"), 3)
            halo(label(ax, 1.12, 1.25, r"$\mathbf{J}_c=(1,1)\,\mathrm{A/m^2}$", color=CURR, size=9, ha="left"), 3)
            label(ax, 1.55, 3.25, T("$\\mathbf{J}_c$ 与 $\\mathbf{E}$ 同向\n", "$\\mathbf{J}_c$ parallel to $\\mathbf{E}$\n") +
                  f"$\\mathbf{{E}}\\cdot\\mathbf{{J}}_c={dot:g}\\,\\mathrm{{W/m^3}}$", size=8.8, color=INK,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=tint(INK, 0.3), lw=0.7))
        else:
            halo(label(ax, 1.12, 0.78, r"$\mathbf{E}=(1,1)\,\mathrm{V/m}$", color=ELEC, size=9, ha="left"), 3)
            halo(label(ax, 1.12, 3.95, r"$\mathbf{J}_c=(1,4)\,\mathrm{A/m^2}$", color=CURR, size=9, ha="left"), 3)
            th0, th1 = np.degrees(np.arctan2(E[1], E[0])), np.degrees(np.arctan2(J[1], J[0]))
            ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=th0, theta2=th1, color=INK, lw=0.9, zorder=4))
            halo(label(ax, 0.36, 0.86, f"{ang:.0f}°", size=8.4), 3)
            label(ax, 2.05, 2.3, T("偏向电导率大的 $y$ 方向\n", "pulled toward the high-$\\sigma$ axis\n") +
                  f"$\\mathbf{{E}}\\cdot\\mathbf{{J}}_c={dot:g}\\,\\mathrm{{W/m^3}}>0$", size=8.8, color=INK,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=tint(INK, 0.3), lw=0.7))
            label(ax, 1.7, 1.55, T("$J_x=1\\cdot1$，$J_y=4\\cdot1$", "$J_x=1\\cdot1$, $J_y=4\\cdot1$"), size=8.4, color=CURR)
        panel_title(ax, title, size=9.4)
    return fig
