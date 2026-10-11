"""Figures for Ch 22 (the edge of the quasistatic picture: RF fields and SAR).

All numbers come from checks/check_ch22.py (teaching inputs of the old edition, rf:3 and rf:5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from style import (ACCENT, AMBER, ELEC, FIG_W, INK, LIGHT, MAG, MAG_T, MUTED, NEUTRAL_T, T, arrow, canvas,
                   card, clean_axes, figure, halo, label, plt, rbox, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch22 as RF  # noqa: E402

HEAT = AMBER


def _arc_arrow(ax, r, th0, th1, color, lw=1.1, ms=8, z=4):
    """Circular arc from angle th0 to th1 (degrees) with an arrow head at th1."""
    th = np.deg2rad(np.linspace(th0, th1, 60))
    ax.plot(r * np.cos(th), r * np.sin(th), color=color, lw=lw, zorder=z, solid_capstyle="round")
    p, q = (r * np.cos(th[-3]), r * np.sin(th[-3])), (r * np.cos(th[-1]), r * np.sin(th[-1]))
    arrow(ax, p, q, color=color, lw=lw, ms=ms, z=z)


def _frame(ax, lim=1.35):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([-lim * 0.92, lim * 0.92], [0, 0], color=LIGHT, lw=0.8, zorder=0)
    ax.plot([0, 0], [-lim * 0.92, lim * 0.8], color=LIGHT, lw=0.8, zorder=0)
    label(ax, lim * 0.97, -0.1, "$x$", size=8.4, color=MUTED)
    label(ax, 0.1, lim * 0.86, "$y$", size=8.4, color=MUTED)


@figure("rotating_components")
def rotating_components():
    fig = plt.figure(figsize=(FIG_W, 2.72))
    a1 = fig.add_axes([0.0, 0.08, 0.33, 0.76])
    a2 = fig.add_axes([0.335, 0.08, 0.33, 0.76])
    a3 = fig.add_axes([0.765, 0.2, 0.225, 0.58])
    ph = np.deg2rad(50.0)
    b = 1.0
    # ---------------- (a) linear drive = co + counter
    _frame(a1)
    a1.plot([-b, b], [0, 0], color=tint(MAG, 0.45), lw=4.0, zorder=1, solid_capstyle="round")
    a1.add_patch(Circle((0, 0), 0.5, fc="none", ec=tint(MAG, 0.5), lw=0.8, ls=(0, (2.5, 2)), zorder=1))
    co = 0.5 * b * np.array([np.cos(ph), -np.sin(ph)])
    cn = 0.5 * b * np.array([np.cos(ph), np.sin(ph)])
    tot = co + cn
    arrow(a1, (0, 0), tuple(co), color=MAG, lw=1.7, ms=9, z=5)
    arrow(a1, (0, 0), tuple(cn), color=tint(MAG, 0.55), lw=1.5, ms=8, z=4)
    a1.plot([co[0], tot[0]], [co[1], tot[1]], color=MUTED, lw=0.7, ls=(0, (1.5, 1.5)), zorder=3)
    a1.plot([cn[0], tot[0]], [cn[1], tot[1]], color=MUTED, lw=0.7, ls=(0, (1.5, 1.5)), zorder=3)
    arrow(a1, (0, 0), tuple(tot), color=INK, lw=2.0, ms=10, z=6)
    _arc_arrow(a1, 0.66, -18, -72, MAG, lw=1.1, ms=7)
    _arc_arrow(a1, 0.66, 18, 72, tint(MAG, 0.55), lw=1.1, ms=7)
    t = label(a1, 0.62, -0.84, T("同向分量 $b/2$\n（顺时针，与自旋同向）", "co-rotating $b/2$\n(clockwise, with the spins)"),
              size=7.6, color=MAG, ha="center")
    halo(t)
    t = label(a1, 0.62, 0.86, T("反向分量 $b/2$（逆时针）", "counter-rotating $b/2$"), size=7.6, color=tint(MAG, 0.7))
    halo(t)
    t = label(a1, -0.95, 0.22, T("线性场的\n端点轨迹", "tip of the\nlinear field"), size=7.4, color=tint(MAG, 0.75))
    halo(t)
    a1.set_title(T("(a) 线性驱动 $(\\widetilde B_x,\\widetilde B_y)=(b,0)$", "(a) linear drive $(\\widetilde B_x,\\widetilde B_y)=(b,0)$"),
                 fontsize=8.8, weight="bold", pad=2)
    # ---------------- (b) quadrature drive
    _frame(a2)
    a2.add_patch(Circle((0, 0), b, fc="none", ec=tint(MAG, 0.5), lw=0.8, ls=(0, (2.5, 2)), zorder=1))
    v = b * np.array([np.cos(ph), -np.sin(ph)])
    arrow(a2, (0, 0), tuple(v), color=MAG, lw=2.0, ms=10, z=5)
    _arc_arrow(a2, 1.1, -12, -80, MAG, lw=1.1, ms=7)
    t = label(a2, -0.02, -0.28, T("同向分量 $b$\n反向分量 0", "co-rotating $b$\ncounter 0"), size=7.6, color=MAG,
              ha="right")
    halo(t)
    # precession sense icon (clockwise seen from +z)
    cx, cy = -1.08, 1.0
    a2.add_patch(Circle((cx, cy), 0.1, fc=tint(ELEC, 0.15), ec=ELEC, lw=0.8, zorder=3))
    th = np.deg2rad(np.linspace(110, -160, 50))
    a2.plot(cx + 0.19 * np.cos(th), cy + 0.19 * np.sin(th), color=ELEC, lw=0.9, zorder=3)
    arrow(a2, (cx + 0.19 * np.cos(th[-3]), cy + 0.19 * np.sin(th[-3])), (cx + 0.19 * np.cos(th[-1]), cy + 0.19 * np.sin(th[-1])),
          color=ELEC, lw=0.9, ms=6, z=4)
    t = label(a2, cx + 0.27, cy + 0.02, T("自旋", "spins"), size=7.4, color=ELEC, ha="left")
    halo(t)
    a2.set_title(T("(b) 正交驱动 $(b,\\,\\mathrm{j}b)$", "(b) quadrature drive $(b,\\,\\mathrm{j}b)$"), fontsize=8.8,
                 weight="bold", pad=2)
    # ---------------- (c) magnitudes
    lin = [abs(z) for z in RF.corotating(b, 0)]
    quad = [abs(z) for z in RF.corotating(b, 1j * b)]
    xg = np.array([0.0, 1.0])
    wbar = 0.34
    a3.bar(xg - wbar / 2, [lin[0], quad[0]], width=wbar, color=MAG, label=T("同向 $|\\widetilde b_{\\rm co}|$", "co $|\\widetilde b_{\\rm co}|$"))
    a3.bar(xg + wbar / 2, [lin[1], quad[1]], width=wbar, color=tint(MAG, 0.4), label=T("反向", "counter"))
    for x0, v0 in zip(np.r_[xg - wbar / 2, xg + wbar / 2], lin[:1] + quad[:1] + lin[1:] + quad[1:]):
        a3.text(x0, v0 + 0.03, f"{v0:g}", ha="center", va="bottom", fontsize=7.6, color=INK)
    a3.set_xticks(xg)
    a3.set_xticklabels([T("线性", "linear"), T("正交", "quadrature")], fontsize=8.0)
    a3.set_ylim(0, 1.25); a3.set_yticks([0, 0.5, 1])
    a3.set_ylabel(T("幅值 / $b$", "amplitude / $b$"), fontsize=8.2)
    a3.legend(loc="upper left", fontsize=7.2, handlelength=1.0, borderaxespad=0.1, labelspacing=0.2)
    clean_axes(a3)
    a3.set_title(T("(c) 同一峰值 $b$ 下", "(c) for the same peak $b$"), fontsize=8.8, weight="bold", pad=4)
    fig.text(0.765, 0.075, T("线性驱动只有一半用于激发", "half of a linear field excites"), fontsize=7.6,
             color=MAG, ha="left")
    fig.text(0.33, 0.005, T("约定：峰值相量 $e^{\\mathrm{j}\\omega t}$；从 $+z$ 俯视，$\\gamma>0$ 的自旋顺时针进动，“同向”即顺时针",
                            "convention: peak phasors $e^{\\mathrm{j}\\omega t}$; seen from $+z$, spins with $\\gamma>0$ precess clockwise"),
             fontsize=7.4, color=ELEC, ha="center", va="bottom")
    return fig


# ---------------------------------------------------------------------------
@figure("rf_averaging_ladder")
def rf_averaging_ladder():
    fig = plt.figure(figsize=(FIG_W, 3.75))
    xl, wl = 0.005, 0.29
    rows = [0.69, 0.375, 0.06]
    hr = 0.25
    axs = [fig.add_axes([0.445, y0 + 0.045, 0.54, hr - 0.05]) for y0 in rows]
    ov = fig.add_axes([0, 0, 1, 1]); ov.axis("off"); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
    p_bar = RF.local_sar_cycle(RF.SAR_E)
    texts = [
        (T("① RF 周期平均", "① RF-cycle average"),
         [T("瞬时 $\\sigma E(t)^2/\\rho_m$ 在 $0$ 与 $5$ 之间振荡", "instantaneous $\\sigma E(t)^2/\\rho_m$: 0 to 5"),
          T("平均：$\\frac{1}{2}\\sigma|\\widetilde E|^2/\\rho_m=2.5$ W/kg", "average $\\frac{1}{2}\\sigma|\\widetilde E|^2/\\rho_m=2.5$ W/kg"),
          T("（若 100 V/m 是有效值：5 W/kg）", "(if 100 V/m is an RMS value: 5 W/kg)")]),
        (T("② 空间（质量）平均", "② spatial (mass) average"),
         [T("逐点 SAR → 区域 $V_a$ 上按质量平均", "point SAR → mass average over $V_a$"),
          T("结果取决于 $V_a$ 的大小与构造", "depends on the size and shape of $V_a$"),
          T("均匀场时与逐点值相同", "equals the point value if uniform")]),
        (T("③ 脉冲串平均", "③ pulse-train average"),
         [T("包络 $\\mathbf{c}(t)$：每 10 ms 发射 1 ms", "envelope $\\mathbf{c}(t)$: 1 ms in every 10 ms"),
          T("脉冲期间 2.5 W/kg", "2.5 W/kg during the pulse"),
          T("长期平均 $0.1\\times2.5=0.25$ W/kg", "long-term $0.1\\times2.5=0.25$ W/kg")]),
    ]
    for (title, lines), y0 in zip(texts, rows):
        rbox(ov, xl, y0 + 0.012, wl + 0.07, hr - 0.01, fc=tint(ACCENT, 0.07), ec=tint(ACCENT, 0.4), lw=0.8, r=0.012)
        label(ov, xl + 0.018, y0 + hr - 0.045, title, size=8.8, weight="bold", color=ACCENT, ha="left")
        for i, s in enumerate(lines):
            label(ov, xl + 0.018, y0 + hr - 0.1 - i * 0.05, s, size=7.5, ha="left", color=INK if i < 2 else MUTED)
    for y0 in rows[:-1]:
        arrow(ov, (xl + 0.19, y0 + 0.012), (xl + 0.19, y0 - 0.045), color=ACCENT, lw=1.3, ms=9)
    # ---- row 1: RF cycle (128 MHz example)
    a = axs[0]
    T_rf = 1e9 / 128e6
    t = np.linspace(0, 2.0 * T_rf, 600)
    inst = RF.SAR_SIGMA * (RF.SAR_E * np.cos(2 * np.pi * t / T_rf)) ** 2 / RF.SAR_RHO
    a.fill_between(t, 0, inst, color=tint(HEAT, 0.22), lw=0)
    a.plot(t, inst, color=HEAT, lw=1.6)
    a.axhline(p_bar, color=INK, lw=1.2, ls=(0, (4, 2)))
    a.set_xlim(0, 2 * T_rf); a.set_ylim(0, 5.9); a.set_yticks([0, 2.5, 5])
    a.set_xlabel(T("时间 (ns)，128 MHz 时一个周期 7.8 ns", "time (ns); one period at 128 MHz is 7.8 ns"), fontsize=7.8, labelpad=1)
    a.text(0.75 * T_rf, p_bar + 0.2, T("平均 2.5 W/kg", "mean 2.5 W/kg"), fontsize=7.8, ha="center", va="bottom", color=INK,
           bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    # ---- row 2: hot spot and two averaging windows
    a = axs[1]
    x = np.linspace(-30e-3, 30e-3, 600)
    a.fill_between(x * 1e3, 0, RF.hotspot_sar(x), color=tint(HEAT, 0.22), lw=0)
    a.plot(x * 1e3, RF.hotspot_sar(x), color=HEAT, lw=1.6)
    for L, c, yv in ((10e-3, ACCENT, RF.window_average(10e-3)), (20e-3, MUTED, RF.window_average(20e-3))):
        a.plot([-L / 2 * 1e3, L / 2 * 1e3], [yv, yv], color=c, lw=2.0, solid_capstyle="butt")
        for xe in (-L / 2 * 1e3, L / 2 * 1e3):
            a.plot([xe, xe], [yv - 0.18, yv + 0.18], color=c, lw=1.0)
        a.text(L / 2 * 1e3 + 0.8, yv, f"{L*1e3:.0f} mm: {yv:.2f}", fontsize=7.6, color=c, va="center", ha="left")
    a.text(1.0, 2.6, T("逐点峰值 2.5", "point peak 2.5"), fontsize=7.6, color=HEAT, ha="left", va="center")
    a.set_xlim(-30, 30); a.set_ylim(0, 3.1); a.set_yticks([0, 1, 2])
    a.set_xlabel(T("位置 (mm)，一维教学热点", "position (mm), 1-D teaching hot spot"), fontsize=7.8, labelpad=1)
    # ---- row 3: pulse train
    a = axs[2]
    tt = np.linspace(0, 39.99, 4000)
    env = ((tt % 10.0) < 1.0).astype(float)
    a.fill_between(tt, 0, env * p_bar, color=tint(HEAT, 0.22), lw=0, step="post")
    a.plot(tt, env * p_bar, color=HEAT, lw=1.4, drawstyle="steps-post")
    a.axhline(0.25, color=INK, lw=1.2, ls=(0, (4, 2)))
    a.text(39.5, 0.45, "0.25 W/kg", fontsize=7.8, ha="right", va="bottom", color=INK)
    a.set_xlim(0, 40); a.set_ylim(0, 3.1); a.set_yticks([0, 1, 2])
    a.set_xlabel(T("时间 (ms)", "time (ms)"), fontsize=7.8, labelpad=1)
    for a in axs:
        clean_axes(a)
        a.tick_params(labelsize=7.5)
    fig.text(0.388, 0.5, T("SAR (W/kg)", "SAR (W/kg)"), rotation=90, fontsize=8.2, va="center", ha="center")
    return fig
