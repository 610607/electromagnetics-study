"""Figures for Ch 18 (unified framework and shared observables).

application_map    three operator families inside one modelling pipeline.
field_observables  one field solution, three kinds of read-out (linear, quadratic, nonlinear/dynamic).
sar_cross_terms    two coherent channels: why single-channel SAR maps cannot be added (computed).
energy_coenergy    lambda-i areas of the saturating C-core and the force by three formulas (computed).

Computed quantities come from checks/check_ch18.py, so text, figures and assertions share one model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from style import (ACCENT, AMBER, CURR, ELEC, FIG_W, INK, LIGHT, MAG, MATL, MUTED, POT, T, arrow, canvas,
                   clean_axes, figure, label, panel_title, plt, rbox, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch18 as FW  # noqa: E402


# ---------------------------------------------------------------------------
@figure("application_map")
def application_map():
    W, H = 10.4, 6.15
    fig, ax = canvas(FIG_W, FIG_W * H / W, xlim=(0, W), ylim=(0, H))
    # --- pipeline strip --------------------------------------------------------
    y0, hh = H - 0.78, 0.6
    steps = [(T("几何", "geometry"), MATL), (T("材料", "materials"), MATL), (T("源", "sources"), MATL),
             (T("近似", "regime"), MATL)]
    x = 0.08
    for i, (s, c) in enumerate(steps):
        w = 1.02
        rbox(ax, x, y0, w, hh, fc=tint(c, 0.16), ec=tint(c, 0.6), lw=0.8, r=0.1)
        label(ax, x + w / 2, y0 + hh / 2, s, size=T(8.8, 8.0), color=INK)
        if i < 3:
            label(ax, x + w + 0.09, y0 + hh / 2, "+", size=10, color=MUTED, weight="bold")
        x += w + 0.18
    xa = x - 0.1
    arrow(ax, (xa, y0 + hh / 2), (xa + 0.5, y0 + hh / 2), color=INK, lw=1.2, ms=9)
    xb = xa + 0.58
    rbox(ax, xb, y0, 2.05, hh, fc=tint(ACCENT, 0.85), ec="none", r=0.1)
    label(ax, xb + 1.025, y0 + hh / 2, T("场问题：三族算子", "field problem"), size=8.8, color="white",
          weight="bold")
    arrow(ax, (xb + 2.1, y0 + hh / 2), (xb + 2.6, y0 + hh / 2), color=INK, lw=1.2, ms=9)
    xc = xb + 2.68
    rbox(ax, xc, y0, W - 0.08 - xc, hh, fc=tint(AMBER, 0.85), ec="none", r=0.1)
    label(ax, (xc + W - 0.08) / 2, y0 + hh / 2, T("观测量", "observables"), size=8.8, color="white", weight="bold")
    # --- three families ----------------------------------------------------------
    rows = [
        (T("电准静态传导", "EQS conduction"), T("TES 的算子", "the TES operator"), ELEC,
         [r"$\nabla\cdot(\boldsymbol{\sigma}\nabla\phi)=0,\quad \mathbf{E}=-\nabla\phi$",
          T("源：电极电流（在边界上）　读出：$\\mathbf{E}$、$\\mathbf{J}$",
            "source: electrode currents (on the boundary)   read: $\\mathbf{E}$, $\\mathbf{J}$")],
         T("TES · 接触阻抗 · 低频生物电", "TES · contact impedance · bioelectric"), T("第 10 章", "Ch. 10")),
        (T("指定源感应", "prescribed-source\ninduction"), T("单向耦合 MQS", "one-way MQS"), CURR,
         [r"$\mathbf{E}=-\partial_t\mathbf{A}_s-\nabla\phi,\quad \nabla\cdot(\boldsymbol{\sigma}\mathbf{E})=0$",
          T("源：线圈的 $\\dot I$（遍布体内）　读出：组织中的 $\\mathbf{E}$",
            "source: coil $\\dot I$ (throughout the volume)   read: $\\mathbf{E}$ in tissue")],
         T("TMS · 梯度切换在人体中的电场", "TMS · gradient-induced fields in the body"), T("第 11 章", "Ch. 11")),
        (T("静磁与耦合 MQS", "magnetostatics,\ncoupled MQS"), T("含磁性材料", "magnetic materials"), MAG,
         [r"$\nabla\cdot\mathbf{B}=0,\quad \nabla\times\mathbf{H}=\mathbf{J}_s\,(+\boldsymbol{\sigma}\mathbf{E}),\quad \mathbf{B}=\mathbf{B}(\mathbf{H})$",
          T("源：绕组电流、剩磁　读出：$\\mathbf{B}$、$\\lambda$、$L$、力",
            "source: winding currents, remanence   read: $\\mathbf{B}$, $\\lambda$, $L$, force")],
         T("主磁体 · 磁轭 · 梯度与匀场线圈 · 屏蔽涡流", "magnets · yokes · gradient/shim coils · shield eddies"),
         T("第 12–15、19–20 章", "Ch. 12–15, 19–20")),
    ]
    h, gap = 1.22, 0.14
    top = y0 - 0.32
    xt, wt = 0.08, 2.0
    for i, (name, sub, col, lines, apps, chap) in enumerate(rows):
        y = top - (i + 1) * h - i * gap
        rbox(ax, xt, y, W - 2 * xt, h, fc=tint(col, 0.07), ec=tint(col, 0.5), lw=0.9, r=0.1)
        rbox(ax, xt, y, wt, h, fc=tint(col, 0.88), ec="none", r=0.1, z=2.2)
        two = "\n" in name
        label(ax, xt + wt / 2, y + h * (0.6 if two else 0.62), name, color="white", size=9.6 if not two else 8.8,
              weight="bold")
        label(ax, xt + wt / 2, y + h * (0.2 if two else 0.30), sub, color="white", size=7.9)
        xb_ = xt + wt + 0.16
        label(ax, xb_, y + h * 0.75, lines[0], size=9.0, ha="left")
        label(ax, xb_, y + h * 0.46, lines[1], size=7.9, ha="left", color=MUTED)
        label(ax, xb_, y + h * 0.17, apps, size=7.9, ha="left", color=INK)
        label(ax, W - 0.2, y + h * 0.17, chap, color=col, size=7.8, ha="right",
              bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.4", fc="white", ec=tint(col, 0.6), lw=0.8))
    yb = top - 3 * h - 2 * gap - 0.2 - 0.5
    rbox(ax, xt, yb, W - 2 * xt, 0.5, fc=tint(INK, 0.82), ec="none", r=0.1)
    label(ax, W / 2, yb + 0.25, T("共同的部分：Maxwell 方程 · 本构 · 边界条件；线性材料中场对源线性 → 叠加与单位场图",
                                  "shared: Maxwell's equations · constitutive laws · boundaries; linear media: fields linear in the source"),
          color="white", size=8.2)
    return fig


# ---------------------------------------------------------------------------
@figure("field_observables")
def field_observables():
    W, H = 10.4, 5.5
    fig, ax = canvas(FIG_W, FIG_W * H / W, xlim=(0, W), ylim=(0, H))
    # source box on the left
    xs, ws = 0.08, 2.0
    ys, hs = 1.55, 2.4
    rbox(ax, xs, ys, ws, hs, fc=tint(ACCENT, 0.1), ec=tint(ACCENT, 0.6), lw=1.0, r=0.12)
    label(ax, xs + ws / 2, ys + hs - 0.32, T("一次场解", "one field solution"), size=9.6, weight="bold", color=ACCENT)
    label(ax, xs + ws / 2, ys + hs - 0.82, r"$\phi,\ \mathbf{A}$", size=10, color=POT)
    label(ax, xs + ws / 2, ys + hs - 1.28, r"$\mathbf{E},\ \mathbf{J}$", size=10, color=ELEC)
    label(ax, xs + ws / 2, ys + hs - 1.74, r"$\mathbf{B},\ \mathbf{H}$", size=10, color=MAG)
    label(ax, xs + ws / 2, ys + 0.24, T("每单位源幅值", "per unit source"), size=7.8, color=MUTED)
    cards = [
        (T("线性读出", "linear read-outs"), ELEC, T("对源线性：可以叠加", "linear in the source: superpose"),
         [T("采样与投影：沿神经的 $\\mathbf{E}\\cdot\\hat{\\mathbf{t}}$、某点的 $b_z$",
            "sample, project: $\\mathbf{E}\\cdot\\hat{\\mathbf{t}}$ along a nerve, $b_z$ at a point"),
          T("空间导数：$G_{ij}=\\partial b_i/\\partial x_j$、$\\partial E_s/\\partial s$",
            "derivatives: $G_{ij}=\\partial b_i/\\partial x_j$, $\\partial E_s/\\partial s$"),
          T("积分：磁链 $\\lambda$、端电压、区域平均", "integrals: linkage $\\lambda$, voltage, averages")],
         T("第 10、11、19 章", "Ch. 10, 11, 19")),
        (T("二次读出", "quadratic read-outs"), AMBER, T("源的二次型：有交叉项", "quadratic: cross terms"),
         [T("损耗与 SAR：$\\mathbf{c}^{\\dagger}\\mathbf{Q}\\,\\mathbf{c}$", "loss and SAR: $\\mathbf{c}^{\\dagger}\\mathbf{Q}\\,\\mathbf{c}$"),
          T("储能：$\\frac{1}{2}\\mathbf{I}^{\\mathsf{T}}\\mathbf{L}\\,\\mathbf{I}$；功率 $\\mathbf{I}^{\\mathsf{T}}\\mathbf{R}\\,\\mathbf{I}$",
            "energy $\\frac{1}{2}\\mathbf{I}^{\\mathsf{T}}\\mathbf{L}\\,\\mathbf{I}$, power $\\mathbf{I}^{\\mathsf{T}}\\mathbf{R}\\,\\mathbf{I}$"),
          T("力与力矩：应力张量 $\\oint\\mathbf{T}_m\\mathbf{n}\\,\\mathrm{d}S$", "force, torque: $\\oint\\mathbf{T}_m\\mathbf{n}\\,\\mathrm{d}S$")],
         T("第 7、15、18、22 章", "Ch. 7, 15, 18, 22")),
        (T("非线性与动力学读出", "nonlinear, dynamic read-outs"), POT, T("先合成场，再计算", "combine fields first"),
         [T("模长与比值：$|\\mathbf{E}|$、聚焦度", "magnitude, ratios: $|\\mathbf{E}|$, focality"),
          T("均匀度：$(\\max-\\min)/B_{\\rm ref}$，ppm", "homogeneity: $(\\max-\\min)/B_{\\rm ref}$ in ppm"),
          T("时间积分与响应：相位、k 空间、膜电位", "time integrals, responses: phase, k-space, membranes")],
         T("第 10、19、21 章", "Ch. 10, 19, 21")),
    ]
    xc, wc = 2.75, W - 2.75 - 0.08
    hc, gap = 1.62, 0.16
    top = H - 0.06
    for i, (title, col, prop, lines, chap) in enumerate(cards):
        y = top - (i + 1) * hc - i * gap
        rbox(ax, xc, y, wc, hc, fc=tint(col, 0.06), ec=tint(col, 0.55), lw=0.9, r=0.1)
        rbox(ax, xc, y + hc - 0.38, wc, 0.38, fc=tint(col, 0.85), ec="none", r=0.1, z=2.2)
        ax.add_patch(plt.Rectangle((xc, y + hc - 0.38), wc, 0.19, fc=tint(col, 0.85), ec="none", zorder=2.2))
        label(ax, xc + 0.15, y + hc - 0.19, title, color="white", size=9.2, weight="bold", ha="left")
        label(ax, xc + wc - 0.15, y + hc - 0.19, prop, color="white", size=8.2, ha="right")
        for k, s in enumerate(lines):
            label(ax, xc + 0.2, y + hc - 0.62 - 0.33 * k, "·  " + s, size=8.2, ha="left")
        label(ax, xc + wc - 0.12, y + 0.2, chap, size=7.6, color=col, ha="right")
        ym = y + hc / 2 - 0.1
        arrow(ax, (xs + ws + 0.04, ys + hs / 2), (xc - 0.06, ym), color=tint(col, 0.9), lw=1.3, ms=9)
    return fig


# ---------------------------------------------------------------------------
@figure("sar_cross_terms")
def sar_cross_terms():
    fig = plt.figure(figsize=(FIG_W, 3.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[0.7, 1.3], hspace=0.12, wspace=0.34,
                          left=0.085, right=0.985, bottom=0.14, top=0.92)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0); a2 = fig.add_subplot(gs[:, 1])
    x = np.linspace(-1, 1, 401)
    e1, e2 = FW.channel_fields(x)
    a0.plot(x, e1, color=ELEC, lw=1.5)
    a0.plot(x, e2, color=ELEC, lw=1.5, ls=(0, (4, 2)))
    a0.text(-0.55, 108, r"$e_1$", fontsize=9, color=ELEC, ha="center")
    a0.text(0.55, 108, r"$e_2$", fontsize=9, color=ELEC, ha="center")
    a0.set_ylim(0, 165); a0.set_yticks([0, 100])
    a0.set_ylabel(T("场（V/m）", "field (V/m)"), fontsize=8.4)
    plt.setp(a0.get_xticklabels(), visible=False)
    clean_axes(a0)
    panel_title(a0, T("(a) 两个通道的场与 SAR", "(a) two channel fields and SAR"), size=9.0)
    s1 = FW.SAR_SIGMA * e1**2 / (2 * FW.SAR_RHO); s2 = FW.SAR_SIGMA * e2**2 / (2 * FW.SAR_RHO)
    a1.plot(x, s1, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    a1.plot(x, s2, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    a1.text(-0.97, 10.45, T("灰色虚线：单个通道", "grey: one channel"), fontsize=7.4, color=MUTED, ha="left")
    a1.plot(x, FW.profile_sum_of_maps(x), color=INK, lw=1.4, ls=(0, (1, 1.5)))
    a1.plot(x, FW.profile_sar(x, 0.0), color=AMBER, lw=1.9)
    a1.plot(x, FW.profile_sar(x, np.pi / 2), color=CURR, lw=1.5, ls=(0, (5, 2)))
    a1.plot(x, FW.profile_sar(x, np.pi), color=ELEC, lw=1.9)
    a1.text(0.0, 10.25, T("同相", "in phase"), fontsize=8.0, color=AMBER, ha="center", va="bottom")
    a1.text(0.0, 0.45, T("反相", "opposite"), fontsize=8.0, color=ELEC, ha="center", va="bottom")
    a1.text(0.97, 5.35, T("相差 90° = 两图相加", "90° apart = sum of maps"), fontsize=7.6, color=CURR,
            ha="right", va="bottom")
    a1.set_xlim(-1, 1); a1.set_ylim(0, 11)
    a1.set_xlabel(T("沿直线的位置 $x$（归一化）", "position $x$ along the line (normalised)"))
    a1.set_ylabel("SAR (W/kg)", fontsize=8.6)
    clean_axes(a1)
    # (b) SAR versus phase difference
    th = np.linspace(0, 2 * np.pi, 361)
    Q = FW.q_point(); Qa = FW.q_line_average()
    sp = np.array([FW.sar_of_drive(Q, [1, np.exp(1j * t)]) for t in th])
    sa = np.array([FW.sar_of_drive(Qa, [1, np.exp(1j * t)]) for t in th])
    deg = np.degrees(th)
    a2.axhline(5.0, color=INK, lw=1.0, ls=(0, (1, 1.5)))
    a2.plot(deg, sp, color=AMBER, lw=1.9, label=T("中心点 $x=0$", "centre point $x=0$"))
    a2.plot(deg, sa, color=POT, lw=1.6, ls=(0, (5, 2)), label=T("整条线的质量平均", "mass average over the line"))
    for t in (0, 90, 180, 270, 360):
        v = FW.sar_of_drive(Q, [1, np.exp(1j * np.radians(t))])
        a2.plot(t, v, "o", ms=4.5, mfc="white", mec=AMBER, mew=1.3, zorder=5)
    a2.text(185, 5.35, T("两张图相加：5 W/kg", "sum of maps: 5 W/kg"), fontsize=T(7.8, 7.0), color=INK, ha="center")
    a2.text(14, 10.0, "10", fontsize=8, color=AMBER, ha="left", va="center")
    a2.text(180, 0.7, "0", fontsize=8, color=AMBER, ha="center", va="bottom")
    a2.text(9, 8.35, "8.18", fontsize=7.8, color=POT, ha="left", va="bottom")
    a2.text(180, 2.15, "1.82", fontsize=7.8, color=POT, ha="center", va="bottom")
    a2.set_xlim(0, 360); a2.set_xticks([0, 90, 180, 270, 360]); a2.set_ylim(0, 11)
    a2.set_xlabel(T("两通道的相位差 $\\theta$（°）", "phase difference $\\theta$ (°)"))
    a2.set_ylabel("SAR (W/kg)", fontsize=8.6)
    a2.legend(loc="upper center", fontsize=7.6, handlelength=1.8, bbox_to_anchor=(0.55, 1.0))
    clean_axes(a2)
    panel_title(a2, T("(b) 交叉项随相位变化", "(b) the cross term follows the phase"), size=9.0)
    return fig


# ---------------------------------------------------------------------------
@figure("energy_coenergy")
def energy_coenergy():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 2.8), gridspec_kw=dict(width_ratios=[1.0, 1.12]))
    i0 = 10.0
    i, lam = FW.lambda_curve(i0, 300)
    lam0 = lam[-1]
    a1.fill_between(i, 0, lam, color=tint(MAG, 0.16), lw=0, zorder=1)
    a1.fill_betweenx(lam, 0, i, color=tint("#6B7785", 0.15), lw=0, zorder=1)
    a1.plot(i, lam, color=MAG, lw=2.0, zorder=3)
    a1.plot([0, i0], [0, lam0], color=MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2)
    a1.plot([i0, i0], [0, lam0], color=MUTED, lw=0.7)
    a1.plot([0, i0], [lam0, lam0], color=MUTED, lw=0.7)
    a1.plot(i0, lam0, "o", ms=5, mfc="white", mec=MAG, mew=1.4, zorder=5)
    r = FW.energy_coenergy(2000.0)
    a1.text(6.9, 0.03, T(f"共能 $W^{{\\prime}}=\\int\\lambda\\,\\mathrm{{d}}i$\n= {r['Wc']:.3f} J",
                         f"co-energy $W^{{\\prime}}=\\int\\lambda\\,\\mathrm{{d}}i$\n= {r['Wc']:.3f} J"),
            fontsize=8.2, color=MAG, ha="center", va="center")
    a1.text(0.45, 0.112, T(f"能量 $W=\\int i\\,\\mathrm{{d}}\\lambda$\n= {r['W']:.3f} J",
                           f"energy $W=\\int i\\,\\mathrm{{d}}\\lambda$\n= {r['W']:.3f} J"),
            fontsize=8.2, color=INK, ha="left", va="center")
    a1.text(7.15, 0.0745, T("线性时沿虚线：$W=W^{\\prime}$", "linear: chord, $W=W^{\\prime}$"), fontsize=7.6,
            color=MUTED, rotation=31, ha="center", va="center")
    a1.set_xlim(0, 10.6); a1.set_ylim(0, 0.142)
    a1.set_xlabel(T("线圈电流 $i$（A）", "coil current $i$ (A)"))
    a1.set_ylabel(T("磁链 $\\lambda$（Wb）", "flux linkage $\\lambda$ (Wb)"))
    clean_axes(a1)
    panel_title(a1, T("(a) 饱和铁芯的 $\\lambda$–$i$ 曲线", "(a) $\\lambda$–$i$ of the saturating core"), size=9.0)
    NIs = np.linspace(50, 3000, 60)
    F = FW.force_curves(NIs)
    a2.plot(NIs, F[:, 2], color=MUTED, lw=1.3, ls=(0, (1.5, 1.5)), label=T("线性铁 $\\mu_r=2000$", "linear iron $\\mu_r=2000$"))
    a2.plot(NIs, F[:, 0], color=MAG, lw=2.0, label=T("非线性共能 = 应力张量", "nonlinear co-energy = stress"))
    a2.plot(NIs, F[:, 1], color=ELEC, lw=1.5, ls=(0, (5, 2)), label=T("误用 $\\frac{1}{2}i^2\\,\\mathrm{d}L_{\\rm sec}/\\mathrm{d}g$",
                                                                      "misused $\\frac{1}{2}i^2\\,\\mathrm{d}L_{\\rm sec}/\\mathrm{d}g$"))
    f2 = FW.force_curves([2000.0])[0]
    a2.plot(2000, f2[0], "o", ms=5, mfc="white", mec=MAG, mew=1.4, zorder=5)
    a2.plot(2000, f2[1], "o", ms=5, mfc="white", mec=ELEC, mew=1.4, zorder=5)
    a2.text(2060, f2[0] + 25, f"{f2[0]:.0f} N", fontsize=8, color=MAG, ha="left", va="bottom")
    a2.text(2060, f2[1] + 25, f"{f2[1]:.0f} N", fontsize=8, color=ELEC, ha="left", va="bottom")
    a2.set_xlim(0, 3000); a2.set_ylim(0, 900)
    a2.set_xlabel(T("磁动势 $Ni$（A）", "MMF $Ni$ (A)"))
    a2.set_ylabel(T("气隙吸力（N）", "gap attraction (N)"))
    a2.legend(loc="upper left", fontsize=7.6, handlelength=1.9)
    clean_axes(a2)
    panel_title(a2, T("(b) 同一气隙力的三种算法", "(b) one gap force, three formulas"), size=9.0)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.17, top=0.9, wspace=0.3)
    return fig
