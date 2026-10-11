"""Figures for Ch 20 (inverse coil design).

inverse_design_flow  from a Maxwell-compatible target to a verified winding (diagram).
tradeoff_curve       biplanar x-gradient teaching example: field error versus power and stored energy for
                     three penalties, and the winding contours of two designs with the same error (computed).

The design model and every number come from checks/check_ch20.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from style import (ACCENT, AMBER, CURR, FIG_W, INK, MAG, MUTED, POT, T, WARN, arrow, canvas, clean_axes,
                   figure, label, panel_title, plt, rbox, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch20 as CD  # noqa: E402


# ---------------------------------------------------------------------------
def _step(ax, x, y, w, h, num, title, lines, col, size=None):
    size = size or T(8.1, 7.7)
    rbox(ax, x, y, w, h, fc=tint(col, 0.07), ec=tint(col, 0.55), lw=0.9, r=0.1)
    rbox(ax, x, y + h - 0.4, w, 0.4, fc=tint(col, 0.85), ec="none", r=0.1, z=2.2)
    ax.add_patch(plt.Rectangle((x, y + h - 0.4), w, 0.2, fc=tint(col, 0.85), ec="none", zorder=2.2))
    label(ax, x + 0.13, y + h - 0.2, num, color="white", size=9.0, weight="bold", ha="left")
    label(ax, x + 0.42, y + h - 0.2, title, color="white", size=8.8, weight="bold", ha="left")
    for k, s in enumerate(lines):
        label(ax, x + 0.14, y + h - 0.62 - 0.3 * k, s, size=size, ha="left")


@figure("inverse_design_flow")
def inverse_design_flow():
    W, H = 10.4, 4.97
    fig, ax = canvas(FIG_W, FIG_W * H / W, xlim=(0, W), ylim=(0, H))
    # forward vs inverse strip
    rbox(ax, 0.08, H - 0.62, 5.0, 0.5, fc=tint(MUTED, 0.1), ec="none", r=0.1)
    label(ax, 2.58, H - 0.37, T("前向：给定源 → 场（解唯一）", "forward: source → field (unique)"), size=8.4, color=INK)
    rbox(ax, 5.32, H - 0.62, 5.0, 0.5, fc=tint(ACCENT, 0.85), ec="none", r=0.1)
    label(ax, 7.82, H - 0.37, T("逆向：想要的场 → 源（不唯一，要选择）", "inverse: wanted field → source (not unique)"),
          size=8.4, color="white", weight="bold")
    w, h = 3.25, 1.72
    xs = [0.08, 3.575, 7.07]
    y1 = H - 0.86 - h
    y2 = y1 - 0.56 - h
    _step(ax, xs[0], y1, w, h, "1", T("目标与约束", "target and limits"), [
        T("ROI 内的场必须满足 Maxwell：", "ROI field must obey Maxwell:"),
        T("用调和基或可实现的场写目标", "write it in a harmonic basis"),
        T("屏蔽：外部区域的场 ≈ 0", "shielding: exterior field ≈ 0"),
        T("功率、电压、力、线距的上限", "limits: power, voltage, force, pitch")], MAG)
    _step(ax, xs[1], y1, w, h, "2", T("源空间", "source space"), [
        T("载体曲面（板、圆柱）", "former surface (plate, cylinder)"),
        T(r"$\psi_s=c_1N_1+c_2N_2+\cdots$", r"$\psi_s=c_1N_1+c_2N_2+\cdots$"),
        T("电流守恒自动成立", "current continuity is built in"),
        T("边界上 $\\psi_s$ 为常数：不流出", "$\\psi_s$ constant on edges: no leakage")], CURR)
    _step(ax, xs[2], y1, w, h, "3", T("一致的算子", "consistent operators"), [
        T(r"场：$\mathbf{b}=\mathbf{C}\,\mathbf{c}$", r"field: $\mathbf{b}=\mathbf{C}\,\mathbf{c}$"),
        T(r"功率：$P=\mathbf{c}^{\mathsf{T}}\mathbf{R}\,\mathbf{c}$", r"power: $P=\mathbf{c}^{\mathsf{T}}\mathbf{R}\,\mathbf{c}$"),
        T(r"储能：$W=\frac{1}{2}\mathbf{c}^{\mathsf{T}}\mathbf{L}\,\mathbf{c}$", r"energy: $W=\frac{1}{2}\mathbf{c}^{\mathsf{T}}\mathbf{L}\,\mathbf{c}$"),
        T("同一组基函数，同一套单位", "one basis, one set of units")], ACCENT)
    _step(ax, xs[2], y2, w, h, "4", T("求解与权衡", "solve, trade off"), [
        T("正则化：误差 + α·功率 + β·储能", "regularise: error + α·P + β·W"),
        T("或：精确约束 + 最小功率", "or: exact constraints, least power"),
        T("扫描 α，画出权衡曲线", "sweep α, draw the trade-off"),
        T("看奇异值：哪些场模式最贵", "singular values: costly modes")], AMBER)
    _step(ax, xs[1], y2, w, h, "5", T("提取绕组", "extract the winding"), [
        T(r"取等值线 $\Delta\psi_s=I_w$", r"contours at $\Delta\psi_s=I_w$"),
        T("加引线、串联、跨线", "leads, series links, crossovers"),
        T("检查线距、线宽、弯曲半径", "check pitch, width, bend radius"),
        T("PCB：铜厚与层间连接", "PCB: copper weight, vias")], CURR)
    _step(ax, xs[0], y2, w, h, "6", T("独立前向复核", "forward re-check"), [
        T("对实际导线重算场、R、L", "recompute field, R, L of the wires"),
        T("邻近导体中的涡流（第 12 章）", "eddy currents nearby (Ch. 12)"),
        T("受力（第 15 章）、PNS（第 21 章）", "forces (Ch. 15), PNS (Ch. 21)"),
        T("与测量对照", "compare with measurement")], WARN)
    ym1 = y1 + h / 2; ym2 = y2 + h / 2
    arrow(ax, (xs[0] + w + 0.03, ym1), (xs[1] - 0.03, ym1), color=INK, lw=1.3, ms=10)
    arrow(ax, (xs[1] + w + 0.03, ym1), (xs[2] - 0.03, ym1), color=INK, lw=1.3, ms=10)
    arrow(ax, (xs[2] + w / 2, y1 - 0.03), (xs[2] + w / 2, y2 + h + 0.03), color=INK, lw=1.3, ms=10)
    arrow(ax, (xs[2] - 0.03, ym2), (xs[1] + w + 0.03, ym2), color=INK, lw=1.3, ms=10)
    arrow(ax, (xs[1] - 0.03, ym2), (xs[0] + w + 0.03, ym2), color=INK, lw=1.3, ms=10)
    # feedback loop: from the check back to the target and the source space
    arrow(ax, (xs[0] + 0.55, y2 + h + 0.03), (xs[0] + 0.55, y1 - 0.03), color=WARN, lw=1.2, ms=9, ls=(0, (4, 2)))
    label(ax, xs[0] + 0.72, (y1 + y2 + h) / 2, T("不满足：回到第 1、2 步，改目标、源空间或约束",
                                                 "fails: back to steps 1–2, change target, source space or limits"),
          size=7.9, color=WARN, ha="left", va="center")
    return fig


# ---------------------------------------------------------------------------
def _contours(ax, c, title):
    x = np.linspace(-CD.A_HALF, CD.A_HALF, 301)
    X, Y = np.meshgrid(x, x)
    psi = CD.psi_grid(c, x).T
    Kx, Ky = CD.sheet_current(c, x)
    Kx, Ky = Kx.T, Ky.T
    top = int(np.ceil(np.abs(psi).max() / CD.I_WIRE))
    lev = (np.arange(top) + 0.5) * CD.I_WIRE
    lev = np.concatenate([-lev[::-1], lev])
    ax.add_patch(plt.Rectangle((-CD.A_HALF, -CD.A_HALF), 2 * CD.A_HALF, 2 * CD.A_HALF, fc=tint(AMBER, 0.08),
                               ec=tint(MUTED, 0.7), lw=0.8, zorder=0))
    cs = ax.contour(X, Y, psi, levels=lev, colors=[CURR], linewidths=0.85, zorder=3, negative_linestyles="solid")
    from scipy.interpolate import RegularGridInterpolator
    fx = RegularGridInterpolator((x, x), Kx, bounds_error=False, fill_value=0.0)
    fy = RegularGridInterpolator((x, x), Ky, bounds_error=False, fill_value=0.0)
    for seg in cs.allsegs:
        for path in seg:
            if len(path) < 20 or np.ptp(path[:, 1]) < 0.05:
                continue
            k = int(np.argmax(path[:, 1]))
            p0 = path[k]
            v = np.array([fx((p0[1], p0[0]))[()], fy((p0[1], p0[0]))[()]])
            if np.hypot(*v) == 0:
                continue
            v = v / np.hypot(*v) * 0.012
            arrow(ax, p0 - v, p0 + v, color=CURR, lw=0.85, ms=6, z=4)
    ax.add_patch(plt.Circle((0, 0), CD.R_ROI, fc="none", ec=MAG, lw=1.0, ls=(0, (3, 2)), zorder=4))
    ax.set_xlim(-0.32, 0.32); ax.set_ylim(-0.32, 0.32); ax.set_aspect("equal")
    ax.set_xticks([-0.3, 0, 0.3]); ax.set_yticks([-0.3, 0, 0.3])
    ax.tick_params(labelsize=7.6, length=2.5)
    ax.set_xlabel(r"$x$ (m)", labelpad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    panel_title(ax, title, size=8.6)


@figure("tradeoff_curve")
def tradeoff_curve():
    fig = plt.figure(figsize=(FIG_W, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.42, wspace=0.3, left=0.095, right=0.985,
                          bottom=0.07, top=0.95)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1], sharey=a1)
    a3 = fig.add_subplot(gs[1, 0]); a4 = fig.add_subplot(gs[1, 1])
    styles = {"R": (ACCENT, "-", 1.8, T("罚功率 $\\mathbf{c}^{\\mathsf{T}}\\mathbf{R}\\mathbf{c}$", "power penalty $\\mathbf{c}^{\\mathsf{T}}\\mathbf{R}\\mathbf{c}$")),
              "L": (POT, (0, (5, 2)), 1.5, T("罚储能 $\\mathbf{c}^{\\mathsf{T}}\\mathbf{L}\\mathbf{c}$", "energy penalty $\\mathbf{c}^{\\mathsf{T}}\\mathbf{L}\\mathbf{c}$")),
              "I": (MUTED, (0, (1.5, 1.5)), 1.6, T("罚系数 $|\\mathbf{c}|^2$", "coefficient penalty $|\\mathbf{c}|^2$"))}
    for nm in ("R", "L", "I"):
        f = CD.family(nm, 61, -13.0, 1.5)
        col, ls, lw, lab = styles[nm]
        a1.loglog(f[:, 2], 100 * f[:, 1], color=col, ls=ls, lw=lw, label=lab)
        a2.loglog(f[:, 3], 100 * f[:, 1], color=col, ls=ls, lw=lw)
        c, _ = CD.design_at(CD.TARGET_ERR, nm)
        mm = CD.metrics(c)
        a1.plot(mm["P"], 100 * mm["err"], "o", ms=4.5, mfc="white", mec=col, mew=1.3, zorder=5)
        a2.plot(mm["W"], 100 * mm["err"], "o", ms=4.5, mfc="white", mec=col, mew=1.3, zorder=5)
    for nf in (96, 160):
        c, p, _, _ = CD.exact_constrained(nf)
        mm = CD.metrics(c)
        a1.plot(mm["P"], 100 * mm["err"], "D", ms=4.5, mfc=tint(WARN, 0.25), mec=WARN, mew=1.1, zorder=5)
        a1.text(mm["P"] * 1.12, 100 * mm["err"], T(f"{len(p)} 点精确约束", f"{len(p)} exact points"), fontsize=7.4,
                color=WARN, ha="left", va="center")
    a1.set_xlim(120, 2.0e4); a1.set_ylim(0.05, 40)
    a1.set_xlabel(T("耗散功率 $P$（W）", "dissipated power $P$ (W)"), labelpad=1)
    a1.set_ylabel(T("场误差（RMS，%）", "field error (RMS, %)"), labelpad=1)
    a1.legend(loc="upper right", fontsize=7.3, handlelength=2.0, borderaxespad=0.2)
    a1.tick_params(labelsize=7.6, length=2.5)
    clean_axes(a1)
    panel_title(a1, T("(a) 误差与功率", "(a) error versus power"), size=8.8)
    a2.set_xlim(0.18, 5.0)
    from matplotlib.ticker import FixedLocator, NullFormatter
    a2.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5]))
    a2.xaxis.set_minor_formatter(NullFormatter())
    a2.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    a2.set_xlabel(T("10 mT/m 时的储能 $W$（J）", "stored energy $W$ at 10 mT/m (J)"), labelpad=1)
    a2.tick_params(labelsize=7.6, length=2.5)
    plt.setp(a2.get_yticklabels(), visible=False)
    clean_axes(a2)
    panel_title(a2, T("(b) 误差与储能", "(b) error versus stored energy"), size=8.8)
    cR, _ = CD.design_at(CD.TARGET_ERR, "R"); cI, _ = CD.design_at(CD.TARGET_ERR, "I")
    mR, mI = CD.metrics(cR), CD.metrics(cI)
    _contours(a3, cR, T(f"(c) 罚功率，误差 1%：P = {mR['P']:.0f} W", f"(c) power penalty, 1 %: P = {mR['P']:.0f} W"))
    _contours(a4, cI, T(f"(d) 罚系数，误差 1%：P = {mI['P']:.0f} W", f"(d) coefficient penalty, 1 %: P = {mI['P']:.0f} W"))
    a3.set_ylabel(r"$y$ (m)", labelpad=0)
    a4.set_yticklabels([])
    return fig
