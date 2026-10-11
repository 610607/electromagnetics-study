"""Figures for Ch 24 (verification, uncertainty and defensible decisions).

verification_validation  real system -> conceptual model -> equations -> discrete model -> code -> numbers;
                         verification acts on the lower links, validation compares numbers with measurement.
error_budget             one observable of the running example (mean |E| in the inclusion), TES and TMS:
                         relative change for each numerical and physical perturbation (computed).

Computed quantities come from checks/check_ch24.py, so text, figure and assertions share one model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from style import (ACCENT, AMBER, ELEC, FIG_W, INK, MAG, MATL, MUTED, NEUTRAL_T, T, arrow, canvas, clean_axes,
                   figure, label, panel_title, plt, rbox, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch24 as C24  # noqa: E402

VER = ACCENT      # verification (numerical)
VAL = AMBER       # validation (physical)


# ---------------------------------------------------------------------------
@figure("verification_validation")
def verification_validation():
    W, H = 10.4, 5.72
    fig, ax = canvas(FIG_W, FIG_W * H / W, xlim=(0, W), ylim=(0, H))
    x0, x1 = 3.35, 7.05
    bh, gap = 0.66, 0.30
    top = H - 0.1
    rows = [
        (T("真实系统", "Real system"), T("线圈、组织、磁体与磁轭", "coil, tissue, magnet, yoke"), tint("#C9B79C", 0.35)),
        (T("概念模型", "Conceptual model"), T("保留哪些物理：近似、几何、材料", "kept physics: regime, geometry"), NEUTRAL_T),
        (T("数学模型", "Mathematical model"), T("方程、边界条件、源与本构", "PDEs, boundaries, sources"), NEUTRAL_T),
        (T("离散模型", "Discrete model"), T("网格、单元、矩阵方程", "mesh, elements, matrices"), NEUTRAL_T),
        (T("程序", "Code"), T("实现与求解器", "implementation, solver"), NEUTRAL_T),
        (T("数值结果", "Numbers"), T("场与观测量", "fields and observables"), NEUTRAL_T),
    ]
    ys = []
    for i, (title, sub, fc) in enumerate(rows):
        y = top - (i + 1) * bh - i * gap
        ys.append(y)
        rbox(ax, x0, y, x1 - x0, bh, fc=fc, ec=tint(INK, 0.35), lw=0.8, r=0.1)
        label(ax, x0 + 0.16, y + bh * 0.66, title, ha="left", weight="bold", size=T(9.4, 8.8))
        label(ax, x0 + 0.16, y + bh * 0.28, sub, ha="left", size=T(8.0, 7.6), color=MUTED)
    steps = [T("理想化", "idealize"), T("写成方程", "write equations"), T("离散化", "discretize"),
             T("编程", "implement"), T("求解", "solve")]
    xc = 0.5 * (x0 + x1)
    for i, s in enumerate(steps):
        ya, yb = ys[i], ys[i + 1] + bh
        arrow(ax, (xc, ya - 0.02), (xc, yb + 0.02), color=INK, lw=1.1, ms=8)
        label(ax, xc + 0.12, 0.5 * (ya + yb), s, ha="left", size=T(7.6, 7.2), color=MUTED, style="italic")
    # --- verification bracket on the right: mathematical model ... numbers
    xb = x1 + 0.22
    ytop, ybot = ys[2] + bh, ys[5]
    ax.plot([xb, xb + 0.14, xb + 0.14, xb], [ytop, ytop, ybot, ybot], color=VER, lw=1.6, solid_capstyle="butt")
    tx = xb + 0.32
    ym = 0.5 * (ytop + ybot)
    label(ax, tx, ym + 0.95, T("数值验证（verification）", "Verification"), ha="left", weight="bold", color=VER,
          size=T(9.4, 9.0))
    label(ax, tx, ym + 0.55, T("方程解对了吗？", "Are the equations solved right?"), ha="left", color=VER,
          size=T(8.8, 8.0))
    label(ax, tx, ym - 0.05, T("代码验证：制造解、收敛阶", "code: manufactured"), ha="left", size=T(8.0, 7.6))
    label(ax, tx, ym - 0.37, T("", "solutions, orders"), ha="left", size=T(8.0, 7.6))
    label(ax, tx, T(ym - 0.45, ym - 0.75), T("解的验证：网格序列、", "solution: mesh series,"), ha="left",
          size=T(8.0, 7.6))
    label(ax, tx, T(ym - 0.77, ym - 1.07), T("残差、守恒检查", "residuals, conservation"), ha="left",
          size=T(8.0, 7.6))
    # --- modelling assumptions bracket (top two links): invisible to verification
    ytop2, ybot2 = ys[0] + bh, ys[2] + bh + 0.06
    ax.plot([xb, xb + 0.14, xb + 0.14, xb], [ytop2, ytop2, ybot2, ybot2], color=VAL, lw=1.6, ls=(0, (3, 2)),
            solid_capstyle="butt")
    label(ax, tx, ytop2 - 0.30, T("建模假设：", "Modelling assumptions:"), ha="left", color=VAL, weight="bold",
          size=T(8.6, 8.2))
    label(ax, tx, ytop2 - 0.62, T("数值验证看不见，", "invisible to verification;"), ha="left", size=T(8.0, 7.6))
    label(ax, tx, ytop2 - 0.94, T("只能由确认来检验", "only validation tests them"), ha="left", size=T(8.0, 7.6))
    # --- validation on the left: numbers <-> measurement of the real system
    mx0, mx1 = 0.12, 2.95
    my = ys[0]
    rbox(ax, mx0, my, mx1 - mx0, bh, fc=tint(VAL, 0.14), ec=tint(VAL, 0.7), lw=0.9, r=0.1)
    label(ax, mx0 + 0.16, my + bh * 0.66, T("测量", "Measurement"), ha="left", weight="bold", size=T(9.4, 8.8),
          color=INK)
    label(ax, mx0 + 0.16, my + bh * 0.28, T("探头、阻抗、场图、阈值", "probe, impedance, maps"), ha="left",
          size=T(8.0, 7.4), color=MUTED)
    arrow(ax, (x0 - 0.04, my + bh / 2), (mx1 + 0.04, my + bh / 2), color=VAL, lw=1.1, ms=8, style="-|>")
    xl = 0.32
    yn = ys[5] + bh / 2
    ax.plot([x0 - 0.04, xl, xl], [yn, yn, my - 0.06], color=VAL, lw=1.8, solid_capstyle="round")
    arrow(ax, (xl, my - 0.3), (xl, my - 0.02), color=VAL, lw=1.8, ms=10)
    arrow(ax, (x0 - 0.34, yn), (x0 - 0.02, yn), color=VAL, lw=1.8, ms=10)
    vx = xl + 0.2
    vy = 0.5 * (my + yn)
    label(ax, vx, vy + 0.62, T("物理确认（validation）", "Validation"), ha="left", weight="bold", color=VAL,
          size=T(9.4, 9.0))
    label(ax, vx, vy + 0.22, T("方程选对了吗？", "Are these the right"), ha="left", color=VAL, size=T(8.8, 8.0))
    label(ax, vx, vy - 0.08, T("", "equations?"), ha="left", color=VAL, size=T(8.8, 8.0))
    label(ax, vx, T(vy - 0.25, vy - 0.48), T("用同一个观测量，", "compare the same"), ha="left", size=T(8.0, 7.6))
    label(ax, vx, T(vy - 0.57, vy - 0.8), T("比较数值结果与测量", "observable with data"), ha="left", size=T(8.0, 7.6))
    return fig


# ---------------------------------------------------------------------------
@figure("error_budget")
def error_budget():
    b = C24.budget()
    rows = [
        (T("离散：网格 64 → 128", "mesh 64 → 128"), b["tes_mesh_64_128"], b["tms_mesh_64_128"], "num"),
        (T("离散：网格 128 → 256", "mesh 128 → 256"), b["tes_mesh_128_256"], b["tms_mesh_128_256"], "num"),
        (T("代数：CG 容差 $10^{-3}$", "solver tol. $10^{-3}$"), b["cg_rel"][1e-3], b["tms_cg_rel"][1e-3], "num"),
        (T("几何：夹杂半径 ±0.01", "radius ±0.01"), max(map(abs, b["tes_R"])), max(map(abs, b["tms_R"])), "phys"),
        (T("材料：夹杂电导率 ±20%", "inclusion σ ±20%"), max(map(abs, b["tes_sig"])), max(map(abs, b["tms_sig"])),
         "phys"),
        (T("电极：等势 vs 均匀", "electrode model"), abs(b["tes_electrode"]["equipotential"]), np.nan, "phys"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 2.75), sharey=True)
    yy = np.arange(len(rows))[::-1]
    for k, (ax, title) in enumerate(zip(axs, (T("TES：恒流电极", "TES: fixed current"),
                                              T("TMS：均匀 d$B$/d$t$", "TMS: uniform d$B$/d$t$")))):
        for y, (name, vt, vm, kind) in zip(yy, rows):
            v = vt if k == 0 else vm
            col = VER if kind == "num" else VAL
            if np.isnan(v):
                ax.text(1.2e-5, y, T("不适用", "n/a"), va="center", ha="left", fontsize=7.8, color=MUTED)
                continue
            ax.barh(y, v, left=1e-5, height=0.58, color=tint(col, 0.55), edgecolor=col, lw=0.8)
            pct = 100 * v
            s = f"{pct:.1f}%" if pct >= 1 else (f"{pct:.2f}%" if pct >= 0.1 else f"{pct:.1g}%")
            ax.text(v * (1.35 if v > 1e-4 else 1.9), y, s, va="center", ha="left", fontsize=7.8, color=INK,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=6)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 3.0)
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ["0.01%", "0.1%", "1%", "10%", "100%"])
        ax.axvline(0.01, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
        clean_axes(ax)
        ax.tick_params(axis="y", length=0)
        panel_title(ax, title, size=9.2, color=ELEC if k == 0 else MAG)   # TES: electric source; TMS: magnetic source
        ax.set_xlabel(T("观测量的相对变化", "relative change of the observable"), fontsize=8.4)
    axs[0].set_yticks(yy, [r[0] for r in rows], fontsize=T(8.2, 7.8))
    # legend
    from matplotlib.patches import Patch
    h = [Patch(fc=tint(VER, 0.55), ec=VER, lw=0.8, label=T("数值误差", "numerical")),
         Patch(fc=tint(VAL, 0.55), ec=VAL, lw=0.8, label=T("模型输入的不确定性", "model input"))]
    axs[1].legend(handles=h, loc="lower right", fontsize=7.6, handlelength=1.4, frameon=True, framealpha=0.95,
                  edgecolor="none")
    fig.subplots_adjust(left=T(0.25, 0.21), right=0.985, bottom=0.17, top=0.9, wspace=0.08)
    return fig
