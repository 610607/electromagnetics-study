"""Figures for Ch 23 (integrated simulation projects).

project_ladder     the projects as a ladder; each rung names the chapters it combines and the check it must pass.
pns_sar_branches   one electric field, two read-out branches (nerve drive vs heating), with a computed
                   comparison: two longitudinal field patterns of equal mean square (equal SAR) give very different
                   membrane polarization; the cable is a spatial band-pass, heating is flat.

Computed quantities come from checks/check_ch23.py (which reuses the Ch 21 compartment cable).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from style import (ACCENT, AMBER, CURR, ELEC, FIG_W, INK, MAG, MUTED, NEUTRAL_T, POT, T, arrow, canvas, clean_axes,
                   figure, label, panel_title, plt, rbox, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch23 as C23  # noqa: E402

NERVE = ACCENT
HEAT = AMBER


# ---------------------------------------------------------------------------
@figure("project_ladder")
def project_ladder():
    W, H = 10.4, 6.75
    fig, ax = canvas(FIG_W, FIG_W * H / W, xlim=(0, W), ylim=(0, H))
    rungs = [
        ("0", MUTED, T("起点：两单元导电棒", "Start: two-element bar"),
         T("组装、边界反力、三种功率", "assembly, reactions, power three ways"), [16],
         T("三种算法的功率都是", "same power by"), T("0.4 mW", "all three routes")),
        ("1", ELEC, T("项目一：同一网格上的 TES 与 TMS", "P1: TES and TMS on one mesh"),
         T("同一个 $\\mathbf{K}$，不同的右端", "same $\\mathbf{K}$, different right-hand sides"), [4, 10, 11, 16],
         T("源线性、守恒、正交性；", "linearity, conservation;"), T("线圈源单独对照解析式", "check the source alone")),
        ("2", CURR, T("项目二：同一源基，多种读出", "P2: one source basis, many read-outs"),
         T("场、损耗、电压、受力、组织电场", "field, loss, voltage, force, tissue E"), [12, 18, 19, 20],
         T("每个读出的单位与归一化；", "units of every read-out;"), T("绕组化之后重新求场", "re-solve the wires")),
        ("3", MAG, T("项目三：先求偏置，再求小信号", "P3: bias first, then small signal"),
         T("非线性磁体与磁轭上的一致切线", "consistent tangent on magnet and yoke"), [13, 14, 15, 17],
         T("偏置先收敛；中心差分", "bias converged first;"), T("出现平台并与切线一致", "plateau = tangent")),
        ("4", HEAT, T("项目四：同一电场，PNS 与加热", "P4: one field, PNS and heating"),
         T("两个不同的读出泛函", "two different read-out functionals"), [11, 18, 21, 22],
         T("四个环节分别加密；", "refine 4 steps apart;"), T("兴奋判据与单调性；平均", "criterion; averages")),
        ("5", POT, T("项目五：先量尺子，再加射频", "P5: check the regime, then add RF"),
         T("需要时换成全波算子", "switch to the full-wave operator if needed"), [8, 22],
         T("三把尺子；端口归一化；", "three rulers; ports;"), T("功率平衡", "power balance")),
        ("", ACCENT, T("可复现的接口", "Reproducible interface"),
         T("单位、坐标与归一化跟着数组走", "units, frame, normalization travel with arrays"), [23, 24],
         T("别人能复现，", "others can reproduce"), T("也能追问", "and question it")),
    ]
    x0, x1 = 0.95, 10.25
    hc, gap = 0.7, 0.14
    ybase = 0.12
    # rails
    ax.plot([x0 - 0.12, x0 - 0.12], [ybase - 0.05, ybase + 7 * hc + 6 * gap + 0.05], color=tint(INK, 0.35), lw=2.2,
            solid_capstyle="round", zorder=1)
    ax.plot([x1 + 0.0, x1 + 0.0], [ybase - 0.05, ybase + 7 * hc + 6 * gap + 0.05], color=tint(INK, 0.35), lw=2.2,
            solid_capstyle="round", zorder=1)
    # column headers
    ytop = ybase + 7 * hc + 6 * gap
    label(ax, 1.55, ytop + 0.24, T("项目", "Project"), ha="left", size=8.2, color=MUTED, weight="bold")
    label(ax, 6.55, ytop + 0.24, T("用到的章", "Chapters"), size=8.2, color=MUTED, weight="bold")
    label(ax, 7.85, ytop + 0.24, T("必须通过的检查", "Checks that must pass"), ha="left", size=8.2, color=MUTED,
          weight="bold")
    for i, (num, col, title, sub, chs, chk1, chk2) in enumerate(rungs):
        y = ybase + i * (hc + gap)
        rbox(ax, x0, y, x1 - x0 - 0.12, hc, fc=tint(col, 0.08), ec=tint(col, 0.55), lw=0.9, r=0.1, z=2)
        cx, cy = x0 + 0.3, y + hc / 2
        ax.add_patch(plt.Circle((cx, cy), 0.2, fc=col, ec="none", zorder=4))
        if num:
            label(ax, cx, cy - 0.005, num, color="white", size=9.5, weight="bold", z=5)
        else:  # check mark for the interface rung
            ax.plot([cx - 0.09, cx - 0.02, cx + 0.1], [cy + 0.0, cy - 0.08, cy + 0.09], color="white", lw=1.8,
                    solid_capstyle="round", zorder=5)
        label(ax, x0 + 0.62, y + hc * 0.68, title, ha="left", size=T(8.9, 8.3), weight="bold", color=INK, z=5)
        label(ax, x0 + 0.62, y + hc * 0.27, sub, ha="left", size=T(7.8, 7.3), color=MUTED, z=5)
        # chapter pills
        pw, pg = 0.4, 0.06
        xs = 6.55 - (len(chs) * pw + (len(chs) - 1) * pg) / 2
        for k, c in enumerate(chs):
            xp = xs + k * (pw + pg)
            rbox(ax, xp, cy - 0.15, pw, 0.3, fc="white", ec=tint(col, 0.75), lw=0.8, r=0.08, z=4)
            label(ax, xp + pw / 2, cy, str(c), size=7.8, color=INK, z=5)
        label(ax, 7.85, y + hc * 0.68, chk1, ha="left", size=T(7.8, 7.2), color=INK, z=5)
        label(ax, 7.85, y + hc * 0.29, chk2, ha="left", size=T(7.8, 7.2), color=INK, z=5)
    # rising arrow on the left
    arrow(ax, (0.3, ybase + 0.1), (0.3, ytop - 0.05), color=MUTED, lw=1.4, ms=10)
    label(ax, 0.12, 0.5 * (ybase + ytop), T("同一份模型约定 · 由浅入深", "one model contract · growing scope"),
          rotation=90, size=T(8.0, 7.6), color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("pns_sar_branches")
def pns_sar_branches():
    FH = 4.6
    fig = plt.figure(figsize=(FIG_W, FH))
    # ---------------- (a) the two branches -------------------------------------------------
    ax = fig.add_axes([0.0, 0.5, 1.0, 0.5])
    W = 10.4
    Hh = W * (0.5 * FH) / FIG_W
    ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.set_aspect("equal"); ax.axis("off")
    top_y = Hh - 0.85
    rbox(ax, 2.75, top_y, 4.9, 0.78, fc=tint(ELEC, 0.1), ec=tint(ELEC, 0.6), lw=1.0, r=0.1)
    label(ax, 0.15, top_y + 0.56, T("(a) 两个分支", "(a) two branches"), size=9.2, weight="bold", ha="left")
    label(ax, 5.2, top_y + 0.56, T("同一个总电场（一次场解）", "one total electric field (one field solution)"),
          size=T(9.0, 8.5), weight="bold", color=ELEC)
    label(ax, 5.2, top_y + 0.22,
          r"$\mathbf{E}(\mathbf{r},t)=\dot I_1(t)\,\mathbf{e}_1(\mathbf{r})+\dot I_2(t)\,\mathbf{e}_2(\mathbf{r})+\cdots$",
          size=9.0)
    branches = [
        (0.15, NERVE, T("神经驱动", "nerve drive"),
         [T("投影到路径：$E_s=\\mathbf{E}\\cdot\\mathbf{t}$", "project on the path: $E_s=\\mathbf{E}\\cdot\\mathbf{t}$"),
          T("路径积分：$\\Delta V_e^{\\rm eff}=-\\int\\mathbf{E}\\cdot\\mathbf{t}\\,\\mathrm{d}s$",
            "path integral: $\\Delta V_e^{\\rm eff}=-\\int\\mathbf{E}\\cdot\\mathbf{t}\\,\\mathrm{d}s$"),
          T("电缆与膜动力学 → 阈值", "cable, membrane → threshold")],
         T("对源线性（被动时）；看沿路径的变化", "linear in the source (passive); sees variation")),
        (5.35, HEAT, T("加热", "heating"),
         [T("损耗密度：$p=\\mathbf{E}^{\\mathsf{T}}\\boldsymbol{\\sigma}\\mathbf{E}$",
            "loss density: $p=\\mathbf{E}^{\\mathsf{T}}\\boldsymbol{\\sigma}\\mathbf{E}$"),
          T("质量平均、时间平均 → SAR", "mass and time averages → SAR"),
          T("再加热模型，才是温度", "a thermal model gives temperature")],
         T("对源二次；不管方向与空间变化", "quadratic; blind to direction, variation")),
    ]
    bw = 4.9
    for x, col, title, steps, note in branches:
        yt = top_y - 0.72
        rbox(ax, x, yt, bw, 0.42, fc=tint(col, 0.85), ec="none", r=0.1)
        label(ax, x + bw / 2, yt + 0.21, title, color="white", weight="bold", size=9.0)
        arrow(ax, (5.2 + (-1.0 if x < 3 else 1.0), top_y - 0.02), (x + bw / 2, yt + 0.44), color=tint(col, 0.9),
              lw=1.3, ms=9)
        y = yt - 0.12
        for k, s in enumerate(steps):
            y -= 0.5
            rbox(ax, x + 0.15, y, bw - 0.3, 0.42, fc=tint(col, 0.08), ec=tint(col, 0.5), lw=0.8, r=0.08)
            label(ax, x + bw / 2, y + 0.21, s, size=T(8.2, 7.8))
            if k:
                arrow(ax, (x + bw / 2, y + 0.5), (x + bw / 2, y + 0.43), color=tint(col, 0.9), lw=1.0, ms=7)
        label(ax, x + bw / 2, y - 0.25, note, size=T(7.8, 7.4), color=col)
    # ---------------- (b) two patterns with the same mean square -------------------------
    axE = fig.add_axes([0.085, 0.3, 0.40, 0.13])
    axV = fig.add_axes([0.085, 0.095, 0.40, 0.185], sharex=axE)
    s, Eu, Vu = C23.cable_response("uniform", ds=0.05)
    s2, Ec, Vc = C23.cable_response("cos", ds=0.05)
    m = np.abs(s) <= 8
    m2 = np.abs(s2) <= 8
    axE.plot(s[m], Eu[m], color=MUTED, lw=1.5, ls=(0, (4, 2)))
    axE.plot(s2[m2], Ec[m2], color=ELEC, lw=1.6)
    axE.set_ylim(-1.7, 1.9); axE.set_yticks([-1, 0, 1], ["−1", "0", "1"])
    axE.set_ylabel(r"$E_s/E_{\rm rms}$", fontsize=8.4)
    plt.setp(axE.get_xticklabels(), visible=False)
    clean_axes(axE)
    panel_title(axE, T("(b) 均方值相同的两种纵向场：SAR 相同", "(b) two fields, same mean square: same SAR"),
                size=8.8)
    axV.plot(s[m], Vu[m], color=MUTED, lw=1.5, ls=(0, (4, 2)))
    axV.plot(s2[m2], Vc[m2], color=NERVE, lw=1.7)
    axV.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)
    axV.set_ylim(-0.95, 1.3); axV.set_yticks([-0.71, 0, 0.71], ["−0.71", "0", "0.71"])
    axV.set_xlim(-8, 8)
    axV.set_ylabel(r"$V_m/(E_{\rm rms}\lambda)$", fontsize=8.4)
    axV.set_xlabel(T("沿轴突的位置 $s/\\lambda$", "position along the axon $s/\\lambda$"), fontsize=8.6)
    axV.text(0.0, 1.27, T("虚线：均匀场，膜电压在内部为零", "dashed: uniform field, zero polarization inside"),
             fontsize=7.6, color=MUTED, ha="center", va="top")
    clean_axes(axV)
    # ---------------- (c) band-pass vs flat ------------------------------------------------
    axc = fig.add_axes([0.6, 0.095, 0.385, 0.335])
    k = np.logspace(-2, 2, 400)
    nerve = C23.bandpass(k) / C23.bandpass(1.0)
    axc.semilogx(k, nerve, color=NERVE, lw=1.9, label=T("膜电压幅值", "membrane polarization"))
    axc.semilogx(k, np.ones_like(k), color=HEAT, lw=1.9, ls=(0, (5, 2)), label=T("SAR", "SAR"))
    axc.plot([1.0], [1.0], "o", ms=5, mfc="white", mec=NERVE, mew=1.4, zorder=5)
    axc.set_ylim(0, 1.4); axc.set_xlim(0.01, 100)
    axc.set_xlabel(T("场沿轴突变化的空间频率 $k\\lambda$", "spatial frequency along the axon $k\\lambda$"),
                   fontsize=8.6)
    axc.set_ylabel(T("以 $k\\lambda=1$ 的值归一化", "relative to $k\\lambda=1$"), fontsize=8.4)
    axc.legend(loc="upper right", fontsize=7.8, handlelength=1.8, ncol=2, bbox_to_anchor=(1.0, 1.02),
               columnspacing=1.0)
    axc.text(0.85, 1.04, T("最强：$k\\lambda=1$", "peak at $k\\lambda=1$"), fontsize=7.8, color=NERVE, ha="right")
    clean_axes(axc)
    panel_title(axc, T("(c) 电缆是空间带通，加热不是", "(c) cable: spatial band-pass; heat: flat"),
                size=8.8)
    return fig


LIGHT_GREY = "#C9CED3"
