"""Figures for Ch 13 (ferromagnetic materials: nonlinearity and hysteresis).

Every curve is computed from the models in checks/check_ch13.py (which use numerics.py), so the
figures, the text and the assertions share one implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Arc, Ellipse, Polygon

from style import (ACCENT, FIG_W, INK, LIGHT, MAG, MUTED, POT, T, WARN, arrow, clean_axes, figure, halo, label,
                   panel_title, plt, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch13 as BH  # noqa: E402
from numerics import MU0  # noqa: E402

DARK_MAG = "#8A3A00"


def _ptitle(ax, s):
    ax.set_title(s, fontsize=9.0, fontweight="bold", pad=4, color=INK)


# ---------------------------------------------------------------------------
@figure("permeabilities")
def permeabilities():
    t = BH.three_permeabilities()
    H0, B0 = BH.H0, t["B0"]
    ms, md = t["mur_sec"] * MU0, t["mur_diff"] * MU0
    fig = plt.figure(figsize=(FIG_W, 2.55))
    a1 = fig.add_axes([0.075, 0.18, 0.27, 0.70])
    a2 = fig.add_axes([0.435, 0.18, 0.235, 0.70])
    a3 = fig.add_axes([0.70, 0.02, 0.30, 0.92])

    # (a) curve, secant, tangent
    H = np.linspace(0, 4200, 600)
    a1.plot(H, BH.B_of_H(H), color=MAG, lw=2.0, zorder=3)
    Hs = np.linspace(0, 2350, 10)
    a1.plot(Hs, ms * Hs, color=ACCENT, lw=1.2, ls=(0, (4, 2)), zorder=2)
    Ht = np.linspace(700, 4200, 10)
    a1.plot(Ht, B0 + md * (Ht - H0), color=DARK_MAG, lw=1.2, ls=(0, (1.2, 1.4)), zorder=2)
    a1.plot([H0], [B0], "o", ms=5.5, mfc="white", mec=MAG, mew=1.5, zorder=5)
    a1.text(H0 - 150, B0 + 0.2, T("工作点", "operating\npoint"), fontsize=8.0, color=INK, ha="right", va="center")
    a1.text(1950, 0.55, T("割线：$\\mu_r=%.0f$" % t["mur_sec"], "secant: $\\mu_r=%.0f$" % t["mur_sec"]),
            fontsize=8.2, color=ACCENT, ha="left", va="center")
    a1.text(1950, 1.1, T("切线：$\\mu_r=%.0f$" % t["mur_diff"], "tangent: $\\mu_r=%.0f$" % t["mur_diff"]),
            fontsize=8.2, color=DARK_MAG, ha="left", va="center")
    a1.set_xlim(0, 4200); a1.set_ylim(0, 2.0)
    a1.set_xticks([0, 1600, 3200])
    a1.set_xlabel(r"$H$ (A/m)")
    a1.set_ylabel(r"$B$ (T)")
    clean_axes(a1)
    _ptitle(a1, T("(a) 一条曲线，两种斜率", "(a) one curve, two slopes"))

    # (b) the two permeabilities versus H
    H = np.linspace(0, 6000, 800)
    a2.semilogy(H, BH.mu_sec(H) / MU0, color=ACCENT, lw=1.5, ls=(0, (4, 2)), label=T("割线", "secant"))
    a2.semilogy(H, BH.mu_diff(H) / MU0, color=DARK_MAG, lw=1.6, label=T("微分", "differential"))
    a2.axhline(1, color=MUTED, lw=0.8, ls=(0, (1.5, 1.5)))
    a2.text(5900, 1.4, T("真空 $\\mu_r=1$", "vacuum $\\mu_r=1$"), fontsize=7.8, color=MUTED, ha="right", va="bottom")
    a2.axvline(H0, color=LIGHT, lw=3, zorder=0)
    a2.plot([H0, H0], [t["mur_sec"], t["mur_diff"]], "o", ms=4, mfc="white", mec=INK, mew=1.0, zorder=5)
    a2.set_xlim(0, 6000); a2.set_ylim(0.6, 4000)
    a2.set_xticks([0, 2000, 4000, 6000])
    a2.set_xlabel(r"$H$ (A/m)")
    a2.set_ylabel(T(r"相对磁导率 $\mu_r$", r"relative permeability $\mu_r$"))
    a2.legend(loc="center right", fontsize=7.8, handlelength=1.8, borderaxespad=0.2)
    clean_axes(a2)
    _ptitle(a2, T("(b) 两者随工作点分开", "(b) they part with bias"))

    # (c) incremental response at the operating point
    ax = a3
    ax.set_xlim(-0.05, 3.5); ax.set_ylim(-1.62, 1.55); ax.set_aspect("equal"); ax.axis("off")
    O = np.array([0.0, -0.12])
    R = 1.9
    tip = O + np.array([R, 0.0])
    # circle |B| = B0 through the tip: a transverse step moves along it
    ax.add_patch(Arc(tuple(O), 2 * R, 2 * R, theta1=-24, theta2=42, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1))
    ang = np.deg2rad(42)
    label(ax, O[0] + R * np.cos(ang) - 0.06, O[1] + R * np.sin(ang), T("$|\\mathbf{B}|=B_0$", "$|\\mathbf{B}|=B_0$"),
          size=8.0, color=MUTED, ha="right", va="center")
    arrow(ax, tuple(O), tuple(tip), color=MAG, lw=2.6, ms=13, z=4)
    label(ax, O[0] + 0.78, O[1] + 0.2, T("偏置 $\\mathbf{B}_0$", "bias $\\mathbf{B}_0$"), size=8.6, color=MAG)
    # response ellipse: delta B for unit |delta H| in every direction, normalised by mu_sec
    s = 0.74
    rpar, rperp = s * md / ms, s
    ax.add_patch(Ellipse(tuple(tip), 2 * rpar, 2 * rperp, fc=tint(MAG, 0.14), ec=MAG, lw=1.0, zorder=2))
    ax.add_patch(Ellipse(tuple(tip), 2 * s, 2 * s, fc="none", ec=MUTED, lw=0.8, ls=(0, (1.5, 1.5)), zorder=2))
    arrow(ax, tuple(tip), (tip[0], tip[1] + rperp), color=INK, lw=1.6, ms=9, z=6)
    arrow(ax, tuple(tip), (tip[0] + rpar + 0.02, tip[1]), color=INK, lw=1.6, ms=6, z=6)
    halo(label(ax, tip[0] + 0.12, tip[1] + rperp * 0.72, T("横向：$\\mu_{\\rm sec}\\,\\delta H$", "across: $\\mu_{\\rm sec}\\,\\delta H$"),
               size=8.2, ha="left"), 3)
    halo(label(ax, tip[0] + 0.2, tip[1] - 0.22, T("平行：$\\mu_{\\rm diff}\\,\\delta H$", "along: $\\mu_{\\rm diff}\\,\\delta H$"),
               size=8.2, ha="left"), 3)
    label(ax, 1.72, -1.5, T("点线圆：各向同性线性材料", "dotted circle: isotropic linear"), size=7.6, color=MUTED)
    fig.text(0.85, 0.88 + 4 / 72 / 2.55, T("(c) 增量响应是各向异性的", "(c) the increment is anisotropic"), fontsize=9.0,
             fontweight="bold", color=INK, ha="center", va="bottom")
    return fig


# ---------------------------------------------------------------------------
@figure("hysteresis")
def hysteresis():
    L = BH.soft_loop_numbers()
    fig = plt.figure(figsize=(FIG_W, 2.6))
    a1 = fig.add_axes([0.075, 0.17, 0.25, 0.71])
    a2 = fig.add_axes([0.405, 0.17, 0.26, 0.71])
    a3 = fig.add_axes([0.735, 0.17, 0.26, 0.71])

    # (a) soft-magnetic loop
    H = np.linspace(-3200, 3200, 900)
    up, dn = BH.soft_branch(H, False), BH.soft_branch(H, True)
    a1.fill(np.concatenate((H, H[::-1])), np.concatenate((up, dn[::-1])), color=tint(MAG, 0.14), lw=0, zorder=1)
    a1.plot(H, BH.B_of_H(H), color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    a1.plot(H, dn, color=MAG, lw=1.8, zorder=3)
    a1.plot(H, up, color=MAG, lw=1.8, zorder=3)
    for Hx, br, d in ((-150, dn, -1), (150, up, 1)):
        i = int(np.argmin(np.abs(H - Hx)))
        arrow(a1, (H[i - d * 6], br[i - d * 6]), (H[i + d * 6], br[i + d * 6]), color=MAG, lw=1.4, ms=9, z=6)
    a1.axhline(0, color=MUTED, lw=0.6); a1.axvline(0, color=MUTED, lw=0.6)
    a1.plot([0, 0], [L["Br"], -L["Br"]], "o", ms=4.5, color=MAG, zorder=6)
    a1.plot([-L["HcJ"], L["HcJ"]], [0, 0], "o", ms=4.5, mfc="white", mec=MAG, mew=1.3, zorder=6)
    halo(a1.text(-130, L["Br"] + 0.12, r"$B_r$", fontsize=9, color=MAG, ha="right", va="bottom", zorder=7), 2.5)
    halo(a1.text(-560, 0.12, r"$-H_c$", fontsize=8.6, color=MAG, ha="right", va="bottom", zorder=7), 2.5)
    halo(a1.text(560, -0.14, r"$H_c$", fontsize=8.6, color=MAG, ha="left", va="top", zorder=7), 2.5)
    a1.annotate(T("面积 = 损耗", "area = loss"), xy=(350, 0.45), xytext=(-3050, 1.32),
                fontsize=7.6, color=DARK_MAG, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=DARK_MAG, lw=0.6, shrinkA=2))
    a1.text(3100, -0.62, T("虚线：单值\n合成曲线", "dashed:\nsingle-valued\ncurve"), fontsize=7.6, color=MUTED,
            ha="right", va="center")
    a1.set_xlim(-3200, 3200); a1.set_ylim(-1.85, 1.85)
    a1.set_xticks([-3000, 0, 3000]); a1.set_yticks([-1.5, 0, 1.5])
    a1.set_xlabel(r"$H$ (A/m)")
    a1.set_ylabel(r"$B$ (T)")
    clean_axes(a1)
    _ptitle(a1, T("(a) 软磁：窄回线", "(a) soft: narrow loop"))

    # (b) hard magnet: normal and intrinsic curves, two coercivities
    hi = BH.hard_numbers(BH.HCJ_HIGH)
    Hk = np.linspace(-1800e3, 60e3, 900)
    a2.plot(Hk / 1e3, BH.hard_mu0M(Hk, BH.HCJ_HIGH), color=MAG, lw=1.4, ls=(0, (4, 2)), zorder=3)
    a2.plot(Hk / 1e3, BH.hard_B(Hk, BH.HCJ_HIGH), color=MAG, lw=2.0, zorder=4)
    a2.axhline(0, color=MUTED, lw=0.6); a2.axvline(0, color=MUTED, lw=0.6)
    a2.plot([0], [BH.BR], "o", ms=4.5, color=MAG, zorder=6)
    a2.plot([-hi["HcB"] / 1e3, -hi["HcJ"] / 1e3], [0, 0], "o", ms=4.5, mfc="white", mec=MAG, mew=1.3, zorder=6)
    halo(a2.text(-40, BH.BR + 0.07, r"$B_r$", fontsize=9, color=MAG, ha="right", va="bottom"), 2.5)
    halo(a2.text(-hi["HcB"] / 1e3 + 45, -0.08, r"$-H_{cB}$", fontsize=8.6, color=MAG, ha="left", va="top"), 2.5)
    halo(a2.text(-hi["HcJ"] / 1e3 + 45, 0.08, r"$-H_{cJ}$", fontsize=8.6, color=MAG, ha="left", va="bottom"), 2.5)
    a2.annotate(T("$\\mu_0M$（内禀曲线）", "$\\mu_0M$ (intrinsic)"), xy=(-1100, float(BH.hard_mu0M(-1100e3, BH.HCJ_HIGH))),
                xytext=(-1050, 1.42), fontsize=7.8, color=MAG, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MAG, lw=0.6))
    a2.annotate(T("$B$（常规曲线）", "$B$ (normal)"), xy=(-560, float(BH.hard_B(-560e3, BH.HCJ_HIGH))),
                xytext=(-930, 0.66), fontsize=7.8, color=MAG, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MAG, lw=0.6))
    a2.set_xlim(-1800, 60); a2.set_ylim(-0.6, 1.55)
    a2.set_xticks([-1500, -1000, -500, 0])
    a2.set_xlabel(r"$H$ (kA/m)")
    clean_axes(a2)
    _ptitle(a2, T("(b) 永磁：两个矫顽力", "(b) magnet: two coercivities"))

    # (c) knee and irreversible loss
    lo = BH.HCJ_LOW
    Hk = np.linspace(-1000e3, 30e3, 900)
    a3.plot(Hk / 1e3, BH.BR + MU0 * BH.MUREC * Hk, color=MUTED, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
    a3.plot(Hk / 1e3, BH.hard_B(Hk, lo), color=MAG, lw=2.0, zorder=4)
    for P, col in ((BH.P_SAFE, ACCENT), (BH.P_KNEE, ACCENT)):
        r = BH.knee_recoil(P)
        Hl = np.linspace(-1000e3, 0, 10)
        a3.plot(Hl / 1e3, -P * MU0 * Hl, color=col, lw=1.0, ls=(0, (4, 2)), zorder=3)
        a3.plot(r["Hp"] / 1e3, r["Bp"], "o", ms=4.5, mfc="white", mec=ACCENT, mew=1.3, zorder=6)
    rk = BH.knee_recoil(BH.P_KNEE)
    Hr = np.linspace(rk["Hp"], 0, 20)
    a3.plot(Hr / 1e3, rk["Bp"] + MU0 * BH.MUREC * (Hr - rk["Hp"]), color=DARK_MAG, lw=1.5, zorder=5)
    a3.plot([0], [rk["Br_new"]], "s", ms=4.2, color=DARK_MAG, zorder=6)
    a3.plot([0], [BH.BR], "o", ms=4.5, color=MAG, zorder=6)
    a3.text(25, BH.BR + 0.02, r"$B_r$", fontsize=9, color=MAG, ha="left", va="bottom")
    a3.text(25, rk["Br_new"] - 0.02, r"$B'_r$", fontsize=9, color=DARK_MAG, ha="left", va="top")
    a3.text(-260, 0.62, "$P=1$", fontsize=7.8, color=ACCENT, ha="left")
    a3.text(-888, 0.33, "$P=0.2$", fontsize=7.8, color=ACCENT, ha="left", va="bottom")
    Hm_ = -330e3
    a3.annotate(T("回复线", "recoil line"), xy=(Hm_ / 1e3, rk["Bp"] + MU0 * BH.MUREC * (Hm_ - rk["Hp"])),
                xytext=(-560, 1.0), fontsize=7.8, color=DARK_MAG, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=DARK_MAG, lw=0.6))
    a3.annotate(T("拐点", "knee"), xy=(-712, 0.24), xytext=(-800, 0.74), fontsize=7.8, color=MAG, ha="center",
                va="center", arrowprops=dict(arrowstyle="-", color=MAG, lw=0.6))
    a3.axhline(0, color=MUTED, lw=0.6); a3.axvline(0, color=MUTED, lw=0.6)
    a3.set_xlim(-900, 130); a3.set_ylim(-0.1, 1.38)
    a3.set_xticks([-800, -400, 0])
    a3.set_xlabel(r"$H$ (kA/m)")
    clean_axes(a3)
    _ptitle(a3, T("(c) 越过拐点：剩磁变小", "(c) past the knee: lower $B_r$"))
    return fig


# ---------------------------------------------------------------------------
@figure("bh_interpolation")
def bh_interpolation():
    Hs, Bs, sp, pc = BH.interpolants()
    I = BH.interpolation_numbers()
    fig = plt.figure(figsize=(FIG_W, 2.6))
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.71])
    a2 = fig.add_axes([0.6, 0.17, 0.38, 0.71])
    H = np.linspace(0, Hs[-1], 3000)
    # (a) B(H)
    a1.plot(H, BH.B_of_H(H), color=INK, lw=0.9, zorder=2, label=T("真实曲线（合成）", "true (synthetic) curve"))
    a1.plot(H, sp(H), color=ACCENT, lw=1.5, ls=(0, (4, 2)), zorder=3, label=T("三次样条（无约束）", "cubic spline (unconstrained)"))
    a1.plot(H, pc(H), color=MAG, lw=1.5, zorder=4, label=T("PCHIP（保单调）", "PCHIP (monotone)"))
    a1.plot(Hs, Bs, "o", ms=4.2, color=INK, zorder=6, label=T("稀疏样本", "sparse samples"))
    a1.set_xlim(0, 10300); a1.set_ylim(0, 1.95)
    a1.set_xlabel(r"$H$ (A/m)")
    a1.set_ylabel(r"$B$ (T)")
    a1.legend(loc="lower right", fontsize=7.6, handlelength=2.0, borderaxespad=0.2)
    clean_axes(a1)
    _ptitle(a1, T("(a) 看起来都很光滑", "(a) all look smooth"))
    # inset zoom on the high-field part
    ins = a1.inset_axes([0.33, 0.58, 0.56, 0.38])
    ins.set_zorder(10)
    Hz = np.linspace(1800, 10000, 600)
    ins.plot(Hz, BH.B_of_H(Hz), color=INK, lw=0.8)
    ins.plot(Hz, sp(Hz), color=ACCENT, lw=1.3, ls=(0, (4, 2)))
    ins.plot(Hz, pc(Hz), color=MAG, lw=1.3)
    m = Hs >= 1800
    ins.plot(Hs[m], Bs[m], "o", ms=3.5, color=INK, zorder=5)
    ins.annotate(T("$B$ 随 $H$ 下降", "$B$ falls as $H$ rises"), xy=(I["H_dip"], I["B_dip"]), xytext=(4700, 1.5705),
                 fontsize=7.4, color=ACCENT, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6))
    ins.set_xlim(1800, 10300); ins.set_ylim(1.563, 1.622)
    ins.set_xticks([2000, 6000, 10000]); ins.set_yticks([1.58, 1.60, 1.62])
    ins.tick_params(labelsize=6.8, length=2)
    for s_ in ins.spines.values():
        s_.set_color(MUTED); s_.set_linewidth(0.6)
    # (b) mu_diff
    H = np.linspace(1500, 10000, 1500)
    true = BH.mu_diff(H) / MU0
    dsp = sp(H, 1) / MU0
    dpc = pc(H, 1) / MU0
    a2.axhspan(-16, 0, color=tint(WARN, 0.08), lw=0, zorder=0)
    a2.plot(H, true, color=INK, lw=1.0, zorder=3, label=T("真实", "true"))
    a2.plot(H, dsp, color=ACCENT, lw=1.6, ls=(0, (4, 2)), zorder=4, label=T("样条", "spline"))
    a2.plot(H, dpc, color=MAG, lw=1.6, zorder=5, label="PCHIP")
    a2.axhline(1, color=MUTED, lw=0.8, ls=(0, (1.5, 1.5)), zorder=2)
    a2.axhline(0, color=MUTED, lw=0.6, zorder=2)
    a2.annotate(T("$\\mu_0$：饱和后的斜率", "$\\mu_0$: slope in saturation"), xy=(2900, 1.0), xytext=(1650, -4.0),
                fontsize=7.6, color=MUTED, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a2.text(6000, -12.5, T("负斜率：非物理", "negative slope: unphysical"), fontsize=7.8, color=WARN, ha="center")
    a2.annotate(T("PCHIP 末端斜率为 0，\n低于 $\\mu_0$", "PCHIP end slope 0,\nbelow $\\mu_0$"), xy=(10000, 0.0),
                xytext=(6900, 15.0), fontsize=7.6, color=MAG, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MAG, lw=0.6))
    a2.plot(Hs, np.interp(Hs, H, true), "o", ms=0)  # keep autoscale stable
    for x in Hs[(Hs >= 1500)]:
        a2.axvline(x, color=LIGHT, lw=0.8, zorder=1)
    a2.set_xlim(1500, 10300); a2.set_ylim(-16, 46)
    a2.set_xlabel(r"$H$ (A/m)")
    a2.set_ylabel(T(r"$\mu_{\rm diff}/\mu_0$", r"$\mu_{\rm diff}/\mu_0$"))
    a2.legend(loc="upper right", fontsize=7.6, handlelength=2.0, borderaxespad=0.2)
    clean_axes(a2)
    _ptitle(a2, T("(b) 斜率暴露了差别", "(b) the slope tells them apart"))
    return fig
