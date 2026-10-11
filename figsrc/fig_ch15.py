"""Figures for Ch 15 (permanent magnets, yokes and MRI magnets: beyond the lumped model).

All computed quantities come from checks/check_ch15.py, so the figures, the text and the
assertions share one implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.tri as mtri
from matplotlib.patches import Circle, Ellipse, Rectangle

from style import (ACCENT, ELEC, FIG_W, INK, IRON_T, LIGHT, MAG, MUTED, T, WARN, arrow, clean_axes, figure,
                   halo, label, plt, seq_cmap, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch15 as PM  # noqa: E402
from numerics import MU0  # noqa: E402

DARK_MAG = "#8A3A00"
IRON_E = "#8E989F"


def _ptitle(ax, s):
    ax.set_title(s, fontsize=9.0, fontweight="bold", pad=4, color=INK)


# ---------------------------------------------------------------------------
@figure("demag_factor")
def demag_factor():
    fig = plt.figure(figsize=(FIG_W, 2.75))
    a1 = fig.add_axes([0.08, 0.16, 0.38, 0.62])
    a2 = fig.add_axes([0.59, 0.16, 0.38, 0.62])
    m = np.logspace(-2, 2, 400)
    Nc = np.array([PM.N_axial(x) for x in m])
    Na = 0.5 * (1 - Nc)
    # (a) N versus aspect ratio
    a1.semilogx(m, Nc, color=MAG, lw=2.0, label=T("沿对称轴 $N_c$", "along the axis $N_c$"))
    a1.semilogx(m, Na, color=MAG, lw=1.3, ls=(0, (4, 2)), label=T("垂直于轴 $N_a=(1-N_c)/2$", "across $N_a=(1-N_c)/2$"))
    a1.plot([1], [1 / 3], "o", ms=5, mfc="white", mec=INK, mew=1.2, zorder=5)
    a1.annotate(T("球：1/3", "sphere: 1/3"), xy=(1, 1 / 3), xytext=(0.025, 0.5), fontsize=8.0, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    a1.text(0.013, 0.8, T("扁：$N_c\\to1$", "flat: $N_c\\to1$"), fontsize=8.0, color=MAG, ha="left", va="center")
    a1.text(90, 0.19, T("细长：$N_c\\to0$", "slender: $N_c\\to0$"), fontsize=8.0, color=MAG, ha="right", va="center")
    a1.set_xlim(0.01, 100); a1.set_ylim(0, 1.02)
    a1.set_xlabel(T("长径比 $c/a$", "aspect ratio $c/a$"))
    a1.set_ylabel(T("退磁因子", "demagnetizing factor"))
    a1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=7.6, handlelength=1.8)
    clean_axes(a1)
    # shape glyphs above panel (a): spheroids with the symmetry axis vertical (field along it)
    g = fig.add_axes([0.08, 0.81, 0.38, 0.17])
    g.set_xlim(np.log10(0.01), np.log10(100)); g.set_ylim(-1, 1); g.axis("off")
    for mm in (0.05, 0.2, 1.0, 5.0, 20.0):
        xc = np.log10(mm)
        s_ = 0.62
        w, h = (s_ * 2 * 0.28, s_ * 2 * 0.28 * mm) if mm <= 1 else (s_ * 2 * 0.28 / mm, s_ * 2 * 0.28)
        g.add_patch(Ellipse((xc, -0.05), w * 1.1, h * 1.6, fc=tint(MAG, 0.25), ec=MAG, lw=0.8))
        g.text(xc, -0.98, f"{mm:g}", fontsize=7.0, color=MUTED, ha="center", va="bottom")
    arrow(g, (np.log10(0.0125), -0.6), (np.log10(0.0125), 0.65), color=INK, lw=1.0, ms=7)
    g.text(np.log10(0.0105), 0.0, T("磁化\n方向", "magnet-\nization"), fontsize=7.0, color=INK, ha="right", va="center")
    fig.text(0.27, 0.995, T("(a) 形状决定退磁因子", "(a) shape sets the factor"), fontsize=9.0, fontweight="bold",
             ha="center", va="top", color=INK)
    # (b) effective response
    chis = (10.0, 100.0, 1e3, 1e4)
    cols = [tint(MAG, 0.4), tint(MAG, 0.65), MAG, DARK_MAG]
    for chi, c in zip(chis, cols):
        a2.loglog(m, PM.chi_eff(chi, Nc), color=c, lw=1.5, label=rf"$\chi={chi:g}$" if chi < 1e4 else r"$\chi=10^4$")
    a2.loglog(m, 1 / Nc, color=INK, lw=1.0, ls=(0, (1.5, 1.5)), label=T("上限 $1/N_c$", "limit $1/N_c$"))
    ce = PM.chi_eff(999.0, 1 / 3)
    a2.plot([1], [ce], "o", ms=5, mfc="white", mec=INK, mew=1.2, zorder=5)
    a2.annotate(T("铁球：%.2f" % ce, "iron sphere: %.2f" % ce), xy=(1, ce), xytext=(0.03, 25), fontsize=8.0,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    a2.set_xlim(0.01, 100); a2.set_ylim(0.8, 2e4)
    a2.set_xlabel(T("长径比 $c/a$", "aspect ratio $c/a$"))
    a2.set_ylabel(T(r"有效磁化率 $M/H_{\rm applied}$", r"effective $M/H_{\rm applied}$"))
    a2.legend(loc="upper left", fontsize=7.4, handlelength=1.6, ncol=1, borderaxespad=0.2)
    clean_axes(a2)
    fig.text(0.78, 0.995, T("(b) 磁化率再大，也越不过 $1/N_c$", "(b) no response above $1/N_c$"), fontsize=9.0,
             fontweight="bold", ha="center", va="top", color=INK)
    return fig


# ---------------------------------------------------------------------------
def _mirror(s):
    """Full cross-section from the quarter solution: A(-x,y) = -A(x,y), A(x,-y) = A(x,y)."""
    p, cells, A = s["p"], s["cells"], s["A"]
    P, C, AA, BM = [], [], [], []
    n = len(p)
    for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
        P.append(p * [sx, sy])
        cc = cells + k * n
        C.append(cc if sx * sy > 0 else cc[:, ::-1])
        AA.append(A * sx)
        BM.append(s["Bmag"])
    return np.vstack(P), np.vstack(C), np.concatenate(AA), np.concatenate(BM)


def _outline(ax, G, lw=0.9, full=True):
    y1, y2, y3, y4 = PM.levels(G)
    a, xp, xq = G["a"], G["xp"], G["xp"] + G["wp"]
    if full:
        ax.plot([-xq, xq, xq, -xq, -xq], [y4, y4, -y4, -y4, y4], color=IRON_E, lw=lw, zorder=6)
        ax.plot([-xp, xp, xp, -xp, -xp], [y3, y3, -y3, -y3, y3], color=IRON_E, lw=lw, zorder=6)
    else:
        ax.plot([0, xq, xq], [y4, y4, 0], color=IRON_E, lw=lw, zorder=6)
        ax.plot([0, xp, xp], [y3, y3, 0], color=IRON_E, lw=lw, zorder=6)
    xa = (-a, a) if full else (0, a)
    for sy in ((1, -1) if full else (1,)):
        for (ya, yb), ec, hatch in (((y1, y2), IRON_E, None), ((y2, y3), MAG, "////")):
            y0 = ya if sy > 0 else -yb
            ax.add_patch(Rectangle((xa[0], y0), xa[1] - xa[0], yb - ya, fc="none", ec=ec, lw=lw, zorder=6, hatch=hatch))


def _numbered(ax, x, y, k, color=INK):
    ax.add_patch(Circle((x, y), 0.028, fc="white", ec=color, lw=0.9, zorder=12))
    ax.text(x, y - 0.002, str(k), fontsize=7.4, color=color, ha="center", va="center", zorder=13, fontweight="bold")


@figure("magnet_assembly")
def magnet_assembly():
    s, o = PM.magnet_case()
    nw = PM.magnet_network()
    G = PM.MAGNET
    y1, y2, y3, y4 = PM.levels(G)
    a, xp = G["a"], G["xp"]
    fig = plt.figure(figsize=(FIG_W, 4.25))
    ax = fig.add_axes([0.0, 0.45, 0.6, 0.55])
    lg = fig.add_axes([0.6, 0.45, 0.4, 0.55])
    b1 = fig.add_axes([0.085, 0.085, 0.37, 0.27])
    b2 = fig.add_axes([0.6, 0.085, 0.37, 0.27])
    # (a) field picture of the whole cross-section (mirrored quarter solution)
    P, C, A, Bm = _mirror(s)
    tri = mtri.Triangulation(P[:, 0], P[:, 1], C)
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.35, 0.45); ax.set_aspect("equal"); ax.axis("off")
    cen = P[C].mean(axis=1)
    inview = (np.abs(cen[:, 0]) < 0.62) & (np.abs(cen[:, 1]) < 0.4)
    tri.set_mask(~inview)
    ax.tripcolor(tri, facecolors=np.where(inview, Bm, 0.0), cmap=seq_cmap(MAG), vmin=0, vmax=2.4, shading="flat",
                 zorder=1, rasterized=True)
    ir = np.concatenate([s["iron"]] * 4)
    tri_s = mtri.Triangulation(P[:, 0], P[:, 1], C, mask=~(ir & (Bm > 1.5) & inview))
    ax.tripcolor(tri_s, facecolors=np.ones(len(C)), cmap=matplotlib.colors.ListedColormap([DARK_MAG]), vmin=0, vmax=2,
                 shading="flat", zorder=2, rasterized=True)
    # flux lines: neighbouring lines carry equal flux (the quarter magnet flux is split into 12 tubes)
    step = o["phi_m"] / 12
    lv = np.arange(-12, 13) * step
    lv = np.sort(lv[lv != 0])
    ax.tricontour(tri, A, levels=lv, colors=[INK], linewidths=0.55, zorder=3, negative_linestyles="solid", alpha=0.8)
    ax.plot([0, 0], [-y1, y1], color=INK, lw=0.55, zorder=3, alpha=0.8)
    _outline(ax, G)
    ax.add_patch(Circle((0, 0), PM.ROI, fc=tint(ELEC, 0.25), ec=ELEC, lw=1.0, ls=(0, (3, 2)), zorder=8, alpha=0.9))
    calls = [(0.415, 0.205, 1, MAG), (0.335, 0.075, 2, MAG), (0.43, 0.372, 3, DARK_MAG), (0.0, -0.155, 4, ELEC),
             (-0.365, 0.255, 5, INK)]
    for x_, y_, k, c_ in calls:
        _numbered(ax, x_, y_, k, c_)
    arrow(ax, (-0.338, 0.248), (-0.305, 0.234), color=INK, lw=0.8, ms=6, z=12)
    arrow(ax, (0.40, 0.358), (0.335, 0.318), color=DARK_MAG, lw=0.8, ms=6, z=12)
    label(ax, -0.1, 0.425, T("(a) 二维截面：$|\\mathbf{B}|$（底色）与磁力线", "(a) 2-D section: $|\\mathbf{B}|$ (shading) and flux lines"),
          size=8.8, weight="bold")
    # call-out legend
    lg.set_xlim(0, 1); lg.set_ylim(0, 1); lg.axis("off")
    rows = [(1, MAG, T("漏磁：只有 %.0f%% 的磁体磁通\n穿过两极之间" % (100 * o["gap_share"]),
                        "leakage: only %.0f%% of the magnet\nflux crosses between the poles" % (100 * o["gap_share"]))),
            (2, MAG, T("极面边缘：磁力线外鼓，磁场减弱", "pole edge: lines bulge, field drops")),
            (3, DARK_MAG, T("局部饱和：轭板中 $>1.5$ T（深色）", "local saturation: plate $>1.5$ T (dark)")),
            (4, ELEC, T("成像区 $r=0.1$ m：$\\Delta B/B=%.1f\\%%$" % (100 * o["dB_disc"]),
                        "imaging region $r=0.1$ m: $\\Delta B/B=%.1f\\%%$" % (100 * o["dB_disc"]))),
            (5, INK, T("磁体尖角：反向场最强", "magnet corner: strongest reverse field"))]
    yy = 0.86
    for k, c_, txt in rows:
        lg.add_patch(Circle((0.05, yy), 0.032, fc="white", ec=c_, lw=0.9, transform=lg.transAxes))
        lg.text(0.05, yy - 0.003, str(k), fontsize=7.4, color=c_, ha="center", va="center", fontweight="bold")
        lg.text(0.11, yy, txt, fontsize=8.0, color=INK, ha="left", va="center", linespacing=1.25)
        yy -= 0.16 if "\n" not in txt else 0.2
    lg.text(0.02, 0.06, T("网络模型给出的只是一个平均 $B_g$", "the network gives one mean $B_g$ only"), fontsize=7.8,
            color=ACCENT, ha="left", va="center", style="italic")
    # (b) mid-plane B_y
    Ag, Bxg, Byg = PM.nodal_B(s)
    xs = s["xs"]
    sel = xs <= 0.7
    b1.axvspan(0, PM.ROI, color=tint(ELEC, 0.18), lw=0)
    b1.axvspan(xp, xp + G["wp"], color=IRON_T, lw=0)
    b1.plot(xs[sel], Byg[0, sel], color=MAG, lw=1.8, label=T("场解", "field solution"))
    b1.plot([0, a], [nw["Bg"]] * 2, color=ACCENT, lw=1.4, ls=(0, (4, 2)), label=T("网络（平均）", "network (mean)"))
    b1.axvline(a, color=MUTED, lw=0.7, ls=(0, (1.5, 1.5)))
    b1.text(a + 0.012, 0.228, T("极面边缘", "pole edge"), fontsize=7.4, color=MUTED, ha="left", va="center")
    b1.text(xp - 0.012, 0.075, T("柱内\n约 $-1$ T", "pillar\n$\\approx-1$ T"), fontsize=7.2, color=MUTED, ha="right")
    b1.set_xlim(0, 0.7); b1.set_ylim(-0.02, 0.245)
    b1.set_xlabel(T("中平面上的 $x$（m）", "$x$ on the mid-plane (m)"), fontsize=8.4)
    b1.set_ylabel(r"$B_y$ (T)", fontsize=8.6)
    b1.tick_params(labelsize=7.8)
    b1.text(0.01, nw["Bg"] + 0.008, T("网络（平均值）", "network (mean)"), fontsize=7.4, color=ACCENT, ha="left", va="bottom")
    b1.text(0.11, 0.138, T("场解", "field solution"), fontsize=7.4, color=MAG, ha="left", va="top")
    clean_axes(b1)
    b1.set_title(T("(b) 气隙磁场是一条曲线", "(b) the gap field is a profile"), fontsize=8.8, fontweight="bold", pad=3)
    # (c) H inside the magnet
    ys = s["ys"]
    nu = s["nu_m"]
    ym = 0.5 * (y2 + y3)
    for yy_, col, lab in ((ym, MAG, T("磁体中间高度", "magnet mid-height")),
                          (y2 + 0.004, DARK_MAG, T("紧靠极靴的一层", "layer next to the pole"))):
        j = int(np.argmin(np.abs(ys - yy_)))
        m_ = xs < a - 1e-9
        b2.plot(xs[m_], nu * (Byg[j, m_] - G["Br"]) / 1e3, color=col, lw=1.6, label=lab)
    b2.axhline(nw["Hm"] / 1e3, color=ACCENT, lw=1.3, ls=(0, (4, 2)), label=T("网络（均匀）", "network (uniform)"))
    b2.axhline(-PM.BH.HC_LINE / 1e3, color=MUTED, lw=0.8, ls=(0, (1.5, 1.5)))
    b2.text(0.265, -PM.BH.HC_LINE / 1e3 + 12, T("直线回复线的截距 $-909$", "recoil-line intercept $-909$"), fontsize=7.2,
            color=MUTED, ha="right", va="bottom")
    b2.set_xlim(0, a); b2.set_ylim(-1400, -470)
    b2.set_xlabel(T("磁体内的 $x$（m）", "$x$ inside the magnet (m)"), fontsize=8.4)
    b2.set_ylabel(r"$H_y$ (kA/m)", fontsize=8.6)
    b2.tick_params(labelsize=7.8)
    b2.legend(loc="lower left", fontsize=7.2, handlelength=1.8, borderaxespad=0.2)
    clean_axes(b2)
    b2.set_title(T("(c) 磁体内部的反向场", "(c) reverse field inside the magnet"), fontsize=8.8, fontweight="bold", pad=3)
    return fig


# ---------------------------------------------------------------------------
@figure("symmetry_bc")
def symmetry_bc():
    s, o = PM.magnet_case()
    G = PM.MAGNET
    y1, y2, y3, y4 = PM.levels(G)
    fig = plt.figure(figsize=(FIG_W, 2.35))
    a1 = fig.add_axes([0.05, 0.04, 0.43, 0.8])
    a2 = fig.add_axes([0.565, 0.04, 0.43, 0.8])
    W, Hh = 0.62, 0.36
    for ax in (a1, a2):
        ax.set_xlim(-0.13, W); ax.set_ylim(-0.07, Hh); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), W, Hh, fc="none", ec=LIGHT, lw=0.8, zorder=0))
    # (1) main field from the quarter solution
    tri = mtri.Triangulation(s["p"][:, 0], s["p"][:, 1], s["cells"])
    cen = s["p"][s["cells"]].mean(axis=1)
    tri.set_mask(~((cen[:, 0] < W + 0.02) & (cen[:, 1] < Hh + 0.02)))
    step = o["phi_m"] / 10
    lv = -np.arange(1, 11) * step
    a1.tricontour(tri, s["A"], levels=np.sort(lv), colors=[MAG], linewidths=0.8, zorder=3, negative_linestyles="solid")
    _outline(a1, G, lw=0.8, full=False)
    a1.add_patch(Rectangle((0, y1), G["a"], y4 - y1, fc=IRON_T, ec="none", zorder=1))
    a1.add_patch(Rectangle((0, y3), G["xp"] + G["wp"], G["tpl"], fc=IRON_T, ec="none", zorder=1))
    a1.add_patch(Rectangle((G["xp"], 0), G["wp"], y3, fc=IRON_T, ec="none", zorder=1))
    # arrows on two flux lines near the axis (field points up in the gap)
    for xx in (0.06, 0.18):
        arrow(a1, (xx, 0.06), (xx, 0.11), color=MAG, lw=1.0, ms=7, z=6)
    # (2) transverse gradient: b = G (y, x), lines are contours of A = G (y^2 - x^2)/2
    xg = np.linspace(0, W, 300); yg = np.linspace(0, Hh, 200)
    X, Y = np.meshgrid(xg, yg)
    Az = PM.gradient_Az(X, Y)
    a2.contour(X, Y, Az, levels=np.linspace(-0.18, 0.06, 13), colors=[MAG], linewidths=0.8, zorder=3,
               negative_linestyles="solid")
    for (xx, yy) in ((0.03, 0.25), (0.42, 0.03)):
        bx, by = yy, xx
        n = np.hypot(bx, by)
        arrow(a2, (xx, yy), (xx + 0.04 * bx / n, yy + 0.04 * by / n), color=MAG, lw=1.0, ms=7, z=6)
    # boundary-condition labels: n.B = 0 (Dirichlet in A_z) versus n x H = 0 (natural)
    DIR, NAT = ACCENT, "#5B6670"
    nat_txt = T("$\\mathbf{n}\\times\\mathbf{H}=0$（$\\partial A_z/\\partial n=0$）", "$\\mathbf{n}\\times\\mathbf{H}=0$ ($\\partial A_z/\\partial n=0$)")
    for ax, left, col_l in ((a1, T("$\\mathbf{n}\\cdot\\mathbf{B}=0$\n（$A_z=0$）\n磁力线沿面走", "$\\mathbf{n}\\cdot\\mathbf{B}=0$\n($A_z=0$)\nlines run along"), DIR),
                            (a2, T("$\\mathbf{n}\\times\\mathbf{H}=0$\n（自然边界）\n磁力线垂直穿过", "$\\mathbf{n}\\times\\mathbf{H}=0$\n(natural)\nlines cross\nat right angles"), NAT)):
        ax.plot([0, 0], [0, Hh], color=col_l, lw=2.4, zorder=8, solid_capstyle="butt")
        ax.plot([0, W], [0, 0], color=NAT, lw=2.4, zorder=8, solid_capstyle="butt")
        label(ax, -0.012, Hh * 0.5, left, size=7.6, color=col_l, ha="right")
        label(ax, W / 2, -0.04, nat_txt, size=7.6, color=NAT)
    a1.set_title(T("(a) 主磁场：磁化平行于 $x=0$ 面", "(a) main field: magnetization in the $x=0$ plane"), fontsize=8.8,
                 fontweight="bold", pad=3)
    a2.set_title(T("(b) 横向梯度场 $\\mathbf{b}=G(y,\\,x)$", "(b) transverse gradient $\\mathbf{b}=G(y,\\,x)$"), fontsize=8.8,
                 fontweight="bold", pad=3)
    return fig
