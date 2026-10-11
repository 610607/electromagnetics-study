"""Figures for Ch 21 (from field to neural response).

All computed curves come from checks/check_ch21.py, so the figures, the text and the
assertions share one implementation.  Everything is dimensionless: lengths in units of the
length constant lambda, times in units of tau_m, fields in units of a reference E0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from style import (ACCENT, AMBER, CURR, ELEC, ELEC_T, FIG_W, INK, LIGHT, MATL_T, MUTED, NEUTRAL_T, POT,
                   T, arrow, canvas, clean_axes, figure, halo, label, plt, rbox, tint)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
import check_ch21 as NR  # noqa: E402

DEP = AMBER          # depolarisation (+)
HYP = ACCENT         # hyperpolarisation (-)
AXON = "#C9B79C"     # axon body (warm sand, like tissue fills)
VM_CMAP = LinearSegmentedColormap.from_list("vm", [HYP, tint(HYP, 0.35), "#EDEDED", tint(DEP, 0.45), DEP])


def _signed_fill(ax, x, y, z=1):
    ax.fill_between(x, 0, y, where=y > 0, color=tint(DEP, 0.30), lw=0, zorder=z, interpolate=True)
    ax.fill_between(x, 0, y, where=y < 0, color=tint(HYP, 0.22), lw=0, zorder=z, interpolate=True)


# ---------------------------------------------------------------------------
@figure("activating_function")
def activating_function():
    s = np.linspace(-4.2, 4.2, 1201)
    Ve, Es, f = NR.gaussian_profile(s)
    ds = 0.02
    sg = np.arange(-15, 15 + ds / 2, ds)
    Veg, _, _ = NR.gaussian_profile(sg)
    Vinf, (V01, V03) = NR.passive_cable(Veg, ds, times=(0.1, 0.3))

    fig = plt.figure(figsize=(FIG_W, 4.4))
    L, W = 0.215, 0.765
    boxes = [(0.712, 0.125), (0.535, 0.125), (0.335, 0.15), (0.105, 0.17)]
    axs = [fig.add_axes([L, b, W, h]) for b, h in boxes]
    sch = fig.add_axes([L, 0.862, W, 0.13])
    # ---- schematic strip: axon, axial current inside, membrane current through the wall
    sch.set_xlim(-4.2, 4.2); sch.set_ylim(-0.95, 0.75); sch.axis("off")
    sch.add_patch(FancyBboxPatch((-4.2, -0.22), 8.4, 0.44, boxstyle="round,pad=0,rounding_size=0.2",
                                 fc=tint(AXON, 0.55), ec=tint("#8C8279", 0.8), lw=0.8, zorder=1))
    for x0 in np.linspace(-3.4, 2.9, 8):
        e = float(np.exp(-x0**2 / 2))
        arrow(sch, (x0, 0.0), (x0 + 0.55 * e + 0.02, 0.0), color=ELEC, lw=0.6 + 2.2 * e, ms=4 + 6 * e, z=4)
    for x0 in (-1.9, -1.3, -0.7, 0.7, 1.3, 1.9):
        out = x0 > 0
        c = DEP if out else HYP
        w_ = abs(x0) * np.exp(-x0**2 / 2) / 0.61
        p, q = (x0, 0.26), (x0, 0.26 + 0.42 * w_ + 0.06)
        arrow(sch, p if out else q, q if out else p, color=c, lw=0.7 + 0.8 * w_, ms=5 + 3 * w_, z=4)
    label(sch, -4.15, -0.66, T("轴突内的轴向电流：箭头粗细随 $E_s$", "axial current (width follows $E_s$)"),
          size=T(7.6, 7.2), color=ELEC, ha="left")
    label(sch, 4.15, -0.66, T("膜电流：向外 → 去极化，向内 → 超极化", "outward → depolarised, inward → hyperpolarised"),
          size=T(7.6, 7.2), color=INK, ha="right")

    names = [
        (T("胞外势", "extracellular\npotential"), r"$V_e/(E_0\ell)$", POT),
        (T("纵向电场", "longitudinal\nfield"), r"$E_s/E_0$", ELEC),
        (T("激活函数", "activating\nfunction"), r"$f\ell/E_0$", INK),
        (T("膜电压", "membrane\nvoltage"), r"$V_m/(E_0\lambda)$", INK),
    ]
    ops = [r"$-\partial_s$", r"$-\partial_s$", T("膜方程", "cable")]
    for a, (nm, sym, c) in zip(axs, names):
        clean_axes(a)
        a.set_xlim(-4.2, 4.2)
        a.axhline(0, color=MUTED, lw=0.6, zorder=1)
        for xv in (-1, 0, 1):
            a.axvline(xv, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
        a.text(-0.255, 0.62, nm, transform=a.transAxes, fontsize=8.6, color=c, ha="left", va="center",
               linespacing=1.15)
        a.text(-0.255, 0.18, sym, transform=a.transAxes, fontsize=8.6, color=c, ha="left", va="center")
        a.tick_params(labelsize=7.8)
    for k, op in enumerate(ops):
        y0 = axs[k].get_position().y0
        y1 = axs[k + 1].get_position().y1
        fig.patches.append(plt.matplotlib.patches.FancyArrowPatch(
            (0.168, y0 + 0.004), (0.168, y1 - 0.004), transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=8, lw=0.9, color=MUTED))
        fig.text(0.158, 0.5 * (y0 + y1), op, fontsize=8.2, color=MUTED, ha="right", va="center")
    # (1) Ve
    a = axs[0]
    a.plot(s, Ve, color=POT, lw=1.9, zorder=3)
    a.set_ylim(-1.5, 1.5); a.set_yticks([-1, 0, 1])
    a.set_xticklabels([])
    # (2) Es
    a = axs[1]
    a.plot(s, Es, color=ELEC, lw=1.9, zorder=3)
    a.plot([0], [1], "o", ms=4.5, color=ELEC, zorder=4)
    a.set_ylim(-0.1, 1.25); a.set_yticks([0, 1])
    a.set_xticklabels([])
    t = a.text(0.35, 1.05, T(r"$|\mathbf{E}|$ 最大处：$f=0$", r"$|\mathbf{E}|$ peak: $f=0$"), fontsize=8.2, color=ELEC,
               ha="left", va="center")
    # (3) f
    a = axs[2]
    _signed_fill(a, s, f)
    a.plot(s, f, color=INK, lw=1.7, zorder=3)
    a.set_ylim(-0.85, 0.85); a.set_yticks([-0.5, 0, 0.5])
    a.set_xticklabels([])
    a.text(1.75, 0.55, T(r"$f>0$：去极化", r"$f>0$: depolarising"), fontsize=8.2, color=DEP, ha="left", va="center")
    a.text(-1.75, -0.55, T(r"$f<0$：超极化", r"$f<0$: hyperpolarising"), fontsize=8.2, color=HYP, ha="right",
           va="center")
    a.text(1.08, -0.42, r"$s=\ell$", fontsize=7.8, color=MUTED, ha="left")
    # (4) Vm
    a = axs[3]
    _signed_fill(a, sg, Vinf)
    a.plot(sg, V01, color=MUTED, lw=1.1, ls=(0, (3, 1.6)), zorder=3, label=r"$t=0.1\tau_m$")
    a.plot(sg, V03, color=MUTED, lw=1.2, ls=(0, (1, 1.2)), zorder=3, label=r"$t=0.3\tau_m$")
    a.plot(sg, Vinf, color=INK, lw=1.8, zorder=3, label=T("稳态", "steady state"))
    a.set_ylim(-0.36, 0.5); a.set_yticks([-0.2, 0, 0.2])
    a.legend(loc="upper left", fontsize=7.6, handlelength=2.2, borderaxespad=0.1, ncol=3, columnspacing=1.2)
    a.set_xlabel(T(r"沿轴突的位置 $s/\ell$（取 $\ell=\lambda$）", r"position along the axon $s/\ell$ (with $\ell=\lambda$)"),
                 fontsize=8.6)
    return fig


# ---------------------------------------------------------------------------
@figure("curved_axon")
def curved_axon():
    r = NR.bent_response(ds=0.01, times=(0.1, 0.5))
    g = r["g"]
    s, Vinf = g["s"], r["Vinf"]
    s1, s2 = g["s1"], g["s2"]
    R = NR.BENT["R"]
    fig = plt.figure(figsize=(FIG_W, 3.15))
    ax = fig.add_axes([0.0, 0.0, 0.42, 0.93])
    b1 = fig.add_axes([0.535, 0.585, 0.45, 0.34])
    b2 = fig.add_axes([0.535, 0.13, 0.45, 0.34])
    # ---------------- (a) geometry in a uniform field
    ax.set_xlim(-3.55, 1.45); ax.set_ylim(-1.3, 4.95); ax.set_aspect("equal"); ax.axis("off")
    for yy in np.arange(-0.62, 4.9, 0.62):
        for xx in np.arange(-3.3, 1.4, 0.7):
            arrow(ax, (xx - 0.2, yy), (xx + 0.2, yy), color=tint(ELEC, 0.42), lw=0.9, ms=6, z=1)
    keep = g["y"] < 4.25
    pts = np.column_stack([g["x"][keep], g["y"][keep]])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    lc = LineCollection(segs, cmap=VM_CMAP, norm=norm, lw=6.5, capstyle="round", zorder=3)
    lc.set_array(Vinf[keep][:-1])
    under = LineCollection(segs, colors=["#8C8279"], lw=8.3, capstyle="round", zorder=2)
    ax.add_collection(under)
    ax.add_collection(lc)
    for yy in (4.45, 4.6, 4.75):
        ax.add_patch(Circle((R, yy), 0.035, fc=MUTED, ec="none", zorder=4))
    t = label(ax, R - 0.25, 3.55, T("远端垂直于 $\\mathbf{E}$\n（$E_s=0$，不极化）", "far end normal to $\\mathbf{E}$\n($E_s=0$, unpolarised)"),
              size=7.6, color=MUTED, ha="right")
    halo(t)
    t = label(ax, -3.0, -0.55, T("末端\n超极化", "terminal:\nhyperpolarised"), size=8.0, color=HYP)
    halo(t)
    t = label(ax, 1.0, 0.02, T("弯曲处\n去极化", "bend:\ndepolarised"), size=8.0, color=DEP)
    halo(t)
    # radius marker
    ax.plot([0, 0], [0, R], color=INK, lw=0.7, ls=(0, (2, 1.5)), zorder=4)
    t = label(ax, -0.08, 0.27, r"$R$", size=8.4, ha="right")
    halo(t)
    t = label(ax, -3.4, 4.55, T("均匀电场 $\\mathbf{E}=E_0\\mathbf{e}_x$\n$|\\mathbf{E}|$ 处处相同", "uniform field $\\mathbf{E}=E_0\\mathbf{e}_x$\n$|\\mathbf{E}|$ the same everywhere"),
              size=8.2, color=ELEC, ha="left")
    halo(t, 4.5)
    # scale bar: one length constant
    ax.plot([-3.4, -2.4], [-1.15, -1.15], color=INK, lw=1.4, solid_capstyle="butt")
    for xv in (-3.4, -2.4):
        ax.plot([xv, xv], [-1.2, -1.1], color=INK, lw=1.0)
    label(ax, -2.3, -1.15, r"$\lambda$", size=8.6, ha="left")
    # colour key (horizontal, under the path)
    cax = ax.inset_axes([-0.9, -1.2, 1.9, 0.14], transform=ax.transData)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=VM_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-1, 0, 1]); cb.ax.tick_params(labelsize=7.0, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    label(ax, -1.0, -1.13, r"$V_m/(E_0\lambda)$", size=7.6, ha="right")
    fig.text(0.0, 0.955, T("(a) 均匀场中的弯曲轴突", "(a) a bent axon in a uniform field"), fontsize=9.0, weight="bold")
    # ---------------- (b) E_s and f along the path
    for a in (b1, b2):
        clean_axes(a)
        a.set_xlim(0, 8.2)
        a.axvspan(s1, s2, color=tint(DEP, 0.13), lw=0, zorder=0)
        a.axhline(0, color=MUTED, lw=0.6, zorder=1)
        a.tick_params(labelsize=7.8)
    b1.plot(s, r["Es"], color=ELEC, lw=1.6, zorder=3, label=r"$E_s/E_0$")
    _signed_fill(b1, s, r["f"])
    b1.plot(s, r["f"], color=INK, lw=1.6, zorder=3, label=r"$f\lambda/E_0$")
    b1.set_ylim(-0.25, 2.3)
    b1.set_xticklabels([])
    b1.legend(loc="upper right", fontsize=7.8, handlelength=1.6, borderaxespad=0.1)
    b1.annotate(T(r"峰值 $E_0/R$", r"peak $E_0/R$"), xy=(s2, 1 / R), xytext=(4.6, 1.75), fontsize=7.8, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
    b1.annotate("", xy=(0.02, 1.45), xytext=(0.02, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=HYP, mutation_scale=8))
    b1.text(0.2, 1.62, T("末端：$f=0$，\n但轴向电流在此须为零，\n相当于一个集中的源", "terminal: $f=0$, yet\nthe sealed end acts\nas a source"),
            fontsize=T(7.4, 7.0), color=HYP, va="center", ha="left", linespacing=1.2)
    b1.set_title(T("(b) 沿路径的纵向场与激活函数", "(b) longitudinal field and activating function"),
                 fontsize=9.0, weight="bold", loc="left", pad=3)
    # ---------------- (c) passive membrane response
    sty = [((0, (3, 1.6)), MUTED, r"$t=0.1\tau_m$"), ((0, (1, 1.2)), MUTED, r"$t=0.5\tau_m$")]
    for v, (ls, c, lab_) in zip(r["Vt"], sty):
        b2.plot(s, v, color=c, lw=1.1, ls=ls, zorder=3, label=lab_)
    _signed_fill(b2, s, Vinf)
    b2.plot(s, Vinf, color=INK, lw=1.8, zorder=3, label=T("稳态", "steady"))
    b2.set_ylim(-1.15, 0.75)
    b2.set_yticks([-1, -0.5, 0, 0.5])
    b2.legend(loc="lower right", fontsize=7.6, handlelength=2.0, borderaxespad=0.1, ncol=3, columnspacing=0.9)
    b2.set_xlabel(T(r"弧长 $s/\lambda$（从末端算起）", r"arc length $s/\lambda$ (from the terminal)"), fontsize=8.4)
    b2.set_ylabel(r"$V_m/(E_0\lambda)$", fontsize=8.4)
    kb = int(np.argmax(Vinf))
    b2.text(s[kb], Vinf[kb] + 0.08, f"{Vinf[kb]:+.2f}", fontsize=7.8, color=DEP, ha="center", va="bottom")
    b2.text(0.18, -1.02, f"{NR.run_cache()['term_V']:+.2f}", fontsize=7.8, color=HYP, ha="left", va="center")
    b2.set_title(T("(c) 被动膜的响应（矩形脉冲开始后）", "(c) passive membrane response after switch-on"),
                 fontsize=9.0, weight="bold", loc="left", pad=3)
    b1.text(0.5 * (s1 + s2), 2.12, T("弯曲", "bend"), fontsize=7.8, color=DEP, ha="center")
    return fig


# ---------------------------------------------------------------------------
def _wire(ax, pts, color=INK, lw=1.1, z=3, ls="-"):
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="round")


def _resistor_h(ax, xc, y, L=0.62, H=0.2, color=INK, fill="white", z=5):
    ax.add_patch(Rectangle((xc - L / 2, y - H / 2), L, H, fc=fill, ec=color, lw=1.0, zorder=z))


def _capacitor_v(ax, x, ytop, ybot, color=INK, w=0.36, gap=0.11, z=5):
    ym = 0.5 * (ytop + ybot)
    _wire(ax, [(x, ytop), (x, ym + gap / 2)], color=color)
    _wire(ax, [(x, ym - gap / 2), (x, ybot)], color=color)
    for yy in (ym + gap / 2, ym - gap / 2):
        ax.plot([x - w / 2, x + w / 2], [yy, yy], color=color, lw=1.6, zorder=z, solid_capstyle="butt")


def _ion_v(ax, x, ytop, ybot, color=CURR, H=0.5, W=0.26, z=5):
    ym = 0.5 * (ytop + ybot)
    _wire(ax, [(x, ytop), (x, ym + H / 2)])
    _wire(ax, [(x, ym - H / 2), (x, ybot)])
    ax.add_patch(Rectangle((x - W / 2, ym - H / 2), W, H, fc=tint(color, 0.15), ec=color, lw=1.0, zorder=z))
    arrow(ax, (x - 0.2, ym - 0.3), (x + 0.2, ym + 0.3), color=color, lw=0.9, ms=6, z=z + 1)


def _vsource(ax, x, ytop, ybot, r=0.2, color=POT, z=5):
    ym = 0.5 * (ytop + ybot)
    _wire(ax, [(x, ytop), (x, ym + r)])
    _wire(ax, [(x, ym - r), (x, ybot)])
    ax.add_patch(Circle((x, ym), r, fc="white", ec=color, lw=1.1, zorder=z))
    label(ax, x, ym + r * 0.45, "+", size=7.5, color=color, weight="bold", z=z + 1)
    label(ax, x, ym - r * 0.5, "−", size=8, color=color, weight="bold", z=z + 1)


@figure("cable_compartments")
def cable_compartments():
    fig, ax = canvas(FIG_W, 3.55, xlim=(0, 10), ylim=(0.0, 5.92))
    xs = [2.5, 5.0, 7.5]
    names = ["n-1", "n", "n+1"]
    # ---------------- top: the fibre path sampled in a computed field
    rbox(ax, 0.15, 3.85, 9.7, 1.98, fc=tint("#C9B79C", 0.22), ec="none", r=0.12, z=0)
    label(ax, 0.3, 5.62, T("组织中的场解 $\\mathbf{E}(\\mathbf{r},t)$（神经只“读取”它，不改变它）",
                           "tissue field $\\mathbf{E}(\\mathbf{r},t)$ from the field solution (read, not changed, by the nerve)"),
          size=8.2, color=ELEC, ha="left")
    X, Y = np.meshgrid(np.linspace(0.6, 9.4, 12), np.linspace(4.1, 5.25, 3))
    ang = 0.25 * np.sin(0.55 * X) - 0.15
    mag = 0.75 + 0.25 * np.cos(0.4 * X + 1.0)
    for x0, y0, a0, m0 in zip(X.ravel(), Y.ravel(), ang.ravel(), mag.ravel()):
        dx, dy = 0.24 * m0 * np.cos(a0), 0.24 * m0 * np.sin(a0)
        arrow(ax, (x0 - dx, y0 - dy), (x0 + dx, y0 + dy), color=tint(ELEC, 0.5), lw=0.8, ms=5, z=1)
    xp = np.linspace(0.45, 9.55, 300)
    yp = 4.55 + 0.32 * np.sin(0.75 * xp - 0.6)
    ax.plot(xp, yp, color="#8C8279", lw=7.5, solid_capstyle="round", zorder=2)
    ax.plot(xp, yp, color=tint(AXON, 0.6), lw=5.6, solid_capstyle="round", zorder=2.1)
    yn = [4.55 + 0.32 * np.sin(0.75 * x - 0.6) for x in xs]
    for x, y, nm in zip(xs, yn, names):
        ax.add_patch(Circle((x, y), 0.085, fc=POT, ec="white", lw=0.8, zorder=6))
        t = label(ax, x, y - 0.36, f"$V_{{e,{nm}}}$", size=8.6, color=POT, z=7)
        halo(t)
    # tangent at node n
    k = 0.75 * 0.32 * np.cos(0.75 * xs[1] - 0.6)
    tv = np.array([1.0, k]) / np.hypot(1.0, k)
    arrow(ax, (xs[1], yn[1]), (xs[1] + 0.62 * tv[0], yn[1] + 0.62 * tv[1]), color=INK, lw=1.1, ms=7, z=7)
    t = label(ax, xs[1] + 0.68 * tv[0] + 0.05, yn[1] + 0.68 * tv[1] + 0.13, r"$\mathbf{t}$", size=8.6, z=7)
    halo(t)
    t = label(ax, 6.25, 5.2, r"$V_{e,n+1}-V_{e,n}=-\int_{s_n}^{s_{n+1}}\mathbf{E}\cdot\mathbf{t}\,\mathrm{d}s$",
              size=8.4, color=POT, z=7)
    halo(t, 4)
    ax.annotate("", xy=(xs[0], 3.98), xytext=(xs[1], 3.98),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=MUTED, shrinkA=0, shrinkB=0))
    label(ax, 3.75, 4.07, r"$\Delta s$", size=8.2, color=MUTED, va="bottom")
    # ---------------- bottom: the compartment chain
    yi, ye, y0 = 3.05, 1.55, 0.35
    _wire(ax, [(0.55, yi), (9.45, yi)])
    for xr in (1.25, 3.75, 6.25, 8.75):
        _resistor_h(ax, xr, yi)
    label(ax, 3.75, yi + 0.3, r"$R_a=r_i\Delta s$", size=8.4)
    label(ax, 6.25, yi + 0.3, r"$G=1/R_a$", size=8.4)
    label(ax, 0.32, yi, "…", size=10, color=MUTED)
    label(ax, 9.68, yi, "…", size=10, color=MUTED)
    label(ax, 0.3, yi + 0.42, T("胞内", "inside"), size=8.0, color=MUTED, ha="left")
    for x, nm in zip(xs, names):
        ax.add_patch(Circle((x, yi), 0.06, fc=INK, ec="none", zorder=6))
        label(ax, x - 0.12, yi + 0.3, f"$V_{{i,{nm}}}$", size=8.4, ha="right")
        xc, xg = x - 0.32, x + 0.32
        _wire(ax, [(x, yi), (x, yi - 0.25), (xc, yi - 0.25)])
        _wire(ax, [(x, yi - 0.25), (xg, yi - 0.25)])
        _capacitor_v(ax, xc, yi - 0.25, ye + 0.25)
        _ion_v(ax, xg, yi - 0.25, ye + 0.25)
        _wire(ax, [(xc, ye + 0.25), (xg, ye + 0.25)])
        _wire(ax, [(x, ye + 0.25), (x, ye)])
        ax.add_patch(Circle((x, ye), 0.07, fc="white", ec=POT, lw=1.1, zorder=6))
        _vsource(ax, x, ye - 0.07, y0)
    label(ax, xs[1] - 0.55, 2.3, r"$C_n$", size=8.4, ha="right")
    label(ax, xs[1] + 0.52, 2.3, r"$I_{\mathrm{ion},n}$", size=8.4, color=CURR, ha="left")
    _wire(ax, [(0.55, y0), (9.45, y0)], color=MUTED, lw=0.9, ls=(0, (3, 2)))
    label(ax, 9.45, y0 - 0.2, T("参考电位", "reference"), size=7.6, color=MUTED, ha="right")
    label(ax, xs[2] + 0.3, 0.92, T("$V_{e,n+1}$：取自场解", "$V_{e,n+1}$: from the field"), size=8.4, color=POT, ha="left")
    label(ax, 0.3, ye + 0.05, T("胞外", "outside"), size=8.0, color=MUTED, ha="left")
    # membrane voltage brace at node n+1 (right side)
    xb = xs[2] + 0.72
    ax.annotate("", xy=(xb, yi - 0.12), xytext=(xb, ye + 0.08),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK, shrinkA=0, shrinkB=0))
    label(ax, xb + 0.12, 0.5 * (yi + ye), T("$V_{m,n+1}=V_{i,n+1}-V_{e,n+1}-V_{\\rm rest}$\n（膜电压）",
                                           "$V_{m,n+1}=V_{i,n+1}-V_{e,n+1}-V_{\\rm rest}$\n(membrane voltage)"),
          size=7.8, ha="left")
    for x, nm in zip(xs[:2], names[:2]):
        label(ax, x - 0.3, 0.92, f"$V_{{e,{nm}}}$", size=8.4, color=POT, ha="right")
    return fig


# ---------------------------------------------------------------------------
@figure("strength_duration")
def strength_duration():
    LN2 = NR.LN2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 2.85), gridspec_kw=dict(width_ratios=[1.12, 1.0]))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.165, top=0.9, wspace=0.3)
    EX, FT = ACCENT, AMBER
    # (a) threshold versus duration
    x = np.logspace(np.log10(0.03), np.log10(12), 600)
    a1.semilogx(x, NR.sd_exact(x), color=EX, lw=2.0, zorder=3,
                label=T("被动膜精确解", "exact passive solution"))
    a1.semilogx(x, NR.sd_fit(x), color=FT, lw=1.8, ls=(0, (4, 2)), zorder=3,
                label=T("双曲线拟合（Lapicque–Weiss 型）", "hyperbolic (Lapicque–Weiss) fit"))
    a1.axhline(1, color=MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    a1.axhline(2, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1, xmax=0.5)
    a1.axvline(LN2, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1, ymax=0.25)
    a1.plot([LN2], [2], "o", ms=5.5, mfc="white", mec=INK, mew=1.2, zorder=5)
    a1.set_xlim(0.03, 12); a1.set_ylim(0, 8)
    a1.set_xlabel(T(r"脉宽 $T/\tau_m$", r"pulse duration $T/\tau_m$"))
    a1.set_ylabel(T(r"阈值 $A_{\rm th}/A_{\rm rh}$", r"threshold $A_{\rm th}/A_{\rm rh}$"))
    a1.text(0.6, 1.5, T("时值 $\\tau_{\\rm ch}=\\tau_m\\ln2$\n两条曲线都等于 2", "chronaxie $\\tau_{\\rm ch}=\\tau_m\\ln 2$:\nboth curves equal 2"),
            fontsize=7.8, color=INK, ha="right", va="center", linespacing=1.3)
    a1.text(11, 0.62, T("基强度", "rheobase"), fontsize=7.8, color=MUTED, ha="right", va="center")
    a1.text(0.034, 4.3, T("短脉冲：精确解\n是拟合的 $1/\\ln2$\n$\\approx1.44$ 倍", "short pulses:\nexact is $1/\\ln2$\n$\\approx1.44\\times$ the fit"),
            fontsize=7.6, color=EX, ha="left", va="center", linespacing=1.25)
    o = NR.run_cache()
    xm = o["sd_overshoot_x"]
    a1.annotate(T(f"$T\\approx{xm:.1f}\\tau_m$：拟合偏高 {o['sd_overshoot']*100:.0f}%", f"$T\\approx{xm:.1f}\\tau_m$: fit {o['sd_overshoot']*100:.0f}% high"),
                xy=(xm, NR.sd_fit(xm)), xytext=(1.25, 4.2), fontsize=7.6, color=FT,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=FT))
    a1.legend(loc="upper left", bbox_to_anchor=(0.31, 1.0), fontsize=7.8, handlelength=2.0, borderaxespad=0.2)
    clean_axes(a1)
    a1.set_title(T("(a) 阈值–时程", "(a) threshold versus duration"), fontsize=9.0, weight="bold", loc="left", pad=4)
    # (b) threshold charge Q = A T
    x = np.linspace(1e-3, 5, 600)
    a2.plot(x, x * NR.sd_exact(x), color=EX, lw=2.0, zorder=3)
    a2.plot(x, x * NR.sd_fit(x), color=FT, lw=1.8, ls=(0, (4, 2)), zorder=3)
    a2.plot([0], [1], "o", ms=5, color=EX, zorder=5, clip_on=False)
    a2.plot([0], [LN2], "o", ms=5, mfc="white", mec=FT, mew=1.2, zorder=5, clip_on=False)
    a2.plot([LN2], [2 * LN2], "o", ms=5, mfc="white", mec=INK, mew=1.2, zorder=5)
    a2.set_xlim(0, 5); a2.set_ylim(0, 6)
    a2.set_xlabel(T(r"脉宽 $T/\tau_m$", r"pulse duration $T/\tau_m$"))
    a2.set_ylabel(r"$A_{\rm th}T/(A_{\rm rh}\tau_m)$")
    a2.annotate(T(r"$T\to0$：$A_{\rm rh}\tau_m$", r"$T\to0$: $A_{\rm rh}\tau_m$"), xy=(0.04, 1.0), xytext=(0.8, 0.95),
                fontsize=7.8, color=EX, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=EX))
    a2.annotate(T(r"拟合：$A_{\rm rh}\tau_{\rm ch}$", r"fit: $A_{\rm rh}\tau_{\rm ch}$"), xy=(0.04, LN2), xytext=(0.8, 0.45),
                fontsize=7.8, color=FT, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=FT))
    a2.text(0.2, 4.6, T("拟合是一条直线：\n$Q=A_{\\rm rh}(T+\\tau_{\\rm ch})$", "the fit is a straight line:\n$Q=A_{\\rm rh}(T+\\tau_{\\rm ch})$"),
            fontsize=7.8, color=FT, ha="left", va="center", linespacing=1.3)
    a2.text(4.0, 3.2, T("精确解：斜率\n趋于 $A_{\\rm rh}$", "exact: slope\ntends to $A_{\\rm rh}$"), fontsize=7.8, color=EX,
            ha="left", va="top", linespacing=1.3)
    clean_axes(a2)
    a2.set_title(T("(b) 阈值电荷 $Q=A_{\\rm th}T$", "(b) threshold charge $Q=A_{\\rm th}T$"), fontsize=9.0, weight="bold",
                 loc="left", pad=4)
    return fig
