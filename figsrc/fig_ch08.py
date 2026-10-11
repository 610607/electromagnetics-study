"""Figures for Chapter 8: static, quasistatic and wave regimes; diffusion and skin depth."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from style import (ACCENT, AMBER, CURR, ELEC, FIG_W, INK, LIGHT, MAG, MATL, MUTED, POT, T,
                   arrow, canvas, clean_axes, figure, halo, label, plt, rbox, tint)

MU0 = 4.0e-7 * np.pi
EPS0 = 8.8541878128e-12
C0 = 1.0 / np.sqrt(MU0 * EPS0)

TISSUE = "#A0814F"   # warm sand-brown: weakly conducting, tissue-like medium
COPPER = "#4F5D6B"   # slate grey: metal (copper)


# ---------------------------------------------------------------------------
@figure("regime_terms")
def regime_terms():
    """Which time-derivative couplings each regime keeps (no material numbers)."""
    fig, ax = canvas(FIG_W, FIG_W * (6.1 - 0.9) / 10.3, xlim=(0, 10.3), ylim=(0.9, 6.1))
    x0, wn = 0.05, 1.75            # regime-name column
    wc = 1.5                        # term columns
    xs = [x0 + wn + 0.1 + i * (wc + 0.08) for i in range(4)]
    xt = xs[-1] + wc + 0.12         # typical-problem column
    wt = 10.25 - xt
    top = 5.05
    hrow = 0.74
    gap = 0.07

    heads = [
        (r"$\partial_t\mathbf{B}$", T("法拉第定律中", "Faraday's law"), T("磁 → 电：感应", "B → E"), MAG),
        (r"$\partial_t\mathbf{D}$", T("安培定律中", "Ampère's law"), T("电 → 磁：位移电流", "E → B"), ELEC),
        (r"$\sigma\mathbf{E}\to\mathbf{B}$", T("感应电流的磁场", "induced current"), T("涡流反馈", "eddy feedback"), CURR),
        (r"$\partial_t\rho$", T("电荷积累率", "charge build-up"), T("进入 $\\nabla\\cdot\\mathbf{J}$", r"enters $\nabla\cdot\mathbf{J}$"), ELEC),
    ]
    for (sym, l1, l2, c), x in zip(heads, xs):
        label(ax, x + wc / 2, top + 0.68, sym, color=c, size=11)
        label(ax, x + wc / 2, top + 0.33, l1, size=7.9, color=INK)
        label(ax, x + wc / 2, top + 0.08, l2, size=7.6, color=MUTED)
    label(ax, xt + wt / 2, top + 0.5, T("典型问题", "typical problems"), size=8.6, weight="bold")
    label(ax, x0 + wn / 2, top + 0.5, T("近似", "regime"), size=8.6, weight="bold")

    K, D, NA, P = "keep", "drop", "na", "part"
    rows = [
        (T("静态", "Static"), r"$\omega=0$", [NA, NA, NA, NA],
         T("直流传导、静电\n静磁、永磁体", "DC conduction, electro-\nstatics, magnets")),
        (T("电准静态 EQS", "EQS"), r"$\eta_m\ll1$", [D, K, D, P],
         T("TES、电极\n容性耦合", "TES, electrodes,\ncapacitive coupling")),
        (T("磁准静态 MQS", "MQS"), r"$\eta_w\ll1$", [K, D, K, D],
         T("涡流、屏蔽\n梯度线圈与冷屏", "eddy currents, shields,\ngradient coil + cryostat")),
        (T("单向耦合 MQS", "One-way MQS"), T(r"$\eta_m\ll1$，非磁性", r"$\eta_m\ll1$, $\mu\approx\mu_0$"), [K, D, D, D],
         T("TMS\n梯度场在人体中", "TMS; gradient\nfields in the body")),
        (T("全波", "Full wave"), T("无", "none"), [K, K, K, K],
         T("MRI 射频、SAR", "MRI RF, SAR")),
    ]
    for i, (name, cond, cells, typ) in enumerate(rows):
        y = top - (i + 1) * (hrow + gap)
        rbox(ax, x0, y, wn, hrow, fc=tint(ACCENT, 0.10), ec=tint(ACCENT, 0.45), lw=0.8, r=0.08)
        label(ax, x0 + wn / 2, y + hrow * 0.66, name, size=8.6, weight="bold", color=ACCENT)
        label(ax, x0 + wn / 2, y + hrow * 0.28, cond, size=7.9, color=MUTED)
        for (sym, _, _, c), x, st in zip(heads, xs, cells):
            if st == K:
                rbox(ax, x, y, wc, hrow, fc=tint(c, 0.22), ec=tint(c, 0.7), lw=0.9, r=0.08)
                label(ax, x + wc / 2, y + hrow / 2, T("保留", "kept"), size=8.6, color=c, weight="bold")
            elif st == D:
                rbox(ax, x, y, wc, hrow, fc="white", ec=tint(MUTED, 0.45), lw=0.8, r=0.08, ls=(0, (3, 2)))
                label(ax, x + wc / 2, y + hrow / 2, T("略去", "dropped"), size=8.4, color=tint(MUTED, 0.85))
            elif st == NA:
                rbox(ax, x, y, wc, hrow, fc=tint(MUTED, 0.06), ec="none", lw=0, r=0.08)
                label(ax, x + wc / 2, y + hrow / 2, T("无时间变化", "no time change"), size=7.8, color=MUTED)
            else:  # partial: kept in general, may be dropped when eta_d << 1
                rbox(ax, x, y, wc, hrow, fc=tint(c, 0.22), ec=tint(c, 0.7), lw=0.9, r=0.08)
                ax.add_patch(Rectangle((x + 0.04, y + 0.04), wc - 0.08, hrow * 0.42, fc="white", ec="none", zorder=2.5))
                label(ax, x + wc / 2, y + hrow * 0.72, T("保留", "kept"), size=8.6, color=c, weight="bold")
                label(ax, x + wc / 2, y + hrow * 0.25, T(r"$\eta_d\ll1$ 时可略", r"drop if $\eta_d\ll1$"), size=7.6, color=MUTED, z=9)
        label(ax, xt + 0.1, y + hrow / 2, typ, size=7.9, ha="left", color=INK)
    return fig


# ---------------------------------------------------------------------------
@figure("skin_depth")
def skin_depth():
    f = np.logspace(1, 7, 400)
    w = 2 * np.pi * f
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FIG_W, 2.85))
    cases = [(0.5, 0.20, TISSUE), (5.8e7, 2e-3, COPPER)]
    f0 = 3000.0
    w0 = 2 * np.pi * f0
    for s, ell, c in cases:
        d = np.sqrt(2 / (w * MU0 * s))
        ax.loglog(f, d, color=c, lw=2.0)
        ax.axhline(ell, color=c, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
        d0 = np.sqrt(2 / (w0 * MU0 * s))
        ax.plot([f0], [d0], "o", ms=5.5, color=c, zorder=5)
        em = w * MU0 * s * ell**2
        bx.loglog(f, em, color=c, lw=2.0)
        bx.plot([f0], [w0 * MU0 * s * ell**2], "o", ms=5.5, color=c, zorder=5)
    for a in (ax, bx):
        a.axvline(f0, color=MUTED, lw=0.8, ls=":", zorder=0)
    # direct labels
    ax.text(1.6e5, 3.2, T(r"$\sigma=0.5\,$S/m", r"$\sigma=0.5\,$S/m"), fontsize=8.4, color=TISSUE, ha="left")
    ax.text(5e4, 4.5e-4, T(r"铜，$5.8\times10^7\,$S/m", "copper"), fontsize=8.4, color=COPPER, ha="left")
    ax.text(2.5e4, 0.27, r"$\ell=0.20\,$m", fontsize=8.0, color=TISSUE)
    ax.text(1.6e5, 2.7e-3, r"$\ell=2\,$mm", fontsize=8.0, color=COPPER)
    ax.annotate(r"$\delta=12.995\,$m", xy=(f0, 12.995), xytext=(9e3, 70), fontsize=8.4, color=TISSUE,
                arrowprops=dict(arrowstyle="-", color=TISSUE, lw=0.7))
    ax.annotate(r"$\delta=1.207\,$mm", xy=(f0, 1.207e-3), xytext=(14, 1.3e-4), fontsize=8.4, color=COPPER,
                arrowprops=dict(arrowstyle="-", color=COPPER, lw=0.7))
    ax.text(3.4e3, 1.6e-5, "3 kHz", fontsize=8.0, color=MUTED)
    ax.set_xlabel(T("频率 $f$ (Hz)", "frequency $f$ (Hz)"))
    ax.set_ylabel(T(r"趋肤深度 $\delta$ (m)", r"skin depth $\delta$ (m)"))
    ax.set_xlim(10, 1e7)
    ax.set_ylim(1e-5, 3e2)
    ax.set_title(T(r"趋肤深度 $\delta$ 与物体尺寸 $\ell$（虚线）", r"skin depth $\delta$ vs. size $\ell$ (dashed)"), fontsize=9.0, pad=5)
    clean_axes(ax)
    # right: magnetic induction number
    bx.axhspan(1, 1e6, color=tint(MAG, 0.08), lw=0, zorder=0)
    bx.axhline(1, color=MUTED, lw=0.9, zorder=1)
    bx.text(14, 1.6e2, T("感应电流显著改写磁场", "induced field matters"), fontsize=8.0, color=MAG)
    bx.text(12, 5.5e-3, T("感应电流的磁场可忽略", "induced field negligible"), fontsize=8.0, color=MUTED)
    bx.annotate(r"$\eta_m=4.74\times10^{-4}$", xy=(f0, 4.74e-4), xytext=(1.5e4, 2.5e-5), fontsize=8.4, color=TISSUE,
                arrowprops=dict(arrowstyle="-", color=TISSUE, lw=0.7))
    bx.annotate(r"$\eta_m=5.50$", xy=(f0, 5.50), xytext=(1.5e1, 9.0), fontsize=8.4, color=COPPER,
                arrowprops=dict(arrowstyle="-", color=COPPER, lw=0.7))
    bx.set_xlabel(T("频率 $f$ (Hz)", "frequency $f$ (Hz)"))
    bx.set_ylabel(r"$\eta_m=\omega\mu_0\sigma\ell^2$")
    bx.set_xlim(10, 1e7)
    bx.set_ylim(1e-6, 1e6)
    bx.set_title(T(r"$\eta_m=2(\ell/\delta)^2$：与 1 比较", r"$\eta_m=2(\ell/\delta)^2$ compared with 1"), fontsize=9.0, pad=5)
    clean_axes(bx)
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.17, top=0.9, wspace=0.36)
    return fig


# ---------------------------------------------------------------------------
@figure("diffusion")
def diffusion():
    from matplotlib.patches import Arc
    fig = plt.figure(figsize=(FIG_W, 2.7))
    ax = fig.add_axes([0.085, 0.17, 0.5, 0.72])
    x = np.linspace(0, 5, 600)
    env = np.exp(-x)
    ax.fill_between(x, -env, env, color=tint(MAG, 0.07), lw=0, zorder=0)
    ax.plot(x, env, color=MAG, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
    ax.plot(x, -env, color=MAG, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
    phases = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    shades = [1.0, 0.7, 0.48, 0.3]
    for ph, a_ in zip(phases, shades):
        ax.plot(x, env * np.cos(ph - x), color=MAG, lw=1.6, alpha=a_, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.plot([1, 1], [-1.05, 1.05], color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.plot([1], [np.exp(-1)], "o", ms=4.5, color=MAG, zorder=4)
    ax.annotate(T(r"$x=\delta$：幅值 $e^{-1}\approx0.37$", r"$x=\delta$: amplitude $e^{-1}\approx0.37$"),
                xy=(1, np.exp(-1)), xytext=(1.55, 0.62), fontsize=8.3, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(0.08, -0.93, T("包络 $\\pm e^{-x/\\delta}$（虚线）", r"envelope $\pm e^{-x/\delta}$ (dashed)"), fontsize=8.0, color=MAG)
    ax.annotate(T(r"$\omega t=0,\ \pi/4,\ \pi/2,\ 3\pi/4$" + "\n（颜色由深到浅）", r"$\omega t=0,\ \pi/4,\ \pi/2,\ 3\pi/4$" + "\n(dark to light)"),
                xy=(2.2, env[np.searchsorted(x, 2.2)] * np.cos(-2.2)), xytext=(2.6, -0.62), fontsize=8.0, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xlim(0, 5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(T(r"进入导体的深度 $x/\delta$", r"depth into the conductor $x/\delta$"))
    ax.set_ylabel(r"$H_y(x,t)/H_0$")
    ax.set_title(T(r"$H_y=H_0e^{-x/\delta}\cos(\omega t-x/\delta)$", r"$H_y=H_0e^{-x/\delta}\cos(\omega t-x/\delta)$"), fontsize=9.0, pad=5)
    clean_axes(ax)
    # ---- right: phasor spiral  H~(x)/H0 = exp(-(1+j) x/delta)
    bx = fig.add_axes([0.63, 0.30, 0.36, 0.5])
    xs = np.linspace(0, 4, 400)
    h = np.exp(-(1 + 1j) * xs)
    bx.axhline(0, color=MUTED, lw=0.6, zorder=1)
    bx.axvline(0, color=MUTED, lw=0.6, zorder=1)
    bx.plot(h.real, h.imag, color=MAG, lw=1.8, zorder=3)
    for k, txt, off in ((0, r"$x=0$", (0.0, 0.07)), (1, r"$x=\delta$", (0.07, -0.05)), (2, r"$2\delta$", (-0.06, -0.1))):
        p = np.exp(-(1 + 1j) * k)
        if k == 1:
            arrow(bx, (0, 0), (p.real, p.imag), color=tint(MAG, 0.6), lw=1.0, ms=7, z=4)
        bx.plot([p.real], [p.imag], "o", ms=4.2, color=MAG, zorder=5)
        halo(bx.text(p.real + off[0], p.imag + off[1], txt, fontsize=8.2, ha="center", color=INK, zorder=8), 2.5)
    bx.add_patch(Arc((0, 0), 0.62, 0.62, theta1=-np.degrees(1.0), theta2=0, color=ACCENT, lw=1.0, zorder=4))
    halo(bx.text(0.36, -0.13, "1 rad", fontsize=7.8, color=ACCENT, va="center"), 2.5)
    bx.set_xlim(-0.2, 1.12)
    bx.set_ylim(-0.42, 0.14)
    bx.set_aspect("equal")
    bx.set_xticks([0, 0.5, 1])
    bx.set_yticks([-0.3, 0])
    bx.tick_params(labelsize=7.8)
    bx.set_xlabel("Re", fontsize=8.5, labelpad=1)
    bx.set_ylabel("Im", fontsize=8.5, labelpad=2)
    bx.set_title(T(r"相量 $\widetilde H_y(x)/H_0=e^{-(1+\mathrm{j})x/\delta}$", r"phasor $\widetilde H_y(x)/H_0=e^{-(1+\mathrm{j})x/\delta}$"), fontsize=8.6, pad=5)
    clean_axes(bx)
    fig.text(0.81, 0.08, T("每深入一个 $\\delta$：幅值乘 $e^{-1}$，相位滞后 1 rad", "each $\\delta$ deeper: amplitude $\\times e^{-1}$,\nphase lags 1 rad"),
             ha="center", fontsize=8.0, color=INK)
    return fig


# ---------------------------------------------------------------------------
@figure("near_far")
def near_far():
    fig = plt.figure(figsize=(FIG_W, 2.75))
    ax = fig.add_axes([0.085, 0.17, 0.5, 0.72])
    kr = np.logspace(-2, 2, 400)
    # soft background: reactive (left) -> radiating (right), no sharp boundary
    n = 200
    xs = np.logspace(-2, 2, n + 1)
    for i in range(n):
        t = i / (n - 1)
        wgt = 1 / (1 + np.exp(-4.0 * np.log10(np.sqrt(xs[i] * xs[i + 1]))))
        col = np.array(tint(ELEC, 0.10)) * (1 - wgt) + np.array(tint(AMBER, 0.12)) * wgt
        ax.axvspan(xs[i], xs[i + 1], color=col, lw=0, zorder=0)
    styles = [(3, ELEC, "-", T(r"$1/r^3$ 项：$(kr)^{-3}$", r"$1/r^3$ term: $(kr)^{-3}$")),
              (2, POT, (0, (5, 2)), T(r"$k/r^2$ 项：$(kr)^{-2}$", r"$k/r^2$ term: $(kr)^{-2}$")),
              (1, AMBER, (0, (1.5, 1.5)), T(r"$k^2/r$ 项：$(kr)^{-1}$", r"$k^2/r$ term: $(kr)^{-1}$"))]
    for p, c, ls, lab in styles:
        ax.loglog(kr, kr ** (-p), color=c, lw=2.0, ls=ls, label=lab, zorder=3)
    ax.axvline(1, color=MUTED, lw=0.8, ls=":", zorder=2)
    ax.text(1.1, 3e5, r"$kr=1$", fontsize=8.2, color=MUTED)
    ax.text(0.013, 3e-6, T("← 反应性近场 $kr\\ll1$", "← reactive, $kr\\ll1$"), fontsize=8.2, color=ELEC)
    ax.text(35, 3e-6, T("辐射区 $kr\\gg1$ →", "radiating, $kr\\gg1$ →"), fontsize=8.2, color="#8A5A12", ha="right")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-6, 1e6)
    ax.set_xlabel(T(r"归一化距离 $kr$", r"normalized distance $kr$"))
    ax.set_ylabel(T("归一化包络", "normalized envelope"))
    ax.legend(loc="upper right", fontsize=7.8, handlelength=2.2, borderaxespad=0.15)
    clean_axes(ax)
    # ---- right: lengths at 128 MHz
    bx = fig.add_axes([0.70, 0.22, 0.28, 0.6])
    lam0 = C0 / 128e6
    lam80 = lam0 / np.sqrt(80)
    names = [T("自由空间波长\n$\\lambda_0$", "free-space\nwavelength $\\lambda_0$"),
             T("$\\varepsilon_r=80$ 无损\n介质中的波长", "wavelength in\nlossless $\\varepsilon_r=80$"),
             T("教学长度 $\\ell$", "teaching length $\\ell$")]
    vals = [lam0, lam80, 0.20]
    cols = [AMBER, AMBER, TISSUE]
    y = np.arange(3)[::-1]
    bx.barh(y, vals, color=[tint(c, 0.45) for c in cols], edgecolor=cols, lw=0.9, height=0.55)
    for yy, txt in zip(y, ("2.34 m", "0.262 m", "0.20 m")):
        bx.text(vals[2 - yy] + 0.06, yy, txt, va="center", fontsize=8.2)
    bx.set_yticks(y, names, fontsize=7.9)
    bx.set_xlim(0, 3.2)
    bx.set_xlabel(T("长度 (m)", "length (m)"), fontsize=8.6)
    bx.set_title(T("128 MHz 时的长度尺度", "length scales at 128 MHz"), fontsize=9.0, pad=5)
    clean_axes(bx)
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    return fig
