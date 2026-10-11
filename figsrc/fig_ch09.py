"""Figures for Chapter 9: potentials as different 'coordinates' of the same field."""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from style import (ACCENT, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, LIGHT, MAG, MAG_T, MUTED, POT, POT_T, T,
                   arrow, canvas, card, clean_axes, curved_arrow, dot, figure, flux_lines, halo, into_page,
                   label, out_of_page, panel_title, pill, plt, rbox, tint)


def _square(ax, L):
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.axis("off")


def _quiver(ax, X, Y, U, V, color, scale, width=0.011, z=4, alpha=1.0):
    return ax.quiver(X, Y, U, V, color=color, angles="xy", scale_units="xy", scale=scale, width=width,
                     headwidth=3.6, headlength=4.0, headaxislength=3.5, zorder=z, alpha=alpha, pivot="mid")


def _loop(ax, r, at, color=ACCENT, lw0=0.6, lw1=4.0, amax=None):
    """Circle of radius r drawn with line width proportional to the tangential component at(theta)."""
    th = np.linspace(0, 2 * np.pi, 361)
    pts = np.stack((r * np.cos(th), r * np.sin(th)), axis=1)
    segs = np.stack((pts[:-1], pts[1:]), axis=1)
    v = at(0.5 * (th[:-1] + th[1:]))
    amax = amax or np.max(np.abs(v))
    lws = lw0 + (lw1 - lw0) * np.abs(v) / amax
    lc = LineCollection(segs, colors=[color], linewidths=lws, zorder=6, capstyle="round")
    ax.add_collection(lc)
    a0 = np.deg2rad(135)
    arrow(ax, (r * np.cos(a0), r * np.sin(a0)), (r * np.cos(a0 + 0.08), r * np.sin(a0 + 0.08)),
          color=color, lw=1.4, ms=11, z=7)


# ---------------------------------------------------------------------------
@figure("gauge_pair")
def gauge_pair():
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.45))
    L = 1.25
    g = np.linspace(-1.0, 1.0, 9)
    X, Y = np.meshgrid(g, g)
    rC = 0.75
    specs = [
        (axs[0], (-Y / 2, X / 2), lambda t: 0.5 * rC + 0 * t,
         T(r"$\mathbf{A}_{\rm sym}=\frac{B_0}{2}(-y,\,x,\,0)$：绕轴打转", r"$\mathbf{A}_{\rm sym}=\frac{B_0}{2}(-y,\,x,\,0)$: swirls"),
         T("沿 $C$ 处处相同：$A_t=B_0r/2$", "uniform along $C$: $A_t=B_0r/2$")),
        (axs[1], (0 * X, X), lambda t: rC * np.cos(t) ** 2,
         T(r"$\mathbf{A}_{\rm alt}=B_0(0,\,x,\,0)$：全部沿 $y$", r"$\mathbf{A}_{\rm alt}=B_0(0,\,x,\,0)$: all along $y$"),
         T(r"左右两侧大，上下为零：$A_t=B_0r\cos^2\theta$", r"large at left/right, zero at top/bottom"))]
    for ax, (U, V), at, title, note in specs:
        _square(ax, L)
        ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fc=tint(MAG, 0.06), ec="none", zorder=0))
        for xx in np.linspace(-1.0, 1.0, 5):
            for yy in np.linspace(-1.0, 1.0, 5):
                out_of_page(ax, xx + 0.125, yy + 0.125, color=tint(MAG, 0.55), r=0.045, lw=0.7, z=1)
        _quiver(ax, X, Y, U, V, POT, scale=4.2, width=0.0105, z=4)
        _loop(ax, rC, at, amax=rC)
        halo(label(ax, 0, -1.13, r"$\oint_C\mathbf{A}\cdot\mathrm{d}\boldsymbol{\ell}=\pi r^2B_0$", color=ACCENT, size=9.4, z=9), 3)
        halo(label(ax, rC * 0.72 + 0.14, rC * 0.72 + 0.12, r"$C$", color=ACCENT, size=10, z=9), 3)
        ax.set_title(title, color=POT, fontsize=9.2, fontweight="bold", pad=17)
        label(ax, 0, L + 0.13, note, color=MUTED, size=8.0, va="center")
    fig.text(0.5, 0.015, T(r"浅红色 $\odot$：处处相同的 $\mathbf{B}=B_0\mathbf{e}_z$（指出纸面）；紫色箭头：$\mathbf{A}$；回路 $C$ 的线宽正比于 $\mathbf{A}$ 沿回路的切向分量",
                           r"light $\odot$: the same uniform $\mathbf{B}=B_0\mathbf{e}_z$; purple arrows: $\mathbf{A}$; width of $C$ $\propto$ tangential component of $\mathbf{A}$"),
             ha="center", fontsize=8.0, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.07, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
@figure("gauge_efield")
def gauge_efield():
    """-dA_alt/dt + (-grad phi_alt) = E = -dA_sym/dt, in units of dB/dt * r0 / 2 (r0 = 8 cm)."""
    fig = plt.figure(figsize=(FIG_W, 2.65))
    w, gap = 0.27, 0.075
    xs = [0.02, 0.02 + w + gap, 0.02 + 2 * (w + gap)]
    axs = [fig.add_axes([x, 0.06, w, 0.74]) for x in xs]
    L = 1.3
    g = np.linspace(-1.0, 1.0, 7)
    X, Y = np.meshgrid(g, g)
    # dB/dt > 0 ; units: x, y in r0 ; E in dB r0 / 2  (= 40 V/m for 1000 T/s, 8 cm)
    fields = [(0 * X, -2 * X), (Y, X), (Y, -X)]
    titles = [T(r"$-\partial_t\mathbf{A}_{\rm alt}=(0,\,-\dot B x)$", r"$-\partial_t\mathbf{A}_{\rm alt}=(0,\,-\dot B x)$"),
              T(r"$-\nabla\phi_{\rm alt},\ \ \phi_{\rm alt}=-\dot Bxy/2$", r"$-\nabla\phi_{\rm alt},\ \ \phi_{\rm alt}=-\dot Bxy/2$"),
              T(r"$\mathbf{E}=\frac{\dot B}{2}(y,\,-x)=-\partial_t\mathbf{A}_{\rm sym}$", r"$\mathbf{E}=\frac{\dot B}{2}(y,\,-x)=-\partial_t\mathbf{A}_{\rm sym}$")]
    vals = [("80", "0"), ("40", "40"), ("40", "40")]
    cols = [tint(ELEC, 0.75), tint(ELEC, 0.75), ELEC]
    gg = np.linspace(-L, L, 301)
    GX, GY = np.meshgrid(gg, gg)
    for k, (ax, (U, V), title, (vp, vq), c) in enumerate(zip(axs, fields, titles, vals, cols)):
        _square(ax, L)
        ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fc=tint(MAG, 0.05), ec="none", zorder=0))
        if k == 1:
            ax.contour(GX, GY, -GX * GY, levels=np.linspace(-1.4, 1.4, 15), colors=[POT], linewidths=0.7,
                       linestyles=[(0, (3, 2))], zorder=1)
        if k == 2:
            ax.add_patch(Circle((0, 0), 1.0, fc="none", ec=tint(ELEC, 0.45), lw=0.8, zorder=1))
        _quiver(ax, X, Y, U, V, c, scale=4.4, width=0.012, z=4)
        # sample points P = (r0, 0) and Q = (0, r0)
        for (px, py), txt, dx_, dy_ in (((1.0, 0.0), vp, 0.18, -0.02), ((0.0, 1.0), vq, 0.0, 0.2)):
            ax.plot([px], [py], "o", ms=5.5, mfc="white", mec=INK, mew=1.1, zorder=8)
            halo(label(ax, px + dx_, py + dy_, txt, size=8.6, ha="left" if dx_ else "center", color=INK, z=9, weight="bold"), 3)
        ax.set_title(title, fontsize=8.6, color=ELEC if k == 2 else INK, pad=3)
    halo(label(axs[0], 1.0, -0.25, "P", size=8.4, color=MUTED, z=9), 2.5)
    halo(label(axs[0], -0.2, 1.0, "Q", size=8.4, color=MUTED, z=9), 2.5)
    for xm, s in ((xs[1] - gap / 2, "+"), (xs[2] - gap / 2, "=")):
        fig.text(xm, 0.43, s, ha="center", va="center", fontsize=20, color=INK)
    fig.text(0.5, 0.925, T(r"同一个变化的均匀磁场，非对称规范中的两部分电场相加，得到与对称规范完全相同的 $\mathbf{E}$（数字：P、Q 处的大小，V/m）",
                           r"Alternative gauge: the two parts add up to exactly the symmetric-gauge $\mathbf{E}$ (numbers: magnitude at P, Q in V/m)"),
             ha="center", fontsize=8.2, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("potential_bridge")
def potential_bridge():
    fig, ax = canvas(FIG_W, 3.05, xlim=(0, 10.4), ylim=(0, 5.3))
    cw, ch = 2.9, 1.15
    xl, xr = 0.35, 7.15
    yt, yb = 3.55, 0.75
    card(ax, xl, yt, cw, ch, T(r"电标势 $\phi$", r"scalar potential $\phi$"), [T("单位 V", "unit: V")], color=POT, band=0.42)
    card(ax, xr, yt, cw, ch, T(r"磁矢势 $\mathbf{A}$", r"vector potential $\mathbf{A}$"), [T("单位 Wb/m = V·s/m", "unit: Wb/m = V·s/m")], color=POT, band=0.42)
    card(ax, xl, yb, cw, ch, T(r"电场 $\mathbf{E}$", r"electric field $\mathbf{E}$"), [T("单位 V/m", "unit: V/m")], color=ELEC, band=0.42)
    card(ax, xr, yb, cw, ch, T(r"磁通密度 $\mathbf{B}$", r"flux density $\mathbf{B}$"), [T("单位 T", "unit: T")], color=MAG, band=0.42)
    # arrows
    arrow(ax, (xl + cw / 2, yt - 0.05), (xl + cw / 2, yb + ch + 0.05), color=POT, lw=1.8, ms=13)
    halo(label(ax, xl + cw / 2 + 0.12, 2.92, r"$-\nabla\phi$", color=POT, size=11, ha="left"), 3)
    arrow(ax, (xr + cw / 2, yt - 0.05), (xr + cw / 2, yb + ch + 0.05), color=POT, lw=1.8, ms=13)
    halo(label(ax, xr + cw / 2 - 0.12, 2.92, r"$\nabla\times\mathbf{A}$", color=POT, size=11, ha="right"), 3)
    arrow(ax, (xr - 0.05, yt + 0.25), (xl + cw + 0.05, yb + ch - 0.25), color=POT, lw=1.8, ms=13)
    halo(label(ax, 5.25, 2.55, r"$-\partial_t\mathbf{A}$", color=POT, size=11), 3)
    # gauge box between potentials
    rbox(ax, 3.55, 4.05, 3.3, 0.9, fc="white", ec=tint(ACCENT, 0.55), lw=0.9, r=0.1, ls=(0, (3, 2)))
    label(ax, 5.2, 4.68, T("规范变换（E、B 不变）", "gauge change (E, B unchanged)"), size=8.4, color=ACCENT, weight="bold")
    label(ax, 5.2, 4.3, r"$\phi\to\phi-\partial_t\chi,\quad \mathbf{A}\to\mathbf{A}+\nabla\chi$", size=9.0, color=ACCENT)
    # built-in constraints
    pill(ax, xr + cw / 2, 0.32, T(r"自动满足 $\nabla\cdot\mathbf{B}=0$（V3）", r"automatically $\nabla\cdot\mathbf{B}=0$ (V3)"), color=MAG, size=8.4)
    pill(ax, xl + cw / 2, 0.32, T(r"自动满足 $\nabla\times\mathbf{E}=-\partial_t\mathbf{B}$", r"automatically $\nabla\times\mathbf{E}=-\partial_t\mathbf{B}$"), color=ELEC, size=8.4)
    label(ax, 5.2, 1.05, T("另外两条方程（高斯、安培）\n变成关于 φ、A 的方程", "Gauss's and Ampère's laws\nbecome equations for φ, A"),
          size=8.2, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("az_contours")
def az_contours():
    fig = plt.figure(figsize=(FIG_W, 3.0))
    ax = fig.add_axes([0.0, 0.02, 0.5, 0.9])
    L = 2.1
    _square(ax, L)
    rw = 0.16
    g = np.linspace(-L, L, 801)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y) + 1e-12
    Az = -np.log(R)                    # units of mu0 I / (2 pi), r0 = 1
    Bx, By = -Y / R**2, X / R**2       # counterclockwise for current out of the page
    q = 1.45
    radii = 0.3 * q ** np.arange(0, 5)
    levels = np.sort(-np.log(radii))
    ax.add_patch(Circle((0, 0), L * 1.5, fc=tint(MAG, 0.04), ec="none", zorder=0))
    flux_lines(ax, X, Y, np.ma.array(Az, mask=R < rw), levels, Bx, By, color=MAG, lw=1.1, arrows=True,
               arrow_pos=0.12, ms=8)
    ax.add_patch(Circle((0, 0), rw, fc=tint(CURR, 0.25), ec=CURR, lw=1.0, zorder=6))
    out_of_page(ax, 0, 0, color=CURR, r=0.085)
    # mark two adjacent contours and the flux between them
    r1, r2 = radii[2], radii[3]
    ang = np.deg2rad(-35)
    p1 = r1 * np.array([np.cos(ang), np.sin(ang)])
    p2 = r2 * np.array([np.cos(ang), np.sin(ang)])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=POT, lw=2.4, zorder=7, solid_capstyle="butt")
    for p in (p1, p2):
        dot(ax, *p, color=POT, r=0.045, z=8)
    halo(label(ax, p1[0] - 0.08, p1[1] - 0.22, r"$A_1$", color=POT, size=9.5, z=9), 3)
    halo(label(ax, p2[0] + 0.18, p2[1] - 0.12, r"$A_2$", color=POT, size=9.5, z=9), 3)
    label(ax, 0.0, -1.85, T(r"相邻两线之间：每米深度的磁通 $=A_1-A_2$（处处相同）", r"neighbouring lines: flux per meter $=A_1-A_2$" + "\n" + "(the same everywhere)"),
          size=8.1, color=POT, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    label(ax, 0.0, 2.02, T(r"线电流 $I$ 指出纸面：$A_z$ 等值线（红）就是 $\mathbf{B}$ 线", r"line current out of page: $A_z$ contours are $\mathbf{B}$ lines"),
          size=8.4, weight="bold")
    # ---- right: profiles for two reference radii
    bx = fig.add_axes([0.6, 0.57, 0.38, 0.33])
    cx = fig.add_axes([0.6, 0.14, 0.38, 0.33], sharex=bx)
    r = np.linspace(rw, 3.0, 400)
    bx.plot(r, -np.log(r), color=POT, lw=2.0, label=r"$r_0=a$")
    bx.plot(r, -np.log(r / 2), color=POT, lw=1.6, ls=(0, (4, 2.5)), label=r"$r_0=2a$")
    bx.axhline(0, color=MUTED, lw=0.6)
    bx.set_ylabel(r"$A_z\,/\,\frac{\mu_0I}{2\pi}$", fontsize=9)
    bx.legend(loc="upper right", fontsize=8.0, handlelength=2.0, borderaxespad=0.1)
    bx.text(0.25, -1.75, T("两条曲线只差一个常数", "the curves differ by a constant"), fontsize=8.2, color=POT)
    bx.set_ylim(-2.05, 2.8)
    plt.setp(bx.get_xticklabels(), visible=False)
    clean_axes(bx)
    cx.plot(r, 1 / r, color=MAG, lw=2.0)
    cx.set_ylabel(r"$B_\theta\,/\,\frac{\mu_0I}{2\pi a}$", fontsize=9)
    cx.set_xlabel(T(r"半径 $r/a$", r"radius $r/a$"))
    cx.set_ylim(0, 4)
    cx.text(0.75, 2.3, T(r"两种 $r_0$ 给出同一个" + "\n" + r"$B_\theta=-\partial_rA_z$", r"both $r_0$ give the same" + "\n" + r"$B_\theta=-\partial_rA_z$"), fontsize=8.2, color=MAG, va="center")
    cx.set_xlim(0, 3)
    clean_axes(cx)
    return fig


# ---------------------------------------------------------------------------
@figure("scalar_cut")
def scalar_cut():
    fig = plt.figure(figsize=(FIG_W, 2.95))
    ax = fig.add_axes([0.0, 0.0, 0.47, 0.98])
    L = 1.55
    _square(ax, L)
    ax.set_ylim(-1.78, 1.6)
    th = np.linspace(0, 2 * np.pi, 400)
    # H circles
    for rr in (0.5, 0.85, 1.2):
        ax.plot(rr * np.cos(th), rr * np.sin(th), color=MAG, lw=1.1, zorder=2)
        for a0 in (np.pi / 2 + 0.3, 3 * np.pi / 2 + 0.3):
            p = rr * np.array([np.cos(a0), np.sin(a0)])
            t = np.array([-np.sin(a0), np.cos(a0)])
            arrow(ax, tuple(p - 0.02 * t), tuple(p + 0.02 * t), color=MAG, lw=1.1, ms=8, z=3)
    # psi contours: rays at equal steps, labelled
    n = 8
    for k in range(n):
        a0 = 2 * np.pi * k / n
        c, s = np.cos(a0), np.sin(a0)
        if k == 0:
            continue
        ax.plot([0.2 * c, 1.42 * c], [0.2 * s, 1.42 * s], color=POT, lw=0.8, ls=(0, (3, 2)), zorder=1)
        txt = rf"$-\frac{{{k}}}{{8}}I$" if k != 4 else r"$-\frac{1}{2}I$"
        halo(label(ax, 1.43 * c + 0.06 * c, 1.43 * s + 0.06 * s, txt, color=POT, size=8.2, z=9), 2.5)
    # the cut
    ax.plot([0.2, 1.5], [0, 0], color=INK, lw=2.4, ls=(0, (2.5, 1.5)), zorder=5)
    halo(label(ax, 1.0, 0.16, T("割线", "cut"), color=INK, size=8.6, weight="bold", z=9), 3)
    halo(label(ax, 1.05, -0.17, T(r"上 $\psi=0$，下 $\psi\to-I$", r"above $\psi=0$, below $\psi\to-I$"), color=INK, size=7.8, z=9), 3)
    ax.add_patch(Circle((0, 0), 0.16, fc=tint(CURR, 0.25), ec=CURR, lw=1.0, zorder=6))
    out_of_page(ax, 0, 0, color=CURR, r=0.08)
    halo(label(ax, -0.36, -1.05, r"$\mathbf{H}=-\nabla\psi$", color=MAG, size=9.5, z=9), 3)
    label(ax, 0, -1.72, T(r"紫色虚线：$\psi$ 等值线，步长 $I/8$", r"purple dashes: $\psi$ contours, step $I/8$"), color=POT, size=8.0)
    # ---- right: psi along a loop
    bx = fig.add_axes([0.57, 0.2, 0.41, 0.66])
    t = np.linspace(0, 4 * np.pi, 801)
    bx.plot(t, -t / (2 * np.pi), color=MUTED, lw=1.2, ls=(0, (4, 2.5)), label=T("连续地绕下去：多值", "keep going: multivalued"))
    branch = -(np.mod(t, 2 * np.pi)) / (2 * np.pi)
    for k in range(2):
        seg = (t >= 2 * np.pi * k) & (t < 2 * np.pi * (k + 1))
        bx.plot(t[seg], branch[seg], color=POT, lw=2.2, label=T("割线处跳回：单值分支", "single-valued branch, jumps at cut") if k == 0 else None)
    bx.plot([2 * np.pi, 2 * np.pi], [-1, 0], color=POT, lw=1.0, ls=":")
    bx.annotate(T("绕一圈：$\\Delta\\psi=-I$", "one turn: $\\Delta\\psi=-I$"), xy=(2 * np.pi, -1.0), xytext=(7.6, -1.45),
                fontsize=8.3, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    bx.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi], ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
    bx.set_yticks([-2, -1, 0])
    bx.set_xlim(0, 4 * np.pi)
    bx.set_ylim(-2.1, 0.25)
    bx.set_xlabel(T(r"沿回路转过的角度 $\theta$", r"angle travelled around the loop $\theta$"))
    bx.set_ylabel(r"$\psi\,/\,I$")
    bx.legend(loc="lower left", fontsize=7.8, handlelength=2.0, borderaxespad=0.1)
    bx.set_title(T(r"$\psi=-I\theta/(2\pi)$：局部无旋，绕一圈却回不到原值", r"$\psi=-I\theta/(2\pi)$: locally curl-free, not single-valued"), fontsize=8.8, pad=5)
    clean_axes(bx, grid=True)
    return fig


# ---------------------------------------------------------------------------
@figure("stream_function")
def stream_function():
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.15))
    L = 1.18
    g = np.linspace(-1, 1, 401)
    X, Y = np.meshgrid(g, g)
    psi = (1 - X**2) * (1 - Y**2)
    Kx, Ky = -2 * Y * (1 - X**2), 2 * X * (1 - Y**2)
    # ---- left: contours and K arrows
    ax = axs[0]
    _square(ax, L)
    ax.add_patch(Rectangle((-1, -1), 2, 2, fc=CURR_T, ec=tint(CURR, 0.7), lw=1.0, zorder=0))
    cs = ax.contour(X, Y, psi, levels=np.linspace(0.1, 0.9, 9), colors=[POT], linewidths=0.8, zorder=2)
    gq = np.linspace(-0.8, 0.8, 7)
    QX, QY = np.meshgrid(gq, gq)
    QKx, QKy = -2 * QY * (1 - QX**2), 2 * QX * (1 - QY**2)
    _quiver(ax, QX, QY, QKx, QKy, CURR, scale=6.0, width=0.011, z=4)
    halo(label(ax, 0, 0, r"$\psi_0$", color=POT, size=9.5, z=9), 3)
    label(ax, 0, -1.11, T(r"边界上 $\psi_s=0$：没有电流流出板边", r"$\psi_s=0$ on the edge: no current leaves the plate"), size=8.0, color=MUTED)
    panel_title(ax, T(r"$\psi_s=\psi_0(1-x^2)(1-y^2)$，$\mathbf{K}_s=\nabla_s\psi_s\times\mathbf{e}_z$", r"$\psi_s=\psi_0(1-x^2)(1-y^2)$, $\mathbf{K}_s=\nabla_s\psi_s\times\mathbf{e}_z$"),
                color=INK, size=8.8)
    # ---- right: discrete winding
    ax = axs[1]
    _square(ax, L)
    ax.add_patch(Rectangle((-1, -1), 2, 2, fc=tint(CURR, 0.05), ec=tint(CURR, 0.5), lw=0.8, zorder=0))
    N = 5
    lev = (np.arange(N) + 0.5) / N
    cs = ax.contour(X, Y, psi, levels=lev, colors=[CURR], linewidths=2.0, zorder=3)
    for seg in cs.allsegs:
        for path in seg:
            k = int(np.argmin(np.abs(path[:, 1]) + 10 * (path[:, 0] < 0)))
            p = path[k]
            arrow(ax, (p[0], p[1] - 0.02), (p[0], p[1] + 0.02), color=CURR, lw=1.6, ms=9, z=5)
    # cut from centre to edge with bracket
    ax.plot([0, 0], [0, -1], color=ACCENT, lw=1.3, ls=(0, (3, 2)), zorder=6)
    dot(ax, 0, 0, color=ACCENT, r=0.03, z=7)
    halo(label(ax, 0.06, -0.5, T("穿过这条虚线的\n总电流 $=\\psi_0-0$", "total current across\nthis line $=\\psi_0-0$"), color=ACCENT, size=8.0, ha="left", z=9), 3)
    label(ax, 0, -1.11, T(r"每条导线承载 $\Delta\psi=\psi_0/N$（$N=5$）", r"each wire carries $\Delta\psi=\psi_0/N$ ($N=5$)"), size=8.0, color=MUTED)
    panel_title(ax, T("取等步长的等值线作导线", "equally spaced contours as wires"), color=CURR, size=8.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06)
    return fig


# ---------------------------------------------------------------------------
@figure("four_potentials")
def four_potentials():
    fig = plt.figure(figsize=(FIG_W, 2.75))
    w = 0.225
    xs = [0.02, 0.265, 0.53, 0.775]
    axs = [fig.add_axes([x, 0.2, w, 0.62]) for x in xs]
    L = 1.15
    g = np.linspace(-L, L, 461)
    X, Y = np.meshgrid(g, g)
    d = 0.5
    r1 = np.hypot(X + d, Y) + 1e-12
    r2 = np.hypot(X - d, Y) + 1e-12
    two_line = np.log(r2 / r1)               # +source at (-d, 0), -source at (+d, 0)
    mask2 = (r1 < 0.09) | (r2 < 0.09)
    quad = (X**2 - Y**2) / 2                   # psi for H = -grad psi = (-x, y)
    plate = (1 - X**2) * (1 - Y**2)
    gq = np.linspace(-0.9, 0.9, 7)
    QX, QY = np.meshgrid(gq, gq)

    def unit(U, V, ln=0.2):
        n = np.hypot(U, V)
        small = n < 1e-9 * max(n.max(), 1e-30)
        n[small] = 1
        U, V = U / n * ln, V / n * ln
        U[small] = np.nan
        V[small] = np.nan
        return U, V

    # ---- 1: phi of two line charges, E = -grad phi
    ax = axs[0]
    _square(ax, L)
    ax.contour(X, Y, np.ma.array(two_line, mask=mask2), levels=np.linspace(-2.4, 2.4, 13), colors=[POT], linewidths=0.75, zorder=1, negative_linestyles="solid")
    q1 = np.hypot(QX + d, QY); q2 = np.hypot(QX - d, QY)
    Ex = (QX + d) / q1**2 - (QX - d) / q2**2
    Ey = QY / q1**2 - QY / q2**2
    keep = (q1 > 0.25) & (q2 > 0.25)
    _quiver(ax, QX[keep], QY[keep], *unit(Ex[keep], Ey[keep]), ELEC, scale=1, width=0.016, z=4)
    from style import charge
    charge(ax, -d, 0, "+", r=0.1, size=7.5)
    charge(ax, d, 0, "-", r=0.1, size=7.5)
    # ---- 2: magnetic scalar potential, quadrupole, H = -grad psi
    ax = axs[1]
    _square(ax, L)
    ax.contour(X, Y, quad, levels=np.linspace(-0.6, 0.6, 13), colors=[POT], linewidths=0.75, zorder=1, negative_linestyles="solid")
    _quiver(ax, QX, QY, *unit(-QX, QY.copy()), MAG, scale=1, width=0.016, z=4)
    # ---- 3: A_z of two antiparallel line currents, B = grad A_z x e_z  (same function as panel 1)
    ax = axs[2]
    _square(ax, L)
    ax.contour(X, Y, np.ma.array(two_line, mask=mask2), levels=np.linspace(-2.4, 2.4, 13), colors=[POT], linewidths=0.75, zorder=1, negative_linestyles="solid")
    # grad(two_line) = (-Ex, -Ey) ; B = (d_y A, -d_x A) = (-Ey, Ex)
    _quiver(ax, QX[keep], QY[keep], *unit(-Ey[keep], Ex[keep]), MAG, scale=1, width=0.016, z=4)
    out_of_page(ax, -d, 0, color=CURR, r=0.085)
    into_page(ax, d, 0, color=CURR, r=0.085)
    # ---- 4: surface stream function, K = grad psi_s x n
    ax = axs[3]
    _square(ax, L)
    ax.add_patch(Rectangle((-1, -1), 2, 2, fc=CURR_T, ec=tint(CURR, 0.6), lw=0.8, zorder=0))
    ax.contour(X, Y, np.ma.array(plate, mask=(np.abs(X) > 1) | (np.abs(Y) > 1)), levels=np.linspace(0.1, 0.9, 9), colors=[POT], linewidths=0.75, zorder=1, negative_linestyles="solid")
    _quiver(ax, QX, QY, *unit(-2 * QY * (1 - QX**2), 2 * QX * (1 - QY**2)), CURR, scale=1, width=0.016, z=4)
    heads = [(r"$\phi\ \to\ \mathbf{E}=-\nabla\phi$", ELEC, T("电标势 · V", "electric potential · V")),
             (r"$\psi\ \to\ \mathbf{H}=-\nabla\psi$", MAG, T("磁标势 · A", "magnetic scalar potential · A")),
             (r"$A_z\ \to\ \mathbf{B}=\nabla A_z\times\mathbf{e}_z$", MAG, T("二维磁矢势 · Wb/m", "2-D vector potential · Wb/m")),
             (r"$\psi_s\to\mathbf{K}_s=\nabla_s\psi_s\times\mathbf{n}$", CURR, T("表面流函数 · A", "surface stream function · A"))]
    for ax, (h, c, sub) in zip(axs, heads):
        ax.text(0.5, 1.05, h, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6, color=c)
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=7.9, color=MUTED)
    notes = [T("两根异号线电荷", "two opposite line charges"), T("无电流区的四极场", "current-free quadrupole"),
             T("两根反向线电流", "two antiparallel currents"), T("方形平板上的电流", "current on a square plate")]
    for ax, n in zip(axs, notes):
        ax.text(0.5, -0.155, n, transform=ax.transAxes, ha="center", va="top", fontsize=7.9, color=INK)
    # group brackets
    for (x0, x1, txt, c) in ((xs[0], xs[1] + w, T("梯度型：场垂直于等值线", "gradient type: field normal to contours"), ACCENT),
                             (xs[2], xs[3] + w, T("流函数型：场沿着等值线", "stream type: field along contours"), ACCENT)):
        fig.add_artist(plt.Line2D([x0 + 0.01, x1 - 0.01], [0.925, 0.925], color=c, lw=1.2))
        for xx in (x0 + 0.01, x1 - 0.01):
            fig.add_artist(plt.Line2D([xx, xx], [0.9, 0.925], color=c, lw=1.2))
        fig.text((x0 + x1) / 2, 0.945, txt, ha="center", va="bottom", fontsize=8.8, color=c, weight="bold")
    return fig
