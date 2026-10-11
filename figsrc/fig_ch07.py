"""Figures for Chapter 7: capacitor, resistor, inductor, magnetic circuit; energy flow."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

from fig_ch04 import loop_B, loop_psi
from style import (ACCENT, AMBER, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, IRON_T, MAG, MAG_T, MATL, MATL_T, MUTED,
                   POT, T, arrow, canvas, clean_axes, figure, flux_lines, halo, into_page, label, out_of_page,
                   panel_title, plt, rbox, tint)

PLATE = "#9AA3AB"


def _chain(ax, x, y, s, color=ACCENT, size=8.7, w=None):
    """Compression chain shown as a rounded tag."""
    return ax.text(x, y, s, fontsize=size, color=INK, ha="center", va="center", zorder=10,
                   bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.25", fc=tint(color, 0.08), ec=tint(color, 0.45), lw=0.8))


def _ring_core(ax, x0, y0, w, h, t, gap, gx=None, fc=IRON_T, ec=None):
    """Rectangular ring core with an air gap in the right limb (centre height)."""
    ec = ec or tint("#7A8691", 0.9)
    gx = x0 + w - t
    yc = y0 + h / 2
    parts = [
        (x0, y0, w, t),                       # bottom
        (x0, y0 + h - t, w, t),               # top
        (x0, y0, t, h),                       # left limb
        (gx, y0, t, h / 2 - gap / 2),         # right limb, lower part
        (gx, yc + gap / 2, t, h / 2 - gap / 2),  # right limb, upper part
    ]
    for (x, y, ww, hh) in parts:
        ax.add_patch(Rectangle((x, y), ww, hh, fc=fc, ec="none", zorder=1))
    # outline
    xi0, yi0, xi1, yi1 = x0 + t, y0 + t, x0 + w - t, y0 + h - t
    ax.plot([x0, x0 + w, x0 + w], [y0, y0, yc - gap / 2], color=ec, lw=0.9, zorder=2)
    ax.plot([x0 + w, x0 + w, x0, x0], [yc + gap / 2, y0 + h, y0 + h, y0], color=ec, lw=0.9, zorder=2)
    ax.plot([xi0, xi1, xi1], [yi0, yi0, yc - gap / 2], color=ec, lw=0.9, zorder=2)
    ax.plot([xi1, xi1, xi0, xi0], [yc + gap / 2, yi1, yi1, yi0], color=ec, lw=0.9, zorder=2)
    ax.plot([xi1, x0 + w], [yc - gap / 2] * 2, color=ec, lw=0.9, zorder=2)
    ax.plot([xi1, x0 + w], [yc + gap / 2] * 2, color=ec, lw=0.9, zorder=2)
    return gx, yc


# ---------------------------------------------------------------------------
@figure("four_systems")
def four_systems():
    fig = plt.figure(figsize=(FIG_W, 4.05))
    W, H = 0.49, 0.495
    pos = [(0.0, 0.505), (0.51, 0.505), (0.0, 0.0), (0.51, 0.0)]
    axs = [fig.add_axes([x, y, W, H]) for x, y in pos]
    for ax in axs:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4.1)
        ax.set_aspect("equal")
        ax.axis("off")

    # ---------------- (a) capacitor
    ax = axs[0]
    xl, xr, yb, yt = 1.0, 5.0, 1.45, 3.05
    ax.add_patch(Rectangle((xl, yb), xr - xl, yt - yb, fc=MATL_T, ec="none", zorder=0))
    ax.add_patch(Rectangle((xl - 0.05, yt), xr - xl + 0.1, 0.13, fc=PLATE, ec="none", zorder=2))
    ax.add_patch(Rectangle((xl - 0.05, yb - 0.13), xr - xl + 0.1, 0.13, fc=PLATE, ec="none", zorder=2))
    for x in np.linspace(xl + 0.25, xr - 0.25, 8):
        label(ax, x, yt + 0.3, "+", color=MAG, size=8.5, weight="bold")
        label(ax, x, yb - 0.32, "−", color=ELEC, size=9.5, weight="bold")
    for x in np.linspace(xl + 0.45, xr - 0.45, 6):
        arrow(ax, (x, yt - 0.08), (x, yb + 0.08), color=ELEC, lw=1.3, ms=9, z=4)
    for s in (-1, 1):  # fringing hint
        xe = xl if s < 0 else xr
        th = np.linspace(-np.pi / 2, np.pi / 2, 50)
        ax.plot(xe + s * 0.38 * np.cos(th), (yt + yb) / 2 + 0.78 * np.sin(th), color=tint(ELEC, 0.4), lw=0.9, ls=(0, (3, 2)), zorder=1)
    label(ax, 0.55, yt + 0.06, r"$+Q$", color=MAG, size=9.5)
    label(ax, 0.55, yb - 0.08, r"$-Q$", color=ELEC, size=9.5)
    ax.annotate("", xy=(5.45, yt), xytext=(5.45, yb), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    label(ax, 5.62, (yt + yb) / 2, r"$d$", color=MUTED, size=9.5, ha="left")
    label(ax, 3.0, 3.85, T(r"电容器：$C=Q/V$", r"Capacitor: $C=Q/V$"), weight="bold", size=9.8, color=ELEC)
    _chain(ax, 3.0, 0.42, T(r"$Q\to D=\frac{Q}{A_c}\to E=\frac{D}{\varepsilon}\to V=Ed$　$\Rightarrow\ C=\frac{\varepsilon A_c}{d}$",
                            r"$Q\to D=\frac{Q}{A_c}\to E=\frac{D}{\varepsilon}\to V=Ed\ \Rightarrow\ C=\frac{\varepsilon A_c}{d}$"), color=ELEC)

    # ---------------- (b) resistor
    ax = axs[1]
    xl, xr, yb, yt = 1.0, 5.0, 1.75, 2.75
    ax.add_patch(Rectangle((xl, yb), xr - xl, yt - yb, fc=MATL_T, ec=tint(MATL, 0.7), lw=0.8, zorder=0))
    ax.add_patch(Rectangle((xl - 0.16, yb - 0.12), 0.16, yt - yb + 0.24, fc=PLATE, ec="none", zorder=2))
    ax.add_patch(Rectangle((xr, yb - 0.12), 0.16, yt - yb + 0.24, fc=PLATE, ec="none", zorder=2))
    for y in np.linspace(yb + 0.2, yt - 0.2, 3):
        arrow(ax, (xl + 0.2, y), (xr - 0.2, y), color=CURR, lw=1.3, ms=9, z=4)
    ax.plot([0.2, xl - 0.16], [(yb + yt) / 2] * 2, color=MUTED, lw=1.6, zorder=1)
    ax.plot([xr + 0.16, 5.8], [(yb + yt) / 2] * 2, color=MUTED, lw=1.6, zorder=1)
    arrow(ax, (0.25, (yb + yt) / 2 + 0.22), (0.7, (yb + yt) / 2 + 0.22), color=CURR, lw=1.5, ms=10)
    label(ax, 0.45, (yb + yt) / 2 + 0.5, r"$I$", color=CURR, size=10)
    ax.annotate("", xy=(xr, yb - 0.35), xytext=(xl, yb - 0.35), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    label(ax, 3.0, yb - 0.58, r"$\ell$", color=MUTED, size=9.5)
    label(ax, 3.0, yt + 0.32, T(r"$\mathbf{J}$（绿）与 $\mathbf{E}$ 同向，沿导体均匀", r"$\mathbf{J}$ (green) parallel to $\mathbf{E}$, uniform"), size=8.4, color=MUTED)
    label(ax, 3.0, 3.85, T(r"电阻：$R=V/I$", r"Resistor: $R=V/I$"), weight="bold", size=9.8, color=CURR)
    _chain(ax, 3.0, 0.42, T(r"$I\to J=\frac{I}{A_c}\to E=\frac{J}{\sigma}\to V=E\ell$　$\Rightarrow\ R=\frac{\ell}{\sigma A_c}$",
                            r"$I\to J=\frac{I}{A_c}\to E=\frac{J}{\sigma}\to V=E\ell\ \Rightarrow\ R=\frac{\ell}{\sigma A_c}$"), color=CURR)

    # ---------------- (c) solenoid inductor: exact field lines of 9 coaxial loops
    ax = axs[2]
    a = 0.55
    zs = np.linspace(-1.5, 1.5, 13)
    x0c, y0c = 3.0, 2.2     # centre of the drawing in panel coordinates (axis horizontal)
    gx = np.linspace(-2.9, 2.9, 581)   # along the axis
    gr = np.linspace(-1.75, 1.75, 351)  # across
    Zg, Rg = np.meshgrid(gx, gr)
    psi = sum(loop_psi(Rg, Zg - zk, a) for zk in zs)
    Br = np.zeros_like(Zg)
    Bz = np.zeros_like(Zg)
    for zk in zs:
        bx, bz = loop_B(Rg, Zg - zk, a)
        Br += bx
        Bz += bz
    mask = np.zeros_like(psi, bool)
    for zk in zs:
        mask |= np.hypot(np.abs(Rg) - a, Zg - zk) < 0.08
    # inside a long solenoid psi ~ rho^2: calibrate on rho = 0.5a, where no wire is close
    ref = float(sum(loop_psi(np.array([0.5 * a]), np.array([0.0 - zk]), a)[0] for zk in zs)) / 0.25
    levels = ref * np.array([0.05, 0.2, 0.37, 0.54])
    sub = ax.inset_axes([0.0, 0.17, 1.0, 0.72], transform=ax.transAxes)
    sub.set_xlim(-2.9, 2.9)
    sub.set_ylim(-1.75, 1.75)
    sub.set_aspect("equal")
    sub.axis("off")
    flux_lines(sub, Zg, Rg, np.ma.array(psi, mask=mask), levels, Bz, Br, color=MAG, lw=0.95, arrows=True,
               arrow_pos=0.5, ms=6.5)
    for zk in zs:
        out_of_page(sub, zk, a, color=CURR, r=0.075, lw=0.8, z=7)
        into_page(sub, zk, -a, color=CURR, r=0.075, lw=0.8, z=7)
    label(ax, 3.0, 3.85, T(r"电感：$L=\lambda/I$", r"Inductor: $L=\lambda/I$"), weight="bold", size=9.8, color=MAG)
    _chain(ax, 3.0, 0.42, T(r"$I\to B=\frac{\mu NI}{\ell}\to\Phi=BA_c\to\lambda=N\Phi$　$\Rightarrow\ L=\frac{\mu N^2A_c}{\ell}$",
                            r"$I\to B=\frac{\mu NI}{\ell}\to\Phi=BA_c\to\lambda=N\Phi\ \Rightarrow\ L=\frac{\mu N^2A_c}{\ell}$"), color=MAG)

    # ---------------- (d) gapped core
    ax = axs[3]
    x0, y0, w, h, t, g = 1.35, 1.05, 3.4, 2.35, 0.55, 0.22
    gxx, yc = _ring_core(ax, x0, y0, w, h, t, g)
    # winding on the left limb
    for k, y in enumerate(np.linspace(y0 + 0.55, y0 + h - 0.55, 6)):
        ax.add_patch(Ellipse((x0 + t / 2, y), t + 0.42, 0.16, fc="none", ec=CURR, lw=1.3, zorder=4))
    label(ax, x0 - 0.42, yc, r"$NI$", color=CURR, size=10, ha="right")
    # mean flux path
    m = t / 2
    path = [(x0 + m, y0 + m), (x0 + w - m, y0 + m), (x0 + w - m, y0 + h - m), (x0 + m, y0 + h - m), (x0 + m, y0 + m)]
    ax.plot(*zip(*path), color=MAG, lw=1.0, ls=(0, (4, 2.5)), zorder=3)
    for p, q in (((x0 + 1.4, y0 + m), (x0 + 1.9, y0 + m)), ((x0 + 1.9, y0 + h - m), (x0 + 1.4, y0 + h - m)),
                 ((x0 + w - m, yc - 0.55), (x0 + w - m, yc - 0.25))):
        arrow(ax, p, q, color=MAG, lw=1.3, ms=10, z=5)
    for dx in (-0.15, 0.0, 0.15):
        arrow(ax, (x0 + w - m + dx, yc - g / 2 - 0.02), (x0 + w - m + dx, yc + g / 2 + 0.02), color=MAG, lw=1.0, ms=6, z=6)
    label(ax, x0 + w + 0.12, yc, r"$g$", color=MAG, size=10, ha="left")
    label(ax, x0 + w / 2, y0 + h / 2, T("铁芯\n$\\mu\\gg\\mu_0$", "iron\n$\\mu\\gg\\mu_0$"), color=INK, size=8.6)
    label(ax, 3.0, 3.85, T(r"磁路：$\mathcal{R}=\mathcal{F}/\Phi$", r"Magnetic circuit: $\mathcal{R}=\mathcal{F}/\Phi$"), weight="bold", size=9.8, color=MAG)
    _chain(ax, 3.0, 0.42, T(r"$NI=H_y\ell_y+H_gg$，$\Phi$ 处处相同　$\Rightarrow\ \mathcal{R}=\frac{\ell_y}{\mu A_c}+\frac{g}{\mu_0A_c}$",
                            r"$NI=H_y\ell_y+H_gg$, same $\Phi\ \Rightarrow\ \mathcal{R}=\frac{\ell_y}{\mu A_c}+\frac{g}{\mu_0A_c}$"), color=MAG)
    return fig


# ---------------------------------------------------------------------------
@figure("poynting_resistor")
def poynting_resistor():
    fig, ax = canvas(FIG_W, 3.0, xlim=(0, 10.4), ylim=(0, 5.2))
    cx, a, yb, yt = 2.8, 0.75, 0.75, 4.25
    # resistor (cylinder seen from the side)
    ax.add_patch(Rectangle((cx - a, yb), 2 * a, yt - yb, fc=MATL_T, ec="none", zorder=1))
    ax.plot([cx - a, cx - a], [yb, yt], color=tint(MATL, 0.8), lw=1.0, zorder=2)
    ax.plot([cx + a, cx + a], [yb, yt], color=tint(MATL, 0.8), lw=1.0, zorder=2)
    ax.add_patch(Ellipse((cx, yt), 2 * a, 0.36, fc=tint(MATL, 0.3), ec=tint(MATL, 0.8), lw=1.0, zorder=2))
    th = np.linspace(np.pi, 2 * np.pi, 60)
    ax.plot(cx + a * np.cos(th), yb + 0.18 * np.sin(th), color=tint(MATL, 0.8), lw=1.0, zorder=2)
    # leads
    ax.plot([cx, cx], [yt, 5.1], color=MUTED, lw=2.0, zorder=0)
    ax.plot([cx, cx], [0.05, yb - 0.18], color=MUTED, lw=2.0, zorder=0)
    arrow(ax, (cx + 0.18, 0.15), (cx + 0.18, 0.5), color=CURR, lw=1.5, ms=10)
    label(ax, cx + 0.32, 0.32, r"$I$", color=CURR, size=10, ha="left")
    # E inside (along the axis) and just outside (tangential E is continuous)
    for y in (1.3, 2.5, 3.7):
        arrow(ax, (cx - 0.25, y - 0.4), (cx - 0.25, y + 0.4), color=ELEC, lw=1.6, ms=10, z=4)
        arrow(ax, (cx + 0.25, y - 0.4), (cx + 0.25, y + 0.4), color=CURR, lw=1.6, ms=10, z=4)
    label(ax, cx - 0.42, 4.0, r"$\mathbf{E}$", color=ELEC, size=10, ha="right")
    label(ax, cx + 0.42, 4.0, r"$\mathbf{J}$", color=CURR, size=10, ha="left")
    # H: circles around the cylinder -> out of page on the left, into the page on the right
    for y in (1.3, 2.5, 3.7):
        out_of_page(ax, cx - a - 0.28, y, color=MAG, r=0.1)
        into_page(ax, cx + a + 0.28, y, color=MAG, r=0.1)
        # S inward
        halo(arrow(ax, (cx - a - 1.35, y + 0.0), (cx - a - 0.48, y), color=AMBER, lw=2.0, ms=12, z=6), 2.5)
        halo(arrow(ax, (cx + a + 1.35, y), (cx + a + 0.48, y), color=AMBER, lw=2.0, ms=12, z=6), 2.5)
    label(ax, cx - a - 0.28, 4.25, r"$\mathbf{H}$", color=MAG, size=10)
    label(ax, cx + a + 0.28, 4.25, r"$\mathbf{H}$", color=MAG, size=10)
    label(ax, cx - a - 0.95, 0.85, r"$\mathbf{S}$", color=AMBER, size=11)
    label(ax, cx + a + 0.95, 0.85, r"$\mathbf{S}$", color=AMBER, size=11)
    # ---- right: vector triad at a point of the right-hand surface
    ox, oy = 7.0, 2.35
    ax.add_patch(Rectangle((ox + 0.0, oy - 1.3), 0.06, 2.6, fc=tint(MATL, 0.6), ec="none", zorder=0))
    label(ax, ox + 0.2, oy - 1.45, T("导体表面（右侧）", "conductor surface (right)"), size=8.4, color=MUTED, ha="left")
    halo(arrow(ax, (ox, oy), (ox, oy + 1.15), color=ELEC, lw=2.0, ms=12, z=6), 2.5)
    label(ax, ox + 0.18, oy + 1.05, T(r"$\mathbf{E}$（$E_z$）", r"$\mathbf{E}$ ($E_z$)"), color=ELEC, size=10, ha="left")
    into_page(ax, ox + 0.75, oy, color=MAG, r=0.16, lw=1.3)
    label(ax, ox + 1.0, oy + 0.0, T(r"$\mathbf{H}$（$H_\theta$，指向纸内）", r"$\mathbf{H}$ ($H_\theta$, into the page)"), color=MAG, size=9.5, ha="left")
    halo(arrow(ax, (ox, oy), (ox - 1.15, oy), color=AMBER, lw=2.2, ms=13, z=6), 2.5)
    label(ax, ox - 0.6, oy - 0.3, r"$\mathbf{S}=\mathbf{E}\times\mathbf{H}$", color=AMBER, size=10)
    label(ax, 7.5, 4.85, T(r"单位长度流入：$2\pi a\,E_zH_\theta=E_z I=I^2R/\ell$",
                           r"inflow per unit length: $2\pi a\,E_zH_\theta=E_z I=I^2R/\ell$"), size=9.3)
    label(ax, 7.5, 4.35, T(r"$H_\theta=\dfrac{I}{2\pi a}$（安培定律）", r"$H_\theta=\dfrac{I}{2\pi a}$ (Ampère)"), size=9.0, color=MAG)
    label(ax, 7.6, 0.35, T("流入的功率恰好等于导体内的欧姆热", "the inflow equals the ohmic heat inside"), size=8.8, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("core_gap")
def core_gap():
    fig = plt.figure(figsize=(FIG_W, 2.7))
    ax = fig.add_axes([0.0, 0.0, 0.47, 1.0])
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 4.25)
    ax.set_aspect("equal")
    ax.axis("off")
    x0, y0, w, h, t, g = 1.1, 0.55, 3.2, 3.0, 0.6, 0.26
    gxx, yc = _ring_core(ax, x0, y0, w, h, t, g)
    for y in np.linspace(y0 + 0.6, y0 + h - 0.6, 7):
        ax.add_patch(Ellipse((x0 + t / 2, y), t + 0.45, 0.17, fc="none", ec=CURR, lw=1.3, zorder=4))
    label(ax, x0 - 0.4, yc + 0.15, r"$N$", color=CURR, size=10, ha="right")
    label(ax, x0 - 0.4, yc - 0.25, r"$I$", color=CURR, size=10, ha="right")
    m = t / 2
    path = [(x0 + m, y0 + m), (x0 + w - m, y0 + m), (x0 + w - m, y0 + h - m), (x0 + m, y0 + h - m), (x0 + m, y0 + m)]
    ax.plot(*zip(*path), color=MAG, lw=1.0, ls=(0, (4, 2.5)), zorder=3)
    for p, q in (((x0 + 1.3, y0 + m), (x0 + 1.9, y0 + m)), ((x0 + 1.9, y0 + h - m), (x0 + 1.3, y0 + h - m)),
                 ((x0 + w - m, y0 + 0.75), (x0 + w - m, y0 + 1.1)), ((x0 + m, y0 + h - 0.75), (x0 + m, y0 + h - 1.1))):
        arrow(ax, p, q, color=MAG, lw=1.3, ms=10, z=5)
    for dx in (-0.18, 0.0, 0.18):
        arrow(ax, (x0 + w - m + dx, yc - g / 2 - 0.02), (x0 + w - m + dx, yc + g / 2 + 0.02), color=MAG, lw=1.0, ms=6, z=6)
    # fringing hint (ignored by the model)
    for s in (-1, 1):
        xe = x0 + w - m + s * (t / 2)
        th = np.linspace(-np.pi / 2, np.pi / 2, 40)
        ax.plot(xe + s * 0.17 * np.cos(th), yc + 0.32 * np.sin(th), color=tint(MAG, 0.55), lw=0.8, ls=(0, (2, 1.5)), zorder=3)
    label(ax, x0 + w + 0.45, yc + 0.0, r"$g$", color=MAG, size=10.5, ha="left")
    label(ax, x0 + w / 2 + 0.05, y0 + h / 2 + 0.18, T(r"平均路径 $\ell_y$", r"mean path $\ell_y$"), color=MAG, size=8.8)
    label(ax, x0 + w / 2 + 0.05, y0 + h / 2 - 0.22, T("（虚线）", "(dashed)"), color=MAG, size=8.4)
    label(ax, x0 + w / 2, y0 + h + 0.28, T(r"铁芯：截面 $A_c$，$\mu_r=1000$", r"iron: cross-section $A_c$, $\mu_r=1000$"), color=INK, size=8.6)
    label(ax, x0 + w / 2, 0.17, T("气隙两侧的点线：边缘效应（本模型忽略）", "dotted at the gap: fringing (ignored here)"), size=8.0, color=MUTED)
    # ---- right: where length, MMF and energy go
    bx = fig.add_axes([0.6, 0.2, 0.37, 0.66])
    cats = [T("路径长度", "path length"), T("磁动势 $NI$", "MMF $NI$"), T("磁能", "magnetic energy")]
    core = np.array([0.2 / 0.201, 1 / 6, 1 / 6])
    gap = 1 - core
    y = np.arange(3)[::-1]
    bx.barh(y, core, color=IRON_T, edgecolor=tint("#7A8691", 0.9), lw=0.8, height=0.55)
    bx.barh(y, gap, left=core, color=tint(MAG, 0.55), edgecolor=MAG, lw=0.8, height=0.55)
    bx.set_yticks(y, cats)
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    texts = [("99.5%", "0.5%"), ("17%", "83%"), ("17%", "83%")]
    for yy, c, (tc, tg) in zip(y, core, texts):
        bx.text(c / 2, yy, tc, ha="center", va="center", fontsize=8.3, color=INK)
        if c < 0.9:
            bx.text(c + (1 - c) / 2, yy, tg, ha="center", va="center", fontsize=8.3, color=INK)
        else:
            bx.annotate(tg, xy=(0.997, yy - 0.28), xytext=(0.9, yy - 0.62), fontsize=8.3, color=MAG,
                        ha="center", va="center", arrowprops=dict(arrowstyle="-", color=MAG, lw=0.7))
    from matplotlib.patches import Patch
    bx.legend([Patch(fc=IRON_T, ec=tint("#7A8691", 0.9), lw=0.8), Patch(fc=tint(MAG, 0.55), ec=MAG, lw=0.8)],
              [T("铁芯", "iron"), T("气隙", "gap")], ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=8.5, handlelength=1.3, borderaxespad=0.2)
    bx.set_title(T("气隙很短，却分走了大部分磁动势和磁能", "the short gap takes most of the MMF and energy"), fontsize=9.0, pad=22)
    clean_axes(bx)
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    return fig
