"""Figures for Chapter 4: currents, magnetic fields and induction."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Rectangle, Wedge
from scipy.special import ellipe, ellipk

from style import (ACCENT, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, LIGHT, MAG, MAG_T, MUTED, POT, POT_T, T,
                   arrow, canvas, clean_axes, dot, figure, flux_lines, halo, into_page, label, out_of_page,
                   panel_title, plt, rbox, tint, unit_quiver)

# ---------------------------------------------------------------------------
# Exact fields of a circular loop (axis z, radius a, in the plane z = 0)
# ---------------------------------------------------------------------------


def loop_psi(rho, z, a=1.0):
    """Stokes flux function rho*A_phi of a unit loop (mu0 I / pi factored out).

    A_phi = (mu0 I / (pi k)) sqrt(a/rho) [(1 - k^2/2) K(k^2) - E(k^2)]  (Jackson 5.37);
    the flux through the coaxial disc of radius rho at height z is 2 pi rho A_phi.
    """
    rho = np.abs(rho) + 1e-12
    k2 = np.clip(4 * a * rho / ((a + rho) ** 2 + z**2), 0, 1 - 1e-12)
    K, E = ellipk(k2), ellipe(k2)
    return rho * np.sqrt(a / rho) / np.sqrt(k2) * ((1 - k2 / 2) * K - E)


def loop_B(x, z, a=1.0):
    """(B_x, B_z) of a unit loop in the x-z plane (shape only, same factor as loop_psi up to a constant)."""
    rho = np.abs(x) + 1e-12
    r2 = rho**2 + z**2
    al2 = a**2 + r2 - 2 * a * rho
    be2 = a**2 + r2 + 2 * a * rho
    be = np.sqrt(be2)
    m = 1 - al2 / be2
    K, E = ellipk(m), ellipe(m)
    with np.errstate(divide="ignore", invalid="ignore"):   # exactly on the wire: singular, masked by callers
        Brho = z / (2 * al2 * be * rho) * ((a**2 + r2) * E - al2 * K)
        Bz = 1 / (2 * al2 * be) * ((a**2 - r2) * E + al2 * K)
    Brho = np.nan_to_num(Brho, nan=0.0, posinf=0.0, neginf=0.0)
    Bz = np.nan_to_num(Bz, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(x) * Brho, Bz


# ---------------------------------------------------------------------------
@figure("magnetic_flux")
def magnetic_flux():
    """Same uniform B, a flat surface (edge-on) at three orientations."""
    fig, ax = canvas(FIG_W, 2.45, xlim=(0, 12.3), ylim=(0, 5.0))
    cy = 2.15
    offsets = np.array([-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75])
    half = 1.0
    cases = [(0, T(r"$\theta=0$：$\Phi_B=BA_c$", r"$\theta=0$: $\Phi_B=BA_c$")),
             (60, T(r"$\theta=60^\circ$：$\Phi_B=BA_c/2$", r"$\theta=60^\circ$: $\Phi_B=BA_c/2$")),
             (90, T(r"$\theta=90^\circ$：$\Phi_B=0$", r"$\theta=90^\circ$: $\Phi_B=0$"))]
    for k, (deg, title) in enumerate(cases):
        x0 = 0.15 + 4.1 * k
        cx = x0 + 1.75
        th = np.deg2rad(deg)
        nvec = np.array([np.cos(th), np.sin(th)])
        tvec = np.array([-np.sin(th), np.cos(th)])  # along the plate
        band = half * np.cos(th)
        for off in offsets:
            y = cy + off
            hit = abs(off) < band - 1e-9
            col = MAG if hit else tint(MAG, 0.35)
            ax.plot([x0, x0 + 3.85], [y, y], color=col, lw=1.15 if hit else 0.9, zorder=2)
            arrow(ax, (x0 + 3.55, y), (x0 + 3.8, y), color=col, lw=1.0, ms=7, z=2.5)
            if hit:
                # crossing point of y = cy + off with the plate line through (cx, cy)
                s = off / tvec[1]
                p = np.array([cx, cy]) + s * tvec
                dot(ax, p[0], p[1], color=MAG, r=0.075, z=7)
        p1 = np.array([cx, cy]) - half * tvec
        p2 = np.array([cx, cy]) + half * tvec
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#4A525A", lw=4.2, solid_capstyle="round", zorder=5)
        # normal
        tip = np.array([cx, cy]) + 1.05 * nvec
        halo(arrow(ax, (cx, cy), tuple(tip), color=ACCENT, lw=1.7, ms=11, z=8), 3)
        lp = tip + 0.22 * nvec + (np.array([0.0, 0.0]) if deg != 90 else np.array([0.2, 0.0]))
        halo(label(ax, lp[0] + (0.12 if deg == 0 else 0.0), lp[1] + (0.15 if deg == 0 else 0.0),
                   r"$\mathbf{n}$", color=ACCENT, size=11, z=9), 3)
        if 0 < deg:
            arc = Arc((cx, cy), 1.25, 1.25, theta1=0, theta2=deg, color=ACCENT, lw=0.9, zorder=8)
            ax.add_patch(arc)
            mid = np.deg2rad(deg / 2)
            halo(label(ax, cx + 0.88 * np.cos(mid), cy + 0.82 * np.sin(mid) + (0.06 if deg == 90 else 0), r"$\theta$",
                       color=ACCENT, size=10, z=9), 3)
        # projected height ("shadow") bracket
        if band > 1e-6:
            xb = x0 + 3.97
            ax.plot([xb, xb], [cy - band, cy + band], color=ACCENT, lw=1.3, zorder=6)
            for yy in (cy - band, cy + band):
                ax.plot([xb - 0.07, xb + 0.07], [yy, yy], color=ACCENT, lw=1.3, zorder=6)
        label(ax, x0 + 1.9, 4.72, title, size=9.3, weight="bold")
    label(ax, 0.25, cy + 2.05, r"$\mathbf{B}$", color=MAG, size=11, ha="left")
    label(ax, 6.15, 0.08, T(r"右侧短竖线：曲面在垂直于 B 的平面上的投影，长度 $\propto A_c\cos\theta$；红点：穿过曲面的场线",
                            r"bar: projection of the surface onto a plane normal to B, length $\propto A_c\cos\theta$; dots: lines crossing it"),
          size=8.2, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
def biot_savart_polygon_axis(z, nseg, a=1.0):
    """B_z on the axis of a regular nseg-gon inscribed in the circle of radius a (unit current, mu0/4pi = 1)."""
    phi = np.linspace(0, 2 * np.pi, nseg + 1)
    P = np.stack((a * np.cos(phi), a * np.sin(phi), np.zeros_like(phi)), axis=1)
    out = []
    for zz in np.atleast_1d(z):
        obs = np.array([0.0, 0.0, zz])
        Bz = 0.0
        for k in range(nseg):
            p1, p2 = P[k], P[k + 1]
            dl = p2 - p1
            r1, r2 = obs - p1, obs - p2
            c = np.cross(dl, r1)
            Bz += c[2] / np.dot(c, c) * (np.dot(dl, r1) / np.linalg.norm(r1) - np.dot(dl, r2) / np.linalg.norm(r2))
        out.append(Bz)
    return np.array(out)


@figure("loop_axis")
def loop_axis():
    fig = plt.figure(figsize=(FIG_W, 2.75))
    # ---- left: geometry of the symmetry argument
    ax = fig.add_axes([0.0, 0.0, 0.40, 1.0])
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-0.75, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    a, h, e = 1.0, 1.45, 0.32
    ax.plot([0, 0], [-0.6, 2.45], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    label(ax, 0.08, 2.47, r"$z$", color=MUTED, size=9.5, ha="left")
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(a * np.cos(th), e * np.sin(th), color=CURR, lw=2.0, zorder=3)
    # current direction: counter-clockwise seen from +z  -> front (lower) part moves left->right
    arrow(ax, (-0.08, -e), (0.1, -e), color=CURR, lw=2.0, ms=11, z=4)
    label(ax, 0.0, -e - 0.27, r"$I$", color=CURR, size=10.5)
    into_page(ax, a, 0, color=CURR, r=0.085)
    out_of_page(ax, -a, 0, color=CURR, r=0.085)
    P = np.array([0.0, h])
    dot(ax, *P, color=INK, r=0.045, z=9)
    halo(label(ax, -0.13, h + 0.2, r"$P$", size=10, ha="right", z=10), 3)
    for sgn in (1, -1):
        src = np.array([sgn * a, 0.0])
        ax.plot([src[0], P[0]], [src[1], P[1]], color=MUTED, lw=0.9, zorder=2)
        # dB perpendicular to R, in the plane: (h, a) for the right element, (-h, a) for the left
        v = np.array([sgn * h, a])
        v = v / np.linalg.norm(v) * 0.85
        tip = P + v
        halo(arrow(ax, tuple(P), tuple(tip), color=MAG, lw=1.6, ms=10, z=8), 3)
        ax.plot([tip[0], tip[0]], [P[1], tip[1]], color=MAG, lw=0.8, ls=(0, (2, 2)), zorder=6)
        ax.plot([P[0], tip[0]], [P[1], P[1]], color=MAG, lw=0.8, ls=(0, (2, 2)), zorder=6)
    label(ax, -0.62, h + 0.75, r"$\mathrm{d}\mathbf{B}_2$", color=MAG, size=9.5)
    label(ax, 0.62, h + 0.75, r"$\mathrm{d}\mathbf{B}_1$", color=MAG, size=9.5)
    label(ax, 0.55, 0.42, r"$R$", color=MUTED, size=9.5)
    label(ax, 1.25, -0.05, "1", color=CURR, size=8.5)
    label(ax, -1.27, -0.05, "2", color=CURR, size=8.5)
    label(ax, 0.0, h - 0.36, T("横向分量相消\n轴向分量相加", "transverse parts cancel,\naxial parts add"), size=8.2, color=MAG,
          bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    # ---- right: on-axis profile, formula vs numerical Biot-Savart
    bx = fig.add_axes([0.50, 0.17, 0.48, 0.75])
    z = np.linspace(-4, 4, 801)
    exact = 1 / (1 + z**2) ** 1.5
    bx.plot(z, exact, color=MAG, lw=1.9, zorder=3, label=T("公式 (4.x)", "formula"))
    zz = np.linspace(-4, 4, 17)
    b0 = 2 * np.pi  # exact centre value with mu0/4pi = 1, I = 1, a = 1
    for nseg, mk, fc in ((6, "s", "white"), (24, "o", MAG)):
        vals = biot_savart_polygon_axis(zz, nseg) / b0
        bx.plot(zz, vals, ls="none", marker=mk, ms=4.3, mfc=fc, mec=MAG if nseg == 24 else INK,
                mew=0.9, zorder=4)
    zf = np.linspace(1.3, 4, 100)
    for s in (1, -1):
        bx.plot(s * zf, zf**-3, color=MUTED, lw=1.1, ls=(0, (4, 2.5)), zorder=2)
    zh = np.sqrt(2 ** (2 / 3) - 1)
    bx.plot([-zh, zh], [0.5, 0.5], color=MUTED, lw=0.8, ls=":", zorder=1)
    bx.annotate(T(r"半高：$|z|\approx0.77a$", r"half height: $|z|\approx0.77a$"), xy=(-zh, 0.5), xytext=(-3.95, 0.6),
                fontsize=8.4, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    bx.annotate(T(r"远处 $\propto|z|^{-3}$（偶极场）", r"far field $\propto|z|^{-3}$ (dipole)"), xy=(2.0, 2.0**-3), xytext=(1.55, 0.55),
                fontsize=8.4, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MAG, lw=1.9),
               Line2D([], [], ls="none", marker="s", ms=4.3, mfc="white", mec=INK, mew=0.9),
               Line2D([], [], ls="none", marker="o", ms=4.3, mfc=MAG, mec=MAG)]
    bx.legend(handles, [T("解析公式", "closed form"), T("6 段折线", "6-segment polygon"),
                        T("24 段折线", "24-segment polygon")],
              loc="upper right", fontsize=8.0, handlelength=1.4, borderaxespad=0.0, bbox_to_anchor=(1.03, 1.04))
    bx.set_xlim(-4, 4)
    bx.set_ylim(0, 1.18)
    bx.set_xlabel(T(r"轴向位置 $z/a$", r"axial position $z/a$"))
    bx.set_ylabel(r"$B_z(z)\,/\,B_z(0)$")
    clean_axes(bx)
    return fig


# ---------------------------------------------------------------------------
@figure("induction")
def induction():
    fig = plt.figure(figsize=(FIG_W, 2.85))
    ax = fig.add_axes([0.0, 0.0, 0.47, 1.0])
    R = 1.0
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.36, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.12, fc=tint(MAG, 0.1), ec="none", zorder=0))
    for x in np.linspace(-0.9, 0.9, 5):
        for y in np.linspace(-0.9, 0.9, 5):
            if x * x + y * y < 1.05:
                out_of_page(ax, x, y, color=tint(MAG, 0.75), r=0.06, lw=0.8, z=1)
    # E: clockwise circles, arrow length grows with r (E_theta = -r/2 dB/dt)
    th = np.linspace(0, 2 * np.pi, 300)
    for rr in (0.35, 0.65, 0.95):
        ax.plot(rr * np.cos(th), rr * np.sin(th), color=tint(ELEC, 0.45), lw=0.9, zorder=2)
        for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False) + 0.3:
            p = rr * np.array([np.cos(ang), np.sin(ang)])
            v = -0.32 * rr * np.array([-np.sin(ang), np.cos(ang)])  # clockwise
            halo(arrow(ax, tuple(p - v / 2), tuple(p + v / 2), color=ELEC, lw=1.5, ms=9, z=5), 2.2)
    # oriented loop C (counter-clockwise, normal out of the page)
    rc = 1.12
    ax.plot(rc * np.cos(th), rc * np.sin(th), color=ACCENT, lw=1.2, ls=(0, (5, 3)), zorder=3)
    a0 = np.deg2rad(140)
    arrow(ax, (rc * np.cos(a0), rc * np.sin(a0)), (rc * np.cos(a0 + 0.12), rc * np.sin(a0 + 0.12)), color=ACCENT, lw=1.4, ms=11, z=6)
    label(ax, -1.02, 1.08, r"$C$", color=ACCENT, size=11)
    label(ax, 0.0, 1.29, T(r"$\mathbf{B}$ 指向纸外且增大：$\partial_tB_z>0$", r"$\mathbf{B}$ out of page, increasing: $\partial_tB_z>0$"), color=MAG, size=9.0)
    label(ax, 0.0, -1.27, T(r"$\oint_C\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}=-\dot\Phi_B<0$：E 顺时针", r"$\oint_C\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}=-\dot\Phi_B<0$: E clockwise"), color=ELEC, size=9.0)
    # ---- right: |E_theta| vs r for dB/dt = 1000 T/s
    bx = fig.add_axes([0.6, 0.2, 0.37, 0.68])
    r = np.linspace(0, 0.1, 50)
    bx.plot(r * 100, 500 * r, color=ELEC, lw=1.9)
    bx.plot([8, 8], [0, 40], color=MUTED, lw=0.8, ls=":")
    bx.plot([0, 8], [40, 40], color=MUTED, lw=0.8, ls=":")
    bx.plot([8], [40], marker="o", ms=5.5, color=ELEC)
    bx.annotate(T(r"$r=8\,$cm：$40\,$V/m", r"$r=8\,$cm: $40\,$V/m"), xy=(8, 40), xytext=(1.0, 44.5), fontsize=8.6, color=INK)
    bx.set_xlim(0, 10)
    bx.set_ylim(0, 52)
    bx.set_xlabel(T(r"半径 $r$ (cm)", r"radius $r$ (cm)"))
    bx.set_ylabel(r"$|E_\theta|$ (V/m)")
    bx.set_title(T(r"$\partial_tB_z=1000\,$T/s 时：$|E_\theta|=\frac{r}{2}\,\partial_tB_z$", r"for $\partial_tB_z=1000\,$T/s: $|E_\theta|=\frac{r}{2}\,\partial_tB_z$"), fontsize=9.0, pad=6)
    clean_axes(bx, grid=True)
    return fig


# ---------------------------------------------------------------------------
@figure("solenoid_profiles")
def solenoid_profiles():
    fig = plt.figure(figsize=(FIG_W, 3.0))
    ax = fig.add_axes([0.0, 0.0, 0.47, 1.0])
    ax.set_xlim(-3.25, 3.25)
    ax.set_ylim(-3.05, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")
    a = 1.0
    th = np.linspace(0, 2 * np.pi, 400)
    ax.add_patch(Circle((0, 0), a, fc=tint(MAG, 0.12), ec="none", zorder=0))
    # winding (current counter-clockwise -> B out of the page inside)
    ax.plot(a * np.cos(th), a * np.sin(th), color=CURR, lw=3.2, zorder=3)
    for ang in (np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6):
        p = a * np.array([np.cos(ang), np.sin(ang)])
        v = 0.05 * np.array([-np.sin(ang), np.cos(ang)])
        arrow(ax, tuple(p - v), tuple(p + v), color=CURR, lw=2.0, ms=11, z=4)
    for x, y in ((0, 0), (0.45, 0.3), (-0.45, 0.3), (0.45, -0.3), (-0.45, -0.3), (0, 0.62), (0, -0.62)):
        out_of_page(ax, x, y, color=MAG, r=0.075, lw=0.9, z=2)
    # induced E (clockwise for increasing current), arrow length ~ |E_theta|
    def Eabs(rr):
        return rr / 2 if rr < a else a**2 / (2 * rr)
    for rr, n in ((0.5, 4), (1.42, 8), (2.25, 10)):
        ax.plot(rr * np.cos(th), rr * np.sin(th), color=tint(ELEC, 0.35), lw=0.8, zorder=1)
        L = 1.35 * Eabs(rr)
        for ang in np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.25:
            p = rr * np.array([np.cos(ang), np.sin(ang)])
            v = -L * np.array([-np.sin(ang), np.cos(ang)])
            halo(arrow(ax, tuple(p - v / 2), tuple(p + v / 2), color=ELEC, lw=1.4, ms=8, z=5), 2)
    # outer loop C
    rc = 2.8
    ax.plot(rc * np.cos(th), rc * np.sin(th), color=ACCENT, lw=1.2, ls=(0, (5, 3)), zorder=2)
    a0 = np.deg2rad(128)
    arrow(ax, (rc * np.cos(a0), rc * np.sin(a0)), (rc * np.cos(a0 + 0.07), rc * np.sin(a0 + 0.07)), color=ACCENT, lw=1.3, ms=10, z=6)
    label(ax, -2.35, 2.3, r"$C$", color=ACCENT, size=11)
    label(ax, 0.0, 3.2, T(r"电流增大时：绕组外 $\mathbf{B}=0$，但 $\mathbf{E}\neq0$", r"current rising: outside, $\mathbf{B}=0$ but $\mathbf{E}\neq0$"), size=9.2, weight="bold")
    label(ax, 0.0, -2.98, T(r"$\oint_C\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}=-\pi a^2\dot B_c$，与 $C$ 的大小无关", r"$\oint_C\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}=-\pi a^2\dot B_c$ for any such $C$"), size=8.8, color=ACCENT)
    # ---- right: radial profiles
    r = np.linspace(0, 4, 801)
    B = np.where(r < a, 1.0, 0.0)
    A = np.where(r < a, r / 2, a**2 / (2 * np.maximum(r, 1e-9)))
    bx1 = fig.add_axes([0.58, 0.6, 0.4, 0.33])
    bx2 = fig.add_axes([0.58, 0.14, 0.4, 0.33], sharex=bx1)
    for bx in (bx1, bx2):
        bx.axvspan(0, 1, color=tint(MAG, 0.1), lw=0, zorder=0)
        bx.axvline(1, color=CURR, lw=1.2, zorder=1)
        clean_axes(bx)
    bx1.plot(r, B, color=MAG, lw=1.9, zorder=3)
    bx1.set_ylim(-0.08, 1.25)
    bx1.set_yticks([0, 1])
    bx1.set_ylabel(r"$B_z/B_c$")
    bx1.text(2.5, 0.25, T("外部 $B=0$", "outside $B=0$"), fontsize=8.6, color=MAG, ha="center")
    bx1.text(0.5, 1.08, T("绕组内", "inside"), fontsize=8.4, color=MUTED, ha="center")
    bx1.text(1.06, 1.08, T("绕组", "winding"), fontsize=8.4, color=CURR, ha="left")
    plt.setp(bx1.get_xticklabels(), visible=False)
    bx2.plot(r, A, color=POT, lw=2.6, zorder=3, label=r"$A_\theta/(B_ca)$")
    bx2.plot(r, A, color=ELEC, lw=1.2, ls=(0, (4, 2.5)), zorder=4, label=r"$|E_\theta|/(\dot B_ca)$")
    bx2.set_ylim(0, 0.62)
    bx2.set_yticks([0, 0.25, 0.5])
    bx2.legend(loc="upper right", fontsize=8.4, handlelength=2.0, borderaxespad=0.1)
    bx2.text(2.6, 0.27, T(r"外部 $\propto 1/r$", r"outside $\propto 1/r$"), fontsize=8.6, color=INK, ha="center")
    bx2.text(0.42, 0.33, r"$\propto r$", fontsize=8.6, color=INK, ha="center")
    bx2.set_xlim(0, 4)
    bx2.set_xlabel(T(r"半径 $r/a$", r"radius $r/a$"))
    return fig


# ---------------------------------------------------------------------------
@figure("static_vs_induced")
def static_vs_induced():
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.2))
    L = 2.2
    g = np.linspace(-L, L, 801)
    X, Y = np.meshgrid(g, g)
    th = np.linspace(0, 2 * np.pi, 300)
    # ---- left: two opposite line charges (2-D electrostatics)
    ax = axs[0]
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    d = 0.9
    r1 = np.hypot(X + d, Y)
    r2 = np.hypot(X - d, Y)
    Ex = (X + d) / r1**2 - (X - d) / r2**2
    Ey = Y / r1**2 - Y / r2**2
    stream_fn = np.arctan2(Y, X + d) - np.arctan2(Y, X - d)  # field lines
    pot = np.log(r2 / r1)  # equipotentials
    mask = (r1 < 0.12) | (r2 < 0.12)
    levels = np.linspace(-np.pi, np.pi, 17)[1:-1]
    flux_lines(ax, X, Y, np.ma.array(stream_fn, mask=mask), levels, Ex, Ey, color=ELEC, lw=1.0,
               arrows=True, arrow_pos=0.5, ms=7)
    ax.contour(X, Y, np.ma.array(pot, mask=mask), levels=np.linspace(-2.4, 2.4, 9), colors=[POT], linewidths=0.8,
               linestyles=[(0, (3, 2))], zorder=2)
    from style import charge
    charge(ax, -d, 0, "+", r=0.13)
    charge(ax, d, 0, "-", r=0.13)
    cc = Circle((0.0, 1.25), 0.52, fc="none", ec=ACCENT, lw=1.6, zorder=6)
    ax.add_patch(cc)
    arrow(ax, (0.52 * np.cos(0.3), 1.25 + 0.52 * np.sin(0.3)), (0.52 * np.cos(0.4), 1.25 + 0.52 * np.sin(0.4)), color=ACCENT, lw=1.6, ms=10, z=7)
    halo(label(ax, 0.0, 1.25, r"$\oint\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}=0$", color=ACCENT, size=9.0, z=9), 3)
    panel_title(ax, T("静电场：起止于电荷", "Electrostatic: starts and ends on charges"), color=ELEC)
    label(ax, 0.0, -2.05, T("紫色虚线：等势线（处处存在）", "purple dashes: equipotentials (exist everywhere)"), color=POT, size=8.4,
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    # ---- right: changing flux tube of radius a
    ax = axs[1]
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    a = 0.6
    R = np.hypot(X, Y) + 1e-12
    Eth = np.where(R < a, R / a, a / R)  # |E_theta| shape
    Ex2 = Eth * Y / R   # clockwise for increasing flux out of page
    Ey2 = -Eth * X / R
    psi = np.where(R < a, R**2 / (2 * a), a / 2 + a * np.log(R / a))  # E = curl(psi e_z) up to sign
    ax.add_patch(Circle((0, 0), a, fc=tint(MAG, 0.18), ec=tint(MAG, 0.7), lw=0.8, zorder=0))
    for x, y in ((0, 0), (0.3, 0.2), (-0.3, 0.2), (0.0, -0.32)):
        out_of_page(ax, x, y, color=MAG, r=0.065, lw=0.8, z=1)
    flux_lines(ax, X, Y, psi, np.linspace(0.12, 1.15, 7), Ex2, Ey2, color=ELEC, lw=1.0, arrows=True,
               arrow_pos=0.12, ms=7)
    rc = 1.22
    c1 = Circle((0, 0), rc, fc="none", ec=ACCENT, lw=1.6, zorder=6)
    ax.add_patch(c1)
    arrow(ax, (rc * np.cos(2.0), rc * np.sin(2.0)), (rc * np.cos(2.08), rc * np.sin(2.08)), color=ACCENT, lw=1.6, ms=10, z=7)
    halo(label(ax, 0.0, 1.63, r"$\oint\mathbf{E}\cdot\mathrm{d}\boldsymbol{\ell}=-\dot\Phi_B\neq0$", color=ACCENT, size=9.0, z=9), 3)
    cx, cy, rr = 1.62, -1.5, 0.36
    ax.add_patch(Circle((cx, cy), rr, fc="none", ec=ACCENT, lw=1.6, zorder=6))
    arrow(ax, (cx + rr * np.cos(0.3), cy + rr * np.sin(0.3)), (cx + rr * np.cos(0.45), cy + rr * np.sin(0.45)), color=ACCENT, lw=1.6, ms=10, z=7)
    halo(label(ax, cx - 0.44, cy + 0.02, r"$\oint=0$", color=ACCENT, size=9.0, ha="right", z=9), 3)
    halo(label(ax, 0.0, -0.2 - a - 0.12, r"$\partial_t\mathbf{B}$", color=MAG, size=9.5, z=9), 2.5)
    panel_title(ax, T("感应电场：绕着变化的磁通闭合", "Induced: closes around the changing flux"), color=ELEC)
    label(ax, 0.0, -2.05, T("找不到处处单值的等势线", "no single-valued equipotentials exist"), color=POT, size=8.4,
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.06)
    return fig


# ---------------------------------------------------------------------------
@figure("flux_linkage")
def flux_linkage():
    fig = plt.figure(figsize=(FIG_W, 3.0))
    ax = fig.add_axes([0.0, 0.0, 0.56, 0.9])
    Rs = 1.0           # source loop radius (at z = 0)
    ap = 0.5           # pick-up turn radius
    zk = np.array([0.55, 1.05, 1.55])
    xl, zl = (-2.25, 2.25), (-0.95, 2.05)
    gx = np.linspace(xl[0] - 0.6, xl[1] + 0.6, 901)
    gz = np.linspace(zl[0] - 0.6, zl[1] + 0.3, 801)
    X, Z = np.meshgrid(gx, gz)
    psi = loop_psi(X, Z, Rs)
    Bx, Bz = loop_B(X, Z, Rs)
    ax.set_xlim(*xl)
    ax.set_ylim(zl[0], zl[1])
    ax.set_aspect("equal")
    ax.axis("off")
    psik = loop_psi(np.full(3, ap), zk, Rs)
    step = psik[0] / 6.0
    levels = step * (np.arange(1, 11) - 0.5)
    mask = np.hypot(np.abs(X) - Rs, Z) < 0.12
    flux_lines(ax, X, Z, np.ma.array(psi, mask=mask), levels, Bx, Bz, color=MAG, lw=0.95, arrows=True,
               arrow_pos=0.5, ms=6.5, mask=None)
    out_of_page(ax, -Rs, 0, color=CURR, r=0.09)
    into_page(ax, Rs, 0, color=CURR, r=0.09)
    label(ax, 0.0, -0.32, T("源线圈", "source coil"), color=CURR, size=8.6,
          bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    for k, z in enumerate(zk):
        ax.plot([-ap, ap], [z, z], color=ACCENT, lw=1.0, ls=(0, (3, 2)), zorder=6)
        for s in (-1, 1):
            ax.add_patch(Circle((s * ap, z), 0.05, fc="white", ec=ACCENT, lw=1.3, zorder=7))
        halo(label(ax, ap + 0.14, z + 0.0, rf"$S_{k+1}$", color=ACCENT, size=9.5, ha="left", z=9), 3)
    fig.text(0.28, 0.955, T("源线圈的磁场穿过三匝拾取线圈（虚线：每匝张成的面）", "Field of a source coil threading a three-turn pick-up winding"),
             size=8.8, color=INK, ha="center", va="center")
    # ---- right: bar chart of the flux of each turn
    bx = fig.add_axes([0.66, 0.2, 0.32, 0.62])
    vals = psik / psik[0]
    names = [r"$\Phi_1$", r"$\Phi_2$", r"$\Phi_3$"]
    bx.bar(range(3), vals, color=tint(MAG, 0.5), edgecolor=MAG, lw=0.9, width=0.62)
    for i, v in enumerate(vals):
        bx.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4)
    bx.set_xticks(range(3), names)
    bx.set_ylim(0, 1.25)
    bx.set_ylabel(T(r"每匝磁通（以 $\Phi_1$ 为单位）", r"flux per turn (units of $\Phi_1$)"), fontsize=8.8)
    lam = vals.sum()
    bx.set_title(T(rf"$\lambda=\Phi_1+\Phi_2+\Phi_3={lam:.2f}\,\Phi_1$" + "\n" + r"$\neq 3\Phi_1$",
                   rf"$\lambda=\Phi_1+\Phi_2+\Phi_3={lam:.2f}\,\Phi_1$" + "\n" + r"$\neq 3\Phi_1$"), fontsize=9.0, pad=6)
    clean_axes(bx)
    return fig
