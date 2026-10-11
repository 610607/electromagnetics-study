"""Figures for the chapter on matter: dipoles, bound sources, D and H, analogies."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle, Arc
from scipy.special import ellipe, ellipk

from style import (flux_lines, halo, ACCENT, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, IRON_T, MAG, MAG_T, MATL,
                   MATL_T, MUTED, NEUTRAL_T, POT, T, arrow, canvas, card, charge, clean_axes,
                   curved_arrow, figure, into_page, label, out_of_page, panel_title, pill, plt, rbox,
                   save, stream, tint)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------
def point_dipole_charges(X, Z, d=0.5):
    """E of +q at z=+d/2 and -q at z=-d/2 evaluated in the x-z plane (3-D Coulomb)."""
    Ex = np.zeros_like(X)
    Ez = np.zeros_like(X)
    for q, z0 in ((1.0, d / 2), (-1.0, -d / 2)):
        rx, rz = X, Z - z0
        r3 = (rx**2 + rz**2) ** 1.5 + 1e-12
        Ex += q * rx / r3
        Ez += q * rz / r3
    return Ex, Ez


def loop_field(X, Z, a=0.5):
    """Exact B of a circular loop (radius a, axis z, in z=0) in the x-z plane.

    Closed form with complete elliptic integrals; see e.g. J. Simpson et al.,
    "Simple analytic expressions for the magnetic field of a circular current loop",
    NASA/TM-2001-210936 (constant prefactor dropped: shape only).
    """
    rho = np.abs(X) + 1e-9
    z = Z
    r2 = rho**2 + z**2
    alpha2 = a**2 + r2 - 2 * a * rho
    beta2 = a**2 + r2 + 2 * a * rho
    beta = np.sqrt(beta2)
    m = 1 - alpha2 / beta2
    K, E = ellipk(m), ellipe(m)
    Brho = z / (2 * alpha2 * beta * rho) * ((a**2 + r2) * E - alpha2 * K)
    Bz = 1 / (2 * alpha2 * beta) * ((a**2 - r2) * E + alpha2 * K)
    return np.sign(X) * Brho, Bz


def loop_flux(X, Z, a=0.5):
    """Stokes flux function rho*A_phi of a circular loop (shape only)."""
    rho = np.abs(X) + 1e-9
    k2 = 4 * a * rho / ((a + rho) ** 2 + Z**2)
    k2 = np.clip(k2, 0, 1 - 1e-12)
    K, E = ellipk(k2), ellipe(k2)
    # Jackson (5.37):  A_phi = (mu0 I/(pi k)) sqrt(a/rho) [(1-k^2/2)K - E]
    Aphi = 1 / np.sqrt(k2 + 1e-30) * np.sqrt(a / rho) * ((1 - k2 / 2) * K - E)
    return rho * Aphi


def charges_flux(X, Z, d=0.6):
    """Stream function of the two-charge field: sum q_i (z - z_i)/|r - r_i|."""
    out = np.zeros_like(X)
    for q, z0 in ((1.0, d / 2), (-1.0, -d / 2)):
        out += q * (Z - z0) / np.hypot(X, Z - z0)
    return out


def sphere_fields(X, Z, kind: str, R=1.0):
    """Uniformly polarised (kind='E') or magnetised (kind='B') sphere, moment along +z.

    Inside: E = -P/(3 eps0)  (opposite P);  B = 2 mu0 M / 3  (along M).
    Outside both are the field of a point dipole p or m = (4/3) pi R^3 P (or M).
    Units: P = eps0 = 1 and M = mu0 = 1.
    """
    r = np.hypot(X, Z) + 1e-12
    inside = r < R
    m = 4 / 3 * np.pi * R**3
    # dipole field  (3 (m.r_hat) r_hat - m) / (4 pi r^3)
    cos = Z / r
    Fx = m / (4 * np.pi * r**3) * (3 * cos * X / r)
    Fz = m / (4 * np.pi * r**3) * (3 * cos * Z / r - 1)
    inner = -1 / 3 if kind == "E" else 2 / 3
    Fx = np.where(inside, 0.0, Fx)
    Fz = np.where(inside, inner, Fz)
    return Fx, Fz


# ---------------------------------------------------------------------------
@figure("dipoles_pair")
def dipoles_pair():
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.2))
    g = np.linspace(-5, 5, 1601)
    X, Z = np.meshgrid(g, g)
    d = 0.6
    V = 1.55
    # --- electric dipole: seed around the + charge, lines end on the - charge
    ax = axs[0]
    Ex, Ez = point_dipole_charges(X, Z, d)
    ax.set_xlim(-V, V); ax.set_ylim(-V, V)
    flux_lines(ax, X, Z, charges_flux(X, Z, d), np.linspace(-2, 0, 11)[1:-1], Ex, Ez,
               color=ELEC, lw=1.0, arrow_pos="xmax")
    charge(ax, 0, d / 2, "+", r=0.09, size=8)
    charge(ax, 0, -d / 2, "-", r=0.09, size=8)
    halo(arrow(ax, (0, -0.17), (0, 0.17), color=INK, lw=2.0, ms=10, z=9), 4)
    halo(label(ax, 0.17, 0.0, r"$\mathbf{p}$", size=11, ha="left", z=10), 3)
    label(ax, 0, -1.42, T("两电荷之间：E 向下，与 p 反向", "between the charges: E points down, against p"), size=8.3, color=ELEC,
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    panel_title(ax, T("电偶极子：一对正负电荷", "Electric dipole: a ± charge pair"), color=ELEC)
    # --- magnetic dipole (current loop seen edge-on)
    ax = axs[1]
    a = 0.42
    Bx, Bz = loop_field(X, Z, a)
    ax.set_xlim(-V, V); ax.set_ylim(-V, V)
    psi = loop_flux(X, Z, a)
    ref = float(loop_flux(np.array([[0.3]]), np.array([[0.0]]), a)[0, 0])
    flux_lines(ax, X, Z, psi, ref * np.linspace(0.1, 1.0, 8) ** 1.0, Bx, Bz,
               color=MAG, lw=1.0, arrow_pos="xmax")
    out_of_page(ax, -a, 0, color=CURR, r=0.07)
    into_page(ax, a, 0, color=CURR, r=0.07)
    halo(arrow(ax, (0, -0.2), (0, 0.2), color=INK, lw=2.0, ms=10, z=9), 4)
    halo(label(ax, 0.1, -0.3, r"$\mathbf{m}$", size=11, ha="left", z=10), 3)
    label(ax, 0, -1.42, T("环内：B 向上，与 m 同向", "inside the loop: B points up, along m"), size=8.3, color=MAG,
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    panel_title(ax, T("磁偶极子：一个小电流环", "Magnetic dipole: a small current loop"), color=MAG)
    for ax in axs:
        ax.set_xlim(-V, V)
        ax.set_ylim(-V, V)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, 0.015, T("远处两者的场线形状相同（都是偶极场）；区别在源附近：电场线起止于电荷，磁场线自成闭合回路。",
                           "Far away the two patterns coincide (both are dipole fields); near the source, E lines start and end on charges, B lines close on themselves."),
             ha="center", fontsize=8.6, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.08, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
@figure("bound_sources")
def bound_sources():
    fig, ax = canvas(FIG_W, 2.75, xlim=(0, 10.9), ylim=(0, 5.0))
    # ----- left: polarised slab
    x0, y0, w, h = 0.55, 0.95, 4.2, 2.6
    rbox(ax, x0, y0, w, h, fc=ELEC_T, ec=tint(ELEC, 0.5), r=0.08)
    for i in range(4):
        for j in range(3):
            cx = x0 + 0.62 + i * 0.98
            cy = y0 + 0.5 + j * 0.8
            ax.add_patch(Ellipse((cx, cy), 0.66, 0.34, fc="white", ec=tint(ELEC, 0.6), lw=0.7, zorder=3))
            label(ax, cx - 0.2, cy, "−", color=ELEC, size=8, weight="bold")
            label(ax, cx + 0.2, cy, "+", color=MAG, size=8, weight="bold")
    for j in range(5):
        yy = y0 + 0.28 + j * 0.51
        label(ax, x0 - 0.2, yy, "−", color=ELEC, size=10, weight="bold")
        label(ax, x0 + w + 0.2, yy, "+", color=MAG, size=10, weight="bold")
    arrow(ax, (x0 + 1.4, y0 + h + 0.38), (x0 + 2.8, y0 + h + 0.38), color=ELEC, lw=2.0, ms=12)
    label(ax, x0 + 3.05, y0 + h + 0.4, r"$\mathbf{P}$", color=ELEC, size=11, ha="left")
    label(ax, x0 + w / 2, y0 + h + 0.95, T("均匀极化的平板", "Uniformly polarized slab"), weight="bold", size=9.6)
    label(ax, x0 + w / 2, 0.55, T(r"体内 $\rho_b=-\nabla\cdot\mathbf{P}=0$", r"interior $\rho_b=-\nabla\cdot\mathbf{P}=0$"), size=8.9)
    label(ax, x0 + w / 2, 0.15, T(r"表面 $q_{b,s}=\mathbf{P}\cdot\mathbf{n}$：左负右正", r"surfaces $q_{b,s}=\mathbf{P}\cdot\mathbf{n}$: − left, + right"), size=8.9)
    # ----- right: magnetised slab (current loops)
    x1 = 6.15
    rbox(ax, x1, y0, w, h, fc=MAG_T, ec=tint(MAG, 0.5), r=0.08)
    for i in range(4):
        for j in range(3):
            cx = x1 + 0.62 + i * 0.98
            cy = y0 + 0.5 + j * 0.8
            ax.add_patch(Circle((cx, cy), 0.27, fc="white", ec=tint(CURR, 0.75), lw=0.9, zorder=3))
            arrow(ax, (cx + 0.27, cy - 0.02), (cx + 0.27, cy + 0.03), color=CURR, lw=0.9, ms=6, z=4)
            out_of_page(ax, cx, cy, color=MAG, r=0.07, lw=0.8)
    # surface current around the slab (counter-clockwise)
    pad = 0.13
    pts = [(x1 - pad, y0 - pad), (x1 + w + pad, y0 - pad), (x1 + w + pad, y0 + h + pad), (x1 - pad, y0 + h + pad), (x1 - pad, y0 - pad)]
    ax.plot(*zip(*pts), color=CURR, lw=1.6, zorder=4)
    for p, q in (((x1 + 1.8, y0 - pad), (x1 + 2.4, y0 - pad)), ((x1 + w + pad, y0 + 1.0), (x1 + w + pad, y0 + 1.6)),
                 ((x1 + 2.4, y0 + h + pad), (x1 + 1.8, y0 + h + pad)), ((x1 - pad, y0 + 1.6), (x1 - pad, y0 + 1.0))):
        arrow(ax, p, q, color=CURR, lw=1.6, ms=11, z=5)
    out_of_page(ax, x1 + 1.75, y0 + h + 0.4, color=MAG, r=0.11)
    label(ax, x1 + 2.05, y0 + h + 0.4, T(r"$\mathbf{M}$ 指向纸外", r"$\mathbf{M}$ out of page"), color=MAG, size=9.5, ha="left")
    label(ax, x1 + w / 2, y0 + h + 0.95, T("均匀磁化的平板（截面）", "Uniformly magnetized slab (cross-section)"), weight="bold", size=9.6)
    label(ax, x1 + w / 2, 0.55, T(r"体内 $\mathbf{J}_M=\nabla\times\mathbf{M}=0$", r"interior $\mathbf{J}_M=\nabla\times\mathbf{M}=0$"), size=8.9)
    label(ax, x1 + w / 2, 0.15, T(r"边界 $\mathbf{K}_M=\mathbf{M}\times\mathbf{n}$：绕边界环流", r"boundary $\mathbf{K}_M=\mathbf{M}\times\mathbf{n}$: circulates"), size=8.9)
    return fig


# ---------------------------------------------------------------------------
@figure("sphere_pair")
def sphere_pair():
    fig = plt.figure(figsize=(FIG_W, 4.25))
    axE = fig.add_axes([0.0, 0.30, 0.49, 0.68])
    axB = fig.add_axes([0.51, 0.30, 0.49, 0.68])
    barE = fig.add_axes([0.08, 0.05, 0.36, 0.2])
    barB = fig.add_axes([0.59, 0.05, 0.36, 0.2])
    g = np.linspace(-2.6, 2.6, 1041)
    X, Z = np.meshgrid(g, g)
    r = np.hypot(X, Z)
    rho2 = X**2
    psi_out = (1 / 3) * rho2 / np.maximum(r, 1e-9) ** 3
    rho_i = np.linspace(0.14, 0.86, 6)
    levels = rho_i**2 / 3
    for ax, kind, col, fill in ((axE, "E", ELEC, ELEC_T), (axB, "B", MAG, MAG_T)):
        Fx, Fz = sphere_fields(X, Z, kind)
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-2.0, 2.15)
        ax.add_patch(Circle((0, 0), 1.0, fc=fill, ec=tint(col, 0.6), lw=1.0, zorder=1))
        flux_lines(ax, X, Z, psi_out, levels, Fx, Fz, color=col, lw=1.0, mask=r < 1.0, arrow_pos="xmax")
        for x in np.concatenate((-rho_i, rho_i)):
            zt = np.sqrt(1 - x**2)
            ax.plot([x, x], [-zt, zt], color=col, lw=1.0, zorder=3)
            if kind == "E":
                arrow(ax, (x, 0.05), (x, -0.05), color=col, lw=1.0, ms=7, z=3.5)
            else:
                arrow(ax, (x, -0.05), (x, 0.05), color=col, lw=1.0, ms=7, z=3.5)
        if kind == "E":
            for t in np.linspace(0.25, np.pi - 0.25, 7):
                label(ax, 1.13 * np.cos(t), 1.13 * np.sin(t), "+", color=MAG, size=8.5, weight="bold")
                label(ax, 1.13 * np.cos(t), -1.13 * np.sin(t), "−", color=ELEC, size=9, weight="bold")
            halo(arrow(ax, (-1.85, -0.45), (-1.85, 0.45), color=INK, lw=2.0, ms=11), 4)
            halo(label(ax, -1.68, 0.0, r"$\mathbf{P}$", size=11, ha="left"), 3)
            panel_title(ax, T("均匀极化球：E 线起止于表面束缚电荷", "Polarized sphere: E ends on bound charge"), color=ELEC, size=T(9.4, 8.8))
        else:
            halo(arrow(ax, (-1.85, -0.45), (-1.85, 0.45), color=INK, lw=2.0, ms=11), 4)
            halo(label(ax, -1.68, 0.0, r"$\mathbf{M}$", size=11, ha="left"), 3)
            panel_title(ax, T("均匀磁化球：B 线穿过球面、自成闭合回路", "Magnetized sphere: B lines close on themselves"), color=MAG, size=T(9.4, 8.8))
        ax.set_aspect("equal")
        ax.axis("off")
    # signed bars (inside values)
    def bars(ax, names, vals, cols):
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=[tint(c, 0.75) for c in cols], edgecolor=cols, lw=0.8, height=0.62)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, names)
        ax.set_xlim(-0.55, 1.12)
        ax.set_xticks([-1 / 3, 0, 2 / 3, 1], [r"$-\frac{1}{3}$", "0", r"$\frac{2}{3}$", "1"])
        for yy, v in zip(y, vals):
            ax.text(v + (0.03 if v >= 0 else -0.03), yy, f"{v:+.2f}".replace("+0.33", "").replace("+", "") if False else "",
                    va="center", fontsize=8)
        clean_axes(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9)
    bars(barE, [r"$\mathbf{P}$", r"$\varepsilon_0\mathbf{E}$", r"$\mathbf{D}$"], [1, -1 / 3, 2 / 3], [ELEC, ELEC, ELEC])
    bars(barB, [r"$\mu_0\mathbf{M}$", r"$\mu_0\mathbf{H}$", r"$\mathbf{B}$"], [1, -1 / 3, 2 / 3], [MAG, MAG, MAG])
    barE.set_xlabel(T(r"球内取值（以 $P$ 为单位）", r"inside value (units of $P$)"), fontsize=8.6)
    barB.set_xlabel(T(r"球内取值（以 $\mu_0M$ 为单位）", r"inside value (units of $\mu_0M$)"), fontsize=8.6)
    return fig


# ---------------------------------------------------------------------------
@figure("em_analogy")
def em_analogy():
    fig, ax = canvas(FIG_W, 3.7, xlim=(0, 10.4), ylim=(0, 6.45))
    rows = [5.0, 3.65, 2.3, 0.95]
    xl, xr, w, h = 0.3, 7.3, 2.8, 0.95
    left = [(r"$\mathbf{E}$", T("电场强度 V/m", "electric field\nV/m")),
            (r"$\mathbf{D}$", T("电位移 C/m²", "displacement\nC/m²")),
            (r"$\mathbf{P}$", T("极化强度 C/m²", "polarization\nC/m²")),
            (r"$\rho_f$", T("自由电荷 C/m³", "free charge\nC/m³"))]
    right = [(r"$\mathbf{B}$", T("磁通密度 T", "flux density\nT")),
             (r"$\mathbf{H}$", T("磁场强度 A/m", "H field\nA/m")),
             (r"$\mathbf{M}$", T("磁化强度 A/m", "magnetization\nA/m")),
             (r"$\mathbf{J}_f$", T("自由电流 A/m²", "free current\nA/m²"))]
    for (sym, txt), y in zip(left, rows):
        rbox(ax, xl, y, w, h, fc=ELEC_T, ec=tint(ELEC, 0.55), lw=0.9)
        label(ax, xl + 0.45, y + h / 2, sym, color=ELEC, size=13)
        label(ax, xl + 0.85, y + h / 2, txt, size=T(8.8, 8.4), ha="left")
    for (sym, txt), y in zip(right, rows):
        rbox(ax, xr, y, w, h, fc=MAG_T, ec=tint(MAG, 0.55), lw=0.9)
        label(ax, xr + 0.45, y + h / 2, sym, color=MAG, size=13)
        label(ax, xr + 0.85, y + h / 2, txt, size=T(8.8, 8.4), ha="left")
    yc = lambda i: rows[i] + h / 2
    L, Rr = xl + w + 0.05, xr - 0.05
    # physical (force) analogy: E-B, P-M, rho-J
    for i, j, txt in ((0, 0, T("力：$q\\mathbf{E}$ 与 $q\\mathbf{v}\\times\\mathbf{B}$", "force: $q\\mathbf{E}$ and $q\\mathbf{v}\\times\\mathbf{B}$")),
                      (2, 2, T("物质的偶极密度", "dipole density of matter")),
                      (3, 3, T("自由源", "free sources"))):
        ax.plot([L, Rr], [yc(i), yc(j)], color=INK, lw=1.5, zorder=1)
        label(ax, 5.2, yc(i) + 0.2, txt, size=8.4, color=INK,
              bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    # formal analogy: E-H and D-B (crossing)
    ax.plot([L, Rr], [yc(0), yc(1)], color=ACCENT, lw=1.5, ls=(0, (4, 2.5)), zorder=1)
    ax.plot([L, Rr], [yc(1), yc(0)], color=ACCENT, lw=1.5, ls=(0, (4, 2.5)), zorder=1)
    label(ax, 5.2, 4.0, T("形式：$\\mathbf{E}\\leftrightarrow\\mathbf{H}$ 用环量与切向分量，\n$\\mathbf{D}\\leftrightarrow\\mathbf{B}$ 用通量与法向分量",
                          "formal: $\\mathbf{E}\\leftrightarrow\\mathbf{H}$ (circulation, tangential);\n$\\mathbf{D}\\leftrightarrow\\mathbf{B}$ (flux, normal)"),
          size=T(8.2, 7.8), color=ACCENT, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    # legend
    ax.plot([0.35, 0.95], [0.35, 0.35], color=INK, lw=1.5)
    label(ax, 1.05, 0.35, T("物理类比（力与物质）", "physical analogy (force, matter)"), size=8.4, ha="left")
    ax.plot([5.3, 5.9], [0.35, 0.35], color=ACCENT, lw=1.5, ls=(0, (4, 2.5)))
    label(ax, 6.0, 0.35, T("形式类比（方程与边界条件）", "formal analogy (equations, interfaces)"), size=8.4, ha="left", color=ACCENT)
    label(ax, xl + w / 2, 6.25, T("电", "Electric"), color=ELEC, weight="bold", size=10.5)
    label(ax, xr + w / 2, 6.25, T("磁", "Magnetic"), color=MAG, weight="bold", size=10.5)
    return fig


# ---------------------------------------------------------------------------
@figure("e_d_capacitor")
def e_d_capacitor():
    fig = plt.figure(figsize=(FIG_W, 2.9))
    ax = fig.add_axes([0, 0.0, 0.56, 1.0])
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")
    for k, (x0, er) in enumerate(((0.3, 1), (3.8, 4))):
        # plates
        ax.add_patch(Rectangle((x0, 3.2), 2.8, 0.14, fc="#9AA3AB", ec="none"))
        ax.add_patch(Rectangle((x0, 0.86), 2.8, 0.14, fc="#9AA3AB", ec="none"))
        for i in range(6):
            xx = x0 + 0.25 + i * 0.46
            label(ax, xx, 3.5, "+", color=MAG, size=9, weight="bold")
            label(ax, xx, 0.68, "−", color=ELEC, size=10, weight="bold")
        if er > 1:
            ax.add_patch(Rectangle((x0 + 0.04, 1.05), 2.72, 2.1, fc=MATL_T, ec=tint(MATL, 0.6), lw=0.8, zorder=1))
            for i in range(6):
                xx = x0 + 0.25 + i * 0.46
                label(ax, xx, 2.96, "−", color=ELEC, size=8, weight="bold", alpha=0.9)
                label(ax, xx, 1.22, "+", color=MAG, size=7.5, weight="bold", alpha=0.9)
        n_lines = 6 if er == 1 else 2
        xs = np.linspace(x0 + 0.35, x0 + 2.45, n_lines)
        for xx in xs:
            arrow(ax, (xx, 2.95 if er > 1 else 3.12), (xx, 1.3 if er > 1 else 1.08), color=ELEC, lw=1.2, ms=8, z=4)
        label(ax, x0 + 1.4, 3.85, T("真空间隙", "vacuum gap") if er == 1 else T(r"介质 $\varepsilon_r=4$", r"dielectric $\varepsilon_r=4$"), weight="bold", size=9.2)
    label(ax, 3.5, 0.2, T("自由电荷 ±Q 相同；E 线（蓝）在介质里变稀", "same free charge ±Q; E lines (blue) thin out in the dielectric"), size=8.4, color=MUTED)
    # stacked bars D = eps0 E + P
    bx = fig.add_axes([0.64, 0.2, 0.34, 0.68])
    cats = [T("真空", "vacuum"), T("介质", "dielectric")]
    e = np.array([1.0, 0.25])
    p = np.array([0.0, 0.75])
    bx.bar(cats, e, color=tint(ELEC, 0.55), edgecolor=ELEC, lw=0.8, width=0.55, label=r"$\varepsilon_0E$")
    bx.bar(cats, p, bottom=e, color=tint(MATL, 0.45), edgecolor=MATL, lw=0.8, width=0.55, hatch="////", label=r"$P$")
    bx.axhline(1.0, color=INK, lw=1.0, ls=(0, (3, 2)))
    bx.text(1.36, 1.0, r"$D=Q/A_c$", va="center", fontsize=9, clip_on=False)
    bx.set_ylim(0, 1.18)
    bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_ylabel(T(r"以 $D$ 为单位", r"in units of $D$"))
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=8.8, handlelength=1.4)
    clean_axes(bx)
    return fig


# ---------------------------------------------------------------------------
@figure("displacement_current")
def displacement_current():
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    fig, ax = canvas(FIG_W, 2.75, xlim=(0, 10.6), ylim=(0, 4.85))
    yw = 2.35
    ax.plot([0.2, 4.6], [yw, yw], color=MUTED, lw=2.4, zorder=2, solid_capstyle="butt")
    ax.plot([5.6, 10.4], [yw, yw], color=MUTED, lw=2.4, zorder=2, solid_capstyle="butt")
    ax.add_patch(Rectangle((4.55, 1.45), 0.14, 1.8, fc="#9AA3AB", ec="none", zorder=3))
    ax.add_patch(Rectangle((5.51, 1.45), 0.14, 1.8, fc="#9AA3AB", ec="none", zorder=3))
    for yy in np.linspace(1.62, 3.08, 4):
        label(ax, 4.42, yy, "+", color=MAG, size=8, weight="bold")
        label(ax, 5.8, yy, "−", color=ELEC, size=9, weight="bold")
        arrow(ax, (4.75, yy), (5.45, yy), color=ELEC, lw=1.1, ms=7, z=4)
    arrow(ax, (0.7, yw + 0.3), (1.9, yw + 0.3), color=CURR, lw=1.8, ms=11)
    label(ax, 1.3, yw + 0.65, r"$I$", color=CURR, size=11)
    arrow(ax, (7.9, yw + 0.3), (9.1, yw + 0.3), color=CURR, lw=1.8, ms=11)
    label(ax, 8.5, yw + 0.65, r"$I$", color=CURR, size=11)
    label(ax, 6.25, 3.45, r"$\partial_t\mathbf{D}$", color=ELEC, size=10.5)
    xc = 3.0
    ax.add_patch(Ellipse((xc, yw), 0.42, 1.4, fc=tint(CURR, 0.2), ec="none", zorder=1))
    ax.add_patch(Ellipse((xc, yw), 0.42, 1.4, fc="none", ec=INK, lw=1.3, zorder=5))
    arrow(ax, (xc + 0.2, yw + 0.22), (xc + 0.19, yw + 0.32), color=INK, lw=1.3, ms=9, z=6)
    label(ax, xc, yw - 0.95, r"$C$", size=10.5)
    label(ax, 1.55, 1.15, T(r"$S_1$：平面圆片，被导线穿过", r"$S_1$: flat disc, pierced by the wire"), size=8.6, color=CURR)
    # S2: a bag bounded by C that wraps the left plate and passes through the gap
    verts = [(xc, yw + 0.7), (3.9, 4.05), (5.1, 3.85), (5.1, 3.4),
             (5.1, 2.4), (5.1, 1.3), (5.1, 0.85), (3.9, 0.65), (xc, yw - 0.7)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    ax.add_patch(PathPatch(MPath(verts, codes), fc="none", ec=ELEC, lw=1.3, ls=(0, (4, 2)), zorder=4))
    label(ax, 7.55, 4.3, T(r"$S_2$：口袋形曲面，包住左极板、从两板之间穿过", r"$S_2$: a bag around the left plate, crossing the gap"), size=8.6, color=ELEC)
    label(ax, 7.55, 3.85, T("穿过它的没有传导电流，只有位移电流", "no conduction current crosses it — only displacement current"), size=8.6, color=ELEC)
    label(ax, 7.6, 0.55, T(r"同一个回路 $C$ 上，$\oint\mathbf{H}\cdot\mathrm{d}\boldsymbol{\ell}$ 只能有一个值", r"on one loop $C$, $\oint\mathbf{H}\cdot\mathrm{d}\boldsymbol{\ell}$ can have only one value"), size=8.8)
    return fig


# ---------------------------------------------------------------------------
@figure("source_bookkeeping")
def source_bookkeeping():
    fig, ax = canvas(FIG_W, 2.95, xlim=(0, 10.6), ylim=(0, 5.1))

    def chips(x, y, items):
        xx = x
        for s_, c in items:
            ww = 0.62 + 0.085 * len(s_.replace("\\", "").replace("$", "").replace("mathbf", "").replace("partial", "d").replace("nabla", "V").replace("times", "x").replace("cdot", "."))
            rbox(ax, xx, y - 0.24, ww, 0.48, fc=tint(c, 0.16), ec=tint(c, 0.6), lw=0.8, r=0.1)
            label(ax, xx + ww / 2, y, s_, color=c, size=9)
            xx += ww + 0.1

    # left ledger
    label(ax, 2.55, 4.8, T("写法一：全部物质源，只用 E、B", "Form 1: all material sources; fields E, B only"), weight="bold", size=9.2)
    label(ax, 0.45, 3.85, T("电荷", "charge"), size=8.6, color=MUTED)
    chips(0.95, 3.85, [(r"$\rho_f$", CURR), (r"$-\nabla\cdot\mathbf{P}$", ELEC)])
    label(ax, 0.45, 3.05, T("电流", "current"), size=8.6, color=MUTED)
    chips(0.95, 3.05, [(r"$\mathbf{J}_f$", CURR), (r"$\partial_t\mathbf{P}$", ELEC), (r"$\nabla\times\mathbf{M}$", MAG)])
    label(ax, 2.55, 2.05, r"$\nabla\cdot(\varepsilon_0\mathbf{E})=\rho_f-\nabla\cdot\mathbf{P}$", size=9.3)
    label(ax, 2.55, 1.4, r"$\nabla\times(\mathbf{B}/\mu_0)=\mathbf{J}_f+\partial_t\mathbf{P}+\nabla\times\mathbf{M}+\varepsilon_0\partial_t\mathbf{E}$", size=8.8)
    # arrow
    arrow(ax, (5.25, 3.45), (6.05, 3.45), color=ACCENT, lw=2.2, ms=14)
    label(ax, 5.65, 2.75, T("束缚项\n并入场变量", "absorb bound\nterms"), size=8.2, color=ACCENT)
    # right ledger
    x1 = 6.35
    label(ax, 8.45, 4.8, T("写法二：只留自由源，引入 D、H", "Form 2: free sources only; introduce D, H"), weight="bold", size=9.2)
    xs = x1 + T(0.3, 0.6)
    label(ax, xs, 3.85, T("电荷", "charge"), size=8.6, color=MUTED)
    chips(xs + 0.5, 3.85, [(r"$\rho_f$", CURR)])
    label(ax, xs, 3.05, T("电流", "current"), size=8.6, color=MUTED)
    chips(xs + 0.5, 3.05, [(r"$\mathbf{J}_f$", CURR)])
    label(ax, 8.45, 2.05, r"$\nabla\cdot\mathbf{D}=\rho_f,\quad \mathbf{D}=\varepsilon_0\mathbf{E}+\mathbf{P}$", size=9.3)
    label(ax, 8.45, 1.4, r"$\nabla\times\mathbf{H}=\mathbf{J}_f+\partial_t\mathbf{D},\quad \mathbf{H}=\mathbf{B}/\mu_0-\mathbf{M}$", size=9.0)
    label(ax, 5.3, 0.45, T("两种写法描述同一个物理场；区别只在哪些项被\u201c记\u201d在源里、哪些被\u201c记\u201d在场变量里。",
                           "Both forms describe the same field; they differ only in which terms are booked as sources and which inside the field variables."),
          size=8.4, color=MUTED)
    return fig
