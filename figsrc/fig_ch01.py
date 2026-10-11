"""Figures for Chapter 1 (one question that opens the course).

intro_scene    side view of a coil over a head (exact loop B) + top view of a thin
               conducting sheet: primary induced E vs total E (exact 2-D Neumann solve)
               + the causal chain.
whole_problem  the four layers of a model on a common foundation.
maxwell_map    the four Maxwell equations, each with a pictogram and the chapter that builds it.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

from style import (ACCENT, AMBER, CURR, ELEC, FIG_W, INK, IRON_T, MAG, MATL, MATL_T, MUTED, NEUTRAL_T,
                   POT, T, arrow, canvas, charge, figure, flux_lines, halo, into_page, label, out_of_page,
                   panel_title, plt, rbox, tint)
from fig_matter import loop_field, loop_flux


# ---------------------------------------------------------------------------
# Thin conducting disc under an off-centre circular coil (dI/dt > 0)
# ---------------------------------------------------------------------------
def _loop_aphi(rho, h, a):
    """Azimuthal vector potential (shape only) of a loop of radius a at height h."""
    rho = np.asarray(rho, float)
    out = loop_flux(rho + 1e-9, np.full_like(rho, h), a) / (rho + 1e-9)
    return out


def sheet_fields(X, Y, R=1.5, c=(0.45, 0.25), a=0.55, h=0.45, nmodes=80):
    """Primary and total in-plane E in an insulated thin disc of radius R.

    Primary field: E_p = -dA/dt of a coil (current increasing, counter-clockwise seen from
    above) -> clockwise circles about the coil axis, |E_p| proportional to A_phi.
    Total field:   E = E_p - grad(phi), with Laplace(phi)=0 in the disc and
    d(phi)/dn = E_p . n on the rim (no current leaves the insulated rim).  The Neumann
    problem is solved exactly with a Fourier series; phi = Re f(z).
    Returns (Epx, Epy, Ex, Ey, psi_total, psi_primary, g(theta) samples).
    """
    cx, cy = c
    # primary field on the grid ------------------------------------------------
    rr = np.linspace(0.0, 6.0, 6001)
    Aphi = _loop_aphi(rr, h, a)
    psi_p_1d = np.concatenate(([0.0], np.cumsum(0.5 * (Aphi[1:] + Aphi[:-1]) * np.diff(rr))))

    def primary(x, y):
        dx, dy = x - cx, y - cy
        r = np.hypot(dx, dy) + 1e-12
        Ephi = -np.interp(r, rr, Aphi)              # clockwise: E_phi < 0
        return Ephi * (-dy / r), Ephi * (dx / r), np.interp(r, rr, psi_p_1d)

    Epx, Epy, psi_p = primary(X, Y)
    # Neumann data on the rim -----------------------------------------------------
    th = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    xr, yr = R * np.cos(th), R * np.sin(th)
    ex, ey, _ = primary(xr, yr)
    g = ex * np.cos(th) + ey * np.sin(th)
    G = np.fft.rfft(g) / len(th)
    n = np.arange(1, nmodes + 1)
    an = 2 * G.real[1:nmodes + 1]
    bn = -2 * G.imag[1:nmodes + 1]
    coef = (an - 1j * bn)
    Z = (X + 1j * Y) / R
    fprime = np.zeros_like(Z)
    f = np.zeros_like(Z)
    Zp = np.ones_like(Z)
    for k, ck in zip(n, coef):
        fprime += ck * Zp            # (z/R)^(k-1)
        Zp = Zp * Z
        f += ck * R * Zp / k         # z^k / (k R^(k-1))
    gphx, gphy = fprime.real, -fprime.imag
    Ex, Ey = Epx - gphx, Epy - gphy
    psi = psi_p - f.imag
    return Epx, Epy, Ex, Ey, psi, psi_p, (th, g)


# ---------------------------------------------------------------------------
@figure("intro_scene")
def intro_scene():
    fig = plt.figure(figsize=(FIG_W, 3.55))
    axB = fig.add_axes([0.0, 0.235, 0.46, 0.76])
    axN = fig.add_axes([0.47, 0.235, 0.53, 0.76])
    axC = fig.add_axes([0.0, 0.0, 1.0, 0.215])

    # ---------------- (a) top view: thin conducting disc ------------------------
    ax = axB
    L = 1.85
    ax.set_xlim(-L, L)
    ax.set_ylim(-L + 0.05, L - 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    R, c, ac = 1.45, (0.42, 0.22), 0.5
    g = np.linspace(-L, L, 721)
    X, Y = np.meshgrid(g, g)
    Epx, Epy, Ex, Ey, psi, psi_p, (th, gn) = sheet_fields(X, Y, R=R, c=c, a=ac, h=0.45)
    inside = np.hypot(X, Y) < R * 0.999
    ax.add_patch(Circle((0, 0), R, fc=MATL_T, ec=tint(MATL, 0.8), lw=1.1, zorder=1))
    # primary circles (dashed) about the coil axis
    for rad in (0.75, 1.12, 1.4):
        ax.add_patch(Circle(c, rad, fc="none", ec=tint(ELEC, 0.55), lw=0.9, ls=(0, (3, 2.2)), zorder=2))
    # total field lines inside the disc (contours of the stream function)
    ps_in = np.where(inside, psi, np.nan)
    rim = float(np.nanmean(psi[(np.hypot(X, Y) > R * 0.985) & inside]))
    ext = float(np.nanmax(ps_in)) if abs(np.nanmax(ps_in) - rim) > abs(np.nanmin(ps_in) - rim) else float(np.nanmin(ps_in))
    levels = np.sort(rim + (ext - rim) * np.array([0.12, 0.3, 0.5, 0.7, 0.88]))
    flux_lines(ax, X, Y, psi, levels, Ex, Ey, color=ELEC, lw=1.15, mask=~inside, arrow_pos=0.3, z=4)
    # coil ring (above the sheet)
    ax.add_patch(Circle(c, ac, fc="none", ec=CURR, lw=1.6, zorder=5))
    arrow(ax, (c[0] + ac * np.cos(1.2), c[1] + ac * np.sin(1.2)), (c[0] + ac * np.cos(1.35), c[1] + ac * np.sin(1.35)),
          color=CURR, lw=1.6, ms=9, z=6)
    # rim charges: sign of E_p . n
    gmax = np.abs(gn).max()
    for t in np.linspace(0, 2 * np.pi, 22, endpoint=False):
        val = np.interp(t, th, gn, period=2 * np.pi)
        if abs(val) < 0.3 * gmax:
            continue
        rr = R + 0.15
        charge(ax, rr * np.cos(t), rr * np.sin(t), "+" if val > 0 else "-", r=0.085, size=7.5)
    halo(label(ax, c[0] + 0.05, c[1] - 0.73, T("线圈（在上方）", "coil (above)"), size=8.0, color=CURR), 3)

    # ---------------- (b) reading guide ------------------------------------------
    ax = axN
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    notes = [
        ("dash", T("初级电场 $\\mathbf{E}_p=-\\partial_t\\mathbf{A}$", "primary field $\\mathbf{E}_p=-\\partial_t\\mathbf{A}$"),
         T("只由线圈决定，绕线圈轴转圈；\n它“不知道”导体的边界在哪里。", "set by the coil alone, circling its axis;\nit does not know where the conductor ends.")),
        ("charge", T("边界电荷", "boundary charge"),
         T("电流不能穿出绝缘的边界，电荷在那里堆积：\n$\\mathbf{E}_p$ 指向外处为正，指向内处为负。", "no current can leave the rim; charge piles up:\npositive where $\\mathbf{E}_p$ points out, negative where in.")),
        ("solid", T("总电场 $\\mathbf{E}=\\mathbf{E}_p+\\mathbf{E}_s$", "total field $\\mathbf{E}=\\mathbf{E}_p+\\mathbf{E}_s$"),
         T("电荷的次级场 $\\mathbf{E}_s$ 把电场“掰”成\n贴着边界流动；刺激与发热看的是它。", "the charges' secondary field $\\mathbf{E}_s$ bends E to run\nalong the rim; stimulation and heating see this field.")),
    ]
    for k, (kind, head, body) in enumerate(notes):
        y = 4.05 - k * 1.5
        if kind == "dash":
            ax.plot([0.1, 0.62], [y, y], color=tint(ELEC, 0.55), lw=1.2, ls=(0, (3, 2.2)))
        elif kind == "solid":
            ax.plot([0.1, 0.62], [y, y], color=ELEC, lw=1.5)
            arrow(ax, (0.3, y), (0.45, y), color=ELEC, lw=1.5, ms=8)
        else:
            charge(ax, 0.22, y, "+", r=0.11, size=7.5)
            charge(ax, 0.52, y, "-", r=0.11, size=7.5)
        label(ax, 0.82, y, head, size=9.0, ha="left", color=ELEC if kind != "charge" else INK, weight="bold")
        label(ax, 0.82, y - 0.55, body, size=8.1, ha="left", color=INK)

    # ---------------- (c) causal chain -------------------------------------------
    ax = axC
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 1.34)
    ax.axis("off")
    steps = [(r"$I(t)$", T("线圈电流", "coil current"), CURR),
             (r"$\mathbf{B}(t)$", T("磁场", "magnetic field"), MAG),
             (r"$\partial_t\mathbf{B}\rightarrow\mathbf{E}_p$", T("感应电场", "induced field"), ELEC),
             (r"$\mathbf{J}=\sigma\mathbf{E}$", T("组织电流", "tissue current"), CURR),
             (r"$q_s\rightarrow\mathbf{E}_s$", T("边界电荷", "boundary charge"), ELEC),
             (r"$\mathbf{E}=\mathbf{E}_p+\mathbf{E}_s$", T("神经刺激、发热", "nerves, heat"), ACCENT)]
    w, gap, x0, y0, hh = 1.5, 0.26, 0.08, 0.1, 0.92
    for i, (sym, txt, col) in enumerate(steps):
        x = x0 + i * (w + gap)
        rbox(ax, x, y0, w, hh, fc=tint(col, 0.1), ec=tint(col, 0.6), lw=0.9, r=0.1)
        label(ax, x + w / 2, y0 + hh * 0.66, sym, color=col, size=9.0)
        label(ax, x + w / 2, y0 + hh * 0.27, txt, color=INK, size=8.2)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.03, y0 + hh / 2), (x + w + gap - 0.03, y0 + hh / 2), color=MUTED, lw=1.2, ms=8)
    # feedback: boundary charge corrects the field that drives the current
    xs = x0 + 4 * (w + gap) + w / 2
    xd = x0 + 3 * (w + gap) + w / 2
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((xs, y0 + hh + 0.02), (xd, y0 + hh + 0.02), connectionstyle="arc3,rad=0.35",
                                 arrowstyle="-|>", mutation_scale=8, lw=1.0, color=ELEC, zorder=5))
    label(ax, xs + 0.18, y0 + hh + 0.2, T("电荷反过来修正电场", "charge feeds back on E"),
          color=ELEC, size=7.9, ha="left")
    return fig


# ---------------------------------------------------------------------------
@figure("whole_problem")
def whole_problem():
    fig, ax = canvas(FIG_W, 3.11, xlim=(0, 10.4), ylim=(0.25, 5.64))
    rows = [
        (T("定律", "Laws"), T("什么必须成立", "what must hold"), ACCENT,
         [r"$\nabla\cdot\mathbf{D}=\rho_f,\quad \nabla\cdot\mathbf{B}=0$",
          r"$\nabla\times\mathbf{E}=-\partial_t\mathbf{B},\quad \nabla\times\mathbf{H}=\mathbf{J}_f+\partial_t\mathbf{D}$"],
         T("第 3–6 章", "Ch. 3–6")),
        (T("材料", "Materials"), T("物质怎样响应", "how matter responds"), MATL,
         [r"$\mathbf{J}_c=\boldsymbol{\sigma}\mathbf{E},\quad \mathbf{D}=\boldsymbol{\varepsilon}\mathbf{E},\quad \mathbf{B}=\boldsymbol{\mu}\mathbf{H}$"
          + T("  或非线性 B–H", "  or nonlinear B–H"),
          T("可随位置、方向、频率与场强改变", "may vary with position, direction, frequency, field")],
         T("第 3、5、13 章", "Ch. 3, 5, 13")),
        (T("表示", "Representation"), T("用什么未知量来算", "choice of unknowns"), POT,
         [r"$\mathbf{B}=\nabla\times\mathbf{A},\quad \mathbf{E}=-\nabla\phi-\partial_t\mathbf{A}$",
          T("参考与规范的选择 · 有限元空间与网格", "reference and gauge · finite-element spaces, mesh")],
         T("第 9、16、17 章", "Ch. 9, 16, 17")),
        (T("观测量", "Observables"), T("最后读出什么", "what is read out"), AMBER,
         [T("沿神经的 $\\mathbf{E}$ · 磁通与端电压 · 成像区均匀度", "$\\mathbf{E}$ along a nerve · flux, terminal voltage · homogeneity"),
          T("损耗与 SAR · 力与力矩 · MRI 相位", "loss and SAR · force, torque · MRI phase")],
         T("第 18–22 章", "Ch. 18–22")),
    ]
    h, gap, top = 0.98, 0.14, 5.55
    xt, wt = 0.08, 2.05
    xb = xt + wt + 0.15
    xc = 8.95
    for i, (name, q, col, lines, chap) in enumerate(rows):
        y = top - (i + 1) * h - i * gap
        rbox(ax, xt, y, 10.4 - 2 * xt, h, fc=tint(col, 0.07), ec=tint(col, 0.5), lw=0.9, r=0.1)
        rbox(ax, xt, y, wt, h, fc=tint(col, 0.88), ec="none", r=0.1, z=2.2)
        label(ax, xt + wt / 2, y + h * 0.64, name, color="white", size=10.5, weight="bold")
        label(ax, xt + wt / 2, y + h * 0.28, q, color="white", size=8.0)
        for k, s in enumerate(lines):
            muted = k > 0 and i > 0
            label(ax, xb, y + h * (0.68 - 0.38 * k), s, size=8.5 if muted else 9.0,
                  color=MUTED if muted else INK, ha="left")
        label(ax, 10.1, y + h / 2, chap, color=col if col != MATL else INK, size=8.4, ha="right",
              bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.4", fc="white", ec=tint(col, 0.6), lw=0.8))
    # foundation
    yb = top - 4 * h - 3 * gap - 0.2 - 0.66
    rbox(ax, xt, yb, 10.4 - 2 * xt, 0.66, fc=tint(INK, 0.82), ec="none", r=0.1)
    label(ax, 5.2, yb + 0.33, T("共同的地基：几何 · 源（线圈电流、电极电流、永磁体）· 边界条件 · 初始条件",
                                "Common ground: geometry · sources (coils, electrodes, magnets) · boundary and initial conditions"),
          color="white", size=8.7)
    return fig


# ---------------------------------------------------------------------------
def _pict_gauss(ax, cx, cy, s):
    ax.add_patch(Circle((cx, cy), 0.62 * s, fc="none", ec=tint(ELEC, 0.6), lw=1.0, ls=(0, (3, 2)), zorder=2))
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        arrow(ax, (cx + 0.2 * s * np.cos(t), cy + 0.2 * s * np.sin(t)),
              (cx + 0.9 * s * np.cos(t), cy + 0.9 * s * np.sin(t)), color=ELEC, lw=1.2, ms=7)
    charge(ax, cx, cy, "+", r=0.15 * s, size=8.5)


def _pict_nodiv(ax, cx, cy, s):
    """Closed B lines and a dashed closed surface: each line that enters it also leaves it."""
    ox = cx - 0.18 * s
    for (aa, bb) in ((0.32, 0.24), (0.54, 0.42), (0.76, 0.6)):
        ax.add_patch(Ellipse((ox, cy), 2 * aa * s, 2 * bb * s, fc="none", ec=MAG, lw=1.2, zorder=3))
        for t in (np.pi / 2, 3 * np.pi / 2):          # counter-clockwise arrows
            p = (ox + aa * s * np.cos(t), cy + bb * s * np.sin(t))
            q = (ox + aa * s * np.cos(t + 0.12), cy + bb * s * np.sin(t + 0.12))
            arrow(ax, p, q, color=MAG, lw=1.2, ms=7, z=4)
    ax.add_patch(Circle((cx + 0.42 * s, cy + 0.04 * s), 0.38 * s, fc=tint(INK, 0.04), ec=tint(INK, 0.6), lw=1.0,
                        ls=(0, (3, 2)), zorder=2))


def _pict_faraday(ax, cx, cy, s):
    ax.add_patch(Circle((cx, cy), 0.8 * s, fc=tint(MAG, 0.08), ec="none", zorder=1))
    for dx, dy in ((-0.3, 0.25), (0.3, 0.25), (-0.3, -0.25), (0.3, -0.25), (0, 0)):
        out_of_page(ax, cx + dx * s, cy + dy * s, color=MAG, r=0.075 * s / 0.85, lw=0.9)
    ax.add_patch(Circle((cx, cy), 0.68 * s, fc="none", ec=ELEC, lw=1.4, zorder=3))
    for t in (np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6):   # clockwise arrows
        p = (cx + 0.68 * s * np.cos(t), cy + 0.68 * s * np.sin(t))
        q = (cx + 0.68 * s * np.cos(t - 0.15), cy + 0.68 * s * np.sin(t - 0.15))
        arrow(ax, p, q, color=ELEC, lw=1.4, ms=9, z=4)
    halo(label(ax, cx, cy - 0.9 * s, T("$\\mathbf{B}$ 增大", "$\\mathbf{B}$ growing"), color=MAG, size=7.8), 3)


def _pict_ampere(ax, cx, cy, s):
    out_of_page(ax, cx, cy, color=CURR, r=0.13 * s, lw=1.2)
    for rad in (0.42, 0.72):
        ax.add_patch(Circle((cx, cy), rad * s, fc="none", ec=MAG, lw=1.2, zorder=3))
        for t in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):   # counter-clockwise
            p = (cx + rad * s * np.cos(t), cy + rad * s * np.sin(t))
            q = (cx + rad * s * np.cos(t + 0.15), cy + rad * s * np.sin(t + 0.15))
            arrow(ax, p, q, color=MAG, lw=1.2, ms=8, z=4)


@figure("maxwell_map")
def maxwell_map():
    fig, ax = canvas(FIG_W, 3.55, xlim=(0, 10.4), ylim=(0, 6.153))
    tiles = [
        (0, 0, _pict_gauss, r"$\nabla\cdot\mathbf{D}=\rho_f$",
         T("电荷是电场通量的源头", "charges are flux sources"),
         T("真空：$\\nabla\\cdot(\\varepsilon_0\\mathbf{E})=\\rho$", "vacuum: $\\nabla\\cdot(\\varepsilon_0\\mathbf{E})=\\rho$"),
         T("第 3 章 → 第 5 章", "Ch. 3 → Ch. 5"), ELEC),
        (0, 1, _pict_nodiv, r"$\nabla\cdot\mathbf{B}=0$",
         T("磁通无源：进多少，出多少", "no flux source: in = out"),
         T("磁场线自成闭合回路", "B lines always close"),
         T("第 4 章", "Ch. 4"), MAG),
        (1, 0, _pict_faraday, r"$\nabla\times\mathbf{E}=-\partial_t\mathbf{B}$",
         T("变化的磁通产生环流电场", "changing B drives E loops"),
         T("TMS 与梯度线圈感应的根源", "root of TMS induction"),
         T("第 4 章", "Ch. 4"), ELEC),
        (1, 1, _pict_ampere, r"$\nabla\times\mathbf{H}=\mathbf{J}_f+\partial_t\mathbf{D}$",
         T("电流与位移电流产生环流磁场", "currents drive B loops"),
         T("真空：$\\nabla\\times(\\mathbf{B}/\\mu_0)=\\mathbf{J}+\\varepsilon_0\\partial_t\\mathbf{E}$",
           "vacuum: $\\nabla\\times(\\mathbf{B}/\\mu_0)=\\mathbf{J}+\\varepsilon_0\\partial_t\\mathbf{E}$"),
         T("第 4 章 → 第 5 章", "Ch. 4 → Ch. 5"), MAG),
    ]
    x0s, w = (0.62, 5.56), 4.78
    y0s, h = (3.42, 0.98), 2.3
    for r, cidx, pict, eq, l1, l2, chap, col in tiles:
        x, y = x0s[cidx], y0s[r]
        rbox(ax, x, y, w, h, fc=tint(col, 0.05), ec=tint(col, 0.45), lw=0.9, r=0.12)
        pict(ax, x + 1.0, y + h / 2, 0.95)
        tx = x + 2.05
        label(ax, tx, y + h - 0.42, eq, color=col, size=10.5, ha="left")
        label(ax, tx, y + h - 0.95, l1, size=8.4, ha="left")
        label(ax, tx, y + h - 1.37, l2, size=8.0, ha="left", color=MUTED)
        label(ax, tx, y + 0.33, chap, size=8.2, ha="left", color=col,
              bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.4", fc="white", ec=tint(col, 0.6), lw=0.8))
    # headers
    label(ax, x0s[0] + w / 2, 5.98, T("电场方程（$\\mathbf{E}$，物质中另有 $\\mathbf{D}$）", "Electric ($\\mathbf{E}$; $\\mathbf{D}$ in matter)"),
          color=ELEC, size=9.4, weight="bold")
    label(ax, x0s[1] + w / 2, 5.98, T("磁场方程（$\\mathbf{B}$，物质中另有 $\\mathbf{H}$）", "Magnetic ($\\mathbf{B}$; $\\mathbf{H}$ in matter)"),
          color=MAG, size=9.4, weight="bold")
    label(ax, 0.28, y0s[0] + h / 2, T("散度：通量从哪里来", "divergence: sources"), size=8.6, rotation=90, color=MUTED)
    label(ax, 0.28, y0s[1] + h / 2, T("旋度：环流从哪里来", "curl: circulation"), size=8.6, rotation=90, color=MUTED)
    # consequences strip
    label(ax, 5.2, 0.42, T("推论与配套：电荷守恒 $\\partial_t\\rho+\\nabla\\cdot\\mathbf{J}=0$（第 3、5 章）· 界面条件（第 6 章）· 材料关系（第 3、5、13 章）",
                           "Consequences and companions: charge conservation (Ch. 3, 5) · interface conditions (Ch. 6) · material laws (Ch. 3, 5, 13)"),
          size=8.3, color=INK)
    return fig
