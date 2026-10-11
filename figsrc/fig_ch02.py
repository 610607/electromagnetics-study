"""Figures for Chapter 2 (geometry of fields: gradient, divergence, curl).

gradient_contours  a computed scalar landscape: contours, gradient arrows, two paths whose
                   line integrals of grad(phi) agree (gradient theorem); local picture of
                   d(phi) = grad(phi).dr.
div_curl_four      source, vortex, uniform and saddle fields with a flux box (divergence probe)
                   and a circulation loop (curl probe), values computed from the fields.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, Circle, Polygon, Rectangle

from style import (ACCENT, AMBER, ELEC, FIG_W, INK, LIGHT, MUTED, POT, T, arrow, figure, halo, label,
                   panel_title, plt, seq_cmap, tint)


# ---------------------------------------------------------------------------
# A smooth teaching landscape phi(x, y) (dimensionless) and its exact gradient
# ---------------------------------------------------------------------------
HILL = (1.0, 0.9, 0.35, 0.9)      # amplitude, x0, y0, width^2
VALLEY = (-0.75, -1.0, -0.45, 0.7)
SLOPE = 0.12


def landscape(x, y):
    out = SLOPE * x
    for A, x0, y0, w in (HILL, VALLEY):
        out = out + A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / w)
    return out


def landscape_grad(x, y):
    gx = SLOPE + 0 * x
    gy = 0 * y
    for A, x0, y0, w in (HILL, VALLEY):
        e = A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / w)
        gx = gx + e * (-2 * (x - x0) / w)
        gy = gy + e * (-2 * (y - y0) / w)
    return gx, gy


PA = np.array([-1.75, 0.95])
PB = np.array([1.35, -0.95])


def path_straight(s):
    return PA[None, :] + (PB - PA)[None, :] * s[:, None]


def path_curved(s):
    """Quadratic Bezier from A to B that detours over the hill."""
    P1 = np.array([0.6, 2.2])
    s = s[:, None]
    return (1 - s) ** 2 * PA + 2 * (1 - s) * s * P1 + s ** 2 * PB


def line_integral_of_grad(path, n=20001):
    s = np.linspace(0, 1, n)
    p = path(s)
    gx, gy = landscape_grad(p[:, 0], p[:, 1])
    dp = np.gradient(p, s, axis=0)
    return float(np.trapezoid(gx * dp[:, 0] + gy * dp[:, 1], s))


# ---------------------------------------------------------------------------
@figure("gradient_contours")
def gradient_contours():
    fig = plt.figure(figsize=(FIG_W, 3.05))
    ax = fig.add_axes([0.0, 0.02, 0.56, 0.9])
    g = np.linspace(-2.4, 2.4, 481)
    X, Y = np.meshgrid(g, g)
    P = landscape(X, Y)
    lev = np.linspace(-0.8, 1.1, 20)
    ax.contourf(X, Y, P, levels=60, cmap=seq_cmap(POT), alpha=0.55, zorder=0)
    ax.contour(X, Y, P, levels=lev, colors=[tint(POT, 0.75)], linewidths=0.7, zorder=1, negative_linestyles="solid")
    # gradient arrows (direction only) on a coarse grid
    q = np.linspace(-2.1, 2.1, 9)
    QX, QY = np.meshgrid(q, q)
    gx, gy = landscape_grad(QX, QY)
    n = np.hypot(gx, gy)
    keep = n > 0.08
    ax.quiver(QX[keep], QY[keep], (gx / n)[keep], (gy / n)[keep], color=INK, angles="xy", scale_units="xy",
              scale=3.2, width=0.0055, headwidth=3.6, headlength=4.0, headaxislength=3.6, zorder=3, alpha=0.7)
    # two paths
    s = np.linspace(0, 1, 400)
    p1, p2 = path_straight(s), path_curved(s)
    ax.plot(p1[:, 0], p1[:, 1], color=ACCENT, lw=2.0, zorder=4)
    ax.plot(p2[:, 0], p2[:, 1], color=ACCENT, lw=2.0, ls=(0, (4, 2.2)), zorder=4)
    for pp, frac in ((p1, 0.55), (p2, 0.5)):
        k = int(frac * len(pp))
        arrow(ax, pp[k - 2], pp[k + 2], color=ACCENT, lw=2.0, ms=11, z=5)
    for P0, name, dx in ((PA, "a", -0.18), (PB, "b", 0.2)):
        ax.plot(*P0, "o", ms=6, color=ACCENT, mec="white", mew=1.2, zorder=6)
        halo(label(ax, P0[0] + dx, P0[1] + 0.18, f"${name}$", size=11, color=ACCENT, z=7), 3)
    I1 = line_integral_of_grad(path_straight)
    I2 = line_integral_of_grad(path_curved)
    dphi = float(landscape(*PB) - landscape(*PA))
    halo(label(ax, 1.6, 0.6, T("山", "hill"), size=9, color=POT, weight="bold"), 3)
    halo(label(ax, -1.05, -0.55, T("谷", "valley"), size=9, color=POT, weight="bold"), 3)
    txt = T(f"实线：$\\int\\nabla\\phi\\cdot d\\boldsymbol{{\\ell}}={I1:.3f}$\n虚线：$\\int\\nabla\\phi\\cdot d\\boldsymbol{{\\ell}}={I2:.3f}$\n$\\phi(b)-\\phi(a)={dphi:.3f}$",
            f"solid: $\\int\\nabla\\phi\\cdot d\\boldsymbol{{\\ell}}={I1:.3f}$\ndashed: $\\int\\nabla\\phi\\cdot d\\boldsymbol{{\\ell}}={I2:.3f}$\n$\\phi(b)-\\phi(a)={dphi:.3f}$")
    label(ax, -2.3, -2.25, txt, size=8.3, ha="left", va="bottom", color=INK,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=tint(ACCENT, 0.4), lw=0.7))
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, T("(a) 等高线与梯度：箭头垂直于等高线、指向上坡", "(a) Contours and gradient (uphill arrows)"), size=9.2)

    # ---------------- (b) local picture -------------------------------------
    bx = fig.add_axes([0.6, 0.02, 0.4, 0.9])
    bx.set_xlim(-1.6, 1.6)
    bx.set_ylim(-1.75, 1.85)
    bx.set_aspect("equal")
    bx.axis("off")
    ang = np.deg2rad(25)                 # direction of grad(phi)
    nvec = np.array([np.cos(ang), np.sin(ang)])
    tvec = np.array([-np.sin(ang), np.cos(ang)])
    for k, lab in zip(range(-2, 3), [r"$\phi_0-2\Delta\phi$", r"$\phi_0-\Delta\phi$", r"$\phi_0$", r"$\phi_0+\Delta\phi$", r"$\phi_0+2\Delta\phi$"]):
        c = k * 0.55 * nvec
        a0, a1 = c - 1.45 * tvec, c + 1.45 * tvec
        bx.plot([a0[0], a1[0]], [a0[1], a1[1]], color=POT, lw=1.0 if k else 1.6, zorder=1)
        if k in (-1, 1):
            t = label(bx, a1[0] + 0.02, a1[1] + 0.12, lab, size=7.8, color=POT, ha="center")
    O = np.zeros(2)
    bx.plot(0, 0, "o", ms=5, color=INK, zorder=6)
    arrow(bx, O, 1.15 * nvec, color=INK, lw=1.9, ms=12, z=5)
    halo(label(bx, 1.25 * nvec[0] + 0.05, 1.25 * nvec[1] - 0.12, r"$\nabla\phi$", size=10.5, ha="left"), 3)
    halo(label(bx, 1.25 * nvec[0] + 0.05, 1.25 * nvec[1] - 0.42, T("上坡最陡", "steepest ascent"), size=8.0, ha="left"), 3)
    arrow(bx, O, -1.15 * nvec, color=ELEC, lw=1.9, ms=12, z=5)
    halo(label(bx, -1.15 * nvec[0] + 0.05, -1.15 * nvec[1] - 0.25, r"$-\nabla\phi$", size=10.5, color=ELEC, ha="center"), 3)
    halo(label(bx, -1.15 * nvec[0] + 0.05, -1.15 * nvec[1] - 0.55, T("下坡最陡", "steepest descent"), size=8.0, color=ELEC, ha="center"), 3)
    arrow(bx, O, 0.95 * tvec, color=ACCENT, lw=1.6, ms=10, z=5)
    halo(label(bx, 0.95 * tvec[0] - 0.12, 0.95 * tvec[1] + 0.02, T("沿等高线：\n$d\\phi=0$", "along a contour:\n$d\\phi=0$"), size=8.3, color=ACCENT, ha="right"), 3)
    th = np.deg2rad(60)
    dr = 0.95 * np.array([np.cos(ang + th), np.sin(ang + th)])
    arrow(bx, O, dr, color=ACCENT, lw=1.6, ms=10, ls=(0, (3, 1.6)), z=5)
    halo(label(bx, dr[0] + 0.12, dr[1] + 0.02, r"$d\mathbf{r}$", size=10, color=ACCENT, ha="left"), 3)
    bx.add_patch(Arc((0, 0), 0.6, 0.6, theta1=np.rad2deg(ang), theta2=np.rad2deg(ang + th), color=INK, lw=0.9, zorder=4))
    halo(label(bx, 0.4 * np.cos(ang + th / 2), 0.4 * np.sin(ang + th / 2), r"$\theta$", size=9), 3)
    label(bx, 0.0, -1.55, r"$d\phi=\nabla\phi\cdot d\mathbf{r}=|\nabla\phi|\,|d\mathbf{r}|\cos\theta$", size=9.2,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=tint(INK, 0.3), lw=0.7))
    panel_title(bx, T("(b) 一点附近：放大后等高线近似平行", "(b) Zoom: contours look parallel"), size=9.2)
    return fig


# ---------------------------------------------------------------------------
FIELDS = {
    "source": (lambda x, y: (x, y), T("源", "Source"), r"$\mathbf{F}=(x,\,y,\,0)$"),
    "vortex": (lambda x, y: (-y, x), T("涡旋", "Vortex"), r"$\mathbf{F}=(-y,\,x,\,0)$"),
    "uniform": (lambda x, y: (1.0 + 0 * x, 0 * y), T("均匀", "Uniform"), r"$\mathbf{F}=(1,\,0,\,0)$"),
    "saddle": (lambda x, y: (x, -y), T("鞍点", "Saddle"), r"$\mathbf{F}=(x,\,-y,\,0)$"),
}


def div_curl_numeric(F, h=1e-4, x=0.37, y=-0.21):
    """Central-difference divergence and z-curl of a 2-D field at a point."""
    fx_p, fy_p = F(x + h, y)
    fx_m, fy_m = F(x - h, y)
    gx_p, gy_p = F(x, y + h)
    gx_m, gy_m = F(x, y - h)
    div = (fx_p - fx_m) / (2 * h) + (gy_p - gy_m) / (2 * h)
    curl = (fy_p - fy_m) / (2 * h) - (gx_p - gx_m) / (2 * h)
    return float(div), float(curl)


@figure("div_curl_four")
def div_curl_four():
    fig = plt.figure(figsize=(FIG_W, 2.3))
    names = ["source", "vortex", "uniform", "saddle"]
    q = np.linspace(-1.75, 1.75, 8)
    QX, QY = np.meshgrid(q, q)
    hb, rl = 0.42, 1.18          # half-width of the flux box, radius of the loop
    for i, key in enumerate(names):
        F, name, formula = FIELDS[key]
        ax = fig.add_axes([0.005 + i * 0.25, 0.215, 0.24, 0.68])
        U, V = F(QX, QY)
        U = np.broadcast_to(U, QX.shape)
        V = np.broadcast_to(V, QX.shape)
        ax.quiver(QX, QY, U, V, color=tint(MUTED, 0.6), angles="xy", scale_units="xy", scale=5.2,
                  width=0.008, headwidth=3.4, headlength=3.8, headaxislength=3.4, zorder=1)
        # flux box (divergence probe)
        ax.add_patch(Rectangle((-hb, -hb), 2 * hb, 2 * hb, fc=tint(AMBER, 0.12), ec=AMBER, lw=1.3, zorder=3))
        for (mx, my), (nx, ny) in (((hb, 0), (1, 0)), ((-hb, 0), (-1, 0)), ((0, hb), (0, 1)), ((0, -hb), (0, -1))):
            fx, fy = F(np.array(mx, float), np.array(my, float))
            fn = float(fx * nx + fy * ny)
            if abs(fn) < 1e-9:
                continue
            L = min(0.95 * abs(fn), 0.55)
            if fn > 0:   # outflow: arrow leaves the face
                p, qq = (mx, my), (mx + nx * L, my + ny * L)
            else:        # inflow: arrow arrives at the face
                p, qq = (mx + nx * L, my + ny * L), (mx, my)
            arrow(ax, p, qq, color=AMBER, lw=1.6, ms=8, z=4)
        # circulation loop (curl probe)
        ax.add_patch(Circle((0, 0), rl, fc="none", ec=ACCENT, lw=1.3, zorder=3))
        for t in np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8:
            px, py = rl * np.cos(t), rl * np.sin(t)
            fx, fy = F(np.array(px), np.array(py))
            tx, ty = -np.sin(t), np.cos(t)
            ft = float(fx * tx + fy * ty)
            if abs(ft) < 1e-9:
                continue
            L = 0.34 * np.sign(ft) * min(abs(ft), 1.3)
            ro = 1.0 + 0.15 / rl                      # draw just outside the ring
            arrow(ax, (ro * px - tx * L / 2, ro * py - ty * L / 2), (ro * px + tx * L / 2, ro * py + ty * L / 2),
                  color=ACCENT, lw=1.6, ms=8.5, z=5)
        ax.set_xlim(-1.95, 1.95)
        ax.set_ylim(-1.95, 1.95)
        ax.set_aspect("equal")
        ax.axis("off")
        panel_title(ax, f"{name}  {formula}", size=9.0)
        d, c = div_curl_numeric(F)
        d, c = round(d, 6) + 0.0, round(c, 6) + 0.0
        fig.text(0.005 + i * 0.25 + 0.12, 0.135, f"$\\nabla\\cdot\\mathbf{{F}}={d:g}$", color=AMBER, ha="center", fontsize=9.2)
        fig.text(0.005 + i * 0.25 + 0.12, 0.035, f"$(\\nabla\\times\\mathbf{{F}})_z={c:g}$", color=ACCENT, ha="center", fontsize=9.2)
    return fig
