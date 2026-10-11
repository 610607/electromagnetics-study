"""Shaded 3-D figures rendered with three.js (see render3d.py); labels by matplotlib."""
from __future__ import annotations

import numpy as np
import contourpy
from scipy.special import ellipe, ellipk

from render3d import Scene, compose, put, put_leader, render, trim
from style import (ACCENT, CURR, ELEC, FIG_W, INK, MAG, MATL, MUTED, POT, T, figure, tint)

COPPER = "#B87333"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def loop_flux(rho, z, a):
    """Stokes flux function rho*A_phi of a circular loop of radius a in z=0 (shape only).
    A_phi = (mu0 I/(pi k)) sqrt(a/rho) [(1-k^2/2)K(k^2) - E(k^2)], k^2 = 4 a rho/((a+rho)^2+z^2)."""
    rho = np.abs(rho) + 1e-9
    k2 = np.clip(4 * a * rho / ((a + rho) ** 2 + z**2), 0, 1 - 1e-12)
    K, E = ellipk(k2), ellipe(k2)
    return rho * np.sqrt(a / rho) / np.sqrt(k2) * ((1 - k2 / 2) * K - E)


def loop_B(rho, z, a):
    """(B_rho, B_z) of a loop (same normalisation as loop_flux)."""
    rho = np.abs(rho) + 1e-9
    r2 = rho**2 + z**2
    al2 = a**2 + r2 - 2 * a * rho
    be2 = a**2 + r2 + 2 * a * rho
    be = np.sqrt(be2)
    m = 1 - al2 / be2
    K, E = ellipk(m), ellipe(m)
    Br = z / (2 * al2 * be * rho) * ((a**2 + r2) * E - al2 * K)
    Bz = 1 / (2 * al2 * be) * ((a**2 - r2) * E + al2 * K)
    return Br, Bz


def meridian_lines(psi_fn, B_fn, levels, rmax, zmax, n=500, min_len=0.3):
    """Field lines in the (rho, z) half-plane as contours of the flux function, oriented along B."""
    r = np.linspace(1e-3, rmax, n)
    z = np.linspace(-zmax, zmax, 2 * n)
    R, Z = np.meshgrid(r, z)
    P = psi_fn(R, Z)
    gen = contourpy.contour_generator(R, Z, P)
    out = []
    for lev in levels:
        for seg in gen.lines(lev):
            if len(seg) < 4:
                continue
            L = np.sum(np.hypot(*np.diff(seg, axis=0).T))
            if L < min_len:
                continue
            k = len(seg) // 2
            t = seg[k + 1] - seg[k - 1]
            br, bz = B_fn(seg[k, 0], seg[k, 1])
            if t[0] * br + t[1] * bz < 0:
                seg = seg[::-1]
            out.append(seg)
    return out


def revolve(seg, phi):
    return np.column_stack([seg[:, 0] * np.cos(phi), seg[:, 0] * np.sin(phi), seg[:, 1]])


def helix(a, z0, z1, turns, n=1600):
    t = np.linspace(0, 1, n)
    th = 2 * np.pi * turns * t
    return np.column_stack([a * np.cos(th), a * np.sin(th), z0 + (z1 - z0) * t])


def circle(r, z, n=200, start=0.0):
    th = start + np.linspace(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, z)])


# ---------------------------------------------------------------------------
@figure("solenoid_3d")
def solenoid_3d():
    a, L, N = 0.5, 2.0, 9
    zs = np.linspace(-L / 2, L / 2, 81)   # dense loops: smooth sheet-current field
    psi = lambda R, Z: sum(loop_flux(R, Z - z0, a) for z0 in zs)

    def Bf(r, z):
        br = sum(loop_B(r, z - z0, a)[0] for z0 in zs)
        bz = sum(loop_B(r, z - z0, a)[1] for z0 in zs)
        return br, bz

    ref = float(psi(np.array(0.45), np.array(0.0)))
    levels = ref * np.array([0.2, 0.55, 0.85])
    lines = meridian_lines(psi, Bf, levels, rmax=3.2, zmax=3.4)

    s = Scene(900, 600)
    s.camera(position=(5.6, -7.4, 3.0), target=(0, 0, -0.05), fov=27)
    # coil (current flows counter-clockwise seen from +z -> B up inside)
    s.tube(helix(a, -L / 2, L / 2, N - 1 + 0.75), radius=0.032, color=CURR, metalness=0.25, roughness=0.4)
    for phi in np.deg2rad([0, 90, 180, 270]):
        for seg in lines:
            p = revolve(seg, phi)
            keep = np.linalg.norm(p, axis=1) < 2.5
            # split into contiguous visible runs
            idx = np.flatnonzero(keep)
            if len(idx) < 4:
                continue
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for rr in runs:
                if len(rr) < 6:
                    continue
                q = p[rr]
                s.tube(q, radius=0.014, color=MAG, arrows=[0.5], head_len=0.13, head_radius=0.045)
    # induced electric field loops (dI/dt > 0: E circulates opposite to the coil current)
    for r, z in ((1.75, 0.0),):
        c = circle(r, z, n=240, start=0.3)[::-1][:-1]
        s.tube(c, radius=0.024, color=ELEC, closed=True, arrows=[0.1, 0.35, 0.6, 0.85], head_len=0.17, head_radius=0.065)
    s.arrow((0, 0, -0.25), (0, 0, 0.35), radius=0.03, color=INK, head_len=0.16, head_radius=0.07)
    s.label("coil", (a * np.cos(-0.6), a * np.sin(-0.6), L / 2 + 0.05))
    s.label("B", (0.0, 0.0, 1.55))
    s.label("Bout", (2.15, -0.3, 1.2))
    s.label("E", (1.75 * np.cos(-1.9), 1.75 * np.sin(-1.9), 0.0))
    s.label("Ein", (1.35 * np.cos(-0.35), 1.35 * np.sin(-0.35), 0.0))
    img, px = render(s)
    img, px = trim(img, px)
    fig, ax = compose(img, px, FIG_W * 0.86)
    put(ax, px, "coil", T(r"绕组电流 $I(t)$", r"winding current $I(t)$"), dx=150, dy=-40, color=CURR, size=9.5)
    put_leader(ax, px, "B", T(r"$\mathbf{B}$：管内向上，绕到管外闭合", r"$\mathbf{B}$: up inside, closing outside"), dx=230, dy=40, color=MAG, size=9.5)
    put(ax, px, "E", T(r"感应 $\mathbf{E}$ 环（$\mathrm{d}I/\mathrm{d}t>0$）", r"induced $\mathbf{E}$ loop ($\mathrm{d}I/\mathrm{d}t>0$)"), dx=-40, dy=60, color=ELEC, size=9.5)
    return fig


# ---------------------------------------------------------------------------
# Coils made of circular loops (axis along z): exact A and B from elliptic integrals
# ---------------------------------------------------------------------------
class Loops:
    """Set of horizontal circular loops: centres (x, y, z), radius a, current sign s."""

    def __init__(self, loops):
        self.loops = [(np.asarray(c, float), a, s) for c, a, s in loops]

    def A(self, P):
        P = np.atleast_2d(P)
        out = np.zeros_like(P)
        for c, a, s in self.loops:
            d = P - c
            rho = np.hypot(d[:, 0], d[:, 1]) + 1e-9
            Aphi = s * loop_flux(rho, d[:, 2], a) / rho
            out[:, 0] += -Aphi * d[:, 1] / rho
            out[:, 1] += Aphi * d[:, 0] / rho
        return out

    def B(self, P):
        P = np.atleast_2d(P)
        out = np.zeros_like(P)
        for c, a, s in self.loops:
            d = P - c
            rho = np.hypot(d[:, 0], d[:, 1]) + 1e-9
            Br, Bz = loop_B(rho, d[:, 2], a)
            out[:, 0] += s * Br * d[:, 0] / rho
            out[:, 1] += s * Br * d[:, 1] / rho
            out[:, 2] += s * Bz
        return out


def trace(field, seed, step=0.01, nmax=2000, bounds=3.0, both=True, stop=None):
    """RK4 field-line tracing of a direction field (unit steps)."""
    def f(p):
        v = field(p[None, :])[0]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    paths = []
    for sgn in ((1, -1) if both else (1,)):
        p = np.asarray(seed, float)
        pts = [p.copy()]
        for _ in range(nmax):
            k1 = f(p); k2 = f(p + 0.5 * step * sgn * k1); k3 = f(p + 0.5 * step * sgn * k2); k4 = f(p + step * sgn * k3)
            p = p + step * sgn * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            pts.append(p.copy())
            if np.any(np.abs(p) > bounds) or (stop is not None and stop(p)):
                break
            if len(pts) > 50 and np.linalg.norm(p - seed) < 0.6 * step:
                break  # closed
        paths.append(np.array(pts))
    if both:
        return np.vstack([paths[1][::-1], paths[0][1:]])
    return paths[0]


def fig8(a=0.45, gap=0.04, h=0.0):
    x = a + gap / 2
    return Loops([((-x, 0, h), a, +1.0), ((x, 0, h), a, -1.0)])


@figure("coil_block_3d")
def coil_block_3d():
    h = 0.42                      # coil height above the conductor's top face
    coil = fig8(a=0.45, h=h)
    s = Scene(900, 620)
    s.camera(position=(4.6, -6.2, 4.2), target=(0, 0, -0.25), fov=26)
    s.lights(ambient=0.6, key=0.6, fill=0.25, key_dir=(2, -5, 7))
    # conductor block (top face at z = 0)
    bx, by, bz = 3.4, 2.4, 0.9
    s.box((0, 0, -bz / 2), (bx, by, bz), color="#F1E6D0", opacity=0.9, edges=True, edge_color="#B9A88A", roughness=1.0)
    # |E_p| on the top face as a light colour map, and arrows
    nx, ny = 69, 49
    X, Y = np.meshgrid(np.linspace(-bx / 2, bx / 2, nx), np.linspace(-by / 2, by / 2, ny))
    P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 0.002)])
    Ep = -coil.A(P)                           # E_p = -dA/dt for dI/dt > 0 (shape only)
    mag = np.linalg.norm(Ep, axis=1).reshape(X.shape)
    from style import seq_cmap
    cm = seq_cmap(ELEC)
    C = cm(np.clip(mag / mag.max(), 0, 1) * 0.85)
    s.surface(X, Y, np.full_like(X, 0.004), colors=C, unlit=True, opacity=0.95)
    xs, ys = np.meshgrid(np.linspace(-1.35, 1.35, 9), np.linspace(-0.9, 0.9, 7))
    Q = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.02)])
    V = -coil.A(Q)
    vmax = np.linalg.norm(V, axis=1).max()
    for q, v in zip(Q, V):
        n = np.linalg.norm(v)
        if n < 0.22 * vmax:
            continue
        d = v / vmax * 0.42
        s.arrow(q - d / 2, q + d / 2, radius=0.012, color=ELEC, head_len=0.08, head_radius=0.032)
    # coil windings (two loops, opposite currents) drawn as tubes with current arrows
    for c, a, sg in coil.loops:
        ring = circle(a, c[2], n=200) + np.array([c[0], c[1], 0])
        if sg < 0:
            ring = ring[::-1]
        s.tube(ring[:-1], radius=0.03, color=CURR, closed=True, arrows=[0.15, 0.65], head_len=0.12, head_radius=0.05)
    # B field lines from seeds inside each loop
    for c, a, sg in coil.loops:
        for dx, dy in ((0.0, 0.0), (0.22, 0.0), (-0.22, 0.0), (0.0, 0.25), (0.0, -0.25)):
            seed = c + np.array([dx, dy, 0.0])
            path = trace(coil.B, seed, step=0.012, nmax=900, bounds=2.6,
                         stop=lambda p: np.linalg.norm(p - np.array([0, 0, h])) > 1.75)
            s.tube(path, radius=0.011, color=MAG, opacity=0.95, arrows=[0.5], head_len=0.1, head_radius=0.035)
    # labels
    s.label("coil", (-0.95, -0.35, h + 0.02))
    s.label("B", (0.0, 0.0, h + 0.9))
    s.label("E", (0.0, -0.55, 0.0))
    s.label("plus", (0.0, -by / 2, -0.3))
    s.label("minus", (0.0, by / 2, -0.3))
    s.label("block", (bx / 2, -by / 2 + 0.2, -bz))
    img, px = render(s)
    img, px = trim(img, px)
    fig, ax = compose(img, px, FIG_W * 0.9)
    put(ax, px, "coil", T("① 线圈电流 $I(t)$", "① coil current $I(t)$"), dx=-70, dy=-50, color=CURR, size=9.5)
    put(ax, px, "B", T(r"② 磁场 $\mathbf{B}$", r"② magnetic field $\mathbf{B}$"), dy=-30, color=MAG, size=9.5)
    put_leader(ax, px, "E", T(r"③ 感应电场 $\mathbf{E}_p=-\partial_t\mathbf{A}$", r"③ induced field $\mathbf{E}_p=-\partial_t\mathbf{A}$"), dx=260, dy=170, color=ELEC, size=9.5)
    for k in (-110, -40, 30, 100):
        put(ax, px, "plus", "+", dx=k, dy=5, color=MAG, size=12, weight="bold")
    put(ax, px, "plus", T("④ 边界上积累电荷 → 次级电场", "④ charge builds up at the boundary → secondary field"), dx=0, dy=75, color=INK, size=9.3)
    put(ax, px, "block", T("导体（组织）", "conductor (tissue)"), dx=-60, dy=-25, color="#7A6A50", size=9)
    return fig


# ---------------------------------------------------------------------------
@figure("div_curl_3d")
def div_curl_3d():
    from render3d import compose_row
    imgs, pxs = [], []
    # ---- source field F = r (divergence) with a small cube
    s = Scene(620, 560)
    s.camera(position=(4.2, -5.6, 3.6), target=(0, 0, 0), fov=28)
    g = np.linspace(-1.1, 1.1, 4)
    for x in g:
        for y in g:
            for z in g:
                p = np.array([x, y, z])
                if np.linalg.norm(p) < 0.5:
                    continue
                d = p / np.linalg.norm(p) * 0.28 * min(np.linalg.norm(p), 1.3)
                s.arrow(p - d / 2, p + d / 2, radius=0.016, color=tint(ACCENT, 0.55), head_len=0.09, head_radius=0.04)
    c = 0.32
    s.box((0, 0, 0), (2 * c, 2 * c, 2 * c), color=tint(ACCENT, 0.18), opacity=0.55, edges=True, edge_color=ACCENT)
    for n in np.vstack([np.eye(3), -np.eye(3)]):
        p0 = n * c
        s.arrow(p0, p0 + n * 0.42, radius=0.025, color=ACCENT, head_len=0.12, head_radius=0.055)
    s.label("box", (c, -c, c))
    img, px = trim(*render(s))
    imgs.append(img); pxs.append(px)
    # ---- vortex field F = (-y, x, 0) with a small loop
    s = Scene(620, 560)
    s.camera(position=(4.2, -5.6, 3.6), target=(0, 0, 0), fov=28)
    th_d = np.linspace(0, 2 * np.pi, 120)
    RR, TH = np.meshgrid(np.linspace(0, 1.45, 8), th_d)
    s.surface(RR * np.cos(TH), RR * np.sin(TH), np.full_like(RR, -0.003), color=tint(ACCENT, 0.08), opacity=0.6)
    for r, n_ in ((0.75, 10), (1.2, 14)):
        for th in np.linspace(0, 2 * np.pi, n_, endpoint=False) + 0.2:
            p = np.array([r * np.cos(th), r * np.sin(th), 0.0])
            d = np.array([-np.sin(th), np.cos(th), 0]) * 0.26 * r
            s.arrow(p - d / 2, p + d / 2, radius=0.018, color=tint(ACCENT, 0.55), head_len=0.1, head_radius=0.045)
    sq = 0.32
    loop = np.array([[sq, -sq, 0], [sq, sq, 0], [-sq, sq, 0], [-sq, -sq, 0]])
    pts = np.vstack([np.linspace(loop[i], loop[(i + 1) % 4], 30)[:-1] for i in range(4)])
    s.tube(pts, radius=0.03, color=ACCENT, closed=True, arrows=[0.125, 0.375, 0.625, 0.875], head_len=0.12, head_radius=0.055)
    s.arrow((0, 0, 0), (0, 0, 0.75), radius=0.022, color=INK, head_len=0.12, head_radius=0.05)
    s.label("n", (0, 0, 0.8))
    s.label("loop", (sq, -sq, 0))
    img, px = trim(*render(s))
    imgs.append(img); pxs.append(px)
    fig, axes = compose_row(imgs, FIG_W, titles=[T(r"散度：小盒子的净流出", r"Divergence: net outflow of a small box"),
                                                  T(r"旋度：小回路的环流", r"Curl: circulation round a small loop")],
                            title_colors=[ACCENT, ACCENT])
    fig.text(0.25, -0.02, r"$\oint\mathbf{F}\cdot\mathbf{n}\,\mathrm{d}S>0\;\Rightarrow\;\nabla\cdot\mathbf{F}>0$", ha="center", va="top", fontsize=9.6)
    fig.text(0.75, -0.02, r"$\oint\mathbf{F}\cdot\mathrm{d}\boldsymbol{\ell}>0\;\Rightarrow\;(\nabla\times\mathbf{F})\cdot\mathbf{n}>0$", ha="center", va="top", fontsize=9.6)
    put(axes[1], pxs[1], "n", r"$\mathbf{n}$", dx=18, dy=0, size=10.5)
    return fig


# ---------------------------------------------------------------------------
@figure("interface_probes_3d")
def interface_probes_3d():
    s = Scene(980, 520)
    s.camera(position=(3.2, -7.2, 2.6), target=(0.0, 0, 0.0), fov=26)
    s.lights(ambient=0.62, key=0.55)
    W, D, Hh = 5.2, 2.2, 0.9
    # materials suggested by opaque back wall and floor, plus a translucent interface plane
    s.box((0, D / 2 + 0.01, -Hh / 2), (W, 0.02, Hh), color="#EDE0C6")
    s.box((0, D / 2 + 0.01, Hh / 2), (W, 0.02, Hh), color=tint(ELEC, 0.12))
    s.box((0, 0, -Hh - 0.01), (W, D, 0.02), color="#F3EADB")
    X, Y = np.meshgrid(np.linspace(-W / 2, W / 2, 2), np.linspace(-D / 2, D / 2, 2))
    s.surface(X, Y, np.zeros_like(X), color=tint(MATL, 0.45), opacity=0.35)
    s.polyline([[-W / 2, -D / 2, 0], [W / 2, -D / 2, 0], [W / 2, D / 2, 0], [-W / 2, D / 2, 0], [-W / 2, -D / 2, 0]], color=MATL)
    # ---- pillbox on the left
    cx = -1.35
    s.cylinder((cx, 0, -0.12), (cx, 0, 0.12), 0.42, color=tint(ACCENT, 0.3), opacity=0.75)
    s.arrow((cx, 0, 0.12), (cx, 0, 0.75), radius=0.03, color=ELEC, head_len=0.13, head_radius=0.06)
    s.arrow((cx, 0, -0.75), (cx, 0, -0.12), radius=0.03, color=ELEC, head_len=0.13, head_radius=0.06)
    s.arrow((cx + 0.62, 0, 0.0), (cx + 0.62, 0, 0.45), radius=0.018, color=INK, head_len=0.1, head_radius=0.045)
    s.label("pill_top", (cx, 0, 0.75))
    s.label("pill_bot", (cx, 0, -0.75))
    s.label("n", (cx + 0.62, 0, 0.47))
    s.label("pill", (cx - 0.42, 0, 0.0))
    # ---- thin rectangular loop on the right (in the x-z plane)
    lx, hw, hh = 1.35, 0.55, 0.12
    rect = np.array([[lx - hw, 0, -hh], [lx + hw, 0, -hh], [lx + hw, 0, hh], [lx - hw, 0, hh]])
    pts = np.vstack([np.linspace(rect[i], rect[(i + 1) % 4], 40)[:-1] for i in range(4)])
    s.tube(pts, radius=0.025, color=ACCENT, closed=True, arrows=[0.13, 0.63], head_len=0.11, head_radius=0.05)
    s.arrow((lx - 0.5, 0, 0.45), (lx + 0.5, 0, 0.45), radius=0.03, color=ELEC, head_len=0.13, head_radius=0.06)
    s.arrow((lx - 0.5, 0, -0.45), (lx + 0.5, 0, -0.45), radius=0.03, color=ELEC, head_len=0.13, head_radius=0.06)
    s.label("loop_top", (lx + 0.5, 0, 0.45))
    s.label("loop_bot", (lx + 0.5, 0, -0.45))
    s.label("loop", (lx + hw, 0, 0.0))
    s.label("m2", (W / 2 - 0.3, D / 2, 0.75))
    s.label("m1", (W / 2 - 0.3, -D / 2, -Hh))
    img, px = trim(*render(s))
    fig, ax = compose(img, px, FIG_W)
    put(ax, px, "pill", T("薄盒（高度→0）", "pillbox (height → 0)"), dx=-15, dy=-5, size=9.2, ha="right", color=ACCENT)
    put(ax, px, "pill_top", r"$\mathbf{D}_2\cdot\mathbf{n}$", dx=40, dy=0, color=ELEC, size=10, ha="left")
    put(ax, px, "pill_bot", r"$\mathbf{D}_1\cdot\mathbf{n}$", dx=40, dy=0, color=ELEC, size=10, ha="left")
    put(ax, px, "n", r"$\mathbf{n}$", dx=16, dy=0, size=10.5)
    put(ax, px, "loop", T("窄回路（宽度→0）", "thin loop (width → 0)"), dx=25, dy=0, size=9.2, ha="left", color=ACCENT)
    put(ax, px, "loop_top", r"$\mathbf{E}_{2t}$", dx=30, dy=0, color=ELEC, size=10, ha="left")
    put(ax, px, "loop_bot", r"$\mathbf{E}_{1t}$", dx=30, dy=0, color=ELEC, size=10, ha="left")
    put(ax, px, "m2", T("材料 2", "material 2"), dx=-10, dy=0, size=9.2, ha="right", color=MUTED)
    put(ax, px, "m1", T("材料 1", "material 1"), dx=-30, dy=-25, size=9.2, ha="right", color=MUTED)
    return fig


# ---------------------------------------------------------------------------
@figure("a_circulation_3d")
def a_circulation_3d():
    s = Scene(900, 560)
    s.camera(position=(4.4, -6.0, 3.4), target=(0, 0, 0.1), fov=26)
    for x in np.linspace(-1.5, 1.5, 5):
        for y in np.linspace(-1.5, 1.5, 5):
            if np.hypot(x, y) > 1.9:
                continue
            s.arrow((x, y, -0.8), (x, y, 1.0), radius=0.014, color=tint(MAG, 0.75), head_len=0.12, head_radius=0.04)
    R = 1.05
    th = np.linspace(0, 2 * np.pi, 120)
    disk_r = np.linspace(0, R, 12)
    TH, RR = np.meshgrid(th, disk_r)
    s.surface(RR * np.cos(TH), RR * np.sin(TH), np.zeros_like(TH), color=tint(MAG, 0.35), opacity=0.55)
    for r, w in ((0.5, 0.018), (R, 0.032), (1.6, 0.018)):
        c = circle(r, 0.0, n=200)[:-1]
        s.tube(c, radius=w, color=POT, closed=True, arrows=[0.0, 0.25, 0.5, 0.75], head_len=0.11 + 0.04 * (w > 0.02), head_radius=0.045 + 0.02 * (w > 0.02))
    s.label("B", (-1.5, 0.0, 1.0))
    s.label("A", (R * np.cos(-1.2), R * np.sin(-1.2), 0))
    s.label("disk", (0.2, -0.3, 0.0))
    img, px = trim(*render(s))
    fig, ax = compose(img, px, FIG_W * 0.82)
    put(ax, px, "B", T(r"均匀 $\mathbf{B}=B_0\mathbf{e}_z$", r"uniform $\mathbf{B}=B_0\mathbf{e}_z$"), dx=-10, dy=-22, color=MAG, size=9.5)
    put_leader(ax, px, "A", T(r"$\mathbf{A}$ 绕圈：$A_\theta=B_0r/2$", r"$\mathbf{A}$ circulates: $A_\theta=B_0r/2$"), dx=170, dy=110, color=POT, size=9.5)
    put(ax, px, "disk", T(r"$\oint\mathbf{A}\cdot\mathrm{d}\boldsymbol{\ell}=\pi r^2B_0=\Phi_B$", r"$\oint\mathbf{A}\cdot\mathrm{d}\boldsymbol{\ell}=\pi r^2B_0=\Phi_B$"), dx=0, dy=-8, size=9.5)
    return fig


# ---------------------------------------------------------------------------
@figure("multipillar_3d")
def multipillar_3d():
    """Schematic four-pillar yoke permanent-magnet assembly (geometry only, not a design)."""
    IRON = "#A9B3BD"
    s = Scene(900, 640)
    s.camera(position=(7.6, -2.9, 2.0), target=(0, 0, 0.0), fov=27)
    s.lights(ambient=0.6, key=0.7, fill=0.3, key_dir=(5, -3, 7))
    P, t = 2.2, 0.22            # plate size, thickness
    zg = 0.45                    # half gap between pole faces
    zm = 0.22                    # magnet thickness
    zp = 0.07                    # pole-piece thickness
    ztop = zg + zp + zm          # underside of the top plate
    # plates
    s.box((0, 0, ztop + t / 2), (P, P, t), color=IRON, roughness=0.6, metalness=0.2)
    s.box((0, 0, -ztop - t / 2), (P, P, t), color=IRON, roughness=0.6, metalness=0.2)
    # pillars
    c = P / 2 - 0.2
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.cylinder((sx * c, sy * c, -ztop), (sx * c, sy * c, ztop), 0.15, color=IRON, roughness=0.6, metalness=0.2,
                       opacity=0.45 if (sx, sy) == (1, -1) else 1.0)
    # magnets (vermilion tint) and pole pieces (iron)
    s.cylinder((0, 0, ztop - zm), (0, 0, ztop), 0.62, color=tint(MAG, 0.55))
    s.cylinder((0, 0, -ztop), (0, 0, -ztop + zm), 0.62, color=tint(MAG, 0.55))
    s.cylinder((0, 0, zg), (0, 0, zg + zp), 0.68, color=IRON, roughness=0.5, metalness=0.25)
    s.cylinder((0, 0, -zg - zp), (0, 0, -zg), 0.68, color=IRON, roughness=0.5, metalness=0.25)
    # imaging region
    s.sphere((0, 0, 0), 0.32, color=tint(ELEC, 0.25), opacity=0.45)
    # main flux path (one of four symmetric loops), drawn slightly in front
    y0 = -0.05
    path = np.array([[0.0, y0, -0.3], [0.0, y0, 0.3], [0.0, y0, ztop + 0.08], [0.45, y0 - 0.2, ztop + 0.12],
                     [c - 0.25, -c + 0.3, ztop + 0.1], [c, -c, ztop - 0.2], [c, -c, 0.0], [c, -c, -ztop + 0.2],
                     [c - 0.25, -c + 0.3, -ztop - 0.1], [0.45, y0 - 0.2, -ztop - 0.12], [0.0, y0, -ztop - 0.08], [0.0, y0, -0.3]])
    from scipy.interpolate import splprep, splev
    tck, _ = splprep(path.T, s=0.002, per=False)
    curve = np.array(splev(np.linspace(0, 1, 300), tck)).T
    s.tube(curve, radius=0.025, color=MAG, arrows=[0.08, 0.3, 0.55, 0.8], head_len=0.13, head_radius=0.055, render_order=5)
    # a leakage path: pole edge straight to the back-right pillar (so that, seen from the camera,
    # it stays clear of the imaging sphere)
    leak = np.array([[0.49, 0.49, zg + 0.04], [0.72, 0.64, 0.30], [c - 0.11, c - 0.13, 0.05]])
    tck2, _ = splprep(leak.T, s=0, k=2)
    lc = np.array(splev(np.linspace(0, 1, 60), tck2)).T
    s.tube(lc, radius=0.012, color=tint(MAG, 0.6), arrows=[0.6], head_len=0.08, head_radius=0.03)
    for name, p in {"plate": (-P / 2 + 0.2, -P / 2, ztop + t), "pillar": (c, -c, -0.3), "magnet": (0.58, -0.22, -ztop + zm / 2),
                    "pole": (0.63, -0.24, zg + zp / 2), "roi": (-0.2, -0.25, 0.0), "flux": (c, -c, 0.0), "leak": (0.72, 0.64, 0.30)}.items():
        s.label(name, p)
    img, px = trim(*render(s))
    fig, ax = compose(img, px, FIG_W * 0.92)
    put_leader(ax, px, "plate", T("磁轭板（软铁）", "yoke plate (soft iron)"), dx=-60, dy=-40, color=INK, size=9.2)
    put_leader(ax, px, "magnet", T("永磁体（上下各一块）", "permanent magnets (top and bottom)"), dx=160, dy=60, color=MAG, size=9.2)
    put_leader(ax, px, "pole", T("极靴", "pole piece"), dx=170, dy=-30, color=INK, size=9.2)
    put_leader(ax, px, "roi", T("成像区（气隙中心）", "imaging region (gap center)"), dx=-170, dy=70, color=ELEC, size=9.2)
    put_leader(ax, px, "pillar", T("回磁柱（×4）", "return pillar (×4)"), dx=140, dy=80, color=INK, size=9.2)
    put_leader(ax, px, "flux", T("主磁通路径", "main flux path"), dx=150, dy=-20, color=MAG, size=9.2)
    put_leader(ax, px, "leak", T("漏磁：不经过气隙", "leakage: bypasses the gap"), dx=170, dy=-70, color=MAG, size=9.2)
    return fig


# ---------------------------------------------------------------------------
@figure("edge_element_3d")
def edge_element_3d():
    from render3d import compose_row
    V = np.array([[0, 0, 0], [1.6, 0, 0], [0.4, 1.5, 0], [0.55, 0.5, 1.4]], float)
    faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
    # barycentric gradients
    M = np.hstack([np.ones((4, 1)), V])
    Cinv = np.linalg.inv(M)              # rows: coefficients? solve N_i = a + b.x
    G = Cinv[1:, :].T                    # grad N_i (4 x 3)
    imgs, pxs = [], []
    cam = dict(position=(-1.6, -4.4, 2.9), target=(0.65, 0.5, 0.45), fov=30)
    # ---- nodal function N_0
    s = Scene(600, 520)
    s.camera(**cam)
    from style import seq_cmap
    cm = seq_cmap(POT)
    # subdivide each face to show the linear colour of N_0
    for f in faces:
        A, B, C = V[f]
        n = 18
        verts, cols, tris, idx = [], [], [], {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                p = A + (B - A) * i / n + (C - A) * j / n
                idx[(i, j)] = len(verts)
                verts.append(p)
                N0 = 1 - (p - V[0]) @ (-G[0]) if False else np.array([1, *p]) @ Cinv[:, 0]
                cols.append(cm(0.08 + 0.85 * np.clip(N0, 0, 1)))
        for i in range(n):
            for j in range(n - i):
                tris.append([idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]])
                if j < n - i - 1:
                    tris.append([idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]])
        s.mesh(np.array(verts), np.array(tris), vertex_colors=np.array(cols), unlit=True, opacity=0.92)
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        s.tube(np.linspace(V[a], V[b], 10), radius=0.012, color=MUTED)
    s.sphere(V[0], 0.05, color=POT)
    s.label("v0", V[0])
    img, px = trim(*render(s)); imgs.append(img); pxs.append(px)
    # ---- edge function W_01 = N0 grad N1 - N1 grad N0
    s = Scene(600, 520)
    s.camera(**cam)
    s.mesh(V, np.array(faces), color=tint(ACCENT, 0.12), opacity=0.35)
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        s.tube(np.linspace(V[a], V[b], 10), radius=0.012, color=MUTED)
    s.tube(np.linspace(V[0], V[1], 10), radius=0.03, color=ACCENT, arrows=[0.62], head_len=0.14, head_radius=0.06)
    # regular barycentric lattice (weights k/4), so the pattern of W_01 reads at a glance
    pts = []
    n = 4
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                w = np.array([i, j, k, n - i - j - k], float) / n
                if w.max() < 1.0:                       # skip the four vertices
                    pts.append(w @ V)
    for p in np.array(pts):
        N = np.array([1, *p]) @ Cinv
        W = N[0] * G[1] - N[1] * G[0]
        if np.linalg.norm(W) < 0.05:                    # W_01 vanishes on the opposite edge 2-3
            continue
        d = W * 0.3
        s.arrow(p - d / 2, p + d / 2, radius=0.016, color=ELEC, head_len=0.09, head_radius=0.038)
    s.label("edge", (V[0] + V[1]) / 2)
    img, px = trim(*render(s)); imgs.append(img); pxs.append(px)
    fig, axes = compose_row(imgs, FIG_W, titles=[T(r"节点基函数 $N_0$：顶点上的值", r"nodal function $N_0$: a vertex value"),
                                                  T(r"棱基函数 $\mathbf{W}_{01}$：沿一条棱的环量", r"edge function $\mathbf{W}_{01}$: circulation along an edge")],
                            title_colors=[POT, ACCENT])
    put_leader(axes[0], pxs[0], "v0", r"$N_0=1$", dx=70, dy=55, color=POT, size=10)
    put(axes[1], pxs[1], "edge", T(r"$\int_{0\to1}\mathbf{W}_{01}\cdot\mathrm{d}\boldsymbol{\ell}=1$", r"$\int_{0\to1}\mathbf{W}_{01}\cdot\mathrm{d}\boldsymbol{\ell}=1$"), dx=0, dy=40, color=ACCENT, size=9.8)
    return fig


# ---------------------------------------------------------------------------
@figure("gradient_3d")
def gradient_3d():
    s = Scene(900, 600)
    s.camera(position=(5.0, -6.4, 3.0), target=(0, 0, 0), fov=26)
    G = 1.0
    L = 1.0
    # coloured mid-plane y=0 showing b_z = G x
    X, Z = np.meshgrid(np.linspace(-L, L, 41), np.linspace(-L, L, 41))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bz", ["#5C6BC0", "#FFFFFF", MAG])
    C = cmap((G * X / L + 1) / 2)
    s.surface(X, np.zeros_like(X), Z, colors=C, unlit=True, opacity=0.85)
    s.polyline([[-L, 0, -L], [L, 0, -L], [L, 0, L], [-L, 0, L], [-L, 0, -L]], color=MUTED)
    # arrows of b = (G z, 0, G x) in the plane y = 0 (the field does not depend on y)
    for x in np.linspace(-0.8, 0.8, 5):
        for z in np.linspace(-0.8, 0.8, 5):
            b = np.array([G * z, 0.0, G * x])
            p = np.array([x, -0.02, z])
            if np.linalg.norm(b) < 1e-6:
                s.sphere(p, 0.03, color=MAG)
                continue
            d = b * 0.3
            s.arrow(p - d / 2, p + d / 2, radius=0.018, color=MAG, head_len=0.1, head_radius=0.042)
    # axes
    for v, name in (((1.35, 0, 0), "x"), ((0, 1.35, 0), "y"), ((0, 0, 1.35), "z")):
        s.arrow((0, 0, 0), v, radius=0.012, color=INK, head_len=0.1, head_radius=0.035)
        s.label(name, np.array(v) * 1.08)
    s.label("plane", (L, 0, -L))
    s.label("top", (0.8, -0.02, 0.8))
    img, px = trim(*render(s))
    fig, ax = compose(img, px, FIG_W * 0.86)
    for name in ("x", "y", "z"):
        put(ax, px, name, f"${name}$", size=11)
    put_leader(ax, px, "plane", T(r"$y=0$ 截面上的颜色：$b_z=G_xx$", r"color on $y=0$: $b_z=G_xx$"), dx=60, dy=-10, color=INK, size=9.3)
    put_leader(ax, px, "top", T(r"箭头：完整的 $\mathbf{b}=(G_xz,\,0,\,G_xx)$", r"arrows: the full $\mathbf{b}=(G_xz,\,0,\,G_xx)$"), dx=140, dy=-40, color=MAG, size=9.3)
    return fig


# ---------------------------------------------------------------------------
@figure("stream_cylinder_3d")
def stream_cylinder_3d():
    """Illustrative transverse-gradient stream function psi = cos(phi) * z exp(-z^2/2s^2) on a cylinder."""
    a, Lz, sz = 1.0, 2.6, 0.55
    ph = np.linspace(0, 2 * np.pi, 361)
    zz = np.linspace(-Lz / 2, Lz / 2, 261)
    PH, ZZ = np.meshgrid(ph, zz)
    psi = np.cos(PH) * ZZ * np.exp(-ZZ**2 / (2 * sz**2))
    s = Scene(900, 600)
    s.camera(position=(7.6, -2.6, 2.4), target=(0, 0, 0), fov=27)
    # former (translucent cylinder)
    s.surface(0.985 * a * np.cos(PH[:, ::6]), 0.985 * a * np.sin(PH[:, ::6]), ZZ[:, ::6], color="#E3E8EE")
    gen = contourpy.contour_generator(PH, ZZ, psi)
    pmax = np.abs(psi).max()
    for lev in pmax * np.array([-0.82, -0.55, -0.3, 0.3, 0.55, 0.82]):
        for seg in gen.lines(lev):
            if len(seg) < 10:
                continue
            P3 = np.column_stack([a * np.cos(seg[:, 0]), a * np.sin(seg[:, 0]), seg[:, 1]])
            # orient so that the arrow shows the current direction K = grad_s psi x n (n outward)
            k = len(seg) // 3
            t = P3[k + 1] - P3[k - 1]
            phi_k, z_k = seg[k]
            dpsi_dphi = -np.sin(phi_k) * z_k * np.exp(-z_k**2 / (2 * sz**2))
            dpsi_dz = np.cos(phi_k) * (1 - z_k**2 / sz**2) * np.exp(-z_k**2 / (2 * sz**2))
            e_phi = np.array([-np.sin(phi_k), np.cos(phi_k), 0.0])
            e_z = np.array([0, 0, 1.0])
            n = np.array([np.cos(phi_k), np.sin(phi_k), 0.0])
            grad = dpsi_dphi / a * e_phi + dpsi_dz * e_z
            K = np.cross(grad, n)
            if K @ t < 0:
                P3 = P3[::-1]
            s.tube(P3, radius=0.016, color=CURR, closed=False, arrows=[0.33], head_len=0.09, head_radius=0.035)
    s.arrow((0, 0, -1.6), (0, 0, 1.75), radius=0.01, color=MUTED, head_len=0.09, head_radius=0.03)
    s.label("z", (0, 0, 1.8))
    s.label("fp", (a * np.cos(-0.4), a * np.sin(-0.4), sz))
    img, px = trim(*render(s))
    fig, ax = compose(img, px, FIG_W * 0.8)
    put(ax, px, "z", "$z$", dx=12, size=10.5)
    put_leader(ax, px, "fp", T(r"$\psi_s$ 等值线 = 电流路径", r"contours of $\psi_s$ = current paths"), dx=150, dy=-60, color=CURR, size=9.3)
    return fig
