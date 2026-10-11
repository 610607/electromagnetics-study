"""Figures for Chapter 11: TMS — induction from a prescribed magnetic source.

All field computations reuse the tested P1 finite elements of numerics.py and the model of
fig_ch10 (unit square, inclusion sigma = 10 at (0.62, 0.56), radius 0.16, element labels).
The primary field E_p = 0.5 (y - 1/2, -(x - 1/2)) is -dA/dt of a spatially uniform dB_z/dt = 1
(normalised), exactly as in numerics.induced_field_example; the outer boundary is insulated.

induction_correction  primary field, gradient correction -grad(phi) and total field, one arrow scale
tms_fem_inclusion     |E| map with current lines; profile across the inclusion (E drops, J rises)
tes_tms_compare       same mesh and same K: right-hand sides and resulting fields of TES and TMS
pulse                 damped-sine coil current and its derivative (old-edition synthetic waveform)
drive_scaling         I scales B through b(r); dI/dt scales E through e(r) (which contains the body correction)
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle
from scipy.sparse.linalg import spsolve

from numerics import stiffness
from style import (ACCENT, AMBER, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, MAG, MAG_T, MATL, MATL_T, MUTED, POT,
                   POT_T, T, arrow, canvas, card, clean_axes, colorbar, figure, halo, label, panel_title, pill, plt,
                   rbox, seq_cmap, tint)
from fig_ch10 import (ANODE, CATHODE, INC_C, INC_R, boundary_nodes, electrodes, grad, model, phi_lines,
                      psi_lines, arrows_on_contours, solve_neumann, square_frame, stream_tes, tes_uniform, tri_of)


# ---------------------------------------------------------------------------
# TMS solver on the shared model
# ---------------------------------------------------------------------------
def primary(xy):
    """E_p = -dA/dt for a uniform dB_z/dt = 1 about the centre (1/2, 1/2): clockwise circles."""
    return np.column_stack((0.5 * (xy[:, 1] - 0.5), -0.5 * (xy[:, 0] - 0.5)))


def tms_rhs(m, Ep=None):
    """b_i = int sigma grad N_i . E_p  (element-wise exact for the linear E_p)."""
    Ep = primary(m["centre"]) if Ep is None else Ep
    be = (m["area"] * m["sigma"])[:, None] * np.einsum("eid,ed->ei", m["grads"], Ep)
    b = np.zeros(len(m["p"]))
    np.add.at(b, m["cells"].ravel(), be.ravel())
    return b


def tms_solve(m):
    Ep = primary(m["centre"])
    b = tms_rhs(m, Ep)
    phi = solve_neumann(m, b)
    phi -= np.sum(m["area"] * phi[m["cells"]].mean(axis=1))          # zero-mean reference
    corr = -grad(m, phi)
    E = Ep + corr
    return dict(Ep=Ep, phi=phi, corr=corr, E=E, J=m["sigma"][:, None] * E, b=b)


def stream_tms(m, bdot=1.0):
    """Current stream function: J = (d_y psi, -d_x psi), psi = 0 on the insulated boundary.

    (curl E)_z = -dB_z/dt  and  E = J/sigma  give  div(sigma^{-1} grad psi) = dB_z/dt.
    """
    p = m["p"]
    Kinv, area, _ = stiffness(p, m["cells"], 1.0 / m["sigma"])
    load = np.zeros(len(p))
    np.add.at(load, m["cells"].ravel(), np.repeat(area / 3.0, 3))
    bnd = boundary_nodes(m)
    free = np.setdiff1d(np.arange(len(p)), bnd)
    psi = np.zeros(len(p))
    psi[free] = spsolve(Kinv[free][:, free].tocsc(), -bdot * load[free])
    return psi


def sample(m, field, pts):
    """Element-wise field at points (element containing each point)."""
    finder = tri_of(m).get_trifinder()
    e = finder(pts[:, 0], pts[:, 1])
    out = field[e]
    out[e < 0] = np.nan
    return out


# ---------------------------------------------------------------------------
@figure("induction_correction")
def induction_correction():
    m = model(128)
    s = tms_solve(m)
    q = np.linspace(0.07, 0.93, 9)
    X, Y = np.meshgrid(q, q)
    pts = np.column_stack((X.ravel(), Y.ravel()))
    fig, axs = plt.subplots(1, 3, figsize=(FIG_W, 2.72))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.80, bottom=0.12, wspace=0.10)
    scale = 3.0                                      # same arrow scale in all panels (data units)
    specs = [(s["Ep"], tint(ELEC, 0.55), T("初级场 $\\mathbf{E}_p=-\\partial_t\\mathbf{A}$", "primary $\\mathbf{E}_p=-\\partial_t\\mathbf{A}$")),
             (s["corr"], POT, T("修正 $-\\nabla\\phi$", "correction $-\\nabla\\phi$")),
             (s["E"], ELEC, T("总场 $\\mathbf{E}=\\mathbf{E}_p-\\nabla\\phi$", "total $\\mathbf{E}=\\mathbf{E}_p-\\nabla\\phi$"))]
    for k, (ax, (F, col, ttl)) in enumerate(zip(axs, specs)):
        square_frame(ax, inclusion=True)
        ax.set_xlim(-0.085, 1.085)
        ax.set_ylim(-0.085, 1.085)
        if k == 1:
            ph = s["phi"]
            lv = np.arange(0.004, 0.0241, 0.004)
            tri = tri_of(m)
            ax.tricontour(tri, ph, levels=lv, colors=[tint(POT, 0.5)], linewidths=0.75, zorder=1)
            ax.tricontour(tri, ph, levels=-lv[::-1], colors=[tint(POT, 0.5)], linewidths=0.75, zorder=1,
                          linestyles=[(0, (3, 2))])
            # sign of the boundary charge layer, read from the computed potential on each half-side
            p = m["p"]
            for pt, off in (((1, .75), (1, 0)), ((1, .25), (1, 0)), ((0, .75), (-1, 0)), ((0, .25), (-1, 0)),
                            ((.75, 1), (0, 1)), ((.25, 1), (0, 1)), ((.75, 0), (0, -1)), ((.25, 0), (0, -1))):
                kk = np.argmin(np.sum((p - np.array(pt)) ** 2, axis=1))
                sgn = "+" if ph[kk] > 0 else "-"
                from style import charge
                charge(ax, pt[0] + 0.045 * off[0], pt[1] + 0.045 * off[1], sgn, r=0.034, size=7.5)
        V = sample(m, F, pts)
        ax.quiver(pts[:, 0], pts[:, 1], V[:, 0], V[:, 1], color=col, angles="xy", scale_units="xy", scale=scale,
                  width=0.0085, headwidth=3.4, headlength=3.8, headaxislength=3.4, zorder=4, pivot="mid")
        panel_title(ax, ttl, color=col if k else ELEC, size=9.2, y=1.03)
    label(axs[0], 0.5, -0.15, T("穿出边界：电流“想”流出去", "pushes current out"), size=7.9, color=MUTED)
    label(axs[1], 0.5, -0.15, T("由边界与界面电荷产生；无旋", "from charges; curl-free"), size=7.9, color=MUTED)
    label(axs[2], 0.5, -0.15, T("贴着边界流；夹杂内变弱", "runs along the boundary"), size=7.9, color=MUTED)
    for x0, sym in ((0.335, "+"), (0.668, "=")):
        fig.text(x0, 0.46, sym, ha="center", va="center", fontsize=15, color=MUTED, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
def profile_data(n=256, y0=0.56):
    """|E| and |J| along y = y0 for the inclusion model and the homogeneous model."""
    xs = np.linspace(0.004, 0.996, 600)
    pts = np.column_stack((xs, np.full_like(xs, y0)))
    out = {"x": xs}
    for key, inc in (("inc", True), ("hom", False)):
        m = model(n, inclusion=inc)
        s = tms_solve(m)
        out[key + "_E"] = np.linalg.norm(sample(m, s["E"], pts), axis=1)
        out[key + "_J"] = np.linalg.norm(sample(m, s["J"], pts), axis=1)
        out[key + "_Ep"] = np.linalg.norm(primary(pts), axis=1)
    return out


@figure("tms_fem_inclusion")
def tms_fem_inclusion():
    m = model(128)
    s = tms_solve(m)
    psi = stream_tms(m)
    fig = plt.figure(figsize=(FIG_W, 2.95))
    ax = fig.add_axes([0.0, 0.07, 0.43, 0.80])
    cax = fig.add_axes([0.425, 0.17, 0.014, 0.6])
    tri = tri_of(m)
    mag = np.linalg.norm(s["E"], axis=1)
    pc = ax.tripcolor(tri, facecolors=mag, cmap=seq_cmap(ELEC), shading="flat", rasterized=True, zorder=0,
                      vmin=0, vmax=0.35)
    ax.add_patch(Circle(INC_C, INC_R, fc="none", ec=INK, lw=0.8, ls=(0, (3, 2)), zorder=6))
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec=INK, lw=1.1, zorder=6))
    lv = np.linspace(psi.min(), 0, 9)[1:-1]
    cs = ax.tricontour(tri, psi, levels=lv, colors=[CURR], linewidths=1.0, zorder=3, linestyles="solid")
    arrows_on_contours(ax, cs, m, s["J"], CURR, frac=0.12, size=7, lw=1.0, min_len=0.1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    halo(label(ax, INC_C[0], INC_C[1] + 0.005, r"$\sigma=10$", size=8.4), 2.5)
    cb = fig.colorbar(pc, cax=cax)
    cax.set_title(r"$|\mathbf{E}|$", fontsize=8.6, pad=4)
    cb.ax.tick_params(labelsize=7.6, length=2)
    cb.outline.set_linewidth(0.5)
    ax.plot([0, 1], [0.56, 0.56], color=INK, lw=0.7, ls=(0, (1, 1.5)), zorder=5)
    label(ax, 0.03, 0.6, "a", size=8, color=INK, ha="left")
    label(ax, 0.97, 0.6, "b", size=8, color=INK, ha="right")
    panel_title(ax, T("总电场大小与电流线", "total |E| and current lines"), y=1.02, size=9.2)
    # ---- profile
    d = profile_data()
    bx = fig.add_axes([0.60, 0.53, 0.38, 0.33])
    cx = fig.add_axes([0.60, 0.14, 0.38, 0.33], sharex=bx)
    x0, x1 = INC_C[0] - INC_R, INC_C[0] + INC_R
    for a in (bx, cx):
        a.axvspan(x0, x1, color=tint("#C9B79C", 0.45), lw=0, zorder=0)
        clean_axes(a)
    bx.plot(d["x"], d["hom_E"], color=ELEC, lw=1.1, ls=(0, (4, 2)), zorder=3)
    bx.plot(d["x"], d["inc_E"], color=ELEC, lw=1.8, zorder=4)
    cx.plot(d["x"], d["hom_J"], color=CURR, lw=1.1, ls=(0, (4, 2)), zorder=3)
    cx.plot(d["x"], d["inc_J"], color=CURR, lw=1.8, zorder=4)
    bx.set_ylabel(r"$|\mathbf{E}|$", fontsize=9)
    cx.set_ylabel(r"$|\mathbf{J}|$", fontsize=9)
    bx.set_ylim(0, 0.32)
    cx.set_ylim(0, 0.95)
    bx.set_yticks([0, 0.1, 0.2, 0.3])
    cx.set_yticks([0, 0.4, 0.8])
    plt.setp(bx.get_xticklabels(), visible=False)
    cx.set_xlim(0, 1)
    cx.set_xticks([0, 1], ["a", "b"])
    cx.set_xlabel(T("沿虚线 a→b", "along the dotted line a→b"), fontsize=8.6, labelpad=1)
    bx.text((x0 + x1) / 2, 0.29, T("夹杂", "inclusion"), ha="center", fontsize=8.0, color=MUTED)
    bx.text(0.25, 0.04, T("夹杂内电场减弱", "E is weaker inside"), ha="center", fontsize=8.2, color=ELEC)
    cx.text(0.2, 0.6, T("跨过界面 $J$ 突然增大\n（这里电流沿界面流）", "J jumps up at the\ninterface (flow along it)"),
            ha="center", fontsize=8.0, color=CURR, linespacing=1.25)
    from matplotlib.lines import Line2D
    bx.legend([Line2D([], [], color=INK, lw=1.8), Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2)))],
              [T("有夹杂", "with inclusion"), T("均匀方块", "homogeneous")], loc="lower center",
              bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=8.2, handlelength=2.0, borderaxespad=0.1)
    return fig


# ---------------------------------------------------------------------------
def _rhs_dots(ax, m, b, vmax, size=26.0):
    nz = np.abs(b) > 1e-12 * np.abs(b).max()
    p = m["p"][nz]
    v = b[nz]
    order = np.argsort(np.abs(v))
    p, v = p[order], v[order]
    col = np.where(v > 0, MAG, ELEC)
    ax.scatter(p[:, 0], p[:, 1], s=size * np.sqrt(np.abs(v) / vmax) + 0.6, c=col, linewidths=0, zorder=7,
               clip_on=False)


@figure("tes_tms_compare")
def tes_tms_compare():
    n = 64
    m = model(n)
    tes = tes_uniform(m)
    tms = tms_solve(m)
    psi_e = stream_tes(m)
    psi_m = stream_tms(m)
    fig = plt.figure(figsize=(FIG_W, 4.9))
    W, H = 0.36, 0.405
    xs = (0.135, 0.585)
    ys = (0.525, 0.068)
    rows = [("TES", tes, psi_e), ("TMS", tms, psi_m)]
    tri = tri_of(m)
    mag_max = 0.0
    for r, (name, sol, psi) in enumerate(rows):
        a1 = fig.add_axes([xs[0], ys[r], W, H])
        a2 = fig.add_axes([xs[1], ys[r], W, H])
        # left: right-hand side b
        square_frame(a1, inclusion=True, fill=tint("#C9B79C", 0.18), inc_fill=tint("#C9B79C", 0.4))
        b = sol["b"]
        _rhs_dots(a1, m, b, np.abs(b).max(), size=17 if name == "TMS" else 20)
        if name == "TES":
            electrodes(a1, signs=False)
        # right: |E| (light) + current lines
        mag = np.linalg.norm(sol["E"], axis=1)
        vmax = np.quantile(mag, 0.985)
        square_frame(a2, inclusion=False, fill="white")
        a2.tripcolor(tri, facecolors=np.minimum(mag, vmax), cmap=seq_cmap(ELEC), shading="flat", rasterized=True,
                     zorder=0.2, vmin=0, vmax=vmax)
        a2.add_patch(Circle(INC_C, INC_R, fc="none", ec=INK, lw=0.7, ls=(0, (3, 2)), zorder=6))
        if name == "TES":
            lv = np.linspace(0, 1, 10)[1:-1]
            electrodes(a2, signs=False)
        else:
            lv = np.linspace(psi.min(), 0, 8)[1:-1]
        cs = a2.tricontour(tri, psi, levels=lv, colors=[CURR], linewidths=0.95, zorder=3, linestyles="solid")
        arrows_on_contours(a2, cs, m, sol["J"], CURR, frac=0.5 if name == "TES" else 0.12, size=6.5, lw=0.9,
                           min_len=0.1)
        # row label
        fig.text(0.055, ys[r] + H / 2, name, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold",
                 color=INK)
    # column titles
    fig.text(xs[0] + W / 2, 0.978, T("右端项 $\\mathbf{b}$：源从哪里进入", "right-hand side $\\mathbf{b}$: where the source enters"),
             ha="center", va="center", fontsize=9.3, fontweight="bold")
    fig.text(xs[1] + W / 2, 0.978, T("解：$|\\mathbf{E}|$（底色）与电流线", "solution: $|\\mathbf{E}|$ (shading) and current lines"),
             ha="center", va="center", fontsize=9.3, fontweight="bold")
    # middle operator badges
    for r in range(2):
        yc = ys[r] + H / 2
        fig.text(0.5 * (xs[0] + W + xs[1]), yc + 0.03, r"$\mathbf{K}\boldsymbol{\phi}=\mathbf{b}$", ha="center",
                 va="center", fontsize=9.5, color=INK)
        fig.patches.append(plt.matplotlib.patches.FancyArrowPatch(
            (xs[0] + W + 0.01, yc - 0.005), (xs[1] - 0.01, yc - 0.005), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK))
    fig.text(0.5 * (xs[0] + W + xs[1]), 0.5 * (ys[0] + ys[1] + H) + 0.0,
             T("同一个\n$\\mathbf{K}$", "same\n$\\mathbf{K}$"), ha="center", va="center", fontsize=8.8,
             color=ACCENT, fontweight="bold", linespacing=1.2)
    # notes on the left panels
    fig.text(xs[0] + W / 2, ys[0] - 0.014, T("只在电极节点上：橙色注入，蓝色引出", "only at electrode nodes: orange in, blue out"),
             ha="center", va="top", fontsize=8.0, color=MUTED)
    fig.text(xs[0] + W / 2, ys[1] - 0.014, T("只在外边界与夹杂界面上：$\\sigma\\mathbf{E}_p$ 穿过 $\\sigma$ 跳变处",
                                            "only on the boundary and the interface, where $\\sigma\\mathbf{E}_p$ meets a jump in $\\sigma$"),
             ha="center", va="top", fontsize=8.0, color=MUTED)
    return fig


# ---------------------------------------------------------------------------
# Waveform (old-edition synthetic damped sine: tau = 200 us, f = 5 kHz)
# ---------------------------------------------------------------------------
PULSE_TAU = 200e-6
PULSE_F = 5000.0


def pulse_waveform(t):
    w = 2 * np.pi * PULSE_F
    I = np.exp(-t / PULSE_TAU) * np.sin(w * t)
    dI = np.exp(-t / PULSE_TAU) * (w * np.cos(w * t) - np.sin(w * t) / PULSE_TAU)
    return I, dI


@figure("pulse")
def pulse():
    t = np.linspace(0, 600e-6, 2401)
    I, dI = pulse_waveform(t)
    In, dIn = I / np.abs(I).max(), dI / np.abs(dI).max()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(FIG_W, 3.1), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.79, top=0.92, bottom=0.14, hspace=0.32)
    tu = t * 1e6
    for a in (ax, bx):
        a.axhline(0, color=MUTED, lw=0.6, zorder=1)
        clean_axes(a)
        a.set_ylim(-1.15, 1.2)
        a.set_yticks([-1, 0, 1])
    ax.plot(tu, In, color=CURR, lw=1.9, zorder=3)
    bx.plot(tu, dIn, color=ELEC, lw=1.9, zorder=3)
    k_peak = np.argmax(In)
    k_zero = np.argmin(np.abs(In[400:1300])) + 400
    for k in (k_peak, k_zero):
        for a in (ax, bx):
            a.axvline(tu[k], color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.plot(tu[k_peak], In[k_peak], "o", ms=4.5, color=CURR, zorder=4)
    bx.plot(tu[k_peak], dIn[k_peak], "o", ms=4.5, color=ELEC, mfc="white", zorder=4)
    ax.plot(tu[k_zero], In[k_zero], "o", ms=4.5, color=CURR, mfc="white", zorder=4)
    bx.plot(tu[k_zero], dIn[k_zero], "o", ms=4.5, color=ELEC, zorder=4)
    ax.text(0.0, 1.06, T("线圈电流 $I$（按自身峰值归一化）", "coil current $I$ (normalized to its peak)"),
            transform=ax.transAxes, fontsize=8.8, color=CURR, ha="left", va="bottom")
    bx.text(0.0, 1.06, T("变化率 $\\mathrm{d}I/\\mathrm{d}t$（按自身峰值归一化）", "rate $\\mathrm{d}I/\\mathrm{d}t$ (normalized to its peak)"),
            transform=bx.transAxes, fontsize=8.8, color=ELEC, ha="left", va="bottom")
    bx.set_xlabel(T(r"时间 ($\mu$s)", r"time ($\mu$s)"))
    bx.set_xlim(0, 600)
    fig.text(0.81, 0.78, T("磁场 $\\mathbf{B}\\propto I$", "field $\\mathbf{B}\\propto I$"), fontsize=9, color=MAG, va="center")
    fig.text(0.81, 0.63, T("电流峰值处\n$\\mathrm{d}I/\\mathrm{d}t=0$：\n此刻感应电场为零", "at the current\npeak $\\mathrm{d}I/\\mathrm{d}t=0$:\nno induced E"),
             fontsize=8.2, color=MUTED, va="center", linespacing=1.3)
    fig.text(0.81, 0.39, T("电场 $\\mathbf{E}\\propto\\mathrm{d}I/\\mathrm{d}t$", "field $\\mathbf{E}\\propto\\mathrm{d}I/\\mathrm{d}t$"), fontsize=9,
             color=ELEC, va="center")
    fig.text(0.81, 0.25, T("电流过零附近，\n$|\\mathrm{d}I/\\mathrm{d}t|$ 取极大", "near the zeros of\n$I$, $|\\mathrm{d}I/\\mathrm{d}t|$\npeaks"),
             fontsize=8.2, color=MUTED, va="center", linespacing=1.3)
    return fig


# ---------------------------------------------------------------------------
@figure("drive_scaling")
def drive_scaling():
    fig, ax = canvas(FIG_W, 2.95, xlim=(0, 10), ylim=(0, 4.92))
    # two rows: I -> B ; dI/dt -> E   (example values: the long solenoid of Chapter 4, r = 1 cm)
    rows = [(3.95, MAG, T("线圈电流 $I(t)$", "coil current $I(t)$"), ["A"],
             T("每安培的场图 $\\mathbf{b}(\\mathbf{r})$", "field per ampere $\\mathbf{b}(\\mathbf{r})$"),
             ["T/A", T("例：螺线管内 $1.26\\,$mT/A", "e.g. solenoid: $1.26\\,$mT/A")],
             T("磁场 $\\mathbf{B}=I\\,\\mathbf{b}$", "field $\\mathbf{B}=I\\,\\mathbf{b}$"), ["T"],
             T("只由线圈和磁环境决定", "set by the coil and its magnetic surroundings")),
            (1.75, ELEC, T("变化率 $\\dot I(t)$", "slew rate $\\dot I(t)$"), ["A/s"],
             T("每 A/s 的场图 $\\mathbf{e}(\\mathbf{r})$", "field per A/s $\\mathbf{e}(\\mathbf{r})$"),
             ["(V/m)/(A/s)", T("例：$r=1\\,$cm 处 $6.28\\times10^{-6}$", "e.g. $r=1\\,$cm: $6.28\\times10^{-6}$")],
             T("电场 $\\mathbf{E}=\\dot I\\,\\mathbf{e}$", "field $\\mathbf{E}=\\dot I\\,\\mathbf{e}$"), ["V/m"],
             T("已含导体修正 $-\\nabla\\phi$，所以取决于组织", "already contains the body correction $-\\nabla\\phi$"))]
    for y, c_out, s_in, b_in, s_map, b_map, s_out, b_out, note in rows:
        card(ax, 0.1, y - 0.6, 2.3, 1.2, s_in, b_in, color=CURR, title_size=8.8, body_size=8.6, band=0.42)
        card(ax, 3.45, y - 0.6, 3.1, 1.2, s_map, b_map, color=ACCENT, title_size=8.8, body_size=8.3, band=0.42)
        card(ax, 7.6, y - 0.6, 2.3, 1.2, s_out, b_out, color=c_out, title_size=8.8, body_size=8.6, band=0.42)
        arrow(ax, (2.45, y - 0.1), (3.4, y - 0.1), color=INK, lw=1.3, ms=11)
        arrow(ax, (6.6, y - 0.1), (7.55, y - 0.1), color=INK, lw=1.3, ms=11)
        label(ax, 2.92, y + 0.17, "×", size=13, color=MUTED)
        label(ax, 7.07, y + 0.17, "=", size=13, color=MUTED)
        label(ax, 5.0, y - 0.82, note, size=8.2, color=MUTED)
    ax.plot([1.25, 1.25], [3.33, 2.37], color=MUTED, lw=1.0, ls=(0, (2, 2)))
    label(ax, 1.38, 2.85, T("对时间求导", "differentiate in time"), size=8.0, color=MUTED, ha="left")
    rbox(ax, 0.1, 0.0, 9.8, 0.46, fc=tint(AMBER, 0.12), ec=tint(AMBER, 0.6), lw=0.8)
    label(ax, 5.0, 0.23, T("单位换算：$1\\,\\mathrm{A/\\mu s}=10^{6}\\,\\mathrm{A/s}$，同一张图按每 A/μs 标注时数值大 $10^{6}$ 倍；匝数只能计入一次",
                           "units: $1\\,\\mathrm{A/\\mu s}=10^{6}\\,\\mathrm{A/s}$, so a map labelled per A/μs is $10^{6}$ times larger; count turns once"),
          size=8.1, color=INK)
    return fig
