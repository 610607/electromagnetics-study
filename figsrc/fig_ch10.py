"""Figures for Chapter 10: TES — volume conduction driven by electrode currents.

The computations reuse the tested P1 finite elements of numerics.py on the unit square with the
same conductivity inclusion as numerics.induced_field_example (centre (0.62, 0.56), radius 0.16,
sigma = 10 inside, 1 outside; the inclusion is approximated by element labels).  Two electrode
patches sit on the top side y = 1:

    anode   x in [0.125, 0.375]   (current I enters the body)
    cathode x in [0.625, 0.875]   (current I leaves the body)

Three electrode models are implemented, all in normalised units (sigma_background = 1, I = 1):
    * prescribed uniform inward current density j_in = I / w        (Neumann data)
    * equipotential electrodes, phi = U on the anode, 0 on the cathode (Dirichlet data)
    * complete electrode model (CEM, eq:cem): U_l - phi = z j_in, int j_in dS = I_l,
      solved together with the electrode voltages U_l (Somersalo, Cheney & Isaacson 1992).

tes_setup        equipotentials and current lines, homogeneous square vs square with inclusion
electrode_models three electrode models: schematic (top) + computed j_in under the anode (bottom)
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve

from numerics import square_mesh, stiffness
from style import (ACCENT, AMBER, CURR, CURR_T, ELEC, ELEC_T, FIG_W, INK, IRON_T, MATL, MATL_T, MUTED, POT,
                   POT_T, T, arrow, clean_axes, figure, halo, label, panel_title, plt, rbox, seq_cmap, tint)

# ---------------------------------------------------------------------------
# Geometry shared with Chapter 11
# ---------------------------------------------------------------------------
INC_C = np.array([0.62, 0.56])
INC_R = 0.16
SIG_IN = 10.0
ANODE = (0.125, 0.375)
CATHODE = (0.625, 0.875)
ELECTRODE_GREY = "#9AA3AB"


def model(n: int, inclusion: bool = True, sig_in: float = SIG_IN):
    """Mesh, element conductivities and stiffness matrix (same labels as numerics.induced_field_example)."""
    p, cells = square_mesh(n)
    centre = p[cells].mean(axis=1)
    inc = np.sum((centre - INC_C) ** 2, axis=1) < INC_R**2
    sigma = np.where(inc & inclusion, sig_in, 1.0)
    K, area, grads = stiffness(p, cells, sigma)
    return dict(n=n, p=p, cells=cells, sigma=sigma, inc=inc, K=K, area=area, grads=grads, centre=centre)


def top_nodes(m) -> np.ndarray:
    n = m["n"]
    return np.arange(n * (n + 1), (n + 1) ** 2)          # y = 1, x increasing


def boundary_nodes(m) -> np.ndarray:
    p = m["p"]
    return np.flatnonzero(np.any((p <= 1e-12) | (p >= 1 - 1e-12), axis=1))


def patch_edges(m, patch):
    """Top-side boundary edges (i, j, length) inside the patch [x0, x1]."""
    idx = top_nodes(m)
    x = m["p"][idx, 0]
    out = []
    for k in range(len(idx) - 1):
        if x[k] >= patch[0] - 1e-12 and x[k + 1] <= patch[1] + 1e-12:
            out.append((idx[k], idx[k + 1], x[k + 1] - x[k]))
    return out


def patch_nodes(m, patch):
    idx = top_nodes(m)
    x = m["p"][idx, 0]
    return idx[(x >= patch[0] - 1e-12) & (x <= patch[1] + 1e-12)]


def uniform_load(m, patch, j_in):
    """Consistent nodal load of a uniform inward current density j_in on a top patch."""
    b = np.zeros(len(m["p"]))
    for i, k, h in patch_edges(m, patch):
        b[i] += j_in * h / 2
        b[k] += j_in * h / 2
    return b


def edge_mass(m, patch):
    """1-D boundary mass matrix and load vector int N_i dS on a top patch."""
    N = len(m["p"])
    rows, cols, vals = [], [], []
    vec = np.zeros(N)
    for i, k, h in patch_edges(m, patch):
        for a, b, v in ((i, i, h / 3), (k, k, h / 3), (i, k, h / 6), (k, i, h / 6)):
            rows.append(a); cols.append(b); vals.append(v)
        vec[i] += h / 2
        vec[k] += h / 2
    M = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return M, vec


def grad(m, phi):
    """Element gradients of a nodal P1 field."""
    return np.einsum("ei,eid->ed", phi[m["cells"]], m["grads"])


def solve_neumann(m, b, ref=None):
    """K phi = b with one potential degree of freedom fixed to zero (reference)."""
    N = len(m["p"])
    ref = int(ref if ref is not None else 0)
    free = np.setdiff1d(np.arange(N), [ref])
    phi = np.zeros(N)
    phi[free] = spsolve(m["K"][free][:, free].tocsc(), b[free])
    return phi


def tes_uniform(m, I=1.0, anode=ANODE, cathode=CATHODE):
    """Uniform current-density electrodes; potential referenced to zero mean."""
    w_a, w_c = anode[1] - anode[0], cathode[1] - cathode[0]
    b = uniform_load(m, anode, I / w_a) + uniform_load(m, cathode, -I / w_c)
    phi = solve_neumann(m, b)
    phi -= np.sum(m["area"] * phi[m["cells"]].mean(axis=1))           # zero-mean reference
    E = -grad(m, phi)
    return dict(phi=phi, E=E, J=m["sigma"][:, None] * E, b=b)


def tes_dirichlet(m, I=1.0, anode=ANODE, cathode=CATHODE):
    """Equipotential electrodes (phi = U on the anode, 0 on the cathode), scaled to total current I."""
    N = len(m["p"])
    a_nodes, c_nodes = patch_nodes(m, anode), patch_nodes(m, cathode)
    fixed = np.concatenate((a_nodes, c_nodes))
    phi = np.zeros(N)
    phi[a_nodes] = 1.0
    free = np.setdiff1d(np.arange(N), fixed)
    K = m["K"]
    phi[free] = spsolve(K[free][:, free].tocsc(), -K[free][:, fixed] @ phi[fixed])
    reaction = K @ phi                                   # nodal currents injected at fixed nodes
    scale = I / reaction[a_nodes].sum()
    phi *= scale
    reaction *= scale
    _, w = edge_mass(m, anode)
    j_nodes = reaction[a_nodes] / w[a_nodes]             # lumped density int j N_i / int N_i
    return dict(phi=phi, E=-grad(m, phi), U=scale, reaction=reaction, j_anode=j_nodes,
                x_anode=m["p"][a_nodes, 0], I_cathode=reaction[c_nodes].sum())


def tes_cem(m, z, I=1.0, anode=ANODE, cathode=CATHODE):
    """Complete electrode model.  Unknowns: nodal phi and the anode voltage U_a (U_c = 0 is the reference).

        int sigma grad phi . grad v + sum_l (1/z) int_l (phi - U_l) v = 0
        (1/z) int_l (U_l - phi) dS = I_l
    """
    N = len(m["p"])
    Ma, wa = edge_mass(m, anode)
    Mc, wc = edge_mass(m, cathode)
    A11 = (m["K"] + (Ma + Mc) / z).tocsr()
    A12 = csr_matrix((-wa / z)[:, None])
    A21 = csr_matrix((-wa / z)[None, :])
    A22 = csr_matrix(np.array([[wa.sum() / z]]))
    A = bmat([[A11, A12], [A21, A22]]).tocsc()
    rhs = np.zeros(N + 1)
    rhs[N] = I
    sol = spsolve(A, rhs)
    phi, Ua = sol[:N], sol[N]
    a_nodes, c_nodes = patch_nodes(m, anode), patch_nodes(m, cathode)
    j_anode = (Ua - phi[a_nodes]) / z
    j_cath = (0.0 - phi[c_nodes]) / z
    I_a = wa[a_nodes] @ j_anode
    I_c = wc[c_nodes] @ j_cath
    return dict(phi=phi, E=-grad(m, phi), U=Ua, j_anode=j_anode, x_anode=m["p"][a_nodes, 0],
                I_anode=I_a, I_cathode=I_c, phi_anode=phi[a_nodes])


def stream_tes(m, I=1.0, anode=ANODE, cathode=CATHODE):
    """Current stream function psi for uniform-density electrodes (J = (d_y psi, -d_x psi)).

    curl(J / sigma) = curl E = 0  ->  div(sigma^{-1} grad psi) = 0, with psi on the boundary equal
    to the current that has crossed it (d psi/ds = J.n along a counter-clockwise traversal).
    Equal steps of psi enclose equal current, so line density shows |J|.
    """
    p = m["p"]
    Kinv, _, _ = stiffness(p, m["cells"], 1.0 / m["sigma"])
    bnd = boundary_nodes(m)
    psi_b = np.zeros(len(p))
    top = np.isclose(p[:, 1], 1.0)
    x = p[:, 0]
    wa, wc = anode[1] - anode[0], cathode[1] - cathode[0]
    # traversing the top side from right to left: cathode adds +I, anode removes it again
    psi_b = np.where(top, np.clip((cathode[1] - x) / wc, 0, 1) * I - np.clip((anode[1] - x) / wa, 0, 1) * I, 0.0)
    free = np.setdiff1d(np.arange(len(p)), bnd)
    psi = psi_b.copy()
    psi[free] = spsolve(Kinv[free][:, free].tocsc(), -Kinv[free][:, bnd] @ psi_b[bnd])
    return psi


# ---------------------------------------------------------------------------
# Plot helpers shared with Chapter 11
# ---------------------------------------------------------------------------
def tri_of(m):
    return Triangulation(m["p"][:, 0], m["p"][:, 1], m["cells"])


def square_frame(ax, inclusion=True, lw=1.1, fill=MATL_T, inc_fill=None, inc_ls=(0, (3, 2))):
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=fill, ec="none", zorder=0))
    if inclusion:
        ax.add_patch(Circle(INC_C, INC_R, fc=inc_fill if inc_fill is not None else tint("#C9B79C", 0.75),
                            ec=MATL, lw=0.8, ls=inc_ls, zorder=0.5))
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec=INK, lw=lw, zorder=6))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.axis("off")


def electrodes(ax, anode=ANODE, cathode=CATHODE, h=0.045, signs=True, size=9.0):
    for (x0, x1), s in ((anode, "+"), (cathode, "−")):
        ax.add_patch(Rectangle((x0, 1.0), x1 - x0, h, fc=ELECTRODE_GREY, ec="none", zorder=7, clip_on=False))
        if signs:
            label(ax, (x0 + x1) / 2, 1.0 + h + 0.055, s, size=size + 2, weight="bold",
                  color=CURR if s == "+" else CURR)


def psi_lines(ax, m, psi, levels, color=CURR, lw=1.0, z=3):
    tri = tri_of(m)
    return ax.tricontour(tri, psi, levels=levels, colors=[color], linewidths=lw, zorder=z,
                         linestyles="solid")


def phi_lines(ax, m, phi, levels, color=POT, lw=0.8, z=2.5, ls=(0, (3.5, 2.2))):
    tri = tri_of(m)
    return ax.tricontour(tri, phi, levels=levels, colors=[color], linewidths=lw, zorder=z, linestyles=[ls])


def arrows_on_contours(ax, cs, m, Jfield, color, frac=0.5, size=7, every=1, lw=1.0, min_len=0.25):
    """Arrow heads on contour paths, oriented along the (element-wise) vector field."""
    from matplotlib.tri import TriFinder  # noqa: F401
    tri = tri_of(m)
    finder = tri.get_trifinder()
    for lev_paths in cs.allsegs:
        for k, path in enumerate(lev_paths):
            if k % every:
                continue
            if len(path) < 5:
                continue
            d = np.hypot(*np.diff(path, axis=0).T)
            if d.sum() < min_len:
                continue
            s = np.concatenate(([0], np.cumsum(d)))
            i = int(np.searchsorted(s, frac * s[-1]))
            i = min(max(i, 1), len(path) - 2)
            q = path[i]
            e = finder(q[0], q[1])
            if e < 0:
                continue
            v = Jfield[e]
            nv = np.hypot(*v)
            if nv == 0:
                continue
            v = v / nv * 0.012
            arrow(ax, tuple(q - v), tuple(q + v), color=color, lw=lw, ms=size, z=5)


# ---------------------------------------------------------------------------
@figure("tes_setup")
def tes_setup():
    n = 128
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.25))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.11, wspace=0.08)
    for ax, inc in zip(axs, (False, True)):
        m = model(n, inclusion=inc)
        sol = tes_uniform(m)
        psi = stream_tes(m)
        square_frame(ax, inclusion=inc)
        lv = np.linspace(0.0, 1.0, 12)[1:-1]
        cs = psi_lines(ax, m, psi, lv, color=CURR, lw=1.15)
        arrows_on_contours(ax, cs, m, sol["J"], CURR, frac=0.5, size=7.5, lw=1.0)
        ph = sol["phi"]
        levs = np.arange(-2.0, 2.01, 0.125)
        levs = levs[(levs > ph.min()) & (levs < ph.max())]
        phi_lines(ax, m, ph, levs, color=POT, lw=0.85)
        electrodes(ax)
        if inc:
            halo(label(ax, INC_C[0], INC_C[1] - 0.02, r"$\sigma=10$", size=8.6, color=INK), 2.5)
        label(ax, 0.06, 0.06, r"$\sigma=1$", size=8.6, color=INK, ha="left")
        title = T("均匀导体", "homogeneous conductor") if not inc else T("含高电导率夹杂", "with a high-conductivity inclusion")
        panel_title(ax, title, y=1.075)
    # shared legend under the panels
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CURR, lw=1.3), Line2D([], [], color=POT, lw=0.9, ls=(0, (3.5, 2.2))),
               Rectangle((0, 0), 1, 1, fc=ELECTRODE_GREY, ec="none")]
    labels = [T("电流线（相邻两线间电流相等）", "current lines (equal current between lines)"),
              T("等势线", "equipotentials"), T("电极：+ 注入，− 引出", "electrodes: + in, − out")]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.4, handlelength=2.0,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.4)
    return fig


# ---------------------------------------------------------------------------
def _electrode_sketch(ax, kind):
    """Small cross-section: metal electrode (grey), optional contact layer, tissue (sand)."""
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.0, 0.0), 3.0, 1.0, fc=MATL_T, ec="none", zorder=0))
    ax.plot([0, 3.0], [1.0, 1.0], color=MATL, lw=0.8, zorder=1)
    x0, x1 = 0.75, 2.25
    if kind == "cem":
        ax.add_patch(Rectangle((x0, 1.0), x1 - x0, 0.16, fc=tint(AMBER, 0.35), ec="none", zorder=2))
        ax.add_patch(Rectangle((x0, 1.16), x1 - x0, 0.2, fc=ELECTRODE_GREY, ec="none", zorder=2))
    else:
        ax.add_patch(Rectangle((x0, 1.0), x1 - x0, 0.22, fc=ELECTRODE_GREY, ec="none", zorder=2))
    # current arrows into the tissue: length ~ local density
    xs = np.linspace(x0 + 0.1, x1 - 0.1, 7)
    u = (xs - (x0 + x1) / 2) / ((x1 - x0) / 2)
    if kind == "dirichlet":
        L = 0.28 + 0.38 * np.abs(u) ** 6
    elif kind == "uniform":
        L = np.full_like(xs, 0.40)
    else:
        L = 0.33 + 0.12 * np.abs(u) ** 6
    for x, l in zip(xs, L):
        arrow(ax, (x, 0.98), (x, 0.98 - l), color=CURR, lw=1.3, ms=7, z=4)
    return x0, x1


CEM_Z = (0.02, 0.2)            # contact impedances used in the figure (normalised; w = 0.25)
CEM_COL = ("#B7791F", "#D9A441")


def electrode_profiles():
    """Data of the bottom panel: j_in/(I/w) under the anode for the three models."""
    w = ANODE[1] - ANODE[0]
    out = {}
    for n in (64, 256):
        d = tes_dirichlet(model(n))
        out[f"dir{n}"] = (d["x_anode"], d["j_anode"] * w)
    m = model(128)
    for z in CEM_Z:
        r = tes_cem(m, z)
        out[f"cem{z}"] = (r["x_anode"], r["j_anode"] * w)
    return out


@figure("electrode_models")
def electrode_models():
    fig = plt.figure(figsize=(FIG_W, 3.75))
    tops = [fig.add_axes([0.015 + k * 0.33, 0.645, 0.30, 0.27]) for k in range(3)]
    kinds = ("dirichlet", "uniform", "cem")
    titles = [T("① 指定电势（等势电极）", "① prescribed potential"),
              T("② 指定电流密度", "② prescribed current density"),
              T("③ 完整电极模型（CEM）", "③ complete electrode model")]
    eqs = [T(r"电极下 $\phi=U_l$（常数）", r"$\phi=U_l$ (constant) under it"),
           T(r"$j_{\rm in}=I_l/|\Gamma_l|$（均匀）", r"$j_{\rm in}=I_l/|\Gamma_l|$ (uniform)"),
           r"$U_l-\phi=z_l\,j_{\rm in}$,  $\int_{\Gamma_l}\,\,j_{\rm in}\,\mathrm{d}S=I_l$"]
    cols = [INK, CURR, CEM_COL[0]]
    for ax, kind, ttl, eq, c in zip(tops, kinds, titles, eqs, cols):
        _electrode_sketch(ax, kind)
        label(ax, 1.5, 1.8, ttl, size=8.9, weight="bold", color=c)
        label(ax, 1.5, 0.27, eq, size=8.3, color=INK)
        if kind == "cem":
            label(ax, 2.32, 1.08, T("接触层 $z_l$", "contact $z_l$"), size=7.8, color=CEM_COL[0], ha="left")
        label(ax, 0.72, 1.16 if kind != "cem" else 1.27, T("金属", "metal"), size=7.6, color=MUTED, ha="right")
        label(ax, 0.06, 0.82, T("组织", "tissue"), size=7.6, color=MUTED, ha="left")
    # ---- computed current density under the anode
    prof = electrode_profiles()
    bx = fig.add_axes([0.105, 0.095, 0.86, 0.44])
    x64, j64 = prof["dir64"]
    x256, j256 = prof["dir256"]
    bx.plot(x256, j256, color=INK, lw=1.7, zorder=5)
    bx.plot(x64, j64, color=INK, lw=1.2, ls=(0, (1.2, 1.6)), zorder=5)
    for z, c, ls in zip(CEM_Z, CEM_COL, ("-", (0, (5, 2)))):
        xx, jj = prof[f"cem{z}"]
        bx.plot(xx, jj, color=c, lw=1.9, ls=ls, zorder=4)
    bx.plot([ANODE[0], ANODE[1]], [1, 1], color=CURR, lw=2.3, zorder=3)
    ytop = 3.3
    bx.set_xlim(ANODE[0] - 0.006, ANODE[1] + 0.006)
    bx.set_ylim(0, ytop)
    bx.set_xticks([ANODE[0], (ANODE[0] + ANODE[1]) / 2, ANODE[1]],
                  [T("左边缘", "left edge"), T("中心", "center"), T("右边缘（朝向阴极）", "right edge (towards cathode)")])
    bx.set_yticks([0, 1, 2, 3])
    bx.set_ylabel(T(r"$j_{\rm in}\,/\,(I/w)$", r"$j_{\rm in}\,/\,(I/w)$"))
    bx.set_title(T("同一总电流 $I$：阳极下的电流密度（计算值，$w$ 为电极宽度）",
                   "same total current $I$: computed density under the anode ($w$ = electrode width)"), fontsize=9.0, pad=5)
    clean_axes(bx, grid=True)
    # direct labels
    xm = ANODE[0] + 0.30 * (ANODE[1] - ANODE[0])
    halo(label(bx, xm, 1.13, T("② 均匀：处处为 1", "② uniform: 1 everywhere"), color=CURR, size=8.5, ha="center"), 2.5)
    halo(label(bx, xm, 0.43, T("① 等势电极：中间少、边缘多", "① equipotential: low in the middle, crowded at edges"),
               color=INK, size=8.5, ha="center"), 2.5)
    xr = ANODE[1] - 0.052
    halo(label(bx, xr, 2.55, T(r"③ $z\sigma/w=0.08$", r"③ $z\sigma/w=0.08$"), color=CEM_COL[0], size=8.5, ha="right"), 2.5)
    halo(label(bx, xr + 0.006, 1.53, T(r"③ $z\sigma/w=0.8$", r"③ $z\sigma/w=0.8$"), color="#B07A1E", size=8.5, ha="right"), 2.5)
    # clipped peaks of the equipotential model grow under refinement
    pk = (j64[-1], j256[-1], j64[0], j256[0])
    note = T(f"① 的边缘峰值超出坐标：右缘 粗网格 {pk[0]:.1f}，细网格 {pk[1]:.0f}；\n"
             f"左缘 {pk[2]:.1f} → {pk[3]:.1f}。网格每加密一次就再增大，CEM 的峰值则不变",
             f"① edge peaks are off scale: right edge {pk[0]:.1f} (coarse) → {pk[1]:.0f} (fine),\n"
             f"left edge {pk[2]:.1f} → {pk[3]:.1f}; they keep growing under refinement, CEM peaks do not")
    label(bx, (ANODE[0] + ANODE[1]) / 2, ytop - 0.33, note, size=8.0, color=MUTED, ha="center")
    from matplotlib.lines import Line2D
    bx.legend([Line2D([], [], color=INK, lw=1.7), Line2D([], [], color=INK, lw=1.2, ls=(0, (1.2, 1.6)))],
              [T("① 细网格（256）", "① fine mesh (256)"), T("① 粗网格（64）", "① coarse mesh (64)")],
              loc="center left", bbox_to_anchor=(0.03, 0.62), fontsize=8.0, handlelength=2.0, frameon=False)
    return fig
