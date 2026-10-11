"""Chapter 15 (permanent magnets, yokes and MRI magnets): models and checks for every number in
zh/ch15_magnets.tex.

    python3 checks/check_ch15.py          # run all assertions, print the quoted values

The figure module figsrc/fig_ch15.py imports the model functions from this file.  Only NumPy,
SciPy and the book's numerics.py are used; the nonlinear iron is the synthetic curve
eq:syntheticbh through checks/check_ch13.py.  Geometry and material values are teaching inputs.

Contents
  * input conventions of a linear permanent magnet (remanence / magnetisation / coercive form);
  * demagnetising factors of spheroids (closed forms of J. A. Osborn, Phys. Rev. 67, 351 (1945),
    checked against the defining elliptic integral by quadrature) and the effective response;
  * a planar (2-D, per unit depth) magnetostatic model of a two-pillar magnet-yoke cross-section:
    A_z formulation, P1 elements on a graded tensor-product mesh (numerics.stiffness for the
    geometry), permanent-magnet source from H = nu_rec (B - B_r), nonlinear iron solved by Newton's
    method with the consistent incremental tangent (eq:bh-inc / eq:inctensor) and a backtracking
    line search on the magnetic energy; quarter model with symmetry conditions, checked against the
    full cross-section, a finer mesh and a larger air box;
  * the lumped network of Ch 14 applied to the same cross-section, for comparison;
  * magnetic pressure B^2/(2 mu0) and the Maxwell stress tensor.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from numerics import MU0, stiffness
import check_ch13 as BH

BR, MUREC = BH.BR, BH.MUREC          # eq:recoil teaching values: 1.2 T, 1.05


# ---------------------------------------------------------------------------
# 15.1  Input conventions of the same linear magnet
# ---------------------------------------------------------------------------
def pm_conventions(Br=BR, mur=MUREC):
    Mr = Br / MU0                    # remanent magnetisation, magnetisation form  B = mu0 (H + M_r e + chi_rec H)
    Hc = Br / (MU0 * mur)            # coercive form  B = mu0 mu_rec (H + H_c e)
    return dict(Br=Br, Mr=Mr, Hc=Hc, chi_rec=mur - 1.0, mix_error=Mr / Hc - 1.0)


# ---------------------------------------------------------------------------
# 15.2  Demagnetising factors of spheroids (Osborn 1945), SI convention N_a + N_b + N_c = 1
# ---------------------------------------------------------------------------
def N_axial(m):
    """Demagnetising factor along the symmetry axis of a spheroid with aspect ratio m = c/a
    (c: semi-axis along the symmetry axis, a = b: equatorial semi-axes)."""
    m = float(m)
    if abs(m - 1.0) < 1e-12:
        return 1.0 / 3.0
    if m > 1.0:                       # prolate
        r = np.sqrt(m * m - 1.0)
        return (m / r * np.log(m + r) - 1.0) / (m * m - 1.0)
    r = np.sqrt(1.0 - m * m)          # oblate
    return (1.0 - m / r * np.arccos(m)) / (1.0 - m * m)


def N_transverse(m):
    return 0.5 * (1.0 - N_axial(m))


def N_quadrature(a, b, c, i):
    """Defining integral N_i = (abc/2) int_0^inf ds / ((s + a_i^2) sqrt((s+a^2)(s+b^2)(s+c^2)))."""
    ai = (a, b, c)[i]
    f = lambda s: 1.0 / ((s + ai * ai) * np.sqrt((s + a * a) * (s + b * b) * (s + c * c)))
    return a * b * c / 2.0 * quad(f, 0.0, np.inf, limit=400, epsabs=0, epsrel=1e-12)[0]


def chi_eff(chi, N):
    """eq:demag with M = chi H_int:  M = chi/(1 + N chi) H_applied."""
    return chi / (1.0 + N * chi)


def isolated_magnet(m, Br=BR, mur=MUREC):
    """Uniformly magnetised spheroidal magnet in free space, magnetised along its axis.
    H_int = -N M and B = mu0 (H + M) give the shape load line B = -mu0 P H, P = (1 - N)/N;
    its intersection with eq:recoil is the operating point."""
    N = N_axial(m)
    P = (1.0 - N) / N
    Bm = Br / (1.0 + mur / P)
    Hm = -Bm / (MU0 * P)
    return dict(N=N, P=P, Bm=Bm, Hm=Hm)


# ---------------------------------------------------------------------------
# 15.3  Planar magnet-yoke cross-section: geometry (teaching inputs, metres)
# ---------------------------------------------------------------------------
MAGNET = dict(g=0.40,        # gap between the pole faces
              tpp=0.03,      # pole-piece thickness
              lm=0.04,       # magnet thickness (each of two magnets), magnetised along +y
              tpl=0.05,      # yoke-plate thickness
              a=0.30,        # half-width of magnets and pole pieces
              xp=0.50,       # inner face of each pillar
              wp=0.05,       # pillar width
              Br=BR, mur=MUREC)
ROI = 0.10                   # radius of the "imaging region" used for the uniformity figure


def levels(G=MAGNET):
    y1 = G["g"] / 2
    y2 = y1 + G["tpp"]
    y3 = y2 + G["lm"]
    y4 = y3 + G["tpl"]
    return y1, y2, y3, y4


def _graded(breaks, hfun, nsub=800):
    pts = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        s = np.linspace(a, b, nsub + 1)
        inv = 1.0 / hfun(s)
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))))
        n = max(1, int(np.ceil(cum[-1])))
        pts.extend(np.interp(np.linspace(0, cum[-1], n + 1)[1:], cum, s))
    return np.array(pts)


def _tensor_mesh(xs, ys):
    nx = len(xs)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    p = np.column_stack((X.ravel(), Y.ravel()))
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(len(ys) - 1), indexing="xy")
    a = (j * nx + i).ravel()
    cells = np.vstack((np.column_stack((a, a + 1, a + nx + 1)), np.column_stack((a, a + nx + 1, a + nx))))
    return p, cells


def _w_iron(b):
    """Reversible magnetic energy density w(b) = int_0^b h db' = b h - co-energy, synthetic curve."""
    h = BH.H_of_B(b)
    x = np.abs(h / BH.HS)
    lncosh = x + np.log1p(np.exp(-2 * x)) - np.log(2.0)
    return b * h - MU0 * h ** 2 / 2 - BH.BS * BH.HS * lncosh


def magnet_fem(box=1.5, h0=2.5e-3, grow=0.18, hmax=0.06, hin=0.01, full=False, G=MAGNET, tol=1e-11, maxit=60):
    """Solve -div(nu grad A_z) = curl-source of the magnets on the cross-section.

    Quarter model (x >= 0, y >= 0): A_z = 0 on x = 0 (n.B = 0: the magnetisation is parallel to that
    plane) and on the outer box; natural condition (n x H = 0) on y = 0, the mid-plane normal to the
    magnetisation.  full=True solves the whole cross-section with A_z = 0 on the outer box only.
    """
    y1, y2, y3, y4 = levels(G)
    a, xp = G["a"], G["xp"]
    xq = xp + G["wp"]
    kx, ky = np.array([a, xp, xq]), np.array([y1, y2, y3, y4])

    def hx(x):
        h = np.minimum(h0 + grow * np.min(np.abs(x[:, None] - kx[None, :]), axis=1), hmax)
        return np.where(x <= xq + 0.02, np.minimum(h, hin), h)        # at most hin over the device

    def hy(y):
        h = np.minimum(h0 + grow * np.min(np.abs(y[:, None] - ky[None, :]), axis=1), hmax)
        return np.where(y <= y4 + 0.02, np.minimum(h, hin), h)

    xs = _graded([0.0, a, xp, xq, box], hx)
    ys = _graded([0.0, y1, y2, y3, y4, box], hy)
    if full:
        xs = np.concatenate((-xs[::-1][:-1], xs))
        ys = np.concatenate((-ys[::-1][:-1], ys))
    p, cells = _tensor_mesh(xs, ys)
    c = p[cells].mean(axis=1)
    ax_, ay_ = np.abs(c[:, 0]), np.abs(c[:, 1])
    pole = (ax_ < a) & (ay_ > y1) & (ay_ < y2)
    mag = (ax_ < a) & (ay_ > y2) & (ay_ < y3)
    plate = (ax_ < xq) & (ay_ > y3) & (ay_ < y4)
    pillar = (ax_ > xp) & (ax_ < xq) & (ay_ < y3)
    iron = pole | plate | pillar
    nu_m = 1.0 / (MU0 * G["mur"])
    nulin = np.where(mag, nu_m, 1.0 / MU0)
    _, area, grads = stiffness(p, cells, np.ones(len(cells)))
    # magnet source: int nu_m (curl w) . B_r = -int nu_m B_r d_x w   for B_r = B_r e_y
    f = np.zeros(len(p))
    loc = -(nu_m * G["Br"] * area)[:, None] * grads[:, :, 0]
    loc[~mag] = 0.0
    np.add.at(f, cells.ravel(), loc.ravel())
    bnd = np.isclose(p[:, 0], xs[-1]) | np.isclose(p[:, 1], ys[-1]) | np.isclose(p[:, 0], xs[0])
    if full:
        bnd |= np.isclose(p[:, 1], ys[0])
    free = np.flatnonzero(~bnd)
    rows = np.repeat(cells, 3, axis=1).ravel()
    cols = np.tile(cells, (1, 3)).ravel()

    def state(A):
        gr = np.einsum("ei,eid->ed", A[cells], grads)
        b = np.hypot(gr[:, 0], gr[:, 1])
        ns = np.where(iron, BH.nu_sec(b), nulin)
        nd = np.where(iron, 1.0 / BH.mu_diff(BH.H_of_B(b)), nulin)
        return gr, b, ns, nd

    def energy(A):
        _, b, _, _ = state(A)
        return float(np.sum(area * np.where(iron, _w_iron(b), 0.5 * nulin * b * b)) - f @ A)

    A = np.zeros(len(p))
    hist = []
    for it in range(maxit):
        gr, b, ns, nd = state(A)
        r = np.zeros(len(p))
        np.add.at(r, cells.ravel(), ((area * ns)[:, None] * np.einsum("eid,ed->ei", grads, gr)).ravel())
        r -= f
        rn = float(np.linalg.norm(r[free]) / np.linalg.norm(f[free]))
        hist.append(rn)
        if rn < tol:
            break
        e = gr / np.maximum(b, 1e-300)[:, None]
        # consistent tangent in the gradient frame: nu_sec I + (nu_diff - nu_sec) e e^T  (2-D form of eq:bh-inc)
        Tn = ns[:, None, None] * np.eye(2)[None] + (nd - ns)[:, None, None] * np.einsum("ei,ej->eij", e, e)
        Ke = area[:, None, None] * np.einsum("eid,edk,ejk->eij", grads, Tn, grads)
        K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(len(p), len(p))).tocsr()
        d = np.zeros(len(p))
        d[free] = spsolve(K[free][:, free].tocsc(), -r[free])
        E0, slope, t = energy(A), float(r[free] @ d[free]), 1.0
        while energy(A + t * d) > E0 + 1e-4 * t * slope and t > 1e-8:
            t *= 0.5
        A = A + t * d
    gr, b, ns, nd = state(A)
    return dict(p=p, cells=cells, A=A, xs=xs, ys=ys, area=area, grads=grads, mag=mag, iron=iron, pole=pole,
                plate=plate, pillar=pillar, Bx=gr[:, 1], By=-gr[:, 0], Bmag=b, nu_m=nu_m, it=it, hist=hist,
                full=full, G=G)


def nodal_B(s):
    """Nodal recovery on the tensor grid: B_x = dA/dy, B_y = -dA/dx (second-order differences).

    For the quarter model the grid is first mirrored with the symmetry of the solution,
    A(-x, y) = -A(x, y) and A(x, -y) = A(x, y), so that the differences on the symmetry lines
    are central rather than one-sided.  Returns arrays on the original grid.
    """
    xs, ys = s["xs"], s["ys"]
    Ag = s["A"].reshape(len(ys), len(xs))
    if s["full"]:
        Bx = np.gradient(Ag, ys, axis=0, edge_order=2)
        By = -np.gradient(Ag, xs, axis=1, edge_order=2)
        return Ag, Bx, By
    X = np.concatenate((-xs[:0:-1], xs))
    Y = np.concatenate((-ys[:0:-1], ys))
    Ax = np.concatenate((-Ag[:, :0:-1], Ag), axis=1)
    Am = np.concatenate((Ax[:0:-1, :], Ax), axis=0)
    Bx = np.gradient(Am, Y, axis=0, edge_order=2)[len(ys) - 1:, len(xs) - 1:]
    By = -np.gradient(Am, X, axis=1, edge_order=2)[len(ys) - 1:, len(xs) - 1:]
    return Ag, Bx, By


def magnet_outputs(s):
    G = s["G"]
    y1, y2, y3, y4 = levels(G)
    a, xp, wp = G["a"], G["xp"], G["wp"]
    xq = xp + wp
    xs, ys = s["xs"], s["ys"]
    Ag, Bxg, Byg = nodal_B(s)
    ix = lambda x: int(np.argmin(np.abs(xs - x)))
    iy = lambda y: int(np.argmin(np.abs(ys - y)))
    j0, i0 = iy(0.0), ix(0.0)
    Aat = lambda x, y: Ag[iy(y), ix(x)]
    ym = 0.5 * (y2 + y3)
    out = {}
    out["Bc"] = float(Byg[j0, i0])
    # fluxes per metre depth through horizontal segments (x >= 0 half): Phi = A(x1) - A(x2)
    out["phi_m"] = float(Aat(0, ym) - Aat(a, ym))
    out["phi_g"] = float(Aat(0, 0) - Aat(a, 0))
    out["phi_mid"] = float(Aat(0, 0) - Aat(xp, 0))
    out["phi_p"] = float(Aat(xq, 0) - Aat(xp, 0))
    out["phi_out"] = float(Aat(xs[-1], 0) - Aat(xq, 0))      # returning (downward) outside the pillars
    out["gap_share"] = out["phi_g"] / out["phi_m"]
    out["mid_share"] = out["phi_mid"] / out["phi_m"]
    out["stray_share"] = out["phi_out"] / out["phi_m"]
    out["Bm_mean"] = out["phi_m"] / a
    out["Bg_mean"] = out["phi_g"] / a
    out["Bp_mid"] = out["phi_p"] / wp
    # horizontal flux in the plate through the vertical cut x = xp (just inside the pillar)
    out["phi_plate"] = float(Aat(xp, y4) - Aat(xp, y3))
    out["Bplate_cut"] = out["phi_plate"] / G["tpl"]
    out["Bplate_edge"] = float(Aat(a, y4) - Aat(a, y3)) / G["tpl"]       # mean plate B above the magnet edge
    # uniformity: B_y on the mid-plane |x| <= ROI and on a disc of radius ROI (nodes)
    sel = (xs >= 0) & (xs <= ROI + 1e-12)
    prof = Byg[j0, sel]
    out["dB_line"] = float((prof.max() - prof.min()) / out["Bc"])
    X, Y = np.meshgrid(xs, ys)
    disc = (X >= 0) & (Y >= 0) & (np.hypot(X, Y) <= ROI + 1e-12)
    Bd = np.hypot(Bxg, Byg)[disc]
    out["dB_disc"] = float((Bd.max() - Bd.min()) / out["Bc"])
    # iron: area above 1.5 T (quarter) and the share of iron area
    c = s["p"][s["cells"]].mean(axis=1)
    q = (c[:, 0] > 0) & (c[:, 1] > 0)
    ir = s["iron"] & q
    out["A_sat"] = float(s["area"][ir & (s["Bmag"] > 1.5)].sum())
    out["A_iron"] = float(s["area"][ir].sum())
    out["sat_share"] = out["A_sat"] / out["A_iron"]
    # magnet: H_y = nu_m (B_y - B_r); mid-height profile and area beyond thresholds
    Hy = s["nu_m"] * (s["By"] - G["Br"])
    m = s["mag"] & q
    jm = iy(ym)
    Hrow = s["nu_m"] * (Byg[jm] - G["Br"])
    inside = (xs >= 0) & (xs < a - 1e-9)
    out["Hm_centre"] = float(Hrow[i0])
    out["Hm_row_min"] = float(Hrow[inside].min())
    out["Hm_row_max"] = float(Hrow[inside].max())
    out["Hm_mean"] = float(np.sum(Hy[m] * s["area"][m]) / s["area"][m].sum())
    out["Hm_min_elem"] = float(Hy[m].min())
    for thr in (700e3, 800e3, 909e3):
        out[f"mag_beyond_{int(thr / 1e3)}"] = float(s["area"][m & (Hy < -thr)].sum() / s["area"][m].sum())
    out["B_far"] = float(np.hypot(Bxg[j0, ix(1.0)], Byg[j0, ix(1.0)]))
    out["B_edge_ratio"] = float(Byg[j0, ix(a)] / out["Bc"])
    return out


def magnet_network(G=MAGNET, Np=2):
    """Ch 14 network on the same cross-section (per metre depth): two Thevenin magnets, gap reluctance,
    ideal pole pieces and plates, Np nonlinear pillars in parallel (synthetic curve), no leakage."""
    y1, y2, y3, y4 = levels(G)
    Br, mur, lm, g = G["Br"], G["mur"], G["lm"], G["g"]
    Am = 2 * G["a"]
    Ap = Np * G["wp"]
    lp = 2 * (y3 + G["tpl"] / 2)                 # pillar length between plate mid-lines

    def res(Hp):
        Phi = Ap * float(BH.B_of_H(Hp))
        Hm = (Phi / Am - Br) / (MU0 * mur)
        return 2 * Hm * lm + Phi / Am * g / MU0 + Hp * lp

    Hp = brentq(res, 0.0, 1e8, xtol=1e-10, rtol=1e-14)
    Phi = Ap * float(BH.B_of_H(Hp))
    Bg = Phi / Am
    return dict(Bg=Bg, Bp=float(BH.B_of_H(Hp)), Hm=(Bg - Br) / (MU0 * mur), ideal=2 * Br * lm / (2 * lm + mur * g),
                lp=lp, Phi=Phi, P=2 * lm / g)


_CACHE: dict = {}


def magnet_case(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        s = magnet_fem(**kw)
        _CACHE[key] = (s, magnet_outputs(s))
    return _CACHE[key]


# ---------------------------------------------------------------------------
# 15.3  Symmetry conditions: main field versus a transverse gradient field
# ---------------------------------------------------------------------------
def gradient_Az(x, y, Gx=1.0):
    """In the cross-section (x horizontal, y along the main field) a transverse gradient
    b = G (y, x) = (dA/dy, -dA/dx) has A_z = G (y^2 - x^2)/2."""
    return Gx * (y ** 2 - x ** 2) / 2


# ---------------------------------------------------------------------------
# 15.4  Maxwell stress
# ---------------------------------------------------------------------------
def pressure(B):
    return B ** 2 / (2 * MU0)


def stress_tensor(B):
    B = np.asarray(B, float)
    return (np.outer(B, B) - 0.5 * B @ B * np.eye(len(B))) / MU0


def midplane_force(s):
    """Attraction between the upper and lower halves of the cross-section (N per metre depth).

    The closed surface around the upper half is the mid-plane y = 0 plus a far arc where the field has
    decayed.  On y = 0 the symmetry gives B_x = 0, so T_yy = B_y^2 / (2 mu0) (tension).  Where the plane
    cuts the pillars it is read as a vanishingly thin air joint: B_y (normal) is continuous, the
    tangential B in the joint is mu0 H_t, negligible.  B_y is taken from the element row on y = 0.
    """
    G = s["G"]
    xp, xq, a = G["xp"], G["xp"] + G["wp"], G["a"]
    c = s["p"][s["cells"]].mean(axis=1)
    row = (c[:, 1] > 0) & (c[:, 1] < s["ys"][1])
    xc, By = c[row, 0], s["By"][row]
    xs = s["xs"]
    i = np.clip(np.searchsorted(xs, xc) - 1, 0, len(xs) - 2)
    dx = xs[i + 1] - xs[i]
    # each grid cell holds two triangles: average their B_y^2 and weight by the cell width once
    t = By ** 2 / (2 * MU0) * dx / 2
    gap = 2 * float(t[xc < xp].sum())
    pil = 2 * float(t[(xc > xp) & (xc < xq)].sum())
    out = 2 * float(t[xc > xq].sum())
    pole = 2 * float(t[xc < a].sum())
    Bc = float(s["By"][row][np.argmin(xc)])
    return dict(gap=gap, pillars=pil, outside=out, pole=pole, uniform=pressure(Bc) * 2 * a, total=gap + pil + out)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
quoted = BH.quoted


def run_checks(verbose=True):
    log = []

    def Q(name, value, text):
        quoted(value, text, name)
        log.append(f"{name:<56s} {value: .6g}  ~ {text}")

    def LT(name, value, bound):
        assert value < bound, f"{name}: {value} is not below {bound}"
        log.append(f"{name:<56s} {value: .6g}  < {bound}")

    # --- 15.1 conventions ---
    c = pm_conventions()
    Q("B_r [T]", c["Br"], "1.2")
    Q("M_r = B_r/mu0 [A/m]", c["Mr"], "9.55e5")
    Q("H_c = B_r/(mu0 mu_rec) [A/m]", c["Hc"], "9.09e5")
    Q("chi_rec", c["chi_rec"], "0.05")
    Q("error when M_r is entered as H_c [%]", 100 * c["mix_error"], "5.0")

    # --- 15.2 demagnetising factors ---
    for m in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
        Nc = N_axial(m)
        assert abs(Nc - N_quadrature(1.0, 1.0, m, 2)) < 1e-8 * max(1.0, 1 / Nc), m
        assert abs(N_transverse(m) - N_quadrature(1.0, 1.0, m, 0)) < 1e-8, m
        assert abs(Nc + 2 * N_transverse(m) - 1.0) < 1e-12
    Q("N sphere", N_axial(1.0), "0.333")
    Q("N prolate c/a = 10", N_axial(10.0), "0.0203")
    Q("N prolate c/a = 100", N_axial(100.0), "4.3e-4")
    Q("N oblate c/a = 0.1", N_axial(0.1), "0.861")
    Q("N oblate c/a = 0.01", N_axial(0.01), "0.984")
    chi = 999.0
    Q("iron sphere chi_eff", chi_eff(chi, 1 / 3), "2.99")
    Q("iron sphere B_in/B_0 = mu_r/(1+N chi)", 1000.0 / (1 + chi / 3), "2.99")
    Q("iron rod c/a = 10 chi_eff", chi_eff(chi, N_axial(10.0)), "47.0")
    Q("iron rod c/a = 100 chi_eff", chi_eff(chi, N_axial(100.0)), "699")
    Q("iron flat c/a = 0.1 chi_eff", chi_eff(chi, N_axial(0.1)), "1.16")
    Q("self-check: sphere with mu_r = 1e4, B_in/B_0", 1e4 / (1 + (1e4 - 1) / 3), "3.00")
    Q("1/N rod c/a = 10", 1 / N_axial(10.0), "49.3")
    s1 = isolated_magnet(1.0)
    s01 = isolated_magnet(0.1)
    Q("isolated sphere magnet: P", s1["P"], "2.00")
    Q("isolated sphere magnet: B_m [T]", s1["Bm"], "0.787")
    Q("isolated sphere magnet: H_m [kA/m]", s1["Hm"] / 1e3, "-313")
    Q("sphere, mu_rec = 1: B_m/B_r = 2/3", isolated_magnet(1.0, mur=1.0)["Bm"] / BR, "0.667")
    Q("isolated flat magnet c/a = 0.1: P", s01["P"], "0.162")
    Q("isolated flat magnet c/a = 0.1: B_m [T]", s01["Bm"], "0.160")
    Q("isolated flat magnet c/a = 0.1: H_m [kA/m]", s01["Hm"] / 1e3, "-788")

    # --- 15.3 inputs quoted in the text (bind the prose to the model) ---
    assert MAGNET == dict(g=0.40, tpp=0.03, lm=0.04, tpl=0.05, a=0.30, xp=0.50, wp=0.05, Br=1.2, mur=1.05)
    assert ROI == 0.10
    Q("iron sphere: 1 + N chi", 1 + chi / 3, "334")
    Q("mu_r at the origin of the synthetic curve", float(BH.mu_diff(0.0)) / MU0, "1593")

    # --- 15.3 network on the 2-D cross-section ---
    nw = magnet_network()
    Q("network: ideal-yoke B_g [T]", nw["ideal"], "0.192")
    Q("network: B_g with nonlinear pillars [T]", nw["Bg"], "0.191")
    Q("network: pillar B [T]", nw["Bp"], "1.15")
    Q("network: magnet H [kA/m]", nw["Hm"] / 1e3, "-765")
    Q("network: pillar length [m]", nw["lp"], "0.59")
    Q("network: load-line slope P = 2 l_m / g", nw["P"], "0.2")

    # --- 15.3 field solution ---
    s, o = magnet_case()
    assert s["hist"][-1] < 1e-11 and s["it"] <= 15, (s["it"], s["hist"][-1])
    h = s["hist"]
    assert h[-1] < 10 * h[-2] ** 1.5          # quadratic convergence at the end
    Q("FEM: Newton iterations", s["it"], "12")
    Q("FEM: nodes of the quarter model [1e3]", len(s["p"]) / 1e3, "9")
    Q("FEM: mid-plane B_y at the pole edge / centre", o["B_edge_ratio"], "0.66")
    Q("FEM: B at the gap centre [T]", o["Bc"], "0.152")
    Q("FEM vs network: B_c / B_g,network", o["Bc"] / nw["Bg"], "0.80")
    Q("FEM: mean B over the pole width at mid-plane [T]", o["Bg_mean"], "0.139")
    Q("FEM: mean B in the magnet [T]", o["Bm_mean"], "0.305")
    Q("FEM: gap flux within pole width / magnet flux [%]", 100 * o["gap_share"], "46")
    Q("FEM: flux crossing mid-plane inside pillars / magnet flux [%]", 100 * o["mid_share"], "56")
    Q("FEM: stray flux returning outside the pillars / magnet flux [%]", 100 * o["stray_share"], "0.9")
    Q("FEM: mean pillar B at mid-plane [T]", o["Bp_mid"], "1.01")
    Q("FEM: mean plate B above the magnet edge [T]", o["Bplate_edge"], "1.73")
    Q("FEM: mean plate B where it meets the pillar [T]", o["Bplate_cut"], "1.15")
    Q("FEM: iron area above 1.5 T [%]", 100 * o["sat_share"], "8")
    Q("FEM: B_y variation on |x| <= 0.1 m [%]", 100 * o["dB_line"], "1.5")
    Q("FEM: |B| variation over the disc r <= 0.1 m [%]", 100 * o["dB_disc"], "2.5")
    Q("FEM: magnet H at centre, mid-height [kA/m]", o["Hm_centre"] / 1e3, "-684")
    Q("FEM: magnet H, area mean [kA/m]", o["Hm_mean"] / 1e3, "-681")
    Q("FEM: magnet area with H < -700 kA/m [%]", 100 * o["mag_beyond_700"], "2")
    Q("FEM: magnet area with H < -909 kA/m [%]", 100 * o["mag_beyond_909"], "0.1")
    Q("FEM: |B| on the mid-plane 1 m from the centre, box 1.5 m [mT]", 1e3 * o["B_far"], "0.97")

    # --- convergence of the field solution: finer mesh, larger boxes, full cross-section ---
    sf, of = magnet_case(h0=1.25e-3, hmax=0.03, hin=0.005)
    sb, ob = magnet_case(box=3.0)
    s6, o6 = magnet_case(box=6.0)
    sF, oF = magnet_case(full=True)
    for k, tol_ in (("Bc", 0.01), ("gap_share", 0.01), ("Bm_mean", 0.01), ("Bplate_cut", 0.01), ("Bplate_edge", 0.01),
                    ("Hm_centre", 0.01), ("dB_line", 0.03), ("dB_disc", 0.03)):
        assert abs(of[k] / o[k] - 1) < tol_, (k, o[k], of[k])
        assert abs(o6[k] / o[k] - 1) < tol_, (k, o[k], o6[k])
    LT("refinement: change of B_c [%]", 100 * abs(of["Bc"] / o["Bc"] - 1), 0.1)
    LT("box 1.5 -> 6 m: change of B_c [%]", 100 * abs(o6["Bc"] / o["Bc"] - 1), 0.1)
    LT("quarter vs full model: change of B_c [%]", 100 * abs(oF["Bc"] / o["Bc"] - 1), 0.01)
    Q("refinement: B_y variation [%]", 100 * of["dB_line"], "1.5")
    Q("refinement: gap share [%]", 100 * of["gap_share"], "46")
    Q("box 3 m: |B| 1 m from the centre [mT]", 1e3 * ob["B_far"], "0.74")
    Q("box 6 m: |B| 1 m from the centre [mT]", 1e3 * o6["B_far"], "0.68")
    Q("box 6 m: stray share [%]", 100 * o6["stray_share"], "1.3")
    for k in ("Bc", "gap_share", "Bm_mean", "phi_p", "dB_line"):
        assert abs(oF[k] / o[k] - 1) < 2e-3, (k, o[k], oF[k])
    # mesh-dependent quantity, quoted only qualitatively: the corner value of H grows with refinement
    assert of["Hm_min_elem"] < o["Hm_min_elem"] < -1.2e6
    Q("corner H, base mesh [MA/m]", o["Hm_min_elem"] / 1e6, "-1.36")
    Q("corner H, finer mesh [MA/m]", of["Hm_min_elem"] / 1e6, "-1.57")
    # antisymmetry of the full solution: A_z vanishes on x = 0 (n.B = 0 there)
    xsF = sF["xs"]
    col = np.isclose(sF["p"][:, 0], xsF[np.argmin(np.abs(xsF))])
    assert np.max(np.abs(sF["A"][col])) < 1e-3 * np.max(np.abs(sF["A"]))
    # the gradient field: zero normal derivative of A_z on both symmetry lines
    eps = 1e-6
    for yv in (0.05, 0.2):
        assert abs((gradient_Az(eps, yv) - gradient_Az(-eps, yv)) / (2 * eps)) < 1e-9
        assert abs((gradient_Az(yv, eps) - gradient_Az(yv, -eps)) / (2 * eps)) < 1e-9

    # --- 15.4 Maxwell stress and magnetic pressure ---
    T = stress_tensor([0.0, 0.0, 0.3])
    assert np.allclose(np.diag(T), [-pressure(0.3), -pressure(0.3), pressure(0.3)])
    Q("pressure at 0.1 T [kPa]", pressure(0.1) / 1e3, "3.98")
    Q("pressure at 0.5 T [kPa]", pressure(0.5) / 1e3, "99.5")
    Q("pressure at 1.0 T [kPa]", pressure(1.0) / 1e3, "398")
    Q("pressure at 1.5 T [kPa]", pressure(1.5) / 1e3, "895")
    import check_ch14 as LM
    mp = LM.multipillar(4, 0.005)
    Q("Ch14 four-pillar example B_g [T]", mp["Bg"], "0.162")
    Q("pressure at that B_g [kPa]", pressure(mp["Bg"]) / 1e3, "10.5")
    Q("pole attraction p A_g [kN]", pressure(mp["Bg"]) * LM.PILLAR["Am"] / 1e3, "2.10")
    Q("... as a weight [kg]", pressure(mp["Bg"]) * LM.PILLAR["Am"] / 9.81, "214")
    Q("2-D model: pressure at B_c [kPa]", pressure(o["Bc"]) / 1e3, "9.2")
    F = midplane_force(s)
    Ff = midplane_force(sf)
    assert abs(Ff["total"] / F["total"] - 1) < 0.005 and abs(Ff["gap"] / F["gap"] - 1) < 0.005
    Q("2-D: attraction through the gap air [kN/m]", F["gap"] / 1e3, "5.2")
    Q("2-D: of which between the pole faces [kN/m]", F["pole"] / 1e3, "4.7")
    Q("2-D: uniform estimate p(B_c) x pole width [kN/m]", F["uniform"] / 1e3, "5.5")
    Q("2-D: uniform estimate / computed pole part", F["uniform"] / F["pole"], "1.18")
    Q("2-D: attraction through the two pillar joints [kN/m]", F["pillars"] / 1e3, "40.6")
    Q("2-D: pillars / gap", F["pillars"] / F["gap"], "7.8")
    Q("2-D: total attraction [kN/m]", F["total"] / 1e3, "45.8")
    LT("2-D: outside the pillars [N/m]", F["outside"], 1.0)
    Q("Ch14 four-pillar: pillar B [T]", mp["Bp"], "1.62")
    Q("Ch14 four-pillar: stress in a pillar joint [MPa]", pressure(mp["Bp"]) / 1e6, "1.05")
    Q("Ch14 four-pillar: four pillar joints [kN]", 4 * 0.005 * pressure(mp["Bp"]) / 1e3, "21.0")
    Q("self-check: pillars of double area, same flux [kN]", 4 * 0.010 * pressure(mp["Bp"] / 2) / 1e3, "10.5")
    Q("Ch14 four-pillar: joints / pole faces", 4 * 0.005 * pressure(mp["Bp"]) / (pressure(mp["Bg"]) * LM.PILLAR["Am"]), "10.0")

    if verbose:
        print("\n".join(log))
        print(f"\nall {len(log)} quoted values check out")
    return log


if __name__ == "__main__":
    run_checks()
