"""Chapter 21 (from field to neural response): models and checks for every number in zh/ch21_neuro.tex.

    python3 checks/check_ch21.py          # run all assertions, print the quoted values

The figure module figsrc/fig_ch21.py imports the model functions from this file, so the text,
the figures and the assertions share one implementation.  Only NumPy and SciPy are used.

Models (all from the old edition, app:23-27):
  * passive cable eq:cablephysical / eq:cablenormalized, solved on a path with the nodal
    (compartment) equation of app:26 with a passive ionic current and sealed ends;
  * the activating function eq:activating and the path form eq:curvatureAF;
  * the lumped passive membrane of app:27 (eq:exactsd) and the hyperbolic fit eq:lapicque.
All geometry is dimensionless (lengths in units of the length constant lambda, times in units
of tau_m, fields in units of a reference field E0).  These are teaching inputs, not nerve data.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.special import erf

LN2 = np.log(2.0)


def close(x, ref, rtol=5e-3, atol=0.0, name=""):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    return x


# ---------------------------------------------------------------------------
# 21.3  A localized longitudinal field on a straight axon (figure activating_function)
# ---------------------------------------------------------------------------
def gaussian_profile(s, ell=1.0, E0=1.0):
    """E_s = E0 exp(-s^2/(2 ell^2)) along a straight axon.

    Returns the path potential V_e (V_e(0) = 0, dV_e/ds = -E_s), E_s and the activating
    function f = d^2 V_e/ds^2 = -dE_s/ds (eq:activating)."""
    s = np.asarray(s, float)
    Es = E0 * np.exp(-s**2 / (2 * ell**2))
    Ve = -E0 * ell * np.sqrt(np.pi / 2) * erf(s / (np.sqrt(2) * ell))
    f = s / ell**2 * Es
    return Ve, Es, f


# ---------------------------------------------------------------------------
# Passive compartment cable on an arbitrary path (app:26 with I_ion = g V, sealed ends)
# ---------------------------------------------------------------------------
def neighbour_laplacian(n):
    """(D v)_k = sum over neighbours m of (v_m - v_k): the axial-coupling sum of app:26 for a
    chain whose end nodes have one neighbour only (sealed ends: no axial current leaves)."""
    D = np.zeros((n, n))
    for k in range(n - 1):
        D[k, k] -= 1.0
        D[k + 1, k + 1] -= 1.0
        D[k, k + 1] += 1.0
        D[k + 1, k] += 1.0
    return D


def passive_cable(Ve, ds, times=(), lam=1.0, tau=1.0):
    """Membrane voltage V_m on a chain of equal compartments of length ds.

    Nodal equation of app:26 divided by g_m ds:
        tau dV_n/dt = -V_n + (lam/ds)^2 sum_m [(V_m + Ve_m) - (V_n + Ve_n)],
    with V(0) = 0 and the extracellular drive Ve switched on at t = 0 (rectangular pulse).
    Returns (V_steady, [V(t) for t in times]).  Exact in time: the matrix is symmetric
    tridiagonal, so V(t) = Q (1 - exp(w t / tau)) Q^T V_steady with its eigenpairs (w, Q)."""
    Ve = np.asarray(Ve, float)
    n = len(Ve)
    c = (lam / ds) ** 2
    deg = np.full(n, 2.0)
    deg[0] = deg[-1] = 1.0                   # sealed ends: one neighbour
    diag = -c * deg - 1.0                    # A = c D - I
    off = np.full(n - 1, c)
    b = c * (np.concatenate(([0.0], Ve[:-1])) + np.concatenate((Ve[1:], [0.0])) - deg * Ve)   # c D Ve
    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1] = diag
    ab[2, :-1] = off
    Vinf = solve_banded((1, 1), ab, -b)
    out = []
    if len(times):
        w, Q = eigh_tridiagonal(diag, off)
        proj = Q.T @ Vinf
        for t in times:
            out.append(Q @ ((1.0 - np.exp(w * t / tau)) * proj))
    return Vinf, out


# ---------------------------------------------------------------------------
# 21.4  Bent axon with a terminal in a uniform field (figure curved_axon)
# ---------------------------------------------------------------------------
BENT = dict(L1=3.0, R=0.5, L2=6.0, E0=1.0)   # lengths in units of lambda; E = E0 e_x


def bent_path(ds=0.02, p=BENT):
    """Nodes of the path: terminal at (-L1, 0), straight along +x to the origin, a quarter
    circle of radius R turning to +y, then straight up for L2 (far end perpendicular to E).

    Node k sits at the centre of compartment k (arc length s_k = (k + 1/2) ds), so the sealed
    terminal lies half a compartment before node 0.  Returns s, x, y, tangent t, curvature
    normal n (unit, towards the centre of curvature; zero on straight parts), curvature kappa."""
    L1, R, L2 = p["L1"], p["R"], p["L2"]
    Ltot = L1 + 0.5 * np.pi * R + L2
    n = int(round(Ltot / ds))
    ds = Ltot / n
    s = (np.arange(n) + 0.5) * ds
    x, y = np.empty(n), np.empty(n)
    tx, ty = np.empty(n), np.empty(n)
    kap = np.zeros(n)
    nx, ny = np.zeros(n), np.zeros(n)
    s1, s2 = L1, L1 + 0.5 * np.pi * R
    a = s < s1
    x[a], y[a], tx[a], ty[a] = s[a] - L1, 0.0, 1.0, 0.0
    b = (s >= s1) & (s < s2)
    th = (s[b] - s1) / R
    x[b], y[b] = R * np.sin(th), R * (1 - np.cos(th))
    tx[b], ty[b] = np.cos(th), np.sin(th)
    kap[b] = 1.0 / R
    nx[b], ny[b] = -np.sin(th), np.cos(th)
    c = s >= s2
    x[c], y[c], tx[c], ty[c] = R, R + (s[c] - s2), 0.0, 1.0
    return dict(s=s, x=x, y=y, tx=tx, ty=ty, nx=nx, ny=ny, kappa=kap, ds=ds, s1=s1, s2=s2, L=Ltot)


def bent_response(ds=0.02, times=(0.1, 0.5), p=BENT):
    g = bent_path(ds, p)
    E0 = p["E0"]
    Es = E0 * g["tx"]                                  # E . t for E = E0 e_x
    # eq:curvatureAF: dE_s/ds = t^T (grad E) t + E . dt/ds; grad E = 0 for a uniform field,
    # dt/ds = kappa n, so f = -dE_s/ds = -kappa (E . n)
    f = -g["kappa"] * E0 * g["nx"]
    Ve = -E0 * g["x"]                                  # path potential: dVe/ds = -E . t
    Vinf, Vt = passive_cable(Ve, g["ds"], times)
    return dict(g=g, Es=Es, f=f, Ve=Ve, Vinf=Vinf, Vt=Vt, times=times)


def straight_finite_exact(s, L, E0=1.0, lam=1.0):
    """Steady passive V_m on a straight sealed cable of length L (s from -L/2 to L/2) in a
    uniform longitudinal field E0: V'' - V/lam^2 = 0 inside, V' = E0 at both ends."""
    return E0 * lam * np.sinh(s / lam) / np.cosh(L / (2 * lam))


# ---------------------------------------------------------------------------
# 21.6  Strength-duration: exact passive result (eq:exactsd) and the hyperbolic fit (eq:lapicque)
# ---------------------------------------------------------------------------
def sd_exact(x):
    """A_th / A_rh for a rectangular pulse of duration T = x tau_m (eq:exactsd)."""
    return 1.0 / (1.0 - np.exp(-np.asarray(x, float)))


def sd_fit(x, chron=LN2):
    """A_th / A_rh of eq:lapicque with chronaxie tau_ch = chron * tau_m."""
    return 1.0 + chron / np.asarray(x, float)


def run() -> dict:
    out = {}
    # ---- 21.3 Gaussian profile: f = 0 where |E| peaks; extrema at s = +-ell, value E0/(ell sqrt e)
    s = np.linspace(-4, 4, 160001)
    Ve, Es, f = gaussian_profile(s)
    h = s[1] - s[0]
    close(np.max(np.abs(np.gradient(np.gradient(Ve, h), h) - f)[100:-100]), 0.0, atol=1e-6, name="f = Ve''")
    close(np.max(np.abs(-np.gradient(Es, h) - f)[100:-100]), 0.0, atol=1e-6, name="f = -Es'")
    kmax = np.argmax(f)
    close(s[kmax], 1.0, rtol=1e-3, name="f maximum at s = ell")
    close(f[kmax], np.exp(-0.5), rtol=1e-6, name="f max = E0/(ell sqrt e)")
    close(np.exp(-0.5), 0.607, rtol=1e-3, name="0.607")
    close(f[np.argmin(np.abs(s))], 0.0, atol=1e-12, name="f(0) = 0 at the |E| peak")
    close(Ve[-1] - Ve[0], -np.sqrt(2 * np.pi) * erf(4 / np.sqrt(2)), rtol=1e-9, name="total drop of Ve")
    out["gauss_fmax"] = float(f[kmax])

    # passive response of an (effectively infinite) straight cable to the Gaussian field, ell = lam
    ds = 0.02
    sg = np.arange(-15, 15 + ds / 2, ds)
    Ve_g, _, f_g = gaussian_profile(sg)
    Vinf_g, (V0, V01) = passive_cable(Ve_g, ds, times=(1e-3, 0.1))
    k_dep = np.argmax(Vinf_g)
    out["gauss_Vpeak"] = float(Vinf_g[k_dep])
    out["gauss_Vpeak_s"] = float(sg[k_dep])
    # the response is antisymmetric: depolarized where f > 0, hyperpolarized where f < 0
    close(Vinf_g[k_dep], -Vinf_g.min(), rtol=1e-6, name="antisymmetric response")
    assert sg[k_dep] > 0
    # early time: tau dV/dt = lam^2 f at t = 0+ (the activating function sets the initial rate)
    inner = np.abs(sg) < 5
    close(np.max(np.abs(V0[inner] / 1e-3 - f_g[inner])), 0.0, atol=2e-3, name="initial rate = lam^2 f / tau")
    k = np.argmin(np.abs(sg - 1.0))
    out["gauss_early_ratio"] = float(V01[k] / (0.1 * f_g[k]))

    # ---- 21.4 verification of the compartment model: straight finite cable, uniform E0
    errs = []
    for dss in (0.02, 0.01):
        L = 6.0
        n = int(round(L / dss))
        sc = (np.arange(n) + 0.5) * dss - L / 2
        Vinf, _ = passive_cable(-sc, dss)                 # Ve = -E0 s, E0 = 1
        err = np.max(np.abs(Vinf - straight_finite_exact(sc, L)))
        close(err, 0.0, atol=2e-4 if dss == 0.02 else 5e-5, name=f"straight cable vs exact (ds={dss})")
        errs.append(err)
    close(errs[1] / errs[0], 0.25, rtol=0.02, name="error quartered when ds is halved (second order)")
    out["straight_errs"] = errs
    close(straight_finite_exact(3.0, 6.0), np.tanh(3.0), rtol=1e-12, name="end value E0 lam tanh(L/2lam)")
    out["straight_end"] = float(np.tanh(3.0))
    close(np.tanh(3.0), 0.995, rtol=1e-3, name="tanh 3")

    # ---- 21.4 bent axon in a uniform field
    r = bent_response()
    g = r["g"]
    # activating function: zero on straight parts, E0 sin(theta)/R in the bend, peak E0/R at its exit
    straight = g["kappa"] == 0
    close(np.max(np.abs(r["f"][straight])), 0.0, atol=1e-15, name="f = 0 on straight parts")
    close(r["f"].max(), 1.0 / BENT["R"], rtol=2e-3, name="f peak = E0/R")
    # |E| is the same everywhere (uniform field): no location is singled out by |E|
    Vinf = r["Vinf"]
    kb = np.argmax(Vinf)
    kt = np.argmin(Vinf)
    out["bend_V"] = float(Vinf[kb])
    out["bend_V_s"] = float(g["s"][kb])
    # value at the sealed end itself: node 0 sits ds/2 inside, and dV_m/ds = E_s there
    term_end = Vinf[0] - r["Es"][0] * g["ds"] / 2
    out["term_V"] = float(term_end)
    out["term_V_s"] = float(g["s"][kt])
    assert g["s1"] - 0.2 < g["s"][kb] < g["s2"] + 0.5, "depolarization peak at the bend"
    assert g["s"][kt] < 0.05, "hyperpolarization peak at the terminal"
    # steady state of a sealed passive cable: sum of V_m (net membrane current) is zero
    close(np.sum(Vinf) * g["ds"], 0.0, atol=1e-9, name="net membrane current zero")
    # convergence in ds
    r2 = bent_response(ds=0.01)
    close(r2["Vinf"].max(), Vinf.max(), rtol=2e-3, name="bend peak converged")
    close(r2["Vinf"][0] - r2["Es"][0] * r2["g"]["ds"] / 2, term_end, rtol=1e-3, name="terminal value converged")
    # far end (perpendicular to E) is unpolarized
    close(Vinf[-1], 0.0, atol=2e-3, name="far end not polarized")
    # bend acts like a smoothed step of E_s: an abrupt step from E0 to 0 on an infinite cable gives E0 lam/2
    ds0 = 0.01
    ss = (np.arange(3000) + 0.5) * ds0 - 15.0
    Ve_step = -np.where(ss < 0, ss, 0.0)                    # E_s = E0 for s < 0, 0 for s > 0
    Vstep, _ = passive_cable(Ve_step, ds0)
    close(Vstep.max(), 0.5, rtol=5e-3, name="abrupt step of E_s: peak E0 lam / 2")
    out["step_V"] = float(Vstep.max())
    # a uniform longitudinal field on a long straight cable: interior unpolarized
    close(np.max(np.abs(passive_cable(-ss, ds0)[0][1000:2000])), 0.0, atol=1e-3, name="straight interior V = 0")
    out["V_times"] = [float(np.max(v)) for v in r["Vt"]]
    # reversing the field reverses everything (linearity)
    Vrev, _ = passive_cable(-r["Ve"], g["ds"])
    close(np.max(np.abs(Vrev + Vinf)), 0.0, atol=1e-12, name="field reversal flips V_m")

    # ---- 21.6 strength-duration
    close(LN2, 0.693, rtol=1e-3, name="tau_ch / tau_m = ln 2")
    close(sd_exact(LN2), 2.0, rtol=1e-12, name="exact: 2 x rheobase at chronaxie")
    close(sd_fit(LN2), 2.0, rtol=1e-12, name="fit: 2 x rheobase at chronaxie")
    close(1 / LN2, 1.443, rtol=1e-3, name="short-pulse ratio exact/fit -> 1/ln2")
    x = np.array([0.1, 1.0, 3.0])
    ex, fi = sd_exact(x), sd_fit(x)
    close(ex[0], 10.51, rtol=1e-3, name="exact at 0.1 tau")
    close(fi[0], 7.93, rtol=1e-3, name="fit at 0.1 tau")
    close(ex[1], 1.582, rtol=1e-3, name="exact at tau")
    close(fi[1], 1.693, rtol=1e-3, name="fit at tau")
    close(ex[2], 1.052, rtol=1e-3, name="exact at 3 tau")
    close(fi[2], 1.231, rtol=1e-3, name="fit at 3 tau")
    close(fi[1] / ex[1] - 1, 0.070, rtol=0.02, name="fit 7% high at tau")
    close(fi[2] / ex[2] - 1, 0.170, rtol=0.01, name="fit 17% high at 3 tau")
    xx = np.linspace(LN2, 50, 400001)
    q = sd_fit(xx) / sd_exact(xx)
    km = np.argmax(q)
    close(q[km] - 1, 0.172, rtol=0.01, name="largest overshoot of the fit")
    close(xx[km], 2.6, rtol=0.03, name="... at T ~ 2.6 tau")
    out["sd_overshoot"] = float(q[km] - 1)
    out["sd_overshoot_x"] = float(xx[km])
    # exact is above the fit for T < tau_ch and below it for T > tau_ch
    assert np.all(sd_exact(np.linspace(1e-3, LN2 * 0.999, 1000)) > sd_fit(np.linspace(1e-3, LN2 * 0.999, 1000)))
    assert np.all(sd_exact(np.linspace(LN2 * 1.001, 40, 1000)) < sd_fit(np.linspace(LN2 * 1.001, 40, 1000)))
    # threshold charge Q = A T: fit is a straight line A_rh (T + tau_ch); exact -> A_rh tau_m as T -> 0
    close(1e-6 * sd_exact(1e-6), 1.0, rtol=1e-5, name="exact Q_th(T->0) = A_rh tau_m")
    close(1e-6 * sd_fit(1e-6), LN2, rtol=1e-5, name="fit Q_th(T->0) = A_rh tau_ch")
    # ---- quoted (rounded) values in zh/ch21_neuro.tex
    close(out["gauss_Vpeak"], 0.267, rtol=2e-3, name="Gaussian: steady V peak 0.267 E0 lam")
    close(out["gauss_Vpeak_s"], 1.30, rtol=1e-2, name="... at s = 1.30 lam")
    close(out["bend_V"], 0.40, rtol=5e-3, name="bend +0.40 E0 lam")
    close(out["term_V"], -0.97, rtol=5e-3, name="terminal -0.97 E0 lam")
    close(out["straight_end"], 0.995, rtol=1e-3, name="straight L = 6 lam: 0.995 E0 lam")
    close(0.5e-3 * LN2 * 1e3, 0.347, rtol=2e-3, name="chronaxie for tau_m = 0.5 ms (ms)")
    return out


_CACHE: dict = {}


def run_cache() -> dict:
    """run() once per process (the figure module reads quoted values from here)."""
    if not _CACHE:
        _CACHE.update(run())
    return _CACHE


if __name__ == "__main__":
    o = run()
    print("Chapter 21 checks passed.")
    print(f"  Gaussian field: f max = {o['gauss_fmax']:.4f} E0/ell at s = ell; f = 0 at the |E| peak")
    print(f"  passive response (ell = lam): V peak = {o['gauss_Vpeak']:.4f} E0 lam at s = {o['gauss_Vpeak_s']:.2f} lam;"
          f" early V/(0.1 f) = {o['gauss_early_ratio']:.4f}")
    print(f"  straight sealed cable L = 6 lam: end V = {o['straight_end']:.4f} E0 lam")
    print(f"  bent axon (R = {BENT['R']} lam): bend V = {o['bend_V']:.4f} E0 lam at s = {o['bend_V_s']:.2f};"
          f" terminal V = {o['term_V']:.4f} E0 lam at s = {o['term_V_s']:.2f}; V(t) peaks {o['V_times']}")
    print(f"  straight cable errors vs exact (ds = 0.02, 0.01): {o['straight_errs']}")
    print(f"  strength-duration: max fit overshoot {o['sd_overshoot']*100:.1f}% at T = {o['sd_overshoot_x']:.2f} tau")
