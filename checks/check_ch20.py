"""Chapter 20 (inverse coil design): every number quoted in zh/ch20_coils.tex.

    python3 checks/check_ch20.py

figsrc/fig_ch20.py imports the model defined here.

Teaching example (stated once in the text; not a device design):
  * a biplanar x-gradient former: two square plates |x|, |y| <= 0.30 m at z = +-0.18 m (the pole gap of the
    Chapter 15 teaching magnet is 0.40 m), main field B0 along z;
  * target b_z = G x with G = 10 mT/m (the Chapter 19 teaching gradient) on the surface of a 20 cm DSV;
  * equivalent conducting sheet sigma*t = 5.8e7 S/m x 1 mm (copper conductivity of Chapter 12, current assumed
    uniform through a 1 mm thickness) -- only the absolute power depends on this choice;
  * the same stream function on both plates (psi odd in x, even in y), expanded in 8 x 8 sine products that
    vanish on the plate edges, so no current leaves a plate (sec:potentials-stream).
Operators (eq:coilobjective): field matrix C (Biot-Savart of the equivalent dipole sheet, checked against the
current-sheet form), resistance matrix R (closed form, checked by quadrature), inductance matrix L (magnetic
energy in the 2-D Fourier domain, checked against a real-space Neumann integral and against coaxial rings).
Also: the long-solenoid analogue of active shielding (ex:eddy-shell geometry), and the field of the
illustrative cylinder stream function of figure stream_cylinder_3d.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0

from numerics import MU0

# ---------------------------------------------------------------------------
# Teaching inputs
# ---------------------------------------------------------------------------
A_HALF = 0.30            # plate half width (m)
D_PLANE = 0.18           # plates at z = +-D_PLANE (m)
R_ROI = 0.10             # DSV radius (m)
G_T = 10e-3              # target gradient (T/m)
SIG_T = 5.8e7 * 1e-3     # sigma * t of the equivalent sheet (S)
B0_LOW = 0.1             # main field for force densities (T)
M_LIST = np.arange(2, 17, 2)      # sin(m pi (x+a)/2a), m even -> odd in x
N_LIST = np.arange(1, 16, 2)      # n odd -> even in y
NB = len(M_LIST) * len(N_LIST)
I_WIRE = 100.0           # current per turn used for contour extraction (A)
RAMP = 1e-4              # 0 -> I_WIRE in 0.1 ms (Ch 19 teaching ramp)
TARGET_ERR = 0.01        # 1 % RMS error on the DSV surface


def basis_1d(m, x, a=A_HALF):
    return np.where(np.abs(x) <= a, np.sin(m * np.pi * (x + a) / (2 * a)), 0.0)


def dbasis_1d(m, x, a=A_HALF):
    al = m * np.pi / (2 * a)
    return np.where(np.abs(x) <= a, al * np.cos(al * (x + a)), 0.0)


def plate_grid(n=150, a=A_HALF):
    h = 2 * a / n
    return -a + (np.arange(n) + 0.5) * h, h


def basis_on_grid(x):
    bx = np.array([basis_1d(m, x) for m in M_LIST])
    by = np.array([basis_1d(k, x) for k in N_LIST])
    return np.einsum("mi,nj->mnij", bx, by).reshape(NB, -1)


def psi_grid(c, x):
    """Stream function on the grid x (both axes) for coefficients c (A)."""
    bx = np.array([basis_1d(m, x) for m in M_LIST])
    by = np.array([basis_1d(k, x) for k in N_LIST])
    return np.einsum("mn,mi,nj->ij", np.asarray(c).reshape(len(M_LIST), len(N_LIST)), bx, by)


def sheet_current(c, x):
    """K = grad psi x e_z = (d psi/dy, -d psi/dx) on the grid (indexing ij: first axis x)."""
    cc = np.asarray(c).reshape(len(M_LIST), len(N_LIST))
    bx = np.array([basis_1d(m, x) for m in M_LIST]); by = np.array([basis_1d(k, x) for k in N_LIST])
    dbx = np.array([dbasis_1d(m, x) for m in M_LIST]); dby = np.array([dbasis_1d(k, x) for k in N_LIST])
    Kx = np.einsum("mn,mi,nj->ij", cc, bx, dby)
    Ky = -np.einsum("mn,mi,nj->ij", cc, dbx, by)
    return Kx, Ky


def fib_sphere(n, r):
    i = np.arange(n) + 0.5
    th = np.arccos(1 - 2 * i / n)
    ph = np.pi * (1 + 5**0.5) * i
    return r * np.column_stack([np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)])


def ball_points(n, r, seed=3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return v * r * rng.random(n)[:, None] ** (1 / 3)


def field_matrix(pts, n=150, form="dipole"):
    """b_z at pts per unit coefficient, both plates carrying the same psi (normals +z).
    form='dipole': psi as a magnetic dipole sheet m = psi e_z (valid because psi = 0 on the plate edges);
    form='current': Biot-Savart of K = grad psi x e_z."""
    x, h = plate_grid(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xf, Yf = X.ravel(), Y.ravel()
    C = np.zeros((len(pts), NB))
    if form == "dipole":
        basis = basis_on_grid(x)
    else:
        bx = np.array([basis_1d(m, x) for m in M_LIST]); by = np.array([basis_1d(k, x) for k in N_LIST])
        dbx = np.array([dbasis_1d(m, x) for m in M_LIST]); dby = np.array([dbasis_1d(k, x) for k in N_LIST])
        Kx = np.einsum("mi,nj->mnij", bx, dby).reshape(NB, -1)
        Ky = -np.einsum("mi,nj->mnij", dbx, by).reshape(NB, -1)
    for zp in (D_PLANE, -D_PLANE):
        dx = pts[:, 0:1] - Xf[None]; dy = pts[:, 1:2] - Yf[None]; dz = pts[:, 2:3] - zp
        R2 = dx * dx + dy * dy + dz * dz
        if form == "dipole":
            C += MU0 / (4 * np.pi) * ((3 * dz * dz - R2) / R2**2.5 * h * h) @ basis.T
        else:
            R3 = R2**1.5
            C += MU0 / (4 * np.pi) * ((dy / R3 * h * h) @ Kx.T - (dx / R3 * h * h) @ Ky.T)
    return C


def full_field(c, pts, n=150):
    """All three components of b at pts for coefficients c (dipole-sheet form, m = psi e_z)."""
    x, h = plate_grid(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    psi = psi_grid(c, x).ravel() * h * h
    b = np.zeros((len(pts), 3))
    for zp in (D_PLANE, -D_PLANE):
        d = pts[:, None, :] - np.stack([X.ravel(), Y.ravel(), np.full(X.size, zp)], axis=1)[None]
        R2 = np.sum(d * d, axis=2)
        mdotr = d[:, :, 2]
        for k in range(3):
            b[:, k] += MU0 / (4 * np.pi) * ((3 * d[:, :, k] * mdotr / R2**2.5 - (k == 2) / R2**1.5) @ psi)
    return b


def R_matrix():
    """Dissipated power P = c^T R c of both plates (closed form; the sine basis is R-orthogonal)."""
    a = A_HALF
    v = [2 * a * a * ((m * np.pi / (2 * a))**2 + (k * np.pi / (2 * a))**2) / SIG_T for m in M_LIST for k in N_LIST]
    return np.diag(v)


def R_matrix_quadrature(n=300):
    x, h = plate_grid(n)
    bx = np.array([basis_1d(m, x) for m in M_LIST]); by = np.array([basis_1d(k, x) for k in N_LIST])
    dbx = np.array([dbasis_1d(m, x) for m in M_LIST]); dby = np.array([dbasis_1d(k, x) for k in N_LIST])
    gx = np.einsum("mi,nj->mnij", dbx, by).reshape(NB, -1)
    gy = np.einsum("mi,nj->mnij", bx, dby).reshape(NB, -1)
    return 2 * (gx @ gx.T + gy @ gy.T) * h * h / SIG_T


def ft_1d(m, k, a=A_HALF):
    """Fourier transform int sin(alpha (x+a)) exp(-i k x) dx over |x| <= a (closed form)."""
    al = m * np.pi / (2 * a)
    return np.exp(1j * k * a) * al * ((-1.0)**m * np.exp(-2j * k * a) - 1) / (k * k - al * al)


def _fourier_kernel_matrix(weight, K=1500.0, dk=0.25):
    """mu0 int w(k) Re(N_i^* N_j) d^2k/(2pi)^2 for the separable sine basis (positive quadrant x 4)."""
    k = (np.arange(int(K / dk)) + 0.5) * dk
    FX = np.array([ft_1d(m, k) for m in M_LIST]); FY = np.array([ft_1d(n, k) for n in N_LIST])
    Ax = np.real(np.einsum("ak,bk->abk", np.conj(FX), FX)).reshape(-1, len(k))
    Ay = np.real(np.einsum("ak,bk->abk", np.conj(FY), FY)).reshape(-1, len(k))
    T = np.zeros((Ax.shape[0], len(k)))
    for i0 in range(0, len(k), 500):
        kk = np.hypot(k[i0:i0 + 500][:, None], k[None, :])
        T += Ax[:, i0:i0 + 500] @ weight(kk)
    full = T @ Ay.T * dk * dk * 4 / (2 * np.pi)**2 * MU0
    nm, nn = len(M_LIST), len(N_LIST)
    return full.reshape(nm, nm, nn, nn).transpose(0, 2, 1, 3).reshape(NB, NB)


@lru_cache(maxsize=4)
def L_matrix(K=1500.0, dk=0.25):
    """Stored energy W = c^T L c / 2 of both plates in series (same psi on both):
    W = (mu0/2) int k |psi^|^2 (1 + exp(-2 D k)) d^2k/(2pi)^2."""
    return _fourier_kernel_matrix(lambda kk: kk * (1 + np.exp(-2 * D_PLANE * kk)), K, dk)


def mutual_fourier(s):
    """Mutual inductance matrix between basis functions on two parallel plates a distance s apart:
    M_ij = (mu0/2) int k Re(N_i^* N_j) exp(-k s) d^2k/(2pi)^2."""
    return 0.5 * _fourier_kernel_matrix(lambda kk: kk * np.exp(-s * kk))


def mutual_realspace(i, j, s, n=60):
    """Same mutual inductance from the Neumann form mu0/(4 pi) int int k_i . k_j / |r - r'| dS dS'."""
    x, h = plate_grid(n)
    mi, ni = divmod(i, len(N_LIST)); mj, nj = divmod(j, len(N_LIST))
    X, Y = np.meshgrid(x, x, indexing="ij")
    def kvec(m, nn):
        kx = (basis_1d(M_LIST[m], X) * dbasis_1d(N_LIST[nn], Y)).ravel()
        ky = (-dbasis_1d(M_LIST[m], X) * basis_1d(N_LIST[nn], Y)).ravel()
        return kx, ky
    kix, kiy = kvec(mi, ni); kjx, kjy = kvec(mj, nj)
    P = np.column_stack([X.ravel(), Y.ravel()])
    tot = 0.0
    for a0 in range(0, len(P), 600):
        d2 = np.sum((P[a0:a0 + 600, None, :] - P[None, :, :])**2, axis=2) + s * s
        tot += (kix[a0:a0 + 600, None] * kjx[None] + kiy[a0:a0 + 600, None] * kjy[None]).ravel() @ (1 / np.sqrt(d2)).ravel()
    return MU0 / (4 * np.pi) * tot * h**4


# --- coaxial "rings": smooth annular stream functions -----------------------------------------
RING = dict(r0=0.15, w=0.03)


def ring_profile(rho):
    r0, w = RING["r0"], RING["w"]
    t = np.clip((rho - (r0 - w / 2)) / w, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * t))


def ring_dprofile(rho):
    r0, w = RING["r0"], RING["w"]
    t = (rho - (r0 - w / 2)) / w
    return np.where((t > 0) & (t < 1), -0.5 * np.pi / w * np.sin(np.pi * t), 0.0)


def ring_mutual_fourier(s):
    """M = (mu0/4pi) int_0^inf k^2 f^(k)^2 exp(-k s) dk, f^ the Hankel transform of the profile."""
    rho, wr = np.polynomial.legendre.leggauss(400)
    lo, hi = RING["r0"] - RING["w"] / 2, RING["r0"] + RING["w"] / 2
    r_in = 0.5 * lo * (rho + 1); w_in = 0.5 * lo * wr
    r_tp = lo + 0.5 * (hi - lo) * (rho + 1); w_tp = 0.5 * (hi - lo) * wr
    rr = np.concatenate([r_in, r_tp]); ww = np.concatenate([w_in, w_tp])
    f = ring_profile(rr)
    k = np.linspace(0, 60 / s, 6001)[1:]
    fh = 2 * np.pi * (j0(np.outer(k, rr)) * (f * rr)) @ ww
    integ = k * k * fh**2 * np.exp(-k * s)
    return MU0 / (4 * np.pi) * np.trapezoid(integ, k)


def ring_mutual_neumann(s):
    """Neumann double line integral (eq:Agreen with eq:Aflux), weighted by the radial current dI = -f'(rho) drho."""
    lo, hi = RING["r0"] - RING["w"] / 2, RING["r0"] + RING["w"] / 2
    x, w = np.polynomial.legendre.leggauss(60)
    r = lo + 0.5 * (hi - lo) * (x + 1); wr = 0.5 * (hi - lo) * w
    dI = -ring_dprofile(r) * wr
    ph, wp = np.polynomial.legendre.leggauss(200)
    ph = np.pi * (ph + 1); wp = np.pi * wp
    R1, R2 = np.meshgrid(r, r, indexing="ij")
    Ml = np.zeros_like(R1)
    for p, q in zip(ph, wp):
        Ml += q * np.cos(p) / np.sqrt(R1**2 + R2**2 - 2 * R1 * R2 * np.cos(p) + s * s)
    Ml *= MU0 * R1 * R2 / 2
    return dI @ Ml @ dI


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def model():
    fit = fib_sphere(400, R_ROI)
    chk = fib_sphere(997, R_ROI)
    vol = ball_points(2000, R_ROI)
    return dict(fit=fit, chk=chk, vol=vol, C=field_matrix(fit), Cc=field_matrix(chk), Cv=field_matrix(vol),
                R=R_matrix(), L=L_matrix(), b=G_T * fit[:, 0], bc=G_T * chk[:, 0], bv=G_T * vol[:, 0])


def penalty(name):
    m = model()
    return {"R": m["R"], "L": m["L"], "I": np.eye(NB)}[name]


def solve(alpha, name="R"):
    """Minimise |C c - b|^2 + alpha * s * c^T P c (s makes alpha dimensionless: trace(C^T C)/trace(P))."""
    m = model(); P = penalty(name)
    s = np.trace(m["C"].T @ m["C"]) / np.trace(P)
    return np.linalg.solve(m["C"].T @ m["C"] + alpha * s * P, m["C"].T @ m["b"])


def metrics(c):
    m = model()
    rms = lambda CC, bb: np.sqrt(np.mean((CC @ c - bb)**2)) / np.sqrt(np.mean(bb**2))
    return dict(err_fit=rms(m["C"], m["b"]), err=rms(m["Cc"], m["bc"]), err_vol=rms(m["Cv"], m["bv"]),
                peak=np.max(np.abs(m["Cc"] @ c - m["bc"])) / (G_T * R_ROI),
                peak_vol=np.max(np.abs(m["Cv"] @ c - m["bv"])) / (G_T * R_ROI),
                P=float(c @ m["R"] @ c), W=float(0.5 * c @ m["L"] @ c))


def design_at(err, name="R"):
    """Regularised design whose RMS error on the (independent) check points equals err."""
    la = brentq(lambda t: np.log(metrics(solve(10**t, name))["err"] / err), -16, 4, xtol=1e-10)
    return solve(10**la, name), 10**la


def family(name, n=61, lo=-13.0, hi=1.5):
    out = []
    for a in np.logspace(lo, hi, n):
        mm = metrics(solve(a, name)); out.append((a, mm["err"], mm["P"], mm["W"]))
    return np.array(out)


def exact_constrained(nfib=96):
    """min c^T R c subject to C_p c = b_p at the first-octant points of a Fibonacci sphere (others follow from
    the symmetry of the basis): c = R^-1 C_p^T (C_p R^-1 C_p^T)^-1 b_p."""
    p = fib_sphere(nfib, R_ROI)
    p = p[(p[:, 0] > 1e-9) & (p[:, 1] > 1e-9) & (p[:, 2] > 1e-9)]
    Cp = field_matrix(p); bp = G_T * p[:, 0]
    Ri = np.linalg.inv(model()["R"])
    c = Ri @ Cp.T @ np.linalg.solve(Cp @ Ri @ Cp.T, bp)
    return c, p, Cp, bp


def winding_stats(c, n=301):
    x = np.linspace(-A_HALF, A_HALF, n)
    psi = psi_grid(c, x)
    xg, h = plate_grid(300)
    Kx, Ky = sheet_current(c, xg)
    Kmax = float(np.max(np.hypot(Kx, Ky)))
    # net force and torque on one plate in a uniform B0 e_z: f = K x B0 e_z = B0 (Ky, -Kx, 0)
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    F = B0_LOW * np.array([np.sum(Ky), -np.sum(Kx)]) * h * h
    tz = B0_LOW * np.sum(X * (-Kx) - Y * Ky) * h * h
    return dict(psimax=float(np.abs(psi).max()), turns=float(np.abs(psi).max() / I_WIRE), Kmax=Kmax,
                fmax=Kmax * B0_LOW, F=F, tz=tz, Ftot=B0_LOW * np.sum(np.hypot(Kx, Ky)) * h * h)


# ---------------------------------------------------------------------------
# Active shielding: long-solenoid analogue of ex:eddy-shell
# ---------------------------------------------------------------------------
SHELL = dict(a=0.25, R=0.30, b=0.28)


def shield_analogue(a=SHELL["a"], b=SHELL["b"], R=SHELL["R"]):
    kap = a * a / R / R
    ks = a * a / b / b
    eff = 1 - ks                   # bore field per primary sheet current
    energy_ratio = 1 / (1 - ks)    # stored energy per length for the same bore field
    # explicit check of the energy ratio
    H0 = 1.0
    K1 = H0 / eff; K2 = -ks * K1
    W_sh = H0**2 * np.pi * a * a + K2**2 * np.pi * (b * b - a * a)
    W_un = H0**2 * np.pi * a * a
    flux_R = np.pi * (a * a * K1 + b * b * K2)      # net enclosed (mu0 omitted) inside radius b <= r <= R
    return dict(kappa=kap, ks=ks, eff=eff, energy_ratio=energy_ratio, energy_ratio_explicit=W_sh / W_un,
                flux_R=flux_R)


# ---------------------------------------------------------------------------
# Field of the illustrative cylinder stream function (figure stream_cylinder_3d)
# ---------------------------------------------------------------------------
CYL = dict(a=1.0, s=0.55)


def cylinder_gradient(n_phi=256, n_z=1200, zmax=8.0, h=1e-3):
    """Gradient tensor at the centre of psi = cos(phi) z exp(-z^2/2s^2) on the infinite cylinder r = a,
    from the equivalent dipole sheet m = psi e_r (psi -> 0 as |z| -> inf)."""
    a, s = CYL["a"], CYL["s"]
    ph = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    z = np.linspace(-zmax, zmax, n_z)
    dz = z[1] - z[0]
    PH, Z = np.meshgrid(ph, z, indexing="ij")
    psi = np.cos(PH) * Z * np.exp(-Z**2 / (2 * s * s))
    dS = a * (2 * np.pi / n_phi) * dz
    src = np.stack([a * np.cos(PH), a * np.sin(PH), Z], axis=-1).reshape(-1, 3)
    mvec = (psi[..., None] * np.stack([np.cos(PH), np.sin(PH), 0 * PH], axis=-1)).reshape(-1, 3) * dS

    def b(r):
        d = r[None, :] - src
        R2 = np.sum(d * d, axis=1)
        md = np.sum(mvec * d, axis=1)
        return MU0 / (4 * np.pi) * np.sum(3 * d * (md / R2**2.5)[:, None] - mvec / (R2**1.5)[:, None], axis=0)

    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3); e[j] = h
        J[:, j] = (b(e) - b(-e)) / (2 * h)
    return J, b(np.zeros(3)), b(np.array([0.2, 0.0, 0.0])), b(np.array([-0.2, 0.0, 0.0]))


# ---------------------------------------------------------------------------
def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:62s} {x:.6g}")
    return x


def run_checks():
    m = model()
    print("20.3 operators")
    C2 = field_matrix(m["fit"], form="current")
    rel = np.abs(m["C"] - C2).max() / np.abs(m["C"]).max()
    assert rel < 3e-4, rel
    print(f"  ok  field matrix: dipole-sheet and current-sheet forms agree ({rel:.1e})")
    rel = np.abs(m["C"] - field_matrix(m["fit"], n=300)).max() / np.abs(m["C"]).max()
    assert rel < 1e-4, rel
    print(f"  ok  field matrix converged in the plate grid ({rel:.1e})")
    Rq = R_matrix_quadrature()
    rel = np.abs(Rq - m["R"]).max() / np.abs(m["R"]).max()
    assert rel < 2e-3, rel
    print(f"  ok  resistance matrix: closed form equals quadrature, diagonal ({rel:.1e})")
    L = m["L"]
    assert np.allclose(L, L.T, rtol=1e-12) and np.linalg.eigvalsh(L).min() > 0
    rel = np.abs(L - L_matrix(K=3000.0)).max() / np.abs(L).max()
    assert rel < 1e-4, rel
    rel2 = np.abs(L - L_matrix(dk=0.125)).max() / np.abs(L).max()
    assert rel2 < 1e-6, rel2
    print(f"  ok  inductance matrix symmetric, positive definite, converged ({rel:.1e}, {rel2:.1e})")
    for (i, j) in ((0, 0), (9, 1), (27, 36)):
        for s in (2 * D_PLANE, 0.12):
            Mf = mutual_fourier(s)[i, j]; Mr = mutual_realspace(i, j, s)
            assert abs(Mf - Mr) < 1e-3 * abs(mutual_fourier(s)[i, i]), (i, j, s, Mf, Mr)
    print("  ok  Fourier mutual inductance = real-space Neumann integral within 0.1 % of the diagonal")
    for s in (0.05, 0.36):
        Mf, Mn = ring_mutual_fourier(s), ring_mutual_neumann(s)
        close(Mf, Mn, rtol=2e-5, name=f"coaxial rings, s = {s} m: Fourier kernel vs Neumann (H)")

    print("20.4 regularised designs (eq:coilobjective)")
    cR, aR = design_at(TARGET_ERR, "R"); mR = metrics(cR)
    cL, aL = design_at(TARGET_ERR, "L"); mL = metrics(cL)
    cI, aI = design_at(TARGET_ERR, "I"); mI = metrics(cI)
    close(mR["err"], 0.01, rtol=1e-6, name="R-penalty design: RMS error on the check points")
    close(mR["err_fit"], 0.01, rtol=0.05, name="... on the fitting points")
    close(mR["P"], 512, rtol=3e-3, name="R-penalty design: dissipated power (W)")
    close(mR["W"], 0.557, rtol=3e-3, name="... stored energy at 10 mT/m (J)")
    close(mR["peak"], 0.0161, rtol=0.02, name="... peak error on the DSV surface / (G R)")
    assert mR["peak_vol"] <= mR["peak"] * 1.0001
    print("  ok  peak error inside the ball does not exceed that on the surface (harmonic error)")
    close(mR["err_vol"], 0.0059, rtol=0.03, name="... RMS error over the ball volume")
    close(mL["P"], 550, rtol=3e-3, name="L-penalty design: power (W)")
    close(mL["W"], 0.544, rtol=3e-3, name="... stored energy (J)")
    close(mI["P"], 807, rtol=3e-3, name="coefficient-norm penalty: power (W)")
    close(mI["W"], 0.605, rtol=3e-3, name="... stored energy (J)")
    close(mI["P"] / mR["P"], 1.58, rtol=5e-3, name="power ratio, |c|^2 penalty / R penalty")
    assert mR["P"] < mL["P"] < mI["P"] and mL["W"] < mR["W"] < mI["W"]
    print("  ok  each physical penalty is best in its own metric; the |c|^2 penalty is worse in both")
    # Pareto check over the whole family: no I-design dominates an R-design
    fR, fI = family("R", 31), family("I", 31)
    for row in fI:
        k = np.argmin(np.abs(np.log(fR[:, 1] / row[1])))
        assert fR[k, 2] <= row[2] * 1.02 or fR[k, 1] < row[1]
    print("  ok  the R-penalty family lies on the error-power Pareto front")
    for e, P_ref in ((0.05, 237), (0.001, 1283)):
        c, _ = design_at(e, "R")
        close(metrics(c)["P"], P_ref, rtol=5e-3, name=f"R-penalty design at {100 * e:g} % error: power (W)")
    w = winding_stats(cR)
    close(w["psimax"], 986, rtol=5e-3, name="max |psi| of the 1 % design (A)")
    close(w["turns"], 9.86, rtol=5e-3, name="... contours per lobe at 100 A per turn")
    Lc = 2 * mR["W"] / I_WIRE**2
    close(Lc * 1e6, 111.4, rtol=5e-3, name="coil inductance 2W/I_w^2 at 100 A per turn (uH)")
    close(mR["P"] / I_WIRE**2, 0.0512, rtol=5e-3, name="sheet-model resistance P/I_w^2 (ohm)")
    close(Lc * I_WIRE / RAMP, 111.4, rtol=5e-3, name="L dI/dt for 0 -> 100 A in 0.1 ms (V)")
    close(G_T / I_WIRE * 1e3, 0.1, rtol=1e-12, name="efficiency G/I_w (mT/m per A)")
    close(Lc * I_WIRE / RAMP * I_WIRE / 1e3, 11.1, rtol=5e-3, name="peak V*I at the end of the ramp, = 2W/ramp (kW)")
    close(2 * mR["W"] / RAMP / 1e3, 11.1, rtol=5e-3, name="... independent of the turn count (kW)")
    close(2 * mR["W"] / 50.0**2 * 1e6, 445.5, rtol=5e-3, name="50 A per turn: inductance (uH)")
    close(2 * mR["W"] / 50.0**2 * 50.0 / RAMP, 222.7, rtol=5e-3, name="... ramp voltage for 0 -> 50 A in 0.1 ms (V)")
    close(w["psimax"] / 25.0, 39.4, rtol=3e-3, name="self-check: contours per lobe at 25 A per turn")
    close(2 * mR["W"] / 25.0**2 * 1e3, 1.78, rtol=3e-3, name="... inductance (mH)")
    close(2 * mR["W"] / 25.0**2 * 25.0 / RAMP, 445.5, rtol=3e-3, name="... ramp voltage 0 -> 25 A in 0.1 ms (V)")
    close(w["Kmax"], 1.72e4, rtol=0.01, name="max sheet current density |K| (A/m)")
    close(w["fmax"], 1.72e3, rtol=0.01, name="... force density |K| B0 at 0.1 T (N/m^2)")
    close(w["Kmax"] * 1.5 / 1e3, 25.8, rtol=0.01, name="... same currents in 1.5 T (kN/m^2)")
    assert np.all(np.abs(w["F"]) < 1e-9 * w["Ftot"]) and abs(w["tz"]) < 1e-9 * w["Ftot"]
    print("  ok  net in-plane force and torque on each plate vanish in a uniform B0")
    close(w["Ftot"], 202, rtol=0.01, name="... while int |f| dS over one plate is (N)")
    s = np.linalg.svd(m["C"] @ np.diag(1 / np.sqrt(np.diag(m["R"]))), compute_uv=False)
    close(s[1] / s[0], 0.213, rtol=0.01, name="second singular value / first (power-weighted)")
    assert s[-1] / s[0] < 1e-11
    print(f"  ok  smallest / largest singular value {s[-1] / s[0]:.1e} (< 1e-11)")

    print("20.4 exact constraints")
    c, p, Cp, bp = exact_constrained(96)
    assert np.allclose(Cp @ c, bp, rtol=0, atol=1e-9 * G_T * R_ROI)
    # KKT system gives the same answer
    Rm = m["R"]; n = len(bp)
    KKT = np.block([[Rm, Cp.T], [Cp, np.zeros((n, n))]])
    sol = np.linalg.solve(KKT, np.concatenate([np.zeros(NB), bp]))
    assert np.allclose(sol[:NB], c, rtol=1e-6, atol=1e-9)
    mc = metrics(c)
    print(f"  ok  {n} exact constraints met; formula equals the KKT solution")
    close(n, 12, rtol=1e-12, name="independent constraint points (first octant of 96)")
    close(mc["err"], 0.0058, rtol=0.01, name="... RMS error on the check points")
    close(mc["P"], 1444, rtol=3e-3, name="... power (W)")
    cr, _ = design_at(mc["err"], "R")
    close(metrics(cr)["P"], 646, rtol=3e-3, name="R-penalty design with the same error: power (W)")
    c2, p2, _, _ = exact_constrained(160)
    m2 = metrics(c2)
    close(len(p2), 20, rtol=1e-12, name="20 constraint points (first octant of 160)")
    close(m2["err"], 0.00106, rtol=0.01, name="... RMS error")
    close(m2["P"], 4920, rtol=3e-3, name="... power (W)")
    cr, _ = design_at(m2["err"], "R")
    close(metrics(cr)["P"], 1256, rtol=3e-3, name="R-penalty design with the same error: power (W)")

    print("20.5 active shielding, long-solenoid analogue of ex:eddy-shell")
    sh = shield_analogue()
    close(sh["kappa"], 0.694, rtol=1e-3, name="unshielded coupling kappa = a^2/R^2")
    assert abs(sh["flux_R"]) < 1e-12
    print("  ok  shielded pair: zero net flux inside the shell radius -> no slow eddy tail")
    close(sh["eff"], 0.203, rtol=3e-3, name="bore field per primary current, 1 - a^2/b^2 (b = 0.28 m)")
    close(sh["energy_ratio"], 4.92, rtol=3e-3, name="stored energy for the same bore field, b^2/(b^2-a^2)")
    close(sh["energy_ratio_explicit"], sh["energy_ratio"], rtol=1e-12, name="... from the two regions explicitly")

    print("figure stream_cylinder_3d: field of the illustrative stream function")
    J, b0, bp_, bm_ = cylinder_gradient()
    Gx = J[2, 0]
    assert abs(b0).max() < 1e-6 * abs(Gx)
    assert abs(J[0, 2] - Gx) < 1e-3 * abs(Gx)
    others = [J[i, j] for i in range(3) for j in range(3) if (i, j) not in ((2, 0), (0, 2))]
    assert max(abs(v) for v in others) < 1e-3 * abs(Gx)
    print(f"  ok  centre: b = 0, only G_zx = G_xz != 0 (an x gradient; G_zx = {Gx:.3e} T/m per A)")
    assert np.isclose(bp_[2], -bm_[2], rtol=1e-9) and np.sign(bp_[2]) == np.sign(Gx)
    print("  ok  b_z is odd in x")
    print("check_ch20: all assertions passed")
    return True


if __name__ == "__main__":
    run_checks()
