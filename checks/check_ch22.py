"""Chapter 22 (the edge of the quasistatic picture: RF fields and SAR): every number in zh/ch22_rf.tex.

    python3 checks/check_ch22.py

The figure module figsrc/fig_ch22.py imports the functions defined here, so the text, the figures
and the assertions share one implementation.  Inputs:
  * the Chapter-8 teaching scales (sigma = 0.5 S/m, l = 0.20 m, eps_r = 80 comparison medium,
    3 kHz gradient example) held fixed across frequencies only to show the scale;
  * the proton gyromagnetic ratio gamma/(2 pi) = 42.577 478 518 MHz/T (CODATA 2018, the value
    used in checks/check_ch19.py) for Larmor frequencies and the flip-angle example;
  * the old-edition SAR teaching values (rf:5): sigma = 0.5 S/m, rho_m = 1000 kg/m^3,
    E_peak = 100 V/m, 1 ms every 10 ms; the rotating-phasor convention of extended_checks.py.
All are teaching inputs, not tissue data, device data or exposure limits.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import erf

import extended_checks
from numerics import EPS0, MU0

C0 = 1.0 / np.sqrt(MU0 * EPS0)
GAMMA_BAR = 42.577478518e6          # Hz/T (CODATA 2018), as in check_ch19.py
GAMMA = 2 * np.pi * GAMMA_BAR       # rad/(s T)
SIGMA, ELL, EPS_R = 0.5, 0.20, 80.0  # Chapter-8 teaching scales
SAR_SIGMA, SAR_RHO, SAR_E = 0.5, 1000.0, 100.0   # rf:5 teaching values (peak amplitude)


def close(x, ref, rtol=5e-3, atol=0.0, name=""):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    return x


# ---------------------------------------------------------------------------
# 22.1 / 22.2  frequency ranges and the three terms of the curl-curl equation
# ---------------------------------------------------------------------------
def scales(f, sigma=SIGMA, ell=ELL, eps_r=EPS_R):
    w = 2 * np.pi * f
    lam0 = C0 / f
    return dict(f=f, lam0=lam0, k0l=w * ell / C0, kl=w * ell * np.sqrt(eps_r) / C0,
                eta_m=w * MU0 * sigma * ell**2, delta=np.sqrt(2 / (w * MU0 * sigma)))


def larmor(B0):
    return GAMMA_BAR * B0


ROWS = [("gradient", 3.0e3), ("RF 0.1 T", larmor(0.1)), ("RF 1.5 T", larmor(1.5)), ("RF 3 T", larmor(3.0))]


# ---------------------------------------------------------------------------
# 22.3  rotating components (eq:corotating), peak phasors with exp(+j w t)
# ---------------------------------------------------------------------------
def corotating(Bx, By):
    """Co- and counter-rotating parts of a transverse phasor (Bx, By), eq:corotating.
    Co-rotating = clockwise seen from +z, the sense of free precession for gamma > 0."""
    Bx, By = complex(Bx), complex(By)
    return (Bx - 1j * By) / 2, (Bx + 1j * By) / 2


def physical(Bx, By, wt):
    """Real field Re{(Bx, By) exp(j w t)} at phases wt."""
    return np.real(Bx * np.exp(1j * wt)), np.real(By * np.exp(1j * wt))


def flip_b1(alpha_deg=90.0, duration=1e-3):
    """Rectangular on-resonance pulse: alpha = gamma b_co T (rotating-wave approximation)."""
    return np.deg2rad(alpha_deg) / (GAMMA * duration)


# ---------------------------------------------------------------------------
# 22.5  averages
# ---------------------------------------------------------------------------
def local_sar_cycle(E_amp, sigma=SAR_SIGMA, rho=SAR_RHO, convention="peak"):
    return (0.5 if convention == "peak" else 1.0) * sigma * E_amp**2 / rho


HOT_W = 5e-3       # width parameter of the 1-D teaching hot spot (m)


def hotspot_sar(x, w=HOT_W):
    """Local (RF-cycle averaged) SAR of a 1-D teaching hot spot: E = 100 V/m exp(-x^2/(2 w^2)) peak."""
    return local_sar_cycle(SAR_E) * np.exp(-np.asarray(x)**2 / w**2)


def window_average(L, w=HOT_W):
    """Average of hotspot_sar over a centred window of length L (uniform density): a 1-D
    stand-in for the mass average over V_a in eq:rfQ."""
    return local_sar_cycle(SAR_E) * np.sqrt(np.pi) * w / L * erf(L / (2 * w))


def q_matrix(fields, Y, rho, dV):
    """Q_ij = 1/(2 m_a) sum_V e_i^H Y_H e_j dV  (eq:rfQ) for fields[k] of shape (npts, 3)."""
    YH = 0.5 * (Y + np.conj(np.swapaxes(Y, -1, -2)))
    m = np.sum(rho * dV)
    n = len(fields)
    Q = np.zeros((n, n), complex)
    for i in range(n):
        for j in range(n):
            Q[i, j] = np.sum(np.einsum("pa,pab,pb->p", np.conj(fields[i]), YH, fields[j]) * dV) / (2 * m)
    return Q


def run() -> dict:
    out = {}
    # ---- 22.1 table
    tab = {name: scales(f) for name, f in ROWS}
    close(larmor(0.1) / 1e6, 4.258, rtol=1e-3, name="Larmor 0.1 T")
    close(larmor(1.5) / 1e6, 63.87, rtol=1e-3, name="Larmor 1.5 T")
    close(larmor(3.0) / 1e6, 127.7, rtol=1e-3, name="Larmor 3 T")
    g, r1, r15, r3 = (tab[n] for n, _ in ROWS)
    close(g["lam0"], 1.0e5, rtol=2e-3, name="lambda0 3 kHz ~ 100 km")
    close(g["k0l"], 1.26e-5, rtol=3e-3, name="eta_w 3 kHz (Ch 8)")
    close(g["eta_m"], 4.74e-4, rtol=1e-3, name="eta_m 3 kHz (Ch 8)")
    close(g["kl"], 1.12e-4, rtol=5e-3, name="k l (eps_r 80) 3 kHz")
    close(r1["lam0"], 70.4, rtol=1e-3, name="lambda0 0.1 T")
    close(r1["k0l"], 0.0178, rtol=3e-3, name="k0 l 0.1 T")
    close(r1["kl"], 0.160, rtol=3e-3, name="k l (eps_r 80) 0.1 T")
    close(r1["eta_m"], 0.672, rtol=2e-3, name="eta_m 0.1 T")
    close(r1["delta"], 0.345, rtol=3e-3, name="delta 0.1 T")
    close(r15["lam0"], 4.69, rtol=2e-3, name="lambda0 1.5 T")
    close(r15["k0l"], 0.268, rtol=3e-3, name="k0 l 1.5 T")
    close(r15["kl"], 2.39, rtol=3e-3, name="k l 1.5 T")
    close(r15["eta_m"], 10.1, rtol=3e-3, name="eta_m 1.5 T")
    close(r3["lam0"], 2.35, rtol=2e-3, name="lambda0 3 T")
    close(r3["k0l"], 0.535, rtol=3e-3, name="k0 l 3 T")
    close(r3["kl"], 4.79, rtol=3e-3, name="k l 3 T")
    close(r3["eta_m"], 20.2, rtol=3e-3, name="eta_m 3 T")
    # curl-curl : conduction : displacement = 1 : eta_m : eta_w^2 (orders of magnitude)
    close(r1["kl"] ** 2, 0.0256, rtol=5e-3, name="eta_w^2 (eps_r 80) at 0.1 T")
    close(r3["kl"] ** 2, 22.9, rtol=5e-3, name="eta_w^2 (eps_r 80) at 3 T")
    # consistency with Ch 8: eta_w^2 = eta_d eta_m
    for d in tab.values():
        eta_d = 2 * np.pi * d["f"] * EPS_R * EPS0 / SIGMA
        close(d["kl"] ** 2, eta_d * d["eta_m"], rtol=1e-10, name="eta_w^2 = eta_d eta_m")
    out["table"] = tab
    out["eta_d_3T"] = 2 * np.pi * larmor(3.0) * EPS_R * EPS0 / SIGMA
    close(out["eta_d_3T"], 1.14, rtol=3e-3, name="eta_d at 127.7 MHz (eps_r 80, sigma 0.5)")

    # ---- 22.3 rotating components
    b = 1.0
    co, cn = corotating(b, 1j * b)
    close(co, b, atol=1e-15, name="circular: co = b")
    close(cn, 0, atol=1e-15, name="circular: counter = 0")
    co, cn = corotating(b, 0)
    close(co, b / 2, atol=1e-15, name="linear: co = b/2")
    close(cn, b / 2, atol=1e-15, name="linear: counter = b/2")
    # time domain: b cos wt e_x = (b/2)(cos, -sin) + (b/2)(cos, +sin)
    wt = np.linspace(0, 2 * np.pi, 721)
    bx, by = physical(b, 0, wt)
    cw = 0.5 * b * np.vstack((np.cos(wt), -np.sin(wt)))
    ccw = 0.5 * b * np.vstack((np.cos(wt), np.sin(wt)))
    assert np.max(np.abs(np.vstack((bx, by)) - cw - ccw)) < 1e-14
    # the circular phasor (b, jb) is the clockwise field b(cos wt, -sin wt)
    cx, cy = physical(b, 1j * b, wt)
    assert np.max(np.abs(cx - b * np.cos(wt))) < 1e-14 and np.max(np.abs(cy + b * np.sin(wt))) < 1e-14
    # free precession for gamma > 0: dM/dt = gamma M x B0 e_z turns M clockwise seen from +z
    M = np.array([1.0, 0.0, 0.0])
    dM = np.cross(M, [0, 0, 1.0])
    assert dM[1] < 0                     # M along +x moves towards -y: clockwise
    # the same conventions as extended_checks.py (old edition)
    ext = extended_checks.run_checks()["rotating_phasor"]
    close(ext["co_rotating"], ext["amplitude"], rtol=1e-12, name="extended_checks co")
    close(ext["counter_rotating"], 0.0, atol=1e-12, name="extended_checks counter")
    # flip angle example
    b90 = flip_b1(90.0, 1e-3)
    close(b90 * 1e6, 5.87, rtol=1e-3, name="b_co for 90 deg in 1 ms (uT)")
    close(2 * b90 * 1e6, 11.74, rtol=1e-3, name="linear peak needed (uT)")
    out["b90"] = b90
    close(np.rad2deg(GAMMA * 2 * b90 * 1e-3), 180.0, rtol=1e-9, name="linear peak taken as b_co: 180 deg")
    # self-check: (3 uT, 0) -> 1.5 uT each; (b, -jb) is counter-rotating
    co, cn = corotating(3e-6, 0)
    close(abs(co), 1.5e-6, rtol=1e-12, name="3 uT linear: co 1.5 uT"); close(abs(cn), 1.5e-6, rtol=1e-12, name="counter 1.5 uT")
    co, cn = corotating(b, -1j * b)
    close(abs(co), 0.0, atol=1e-15, name="(b,-jb): co 0"); close(abs(cn), b, rtol=1e-12, name="(b,-jb): counter b")
    # |b_co|^2 - |b_counter|^2 = Im(Bx* By) for arbitrary phasors
    rng0 = np.random.default_rng(3)
    for _ in range(5):
        Bx, By = rng0.normal(size=2) + 1j * rng0.normal(size=2)
        co, cn = corotating(Bx, By)
        close(abs(co) ** 2 - abs(cn) ** 2, np.imag(np.conj(Bx) * By), rtol=1e-12, atol=1e-14, name="|co|^2-|counter|^2")
    # receive: B1 . M for clockwise M = M_perp (1, j) equals 2 M_perp b_counter
    for _ in range(3):
        Bx, By = rng0.normal(size=2) + 1j * rng0.normal(size=2)
        close(Bx * 1 + By * 1j, 2 * corotating(Bx, By)[1], rtol=1e-12, name="B1.M = 2 M b_counter")
    # ---- 22.4 linear vs quadrature drive with equal, uncoupled channels: Q = q I
    q = 1.0
    Qd = q * np.eye(2)
    sar = lambda c: float(np.real(np.conj(c) @ Qd @ c))
    c_lin, c_quad = np.array([2.0, 0.0]), np.array([1.0, 1j])
    # same co-rotating field: channel x gives (b, 0), channel y gives (0, b) per unit drive
    co_lin = corotating(c_lin[0] * b, c_lin[1] * b)[0]
    co_quad = corotating(c_quad[0] * b, c_quad[1] * b)[0]
    close(co_lin, co_quad, atol=1e-15, name="same B1+")
    close(sar(c_lin), 4 * q, rtol=1e-12, name="linear: 4 q")
    close(sar(c_quad), 2 * q, rtol=1e-12, name="quadrature: 2 q")
    close(sar(c_lin) / sar(c_quad), 2.0, rtol=1e-12, name="factor 2")
    # amplitude scaling: fields x a, SAR x |a|^2
    a = 1.5 * np.exp(0.3j)
    close(sar(a * c_quad), abs(a) ** 2 * sar(c_quad), rtol=1e-12, name="SAR scales with |a|^2")

    # ---- 22.5 averages (old rf:5 numbers)
    p_bar = local_sar_cycle(SAR_E)
    close(p_bar, 2.5, rtol=1e-12, name="RF-cycle average 2.5 W/kg")
    close(SAR_SIGMA * SAR_E**2 / SAR_RHO, 5.0, rtol=1e-12, name="instantaneous peak 5 W/kg")
    close(local_sar_cycle(SAR_E, convention="rms"), 5.0, rtol=1e-12, name="RMS reading: 5 W/kg")
    train = p_bar * 1e-3 / 1e-2
    close(train, 0.25, rtol=1e-12, name="pulse-train average 0.25 W/kg")
    close(local_sar_cycle(SAR_E, convention="rms") * 0.1, 0.5, rtol=1e-12, name="RMS reading: 0.5 W/kg")
    close(local_sar_cycle(SAR_E, convention="rms") * 0.05, 0.25, rtol=1e-12, name="RMS, 5% duty: 0.25 W/kg")
    close(C0 / larmor(3.0) / np.sqrt(EPS_R), 0.26, rtol=1e-2, name="wavelength at 3 T in eps_r 80 medium")
    ext = extended_checks.run_checks()["sar"]
    close(ext["during_pulse_W_per_kg"], 2.5, rtol=1e-12, name="extended_checks pulse")
    close(ext["pulse_train_W_per_kg"], 0.25, rtol=1e-12, name="extended_checks train")
    # instantaneous loss sigma E(t)^2 averaged over one RF period = sigma E^2/2
    wt = np.linspace(0, 2 * np.pi, 100001)
    inst = SAR_SIGMA * (SAR_E * np.cos(wt)) ** 2 / SAR_RHO
    close(np.trapezoid(inst, wt) / (2 * np.pi), 2.5, rtol=1e-9, name="cycle average by quadrature")
    # eq:rftimeaverage with a rectangular envelope, 1 ms on every 10 ms, over 10 periods
    t = np.linspace(0, 0.1, 1000001)
    env = ((t % 1e-2) < 1e-3).astype(float)
    close(np.trapezoid(env**2 * p_bar, t) / 0.1, 0.25, rtol=2e-4, name="eq:rftimeaverage")
    # spatial (1-D teaching) averages of a hot spot
    close(hotspot_sar(0.0), 2.5, rtol=1e-12, name="hot-spot peak local SAR")
    for L, ref in ((10e-3, 1.87), (20e-3, 1.10)):
        xs = np.linspace(-L / 2, L / 2, 200001)
        num = np.trapezoid(hotspot_sar(xs), xs) / L
        close(num, window_average(L), rtol=1e-8, name="window average closed form")
        close(window_average(L), ref, rtol=3e-3, name=f"window average L={L}")
    out["win10"], out["win20"] = window_average(10e-3), window_average(20e-3)
    # Q is Hermitian positive semidefinite for passive material (and Y_H = sigma + w eps'')
    rng = np.random.default_rng(7)
    npts = 400
    fields = [rng.normal(size=(npts, 3)) + 1j * rng.normal(size=(npts, 3)) for _ in range(3)]
    w = 2 * np.pi * larmor(3.0)
    A = rng.normal(size=(npts, 3, 3))
    sig = 0.2 * np.eye(3) + 0.05 * np.einsum("pij,pkj->pik", A, A)           # SPD conductivity
    B = rng.normal(size=(npts, 3, 3))
    epsp = EPS0 * (50 * np.eye(3) + np.einsum("pij,pkj->pik", B, B))           # SPD permittivity
    epspp = EPS0 * 5 * np.eye(3)                                               # dielectric loss eps''
    Y = sig + w * epspp + 1j * w * epsp
    rho = np.full(npts, 1000.0)
    dV = np.full(npts, 1e-6)
    Q = q_matrix(fields, Y, rho, dV)
    assert np.allclose(Q, Q.conj().T, atol=1e-12 * np.abs(Q).max()), "Q Hermitian"
    assert np.linalg.eigvalsh(Q).min() > -1e-12 * np.abs(Q).max(), "Q positive semidefinite"
    YH = 0.5 * (Y + np.conj(np.swapaxes(Y, -1, -2)))
    assert np.allclose(YH, sig + w * epspp), "Y_H = sigma + w eps''"
    c = rng.normal(size=3) + 1j * rng.normal(size=3)
    sar_direct = np.sum(np.einsum("pa,pab,pb->p", np.conj(sum(ck * fk for ck, fk in zip(c, fields))), YH,
                                  sum(ck * fk for ck, fk in zip(c, fields))) * dV) / (2 * np.sum(rho * dV))
    close(np.real(c.conj() @ Q @ c), np.real(sar_direct), rtol=1e-10, name="c^H Q c = direct SAR")
    assert abs(np.imag(c.conj() @ Q @ c)) < 1e-12 * abs(c.conj() @ Q @ c)
    return out


_CACHE: dict = {}


def run_cache() -> dict:
    if not _CACHE:
        _CACHE.update(run())
    return _CACHE


if __name__ == "__main__":
    o = run()
    print("Chapter 22 checks passed.")
    for name, f in ROWS:
        d = o["table"][name]
        print(f"  {name:9s} f = {f/1e6:9.4f} MHz  lambda0 = {d['lam0']:.4g} m  k0l = {d['k0l']:.3g}  "
              f"kl(80) = {d['kl']:.3g}  eta_m = {d['eta_m']:.3g}  delta = {d['delta']:.3g} m")
    print(f"  eta_d (eps_r 80, 127.7 MHz) = {o['eta_d_3T']:.3f}")
    print(f"  90 deg in 1 ms: b_co = {o['b90']*1e6:.3f} uT, linear peak {2*o['b90']*1e6:.2f} uT")
    print(f"  hot spot window averages: 10 mm {o['win10']:.3f} W/kg, 20 mm {o['win20']:.3f} W/kg")
