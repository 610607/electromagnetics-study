"""Chapter 18 (unified framework and shared observables): every number quoted in zh/ch18_framework.tex.

    python3 checks/check_ch18.py

The figure module figsrc/fig_ch18.py imports the model functions defined here, so the text, the
figures and the assertions share one implementation.  Inputs:
  * SAR: the old-edition teaching values (sigma = 0.5 S/m, E = 100 V/m, rho_m = 1000 kg/m^3) and the
    two-channel Q matrix of extended_checks.py (copied unchanged from the old edition);
  * nonlinear energy / co-energy: the gapped C-core of Chapter 14 (N = 200, iron path 0.30 m,
    A = 4 cm^2, g = 1 mm) with the synthetic B-H curve eq:syntheticbh of numerics.py;
  * reciprocity: the conducting square with the sigma = 10 inclusion of Chapters 10-11 (numerics P1 FEM);
  * adjoint sensitivity: the two-element bar of extended_checks.py (sigma = 0.5 S/m, A = 1 cm^2,
    h = 1 cm, I = 1 mA).
All are teaching inputs, not device data or exposure limits.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

import extended_checks
from numerics import MU0, square_mesh, stiffness, synthetic_B

# ---------------------------------------------------------------------------
# 18.5  local SAR, peak versus RMS
# ---------------------------------------------------------------------------
SAR_SIGMA, SAR_E, SAR_RHO = 0.5, 100.0, 1000.0


def local_sar(sigma, E_amp, rho, convention="peak"):
    """eq:localsar for a scalar conductivity; E_amp is a peak or an RMS amplitude."""
    f = 0.5 if convention == "peak" else 1.0
    return f * sigma * E_amp**2 / rho


# ---------------------------------------------------------------------------
# 18.6  two coherent channels: Q matrix and the cross-term profile of figure sar_cross_terms
# ---------------------------------------------------------------------------
E0_PROFILE = 100.0 * np.sqrt(2.0)      # peak amplitude of each channel field on the line (V/m)


def q_point():
    """2x2 Q (W/kg) at a point where both unit-drive fields are parallel, 100 V/m peak (extended_checks)."""
    return SAR_SIGMA * SAR_E**2 / (2 * SAR_RHO) * np.ones((2, 2))


def sar_of_drive(Q, c):
    c = np.asarray(c, complex)
    return float(np.real(np.conj(c) @ Q @ c))


def channel_fields(x):
    """Teaching profile on -1 <= x <= 1: parallel real fields e1, e2 (V/m per unit drive).
    e1^2 + e2^2 is constant, and at x = 0 both equal 100 V/m (the point of q_point)."""
    u = np.pi * (np.asarray(x) + 1) / 4
    return E0_PROFILE * np.cos(u), E0_PROFILE * np.sin(u)


def profile_sar(x, theta):
    """Local SAR along the line for c = (1, exp(j theta))."""
    e1, e2 = channel_fields(x)
    return SAR_SIGMA * np.abs(e1 + np.exp(1j * theta) * e2) ** 2 / (2 * SAR_RHO)


def profile_sum_of_maps(x):
    e1, e2 = channel_fields(x)
    return SAR_SIGMA * (e1**2 + e2**2) / (2 * SAR_RHO)


def q_line_average():
    """Q averaged over the whole line (uniform mass density): full rank, cross term survives."""
    x = np.linspace(-1, 1, 200001)
    e1, e2 = channel_fields(x)
    k = SAR_SIGMA / (2 * SAR_RHO)
    m = lambda f: np.trapezoid(f, x) / 2.0
    return k * np.array([[m(e1 * e1), m(e1 * e2)], [m(e2 * e1), m(e2 * e2)]])


# ---------------------------------------------------------------------------
# 18.3 / 18.4  nonlinear energy and co-energy of the Chapter-14 C-core
# ---------------------------------------------------------------------------
CC = dict(N=200, l=0.30, A=4.0e-4, g=1.0e-3)
BS, HS = 1.6, 800.0


def core_H(NI, g=CC["g"], l=CC["l"]):
    """eq:coreconnection with equal areas: l H + g B(H)/mu0 = NI."""
    return brentq(lambda H: l * H + g * synthetic_B(H) / MU0 - NI, 0.0, 1e7, xtol=1e-13, rtol=1e-15)


def linkage(i, g=CC["g"]):
    """Flux linkage lambda(i) of the N-turn coil (Wb)."""
    return CC["N"] * CC["A"] * float(synthetic_B(core_H(CC["N"] * i, g)))


def current_of_linkage(lam, g=CC["g"]):
    B = lam / (CC["N"] * CC["A"])
    H = brentq(lambda H: synthetic_B(H) - B, 0.0, 1e9, xtol=1e-13, rtol=1e-15)
    return (CC["l"] * H + g * B / MU0) / CC["N"]


def coenergy(i, g=CC["g"]):
    """W'(i, g) = int_0^i lambda di' at fixed geometry."""
    return quad(lambda s: linkage(s, g), 0.0, i, epsabs=1e-13, epsrel=1e-12, limit=200)[0]


def energy(lam, g=CC["g"]):
    """W(lambda, g) = int_0^lambda i dlambda' at fixed geometry."""
    return quad(lambda s: current_of_linkage(s, g), 0.0, lam, epsabs=1e-13, epsrel=1e-12, limit=200)[0]


def energy_coenergy(NI):
    i = NI / CC["N"]
    lam = linkage(i)
    W, Wc = energy(lam), coenergy(i)
    h = 1e-7
    F_co = (coenergy(i, CC["g"] + h) - coenergy(i, CC["g"] - h)) / (2 * h)          # +dW'/dg at fixed i
    F_en = -(energy(lam, CC["g"] + h) - energy(lam, CC["g"] - h)) / (2 * h)         # -dW/dg at fixed lambda
    Lsec = lambda gg: linkage(i, gg) / i
    F_naive = 0.5 * i**2 * (Lsec(CC["g"] + h) - Lsec(CC["g"] - h)) / (2 * h)
    B = lam / (CC["N"] * CC["A"])
    return dict(i=i, lam=lam, W=W, Wc=Wc, B=B, F_co=F_co, F_en=F_en, F_naive=F_naive,
                F_stress=-B**2 * CC["A"] / (2 * MU0), half=0.5 * lam * i)


def iron_densities(NI):
    """Energy and co-energy density in the iron at the operating point (J/m^3)."""
    H = core_H(NI)
    B = float(synthetic_B(H))
    wc = MU0 * H**2 / 2 + BS * HS * np.log(np.cosh(H / HS))      # int_0^H B dH
    return dict(H=H, B=B, wc=wc, w=B * H - wc, half=B * H / 2)


def lambda_curve(imax, n=400):
    i = np.linspace(0.0, imax, n)
    return i, np.array([linkage(s) for s in i])


MUR_LIN = 2000.0     # linear-iron model of ex:lumped-gapcore


def force_curves(NIs):
    """Attractive gap force (N) versus NI: nonlinear co-energy (= Maxwell stress), the naive
    1/2 i^2 dL_sec/dg, and the linear-iron model mu_r = 2000."""
    out = []
    for NI in NIs:
        r = energy_coenergy(NI)
        Blin = MU0 * NI / (CC["g"] + CC["l"] / MUR_LIN)
        out.append((-r["F_co"], -r["F_naive"], Blin**2 * CC["A"] / (2 * MU0)))
    return np.array(out)


# ---------------------------------------------------------------------------
# 18.7  reciprocity on the conducting square with an inclusion (numerics P1 FEM)
# ---------------------------------------------------------------------------
INC_C, INC_R, SIG_IN = np.array([0.62, 0.56]), 0.16, 10.0
ANODE, CATHODE = (0.125, 0.375), (0.625, 0.875)


def square_model(n=128):
    p, cells = square_mesh(n)
    centre = p[cells].mean(axis=1)
    sigma = np.where(np.sum((centre - INC_C) ** 2, axis=1) < INC_R**2, SIG_IN, 1.0)
    K, area, grads = stiffness(p, cells, sigma)
    return dict(n=n, p=p, cells=cells, K=K, area=area, grads=grads, sigma=sigma)


def electrode_load(m, patch, j):
    n = m["n"]
    top = np.arange(n * (n + 1), (n + 1) ** 2)
    x = m["p"][top, 0]
    b = np.zeros(len(m["p"]))
    for k in range(len(top) - 1):
        if x[k] >= patch[0] - 1e-12 and x[k + 1] <= patch[1] + 1e-12:
            h = x[k + 1] - x[k]
            b[top[k]] += j * h / 2
            b[top[k + 1]] += j * h / 2
    return b


def solve_floating(m, b):
    N = len(m["p"])
    free = np.arange(1, N)
    phi = np.zeros(N)
    phi[free] = spsolve(m["K"][free][:, free].tocsc(), b[free])
    return phi


def reciprocity_check(n=128, r0=(0.40, 0.45), Id=1e-3):
    m = square_model(n)
    w = ANODE[1] - ANODE[0]
    b1 = electrode_load(m, ANODE, 1.0 / w) + electrode_load(m, CATHODE, -1.0 / w)      # unit current pattern
    phi1 = solve_floating(m, b1)
    # a discrete current dipole: +Id at node a, -Id at the node one cell to its left -> p = Id h e_x
    h = 1.0 / n
    ia = int(np.argmin(np.sum((m["p"] - np.array(r0)) ** 2, axis=1)))
    ib = int(np.argmin(np.sum((m["p"] - (m["p"][ia] - [h, 0])) ** 2, axis=1)))
    b2 = np.zeros(len(m["p"])); b2[ia] += Id; b2[ib] -= Id
    phi2 = solve_floating(m, b2)
    V_meas = b1 @ phi2                       # electrode-weighted voltage "anode minus cathode"
    V_recip = phi1 @ b2
    # lead field at the dipole: element field of the stimulation solution near r0
    E1 = -np.einsum("ei,eid->ed", phi1[m["cells"]], m["grads"])
    mid = (m["p"][ia] + m["p"][ib]) / 2
    near = np.argsort(np.sum((m["p"][m["cells"]].mean(axis=1) - mid) ** 2, axis=1))[:2]
    e_lead = E1[near].mean(axis=0)
    p_vec = np.array([Id * h, 0.0])
    return dict(V_meas=V_meas, V_recip=V_recip, V_lead=-(e_lead @ p_vec), e_lead=e_lead)


# ---------------------------------------------------------------------------
# 18.7  adjoint sensitivity on the two-element bar
# ---------------------------------------------------------------------------
BAR = dict(sigma=(0.5, 0.5), A=1e-4, h=0.01, I=1e-3)


def bar_K(s1, s2, A=BAR["A"], h=BAR["h"]):
    """Reduced stiffness of the bar with node 2 grounded (unknowns phi0, phi1)."""
    return (A / h) * np.array([[s1, -s1], [-s1, s1 + s2]])


def bar_adjoint():
    s1, s2 = BAR["sigma"]
    K = bar_K(s1, s2)
    f = np.array([BAR["I"], 0.0])
    phi = np.linalg.solve(K, f)
    dK = [(BAR["A"] / BAR["h"]) * np.array([[1.0, -1.0], [-1.0, 1.0]]),
          (BAR["A"] / BAR["h"]) * np.array([[0.0, 0.0], [0.0, 1.0]])]
    out = {}
    for name, q in (("phi1", np.array([0.0, 1.0])), ("phi0", np.array([1.0, 0.0]))):
        z = np.linalg.solve(K.T, q)
        sens = np.array([-(z @ (dKe @ phi)) for dKe in dK])
        eps = 1e-7
        fd = np.array([(q @ np.linalg.solve(bar_K(s1 + eps * (k == 0), s2 + eps * (k == 1)), f) - q @ phi) / eps
                       for k in range(2)])
        out[name] = dict(Q=q @ phi, z=z, sens=sens, fd=fd)
    return phi, out


# ---------------------------------------------------------------------------
def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:62s} {x:.6g}")
    return x


def run_checks():
    print("18.5 SAR")
    close(local_sar(SAR_SIGMA, SAR_E, SAR_RHO, "peak"), 2.5, rtol=1e-12, name="local SAR, 100 V/m peak (W/kg)")
    close(local_sar(SAR_SIGMA, SAR_E, SAR_RHO, "rms"), 5.0, rtol=1e-12, name="local SAR, 100 V/m RMS (W/kg)")
    close(0.5 * SAR_SIGMA * SAR_E**2, 2500.0, rtol=1e-12, name="Ch 7 loss density, peak (W/m^3)")
    ext = extended_checks.run_checks()["sar"]
    Q = q_point()
    assert np.allclose(Q, np.array(ext["Q_W_per_kg"])), "Q equals extended_checks"
    close(sar_of_drive(Q, [1, 1]), ext["two_equal_in_phase_W_per_kg"], rtol=1e-12, name="in phase (W/kg)")
    close(sar_of_drive(Q, [1, 1]), 10.0, rtol=1e-12, name="in phase = 10 W/kg")
    assert abs(sar_of_drive(Q, [1, -1])) < 1e-12 and ext["two_equal_opposed_W_per_kg"] == 0
    print("  ok  opposite phase = 0 W/kg")
    close(sar_of_drive(Q, [1, 1j]), 5.0, rtol=1e-12, name="quadrature (90 deg) (W/kg)")
    close(Q[0, 0] + Q[1, 1], 5.0, rtol=1e-12, name="sum of the two single-channel maps (W/kg)")
    ev = np.linalg.eigvalsh(Q)
    assert np.allclose(ev, [0.0, 5.0]), ev
    print("  ok  eigenvalues of Q: 0 and 5 W/kg (per unit-norm drive)")
    th = np.linspace(0, 2 * np.pi, 361)
    s = np.array([sar_of_drive(Q, [1, np.exp(1j * t)]) for t in th])
    assert np.allclose(s, 5 * (1 + np.cos(th))), "SAR(theta) = 5 (1 + cos theta)"
    print("  ok  SAR(theta) = 5(1 + cos theta) W/kg")
    x = np.linspace(-1, 1, 2001)
    assert np.allclose(profile_sum_of_maps(x), 5.0), "sum of maps is flat"
    close(profile_sar(0.0, 0.0), 10.0, rtol=1e-12, name="profile centre, in phase")
    assert abs(profile_sar(0.0, np.pi)) < 1e-12
    close(profile_sar(-1.0, 0.0), 5.0, rtol=1e-12, name="profile end, in phase")
    assert np.allclose(profile_sar(x, np.pi / 2), 5.0)
    e1, e2 = channel_fields(0.0)
    close(e1, 100.0, rtol=1e-12, name="each channel field at the centre (V/m)")
    Qa = q_line_average()
    close(Qa[0, 0], 2.5, rtol=1e-6, name="line-averaged Q11 (W/kg)")
    close(Qa[0, 1], 5 / np.pi, rtol=1e-6, name="line-averaged Q12 = 5/pi (W/kg)")
    close(sar_of_drive(Qa, [1, 1]), 8.18, rtol=2e-3, name="line average, in phase (W/kg)")
    close(sar_of_drive(Qa, [1, -1]), 1.82, rtol=3e-3, name="line average, opposite (W/kg)")
    close(sar_of_drive(Qa, [1, 1j]), 5.0, rtol=1e-6, name="line average, quadrature (W/kg)")

    print("self-check numbers")
    Erms = 100.0
    close(SAR_SIGMA * (2 * Erms)**2 / SAR_RHO, 20.0, rtol=1e-12, name="two RMS channels in phase (W/kg)")
    close(2 * SAR_SIGMA * Erms**2 / SAR_RHO, 10.0, rtol=1e-12, name="... sum of the two single-channel maps (W/kg)")
    close(0.2 / 1e-3 * 10e-9 * 1e6, 2.0, rtol=1e-12, name="lead field 0.2 V/m per mA, p = 10 nA m: |V| (uV)")
    print("18.3/18.4 energy and co-energy (Ch 14 C-core, synthetic B-H)")
    r = energy_coenergy(2000.0)
    close(r["B"], 1.60, rtol=3e-3, name="B at NI = 2000 A (Ch 14: 1.60 T)")
    close(r["lam"], 0.128, rtol=3e-3, name="flux linkage (Wb)")
    close(r["W"], 0.510, rtol=2e-3, name="energy W (J)")
    close(r["Wc"], 0.767, rtol=2e-3, name="co-energy W' (J)")
    close(r["W"] + r["Wc"], r["lam"] * r["i"], rtol=1e-8, name="W + W' = lambda i (J)")
    close(r["half"], 0.638, rtol=2e-3, name="lambda i / 2 (J)")
    close(r["Wc"] / r["W"], 1.50, rtol=5e-3, name="W'/W")
    close(r["F_co"], r["F_en"], rtol=1e-5, name="force: +dW'/dg|_i equals -dW/dg|_lambda (N)")
    close(r["F_co"], r["F_stress"], rtol=1e-5, name="... equals -B^2 A/(2 mu0) (N)")
    close(-r["F_co"], 405.0, rtol=3e-3, name="attractive force (N)")
    close(-r["F_naive"], 31.2, rtol=5e-3, name="naive 1/2 i^2 dL_sec/dg (N)")
    close(r["F_naive"] / r["F_co"], 0.077, rtol=1e-2, name="naive / correct")
    fc = force_curves([2000.0])[0]
    close(fc[2], 760.0, rtol=3e-3, name="linear-iron model (mu_r = 2000) force at 2000 A (N)")
    close(MU0 * 2000.0 / (CC["g"] + CC["l"] / MUR_LIN), 2.19, rtol=3e-3, name="... its gap field (Ch 14: 2.19 T)")
    r4 = energy_coenergy(400.0)
    close(r4["Wc"] / r4["W"], 1.004, rtol=1e-3, name="W'/W at NI = 400 A")
    close(r4["W"], 0.0336, rtol=3e-3, name="W at NI = 400 A (J)")
    close(r4["F_naive"] / r4["F_co"], 0.992, rtol=1e-3, name="naive / correct at 400 A")
    close(-r4["F_co"], 28.3, rtol=3e-3, name="force at 400 A (N)")
    Wgap = r["B"]**2 / (2 * MU0) * CC["A"] * CC["g"]
    close(Wgap, 0.405, rtol=3e-3, name="energy in the gap (J)")
    d0 = iron_densities(2000.0)
    close(d0["w"] * CC["A"] * CC["l"], 0.104, rtol=5e-3, name="energy in the iron (J)")
    close(Wgap + d0["w"] * CC["A"] * CC["l"], r["W"], rtol=2e-3, name="gap + iron = W (J)")
    close(d0["wc"] * CC["A"] * CC["l"], 0.362, rtol=3e-3, name="co-energy in the iron (J)")
    close(Wgap / CC["g"], -r["F_co"], rtol=2e-3, name="gap energy / g = force (N)")
    d = iron_densities(2000.0)
    close(d["H"], 2434.0, rtol=1e-3, name="iron H (A/m)")
    close(d["w"], 870.0, rtol=2e-3, name="iron energy density (J/m^3)")
    close(d["wc"], 3013.0, rtol=2e-3, name="iron co-energy density (J/m^3)")
    close(d["half"], 1942.0, rtol=2e-3, name="BH/2 (J/m^3)")
    close(d["wc"] / d["w"], 3.46, rtol=3e-3, name="w'/w in the iron")

    print("18.7 reciprocity (square with inclusion, P1 FEM)")
    rc = reciprocity_check()
    rel = abs(rc["V_meas"] - rc["V_recip"]) / abs(rc["V_meas"])
    assert rel < 1e-13, rc
    print(f"  ok  discrete reciprocity b1.phi2 = phi1.b2 ({rc['V_meas']:.6e}, relative difference {rel:.1e} < 1e-13)")
    rel = abs(rc["V_lead"] / rc["V_meas"] - 1)
    assert rel < 1e-13, rel
    print(f"  ok  V = -e_lead . p (relative difference {rel:.1e} < 1e-13)")

    print("18.7 adjoint sensitivity (two-element bar)")
    phi, out = bar_adjoint()
    assert np.allclose(phi, [0.4, 0.2]) and np.allclose(extended_checks.run_checks()["two_element_bar"]["phi_V"][:2], phi)
    print("  ok  bar potentials 0.4 V, 0.2 V (extended_checks)")
    a = out["phi1"]
    close(a["Q"], 0.2, rtol=1e-12, name="Q = phi1 (V)")
    assert np.allclose(a["z"], [200.0, 200.0]), a["z"]
    print("  ok  adjoint potentials z = (200, 200) ohm")
    assert abs(a["sens"][0]) < 1e-14
    close(a["sens"][1], -0.4, rtol=1e-12, name="dphi1/dsigma2 (V per S/m)")
    assert np.allclose(a["sens"], a["fd"], atol=1e-6), (a["sens"], a["fd"])
    print("  ok  adjoint sensitivities agree with finite differences")
    b = out["phi0"]
    assert np.allclose(b["z"], phi / BAR["I"])
    close(b["sens"][0], -0.4, rtol=1e-12, name="dphi0/dsigma1 (V per S/m)")
    close(b["sens"][1], -0.4, rtol=1e-12, name="dphi0/dsigma2 (V per S/m)")
    # -int_e E_fwd . E_adj dV : E_fwd = 20 V/m, E_adj = 2e4 V/m per A, volume 1e-6 m^3
    close(-20.0 * 2.0e4 * 1e-6, -0.4, rtol=1e-12, name="-E_fwd E_adj V_e for element 2")
    print("check_ch18: all assertions passed")
    return True


if __name__ == "__main__":
    run_checks()
