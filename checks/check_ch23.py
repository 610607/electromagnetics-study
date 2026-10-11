"""Chapter 23 (integrated simulation projects): every number quoted in zh/ch23_projects.tex.

    python3 checks/check_ch23.py

The projects reuse the models of earlier chapters instead of recomputing them differently:
  * model contract: the anisotropic conductivity of ex:electric-aniso (sigma = diag(1, 4) S/m) rotated by 30 deg,
    and a Biot-Savart polygon loop mirrored in a plane (axial-vector rule);
  * project 1: the conducting square with the sigma = 10 inclusion (fig_ch10 / fig_ch11, numerics.py P1 FEM)
    and the polygon loop of check_ch04 (eq:loopaxis);
  * project 2: the biplanar x-gradient teaching design of Chapter 20 (check_ch20) and the slew numbers of Ch 19;
  * project 3: the multi-pillar network of ex:lumped-pillars (check_ch14 inputs, synthetic B-H curve) with a small
    coil linking the main flux loop: frozen secant vs consistent tangent, and the central-difference check
    eq:biascheck;
  * project 4: the passive compartment cable of Chapter 21 (check_ch21.passive_cable) driven by two longitudinal
    field patterns with the same mean square (same SAR), plus the spatial band-pass response of the cable;
  * interface: the two-element bar of Ch 16 (old proj:2) read with the wrong length unit, and the old
    extended_checks unit-map conversion.
All inputs are teaching inputs, not device data or thresholds.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "figsrc", ROOT / "checks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from numerics import MU0, synthetic_B  # noqa: E402


def close(x, ref, rtol=5e-3, atol=0.0, name=""):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:66s} {x:.6g}")
    return x


# ---------------------------------------------------------------------------
# 23.1  Model contract: points, polar vectors, axial vectors and tensors under a change of frame
# ---------------------------------------------------------------------------
def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def tensor_example(theta_deg=30.0):
    """Fibre frame rotated by theta against the lab frame; sigma_fibre = diag(1, 4) S/m (ex:electric-aniso)."""
    Q = rot2(np.radians(theta_deg))
    sig_f = np.diag([1.0, 4.0])
    sig_lab = Q @ sig_f @ Q.T
    E_lab = np.array([1.0, 0.0])                 # V/m, applied along the lab x axis
    J_lab = sig_lab @ E_lab
    E_f = Q.T @ E_lab                            # the same field in fibre coordinates
    J_f = sig_f @ E_f
    wrong_once = (Q @ sig_f) @ E_lab             # tensor rotated "like a vector"
    return dict(Q=Q, sig_lab=sig_lab, J_lab=J_lab, J_f=J_f, E_f=E_f, wrong_unrotated=sig_f @ E_lab,
                wrong_once=wrong_once, p_lab=E_lab @ J_lab, p_f=E_f @ J_f)


def biot_savart_polygon(P, obs):
    """B of a closed polygon P (k x 3, last = first) carrying 1 A, at the points obs (n x 3); exact segment formula."""
    B = np.zeros((len(obs), 3))
    for k in range(len(P) - 1):
        p1, p2 = P[k], P[k + 1]
        dl = p2 - p1
        r1, r2 = obs - p1, obs - p2
        n1, n2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
        c = np.cross(dl, r1)
        cc = np.sum(c * c, axis=1)
        B += MU0 / (4 * np.pi) * c * ((r1 @ dl) / n1 - (r2 @ dl) / n2)[:, None] / cc[:, None]
    return B


def axial_vector_check():
    """Mirror a tilted current loop in the plane x = 0 (det Q = -1): B' = det(Q) Q B, not Q B."""
    t = np.linspace(0, 2 * np.pi, 201)
    loop = np.stack([0.3 + 0.1 * np.cos(t), 0.1 * np.sin(t), 0.05 * np.sin(t)], axis=1)
    Q = np.diag([-1.0, 1.0, 1.0])
    obs = np.array([[0.05, 0.02, 0.1], [0.4, -0.1, 0.0]])
    B = biot_savart_polygon(loop, obs)
    Bm = biot_savart_polygon(loop @ Q.T, obs @ Q.T)
    return B, Bm, Q


# ---------------------------------------------------------------------------
# Project 1: TES and TMS on one mesh
# ---------------------------------------------------------------------------
def project1(n=64):
    from fig_ch10 import model, tes_uniform
    from fig_ch11 import tms_solve
    from numerics import stiffness
    m = model(n)
    tes, tms = tes_uniform(m), tms_solve(m)
    out = dict(sum_b_tes=float(tes["b"].sum()), sum_b_tms=float(tms["b"].sum()),
               scale_b=float(np.abs(tms["b"]).max()))
    # adding a constant to phi does not change E
    E_shift = -np.einsum("ei,eid->ed", (tes["phi"] + 0.37)[m["cells"]], m["grads"])
    out["shift"] = float(np.abs(E_shift - tes["E"]).max())
    # doubling the source doubles E (TES current, TMS dB/dt): same K, right-hand side x2
    from fig_ch10 import solve_neumann
    phi2 = solve_neumann(m, 2 * tes["b"])
    E2 = -np.einsum("ei,eid->ed", phi2[m["cells"]], m["grads"])
    out["tes_double"] = float(np.abs(E2 - 2 * tes["E"]).max() / np.abs(tes["E"]).max())
    phi2 = solve_neumann(m, 2 * tms["b"])
    E2 = 2 * tms["Ep"] - np.einsum("ei,eid->ed", phi2[m["cells"]], m["grads"])
    out["tms_double"] = float(np.abs(E2 - 2 * tms["E"]).max() / np.abs(tms["E"]).max())
    # weak orthogonality of the TMS solution (eq:tmsorthogonal) and weak conservation of TES
    w = m["area"] * m["sigma"]
    out["ortho"] = float(abs(np.sum(w * np.einsum("ed,ed->e", tms["E"], tms["corr"]))))
    out["K_same"] = bool(np.allclose(m["K"].toarray(), stiffness(m["p"], m["cells"], m["sigma"])[0].toarray()))
    return out


def loop_center_ratio(nseg):
    """Centre field of an n-gon / circle of the same radius: n tan(pi/n) / pi (check_ch04)."""
    import check_ch04 as C04  # noqa: F401  (imports the same Biot-Savart routine)
    return C04.biot_savart_loop_axis(0.0, nseg=nseg) / C04.loop_axis_exact(0.0)


# ---------------------------------------------------------------------------
# Project 2: one source basis, many read-outs (Ch 20 biplanar x-gradient)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def project2():
    import check_ch20 as C20
    c, _ = C20.design_at(C20.TARGET_ERR, "R")
    m = C20.metrics(c)
    w = C20.winding_stats(c)
    Lc = 2 * m["W"] / C20.I_WIRE**2
    eta = C20.G_T / C20.I_WIRE                  # T/(m A)
    Idot = C20.I_WIRE / C20.RAMP                # A/s
    return dict(err=m["err"], P=m["P"], W=m["W"], L=Lc, R=m["P"] / C20.I_WIRE**2, V=Lc * Idot, eta=eta,
                Idot=Idot, Gdot=eta * Idot, Kmax=w["Kmax"], fmax=w["fmax"], Ftot=w["Ftot"], F=w["F"])


# ---------------------------------------------------------------------------
# Project 3: nonlinear bias first, then the small signal (multi-pillar network, eq:biascheck)
# ---------------------------------------------------------------------------
def pillar_parts():
    import check_ch14 as C14
    p = C14.PILLAR
    Rm = p["lm"] / (MU0 * p["mu_rec"] * p["Am"])
    Rg = p["g"] / (MU0 * p["Am"])
    return p, Rm, Rg, C14.mu_diff


def pillar_solve(N, Ap=0.005, dF=0.0):
    """Loop law of ex:lumped-pillars with an extra MMF dF (a small coil linking the main loop)."""
    p, Rm, Rg, _ = pillar_parts()

    def res(Hp):
        Phi = N * Ap * synthetic_B(Hp)
        Hm = (Phi / p["Am"] - p["Br"]) / (MU0 * p["mu_rec"])
        return 2 * Hm * p["lm"] + Phi / p["Am"] * p["g"] / MU0 + Hp * p["lp"] - dF

    Hp = brentq(res, 0.0, 1.0e8, xtol=1e-13, rtol=1e-15)
    Phi = N * Ap * float(synthetic_B(Hp))
    return Hp, Phi, Phi / p["Am"]


def pillar_small_signal(N, Ap=0.005):
    p, Rm, Rg, mu_diff = pillar_parts()
    Hp, Phi, Bg = pillar_solve(N, Ap)
    mus = float(synthetic_B(Hp)) / Hp
    mud = float(mu_diff(Hp))
    Rs = p["lp"] / (mus * N * Ap)
    Rd = p["lp"] / (mud * N * Ap)
    tan = 1.0 / (2 * Rm + Rg + Rd) / p["Am"]     # dB_g/dF, T/A
    sec = 1.0 / (2 * Rm + Rg + Rs) / p["Am"]
    return dict(Hp=Hp, Bp=float(synthetic_B(Hp)), Bg=Bg, mus=mus / MU0, mud=mud / MU0, tan=tan, sec=sec,
                share_d=Rd / (2 * Rm + Rg + Rd))


def central_difference(N, eps, Ap=0.005):
    return (pillar_solve(N, Ap, eps)[2] - pillar_solve(N, Ap, -eps)[2]) / (2 * eps)


# ---------------------------------------------------------------------------
# Project 4: one electric field, nerve drive and heating
# ---------------------------------------------------------------------------
def cable_response(pattern, ds=0.1, L=40.0):
    """Steady passive V_m on a long sealed straight cable (lengths in lambda, fields in E_rms).

    pattern 'uniform': E_s = 1;  pattern 'cos': E_s = sqrt(2) cos(s/lambda) (same mean square).
    The extracellular drive is the exact path integral V_e = -int E_s ds (eq:nervedifference)."""
    import check_ch21 as C21
    n = int(round(L / ds))
    s = (np.arange(n) + 0.5) * ds - L / 2
    if pattern == "uniform":
        Es, Ve = np.ones_like(s), -s
    else:
        Es, Ve = np.sqrt(2) * np.cos(s), -np.sqrt(2) * np.sin(s)
    V, _ = C21.passive_cable(Ve, ds)
    return s, Es, V


def bandpass(k_lambda):
    """Steady passive V_m amplitude for E_s = sqrt(2) E_rms cos(k s) on an infinite cable, in E_rms lambda:
    tau dV/dt = lambda^2 V'' - V + lambda^2 f,  f = -dE_s/ds  ->  |V| = sqrt(2) k lambda^2 / (1 + k^2 lambda^2)."""
    k = np.asarray(k_lambda, float)
    return np.sqrt(2) * k / (1 + k * k)


# ---------------------------------------------------------------------------
def main():
    print("23.1 model contract")
    t = tensor_example()
    close(t["sig_lab"][0, 0], 1.75, rtol=1e-12, name="rotated tensor sigma_xx (S/m)")
    close(t["sig_lab"][1, 1], 3.25, rtol=1e-12, name="rotated tensor sigma_yy (S/m)")
    close(abs(t["sig_lab"][0, 1]), 1.299, rtol=1e-3, name="rotated tensor |sigma_xy| (S/m)")
    assert np.allclose(t["J_lab"], t["Q"] @ t["J_f"])           # J transforms like E
    close(t["J_lab"][1], -1.299, rtol=1e-3, name="J_y in the lab frame (A/m^2)")
    close(t["wrong_once"][0], 0.866, rtol=1e-3, name="tensor rotated like a vector: J_x")
    close(t["wrong_once"][1], 0.5, rtol=1e-3, name="tensor rotated like a vector: J_y (wrong side)")
    close(t["p_lab"], t["p_f"], rtol=1e-12, name="loss density E.J invariant (W/m^3)")
    close(t["p_lab"], 1.75, rtol=1e-12, name="E.J for E = (1, 0) V/m")
    close(np.degrees(np.arctan2(abs(t["J_lab"][1]), t["J_lab"][0])), 36.6, rtol=2e-3,
          name="angle between J and E (deg)")
    assert not np.allclose(t["wrong_once"], t["J_lab"]) and not np.allclose(t["wrong_unrotated"], t["J_lab"])
    print("  ok  rotating the tensor once (Q sigma) or not at all gives a different current")
    B, Bm, Q = axial_vector_check()
    assert np.allclose(Bm, (np.linalg.det(Q) * (Q @ B.T)).T, rtol=1e-10, atol=1e-18)
    assert not np.allclose(Bm, (Q @ B.T).T, rtol=1e-3)
    print("  ok  mirrored loop: B' = det(Q) Q B (axial vector), not Q B")

    print("project 1: TES and TMS on one mesh")
    from fig_ch10 import INC_C, INC_R, SIG_IN
    assert np.allclose(INC_C, (0.62, 0.56)) and INC_R == 0.16 and SIG_IN == 10.0
    print("  ok  contract example: inclusion centre (0.62, 0.56), radius 0.16, sigma = 10 (fig_ch10)")
    p1 = project1()
    assert p1["K_same"]
    assert abs(p1["sum_b_tes"]) < 1e-14 and abs(p1["sum_b_tms"]) < 1e-13 * p1["scale_b"]
    print(f"  ok  sum of right-hand sides: TES {p1['sum_b_tes']:.1e}, TMS {p1['sum_b_tms']:.1e} (exact conservation)")
    assert p1["shift"] < 1e-12 and p1["tes_double"] < 1e-10 and p1["tms_double"] < 1e-10
    print("  ok  phi + const leaves E unchanged; doubling the source doubles E (TES and TMS)")
    assert p1["ortho"] < 1e-14
    print(f"  ok  TMS weak orthogonality {p1['ortho']:.1e}")
    for nseg, ref in ((6, 0.103), (12, 0.0235), (24, 0.0057), (48, 0.00143)):
        close(loop_center_ratio(nseg) - 1, ref, rtol=0.02, name=f"{nseg}-gon loop: centre-field excess")
    close((loop_center_ratio(24) - 1) / (loop_center_ratio(48) - 1), 4.0, rtol=0.01,
          name="polygon source error quartered when segments double")

    print("project 2: one source basis, many read-outs (Ch 20 biplanar x-gradient)")
    p2 = project2()
    close(p2["err"], 0.01, rtol=1e-6, name="RMS field error on the check points")
    close(p2["P"], 513, rtol=2e-3, name="dissipated power at 10 mT/m, 1 mm Cu sheet (W)")
    close(p2["W"], 0.557, rtol=2e-3, name="stored energy at 10 mT/m (J)")
    close(p2["L"] * 1e6, 111, rtol=5e-3, name="inductance at 100 A per turn (uH)")
    close(p2["R"], 0.0513, rtol=2e-3, name="sheet resistance P/I^2 (ohm)")
    close(p2["eta"] * 1e3, 0.1, rtol=1e-12, name="efficiency eta (mT/m per A)")
    close(p2["Idot"], 1e6, rtol=1e-12, name="0 -> 100 A in 0.1 ms (A/s)")
    close(p2["V"], 111, rtol=5e-3, name="inductive voltage L dI/dt (V)")
    close(p2["Gdot"], 100.0, rtol=1e-12, name="gradient slew eta dI/dt (T/m/s)")
    close(p2["Gdot"] * 0.2, 20.0, rtol=1e-12, name="dB_z/dt at x = 0.2 m (T/s), Ch 19")
    close(p2["fmax"] / 1e3, 1.72, rtol=0.01, name="max force density |K| B0 at 0.1 T (kN/m^2)")
    close(p2["Ftot"], 202, rtol=0.01, name="int |f| dS over one plate (N)")
    assert np.all(np.abs(p2["F"]) < 1e-9 * p2["Ftot"])
    print("  ok  net force on each plate in a uniform B0 vanishes")
    close(p2["P"] * 0.1**2 / p2["P"], 0.01, rtol=1e-12, name="power at 1 mT/m relative to 10 mT/m (quadratic)")
    # self-check 4: 10 -> 5 mT/m (current halved)
    close(p2["P"] / 4, 128, rtol=3e-3, name="self-check: power at 5 mT/m (W)")
    close(p2["W"] / 4, 0.139, rtol=3e-3, name="self-check: stored energy at 5 mT/m (J)")
    close(p2["V"] / 2, 56, rtol=6e-3, name="self-check: ramp voltage at 5 mT/m, same rise time (V)")

    print("project 3: nonlinear bias, then the small signal (multi-pillar network)")
    s4, s5, s6 = (pillar_small_signal(N) for N in (4, 5, 6))
    close(s4["Bg"], 0.162, rtol=3e-3, name="N = 4: B_g (T), as in ex:lumped-pillars")
    close(s4["Bp"], 1.62, rtol=3e-3, name="N = 4: pillar B (T)")
    close(s4["mus"], 69, rtol=1e-3, name="N = 4: pillar mu_sec / mu0")
    close(s4["mud"], 1.0, rtol=1e-6, name="N = 4: pillar mu_diff / mu0")
    close(s4["tan"] * 1e6, 0.194, rtol=2e-3, name="N = 4: tangent dB_g/dF (uT/A)")
    close(s4["sec"] * 1e6, 2.23, rtol=2e-3, name="N = 4: frozen secant dB_g/dF (uT/A)")
    close(s4["sec"] / s4["tan"], 11.5, rtol=1e-3, name="N = 4: secant / tangent")
    close(s6["Bp"], 1.27, rtol=2e-3, name="N = 6: pillar B (T)")
    close(s6["Bg"], 0.191, rtol=2e-3, name="N = 6: B_g (T)")
    close(s6["tan"] * 1e6, 2.60, rtol=2e-3, name="N = 6: tangent dB_g/dF (uT/A)")
    close(s6["sec"] / s6["tan"] - 1, 0.0070, rtol=0.02, name="N = 6: secant overestimates by")
    close(s6["mus"], 1170, rtol=2e-3, name="N = 6: pillar mu_sec / mu0")
    close(s6["mud"], 590, rtol=2e-3, name="N = 6: pillar mu_diff / mu0 (about half the secant)")
    close(s6["share_d"], 0.014, rtol=0.02, name="N = 6: pillars' incremental reluctance / loop total")
    close(s6["tan"] / s4["tan"], 13.4, rtol=5e-3, name="small-signal efficiency N = 6 / N = 4")
    close(s5["Bp"], 1.52, rtol=2e-3, name="N = 5: pillar B (T)")
    close(s5["sec"] / s5["tan"] - 1, 0.049, rtol=0.02, name="N = 5: secant overestimates by")
    err = {e: abs(central_difference(5, e) - s5["tan"]) / s5["tan"] for e in (3000.0, 1000.0, 100.0, 10.0, 1.0, 0.1,
                                                                                1e-3, 1e-6)}
    close(err[3000.0], 1.1e-2, rtol=0.03, name="N = 5, eps = 3000 A: relative difference")
    close(err[1000.0], 9.6e-4, rtol=0.03, name="eps = 1000 A")
    close(err[100.0], 9.4e-6, rtol=0.03, name="eps = 100 A")
    close(err[10.0], 9.4e-8, rtol=0.05, name="eps = 10 A")
    assert err[1.0] < 1e-9 and err[0.1] < 1e-10
    print(f"  ok  eps = 1 A, 0.1 A: {err[1.0]:.1e}, {err[0.1]:.1e} (plateau)")
    assert err[1e-6] > 10 * err[0.1]
    print(f"  ok  eps = 1e-3 A, 1e-6 A: {err[1e-3]:.1e}, {err[1e-6]:.1e} (round-off grows again)")
    close(err[1000.0] / err[100.0], 100, rtol=0.03, name="truncation error ~ eps^2")
    # deep saturation: the response is linear over a wide range (exactly linear branch of the curve)
    assert abs(central_difference(4, 1000.0) - s4["tan"]) / s4["tan"] < 1e-10

    print("project 4: one field, nerve drive and heating")
    s, Es, Vu = cable_response("uniform")
    inner = np.abs(s) < 12
    close(np.mean(Es**2), 1.0, rtol=1e-12, name="uniform pattern: mean E_s^2 / E_rms^2")
    assert np.max(np.abs(Vu[inner])) < 1e-3
    print(f"  ok  uniform pattern: interior |V_m| < 1e-3 E_rms lambda ({np.max(np.abs(Vu[inner])):.1e})")
    s, Es, Vc = cable_response("cos", ds=0.01)
    xs = np.linspace(0, 2 * np.pi, 200001)
    close(np.trapezoid(2 * np.cos(xs) ** 2, xs) / (2 * np.pi), 1.0, rtol=1e-9,
          name="cosine pattern: mean E_s^2 over a period / E_rms^2")
    inner = np.abs(s) < 12
    close(np.max(np.abs(Vc[inner])), 1 / np.sqrt(2), rtol=1e-3, name="cosine pattern: |V_m| max (E_rms lambda)")
    close(bandpass(1.0), 0.707, rtol=1e-3, name="band-pass amplitude at k lambda = 1")
    kk = np.logspace(-3, 3, 20001)
    close(kk[np.argmax(bandpass(kk))], 1.0, rtol=2e-3, name="largest response at k lambda")
    close(bandpass(0.1) / bandpass(1.0), 0.198, rtol=3e-3, name="k lambda = 0.1 relative to 1")
    close(bandpass(10.0) / bandpass(1.0), 0.198, rtol=3e-3, name="k lambda = 10 relative to 1")
    for ds, ref in ((0.5, 0.011), (0.25, 0.0028), (0.1, 0.00042)):
        sd, _, V = cable_response("cos", ds=ds)
        close(1 - np.max(np.abs(V[np.abs(sd) < 12])) * np.sqrt(2), ref, rtol=0.03, name=f"cable discretization error, ds = {ds}")

    print("23.7 from field arrays to an interface")
    phi = np.array([0.4, 0.2, 0.0])
    close(-(phi[1] - phi[0]) / 0.01, 20.0, rtol=1e-12, name="bar: E with h = 0.01 m (V/m)")
    close(-(phi[1] - phi[0]) / 10.0, 0.02, rtol=1e-12, name="bar: E if h = 10 (mm) is read as metres (V/m)")
    close(0.5 * 20.0**2, 200.0, rtol=1e-12, name="loss density sigma E^2 (W/m^3)")
    close(0.5 * 0.02**2, 2e-4, rtol=1e-12, name="... with the wrong length unit (W/m^3)")
    close(2e-5 * 1e6, 20.0, rtol=1e-12, name="2e-5 (V/m)/(A/s) at 1 A/us (V/m)")
    close(2e-5 * 1.0, 2e-5, rtol=1e-12, name="the same map read as per A/us, at 1 A/us (V/m)")
    print("All Chapter 23 checks passed.")


if __name__ == "__main__":
    main()
