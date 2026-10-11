"""Numerical checks for Chapter 3 (zh/ch03_electric.tex).

Inputs come from the old (4th) edition: atlas sub 2 (work example), worked check 1 (phi levels),
worked check 3 (capacitor: Q = 1 nC, A = 0.01 m^2, d = 1 mm), the two-layer conductor of
concept_checks.py (upper layer: 8 mm, 0.2 S/m, 1 cm^2, 1 mA) and atlas sub 7 (sigma = diag(1, 4)).
Run from the project root:  python3 checks/check_ch03.py
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12


def close(a, b, rel=1e-3):
    return abs(a - b) <= rel * abs(b)


def main() -> None:
    # --- work in a uniform field (atlas sub 2) ---------------------------------
    Ex, dx = 100.0, 0.01
    W_per_C = Ex * dx
    assert close(W_per_C, 1.0, 1e-12)
    print(f"E = {Ex:g} V/m over {dx} m: work per coulomb = {W_per_C:g} J/C -> potential drops {W_per_C:g} V")

    # --- worked check 1: large potential is not large field --------------------
    x = np.linspace(0, 0.01, 101)
    phi1 = np.full_like(x, 1e6)
    phi2 = 1.0 - 100.0 * x
    E1, E2 = -np.gradient(phi1, x), -np.gradient(phi2, x)
    assert np.allclose(E1, 0.0, atol=1e-5) and np.allclose(E2, 100.0)
    assert np.allclose(-np.gradient(phi2 + 5.0, x), 100.0)       # adding a constant changes nothing
    assert close(phi2[0] - phi2[-1], 1.0, 1e-12)                  # 1 V across 1 cm
    print("phi1 = 1e6 V -> E = 0;  phi2 = 1 V - 100 V/m x -> E = 100 V/m (also for phi2 + const)")

    # --- parallel-plate capacitor in vacuum (inputs of worked check 3, no dielectric) ----
    Q, A, d = 1e-9, 0.01, 1e-3
    qs = Q / A
    E = qs / EPS0
    V = E * d
    assert close(qs, 1.0e-7, 1e-12)
    assert close(E, 1.129e4) and close(V, 11.29)
    # consistency with the old edition's eps_r = 4 values (E ~ 2.82e3 V/m, V ~ 2.82 V): vacuum is 4x
    assert close(E / 4, 2.82e3, 2e-3) and close(V / 4, 2.82, 2e-3)
    print(f"capacitor: q_s = {qs:.2e} C/m^2, E = {E:.4g} V/m, V = {V:.4g} V (vacuum gap)")
    qs2, E2c = Q / (2 * A), Q / (2 * A) / EPS0          # self-check 1: doubled plate area
    assert close(qs2, 5.0e-8, 1e-12) and close(E2c, 5.65e3)
    print(f"doubled area: q_s = {qs2:.2e} C/m^2, E = {E2c:.4g} V/m")
    # self-check 5: 3 mA in, 1 mA out -> dQ/dt = +2 mA
    assert close(3e-3 - 1e-3, 2e-3, 1e-12)

    # --- one conducting layer (upper layer of the two-layer example) -----------
    I, Acs, ell, sigma = 1e-3, 1e-4, 0.008, 0.2
    J = I / Acs
    Ef = J / sigma
    Vl = Ef * ell
    Rl = ell / (sigma * Acs)
    P_vi = Vl * I
    p = sigma * Ef**2
    P_vol = p * Acs * ell
    assert close(J, 10.0, 1e-12) and close(Ef, 50.0, 1e-12) and close(Vl, 0.4, 1e-12)
    assert close(Rl, 400.0, 1e-12) and close(P_vi, 4e-4, 1e-12) and close(P_vol, P_vi, 1e-12)
    assert close(p, 500.0, 1e-12) and close(Rl * I, Vl, 1e-12)
    print(f"layer: J = {J:g} A/m^2, E = {Ef:g} V/m, V = {Vl:g} V, R = {Rl:g} ohm, "
          f"p = sigma E^2 = {p:g} W/m^3, P = {P_vi*1e3:g} mW (= VI)")

    # --- anisotropic conductivity (atlas sub 7, materials sub 2) ----------------
    S = np.diag([1.0, 4.0])
    E = np.array([1.0, 1.0])
    Jc = S @ E
    assert np.allclose(Jc, [1.0, 4.0]) and close(float(E @ Jc), 5.0, 1e-12)
    ang = np.degrees(np.arctan2(Jc[1], Jc[0]) - np.arctan2(E[1], E[0]))
    assert close(ang, 30.96, 1e-3)
    Ji = np.eye(2) @ E
    assert close(float(E @ Ji), 2.0, 1e-12)
    print(f"anisotropic: J = {Jc}, E.J = {E @ Jc:g} W/m^3, angle(E, J) = {ang:.2f} deg; isotropic E.J = 2")
    # symmetric positive-definite tensor: rotate, scale, rotate back; E^T sigma E > 0 for every E
    rng = np.random.default_rng(1)
    for _ in range(200):
        Qm, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        sv = rng.uniform(0.05, 5.0, 3)
        St = Qm @ np.diag(sv) @ Qm.T
        e = rng.normal(size=3)
        assert e @ St @ e > 0 and np.allclose(St, St.T)
    print("sigma = Q diag(s) Q^T with s_i > 0 gives E^T sigma E > 0 for all E (200 random cases)")

    # --- net-neutral current: +/- carriers with opposite drift (worked check 4) ----
    n, q, v = 1.0, 1.0, 1.0       # symbolic unit check: rho cancels, J adds
    rho = n * q + n * (-q)
    Jx = n * q * v + n * (-q) * (-v)
    assert rho == 0.0 and Jx == 2.0
    print("equal +/- carrier densities, opposite drift: rho = 0, J = 2 n q v")
    print("Chapter 3 checks passed.")


if __name__ == "__main__":
    main()
