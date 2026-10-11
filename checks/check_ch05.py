"""Numerical checks for Chapter 5 (zh/ch05_matter.tex) and its computed figures.

Every number quoted in the chapter is recomputed from its stated inputs and asserted:
  * ex:dielectriccap / fig:e_d_capacitor   eps_r = 4, fixed free charge: E -> 1/4, P = 3/4 q_f, D unchanged
                                            (old edition: 02_atlas fig displacement_capacitor, 03_connections)
  * pitfall + self-check 3                  fixed voltage vs fixed charge: E, D, Q, V ratios
  * ex:ironcore                             mu_r = 1000, fixed current: K_M = 999 n_t I, B = mu0 mu_r n_t I
  * self-check 6                            water chi_m = -9e-6: K_M/(n_t I) = chi_m, opposite sense
  * eq:atlassphere + the deeper box         sphere: H = -M/3, B = 2 mu0 M/3; eps0 E = -P/3, D = 2P/3
                                            (also the interface conditions on fig_matter.sphere_fields)
  * eq:boundcharge, eq:boundcurrent         1-D bound-charge bookkeeping, net bound charge zero,
                                            K_M = M x n directions (fig:bound_sources, ex:ironcore)
  * fig:dipoles_pair                        far fields of the charge pair and the loop have the same shape
  * eq:allsourceampere <-> eq:freeDH,       on test fields: the two bookkeepings are identical; div of the
    eq:continuityfromampere, J_amp - J_mat  Ampere right-hand side gives continuity; J_amp - J_mat
  * fig:displacement_current                d/dt of the D flux through S2 equals the conduction current I

Run from the project root:  python3 checks/check_ch05.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figsrc"))

EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi  # teaching value used throughout the book


def close(x, ref, rtol=1e-12, atol=0.0, name=""):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    return x


def capacitor() -> None:
    # ex:dielectriccap, eps_r = 4, fixed free surface charge q_f (any value; use the Ch 3 capacitor)
    A, d, Q, er = 0.01, 1e-3, 1e-9, 4.0
    qf = Q / A
    E_vac = qf / EPS0
    # eps0 E = q_f - P and P = eps0 (eps_r - 1) E  ->  solve the 2x2 linear system
    M = np.array([[EPS0, 1.0], [-EPS0 * (er - 1), 1.0]])
    E, P = np.linalg.solve(M, [qf, 0.0])
    close(E / E_vac, 1 / er, name="E ratio 1/eps_r")
    close(E / E_vac, 0.25, name="E drops to 1/4")
    close(P / qf, 1 - 1 / er, name="P = (1 - 1/eps_r) q_f")
    close(P / qf, 0.75, name="bound charge cancels 3/4")
    close(EPS0 * E + P, qf, name="D = eps0 E + P = q_f")
    # bound surface charge P.n: negative next to the + plate (n points from dielectric to that plate)
    n_top = +1.0   # P points from + plate (top) to - plate: P = -P e_z; the face at the + plate has n = +e_z
    assert (-P) * n_top < 0 and (-P) * (-n_top) > 0
    print(f"capacitor eps_r=4: E/E_vac = {E / E_vac:.4f}, P/q_f = {P / qf:.4f}, (eps0 E + P)/q_f = {(EPS0 * E + P) / qf:.4f}")

    # pitfall and self-check 3: fixed voltage (battery connected) vs fixed charge (disconnected)
    V0 = E_vac * d
    E_V, D_V = V0 / d, er * EPS0 * V0 / d                 # fixed V
    close(E_V / E_vac, 1.0, name="fixed V: E unchanged")
    close(D_V / (EPS0 * E_vac), 4.0, name="fixed V: D x4")
    close(D_V * A / Q, 4.0, name="fixed V: Q x4 (= eps_r)")
    E_Q = Q / A / (er * EPS0)                             # fixed Q
    close(E_Q / E_vac, 0.25, name="fixed Q: E x1/4")
    close(E_Q * d / V0, 0.25, name="fixed Q: V x1/4")
    print("fixed V: E x1, D x4, Q x4;  fixed Q: Q, D x1, E and V x1/4")


def iron_core() -> None:
    # ex:ironcore: long solenoid, H = n_t I independent of the core; M = (mu_r - 1) H; K_M = M
    n_t, I = 1000.0, 5.0          # Ch 4 teaching solenoid (ratios below do not depend on them)
    for mu_r, ratio_ref in ((1000.0, 999.0), (1 - 9e-6, -9e-6)):
        H = n_t * I
        M = (mu_r - 1) * H
        K_M = M
        B = MU0 * (n_t * I + K_M)
        close(K_M / (n_t * I), ratio_ref, rtol=1e-9, name="K_M / n_t I")
        close(B, MU0 * mu_r * n_t * I, rtol=1e-12, name="B = mu0 mu_r n_t I")
    close(MU0 * 1000.0 * n_t * I / (MU0 * n_t * I), 1000.0, name="B grows 1000x")
    # self-check 6: |chi_m| = 9e-6 is 'about one part in 1e5' (same order of magnitude as 1e-5)
    assert 0.5e-5 < 9e-6 < 2e-5
    # direction: winding current along +e_phi gives B, M along +e_z; on the side surface n = e_r and
    # K_M = M x n = e_z x e_r = +e_phi (same sense as the winding); for water (M along -e_z) opposite.
    ez, er_, ephi = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert np.allclose(np.cross(ez, er_), ephi)
    assert np.allclose(np.cross(-ez, er_), -ephi)
    print("iron core mu_r=1000: K_M = 999 n_t I, B = 1000 mu0 n_t I;  water chi_m=-9e-6: K_M/(n_t I) = -9e-6 (opposite)")


def bound_sources() -> None:
    # 1-D bound charge: rho_b = -dP/dx; with the end faces q_b = P.n the net bound charge is zero
    L = 0.02
    x = np.linspace(0.0, L, 4001)
    P = 1e-7 * (1 + 3 * x / L + np.sin(7 * x / L))     # arbitrary non-uniform polarisation, C/m^2
    rho_b = -np.gradient(P, x)
    net = np.trapezoid(rho_b, x) + (P[-1] * (+1)) + (P[0] * (-1))
    assert abs(net) < 1e-12 * 1e-7 * 10, net
    # uniform slab, P = P0 e_x: left face P.n = -P0, right face +P0 (fig:bound_sources, left)
    P0 = 1.0
    assert P0 * (-1) == -P0 and P0 * (+1) == +P0
    # uniform magnetised slab, M = M0 e_z (out of page): K_M = M x n circulates counter-clockwise
    ez = np.array([0, 0, 1.0])
    for n, k in (((0, 1, 0), (-1, 0, 0)), ((1, 0, 0), (0, 1, 0)), ((0, -1, 0), (1, 0, 0)), ((-1, 0, 0), (0, -1, 0))):
        assert np.allclose(np.cross(ez, n), k)
    print("bound sources: net bound charge of a non-uniform slab = 0; K_M = M x n circulates (top: -x, right: +y)")


def spheres() -> None:
    # deeper box: psi_in = c r cos(th), psi_out = d cos(th)/r^2; at r = a, tangential H continuous
    # (c = d/a^3) and normal B continuous (2 d/a^3 = -c + M0).  Solve for c and d/a^3 with M0 = 1.
    c, d_a3 = np.linalg.solve(np.array([[1.0, -1.0], [1.0, 2.0]]), [0.0, 1.0])
    close(c, 1 / 3, name="c = M0/3")
    close(d_a3, 1 / 3, name="d = M0 a^3/3")
    H_in = -c                            # H_in = -grad(c z) = -c e_z  (units of M0)
    close(H_in, -1 / 3, name="H_in = -M/3")
    close(H_in + 1.0, 2 / 3, name="B_in/mu0 = H_in + M = 2M/3")
    print("sphere (deeper box): c = M0/3 -> H_in = -M/3, B_in = 2 mu0 M/3")

    # fig_matter.sphere_fields (used for fig:sphere_pair): check both interface conditions numerically
    import fig_matter as fm
    th = np.linspace(0.05, np.pi - 0.05, 181)
    for kind, mat in (("B", 1.0), ("E", 1.0)):
        rin, rout = 1 - 1e-9, 1 + 1e-9
        Xi, Zi = rin * np.sin(th), rin * np.cos(th)
        Xo, Zo = rout * np.sin(th), rout * np.cos(th)
        Fxi, Fzi = fm.sphere_fields(Xi, Zi, kind)
        Fxo, Fzo = fm.sphere_fields(Xo, Zo, kind)
        nr = (np.sin(th), np.cos(th))
        nt = (np.cos(th), -np.sin(th))
        rad = lambda fx, fz: fx * nr[0] + fz * nr[1]
        tan = lambda fx, fz: fx * nt[0] + fz * nt[1]
        if kind == "B":   # inside value is B/mu0; outside is B/mu0 = H. H_in = B_in - M
            assert np.allclose(rad(Fxi, Fzi), rad(Fxo, Fzo), atol=1e-6)                    # normal B
            assert np.allclose(tan(Fxi, Fzi - mat), tan(Fxo, Fzo), atol=1e-6)              # tangential H
            close(Fzi[0] - mat, -1 / 3, name="mu0 H_in")
            close(Fzi[0], 2 / 3, name="B_in")
        else:             # inside is E/eps0-units; D_in = E_in + P
            assert np.allclose(tan(Fxi, Fzi), tan(Fxo, Fzo), atol=1e-6)                    # tangential E
            assert np.allclose(rad(Fxi, Fzi + mat), rad(Fxo, Fzo), atol=1e-6)              # normal D
            # jump of eps0 E_r equals the bound surface charge P.n = P cos(theta)
            assert np.allclose(rad(Fxo, Fzo) - rad(Fxi, Fzi), np.cos(th), atol=1e-6)
            close(Fzi[0], -1 / 3, name="eps0 E_in")
            close(Fzi[0] + mat, 2 / 3, name="D_in")
    print("sphere_fields: normal B, tangential H, tangential E, normal D continuous; eps0 E_in = -P/3, D_in = 2P/3")


def dipole_far_fields() -> None:
    # fig:dipoles_pair: far from the source both fields approach the same point-dipole pattern
    import fig_matter as fm
    th = np.linspace(0.1, np.pi - 0.1, 50)
    R = 40.0
    X, Z = R * np.sin(th), R * np.cos(th)
    Ex, Ez = fm.point_dipole_charges(X, Z, d=0.5)
    Bx, Bz = fm.loop_field(X, Z, a=0.5)
    dx, dz = 3 * np.cos(th) * np.sin(th), 3 * np.cos(th) ** 2 - 1        # point-dipole pattern
    def ang(ax_, az_):
        return np.arctan2(ax_, az_)
    err_E = np.abs(np.angle(np.exp(1j * (ang(Ex, Ez) - ang(dx, dz))))).max()
    err_B = np.abs(np.angle(np.exp(1j * (ang(Bx, Bz) - ang(dx, dz))))).max()
    assert err_E < 2e-3 and err_B < 2e-3, (err_E, err_B)
    # near the source the interior fields are opposite: E between the charges along -z, B in the loop along +z
    Ex0, Ez0 = fm.point_dipole_charges(np.array([1e-6]), np.array([0.0]), d=0.5)
    Bx0, Bz0 = fm.loop_field(np.array([1e-6]), np.array([0.0]), a=0.5)
    assert Ez0[0] < 0 < Bz0[0]
    print(f"dipoles_pair: far-field direction error vs point dipole: E {err_E:.1e} rad, B {err_B:.1e} rad; "
          "inside: E opposite p, B along m")


def bookkeeping() -> None:
    # Smooth test fields on a 4-D grid (x, y, z, t); derivatives by central differences.  The checks are
    # the algebra behind eq:allsourceampere <-> eq:freeDH, eq:continuityfromampere and J_amp - J_mat.
    n = 18
    g = np.linspace(-1.0, 1.0, n)
    tg = np.linspace(0.0, 1.0, n)
    X, Y, Z, Tt = np.meshgrid(g, g, g, tg, indexing="ij")
    h, ht = g[1] - g[0], tg[1] - tg[0]
    rng = np.random.default_rng(5)
    def field():
        a = rng.normal(size=(3, 4))
        return np.stack([np.sin(a[k, 0] * X + a[k, 1] * Y) * np.cos(a[k, 2] * Z + a[k, 3] * Tt) for k in range(3)])
    eps0, mu0 = 0.7, 1.3           # any positive constants (the identities are linear)
    E, B, P, M, Jf = field(), field(), field(), field(), field()
    dx = lambda F: np.gradient(F, h, axis=0)
    dy = lambda F: np.gradient(F, h, axis=1)
    dz = lambda F: np.gradient(F, h, axis=2)
    dt = lambda F: np.gradient(F, ht, axis=-1)
    div = lambda F: dx(F[0]) + dy(F[1]) + dz(F[2])
    curl = lambda F: np.stack([dy(F[2]) - dz(F[1]), dz(F[0]) - dx(F[2]), dx(F[1]) - dy(F[0])])
    D = eps0 * E + P
    H = B / mu0 - M
    rho_f = div(D)                 # take rho_f from Gauss's law
    tol = 1e-10
    # eq:allsourcegauss <-> div D = rho_f
    assert np.abs((div(eps0 * E) - (rho_f - div(P))) - (div(D) - rho_f)).max() < tol
    # eq:allsourceampere <-> curl H = J_f + d_t D
    lhs_all = curl(B / mu0) - (Jf + dt(P) + curl(M) + eps0 * dt(E))
    lhs_free = curl(H) - (Jf + dt(D))
    assert np.abs(lhs_all - lhs_free).max() < tol
    # eq:continuityfromampere: div curl H = 0, so div(J_f + d_t D) = div J_f + d_t rho_f
    assert np.abs(div(curl(H))).max() < tol
    assert np.abs(div(Jf + dt(D)) - (div(Jf) + dt(rho_f))).max() < tol
    # J_amp - J_mat = eps0 d_t E - curl M
    J_amp = Jf + dt(D)
    J_mat = Jf + dt(P) + curl(M)
    assert np.abs(J_amp - J_mat - (eps0 * dt(E) - curl(M))).max() < tol
    print("bookkeeping: all-source and free-source forms identical; div(J_f + d_t D) -> continuity; "
          "J_amp - J_mat = eps0 d_t E - curl M")


def displacement() -> None:
    # fig:displacement_current: charging current I(t); between the plates D = Q/A, so d/dt (D A) = I
    A = 0.01
    tt = np.linspace(0, 1e-3, 20001)
    I = 2e-3 * np.sin(2 * np.pi * 3e3 * tt)                    # any waveform
    Q = np.concatenate(([0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(tt))))
    D = Q / A
    dflux = np.gradient(D * A, tt)
    assert np.allclose(dflux[5:-5], I[5:-5], atol=1e-3 * np.abs(I).max())
    print("displacement current through S2 = conduction current I (numerically, any waveform)")


def main() -> None:
    capacitor()
    iron_core()
    bound_sources()
    spheres()
    dipole_far_fields()
    bookkeeping()
    displacement()
    print("Chapter 5 checks passed.")


if __name__ == "__main__":
    main()
