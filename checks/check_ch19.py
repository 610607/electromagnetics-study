"""Chapter 19 (magnetic observables of MRI): every number quoted in zh/ch19_mri.tex.

    python3 checks/check_ch19.py

figsrc/fig_ch19.py imports the field models defined here.  Teaching inputs (stated once in the text):
gradient strength G = 10 mT/m, bias fields B0 = 0.1 T (low field) and 1.5 T, distances 10 cm,
a 1 ms gradient lobe, a 20 cm DSV with a Z2 error of 30 ppm peak-to-peak, a ramp of 10 mT/m in 0.1 ms;
self-check inputs 0.05 T with 20 mT/m, and a 10 ppm peak-to-peak Z2 error.
Physical constant: proton gyromagnetic ratio gamma/(2 pi) = 42.577 MHz/T (CODATA 2018,
42.577 478 518 MHz/T for the free proton).  No device data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from numerics import MU0

GAMMA_BAR = 42.577478518e6          # Hz/T, CODATA 2018 proton gyromagnetic ratio / (2 pi)
G_T = 10e-3                         # teaching gradient strength, T/m
B0_LOW, B0_HIGH = 0.1, 1.5          # teaching bias fields, T


# ---------------------------------------------------------------------------
# Linear gradient fields (eq:xgradient, eq:zgradient) and generic checks
# ---------------------------------------------------------------------------
def x_gradient(x, y, z, G=G_T):
    return np.array([G * z, 0 * x, G * x])


def z_gradient(x, y, z, G=G_T):
    return np.array([-G * x / 2, -G * y / 2, G * z])


def z2_shim(x, y, z, c=1.0):
    """Full vector field whose z-component is the Z2 harmonic c (2 z^2 - x^2 - y^2)."""
    return np.array([-2 * c * x * z, -2 * c * y * z, c * (2 * z**2 - x**2 - y**2)])


def jacobian(field, r, h=1e-6):
    """G_ij = d b_i / d x_j by central differences."""
    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3); e[j] = h
        J[:, j] = (field(*(r + e)) - field(*(r - e))) / (2 * h)
    return J


def div_curl(field, r):
    J = jacobian(field, r)
    div = np.trace(J)
    curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    return div, curl


# ---------------------------------------------------------------------------
# Concomitant field (eq:concomitant)
# ---------------------------------------------------------------------------
def magnitude_exact(b, B0):
    return np.sqrt(b[0] ** 2 + b[1] ** 2 + (B0 + b[2]) ** 2)


def concomitant(b, B0):
    """Second-order term (b_x^2 + b_y^2)/(2 B0)."""
    return (b[0] ** 2 + b[1] ** 2) / (2 * B0)


def bc_xgrad(z, B0, G=G_T):
    return (G * z) ** 2 / (2 * B0)


def bc_zgrad(r, B0, G=G_T):
    return (G * r) ** 2 / (8 * B0)


# ---------------------------------------------------------------------------
# Homogeneity metrics of a Z2 error over a sphere (DSV)
# ---------------------------------------------------------------------------
def z2_metrics(R=0.1, c=1.0, n=400000, seed=1):
    """Peak-to-peak and RMS of c(2z^2-x^2-y^2) on the DSV surface, in its volume, and with an
    unweighted radial grid (equal node count per shell, equally spaced radii) -- Monte Carlo
    check of the closed forms."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    f = lambda p: c * (2 * p[:, 2] ** 2 - p[:, 0] ** 2 - p[:, 1] ** 2)
    surf = f(R * v)
    vol = f(R * v * rng.random(n)[:, None] ** (1 / 3))
    rad = f(R * v * rng.random(n)[:, None])
    return dict(pp=3 * c * R**2, rms_surf=np.sqrt(np.mean(surf**2)), rms_vol=np.sqrt(np.mean(vol**2)),
                rms_rad=np.sqrt(np.mean(rad**2)), mean_surf=np.mean(surf))


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:62s} {x:.6g}")
    return x


def run_checks():
    print("19.2 Maxwell constraints")
    rng = np.random.default_rng(0)
    for fld, name in ((x_gradient, "x gradient"), (z_gradient, "z gradient"), (z2_shim, "Z2 shim, full vector")):
        for r in rng.uniform(-0.2, 0.2, size=(5, 3)):
            d, c = div_curl(fld, r)
            assert abs(d) < 1e-9 and np.all(np.abs(c) < 1e-9), (name, d, c)
        print(f"  ok  {name}: div = 0 and curl = 0")
    Gx = jacobian(x_gradient, np.zeros(3)); Gz = jacobian(z_gradient, np.zeros(3))
    assert np.allclose(Gx, [[0, 0, G_T], [0, 0, 0], [G_T, 0, 0]]) and np.allclose(Gz, np.diag([-G_T / 2, -G_T / 2, G_T]))
    assert np.allclose(Gx, Gx.T) and abs(np.trace(Gz)) < 1e-12
    print("  ok  gradient matrices symmetric and traceless")
    # symmetric traceless 3x3: 9 - 3 - 1 = 5 parameters (dimension of the space)
    basis = []
    for i in range(3):
        for j in range(3):
            E = np.zeros((3, 3)); E[i, j] = 1
            S = (E + E.T) / 2; S -= np.trace(S) / 3 * np.eye(3)
            basis.append(S.ravel())
    assert np.linalg.matrix_rank(np.array(basis)) == 5
    print("  ok  symmetric traceless matrices form a 5-dimensional space")
    # incompatible targets
    bad_x = lambda x, y, z: np.array([0 * x, 0 * x, G_T * x])
    d, c = div_curl(bad_x, np.array([0.05, 0.02, 0.01]))
    assert abs(d) < 1e-12 and np.allclose(c, [0, -G_T, 0])
    close(G_T / MU0, 7958.0, rtol=1e-3, name="current density needed by b=(0,0,Gx): G/mu0 (A/m^2)")
    bad_z = lambda x, y, z: np.array([0 * x, 0 * x, G_T * z])
    d, c = div_curl(bad_z, np.array([0.05, 0.02, 0.01]))
    assert np.isclose(d, G_T) and np.allclose(c, 0)
    print("  ok  b=(0,0,Gz) has div b = G (forbidden)")
    # scalar potentials generate the fields: b = -mu0 grad psi
    for psi, fld in ((lambda x, y, z: -G_T * x * z / MU0, x_gradient),
                     (lambda x, y, z: -(G_T / (2 * MU0)) * (z**2 - (x**2 + y**2) / 2), z_gradient)):
        for r in rng.uniform(-0.2, 0.2, size=(3, 3)):
            h = 1e-6
            grad = np.array([(psi(*(r + h * e)) - psi(*(r - h * e))) / (2 * h) for e in np.eye(3)])
            assert np.allclose(-MU0 * grad, fld(*r), atol=1e-12)
            lap = sum((psi(*(r + 1e-3 * e)) - 2 * psi(*r) + psi(*(r - 1e-3 * e))) / 1e-6 for e in np.eye(3))
            assert abs(lap) < 1e-3 * abs(G_T / MU0)
    print("  ok  scalar potentials -G x z/mu0 and -(G/2mu0)(z^2-(x^2+y^2)/2) are harmonic and generate the fields")
    # spatial gradient versus time derivative
    slew = G_T / 1e-4
    close(slew, 100.0, rtol=1e-12, name="slew rate 10 mT/m in 0.1 ms (T/m/s)")
    close(slew * 0.2, 20.0, rtol=1e-12, name="d b_z/dt at x = 0.2 m (T/s)")
    close(np.linalg.norm(slew * np.array([0.2, 0, 0.2])), 28.3, rtol=1e-3, name="|d b/dt| at x = z = 0.2 m (T/s)")

    print("19.3 concomitant fields")
    for B0 in (B0_LOW, B0_HIGH):
        b = x_gradient(0.1, 0.0, 0.1)
        approx = B0 + b[2] + concomitant(b, B0)
        assert abs(magnitude_exact(b, B0) - approx) < 0.011 * concomitant(b, B0)
    print("  ok  second-order expansion within 1.1 % of the concomitant term at x = z = 10 cm")
    close(bc_xgrad(0.1, B0_LOW) * 1e6, 5.0, rtol=1e-12, name="x gradient, z = 10 cm, 0.1 T (uT)")
    close(bc_xgrad(0.1, B0_LOW) / B0_LOW * 1e6, 50.0, rtol=1e-12, name="... in ppm of B0")
    close(bc_xgrad(0.1, B0_HIGH) * 1e6, 0.333, rtol=2e-3, name="x gradient, z = 10 cm, 1.5 T (uT)")
    close(bc_xgrad(0.1, B0_HIGH) / B0_HIGH * 1e6, 0.222, rtol=2e-3, name="... in ppm of B0")
    close(bc_zgrad(0.1, B0_LOW) * 1e6, 1.25, rtol=1e-12, name="z gradient, r = 10 cm, 0.1 T (uT)")
    close(bc_zgrad(0.1, B0_HIGH) * 1e6, 0.0833, rtol=2e-3, name="z gradient, r = 10 cm, 1.5 T (uT)")
    close(bc_xgrad(0.1, B0_LOW) / G_T * 1e3, 0.5, rtol=1e-12, name="apparent shift along x, 0.1 T (mm)")
    close(bc_xgrad(0.1, B0_HIGH) / G_T * 1e3, 0.0333, rtol=2e-3, name="apparent shift along x, 1.5 T (mm)")
    close(G_T * 0.1 / B0_LOW, 0.01, rtol=1e-12, name="|b|/B0 for b_z = G x at 10 cm, 0.1 T")
    # second-order error grows with |b|/B0: at |b_x|/B0 = 0.3 (b_z = 0) the relative error of b_c
    for ratio, tol in ((0.01, 3e-5), (0.3, 0.03)):
        b = np.array([ratio, 0.0, 0.0])
        rel = (magnitude_exact(b, 1.0) - 1.0) / concomitant(b, 1.0) - 1
        assert abs(rel) < tol, (ratio, rel)
    close((np.sqrt(1 + 0.3**2) - 1) / (0.3**2 / 2) - 1, -0.0215, rtol=0.01, name="relative error of b_c at |b_perp|/B0 = 0.3")
    larmor = GAMMA_BAR * B0_LOW
    close(larmor / 1e6, 4.258, rtol=1e-3, name="Larmor frequency at 0.1 T (MHz)")
    close(GAMMA_BAR * B0_LOW * 1e-6, 4.258, rtol=1e-3, name="1 ppm at 0.1 T (Hz)")
    close(GAMMA_BAR * B0_HIGH / 1e6, 63.87, rtol=1e-3, name="Larmor frequency at 1.5 T (MHz)")
    th1 = 2 * np.pi * GAMMA_BAR * bc_xgrad(0.1, B0_LOW) * 1e-3
    close(th1, 1.338, rtol=1e-3, name="concomitant phase, 1 ms lobe, 0.1 T, z = 10 cm (rad)")
    close(np.degrees(th1), 76.6, rtol=1e-3, name="... in degrees")
    th15 = 2 * np.pi * GAMMA_BAR * bc_xgrad(0.1, B0_HIGH) * 1e-3
    close(th15, 0.0892, rtol=1e-3, name="concomitant phase at 1.5 T (rad)")
    close(np.degrees(th15), 5.11, rtol=2e-3, name="... in degrees")
    # bipolar lobes +G, -G for 1 ms each: linear phase cancels, concomitant adds
    t = (np.arange(200000) + 0.5) * 1e-8            # midpoint rule on 0 .. 2 ms (lobe edge on a cell edge)
    g = np.where(t < 1e-3, G_T, -G_T)
    k = GAMMA_BAR * np.sum(g) * 1e-8
    assert abs(k) < 1e-9 * GAMMA_BAR * G_T * 1e-3
    thc = 2 * np.pi * GAMMA_BAR * np.sum((g * 0.1) ** 2 / (2 * B0_LOW)) * 1e-8
    close(thc, 2.675, rtol=1e-3, name="bipolar pair: concomitant phase at 0.1 T, z = 10 cm (rad)")
    close(np.degrees(thc), 153.3, rtol=1e-3, name="... in degrees")

    # x and z gradients together: the cross term -Gx Gz x z /(2 B0) of the expansion
    Gx_, Gz_ = G_T, 0.5 * G_T
    for r in rng.uniform(-0.15, 0.15, size=(5, 3)):
        b = x_gradient(*r, G=Gx_) + z_gradient(*r, G=Gz_)
        ref = (Gx_**2 * r[2]**2 - Gx_ * Gz_ * r[0] * r[2] + Gz_**2 * (r[0]**2 + r[1]**2) / 4) / (2 * B0_LOW)
        assert np.isclose(concomitant(b, B0_LOW), ref, rtol=1e-12)
    print("  ok  combined x and z gradients: b_c = [Gx^2 z^2 - Gx Gz x z + Gz^2 (x^2+y^2)/4] / (2 B0)")
    # b_z = G.r fixes the third row (and column) of the symmetric traceless matrix; two parameters stay free
    def gmat(gx, gy, gz, p, q):
        return np.array([[p, q, gx], [q, -gz - p, gy], [gx, gy, gz]])
    for p, q in ((0.0, 0.0), (0.3 * G_T, 0.0), (0.0, 0.2 * G_T)):
        Gm = gmat(G_T, 0, 0, p, q)
        assert np.allclose(Gm, Gm.T) and abs(np.trace(Gm)) < 1e-15
    print("  ok  for b_z = G.r the entries G_xx - G_yy and G_xy remain free (3 + 2 = 5)")
    # z gradient without rotational symmetry: b = (-a G x, -(1-a) G y, G z)
    def zgrad_alpha(r, a, G=G_T):
        return np.array([-a * G * r[0], -(1 - a) * G * r[1], G * r[2]])
    for a in (0.5, 1.0, 0.0):
        r = np.array([0.03, -0.07, 0.05])
        d, c = div_curl(lambda x, y, z: zgrad_alpha(np.array([x, y, z]), a), r)
        assert abs(d) < 1e-9 and np.all(np.abs(c) < 1e-9)
        assert np.isclose(concomitant(zgrad_alpha(r, a), B0_LOW),
                          G_T**2 * (a**2 * r[0]**2 + (1 - a)**2 * r[1]**2) / (2 * B0_LOW), rtol=1e-12)
    close(concomitant(zgrad_alpha(np.array([0.1, 0, 0]), 1.0), B0_LOW) * 1e6, 5.0, rtol=1e-12,
          name="z gradient with all transverse field along x: b_c at x = 10 cm, 0.1 T (uT)")
    close(concomitant(zgrad_alpha(np.array([0.1, 0, 0]), 0.5), B0_LOW) * 1e6, 1.25, rtol=1e-12,
          name="... axisymmetric z gradient at the same point (uT)")

    print("19.4 phase and k-space")
    kk = GAMMA_BAR * G_T * 1e-3
    close(kk, 425.8, rtol=1e-3, name="k after 10 mT/m for 1 ms (cycles/m)")
    close(2 * np.pi * kk * 1e-3, 2.675, rtol=1e-3, name="phase difference across 1 mm (rad)")
    close(np.degrees(2 * np.pi * kk * 1e-3), 153.3, rtol=1e-3, name="... in degrees")
    close(2 * bc_xgrad(0.1, B0_LOW), G_T * 1e-3, rtol=1e-12, name="2 b_c(z=10 cm, 0.1 T) = G * 1 mm (T)")

    print("19.5 homogeneity metrics (Z2 error on a 20 cm DSV)")
    m = z2_metrics()
    assert abs(m["mean_surf"]) < 5e-3 * m["pp"]
    close(m["pp"] / m["rms_surf"], 3 / np.sqrt(4 / 5), rtol=5e-3, name="pk-pk / RMS on the surface (closed form 3.354)")
    close(m["pp"] / m["rms_vol"], 3 / np.sqrt(12 / 35), rtol=5e-3, name="pk-pk / RMS in the volume (5.123)")
    close(m["pp"] / m["rms_rad"], 3 / 0.4, rtol=5e-3, name="pk-pk / RMS, unweighted radial grid (7.5)")
    pp_ppm = 30.0
    close(np.sqrt(4 / 5) / 3, 0.298, rtol=2e-3, name="RMS / pk-pk on the surface")
    close(np.sqrt(12 / 35) / 3, 0.195, rtol=2e-3, name="RMS / pk-pk in the volume")
    close(pp_ppm * np.sqrt(4 / 5) / 3, 8.94, rtol=1e-3, name="RMS on the DSV surface (ppm)")
    close(pp_ppm * np.sqrt(12 / 35) / 3, 5.86, rtol=1e-3, name="RMS in the DSV volume (ppm)")
    close(pp_ppm * 0.4 / 3, 4.0, rtol=1e-12, name="RMS, unweighted radial grid (ppm)")
    close(pp_ppm * B0_LOW, 3.0, rtol=1e-12, name="30 ppm at 0.1 T (uT)")
    close(pp_ppm * GAMMA_BAR * B0_LOW * 1e-6, 127.7, rtol=1e-3, name="30 ppm at 0.1 T (Hz)")
    # maximum principle: the extremes of the harmonic Z2 error over the ball lie on the DSV surface
    rng2 = np.random.default_rng(5)
    v = rng2.normal(size=(200000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    ball = 0.1 * v * rng2.random(200000)[:, None] ** (1 / 3)
    f = 2 * ball[:, 2]**2 - ball[:, 0]**2 - ball[:, 1]**2
    assert f.max() <= 2 * 0.1**2 and f.min() >= -0.1**2
    print("  ok  Z2 over the ball stays within its surface extremes (+2R^2, -R^2): pk-pk = 3R^2 either way")
    # fitting a free constant c0 (sec 19.5): the mean of a Z2 error over the sphere is zero, so c0 = 0
    close(abs(m["mean_surf"]) / m["pp"] + 1.0, 1.0, rtol=5e-3, name="mean of Z2 over the DSV surface / pk-pk (+1)")
    print("self-check numbers")
    close(bc_xgrad(0.1, 0.05, G=20e-3) * 1e6, 40.0, rtol=1e-12, name="x gradient 20 mT/m, z = 10 cm, 0.05 T: b_c (uT)")
    close(bc_xgrad(0.1, 0.05, G=20e-3) / 0.05 * 1e6, 800.0, rtol=1e-12, name="... in ppm")
    close(20e-3 * 0.1 / 0.05, 0.04, rtol=1e-12, name="... |b_x|/B0 (expansion still valid)")
    close(10.0 * np.sqrt(12 / 35) / 3, 1.95, rtol=2e-3, name="Z2 error, 10 ppm pk-pk -> volume RMS (ppm)")
    d, c = div_curl(lambda x, y, z: np.array([0 * x, G_T * z, G_T * y]), np.array([0.01, 0.02, 0.03]))
    assert abs(d) < 1e-12 and np.allclose(c, 0)
    print("  ok  y gradient b = G(0, z, y) is divergence- and curl-free")

    print("19.6 a uniform transverse field is invisible in B_z but not in |B|")
    close(concomitant(np.array([1e-3, 0, 0]), B0_LOW) * 1e6, 5.0, rtol=1e-12, name="b_x = 1 mT at 0.1 T shifts |B| by (uT)")
    # self-check 6: B_z uniform does not exclude a non-uniform transverse field, e.g. b = a (x, -y, 0)
    a_t = 0.01                                           # T/m, any value
    trans = lambda x, y, z: np.array([a_t * x, -a_t * y, 0 * z])
    for r in (np.array([0.05, -0.03, 0.02]), np.array([-0.08, 0.04, -0.06])):
        d, c = div_curl(trans, r)
        assert abs(d) < 1e-12 and np.allclose(c, 0)
        assert np.isclose(concomitant(trans(*r), B0_LOW), a_t**2 * (r[0]**2 + r[1]**2) / (2 * B0_LOW), rtol=1e-12)
    print("  ok  b = a(x, -y, 0): div = curl = 0, b_z = 0, |B| - B0 = a^2 (x^2 + y^2)/(2 B0) to second order")
    print("check_ch19: all assertions passed")
    return True


if __name__ == "__main__":
    run_checks()
