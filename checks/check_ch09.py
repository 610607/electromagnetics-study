"""Numbers and identities used in Chapter 9 (zh/ch09_potentials.tex).

The uniform-B two-gauge example (old atlas:12, form:2, conn:5, wc:2) is checked
symbolically and numerically, with the induction numbers of Chapter 4
(dB/dt = 1000 T/s, r = 8 cm -> 40 V/m).  The Biot-Savart derivation is checked
numerically for a circular loop (A from the Green-function integral, its curl
and divergence).  Run:  python3 checks/check_ch09.py
"""
from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi  # same teaching value as numerics.py


def close(x, ref, rtol=5e-3, atol=0.0, name=""):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    return x


# --- 1. Two vector potentials of the same uniform B ----------------------------------
def A_sym(x, y, B):
    return np.array([-B * y / 2, B * x / 2])


def A_alt(x, y, B):
    return np.array([0.0 * x, B * x])


def curl_z(Af, x, y, B, h=1e-6):
    Ay_x = (Af(x + h, y, B)[1] - Af(x - h, y, B)[1]) / (2 * h)
    Ax_y = (Af(x, y + h, B)[0] - Af(x, y - h, B)[0]) / (2 * h)
    return Ay_x - Ax_y


B0 = 0.004
for (x, y) in ((0.3, -0.2), (0.0, 0.05), (-0.07, 0.11)):
    close(curl_z(A_sym, x, y, B0), B0, rtol=1e-8, name="curl A_sym")
    close(curl_z(A_alt, x, y, B0), B0, rtol=1e-8, name="curl A_alt")
    # A_alt - A_sym = grad(chi), chi = B x y / 2
    grad_chi = np.array([B0 * y / 2, B0 * x / 2])
    assert np.allclose(A_alt(x, y, B0) - A_sym(x, y, B0), grad_chi, atol=1e-15)

# circulation on a circle = pi r^2 B in both gauges (old concept_checks numbers)
r = 0.037
th = np.linspace(0, 2 * np.pi, 20001)
x, y = r * np.cos(th), r * np.sin(th)
dx, dy = -r * np.sin(th), r * np.cos(th)
for Af in (A_sym, A_alt):
    Ax, Ay = Af(x, y, B0)
    circ = np.trapezoid(Ax * dx + Ay * dy, th)
    close(circ, np.pi * r**2 * B0, rtol=1e-9, name="circulation")
close(np.pi * r**2 * B0, 1.72e-5, rtol=3e-3, name="flux through r = 3.7 cm")
# tangential component along the circle: uniform B r/2 (sym) vs B r cos^2(theta) (alt)
At_sym = A_sym(x, y, B0)[0] * (-np.sin(th)) + A_sym(x, y, B0)[1] * np.cos(th)
At_alt = A_alt(x, y, B0)[0] * (-np.sin(th)) + A_alt(x, y, B0)[1] * np.cos(th)
assert np.allclose(At_sym, B0 * r / 2)
assert np.allclose(At_alt, B0 * r * np.cos(th) ** 2)

# --- 2. Time-dependent case: phi compensates (numbers from Chapter 4) ----------------
dB = 1000.0      # T/s
r0 = 0.08        # m


def E_sym(x, y):            # -dA_sym/dt, phi_sym = 0
    return np.array([dB * y / 2, -dB * x / 2])


def mdA_alt(x, y):          # -dA_alt/dt
    return np.array([0.0 * x, -dB * x])


def mgradphi_alt(x, y):     # phi_alt = -dB x y / 2  ->  -grad phi_alt
    return np.array([dB * y / 2, dB * x / 2])


for (px, py) in ((r0, 0.0), (0.0, r0), (0.05, -0.03)):
    assert np.allclose(mdA_alt(px, py) + mgradphi_alt(px, py), E_sym(px, py), atol=1e-12)
# the two sample points quoted in the chapter
assert np.allclose(E_sym(r0, 0), [0, -40]) and np.allclose(E_sym(0, r0), [40, 0])
assert np.allclose(mdA_alt(r0, 0), [0, -80]) and np.allclose(mdA_alt(0, r0), [0, 0])
assert np.allclose(mgradphi_alt(r0, 0), [0, 40]) and np.allclose(mgradphi_alt(0, r0), [40, 0])
close(np.hypot(*E_sym(r0 / np.sqrt(2), r0 / np.sqrt(2))), 40.0, rtol=1e-12, name="|E| on r = 8 cm")
# phi_alt at (8 cm, 8 cm)
close(-dB * r0 * r0 / 2, -3.2, rtol=1e-12, name="phi_alt(8 cm, 8 cm)")
# self-check: A3 = (-B y, 0, 0) needs phi3 = + dB x y / 2
for (px, py) in ((0.02, 0.07), (-0.05, 0.01)):
    mdA3 = np.array([dB * py, 0.0])
    mgradphi3 = -np.array([dB * py / 2, dB * px / 2])
    assert np.allclose(mdA3 + mgradphi3, E_sym(px, py))
    # A_alt - A3 = grad(B x y)
    assert np.allclose(A_alt(px, py, B0) - np.array([-B0 * py, 0.0]), [B0 * py, B0 * px])

# --- 3. Coulomb gauge and Biot-Savart for a circular loop ----------------------------
a, I = 0.05, 1.0
phi_s = np.linspace(0, 2 * np.pi, 4001)[:-1]
dphi = phi_s[1] - phi_s[0]
src = np.stack((a * np.cos(phi_s), a * np.sin(phi_s), 0 * phi_s), axis=1)
dl = np.stack((-a * np.sin(phi_s), a * np.cos(phi_s), 0 * phi_s), axis=1) * dphi


def A_green(p):
    R = np.linalg.norm(p - src, axis=1)
    return MU0 * I / (4 * np.pi) * np.sum(dl / R[:, None], axis=0)


def B_biot(p):
    d = p - src
    R = np.linalg.norm(d, axis=1)
    return MU0 * I / (4 * np.pi) * np.sum(np.cross(dl, d) / R[:, None] ** 3, axis=0)


def curl_num(F, p, h=1e-5):
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = (F(p + e) - F(p - e)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]]), np.trace(J)


for p in (np.array([0.0, 0.0, 0.03]), np.array([0.02, 0.01, 0.04]), np.array([0.08, -0.03, -0.02])):
    cB, divA = curl_num(A_green, p)
    Bb = B_biot(p)
    assert np.allclose(cB, Bb, rtol=1e-5, atol=1e-12 * np.linalg.norm(Bb)), (cB, Bb)
    assert abs(divA) < 1e-5 * np.linalg.norm(cB), (divA, cB)     # Coulomb gauge holds
# on the axis, Biot-Savart reproduces eq:loopaxis
for z in (0.0, 0.03):
    close(B_biot(np.array([0, 0, z]))[2], MU0 * I * a**2 / (2 * (a**2 + z**2) ** 1.5), rtol=1e-6, name="loop axis")

# --- 4. Magnetic scalar potential around a wire --------------------------------------
I_w = 5.0
theta = np.linspace(0, 2 * np.pi, 1001)
psi = -I_w * theta / (2 * np.pi)
close(psi[-1] - psi[0], -I_w, rtol=1e-12, name="psi jump around the wire")
# H = -grad psi = I/(2 pi r) e_theta, circulation I
rr = 0.01
close(I_w / (2 * np.pi * rr) * 2 * np.pi * rr, I_w, rtol=1e-12, name="circulation of H")

# --- 5. A_z of a line current: flux between contours (per metre of depth) -----------
flux_12 = MU0 * I_w / (2 * np.pi) * np.log(0.02 / 0.01)
close(flux_12, 6.93e-7, rtol=3e-3, name="flux per metre between 1 cm and 2 cm")
close(MU0 * I_w / (2 * np.pi) * np.log(0.04 / 0.02), flux_12, rtol=1e-12, name="same between 2 cm and 4 cm")
# equal A_z steps -> equal radius ratios; B = |dA_z/dr| = mu0 I / (2 pi r)
Az = lambda r_, r_ref=1.0: -MU0 * I_w / (2 * np.pi) * np.log(r_ / r_ref)
h = 1e-7
for r_ in (0.01, 0.03):
    dAz = (Az(r_ + h) - Az(r_ - h)) / (2 * h)
    close(-dAz, MU0 * I_w / (2 * np.pi * r_), rtol=1e-6, name="B from A_z")
    # a different reference radius only adds a constant
    close(Az(r_, 2.0) - Az(r_, 1.0), Az(1.0, 2.0), rtol=1e-12, name="reference shift")
close(MU0 * I_w / (2 * np.pi * 0.01), 1.0e-4, rtol=1e-9, name="B at 1 cm from 5 A")

# axisymmetric flux function: for uniform B, 2 pi r A_theta = pi r^2 B
for r_ in (0.01, 0.05):
    close(2 * np.pi * r_ * (B0 * r_ / 2), np.pi * r_**2 * B0, rtol=1e-12, name="2 pi r A_theta")

# --- 6. Surface stream function on a square plate -----------------------------------
psi0 = 10.0                         # A, teaching value


def psi_s(x, y):
    return psi0 * (1 - x**2) * (1 - y**2)


def K(x, y):                        # K = grad psi x e_z = (d_y psi, -d_x psi)
    return np.array([psi0 * (1 - x**2) * (-2 * y), -psi0 * (-2 * x) * (1 - y**2)])


# divergence-free (numerically)
for (x0, y0) in ((0.3, -0.4), (-0.6, 0.2)):
    hh = 1e-6
    div = (K(x0 + hh, y0)[0] - K(x0 - hh, y0)[0]) / (2 * hh) + (K(x0, y0 + hh)[1] - K(x0, y0 - hh)[1]) / (2 * hh)
    assert abs(div) < 1e-5
# current crossing the segment y = 0, 0 <= x <= 1 equals psi(0,0) - psi(1,0) = psi0
xs = np.linspace(0, 1, 20001)
crossing = np.trapezoid(K(xs, 0 * xs)[1], xs)
close(crossing, psi0, rtol=1e-8, name="current crossing the cut")
# no current leaves the plate edge (normal component zero on x = 1)
ys = np.linspace(-1, 1, 11)
assert np.allclose(K(1.0 + 0 * ys, ys)[0], 0.0)
# counterclockwise circulation seen from +z: at (x>0, 0) K points along +y
assert K(0.5, 0.0)[1] > 0
# winding with N = 5 equally spaced contours: each turn carries psi0 / N
N = 5
close(psi0 / N, 2.0, rtol=1e-12, name="current per turn")
close(psi0 / 10, 1.0, rtol=1e-12, name="current per turn, N = 10")

print("Chapter 9 checks passed.")
print(f"  gauge example: E_sym(8 cm,0) = {E_sym(r0, 0)}, -dA_alt = {mdA_alt(r0, 0)}, -grad phi_alt = {mgradphi_alt(r0, 0)}")
print(f"  flux per metre between 1 and 2 cm (5 A) = {flux_12:.4e} Wb/m")
print(f"  stream plate: crossing current = {crossing:.6f} A (psi0 = {psi0} A)")
