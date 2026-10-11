"""Numbers used in Chapter 4 (zh/ch04_magnetic.tex).

Old-edition inputs (conn:8, wc:6, wc:7, form:5) are reproduced; every derived
number quoted in the chapter is asserted here.  Run:  python3 checks/check_ch04.py
"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4.0e-7 * np.pi  # same teaching value as numerics.py


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    return x


# --- Uniform dB/dt inside a cylinder (form:5): E = r/2 dB/dt -------------------
dBdt, r = 1000.0, 0.08
E_uniform = r / 2 * dBdt
close(E_uniform, 40.0, name="uniform induction E")

# --- Ideal long solenoid (conn:8, wc:6, wc:7) ----------------------------------
n_t, a, I, dI, r_in, sigma = 1000.0, 0.02, 5.0, 1.0e6, 0.01, 0.2
Bc = MU0 * n_t * I
dBc = MU0 * n_t * dI
close(Bc, 6.28e-3, name="B_c")
close(dBc, 1.26e3, name="dB_c/dt")
A_in = Bc * r_in / 2
E_in = -dBc * r_in / 2
close(A_in, 3.14e-5, name="A_theta(1 cm)")
close(E_in, -6.28, name="E_theta(1 cm)")
close(sigma * E_in, -1.26, name="J_theta(1 cm)")
# A_theta from 'circulation = flux' agrees with the formulas, inside and outside
for rr in (0.005, 0.01, 0.02, 0.04, 0.1):
    flux = np.pi * min(rr, a) ** 2 * Bc
    A_formula = Bc * rr / 2 if rr < a else Bc * a**2 / (2 * rr)
    close(flux / (2 * np.pi * rr), A_formula, rtol=1e-12, name="A from flux")
# outside the winding (r = 4 cm = 2a): B = 0 but E is as large as at r = 1 cm
r_out = 0.04
E_out = -dBc * a**2 / (2 * r_out)
close(E_out, -6.28, name="E_theta(4 cm)")
close(-dBc * a / 2, -12.57, name="E_theta at r=a (extreme)")
# EMF of one turn around the solenoid, any radius > a
emf_turn = -np.pi * a**2 * dBc
close(emf_turn, -1.579, name="EMF per turn")
for rr in (0.03, 0.05, 0.2):
    close(2 * np.pi * rr * (-dBc * a**2 / (2 * rr)), emf_turn, rtol=1e-12, name="EMF independent of r")
# outside region: local curl of E_theta(r) e_theta = (1/r) d(r E)/dr = 0
rr = np.linspace(0.03, 0.2, 200)
rE = rr * (-dBc * a**2 / (2 * rr))
assert np.allclose(np.gradient(rE, rr), 0.0, atol=1e-9)

# --- Circular loop on axis (eq:loopaxis) ---------------------------------------
z_half = np.sqrt(2 ** (2 / 3) - 1)  # B_z(z)/B_z(0) = 1/2  (in units of a)
close(z_half, 0.766, name="half-maximum distance")
close((1 + z_half**2) ** -1.5, 0.5, rtol=1e-12, name="half maximum check")


def biot_savart_loop_axis(z, a=1.0, nseg=12):
    """B_z on the axis of a unit-current loop from a polygon of nseg straight segments
    (mu0/4pi = 1); exact straight-segment formula summed over segments."""
    phi = np.linspace(0, 2 * np.pi, nseg + 1)
    P = np.stack((a * np.cos(phi), a * np.sin(phi), np.zeros_like(phi)), axis=1)
    obs = np.array([0.0, 0.0, z])
    Bz = 0.0
    for k in range(nseg):
        p1, p2 = P[k], P[k + 1]
        dl = p2 - p1
        r1, r2 = obs - p1, obs - p2
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        c = np.cross(dl, r1)
        Bz += c[2] / np.dot(c, c) * (np.dot(dl, r1) / n1 - np.dot(dl, r2) / n2)
    return Bz


def loop_axis_exact(z, a=1.0):
    return 2 * np.pi * a**2 / (a**2 + z**2) ** 1.5  # mu0/4pi = 1, I = 1


for z in (0.0, 0.5, 1.0, 3.0):
    # 360 straight segments reproduce the circle to < 1e-4
    close(biot_savart_loop_axis(z, nseg=360), loop_axis_exact(z), rtol=1e-4, name="Biot-Savart vs loop formula")

# polygon (n-gon) approximations used in the figure loop_axis: centre-field ratio n tan(pi/n)/pi
for nseg, ratio in ((6, 1.103), (24, 1.0057)):
    close(biot_savart_loop_axis(0.0, nseg=nseg) / loop_axis_exact(0.0), ratio, rtol=1e-3, name=f"{nseg}-gon")
    close(nseg * np.tan(np.pi / nseg) / np.pi, ratio, rtol=1e-3, name=f"{nseg}-gon closed form")

# --- Self-check: the same 5 A reached by a 0.1 ms linear ramp -------------------
ramp = 5.0 / 1e-4                     # A/s
dB_ramp = MU0 * n_t * ramp
close(ramp, 5e4, name="ramp rate")
close(dB_ramp, 62.8, name="dB/dt during ramp")
close(dB_ramp * r_in / 2, 0.314, name="E(1 cm) during ramp")
# self-check: the alternative potential (0, B0 x, 0) has circulation pi r^2 B0 on a circle
B0, rr = 1.0, 0.7
t = np.linspace(0, 2 * np.pi, 20001)
circ = np.trapezoid(B0 * rr * np.cos(t) * rr * np.cos(t), t)
close(circ, np.pi * rr**2 * B0, rtol=1e-9, name="circulation of A_alt")

# --- Figure flux_linkage: flux of three pick-up turns in the field of a source loop ----
def loop_psi(rho, z, a=1.0):
    """rho*A_phi of a unit loop (constant factor dropped), Jackson (5.37)."""
    k2 = 4 * a * rho / ((a + rho) ** 2 + z**2)
    return rho * np.sqrt(a / rho) / np.sqrt(k2) * ((1 - k2 / 2) * ellipk(k2) - ellipe(k2))


phis = loop_psi(0.5, np.array([0.55, 1.05, 1.55]), 1.0)
ratios = phis / phis[0]
assert np.allclose(np.round(ratios, 2), [1.00, 0.46, 0.23]), ratios
close(ratios.sum(), 1.69, rtol=3e-3, name="lambda / Phi_1")
assert ratios[2] < 0.25                       # far turn links less than a quarter
close(3 / ratios.sum() - 1, 0.775, rtol=1e-2, name="overestimate of N*Phi_1")
print("Chapter 4 checks passed.")
print(f"  B_c = {Bc*1e3:.3f} mT, dB_c/dt = {dBc:.4g} T/s")
print(f"  A_theta(1 cm) = {A_in:.4g} Wb/m, E_theta(1 cm) = {E_in:.4f} V/m, J = {sigma*E_in:.4f} A/m^2")
print(f"  E_theta(4 cm) = {E_out:.4f} V/m, E_theta(a) = {-dBc*a/2:.3f} V/m, EMF/turn = {emf_turn:.4f} V")
print(f"  uniform induction E = {E_uniform:.1f} V/m, half-max z = {z_half:.4f} a")
print(f"  flux_linkage turns: {np.round(ratios, 3)}, lambda = {ratios.sum():.3f} Phi_1")
