"""Numbers used in Chapter 6 (zh/ch06_interfaces.tex).

The two-layer conductor (old conn:7, form:10, wc:4) is reproduced and every
derived value quoted in the chapter is asserted.  Run:  python3 checks/check_ch06.py
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12
QE = 1.602176634e-19


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    return x


# --- Two-layer conductor (teaching inputs from the old edition) ---------------
A = 1e-4                       # m^2
d = np.array([8e-3, 2e-3])     # m
sig = np.array([0.2, 0.02])    # S/m
I = 1e-3                       # A
J = I / A
E = J / sig
V_layer = E * d
V = V_layer.sum()
R = V / I
P = I * V
p_density = sig * E**2
P_layer = p_density * A * d
R_layer = d / (sig * A)
close(J, 10.0, name="J")
assert np.allclose(E, [50.0, 500.0])
assert np.allclose(V_layer, [0.4, 1.0])
close(V, 1.4, name="Delta V")
close(R, 1400.0, name="R")
close(P, 1.4e-3, name="P")
assert np.allclose(p_density, [500.0, 5000.0])
assert np.allclose(P_layer, [0.4e-3, 1.0e-3])
assert np.allclose(R_layer, [400.0, 1000.0])
close(P_layer.sum(), P, rtol=1e-12, name="sum of layer powers")
close(A * np.sum(sig * E**2 * d), I**2 * np.sum(d / (sig * A)), rtol=1e-12, name="two power formulas")
close(V_layer[1] / V, 0.714, name="share of voltage in thin layer")

# total (free + bound) surface charge at the interface: eps0 (E2n - E1n)
q_total = EPS0 * (E[1] - E[0])
close(q_total, 3.98e-9, name="total interface charge")
per_mm2 = q_total / QE * 1e-6
close(per_mm2, 2.49e4, rtol=1e-2, name="elementary charges per mm^2")

# --- Refraction at a conductivity interface (figure interface, sigma2 = sigma1/4)
theta1 = np.deg2rad(60.0)
theta2 = np.arctan(np.tan(theta1) * 0.25)
close(np.rad2deg(theta2), 23.4, rtol=2e-3, name="refracted angle")
# components: E_t continuous, sigma1 E1n = sigma2 E2n
E1 = np.array([np.sin(theta1), np.cos(theta1)])          # (t, n), |E1| = 1
E2 = np.array([E1[0], E1[1] * 4.0])
close(np.arctan2(E2[0], E2[1]), theta2, rtol=1e-12, name="angle from components")

# --- Air / iron interface, mu_r = 1000 ------------------------------------------
mu_r = 1000.0
theta_iron = np.deg2rad(89.0)
theta_air = np.arctan(np.tan(theta_iron) / mu_r)
close(np.rad2deg(theta_air), 3.28, rtol=5e-3, name="air-side angle")

print("Chapter 6 checks passed.")
print(f"  J = {J} A/m^2, E = {E} V/m, layer V = {V_layer} V, V = {V:.3f} V, R = {R:.1f} ohm")
print(f"  p = {p_density} W/m^3, layer P = {P_layer*1e3} mW, R_i = {R_layer} ohm")
print(f"  interface charge = {q_total:.4g} C/m^2 = {per_mm2:.3g} e/mm^2")
print(f"  refraction 60 deg -> {np.rad2deg(theta2):.2f} deg; iron 89 deg -> air {np.rad2deg(theta_air):.3f} deg")
