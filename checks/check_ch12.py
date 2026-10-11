"""Numbers used in Chapter 12 (zh/ch12_eddy.tex) and its figures eddy_oneway_vs_coupled, coil_voltage.

Teaching inputs: Chapter 8 diffusion examples (copper 5.8e7 S/m, 2 mm; tissue-like 0.5 S/m, 0.2 m; 3 kHz),
the thin copper shell of Example 12.1 (R = 0.3 m, Delta = 2 mm, coil radius a = 0.25 m) and the old-edition
coil circuit (L = 20 uH, R = 0.1 ohm, I = 5000 A, dI/dt = 50 A/us).  Run:  python3 checks/check_ch12.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "figsrc")]

from numerics import MU0  # noqa: E402
from fig_ch12 import (COIL_A, I_PEAK, L_COIL, R_COIL, SHELL_DELTA, SHELL_KAPPA, SHELL_R, SHELL_SIGMA, SLEW,  # noqa: E402
                      coil_waveform, shell_response, shell_tau, trapezoid)


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:58s} {x:.6g}")
    return x


# --- magnetic diffusion times (Sec. eddy-diffusion), consistent with Chapter 8 ------------------------
w3k = 2 * np.pi * 3000.0
tau_cu = MU0 * 5.8e7 * 0.002**2
tau_ti = MU0 * 0.5 * 0.2**2
close(tau_cu * 1e3, 0.29, rtol=0.01, name="tau_m copper 2 mm (ms)")
close(tau_ti * 1e9, 25.0, rtol=0.01, name="tau_m tissue-like 0.2 m (ns)")
close(w3k * tau_cu, 5.50, rtol=0.002, name="eta_m copper at 3 kHz (Chapter 8: 5.50)")
close(w3k * tau_ti, 4.74e-4, rtol=0.002, name="eta_m tissue at 3 kHz (Chapter 8: 4.74e-4)")
delta3k = np.sqrt(2 / (w3k * MU0 * 5.8e7))
close(delta3k * 1e3, 1.207, rtol=1e-3, name="copper skin depth at 3 kHz (mm)")

# --- thin shell (Example 12.1) ------------------------------------------------------------------------
assert (SHELL_SIGMA, SHELL_DELTA, SHELL_R, COIL_A) == (5.8e7, 2e-3, 0.3, 0.25)
tau = shell_tau(SHELL_SIGMA, SHELL_DELTA, SHELL_R)
close(tau * 1e3, 21.9, rtol=0.002, name="shell time constant (ms)")
close(SHELL_KAPPA, 0.694, rtol=0.001, name="kappa = a^2/R^2")
close(1 - SHELL_KAPPA, 0.31, rtol=0.02, name="bore field right after a fast ramp")
Tr = 1e-4
close(1 - SHELL_KAPPA * (tau / Tr) * (1 - np.exp(-Tr / tau)), 0.307, rtol=0.002, name="exact value for a 0.1 ms ramp")
t99 = tau * np.log(100 * SHELL_KAPPA)
close(t99 * 1e3, 93.0, rtol=0.005, name="time to 99 % (ms)")
close(t99 / Tr, 927, rtol=0.005, name="... in units of the ramp time (about a thousand)")
fc = 1 / (2 * np.pi * tau)
close(fc, 7.3, rtol=0.01, name="1/(2 pi tau) (Hz)")
close(np.sqrt(2 / (2 * np.pi * fc * MU0 * SHELL_SIGMA)) * 100, 2.4, rtol=0.03, name="skin depth at that frequency (cm)")
close(MU0 * SHELL_SIGMA * SHELL_DELTA**2 * 1e3, 0.29, rtol=0.01, name="diffusion time through the thickness (ms)")
close(tau / (MU0 * SHELL_SIGMA * SHELL_DELTA**2), SHELL_R / (2 * SHELL_DELTA), rtol=1e-12, name="ratio = R/(2 Delta)")
close(SHELL_R / (2 * SHELL_DELTA), 75.0, rtol=1e-12, name="R/(2 Delta)")
# self-check 4: half the conductivity
close(shell_tau(SHELL_SIGMA / 2, SHELL_DELTA, SHELL_R) * 1e3, 10.9, rtol=0.005, name="tau with sigma/2 (ms)")

# the shell ODE derived from Faraday + Ohm agrees with the field picture (direct integration)
n_t = 1000.0
def rhs(t, y, ramp=Tr):
    # y = [K_s];  flux inside R: mu0 (pi a^2 K_c + pi R^2 K_s);  2 pi R E = -dPhi/dt;  K_s = sigma Delta E
    Kc_dot = n_t * (1.0 / ramp if t < ramp else 0.0)          # I rises 0 -> 1 A in Tr
    Ks = y[0]
    # sigma Delta (-mu0/(2R)) (a^2 Kc' + R^2 Ks') = Ks  ->  Ks' = -(Ks/(sigma Delta mu0 R/2) + a^2 Kc'/R)/R
    Ks_dot = -(Ks / (SHELL_SIGMA * SHELL_DELTA * MU0 / (2 * SHELL_R)) + COIL_A**2 * Kc_dot) / SHELL_R**2
    return [Ks_dot]
sol = solve_ivp(rhs, (0, Tr), [0.0], max_step=Tr / 2000, rtol=1e-10, atol=1e-12)
Ks_end = sol.y[0, -1]
close((n_t + Ks_end) / n_t, 0.307, rtol=0.002, name="direct integration: bore field after the ramp")

# circuit picture: shorted single turn gives the same tau and kappa (length ell, end effects ignored)
ell = 1.0
L11 = MU0 * np.pi * COIL_A**2 * n_t**2 * ell
L22 = MU0 * np.pi * SHELL_R**2 / ell
L12 = MU0 * np.pi * COIL_A**2 * n_t
R2 = 2 * np.pi * SHELL_R / (SHELL_SIGMA * SHELL_DELTA * ell)
close(L22 / R2, tau, rtol=1e-12, name="L22/R2 = tau")
close(L12**2 / (L11 * L22), SHELL_KAPPA, rtol=1e-12, name="coupling^2 = kappa")
for w, ref in ((1e-3 / tau, 1.0), (1e3 / tau, 1 - SHELL_KAPPA)):
    Z = 1j * w * L11 + w**2 * L12**2 / (R2 + 1j * w * L22)
    close(Z.imag / w / L11, ref, rtol=1e-4, name=f"effective L/L11 at omega tau = {w * tau:g}")

# normalised figure curve: exact superposition solution vs ODE integration
t = np.linspace(0, 5.5, 5501)
tr, tf = 0.06, 2.4
Ks_fig = shell_response(t, SHELL_KAPPA, tr, tf)
Kc_fig = trapezoid(t, tr, tf)
dKc = np.gradient(Kc_fig, t)
ode = solve_ivp(lambda tt, y: [-y[0] - SHELL_KAPPA * np.interp(tt, t, dKc)], (0, 5.5), [0.0], t_eval=t,
                max_step=0.002, rtol=1e-9, atol=1e-12)
assert np.max(np.abs(ode.y[0] - Ks_fig)) < 5e-3
print("  ok  figure curve = solution of tau Ks' + Ks = -kappa tau Kc'")

# --- coil voltage (Example 12.2, old-edition teaching values) ---------------------------------------
assert (L_COIL, R_COIL, I_PEAK, SLEW) == (20e-6, 0.1, 5000.0, 50e6)
close(R_COIL * I_PEAK, 500.0, name="resistive voltage (V)")
close(L_COIL * SLEW, 1000.0, name="inductive voltage (V)")
close(R_COIL * I_PEAK + L_COIL * SLEW, 1500.0, name="voltage at the end of the ramp (V)")
close(0.5 * L_COIL * I_PEAK**2, 250.0, name="stored energy (J)")
close(I_PEAK / SLEW * 1e6, 100.0, name="ramp time (us)")
close(I_PEAK**2 * R_COIL / 1e6, 2.5, name="resistive power on the flat top (MW)")
tt = np.linspace(-50e-6, 600e-6, 65001)
I, dI = coil_waveform(tt)
V = R_COIL * I + L_COIL * dI
close(V.max(), 1500.0, rtol=2e-3, name="max of the plotted voltage (V)")
close(V.min(), -1000.0, rtol=2e-3, name="min of the plotted voltage (V)")
# self-check 5: half the slew rate
close(L_COIL * SLEW / 2, 500.0, name="inductive voltage at half slew (V)")
close(I_PEAK / (SLEW / 2) * 1e6, 200.0, name="ramp time at half slew (us)")

# --- moving conductor ------------------------------------------------------------------------------
close(1.0 * 3.0, 3.0, name="|u x B| for 1 m/s across 3 T (V/m)")
# rigid rotation about an axis parallel to uniform B: u x B = Omega B (x, y, 0) is curl-free
Om, B = 2.0, 3.0
def uxB(x, y):
    u = np.array([-Om * y, Om * x, 0.0])
    return np.cross(u, [0.0, 0.0, B])
h = 1e-6
x0, y0 = 0.3, -0.2
curl_z = ((uxB(x0 + h, y0)[1] - uxB(x0 - h, y0)[1]) - (uxB(x0, y0 + h)[0] - uxB(x0, y0 - h)[0])) / (2 * h)
assert abs(curl_z) < 1e-6 and np.allclose(uxB(x0, y0), Om * B * np.array([x0, y0, 0]))
print("  ok  rotation parallel to B: u x B = Omega B (x, y, 0), curl-free")
# rotation about x (perpendicular to B along z): curl(u x B) = Omega x B != 0
def uxB2(p):
    u = np.cross([Om, 0, 0], p)
    return np.cross(u, [0.0, 0.0, B])
p0 = np.array([0.1, 0.2, 0.3])
J = np.zeros((3, 3))
for k in range(3):
    e = np.zeros(3); e[k] = h
    J[:, k] = (uxB2(p0 + e) - uxB2(p0 - e)) / (2 * h)
curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
assert np.allclose(curl, np.cross([Om, 0, 0], [0, 0, B]), atol=1e-6)
print("  ok  rotation perpendicular to B: curl(u x B) = Omega x B")
print("all Chapter 12 checks passed")
