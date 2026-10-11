"""Numbers used in Chapter 7 (zh/ch07_systems.tex).

Capacitor inputs follow the old edition (conn:6, wc:3); the solenoid follows the
running example of Chapter 4 (conn:8); the gapped ring core is a teaching example
(linear iron, mu_r = 1000 as in Ch. 5).  Run:  python3 checks/check_ch07.py
"""
from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi
EPS0 = 8.8541878128e-12


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    return x


# --- Parallel-plate capacitor (old conn:6 / wc:3) -------------------------------
A, d, er, Q = 0.01, 1e-3, 4.0, 1e-9
eps = EPS0 * er
D = Q / A
E = D / eps
V = E * d
C = eps * A / d
P = D - EPS0 * E
W = 0.5 * C * V**2
close(D, 1e-7, name="D")
close(E, 2.82e3, name="E")
close(V, 2.82, name="V")
close(C, 354e-12, name="C")
close(P, 7.5e-8, name="P")
close(W, 1.41e-9, name="W_e")
close(Q * d / (EPS0 * A), 11.3, name="vacuum-gap voltage")
close(W, 0.5 * E * D * A * d, rtol=1e-12, name="field energy = circuit energy")
close(W, Q**2 / (2 * C), rtol=1e-12, name="Q^2/2C")

# --- Resistors: the two layers of Chapter 6 in series ---------------------------
Ar = 1e-4
R1 = 8e-3 / (0.2 * Ar)
R2 = 2e-3 / (0.02 * Ar)
assert np.isclose(R1, 400.0) and np.isclose(R2, 1000.0) and np.isclose(R1 + R2, 1400.0)

# Poynting inflow into a cylindrical resistor equals I^2 R per unit length
sigma, a_r, I_r = 0.2, 0.01, 1e-3
Ez = I_r / (sigma * np.pi * a_r**2)
Htheta = I_r / (2 * np.pi * a_r)
close(2 * np.pi * a_r * Ez * Htheta, I_r**2 / (sigma * np.pi * a_r**2), rtol=1e-12, name="Poynting inflow")

# --- Long solenoid inductor (running example of Chapter 4) ----------------------
n_t, a, ell, I, dI = 1000.0, 0.02, 1.0, 5.0, 1.0e6
N = n_t * ell
Ac = np.pi * a**2
L = MU0 * N**2 * Ac / ell
B = MU0 * n_t * I
lam = N * B * Ac
Wm = 0.5 * L * I**2
wm = B**2 / (2 * MU0)
close(L, 1.579e-3, name="L")
close(lam, 7.90e-3, name="lambda")
close(lam / I, L, rtol=1e-12, name="L = lambda/I")
close(Wm, 19.7e-3, name="W_m")
close(wm, 15.7, name="w_m")
close(wm * Ac * ell, Wm, rtol=1e-12, name="field energy = LI^2/2")
close(L * dI, 1579.0, name="L dI/dt")
close(L * dI, N * np.pi * a**2 * MU0 * n_t * dI, rtol=1e-12, name="= N x EMF per turn")

# --- Gapped ring core (teaching example, linear iron) ---------------------------
ly, g, Acore, mu_r, Nc, Ic = 0.2, 1e-3, 4e-4, 1000.0, 200, 2.0
F = Nc * Ic
Ry = ly / (mu_r * MU0 * Acore)
Rg = g / (MU0 * Acore)
Phi = F / (Ry + Rg)
Bg = Phi / Acore
Hy = Bg / (mu_r * MU0)
Hg = Bg / MU0
Lc = Nc**2 / (Ry + Rg)
close(F, 400.0, name="MMF")
close(Ry, 3.98e5, name="R_yoke")
close(Rg, 1.99e6, name="R_gap")
close(Rg / Ry, 5.0, rtol=1e-12, name="gap/yoke reluctance ratio")
close(Rg / (Ry + Rg), 0.833, name="MMF share of gap")
close(g / (g + ly), 0.005, rtol=1e-2, name="length share of gap")
close(Phi, 1.68e-4, name="flux")
close(Bg, 0.419, name="B")
close(Bg, MU0 * F / (g + ly / mu_r), rtol=1e-12, name="equivalent-gap formula")
close(ly / mu_r, 2e-4, rtol=1e-12, name="equivalent gap length")
close(Hy * ly, 66.7, name="H_y l_y")
close(Hg * g, 333.3, name="H_g g")
close(Hy * ly + Hg * g, F, rtol=1e-12, name="Ampere loop")
close(Lc, 16.8e-3, name="L with gap")
close(Nc**2 / Ry, 100.5e-3, name="L without gap")
close(MU0 * mu_r * F / ly, 2.51, name="B without gap (linear model)")
# energy: same flux through both series elements, W_i = Phi^2 R_i / 2
Wy, Wg = 0.5 * Phi**2 * Ry, 0.5 * Phi**2 * Rg
close(Wg / (Wy + Wg), 0.833, name="energy share of gap")
close(Wy + Wg, 0.5 * Lc * Ic**2, rtol=1e-12, name="W = LI^2/2")
# field-integral check of the gap energy
close(Wg, Bg**2 / (2 * MU0) * Acore * g, rtol=1e-12, name="gap field energy")

# --- Self-check numbers ---------------------------------------------------------
# capacitor: vacuum capacitance and the factor 1/4 in stored energy at fixed Q
C0 = EPS0 * A / d
close(Q**2 / (2 * C0) / W, 4.0, rtol=1e-12, name="fixed-Q energy ratio")
# gapped core with the gap doubled
Rg2 = 2 * g / (MU0 * Acore)
B2 = F / (Ry + Rg2) / Acore
L2 = Nc**2 / (Ry + Rg2)
close(B2, 0.228, name="B, gap doubled")
close(L2, 9.14e-3, name="L, gap doubled")
close(B2 / Bg, 0.545, name="B ratio, gap doubled")
# peak vs RMS dissipation: sigma = 0.5 S/m, 100 V/m (old extended_checks inputs)
sig_t, E0 = 0.5, 100.0
close(sig_t * E0**2 / 2, 2500.0, rtol=1e-12, name="time-average loss, peak amplitude")
close(sig_t * E0**2, 5000.0, rtol=1e-12, name="time-average loss, RMS amplitude")

print("Chapter 7 checks passed.")
print(f"  capacitor: D={D:.3g}, E={E:.4g} V/m, V={V:.4f} V, C={C*1e12:.1f} pF, P={P:.3g}, W={W:.3g} J")
print(f"  solenoid: L={L*1e3:.3f} mH, lambda={lam*1e3:.3f} mWb, W={Wm*1e3:.2f} mJ, w={wm:.2f} J/m^3, L dI/dt={L*dI:.0f} V")
print(f"  core: Ry={Ry:.4g}, Rg={Rg:.4g}, Phi={Phi:.4g} Wb, B={Bg:.4f} T, HyLy={Hy*ly:.2f} A, Hg g={Hg*g:.2f} A")
print(f"        L={Lc*1e3:.2f} mH, L_nogap={Nc**2/Ry*1e3:.1f} mH, B_nogap={MU0*mu_r*F/ly:.3f} T")
