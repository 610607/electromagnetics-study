"""Numbers used in Chapter 8 (zh/ch08_regimes.tex).

Old-edition inputs (mat:8: f = 3 kHz, sigma = 0.5 S/m with l = 0.20 m, copper
5.8e7 S/m with l = 2 mm; mat:10: 128 MHz, eps_r = 80 comparison medium) are
reproduced with the same formulas as numerics.run_all("scales_at_3kHz");
every derived number quoted in the chapter is asserted here.
Run:  python3 checks/check_ch08.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numerics  # noqa: E402

MU0 = numerics.MU0          # 4 pi 1e-7, same teaching value as numerics.py
EPS0 = numerics.EPS0
C0 = 1.0 / np.sqrt(MU0 * EPS0)


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    return x


# --- Section 8.3 / 8.8: tissue-like conductor vs copper at 3 kHz (old mat:8) ---------
f = 3000.0
w = 2 * np.pi * f
close(w, 1.885e4, name="omega")
close(1 / w, 5.31e-5, name="1/omega")


def delta(sigma, mu=MU0):
    return np.sqrt(2.0 / (w * mu * sigma))


def eta_m(sigma, ell, mu=MU0):
    return w * mu * sigma * ell**2


cases = {"tissue": (0.5, 0.20), "copper": (5.8e7, 2e-3)}
d_t, d_c = delta(0.5), delta(5.8e7)
em_t, em_c = eta_m(0.5, 0.20), eta_m(5.8e7, 2e-3)
close(d_t, 12.995, rtol=1e-4, name="delta tissue")
close(d_c, 1.207e-3, rtol=1e-3, name="delta copper")
close(em_t, 4.74e-4, rtol=1e-3, name="eta_m tissue")
close(em_c, 5.50, rtol=2e-3, name="eta_m copper")
close(em_c / em_t, 1.16e4, rtol=5e-3, name="ratio of eta_m")
# eta_m = 2 (l/delta)^2  (eq:skindepth)
for (s, ell), d in zip(cases.values(), (d_t, d_c)):
    close(eta_m(s, ell), 2 * (ell / d) ** 2, rtol=1e-12, name="eta_m = 2(l/delta)^2")
close(0.20 / d_t, 0.0154, rtol=3e-3, name="l/delta tissue")
close(2e-3 / d_c, 1.66, rtol=3e-3, name="l/delta copper")
# magnetic diffusion times tau_m = mu sigma l^2 and eta_m = omega tau_m
tau_t, tau_c = MU0 * 0.5 * 0.2**2, MU0 * 5.8e7 * 2e-3**2
close(tau_t, 2.51e-8, name="tau_m tissue")
close(tau_c, 2.92e-4, name="tau_m copper")
close(w * tau_c, em_c, rtol=1e-12, name="eta_m = omega tau_m")

# Same values from the shared numerics module (the old edition's own script)
with tempfile.TemporaryDirectory() as tmp:
    res = numerics.run_all(Path(tmp) / "r.json")["scales_at_3kHz"]
close(res["tissue_illustration"]["delta_m"], d_t, rtol=1e-12, name="numerics delta tissue")
close(res["copper"]["delta_m"], d_c, rtol=1e-12, name="numerics delta copper")
close(res["tissue_illustration"]["magnetic_induction_number"], em_t, rtol=1e-12, name="numerics eta tissue")
close(res["copper"]["magnetic_induction_number"], em_c, rtol=1e-12, name="numerics eta copper")

# Transit (wave) number in the air around the device, eta_w = omega l / c
close(w * 0.20 / C0, 1.26e-5, rtol=3e-3, name="eta_w for l = 0.2 m at 3 kHz")
close(w * 2e-3 / C0, 1.26e-7, rtol=3e-3, name="eta_w for l = 2 mm at 3 kHz")
close(C0 / f, 1.0e5, rtol=3e-3, name="free-space wavelength at 3 kHz")
close(0.20 / C0, 6.67e-10, rtol=3e-3, name="transit time over 0.2 m")

# eta_w^2 = eta_d eta_m (identity), checked for arbitrary values
for sig, eps_r, ell in ((0.5, 1e3, 0.2), (5.8e7, 1.0, 2e-3), (2.0, 50.0, 0.05)):
    eta_w = w * ell * np.sqrt(MU0 * eps_r * EPS0)
    eta_d = w * eps_r * EPS0 / sig
    close(eta_w**2, eta_d * eta_m(sig, ell), rtol=1e-12, name="eta_w^2 = eta_d eta_m")

# Example: permittivity at which displacement current equals conduction current
# (sigma = 0.5 S/m, 3 kHz):  eta_d = omega eps0 eps_r / sigma = 1
eps_r_break = 0.5 / (w * EPS0)
close(eps_r_break, 3.0e6, rtol=3e-3, name="break-even eps_r")
close(w * EPS0 / 0.5, 3.34e-7, rtol=3e-3, name="eta_d per unit eps_r")
close(0.01 * eps_r_break, 3.0e4, rtol=3e-3, name="eps_r for eta_d = 0.01")

# --- Section 8.2: phasors.  Heating from two peak phasors (sigma = 0.5 S/m, 100 V/m)
sigma = 0.5
E1 = 100.0
for phase, p_ref in ((0.0, 1.0e4), (np.pi / 2, 5.0e3), (np.pi, 0.0)):
    E2 = 100.0 * np.exp(1j * phase)
    p_avg = 0.5 * sigma * abs(E1 + E2) ** 2
    assert np.isclose(p_avg, p_ref, atol=1e-9), (phase, p_avg)
    # time average of sigma E(t)^2 with E(t) = Re{(E1 + E2) e^{jwt}}
    t = np.linspace(0, 2 * np.pi, 20001)[:-1]
    e_t = np.real((E1 + E2) * np.exp(1j * t))
    close(np.mean(sigma * e_t**2) + 1e-30, p_ref + 1e-30, rtol=1e-9, name="time average")
p_sep = 0.5 * sigma * E1**2 * 2
close(p_sep, 5.0e3, rtol=1e-12, name="sum of separate heatings")
close(0.5 * sigma * E1**2, 2.5e3, rtol=1e-12, name="single source peak-phasor heating")

# --- Section 8.8: half-space diffusion  H = H0 exp(-q x), q^2 = j w mu sigma, q = (1+j)/delta
for s in (0.5, 5.8e7):
    q = np.sqrt(1j * w * MU0 * s)
    close(q.real, 1 / delta(s), rtol=1e-12, name="Re q = 1/delta")
    close(q.imag, 1 / delta(s), rtol=1e-12, name="Im q = 1/delta")
# one skin depth: amplitude e^-1 and phase lag 1 rad
h = np.exp(-(1 + 1j))
close(abs(h), np.exp(-1), rtol=1e-12, name="amplitude at delta")
close(-np.angle(h), 1.0, rtol=1e-12, name="phase lag at delta")
close(np.exp(-1), 0.368, rtol=2e-3, name="1/e")
# current density J_z = dH_y/dx has the same envelope, |J| = sqrt(2) H0/delta e^{-x/delta}
close(abs(1 + 1j), np.sqrt(2), rtol=1e-12, name="|q| delta")
# depth where the amplitude falls to 1 %: x = ln(100) delta
close(np.log(100), 4.6, rtol=2e-3, name="1 % depth in deltas")

# --- Section 8.9: wave number; good-conductor limit equals the diffusion result
for s in (5.8e7,):
    k2 = w**2 * MU0 * EPS0 - 1j * w * MU0 * s
    k = np.sqrt(k2)
    if k.imag > 0:
        k = -k
    k = k if k.real >= 0 else -k
    kc = (1 - 1j) / delta(s)
    assert abs(k - kc) / abs(kc) < 1e-12, (k, kc)
    close((1j * kc).real, 1 / delta(s), rtol=1e-12, name="q = j k")
# eta_d of copper at 3 kHz with eps = eps0 (how small the displacement term is)
close(w * EPS0 / 5.8e7, 2.88e-15, rtol=5e-3, name="eta_d copper (eps0)")

# --- Section 8.10: 128 MHz wavelengths (old mat:10)
f_rf = 128e6
lam0 = C0 / f_rf
lam80 = lam0 / np.sqrt(80.0)
close(lam0, 2.34, rtol=2e-3, name="free-space wavelength 128 MHz")
close(lam80, 0.262, rtol=2e-3, name="wavelength eps_r = 80")
close(2 * np.pi * 0.20 / lam0, 0.537, rtol=3e-3, name="k0 l (l = 0.2 m)")
close(2 * np.pi * 0.20 / lam80, 4.80, rtol=3e-3, name="k l in eps_r = 80 (l = 0.2 m)")
close(lam0 / (2 * np.pi), 0.373, rtol=3e-3, name="lambda0/2pi")
close(lam80 / (2 * np.pi), 0.0417, rtol=3e-3, name="lambda80/2pi")
close(f_rf / f, 4.27e4, rtol=3e-3, name="128 MHz / 3 kHz")

# --- good-conductor |k| l = sqrt(eta_m) (copper, 3 kHz)
k_cu = np.sqrt(-1j * w * MU0 * 5.8e7)
close(abs(k_cu) * 2e-3, np.sqrt(em_c), rtol=1e-12, name="|k| l = sqrt(eta_m)")
close(np.sqrt(em_c), 2.34, rtol=3e-3, name="sqrt(eta_m) copper")
# general: |k l|^2 = eta_m sqrt(1 + eta_d^2) and k^2 = -j w mu sigma (1 + j eta_d)
for sig, eps_r, ell in ((0.5, 1e3, 0.2), (2.0, 50.0, 0.05)):
    k2 = w**2 * MU0 * eps_r * EPS0 - 1j * w * MU0 * sig
    ed = w * eps_r * EPS0 / sig
    close(abs(k2) * ell**2, eta_m(sig, ell) * np.sqrt(1 + ed**2), rtol=1e-12, name="|k l|^2")
    assert abs(k2 - (-1j * w * MU0 * sig * (1 + 1j * ed))) < 1e-12 * abs(k2)

# --- Self-check answers
close(eta_m(5.8e7, 0.2e-3), 0.055, rtol=3e-3, name="eta_m for 0.2 mm copper")
w30 = 2 * np.pi * 3e4
close(np.sqrt(2 / (w30 * MU0 * 0.5)), 4.11, rtol=3e-3, name="delta tissue 30 kHz")
close(w30 * MU0 * 0.5 * 0.2**2, 4.74e-3, rtol=3e-3, name="eta_m tissue 30 kHz")
close(0.5 * 0.5 * 100.0**2 * abs(1 + np.exp(1j * np.pi / 3)) ** 2, 7500.0, rtol=1e-12, name="60 degree phasor heating")
close(2 * np.pi * 0.05 / lam0, 0.134, rtol=3e-3, name="kr at 5 cm, 128 MHz")
# rise-time scale: omega ~ 1/t_r for a few tens of microseconds is ~1e4 rad/s
for tr in (20e-6, 50e-6):
    assert 1e4 <= 1 / tr <= 1e5

print("Chapter 8 checks passed.")
print(f"  3 kHz: omega = {w:.5g} rad/s, 1/omega = {1/w:.3e} s")
print(f"  tissue: delta = {d_t:.4f} m, eta_m = {em_t:.4e}, l/delta = {0.2/d_t:.4f}, tau_m = {tau_t:.3e} s")
print(f"  copper: delta = {d_c*1e3:.4f} mm, eta_m = {em_c:.4f}, l/delta = {2e-3/d_c:.4f}, tau_m = {tau_c:.3e} s")
print(f"  eta_w(0.2 m) = {w*0.2/C0:.3e}, break-even eps_r = {eps_r_break:.4g}")
print(f"  128 MHz: lambda0 = {lam0:.4f} m, lambda(80) = {lam80:.4f} m, k0 l = {2*np.pi*0.2/lam0:.3f}, k l = {2*np.pi*0.2/lam80:.3f}")
