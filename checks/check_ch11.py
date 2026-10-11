"""Numbers used in Chapter 11 (zh/ch11_tms.tex) and its computed figures.

Teaching model: the insulated unit square of numerics.induced_field_example driven by
E_p = 0.5 (y - 1/2, -(x - 1/2)), i.e. -dA/dt of a uniform dB_z/dt = 1 about the centre.  The worked
example scales it to a long square prism of side L = 0.16 m in dB_z/dt = 1000 T/s (Chapter 4 numbers:
r = 0.08 m -> 40 V/m).  Run:  python3 checks/check_ch11.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "figsrc")]

from numerics import MU0, induced_field_example, stiffness  # noqa: E402
from fig_ch10 import grad, model, solve_neumann, tes_uniform  # noqa: E402
from fig_ch11 import (PULSE_F, PULSE_TAU, primary, profile_data, pulse_waveform, sample, stream_tms,  # noqa: E402
                      tms_rhs, tms_solve)


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:58s} {x:.6g}")
    return x


# --- the solver reproduces numerics.induced_field_example ----------------------------------------
p, cells, phi, E, sigma, res, ortho = induced_field_example(64)
m = model(64)
s = tms_solve(m)
assert np.abs(s["E"] - E).max() < 1e-12
print("  ok  fig_ch11.tms_solve == numerics.induced_field_example (n = 64)")
assert abs(ortho) < 1e-12 and res < 1e-10
close(abs(ortho) + 1e-17, 1e-17, rtol=2.0, name="weak orthogonality sum (order 1e-17)")

# --- primary field: uniform dB/dt, curl = -1 ------------------------------------------------------
xy = np.array([[0.3, 0.7], [0.9, 0.1]])
h = 1e-6
dEy_dx = (primary(xy + [h, 0])[:, 1] - primary(xy - [h, 0])[:, 1]) / (2 * h)
dEx_dy = (primary(xy + [0, h])[:, 0] - primary(xy - [0, h])[:, 0]) / (2 * h)
assert np.allclose(dEy_dx - dEx_dy, -1.0)
print("  ok  curl_z E_p = -dB_z/dt = -1 (clockwise for B out of the page, increasing)")

# --- Example 11.1: square prism, L = 0.16 m, dB/dt = 1000 T/s --------------------------------------
L, Bdot = 0.16, 1000.0
scale = L * Bdot
close(scale, 160.0, name="field scale L dB/dt (V/m)")
close(0.5 * 0.08 * Bdot, 40.0, name="primary at edge midpoint (V/m), r = 8 cm")
close(0.5 * 0.08 * np.sqrt(2) * Bdot, 56.6, rtol=1e-3, name="primary at corner (V/m)")
close(np.linalg.norm(primary(np.array([[0.5, 1.0]])))[()] * scale, 40.0, name="primary at edge midpoint, model")
mid = {}
for n in (128, 256):
    mh = model(n, inclusion=False)
    sh = tms_solve(mh)
    pts = np.array([[0.0005, 0.5], [0.5, 0.9995], [0.9995, 0.5], [0.5, 0.0005], [0.001, 0.001], [0.999, 0.999]])
    vals = np.linalg.norm(sample(mh, sh["E"], pts), axis=1)
    mid[n] = vals[:4].max()
    corner = vals[4:].max()
    print(f"      n={n}: edge-mid |E| = {vals[:4].round(4)}, corner |E| = {vals[4:].round(4)}")
    assert corner < 0.035 * (128 / n), "corner field must vanish under refinement"
close(mid[256], 0.338, rtol=0.003, name="normalised total field at edge midpoint (n=256)")
close(mid[256] * scale, 54.0, rtol=0.005, name="total field at edge midpoint (V/m)")
close(mid[256] * scale / 40.0 - 1.0, 0.35, rtol=0.03, name="increase over the primary field")
# loop average of the tangential component equals the primary one (0.25), O(h) sampling error
for n, ref in ((128, 0.2486), (256, 0.2493)):
    mm = model(n, inclusion=False)          # Example 11.1 is the homogeneous prism
    ss = tms_solve(mm)
    xs = (np.arange(4000) + 0.5) / 4000
    eps = 1e-4
    tot = 0.0
    for pts, t in ((np.c_[xs, np.full_like(xs, 1 - eps)], (1, 0)), (np.c_[np.full_like(xs, 1 - eps), xs], (0, -1)),
                   (np.c_[xs, np.full_like(xs, eps)], (-1, 0)), (np.c_[np.full_like(xs, eps), xs], (0, 1))):
        tot += (sample(mm, ss["E"], pts) @ np.array(t)).mean()
    close(tot / 4, ref, rtol=5e-4, name=f"perimeter-average tangential E (n={n})")
close(0.25 * scale, 40.0, name="exact perimeter average (V/m)")

# --- inclusion: E weaker, J stronger (Sec. 11.3) ------------------------------------------------------
mi, mh = model(256, True), model(256, False)
si, sh = tms_solve(mi), tms_solve(mh)
inc = mi["inc"]
a = mi["area"][inc]
Ei = np.sum(a * np.linalg.norm(si["E"][inc], axis=1)) / a.sum()
Eh = np.sum(a * np.linalg.norm(sh["E"][inc], axis=1)) / a.sum()
Ji = np.sum(a * np.linalg.norm(si["J"][inc], axis=1)) / a.sum()
Jh = np.sum(a * np.linalg.norm(sh["J"][inc], axis=1)) / a.sum()
close(Eh, 0.080, rtol=0.01, name="mean |E| over inclusion region, homogeneous")
close(Ei, 0.054, rtol=0.01, name="mean |E| over inclusion, with inclusion")
close(Ei / Eh, 2 / 3, rtol=0.02, name="ratio of mean |E| (about two thirds)")
close(Ji / Jh, 6.7, rtol=0.01, name="ratio of mean |J|")
d = profile_data()
x = d["x"]
k_left = np.argmin(np.abs(x - 0.455))     # just outside / inside the left rim along y = 0.56
k_in = np.argmin(np.abs(x - 0.475))
print(f"      profile at the left rim: |E| {d['inc_E'][k_left]:.3f} -> {d['inc_E'][k_in]:.3f}, "
      f"|J| {d['inc_J'][k_left]:.3f} -> {d['inc_J'][k_in]:.3f}")
close(d["inc_J"][k_in] / d["inc_J"][k_left], 10 * d["inc_E"][k_in] / d["inc_E"][k_left], rtol=1e-6,
      name="J jump = 10 x E ratio at the rim")
assert 7.0 < d["inc_J"][k_in] / d["inc_J"][k_left] < 11.0

# --- projection: Pythagoras, idempotence, power ratio ---------------------------------------------
for n in (64, 128):
    mm = model(n)
    ss = tms_solve(mm)
    w = mm["area"] * mm["sigma"]
    P0 = np.sum(w * np.sum(ss["Ep"] ** 2, 1))
    P = np.sum(w * np.sum(ss["E"] ** 2, 1))
    Pc = np.sum(w * np.sum(ss["corr"] ** 2, 1))
    cross = np.sum(w * np.sum(ss["E"] * ss["corr"], 1))
    close(P + Pc, P0, rtol=1e-12, name=f"Pythagoras ||Ep||^2 = ||E||^2 + ||grad phi||^2 (n={n})")
    assert abs(cross) < 1e-14
    close(P / P0, 0.80, rtol=0.01, name=f"power ratio with/without correction (n={n})")
    # idempotence: correcting the corrected field changes nothing
    b2 = tms_rhs(mm, ss["E"])
    assert np.abs(b2).max() < 1e-13, "projection must be idempotent (weak residual of E vanishes)"
print("  ok  projection is idempotent: weak residual of the total field ~ 0")

# --- same K for TES and TMS; scaling with a uniform conductivity factor --------------------------
mm = model(64)
K1 = mm["K"].toarray()
tes = tes_uniform(mm)
tms = tms_solve(mm)
assert np.allclose(K1, stiffness(mm["p"], mm["cells"], mm["sigma"])[0].toarray())
for fac in (2.0,):
    m2 = model(64)
    m2["sigma"] = m2["sigma"] * fac
    m2["K"], _, _ = stiffness(m2["p"], m2["cells"], m2["sigma"])
    t2, e2 = tms_solve(m2), tes_uniform(m2)
    assert np.allclose(t2["E"], tms["E"], atol=1e-12)          # TMS: E unchanged
    assert np.allclose(t2["J"], fac * tms["J"], atol=1e-12)    # TMS: J scales
    assert np.allclose(e2["E"], tms["E"] * 0 + tes["E"] / fac, atol=1e-12)   # TES (fixed I): E ~ 1/s
print("  ok  uniform sigma x2: TMS E unchanged and J doubled; TES (fixed I) E halved")
# TMS right-hand side vanishes at interior nodes of homogeneous regions
b = tms["b"]
bnd = np.any((mm["p"] <= 1e-12) | (mm["p"] >= 1 - 1e-12), axis=1)
node_on_jump = np.zeros(len(mm["p"]), bool)
for e in np.flatnonzero(mm["inc"]):
    node_on_jump[mm["cells"][e]] = True
outside_nodes = np.zeros(len(mm["p"]), bool)
for e in np.flatnonzero(~mm["inc"]):
    outside_nodes[mm["cells"][e]] = True
interface = node_on_jump & outside_nodes
assert np.abs(b[~bnd & ~interface]).max() < 1e-14
print("  ok  TMS right-hand side is zero away from the boundary and the sigma jump")

# --- current stream function agrees with sigma E (two discretisations, used for the current lines) ----
def rel_diff(mm):
    psi = stream_tms(mm)
    J_psi = np.einsum("ei,eid->ed", psi[mm["cells"]], mm["grads"]) @ np.array([[0, -1], [1, 0]])
    J_phi = tms_solve(mm)["J"]
    w = mm["area"]
    return np.sqrt(np.sum(w * np.sum((J_psi - J_phi) ** 2, 1)) / np.sum(w * np.sum(J_phi ** 2, 1)))


r_h = [rel_diff(model(n, inclusion=False)) for n in (64, 128)]
close(r_h[0] / r_h[1], 2.0, rtol=0.02, name="homogeneous: stream-function vs sigma E, O(h) agreement")
r_i = [rel_diff(model(n)) for n in (64, 128, 256)]
assert r_i[0] > r_i[1] > r_i[2], r_i       # slower with the staircase inclusion, but converging
print(f"  ok  with the staircase inclusion the difference still decreases: {np.round(r_i, 3)}")

# --- unit field maps (Chapter 4 solenoid: n_t = 1000 /m, r = 1 cm) ----------------------------------
n_t, r = 1000.0, 0.01
b_per_A = MU0 * n_t
e_per_As = MU0 * n_t * r / 2
close(b_per_A * 1e3, 1.26, rtol=0.003, name="b = mu0 n_t (mT/A)")
close(e_per_As, 6.28e-6, rtol=0.001, name="e per A/s ((V/m)/(A/s))")
close(e_per_As * 1e6, 6.28, rtol=0.001, name="e per A/us ((V/m)/(A/us))")
close(5.0 * b_per_A * 1e3, 6.28, rtol=0.001, name="5 A -> B (mT), Chapter 4")
close(1e6 * e_per_As, 6.28, rtol=0.001, name="1e6 A/s -> E (V/m), Chapter 4")
# self-check 7
close(1.2 * 80, 96.0, name="1.2 (V/m)/(A/us) x 80 A/us")
close(1.2 * 80e6, 9.6e7, name="the same map misread as per A/s")

# --- pulse: old-edition synthetic damped sine ---------------------------------------------------------
assert PULSE_TAU == 200e-6 and PULSE_F == 5000.0
t = np.linspace(0, 600e-6, 240001)
I, dI = pulse_waveform(t)
k = np.argmax(I)
assert abs(dI[k]) / np.abs(dI).max() < 1e-3                       # dI/dt = 0 at the current peak
assert np.argmax(np.abs(dI)) == 0                                  # largest |dI/dt| at t = 0
print("  ok  pulse: dI/dt = 0 at the current peak, |dI/dt| largest at t = 0")
print("all Chapter 11 checks passed")
