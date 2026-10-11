"""Numbers used in Chapter 10 (zh/ch10_tes.tex) and its figures tes_setup, electrode_models.

The teaching model is the unit square of numerics.square_mesh with the conductivity inclusion of
numerics.induced_field_example (sigma = 10 inside a circle of radius 0.16 at (0.62, 0.56), 1 outside)
and two electrodes of width w = 0.25 on the top side.  Everything is normalised: lengths by the side,
conductivity by the background value, current by I.  Run:  python3 checks/check_ch10.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "figsrc")]

from numerics import induced_field_example, stiffness  # noqa: E402
from fig_ch10 import (ANODE, CATHODE, CEM_Z, model, tes_cem, tes_dirichlet, tes_uniform,  # noqa: E402
                      uniform_load, electrode_profiles)
from fig_ch11 import sample, tms_solve  # noqa: E402


def close(x, ref, rtol=5e-3, name=""):
    assert np.isclose(x, ref, rtol=rtol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:58s} {x:.6g}")
    return x


W = ANODE[1] - ANODE[0]
assert np.isclose(W, 0.25) and np.isclose(CATHODE[1] - CATHODE[0], 0.25)

# --- the model's inclusion is exactly the one of numerics.induced_field_example ----------------------
m64 = model(64)
p, cells, phi, E, sigma, res, ortho = induced_field_example(64)
assert np.array_equal(m64["sigma"], sigma), "inclusion labels differ from numerics.py"

# --- current compatibility (Sec. 10.3): +2, -0.5 -> -1.5 mA; self-check: +1, +1, -0.5 -> -1.5 --------
close(-(2.0 - 0.5), -1.5, name="third electrode current (mA)")
close(-(1.0 + 1.0 - 0.5), -1.5, name="fourth electrode current, self-check (mA)")

# --- Example 10.1: injection only, one grounded node absorbs all the current ------------------------
m = model(128)
b = uniform_load(m, ANODE, 1.0 / W)
close(b.sum(), 1.0, rtol=1e-12, name="net injected current")
N = len(m["p"])
ref = 0                                                     # lower-left corner node (x, y) = (0, 0)
assert np.allclose(m["p"][ref], [0.0, 0.0])
free = np.setdiff1d(np.arange(N), [ref])
ph = np.zeros(N)
ph[free] = spsolve(m["K"][free][:, free].tocsc(), b[free])
reaction = m["K"] @ ph - b
close(reaction[ref], -1.0, rtol=1e-9, name="reaction at the grounded node / I")
assert np.abs(np.delete(reaction, ref)).max() < 1e-10

# --- electrode models (Fig. electrode_models, Sec. 10.4) -------------------------------------------
peaks = {}
for n in (64, 128, 256):
    d = tes_dirichlet(model(n))
    close(-d["I_cathode"], 1.0, rtol=1e-8, name=f"Dirichlet current balance n={n}")
    j = d["j_anode"] * W
    peaks[n] = (j[0], j[-1])
close(peaks[64][1], 6.9, rtol=0.01, name="equipotential right-edge peak, n=64")
close(peaks[128][1], 9.8, rtol=0.01, name="equipotential right-edge peak, n=128")
close(peaks[256][1], 14.0, rtol=0.01, name="equipotential right-edge peak, n=256")
close(peaks[64][0], 3.0, rtol=0.02, name="equipotential left-edge peak, n=64")
close(peaks[256][0], 6.0, rtol=0.01, name="equipotential left-edge peak, n=256")
# growth ~ h^(-1/2): factor ~ 2 when h shrinks fourfold
close(peaks[256][1] / peaks[64][1], 2.0, rtol=0.02, name="peak ratio n=256/n=64 (expect ~2)")
close(peaks[128][1] / peaks[64][1], np.sqrt(2), rtol=0.02, name="peak ratio n=128/n=64 (expect ~sqrt 2)")

cem_peak = {}
for n in (128, 256):
    mm = model(n)
    for z in CEM_Z:
        r = tes_cem(mm, z)
        close(r["I_anode"], 1.0, rtol=1e-9, name=f"CEM anode current n={n} z={z}")
        close(-r["I_cathode"], 1.0, rtol=1e-6, name=f"CEM cathode current n={n} z={z}")
        jj = r["j_anode"] * W
        cem_peak[(n, z)] = (jj.min(), jj.max(), r["U"])
close(CEM_Z[0] / W, 0.08, rtol=1e-12, name="z sigma / w, first CEM curve")
close(CEM_Z[1] / W, 0.8, rtol=1e-12, name="z sigma / w, second CEM curve")
close(cem_peak[(128, 0.02)][1], 2.9, rtol=0.03, name="CEM peak (z sigma/w = 0.08), n=128")
close(cem_peak[(256, 0.02)][1], cem_peak[(128, 0.02)][1], rtol=0.01, name="CEM peak mesh independent")
close(cem_peak[(128, 0.2)][0], 0.92, rtol=0.01, name="CEM min (z sigma/w = 0.8)")
close(cem_peak[(128, 0.2)][1], 1.32, rtol=0.01, name="CEM max (z sigma/w = 0.8)")

# electrode voltages (units I/sigma) and the voltage-control consequence
d128 = tes_dirichlet(model(128))
close(d128["U"], 1.34, rtol=0.005, name="equipotential electrode voltage")
close(cem_peak[(128, 0.02)][2], 1.56, rtol=0.005, name="CEM voltage, z sigma/w = 0.08")
close(cem_peak[(128, 0.2)][2], 3.05, rtol=0.005, name="CEM voltage, z sigma/w = 0.8")
close(d128["U"] / cem_peak[(128, 0.2)][2], 0.44, rtol=0.01, name="current under fixed voltage 1.34")

# body dissipation for the same I (equipotential < uniform: free distribution minimises loss)
m = model(128)
u = tes_uniform(m)
P_uni = np.sum(m["area"] * m["sigma"] * np.sum(u["E"] ** 2, axis=1))
P_dir = np.sum(m["area"] * m["sigma"] * np.sum(d128["E"] ** 2, axis=1))
close(P_dir, d128["U"] * 1.0, rtol=1e-6, name="P_dir = U I (energy balance)")
close(P_dir, 1.34, rtol=0.005, name="body power, equipotential electrodes")
close(P_uni, 1.47, rtol=0.005, name="body power, uniform electrodes")
assert P_dir < P_uni

# field away from the electrodes, point (0.5, 0.4)
pt = np.array([[0.5, 0.4]])
vals = {}
for name, s in (("dir", d128), ("cem08", tes_cem(m, 0.02)), ("cem8", tes_cem(m, 0.2)), ("uni", u)):
    vals[name] = float(np.linalg.norm(sample(m, s["E"], pt), axis=1)[0])
close(vals["dir"], 0.41, rtol=0.01, name="|E|(0.5,0.4) equipotential")
close(vals["cem08"], 0.42, rtol=0.01, name="|E|(0.5,0.4) CEM 0.08")
close(vals["cem8"], 0.44, rtol=0.01, name="|E|(0.5,0.4) CEM 0.8")
close(vals["uni"], 0.45, rtol=0.01, name="|E|(0.5,0.4) uniform")
close(vals["uni"] / vals["dir"] - 1, 0.10, rtol=0.1, name="relative difference uniform vs equipotential")

# electrode_models figure data are the same as above
prof = electrode_profiles()
close(prof["dir256"][1][-1], peaks[256][1], rtol=1e-12, name="figure data = checked data")

# --- scaling with a uniform change of conductivity (compare box, self-check 3) ----------------------
for fac in (1.0, 2.0):
    mm = model(64)
    mm["sigma"] = mm["sigma"] * fac
    mm["K"], _, _ = stiffness(mm["p"], mm["cells"], mm["sigma"])
    tes = tes_uniform(mm)
    if fac == 1.0:
        E1, J1 = tes["E"], tes["J"]
    else:
        assert np.allclose(tes["E"], E1 / 2, atol=1e-12), "fixed current: E must halve"
        assert np.allclose(tes["J"], J1, atol=1e-12), "fixed current: J must not change"
print("  ok  uniform sigma x2 at fixed current: E halves, J unchanged")

# --- Example 10.2: superposition --------------------------------------------------------------------
e1, e2 = np.array([3.0, 0.0]), np.array([-3.0, 4.0])
close(np.linalg.norm(e1), 3.0, name="|e1|")
close(np.linalg.norm(e2), 5.0, name="|e2|")
close(np.linalg.norm(e1 + e2), 4.0, name="|e1 + e2|")
close(np.linalg.norm(e1 - e2), 7.2, rtol=0.01, name="|e1 - e2|")
close((e1 + e2) @ [0, 1], 4.0, name="(e1+e2).t")
# linearity of the TES solution itself (same mesh): two montages superpose exactly
mm = model(64)
mid = (0.4375, 0.5625)                    # a third electrode patch between the two
from fig_ch10 import solve_neumann, grad  # noqa: E402
b1 = uniform_load(mm, ANODE, 1 / W) + uniform_load(mm, CATHODE, -1 / W)
b2 = uniform_load(mm, mid, 1 / 0.125) + uniform_load(mm, CATHODE, -1 / W)
f1, f2, f12 = (-grad(mm, solve_neumann(mm, bb)) for bb in (b1, b2, b1 + b2))
assert np.allclose(f12, f1 + f2, atol=1e-10)
print("  ok  two montages superpose exactly on the FEM mesh")
print("all Chapter 10 checks passed")
