"""Chapter 24 (verification, uncertainty and defensible decisions) and appendix B: every number quoted in
zh/ch24_practice.tex and in the constants table of zh/appendices.tex.

    python3 checks/check_ch24.py

Models are reused, not re-implemented:
  * the conducting square with the sigma = 10 inclusion of Chapters 10-11 (fig_ch10.model, fig_ch10.tes_*,
    fig_ch11.tms_solve; numerics.py P1 finite elements).  The observable of the error-budget example is the
    area-weighted mean |E| over the inclusion ("target tissue"), for TES (uniform-density electrodes, unit
    current) and for TMS (uniform dB/dt = 1 about the centre).  The perturbations (+-20 % inclusion
    conductivity, +-0.01 inclusion radius, solver tolerances) are teaching inputs, not tissue data;
  * the two-layer conductor meshed with equal elements of Chapter 16 (check_ch16.thin_layer_mesh);
  * the frozen-secant / tangent ratio of Chapter 17 (check_ch17) and the network / field numbers of
    Chapters 14-15 are quoted from their own check scripts, which assert them.
figsrc/fig_ch24.py imports budget() from here, so text, figure and assertions share one implementation.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import cg, spsolve

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "figsrc", ROOT / "checks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from numerics import EPS0, MU0, square_mesh, stiffness  # noqa: E402
from fig_ch10 import ANODE, CATHODE, INC_C, INC_R, SIG_IN, tes_cem, tes_dirichlet, tes_uniform, uniform_load  # noqa: E402,F401
from fig_ch11 import primary, tms_rhs, tms_solve  # noqa: E402


def close(x, ref, rtol=5e-3, atol=0.0, name=""):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:64s} {x:.6g}")
    return x


# ---------------------------------------------------------------------------
# 24.5  Error budget on the running example (inclusion = target tissue)
# ---------------------------------------------------------------------------
N_REF = 128
SIG_PERT = (8.0, 12.0)          # inclusion conductivity +-20 % (teaching input)
R_PERT = (0.15, 0.17)           # inclusion radius +-0.01 (teaching input; h = 1/128 = 0.0078)
CG_TOLS = (1e-3, 1e-6)          # relative-residual tolerances of an iterative solver


def mk(n: int, sig_in: float = SIG_IN, R: float = INC_R):
    """Same mesh, labels and stiffness as fig_ch10.model, with the inclusion radius as a parameter."""
    p, cells = square_mesh(n)
    centre = p[cells].mean(axis=1)
    inc = np.sum((centre - INC_C) ** 2, axis=1) < R**2
    sigma = np.where(inc, sig_in, 1.0)
    K, area, grads = stiffness(p, cells, sigma)
    return dict(n=n, p=p, cells=cells, sigma=sigma, inc=inc, K=K, area=area, grads=grads, centre=centre)


def y_mean(m, E):
    """Area-weighted mean |E| over the inclusion elements."""
    a = m["area"][m["inc"]]
    return float(np.sum(a * np.linalg.norm(E[m["inc"]], axis=1)) / a.sum())


def y_tes(m):
    return y_mean(m, tes_uniform(m)["E"])


def y_tms(m):
    return y_mean(m, tms_solve(m)["E"])


def y_cg(m, rtol, kind="tes"):
    """Observable when the reference-fixed system is solved by CG to a relative residual rtol."""
    N = len(m["p"])
    if kind == "tes":
        W = ANODE[1] - ANODE[0]
        b = uniform_load(m, ANODE, 1 / W) + uniform_load(m, CATHODE, -1 / W)
        Ep = 0.0
    else:
        Ep = primary(m["centre"])
        b = tms_rhs(m, Ep)
    free = np.arange(1, N)
    Kf = m["K"][free][:, free].tocsr()
    phi = np.zeros(N)
    phi[free], info = cg(Kf, b[free], rtol=rtol, maxiter=200000)
    assert info == 0
    res = np.linalg.norm(Kf @ phi[free] - b[free]) / np.linalg.norm(b[free])
    E = Ep - np.einsum("ei,eid->ed", phi[m["cells"]], m["grads"])
    return y_mean(m, E), res


@lru_cache(maxsize=None)
def budget():
    """Relative changes of the observable for each perturbation (dictionary of floats)."""
    out = {}
    for name, fun in (("tes", y_tes), ("tms", y_tms)):
        ys = {n: fun(mk(n)) for n in (64, 128, 256)}
        out[f"{name}_y"] = ys
        out[f"{name}_mesh_64_128"] = abs(ys[128] - ys[64]) / abs(ys[128])
        out[f"{name}_mesh_128_256"] = abs(ys[256] - ys[128]) / abs(ys[256])
        y0 = ys[N_REF]
        out[f"{name}_sig"] = tuple(fun(mk(N_REF, sig_in=s)) / y0 - 1 for s in SIG_PERT)
        out[f"{name}_R"] = tuple(fun(mk(N_REF, R=r)) / y0 - 1 for r in R_PERT)
    m = mk(N_REF)
    y0 = out["tes_y"][N_REF]
    out["tes_electrode"] = {
        "equipotential": y_mean(m, tes_dirichlet(m)["E"]) / y0 - 1,
        "cem_0.08": y_mean(m, tes_cem(m, 0.02)["E"]) / y0 - 1,
        "cem_0.8": y_mean(m, tes_cem(m, 0.2)["E"]) / y0 - 1,
    }
    W = ANODE[1] - ANODE[0]
    b = uniform_load(m, ANODE, 1 / W) + uniform_load(m, CATHODE, -1 / W)
    free = np.arange(1, len(m["p"]))
    Kf = m["K"][free][:, free].tocsc()
    phi = spsolve(Kf, b[free])
    out["direct_residual"] = float(np.linalg.norm(Kf @ phi - b[free]) / np.linalg.norm(b[free]))
    out["cg"] = {t: y_cg(m, t, "tes") for t in CG_TOLS}
    out["cg_rel"] = {t: abs(v[0] / y0 - 1) for t, v in out["cg"].items()}
    out["tms_cg_rel"] = {t: abs(y_cg(m, t, "tms")[0] / out["tms_y"][N_REF] - 1) for t in CG_TOLS}
    return out


def run_budget():
    print("24.5 error budget: mean |E| over the inclusion, n = 128 reference")
    b = budget()
    t, s = b["tes_y"], b["tms_y"]
    close(t[64], 0.1335, rtol=1e-3, name="TES y, n = 64")
    close(t[128], 0.1301, rtol=1e-3, name="TES y, n = 128")
    close(t[256], 0.1291, rtol=1e-3, name="TES y, n = 256")
    close(b["tes_mesh_64_128"], 0.026, rtol=0.03, name="TES adjacent-mesh change 64 -> 128")
    close(b["tes_mesh_128_256"], 0.0080, rtol=0.03, name="TES adjacent-mesh change 128 -> 256")
    close(b["tes_sig"][0], 0.225, rtol=0.01, name="TES: sigma_inc 10 -> 8")
    close(b["tes_sig"][1], -0.155, rtol=0.01, name="TES: sigma_inc 10 -> 12")
    close(b["tes_R"][0], 0.014, rtol=0.05, name="TES: radius 0.16 -> 0.15")
    close(b["tes_R"][1], -0.0039, rtol=0.05, name="TES: radius 0.16 -> 0.17")
    e = b["tes_electrode"]
    close(e["equipotential"], -0.085, rtol=0.01, name="TES: equipotential electrodes vs uniform")
    close(e["cem_0.08"], -0.052, rtol=0.01, name="TES: CEM z sigma/w = 0.08 vs uniform")
    close(e["cem_0.8"], -0.0145, rtol=0.01, name="TES: CEM z sigma/w = 0.8 vs uniform")
    close(s[64], 0.05362, rtol=1e-3, name="TMS y, n = 64")
    close(s[128], 0.05396, rtol=1e-3, name="TMS y, n = 128")
    close(s[256], 0.05402, rtol=1e-3, name="TMS y, n = 256")
    close(b["tms_mesh_64_128"], 0.0062, rtol=0.03, name="TMS adjacent-mesh change 64 -> 128")
    close(b["tms_mesh_128_256"], 0.0012, rtol=0.05, name="TMS adjacent-mesh change 128 -> 256")
    close(b["tms_sig"][0], 0.0078, rtol=0.02, name="TMS: sigma_inc 10 -> 8")
    close(b["tms_sig"][1], -0.0045, rtol=0.02, name="TMS: sigma_inc 10 -> 12")
    close(b["tms_R"][0], -0.061, rtol=0.02, name="TMS: radius 0.16 -> 0.15")
    close(b["tms_R"][1], 0.061, rtol=0.02, name="TMS: radius 0.16 -> 0.17")
    # TES is dominated by the material, TMS by the segmentation of the target region
    assert abs(b["tes_sig"][0]) > 10 * abs(b["tes_R"][0])
    assert abs(b["tms_R"][0]) > 5 * abs(b["tms_sig"][0])
    # sensitivities to the same +-20 % differ by a factor of ~30 between TES and TMS
    ratio = abs(b["tes_sig"][0]) / abs(b["tms_sig"][0])
    close(ratio, 29, rtol=0.05, name="TES / TMS sensitivity to sigma_inc 10 -> 8")
    assert b["direct_residual"] < 1e-12
    close(b["direct_residual"], 1.7e-13, rtol=0.2, name="direct solve: relative residual")
    close(b["cg_rel"][1e-3], 2.4e-4, rtol=0.05, name="CG rtol 1e-3: change of y")
    print(f"      TMS, CG rtol 1e-3: change of y {b['tms_cg_rel'][1e-3]:.1e}")
    assert b["tms_cg_rel"][1e-3] < 1e-3 and b["tms_cg_rel"][1e-6] < 1e-6
    assert b["cg_rel"][1e-6] < 1e-6
    print(f"  ok  CG rtol 1e-6: change of y {b['cg_rel'][1e-6]:.1e} (< 1e-6)")
    # in this example the discretization error at n = 128 is far below the material uncertainty
    assert b["tes_mesh_128_256"] < 0.05 * abs(b["tes_sig"][0])
    # exact scalings with a common conductivity factor (Ch 10 / Ch 11): TES E ~ 1/s, TMS E unchanged
    for fac in (2.0,):
        m1, m2 = mk(64), mk(64)
        m2["sigma"] = m2["sigma"] * fac
        m2["K"], _, _ = stiffness(m2["p"], m2["cells"], m2["sigma"])
        assert np.isclose(y_tes(m2), y_tes(m1) / fac, rtol=1e-10)
        assert np.isclose(y_tms(m2), y_tms(m1), rtol=1e-10)
    print("  ok  all conductivities x2: TES y halves, TMS y unchanged (exact)")
    run_tms_split()


def run_tms_split():
    """Why the TMS observable ignores sigma_inc but follows the radius (text of ex:practice-budget).

    Relative to the inclusion centre c, the primary field of a uniform dB_z/dt splits exactly into a part that
    circles c (tangent to the inclusion boundary, so it leaves no charge there and does not depend on sigma_inc)
    and a uniform part U = E_p(c), which crosses the boundary and is screened by the conducting inclusion."""
    m = mk(N_REF)
    inc, a = m["inc"], m["area"][m["inc"]]
    U = primary(np.array(INC_C)[None, :])[0]
    circ = primary(m["centre"][inc]) - U
    mean_circ = float(np.sum(a * np.linalg.norm(circ, axis=1)) / a.sum())
    close(mean_circ, INC_R / 3, rtol=0.01, name="TMS: mean |circling part| over the inclusion = R/3 (dB/dt = 1)")
    rem = {}
    for s in (SIG_PERT[0], SIG_IN, SIG_PERT[1]):
        ms = mk(N_REF, sig_in=s)
        E = tms_solve(ms)["E"][inc]
        rem[s] = float(np.sum(a * np.linalg.norm(E - circ, axis=1)) / a.sum()) / np.linalg.norm(U)
    assert rem[SIG_IN] < 0.2                                   # "不足五分之一"
    close(rem[SIG_IN], 0.173, rtol=0.02, name="TMS: uniform remainder inside / |U| (sigma_inc = 10)")
    # the remainder itself follows sigma_inc by ~ +-20 %, yet y moves by < 1 % (it adds to a rotating field)
    assert rem[SIG_PERT[0]] / rem[SIG_IN] - 1 > 0.15 and 1 - rem[SIG_PERT[1]] / rem[SIG_IN] > 0.1
    b = budget()
    assert max(abs(v) for v in b["tms_sig"]) < 0.01
    print(f"  ok  remainder changes {rem[SIG_PERT[0]] / rem[SIG_IN] - 1:+.1%} / {rem[SIG_PERT[1]] / rem[SIG_IN] - 1:+.1%}, "
          f"y only {b['tms_sig'][0]:+.2%} / {b['tms_sig'][1]:+.2%}")


# ---------------------------------------------------------------------------
# 24.5  Two meshes that agree can both be wrong (Ch 16 two-layer conductor, equal elements)
# ---------------------------------------------------------------------------
def run_thin_layer():
    import check_ch16 as C16
    print("24.5 two-layer conductor with equal elements (Ch 16)")
    R = {k: C16.thin_layer_mesh(k)["R"] for k in range(1, 11)}
    for k, ref in ((1, 500.0), (2, 500.0), (3, 2000.0), (4, 1625.0), (5, 1400.0), (6, 1250.0), (7, 1142.9),
                   (10, 1400.0)):
        close(R[k], ref, rtol=1e-4, name=f"R with {k} equal elements (ohm)")
    close(1 - R[1] / 1400.0, 0.643, rtol=1e-3, name="1 and 2 elements agree, both low by")
    assert all(C16.thin_layer_mesh(k)["residual"] < 1e-12 for k in range(1, 11))
    print("  ok  every mesh is solved to round-off: the residual cannot see the lost layer")


# ---------------------------------------------------------------------------
# Numbers quoted from other chapters (asserted again where cheap)
# ---------------------------------------------------------------------------
def run_quoted():
    print("numbers quoted from other chapters")
    import check_ch17 as C17
    Rm, Rg, _ = C17.pm_parts()
    Phi = C17.newton_solve()
    Rs, Rd = C17.yoke_reluctances(Phi)
    close((1 / (Rm + Rg + Rs)) / (1 / (Rm + Rg + Rd)), 12.1, rtol=5e-3, name="Ch 17: frozen secant / tangent")
    # Ch 18: SAR peak vs RMS (old edition teaching values)
    close(0.5 * 0.5 * 100**2 / 1000, 2.5, rtol=1e-12, name="SAR, 100 V/m peak (W/kg)")
    close(0.5 * 100**2 / 1000, 5.0, rtol=1e-12, name="SAR, 100 V/m RMS (W/kg)")
    # Ch 16: manufactured solution, measured orders and the gap between residual and error
    import numerics
    rows = [numerics.manufactured_solution(n)[0] for n in (32, 64)]
    p_l2 = np.log2(rows[0]["L2_error"] / rows[1]["L2_error"])
    p_h1 = np.log2(rows[0]["H1_seminorm_error"] / rows[1]["H1_seminorm_error"])
    close(p_l2, 1.998, rtol=1e-3, name="Ch 16: L2 order (32 -> 64)")
    close(p_h1, 0.999, rtol=1e-3, name="Ch 16: H1 order (32 -> 64)")
    close(rows[1]["H1_seminorm_error"] / (np.pi / np.sqrt(2)), 0.0245, rtol=5e-3, name="Ch 16: gradient error n = 64")
    # Ch 10: equipotential-electrode edge peak grows under refinement (asserted in check_ch10.py)
    peaks = (6.9, 9.8, 14.0)
    close(peaks[2] / peaks[0], 2.0, rtol=0.02, name="Ch 10: edge peak n = 256 / n = 64")
    # Ch 11: unit map conversion 1 A/us = 1e6 A/s; old extended check 2e-5 (V/m)/(A/s) at 1 A/us
    close(2e-5 * 1e6, 20.0, rtol=1e-12, name="2e-5 (V/m)/(A/s) x 1 A/us (V/m)")


# ---------------------------------------------------------------------------
# Appendix B: constants and conversions
# ---------------------------------------------------------------------------
MU0_CODATA2018 = 1.25663706212e-6          # N/A^2, CODATA 2018 recommended value
GAMMA_BAR = 42.577478518e6                 # Hz/T, CODATA 2018 proton (as in check_ch19.py)


def run_appendix():
    print("appendix B: constants and conversions")
    rel = abs(MU0_CODATA2018 - MU0) / MU0
    assert rel < 1e-9
    print(f"  ok  |mu0(CODATA 2018) - 4 pi 1e-7| / mu0 = {rel:.1e} (< 1e-9)")
    close(EPS0, 8.854e-12, rtol=1e-4, name="eps0 (F/m)")
    close(1 / np.sqrt(MU0 * EPS0), 2.998e8, rtol=1e-4, name="c = 1/sqrt(mu0 eps0) (m/s)")
    close(GAMMA_BAR / 1e6, 42.577, rtol=2e-5, name="gamma bar (MHz/T)")
    close(GAMMA_BAR * 0.1 / 1e6, 4.258, rtol=1e-3, name="Larmor frequency at 0.1 T (MHz)")
    close(GAMMA_BAR * 0.1 * 1e-6, 4.258, rtol=1e-3, name="1 ppm of 0.1 T (Hz)")
    close(1000 / (4 * np.pi), 79.58, rtol=1e-4, name="1 Oe in A/m")
    close(1.0**2 / (2 * MU0) / 1e3, 398, rtol=1e-3, name="magnetic pressure at 1 T (kPa)")


if __name__ == "__main__":
    run_budget()
    run_thin_layer()
    run_quoted()
    run_appendix()
    print("All Chapter 24 / appendix checks passed.")
