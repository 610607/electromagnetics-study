"""Chapter 14 (lumped magnetic circuits): models and checks for every number in zh/ch14_lumped.tex.

    python3 checks/check_ch14.py          # run all assertions, print the quoted values

The figure module figsrc/fig_ch14.py imports the model functions from this file, so the
text, the figures and the checks share one implementation.  Only NumPy, SciPy and the
book's numerics.py are used (synthetic B-H curve, single-loop magnet example, P1 FEM
assembly).  All geometry and material values are teaching inputs, not device data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

import numerics
from numerics import MU0, stiffness, synthetic_B

BS, HS = 1.6, 800.0          # parameters of eq:syntheticbh (Ch 13), as in numerics.synthetic_B


def mu_diff(H):
    """Differential permeability dB/dH of the synthetic curve (eq:permeabilities)."""
    return MU0 + BS / HS / np.cosh(np.asarray(H) / HS) ** 2


# ---------------------------------------------------------------------------
# Example 14.1  Gapped C-core, linear iron
# ---------------------------------------------------------------------------
CORE = dict(N=200, I=2.0, lc=0.30, A=4.0e-4, mur=2000.0, g=1.0e-3)


def gapped_core(N=CORE["N"], I=CORE["I"], lc=CORE["lc"], A=CORE["A"], mur=CORE["mur"], g=CORE["g"]):
    Rc = lc / (MU0 * mur * A)
    Rg = g / (MU0 * A)
    R = Rc + Rg
    F = N * I
    Phi = F / R
    B = Phi / A
    return dict(Rc=Rc, Rg=Rg, R=R, ratio=Rg / Rc, F=F, Phi=Phi, B=B, Hg=B / MU0, Hc=B / (MU0 * mur),
                Fg=Rg * Phi, Fc=Rc * Phi, frac_gap=Rg / R, L=N**2 / R, W=0.5 * Phi**2 * R,
                W_check=0.5 * (N**2 / R) * I**2, iron_equiv=mur * g, Pg=1 / Rg, Pc=1 / Rc)


# ---------------------------------------------------------------------------
# Example 14.2  Three-limb (E-type) core, nodal analysis; mutual inductance in 14.6
# ---------------------------------------------------------------------------
ECORE = dict(lo=0.18, Ao=4.0e-4, lc=0.08, Ac=8.0e-4, gC=1.0e-3, gR=0.25e-3, mur=2000.0,
             N1=200, I1=1.0, N2=100)


def ecore_reluctances(p=ECORE):
    """Branch reluctances between the bottom node (reference) and the top node."""
    Ri_o = p["lo"] / (MU0 * p["mur"] * p["Ao"])
    Ri_c = p["lc"] / (MU0 * p["mur"] * p["Ac"])
    RgC = p["gC"] / (MU0 * p["Ac"])
    RgR = p["gR"] / (MU0 * p["Ao"])
    return dict(L=Ri_o, C=Ri_c + RgC, R=Ri_o + RgR, Ri_o=Ri_o, Ri_c=Ri_c, RgC=RgC, RgR=RgR)


def ecore_nodal(F, p=ECORE):
    """Node method: one unknown magnetic potential psi of the top node (bottom = 0).

    Branch k carries upward flux Phi_k = P_k (F_k - psi); flux conservation at the top
    node, sum_k Phi_k = 0, gives psi = sum P_k F_k / sum P_k.
    """
    R = ecore_reluctances(p)
    P = {k: 1.0 / R[k] for k in "LCR"}
    psi = sum(P[k] * F.get(k, 0.0) for k in "LCR") / sum(P.values())
    Phi = {k: P[k] * (F.get(k, 0.0) - psi) for k in "LCR"}
    return psi, Phi, R, P


def ecore_mesh(F, p=ECORE):
    """Independent check by loop (mesh) analysis with two loop fluxes."""
    R = ecore_reluctances(p)
    # loop a: up L, down C ; loop b: up C, down R
    M = np.array([[R["L"] + R["C"], -R["C"]], [-R["C"], R["C"] + R["R"]]])
    rhs = np.array([F.get("L", 0) - F.get("C", 0), F.get("C", 0) - F.get("R", 0)])
    a, b = np.linalg.solve(M, rhs)
    return {"L": a, "C": b - a, "R": -b}


def ecore():
    p = ECORE
    F1 = p["N1"] * p["I1"]
    psi, Phi, R, P = ecore_nodal({"L": F1})
    mesh = ecore_mesh({"L": F1})
    B = {"L": Phi["L"] / p["Ao"], "C": Phi["C"] / p["Ac"], "R": Phi["R"] / p["Ao"]}
    # inductances (coil 2 on the right limb, reference direction = downward flux in that limb)
    R_in1 = R["L"] + 1 / (P["C"] + P["R"])
    L11 = p["N1"] ** 2 / R_in1
    L11_direct = p["N1"] * Phi["L"] / p["I1"]
    M21 = p["N2"] * (-Phi["R"]) / p["I1"]
    # coil 2 driven alone: MMF in branch R acting downward = -N2 I2 in the upward convention
    I2 = 1.0
    _, Phi2, _, _ = ecore_nodal({"R": -p["N2"] * I2})
    L22 = p["N2"] * (-Phi2["R"]) / I2
    M12 = p["N1"] * Phi2["L"] / I2
    R_in2 = R["R"] + 1 / (P["L"] + P["C"])
    split_R = P["R"] / (P["C"] + P["R"])
    k = M21 / np.sqrt(L11 * L22)
    return dict(psi=psi, Phi=Phi, mesh=mesh, B=B, R=R, P=P, R_in1=R_in1, L11=L11, L11_direct=L11_direct,
                M21=M21, M12=M12, L22=L22, L22_formula=p["N2"] ** 2 / R_in2, k=k, split_R=split_R,
                F1=F1, Psum=sum(P[x] for x in "LCR"))


# ---------------------------------------------------------------------------
# Example 14.3  Electromagnet with armature: inductance, energy, force
# ---------------------------------------------------------------------------
ARM = dict(N=300, I=2.0, A=4.0e-4, li=0.20, mur=2000.0, x=1.0e-3)


def armature(x=ARM["x"], N=ARM["N"], I=ARM["I"], A=ARM["A"], li=ARM["li"], mur=ARM["mur"]):
    Ri = li / (MU0 * mur * A)
    R = Ri + 2 * x / (MU0 * A)
    dRdx = 2 / (MU0 * A)
    F = N * I
    Phi = F / R
    B = Phi / A
    L = N**2 / R
    W = 0.5 * Phi**2 * R
    f_flux = -0.5 * Phi**2 * dRdx                 # constant flux (energy)
    f_mmf = 0.5 * F**2 * (-dRdx / R**2)           # constant MMF (co-energy)
    # finite-difference check of both derivatives
    h = 1e-9
    Wf = lambda xx: 0.5 * Phi**2 * (Ri + 2 * xx / (MU0 * A))
    Wc = lambda xx: 0.5 * F**2 / (Ri + 2 * xx / (MU0 * A))
    f_fd_flux = -(Wf(x + h) - Wf(x - h)) / (2 * h)
    f_fd_mmf = (Wc(x + h) - Wc(x - h)) / (2 * h)
    f_pressure = 2 * (B**2 / (2 * MU0)) * A        # two pole faces, Maxwell-stress scale B^2/(2 mu0)
    return dict(Ri=Ri, Rgap=2 * x / (MU0 * A), R=R, Phi=Phi, B=B, L=L, W=W, f_flux=f_flux, f_mmf=f_mmf,
                f_fd_flux=f_fd_flux, f_fd_mmf=f_fd_mmf, f_pressure=f_pressure, pressure=B**2 / (2 * MU0))


# ---------------------------------------------------------------------------
# Example 14.4  Nonlinear core (same C-core, synthetic B-H curve)
# ---------------------------------------------------------------------------
def core_residual(H, NI, l=CORE["lc"], g=CORE["g"], area_ratio=1.0):
    """eq:coreconnection with the iron intensity H as unknown: l H + g (A_y/A_g) B(H)/mu0 - NI."""
    return l * H + g * area_ratio * synthetic_B(H) / MU0 - NI


def core_newton(NI, H0=0.0, n=7, l=CORE["lc"], g=CORE["g"], area_ratio=1.0):
    H = float(H0)
    hist = []
    for k in range(n):
        f = float(core_residual(H, NI, l, g, area_ratio))
        hist.append((k, H, f))
        df = l + g * area_ratio * float(mu_diff(H)) / MU0
        H = H - f / df
    return hist


def core_operating_point(NI, l=CORE["lc"], g=CORE["g"], A=CORE["A"], N=CORE["N"]):
    H = brentq(core_residual, 0.0, 1.0e7, args=(NI, l, g), xtol=1e-12, rtol=1e-15)
    B = float(synthetic_B(H))
    Phi = B * A
    I = NI / N
    Rdiff = l / (float(mu_diff(H)) * A) + g / (MU0 * A)
    return dict(H=H, B=B, Phi=Phi, Firon=l * H, Fgap=g * B / MU0, frac_iron=l * H / NI,
                ideal=MU0 * NI / g, linear=MU0 * NI / (g + l / CORE["mur"]),
                Lsec=N * Phi / I, Ldiff=N**2 / Rdiff, mur_sec=B / (MU0 * H), mur_diff=float(mu_diff(H)) / MU0)


# ---------------------------------------------------------------------------
# Section 14.8  Permanent magnet in the circuit (old single-loop example)
# ---------------------------------------------------------------------------
def pm_single_loop():
    r = numerics.magnetic_circuit()
    Br, mur, lm, g, ly = r["Br_T"], r["mu_rec_relative"], r["magnet_length_m"], r["gap_m"], r["yoke_length_m"]
    Am, Ag, Ay = r["Am_m2"], r["Ag_m2"], r["Ay_m2"]
    Hc = Br / (MU0 * mur)                    # B = 0 intercept of the straight recoil line
    Fc = Hc * lm                             # Thevenin MMF
    Rm = lm / (MU0 * mur * Am)               # internal reluctance
    Phir = Br * Am                           # Norton flux
    Pm = 1 / Rm
    Rg = g / (MU0 * Ag)
    Phi_ideal = Fc / (Rm + Rg)
    Hm_ideal = (Phi_ideal / Am - Br) / (MU0 * mur)
    BHmax = Br**2 / (4 * MU0 * mur)
    Fy = r["Hy_A_per_m"] * ly
    # Ampere balance terms of the nonlinear solution
    Fm_drop = r["Hm_A_per_m"] * lm
    Fg_drop = r["Bg_T"] * g / MU0
    # loop residual of eq:nonlinearmagcircuit evaluated independently
    Phi = r["flux_Wb"]
    resid = lm / (MU0 * mur) * (Phi / Am - Br) + g * Phi / (MU0 * Ag) + ly * r["Hy_A_per_m"]
    # g at which the ideal operating point sits at (BH)max: slope mu0 lm/g = mu0 mu_rec
    g_opt = lm / mur
    Vg = Ag * g
    Vm_formula = Phi_ideal**2 / Ag**2 * Vg / (MU0 * abs(Phi_ideal / Am * Hm_ideal))
    return dict(r=r, Hc=Hc, Fc=Fc, Rm=Rm, Phir=Phir, Pm=Pm, Rg=Rg, Phi_ideal=Phi_ideal,
                Bg_ideal=Phi_ideal / Ag, Hm_ideal=Hm_ideal, BHmax=BHmax, Fy=Fy, frac_yoke=Fy / Fc,
                deficit=1 - r["Bg_T"] / r["ideal_infinite_mu_yoke_Bg_T"], Fm_drop=Fm_drop, Fg_drop=Fg_drop,
                resid=resid, g_opt=g_opt, BH_ideal=abs(Phi_ideal / Am * Hm_ideal), Vm=Am * lm,
                Vm_formula=Vm_formula, Rm_share=Rm / (Rm + Rg), thevenin_check=Fc - Rm * Phi)


def recoil_operating_point(g, Br=1.2, mur=1.05, lm=0.05):
    """Ideal equal-area circuit, yoke drop neglected: intersection of eq:recoil and eq:loadline."""
    Hm = -Br / (MU0 * (mur + lm / g))
    return Hm, Br + MU0 * mur * Hm


# ---------------------------------------------------------------------------
# Example 14.6  Multi-pillar yoke (teaching inputs)
# ---------------------------------------------------------------------------
PILLAR = dict(Br=1.2, mu_rec=1.05, lm=0.04, Am=0.20, g=0.40, lp=0.60)


def multipillar(N, Ap, p=PILLAR):
    """Two magnets (each l_m, A_m) + ideal pole pieces/plates + gap + N identical nonlinear pillars.

    Loop law: 2 H_m l_m + (Phi/A_g) g / mu0 + H_p l_p = 0,  Phi = N A_p B(H_p),  A_g = A_m.
    """
    Br, mur, lm, Am, g, lp = (p[k] for k in ("Br", "mu_rec", "lm", "Am", "g", "lp"))

    def res(Hp):
        Phi = N * Ap * synthetic_B(Hp)
        Hm = (Phi / Am - Br) / (MU0 * mur)
        return 2 * Hm * lm + Phi / Am * g / MU0 + Hp * lp

    Hp = brentq(res, 0.0, 1.0e8, xtol=1e-10, rtol=1e-14)
    Phi = N * Ap * float(synthetic_B(Hp))
    Bg = Phi / Am
    ideal = 2 * Br * lm / (2 * lm + mur * g)
    Fc2 = 2 * Br * lm / (MU0 * mur)
    return dict(Hp=Hp, Bp=float(synthetic_B(Hp)), Phi=Phi, Bg=Bg, ideal=ideal, ratio=Bg / ideal,
                frac_pillars=Hp * lp / Fc2, Fc2=Fc2, resid=float(res(Hp)))


# ---------------------------------------------------------------------------
# Section 14.5 / 14.10  2-D field solution of the gapped C-core (validation study)
# ---------------------------------------------------------------------------
FEM = dict(w=0.02, outer=(0.10, 0.09), window=((0.02, 0.08), (0.02, 0.07)), yc=0.045,
           coil_y=(0.025, 0.065), coil_out=(-0.01, 0.0), coil_in=(0.02, 0.03), mur=2000.0, NI=400.0,
           perimeter=0.30)


def _graded(breaks, hfun, nsub=600):
    pts = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        s = np.linspace(a, b, nsub + 1)
        inv = 1.0 / hfun(s)
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))))
        n = max(1, int(np.ceil(cum[-1])))
        pts.extend(np.interp(np.linspace(0, cum[-1], n + 1)[1:], cum, s))
    return np.array(pts)


def _tensor_mesh(xs, ys):
    nx = len(xs)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    p = np.column_stack((X.ravel(), Y.ravel()))
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(len(ys) - 1), indexing="xy")
    a = (j * nx + i).ravel()
    cells = np.vstack((np.column_stack((a, a + 1, a + nx + 1)), np.column_stack((a, a + nx + 1, a + nx))))
    return p, cells


def c_core_fem(g, refine=1.0, half=0.5, mur=FEM["mur"], NI=FEM["NI"], keep=False):
    """Planar (per unit depth) magnetostatic A_z solution of the gapped C-core.

    -div(nu grad A_z) = J_z, A_z = 0 on a square box of half-width `half` around the core.
    P1 elements on a graded tensor-product mesh whose lines follow every material edge;
    assembly by numerics.stiffness with coefficient nu = 1/mu.  Linear iron.
    """
    w, yc = FEM["w"], FEM["yc"]
    (W, Hh), ((wx0, wx1), (wy0, wy1)) = FEM["outer"], FEM["window"]
    cy0, cy1 = FEM["coil_y"]
    cx, cyy = W / 2, Hh / 2
    xb = sorted({cx - half, FEM["coil_out"][0], 0.0, wx0, FEM["coil_in"][1], wx1, W, cx + half})
    yb = sorted({cyy - half, 0.0, wy0, cy0, yc - g / 2, yc + g / 2, cy1, wy1, Hh, cyy + half})
    h0 = g / 6 / refine

    def hx(x):
        d = np.minimum(np.abs(x - wx1), np.abs(x - W))
        h = np.minimum(h0 + 0.15 * d, 4e-3 / refine)
        return np.where((x > -0.012) & (x < W + 0.002), np.minimum(h, 1e-3 / refine), h)

    def hy(y):
        d = np.minimum(np.abs(y - (yc - g / 2)), np.abs(y - (yc + g / 2)))
        h = np.minimum(h0 + 0.15 * d, 4e-3 / refine)
        return np.where((y > -0.002) & (y < Hh + 0.002), np.minimum(h, 1e-3 / refine), h)

    xs, ys = _graded(xb, hx), _graded(yb, hy)
    p, cells = _tensor_mesh(xs, ys)
    c = p[cells].mean(axis=1)
    x, y = c[:, 0], c[:, 1]
    window = (x > wx0) & (x < wx1) & (y > wy0) & (y < wy1)
    gap = (x > wx1) & (x < W) & (np.abs(y - yc) < g / 2)
    iron = (x > 0) & (x < W) & (y > 0) & (y < Hh) & ~window & ~gap
    nu = np.where(iron, 1 / (MU0 * mur), 1 / MU0)
    s_out = (x > FEM["coil_out"][0]) & (x < FEM["coil_out"][1]) & (y > cy0) & (y < cy1)
    s_in = (x > FEM["coil_in"][0]) & (x < FEM["coil_in"][1]) & (y > cy0) & (y < cy1)
    J = NI / ((FEM["coil_in"][1] - FEM["coil_in"][0]) * (cy1 - cy0))
    Jz = np.where(s_out, J, 0.0) - np.where(s_in, J, 0.0)
    K, area, grads = stiffness(p, cells, nu)
    rhs = np.zeros(len(p))
    np.add.at(rhs, cells.ravel(), np.repeat(Jz * area / 3, 3))
    bnd = (np.isclose(p[:, 0], xs[0]) | np.isclose(p[:, 0], xs[-1]) |
           np.isclose(p[:, 1], ys[0]) | np.isclose(p[:, 1], ys[-1]))
    free = np.flatnonzero(~bnd)
    Az = np.zeros(len(p))
    Az[free] = spsolve(K[free][:, free].tocsc(), rhs[free])
    nx = len(xs)

    def at(xq, yq):
        i = int(np.argmin(np.abs(xs - xq)))
        j = int(np.argmin(np.abs(ys - yq)))
        return Az[j * nx + i]

    ymid = ys[int(np.argmin(np.abs(ys - yc)))]
    phi_limb = at(0.0, ymid) - at(wx0, ymid)          # flux up the coil limb (per metre depth)
    phi_pole = at(wx1, ymid) - at(W, ymid)            # flux across the gap mid-plane, pole width only
    grad = np.einsum("ei,eid->ed", Az[cells], grads)
    Bx, By = grad[:, 1], -grad[:, 0]
    Bmag = np.hypot(Bx, By)
    centre = gap & (np.abs(x - (wx1 + W) / 2) < w / 10)
    Bc = float(np.sum(Bmag[centre] * area[centre]) / area[centre].sum())
    energy = 0.5 * float(np.sum(area * nu * Bmag**2))     # J per metre depth
    lc = FEM["perimeter"] - g
    Rc = lc / (MU0 * mur * w)
    Rg = g / (MU0 * w)
    phi_lumped = NI / (Rc + Rg)
    k_eff = Rg / (NI / phi_limb - Rc)                # calibrated "effective area" factor
    out = dict(g=g, gw=g / w, nodes=len(p), phi_limb=phi_limb, phi_pole=abs(phi_pole), Bc=Bc,
               energy=energy, phi_energy=2 * energy / NI, phi_lumped=phi_lumped, Bg_lumped=phi_lumped / w,
               limb_ratio=phi_limb / phi_lumped, Bc_ratio=Bc / (phi_lumped / w), k_eff=k_eff,
               offpath=(phi_limb - abs(phi_pole)) / phi_limb)
    if keep:
        out.update(p=p, cells=cells, Az=Az, xs=xs, ys=ys, Bmag=Bmag, iron=iron, gap=gap, s_in=s_in, s_out=s_out)
    return out


def predict_with_k(g, k, mur=FEM["mur"], NI=FEM["NI"]):
    """Lumped prediction of the limb flux with an effective gap area factor k (A_eff = k A)."""
    w = FEM["w"]
    lc = FEM["perimeter"] - g
    return NI / (lc / (MU0 * mur * w) + g / (MU0 * k * w))


FEM_GAPS = (0.5e-3, 1.0e-3, 2.0e-3, 4.0e-3, 6.0e-3, 8.0e-3)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def quoted(value, text, name=""):
    """Assert that `value` rounds to the number printed in the chapter as `text`.

    `text` is written as in the chapter: '0.437', '2.98e5', '-41.0', '87.0' ...; the tolerance is
    half a unit in the last printed digit.
    """
    t = text.replace("−", "-")
    mant, _, exp = t.lower().partition("e")
    e = int(exp) if exp else 0
    dec = len(mant.split(".")[1]) if "." in mant else 0
    tol = 0.5 * 10.0 ** (e - dec) * (1 + 1e-9)
    q = float(t)
    assert abs(value - q) <= tol, f"{name}: computed {value!r} does not round to {text}"
    return value


def run_checks(verbose=True):
    log = []

    def Q(name, value, text):
        quoted(value, text, name)
        log.append(f"{name:<46s} {value: .6g}  ~ {text}")

    # --- inputs quoted in the text (bind the prose to the models) ---
    assert CORE == dict(N=200, I=2.0, lc=0.30, A=4.0e-4, mur=2000.0, g=1.0e-3)
    assert ECORE == dict(lo=0.18, Ao=4.0e-4, lc=0.08, Ac=8.0e-4, gC=1.0e-3, gR=0.25e-3, mur=2000.0,
                         N1=200, I1=1.0, N2=100)
    assert ARM == dict(N=300, I=2.0, A=4.0e-4, li=0.20, mur=2000.0, x=1.0e-3)
    assert PILLAR == dict(Br=1.2, mu_rec=1.05, lm=0.04, Am=0.20, g=0.40, lp=0.60)
    assert (BS, HS) == (1.6, 800.0) and float(synthetic_B(800.0)) == float(numerics.synthetic_B(800.0))
    assert FEM["mur"] == 2000.0 and FEM["NI"] == 400.0 and FEM["w"] == 0.02

    # --- 14.2 / Example 14.1: gapped C-core, linear iron ---
    a = gapped_core()
    Q("A: R_Fe [A/Wb]", a["Rc"], "2.98e5")
    Q("A: R_g [A/Wb]", a["Rg"], "1.99e6")
    Q("A: R_g/R_Fe", a["ratio"], "6.67")
    Q("A: iron length equivalent of 1 mm gap [m]", a["iron_equiv"], "2.0")
    Q("A: g/l_Fe = 1/300", CORE["g"] / CORE["lc"] * 300, "1.000")
    Q("A: total R [A/Wb]", a["R"], "2.29e6")
    Q("A: Phi [Wb]", a["Phi"], "1.75e-4")
    Q("A: B [T]", a["B"], "0.437")
    Q("A: MMF on gap [A]", a["Fg"], "348")
    Q("A: MMF on iron [A]", a["Fc"], "52")
    Q("A: gap share [%]", 100 * a["frac_gap"], "87.0")
    Q("A: H_g [A/m]", a["Hg"], "3.48e5")
    Q("A: H_Fe [A/m]", a["Hc"], "174")
    # 14.6 inductance and energy of the same core
    Q("A: L [mH]", 1e3 * a["L"], "17.5")
    Q("A: L with ideal iron N^2/R_g [mH]", 1e3 * CORE["N"] ** 2 / a["Rg"], "20.1")
    Q("A: W [J]", a["W"], "0.035")
    assert abs(a["W"] - a["W_check"]) < 1e-12
    Q("A: energy share in gap [%]", 100 * a["frac_gap"], "87")
    # self-check 1: gap doubled
    a2 = gapped_core(g=2.0e-3)
    Q("A': R_g at 2 mm [A/Wb]", a2["Rg"], "3.98e6")
    Q("A': total R at 2 mm [A/Wb]", a2["R"], "4.28e6")
    Q("A': B at 2 mm [T]", a2["B"], "0.234")

    # --- 14.4 / Example 14.2: three-limb core ---
    e = ecore()
    Q("B: R_L [A/Wb]", e["R"]["L"], "1.79e5")
    Q("B: R_C iron [A/Wb]", e["R"]["Ri_c"], "3.98e4")
    Q("B: R_C gap [A/Wb]", e["R"]["RgC"], "9.95e5")
    Q("B: R_C [A/Wb]", e["R"]["C"], "1.03e6")
    Q("B: R_C in figure [1e5 A/Wb]", e["R"]["C"] / 1e5, "10.3")
    Q("B: R_R gap [A/Wb]", e["R"]["RgR"], "4.97e5")
    Q("B: R_R [A/Wb]", e["R"]["R"], "6.76e5")
    Q("B: P_L [uH]", 1e6 * e["P"]["L"], "5.59")
    Q("B: P_C [uH]", 1e6 * e["P"]["C"], "0.967")
    Q("B: P_R [uH]", 1e6 * e["P"]["R"], "1.48")
    Q("B: psi_T [A]", e["psi"], "139")
    Q("B: Phi_L [Wb]", e["Phi"]["L"], "3.40e-4")
    Q("B: Phi_C (down) [Wb]", -e["Phi"]["C"], "1.34e-4")
    Q("B: Phi_R (down) [Wb]", -e["Phi"]["R"], "2.06e-4")
    Q("B: B_L [T]", e["B"]["L"], "0.850")
    Q("B: B_C [T]", -e["B"]["C"], "0.168")
    Q("B: B_R [T]", -e["B"]["R"], "0.514")
    assert abs(sum(e["Phi"].values())) < 1e-15, "node law"
    for k_ in "LCR":
        assert abs(e["Phi"][k_] - e["mesh"][k_]) < 1e-15, "nodal and loop analysis disagree"
    Q("B: right-limb share [%]", 100 * e["split_R"], "60")
    # 14.6 mutual inductance
    Q("B: R seen by coil 1 [A/Wb]", e["R_in1"], "5.88e5")
    Q("B: L11 [mH]", 1e3 * e["L11"], "68.0")
    assert abs(e["L11"] - e["L11_direct"]) < 1e-12
    Q("B: M = L21 [mH]", 1e3 * e["M21"], "20.6")
    Q("B: L22 [mH]", 1e3 * e["L22"], "12.1")
    assert abs(e["L22"] - e["L22_formula"]) < 1e-12
    assert abs(e["M21"] - e["M12"]) < 1e-12, "reciprocity L12 = L21"
    Q("B: coupling k", e["k"], "0.72")
    # self-check 3: right gap closed -> psi_T falls, total flux rises
    psi0, Phi0, _, _ = ecore_nodal({"L": ECORE["N1"] * ECORE["I1"]})
    p_closed = dict(ECORE, gR=0.0)
    psi1, Phi1, R1, _ = ecore_nodal({"L": ECORE["N1"] * ECORE["I1"]}, p_closed)
    Q("B': R_R with closed gap [A/Wb]", R1["R"], "1.79e5")
    assert psi1 < psi0 and Phi1["L"] > Phi0["L"] and -Phi1["R"] > -Phi0["R"]

    # --- 14.5 conductivity contrast ---
    assert 5.8e7 / 1e-14 > 1e21, "sigma_Cu / sigma_air > 10^21"

    # --- 14.6 / Example 14.3: electromagnet with armature ---
    m = armature()
    Q("C: R_Fe [A/Wb]", m["Ri"], "1.99e5")
    Q("C: R of two gaps [A/Wb]", m["Rgap"], "3.98e6")
    Q("C: R total [A/Wb]", m["R"], "4.18e6")
    Q("C: Phi [Wb]", m["Phi"], "1.44e-4")
    Q("C: B [T]", m["B"], "0.359")
    Q("C: L [mH]", 1e3 * m["L"], "21.5")
    Q("C: W [J]", m["W"], "0.043")
    Q("C: F, constant flux [N]", m["f_flux"], "-41.0")
    Q("C: F, constant MMF [N]", m["f_mmf"], "-41.0")
    assert abs(m["f_flux"] - m["f_fd_flux"]) < 1e-5 * abs(m["f_flux"])
    assert abs(m["f_mmf"] - m["f_fd_mmf"]) < 1e-5 * abs(m["f_mmf"])
    Q("C: magnetic pressure [kPa]", m["pressure"] / 1e3, "51.3")
    Q("C: pressure x two faces [N]", m["f_pressure"], "41.0")
    m2 = armature(x=0.5e-3)
    Q("C: |F| at x = 0.5 mm [N]", -m2["f_flux"], "150")
    Q("C: force ratio 0.5 mm / 1 mm", m2["f_flux"] / m["f_flux"], "3.6")

    # --- 14.7 / Example 14.4: nonlinear core ---
    o4, o10, o20, o30 = (core_operating_point(x) for x in (400.0, 1000.0, 2000.0, 3000.0))
    Q("D: B_g at 400 A [T]", o4["B"], "0.421")
    Q("D: initial relative permeability of the synthetic curve", float(mu_diff(0.0)) / MU0, "1.59e3")
    Q("D: linear-model B at 400 A [T]", o4["linear"], "0.437")
    Q("D: iron MMF share at 400 A [%]", 100 * o4["frac_iron"], "16")
    Q("D: B_g at 1000 A [T]", o10["B"], "1.03")
    Q("D: B_g at 2000 A [T]", o20["B"], "1.60")
    Q("D: linear-model B at 2000 A [T]", o20["linear"], "2.19")
    Q("D: ideal-iron B at 2000 A [T]", o20["ideal"], "2.51")
    Q("D: H_Fe at 2000 A [A/m]", o20["H"], "2.43e3")
    Q("D: iron MMF share at 2000 A [%]", 100 * o20["frac_iron"], "37")
    Q("D: MMF increase 2000 -> 3000 A [%]", 100 * (3000.0 / 2000.0 - 1), "50")
    Q("D: B increase 2000 -> 3000 A [%]", 100 * (o30["B"] / o20["B"] - 1), "0.7")
    Q("D: H ratio 3000 A / 2000 A", o30["H"] / o20["H"], "2.4")
    Q("D: L_sec at 400 A [mH]", 1e3 * o4["Lsec"], "16.9")
    Q("D: L_diff at 400 A [mH]", 1e3 * o4["Ldiff"], "16.7")
    Q("D: mu_r secant at 2000 A", o20["mur_sec"], "522")
    Q("D: mu_r differential at 2000 A", o20["mur_diff"], "15.4")
    Q("D: L_sec at 2000 A [mH]", 1e3 * o20["Lsec"], "12.8")
    Q("D: L_diff at 2000 A [mH]", 1e3 * o20["Ldiff"], "0.98")
    hist = core_newton(2000.0)
    table = [("0", "-2000"), ("1057", "-578"), ("1887", "-181"), ("2395", "-12.2"),
             ("2433.6", "-2.8e-2"), ("2433.72", "-1.5e-7")]
    for (k, H, f), (Ht, ft) in zip(hist, table):
        Q(f"D: Newton H_{k} [A/m]", H, Ht)
        Q(f"D: Newton f_{k} [A]", f, ft)
    assert all(hist[i][1] <= hist[i + 1][1] <= o20["H"] + 1e-9 for i in range(5)), "monotone from the left"
    assert abs(hist[6][2]) < 1e-9

    # --- 14.8 / Example 14.5: permanent magnet (old example, numerics.magnetic_circuit) ---
    pm = pm_single_loop()
    r = pm["r"]
    assert (r["Br_T"], r["mu_rec_relative"], r["magnet_length_m"], r["gap_m"], r["yoke_length_m"],
            r["Am_m2"], r["Ag_m2"], r["Ay_m2"]) == (1.2, 1.05, 0.05, 0.02, 0.40, 0.002, 0.002, 0.001)
    Q("E: F_c [A]", pm["Fc"], "4.55e4")
    Q("E: R_m [A/Wb]", pm["Rm"], "1.89e7")
    Q("E: R_g [A/Wb]", pm["Rg"], "7.96e6")
    Q("E: share of F_c on R_m (ideal yoke) [%]", 100 * pm["Rm_share"], "70")
    Q("E: Phi [Wb]", r["flux_Wb"], "1.60702e-3")
    Q("E: B_g [T]", r["Bg_T"], "0.803511")
    Q("E: B_m [T]", r["Bm_T"], "0.803511")
    Q("E: B_y [T]", r["By_T"], "1.607023")
    Q("E: B_y (short) [T]", r["By_T"], "1.607")
    Q("E: H_y [A/m]", r["Hy_A_per_m"], "5590.68")
    Q("E: H_m [A/m]", r["Hm_A_per_m"], "-3.00491e5")
    assert abs(r["loop_residual_A"]) < 2e-12 and abs(pm["resid"]) < 1e-9
    Q("E: ideal-yoke B_g [T]", r["ideal_infinite_mu_yoke_Bg_T"], "0.845070")
    Q("E: ideal-yoke B_g in figure [T]", r["ideal_infinite_mu_yoke_Bg_T"], "0.8451")
    Q("E: nonlinear B_g in figure [T]", r["Bg_T"], "0.8035")
    Q("E: deficit [%]", 100 * pm["deficit"], "4.92")
    Q("E: yoke MMF H_y l_y [A]", pm["Fy"], "2236")
    Q("E: yoke share of F_c [%]", 100 * pm["frac_yoke"], "4.92")
    assert abs(pm["frac_yoke"] - pm["deficit"]) < 1e-12, "B_g/B_g,ideal = 1 - H_y l_y / F_c"
    Q("E: magnet MMF -H_m l_m [A]", -pm["Fm_drop"], "1.50e4")
    Q("E: gap MMF [A]", pm["Fg_drop"], "1.28e4")
    Q("E: yoke MMF [A]", pm["Fy"], "2.24e3")
    assert abs(-pm["Fm_drop"] - pm["Fg_drop"] - pm["Fy"]) < 1e-8, "MMF budget closes"
    Q("E: B_y / B_g", r["By_T"] / r["Bg_T"], "2.000")
    Q("E: gap field lower than ideal (about) [%]", 100 * pm["deficit"], "5")
    Q("E: ideal operating point H_m [A/m]", pm["Hm_ideal"], "-2.69e5")
    Hm20, Bm20 = recoil_operating_point(0.02)
    Q("E: ideal operating point B_m [T]", Bm20, "0.845")
    assert abs(Hm20 - pm["Hm_ideal"]) < 1e-6
    for gg in (0.01, 0.02, 0.05 / 1.05):
        Hm, Bm = recoil_operating_point(gg)
        assert abs(Bm + MU0 * 0.05 / gg * Hm) < 1e-12        # on the load line
        assert abs(Bm - (1.2 + MU0 * 1.05 * Hm)) < 1e-12     # on the recoil line
    Q("E: (BH)max [J/m^3]", pm["BHmax"], "2.73e5")
    Q("E: B*_m [T]", 1.2 / 2, "0.6")
    Q("E: H*_m [A/m]", -pm["Hc"] / 2, "-4.55e5")
    Q("E: g for (BH)max [mm]", 1e3 * pm["g_opt"], "47.6")
    Q("E: |B_m H_m| / (BH)max at g = 20 mm [%]", 100 * pm["BH_ideal"] / pm["BHmax"], "83")
    assert abs(pm["Vm_formula"] - pm["Vm"]) < 1e-15, "V_m = B_g^2 V_g / (mu0 B_m |H_m|)"

    # --- 14.9 / Example 14.6: multi-pillar yoke ---
    base = multipillar(4, 0.005)
    Q("F: ideal-yoke B_g [T]", base["ideal"], "0.192")
    Q("F: 2 F_c [A]", base["Fc2"], "7.28e4")
    for N_, txt in ((2, "0.084"), (3, "0.124"), (4, "0.162"), (6, "0.191")):
        Q(f"F: B_g, N={N_}, A_p=0.005", multipillar(N_, 0.005)["Bg"], txt)
    Q("F: B_p, N=4 [T]", base["Bp"], "1.62")
    Q("F: pillar MMF share, N=4 [%]", 100 * base["frac_pillars"], "15")
    Q("F: B_p, N=6 [T]", multipillar(6, 0.005)["Bp"], "1.27")
    knee = base["ideal"] * PILLAR["Am"] / 1.6
    Q("F: knee estimate [m^2]", knee, "0.024")
    Q("F: ideal-yoke flux [Wb]", base["ideal"] * PILLAR["Am"], "0.0384")
    for tot in np.linspace(0.024, 0.2, 30):
        assert multipillar(4, tot / 4)["ratio"] > 0.98, "within 2 % above the knee"
    for tot in (0.008, 0.012, 0.016):
        rr = multipillar(4, tot / 4)
        assert rr["Bp"] > 1.6 and rr["ratio"] < 0.7, "saturated below the knee"
    assert abs(multipillar(2, 0.01)["Bg"] - multipillar(4, 0.005)["Bg"]) < 1e-12, "only N*A_p matters"
    a_min = brentq(lambda t: multipillar(4, t / 4)["Bg"] - 0.18, 0.01, 0.03)
    Q("F: minimal total area for B_g >= 0.18 T [m^2]", a_min, "0.022")
    # magnet operating point (permeance coefficient P of eq:loadline / Sec. bh-hysteresis)
    Q("E: permeance coefficient P = l_m/g", 0.05 / 0.02, "2.5")
    Q("F: permeance coefficient P = 2 l_m/g", 2 * PILLAR["lm"] / PILLAR["g"], "0.2")
    def magnet_H(Bg, p=PILLAR):                      # B_m = B_g (A_g = A_m) on the straight recoil line
        return (Bg - p["Br"]) / (MU0 * p["mu_rec"])
    Q("F: magnet H_m, ideal yoke [kA/m]", magnet_H(base["ideal"]) / 1e3, "-764")
    Q("F: magnet H_m, N=2, A_p=0.005 [kA/m]", magnet_H(multipillar(2, 0.005)["Bg"]) / 1e3, "-846")
    assert abs(magnet_H(base["ideal"]) - recoil_operating_point(0.40, lm=0.08)[0]) < 1e-6, "P = 2 l_m / g"

    # --- 14.5 and 14.10: 2-D field solution versus network ---
    assert FEM_GAPS[0] / FEM["w"] == 0.025 and FEM_GAPS[-1] / FEM["w"] == 0.4
    fem = {g: c_core_fem(g) for g in FEM_GAPS}
    Q("V: off-path flux fraction at g = 4 mm [%]", 100 * fem[4.0e-3]["offpath"], "47")
    f1, f8 = fem[1.0e-3], fem[8.0e-3]
    Q("V: B centre ratio, g = 1 mm", f1["Bc_ratio"], "0.994")
    Q("V: B centre ratio, g = 8 mm", f8["Bc_ratio"], "0.987")
    assert max(abs(f["Bc_ratio"] - 1) for f in fem.values()) < 0.015, "gap-centre B within 1.5 %"
    Q("V: limb flux ratio, g = 1 mm", f1["limb_ratio"], "1.26")
    Q("V: limb flux excess, g = 1 mm [%]", 100 * (f1["limb_ratio"] - 1), "26")
    Q("V: limb flux ratio, g = 8 mm", f8["limb_ratio"], "2.50")
    Q("V: k_eff, g = 1 mm", f1["k_eff"], "1.31")
    Q("V: k_eff, g = 8 mm", f8["k_eff"], "2.57")
    pred = predict_with_k(8.0e-3, f1["k_eff"])
    Q("V: hold-out error at 8 mm with k(1 mm) [%]", 100 * (1 - pred / f8["phi_limb"]), "48")
    # the field solution itself: mesh refinement x2 and outer box x2 change the quoted ratios < 0.5 %
    for g in (1.0e-3, 8.0e-3):
        fine = c_core_fem(g, refine=2.0)
        big = c_core_fem(g, half=1.0)
        for key in ("limb_ratio", "Bc_ratio", "k_eff"):
            assert abs(fine[key] / fem[g][key] - 1) < 5e-3, ("mesh", g, key)
            assert abs(big[key] / fem[g][key] - 1) < 5e-3, ("domain", g, key)
    for g, f in fem.items():
        # linkage from the field energy agrees with the limb flux to a few per cent (consistency)
        assert abs(f["phi_energy"] / f["phi_limb"] - 1) < 0.03
        log.append(f"V: g={g*1e3:.1f} mm g/w={f['gw']:.3f} nodes={f['nodes']} limb/lumped={f['limb_ratio']:.4f} "
                   f"Bc/lumped={f['Bc_ratio']:.4f} k_eff={f['k_eff']:.3f} offpath={f['offpath']:.3f}")

    if verbose:
        print("\n".join(log))
        print("check_ch14: all assertions passed")
    return True


if __name__ == "__main__":
    run_checks()
