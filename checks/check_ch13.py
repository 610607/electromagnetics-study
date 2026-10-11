"""Chapter 13 (ferromagnetic materials): models and checks for every number in zh/ch13_bh.tex.

    python3 checks/check_ch13.py          # run all assertions, print the quoted values

The figure module figsrc/fig_ch13.py imports the model functions from this file, so the text,
the figures and the checks share one implementation.  Only NumPy, SciPy and the book's
numerics.py (synthetic B-H curve, MU0) are used.  Every material value is a teaching input,
not data for a real steel or magnet grade.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq

import numerics
from numerics import MU0, synthetic_B

# ---------------------------------------------------------------------------
# eq:syntheticbh  B(H) = mu0 H + B_s tanh(H / H_s)   (same function as numerics.synthetic_B)
# ---------------------------------------------------------------------------
BS, HS = 1.6, 800.0


def B_of_H(H):
    return MU0 * np.asarray(H, float) + BS * np.tanh(np.asarray(H, float) / HS)


def sech2(x):
    """sech^2 without overflow for large |x|."""
    e = np.exp(-2.0 * np.abs(np.asarray(x, float)))
    return 4.0 * e / (1.0 + e) ** 2


def mu_diff(H):
    """eq:permeabilities, differential: dB/dH = mu0 + (B_s/H_s) sech^2(H/H_s)."""
    return MU0 + BS / HS * sech2(np.asarray(H, float) / HS)


def mu_sec(H):
    """eq:permeabilities, secant: B/H (continuous limit mu_diff(0) at H = 0)."""
    H = np.asarray(H, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(H) > 1e-9, B_of_H(H) / np.where(H == 0, 1, H), mu_diff(0.0))
    return out


def H_of_B(b, iters=80):
    """Inverse curve h(b) for b >= 0 by Newton's method from H = 0.

    f(H) = B(H) - b is increasing and concave for H >= 0, so Newton started at H = 0 increases
    monotonically to the root (no overshoot).  Vectorised; accurate to machine precision.
    """
    b = np.asarray(b, float)
    sgn = np.sign(b)
    bb = np.abs(b)
    H = np.zeros_like(bb)
    for _ in range(iters):
        H = H - (B_of_H(H) - bb) / mu_diff(H)
    return sgn * H


def nu_sec(b):
    """eq:HofB: h(b)/b with the continuous limit 1/mu_diff(0) at b = 0."""
    b = np.asarray(b, float)
    small = np.abs(b) < 1e-12
    return np.where(small, 1.0 / mu_diff(0.0), H_of_B(np.where(small, 1.0, b)) / np.where(small, 1.0, b))


# ---------------------------------------------------------------------------
# Section 13.3  Saturation is not a hard limit
# ---------------------------------------------------------------------------
SAT_PATH = 0.20      # m, mean iron path of ex:systems-core (Ch 7)


def saturation_table():
    rows = {}
    for b in (1.5, 1.6, 1.7):
        H = float(H_of_B(b))
        rows[b] = dict(H=H, mmf=H * SAT_PATH, mur_sec=b / (MU0 * H), mur_diff=float(mu_diff(H)) / MU0,
                       mu0M=b - MU0 * H)
    return rows


# ---------------------------------------------------------------------------
# Section 13.4  One iron, three questions (operating point H0 = 2 H_s)
# ---------------------------------------------------------------------------
H0 = 1600.0
DH = 10.0            # A/m, small perturbation


def three_permeabilities(H0=H0, dH=DH):
    B0 = float(B_of_H(H0))
    ms, md = B0 / H0, float(mu_diff(H0))
    dB_par = md * dH
    dB_perp = ms * dH
    # exact check of the transverse response: rotate H by dH/H0 and evaluate B along the new direction
    Hvec = np.array([H0, dH])
    Bvec = B_of_H(np.hypot(*Hvec)) * Hvec / np.hypot(*Hvec)
    dB_perp_exact = Bvec[1]
    # exact check of the parallel response (finite difference)
    dB_par_exact = float(B_of_H(H0 + dH) - B_of_H(H0))
    return dict(B0=B0, mur_sec=ms / MU0, mur_diff=md / MU0, ratio=ms / md, dB_par=dB_par, dB_perp=dB_perp,
                dB_par_exact=dB_par_exact, dB_perp_exact=dB_perp_exact, angle=dB_perp / B0, angleH=dH / H0,
                dB_par_if_sec=ms * dH, mu0M=B0 - MU0 * H0, mu0M_share=(B0 - MU0 * H0) / B0)


def incremental_reluctivity(Bvec):
    """Incremental reluctivity tensor of H = nu_sec(|B|) B (isotropic, anhysteretic):
    nu_inc = nu_sec I + (nu_diff - nu_sec) b b^T, b = B/|B|."""
    Bvec = np.asarray(Bvec, float)
    b = np.linalg.norm(Bvec)
    e = Bvec / b
    H = float(H_of_B(b))
    ns = H / b
    nd = 1.0 / float(mu_diff(H))
    return ns * np.eye(len(Bvec)) + (nd - ns) * np.outer(e, e), ns, nd


# ---------------------------------------------------------------------------
# Section 13.5  Hysteresis: teaching loops
# ---------------------------------------------------------------------------
# Soft loop: the two branches are eq:syntheticbh with the magnetisation part shifted by +/- HC0.
HC0 = 400.0          # A/m (teaching, exaggerated so that the loop is visible)


def soft_branch(H, descending):
    s = -1.0 if descending else 1.0
    H = np.asarray(H, float)
    return MU0 * H + BS * np.tanh((H - s * HC0) / HS)


def soft_loop_numbers(Hmax=6000.0, n=200001):
    Br = float(soft_branch(0.0, True))
    HcJ = HC0                                   # mu0 M = B - mu0 H vanishes at H = -HC0 on the descending branch
    HcB = -brentq(lambda h: float(soft_branch(h, True)), -2 * HC0, 0.0, xtol=1e-12)
    # loop area  \oint H dB  (counter-clockwise traversal: up the ascending branch, down the descending one)
    H = np.linspace(-Hmax, Hmax, n)
    up = soft_branch(H, False)
    dn = soft_branch(H, True)
    # close the loop with the (tiny) vertical segments at +/- Hmax:  \oint H dB = \int H dB_up - \int H dB_dn + ends
    area = np.trapezoid(H, up) - np.trapezoid(H, dn)
    area += Hmax * (dn[-1] - up[-1]) - (-Hmax) * (dn[0] - up[0])
    return dict(Br=Br, HcJ=HcJ, HcB=HcB, area=area, rect=4 * HC0 * BS, Hmax=Hmax)


# Hard (permanent-magnet) teaching material: straight recoil line above a knee.
BR, MUREC = 1.2, 1.05            # eq:recoil teaching values (also used in Ch 14)
HC_LINE = BR / (MU0 * MUREC)     # B = 0 intercept of the straight recoil line
KNEE_W = 60.0e3                  # A/m, width of the (teaching) knee


def hard_mu0M(H, HcJ, w=KNEE_W):
    """Descending intrinsic curve mu0 M(H) of a teaching permanent-magnet material.

    mu0 M = [B_r + mu0 (mu_rec - 1) H] * tanh((H + H_cJ)/w): straight (recoil-line) behaviour for
    H >> -H_cJ, zero exactly at H = -H_cJ.  A schematic shape, not a material model.
    """
    H = np.asarray(H, float)
    return (BR + MU0 * (MUREC - 1.0) * H) * np.tanh((H + HcJ) / w)


def hard_B(H, HcJ, w=KNEE_W):
    return MU0 * np.asarray(H, float) + hard_mu0M(H, HcJ, w)


def hard_numbers(HcJ, w=KNEE_W):
    HcB = -brentq(lambda h: float(hard_B(h, HcJ, w)), -HcJ, 0.0, xtol=1e-6)
    return dict(HcJ=HcJ, HcB=HcB, line=HC_LINE)


HCJ_HIGH, HCJ_LOW = 1500.0e3, 800.0e3    # teaching values on either side of HC_LINE = 909 kA/m


def knee_recoil(P, HcJ=HCJ_LOW, w=KNEE_W):
    """Operating point of an isolated magnet with permeance coefficient P (load line B = -P mu0 H)
    on the knee curve; after the field is removed the magnet recoils along a straight line of slope
    mu0 mu_rec to the new remanence B_r' (the recoil-line approximation)."""
    f = lambda h: float(hard_B(h, HcJ, w)) + P * MU0 * h
    Hp = brentq(f, -3.0e6, -1.0, xtol=1e-6)
    Bp = float(hard_B(Hp, HcJ, w))
    Br_new = Bp - MU0 * MUREC * Hp
    # the same load line on the ideal straight line (no knee)
    Hi = -BR / (MU0 * (MUREC + P))
    return dict(P=P, Hp=Hp, Bp=Bp, Br_new=Br_new, loss=1 - Br_new / BR, Hi=Hi, Bi=BR + MU0 * MUREC * Hi)


P_SAFE, P_KNEE = 1.0, 0.2         # permeance coefficients used in the figure (0.2 = 2 l_m / g of the Ch 14 multi-pillar example)


# ---------------------------------------------------------------------------
# Section 13.6  Scalar curve in a vector field: correct (eq:HofB) vs component-wise
# ---------------------------------------------------------------------------
def componentwise(bmag=1.5, angle_deg=45.0):
    th = np.deg2rad(angle_deg)
    B = bmag * np.array([np.cos(th), np.sin(th)])
    H_ok = nu_sec(bmag) * B
    H_cw = np.array([float(H_of_B(B[0])), float(H_of_B(B[1]))])
    ang_cw = np.rad2deg(np.arctan2(H_cw[1], H_cw[0]))
    return dict(B=B, H_ok=H_ok, H_cw=H_cw, Hmag_ok=float(np.hypot(*H_ok)), Hmag_cw=float(np.hypot(*H_cw)),
                ang_cw=ang_cw, ratio=float(np.hypot(*H_cw) / np.hypot(*H_ok)))


# ---------------------------------------------------------------------------
# Section 13.7  Importing data: units, interpolation, extrapolation
# ---------------------------------------------------------------------------
GAUSS = 1.0e-4                 # T
OERSTED = 1000.0 / (4 * np.pi)  # A/m


def units_example(B_kG=15.0, H_Oe=10.0):
    B = B_kG * 1e3 * GAUSS
    H = H_Oe * OERSTED
    mur = B / (MU0 * H)
    mur_wrong = B / (MU0 * H_Oe)        # oersted read as A/m
    return dict(B=B, H=H, mur=mur, mur_wrong=mur_wrong, factor=mur_wrong / mur)


SAMPLES_H = np.array([0.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0])


def interpolants(Hs=SAMPLES_H):
    Bs = B_of_H(Hs)
    spline = CubicSpline(Hs, Bs)                 # SciPy default: not-a-knot end conditions
    pchip = PchipInterpolator(Hs, Bs)            # monotone piecewise cubic Hermite
    return Hs, Bs, spline, pchip


def interpolation_numbers(n=90001):
    Hs, Bs, sp, pc = interpolants()
    H = np.linspace(Hs[0], Hs[-1], n)
    dsp = sp(H, 1)
    neg = H[dsp < 0]
    bsp = sp(H)
    i_over = int(np.argmax(bsp - B_of_H(H)))
    i_min = int(np.argmin(dsp))
    # PCHIP: monotone (slope >= 0) but its end slope is set by a one-sided formula
    dpc = pc(H, 1)
    out = dict(neg_lo=float(neg.min()), neg_hi=float(neg.max()), min_mur=float(dsp[i_min] / MU0),
               H_min=float(H[i_min]), overshoot=float(bsp[i_over] - B_of_H(H[i_over])), H_over=float(H[i_over]),
               Bmax_sp=float(bsp.max()), pc_min=float(dpc.min() / MU0), pc_end=float(pc(Hs[-1], 1) / MU0),
               pc_monotone=bool(np.all(np.diff(pc(H)) >= -1e-15)))
    for Hq in (3000.0, 6000.0):
        out[f"true_{int(Hq)}"] = float(mu_diff(Hq) / MU0)
        out[f"sp_{int(Hq)}"] = float(sp(Hq, 1) / MU0)
        out[f"pc_{int(Hq)}"] = float(pc(Hq, 1) / MU0)
    # B(H) of the spline is not monotone, so its inverse H(B) is multivalued near the dip
    j = int(np.argmax(bsp[: int(0.7 * n)]))
    k = j + int(np.argmin(bsp[j:]))
    out.update(B_peak=float(bsp[j]), H_peak=float(H[j]), B_dip=float(bsp[k]), H_dip=float(H[k]))
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def quoted(value, text, name=""):
    """Assert that `value` rounds to the number printed in the chapter as `text`
    (tolerance: half a unit in the last printed digit)."""
    t = text.replace("−", "-")
    mant, _, exp = t.lower().partition("e")
    e = int(exp) if exp else 0
    dec = len(mant.split(".")[1]) if "." in mant else 0
    tol = 0.5 * 10.0 ** (e - dec) * (1 + 1e-9)
    assert abs(value - float(t)) <= tol, f"{name}: computed {value!r} does not round to {text}"
    return value


def run_checks(verbose=True):
    log = []

    def Q(name, value, text):
        quoted(value, text, name)
        log.append(f"{name:<52s} {value: .6g}  ~ {text}")

    # --- the synthetic curve is the book's numerics.synthetic_B ---
    Ht = np.linspace(-5e4, 5e4, 1001)
    assert np.allclose(B_of_H(Ht), numerics.synthetic_B(Ht), rtol=0, atol=1e-15)
    assert (BS, HS) == (1.6, 800.0)
    # mu_diff is the derivative of B(H) (central differences)
    h = 1e-3
    assert np.allclose(mu_diff(Ht), (B_of_H(Ht + h) - B_of_H(Ht - h)) / (2 * h), rtol=1e-6)
    # inverse curve
    bt = np.linspace(-2.5, 2.5, 501)
    assert np.allclose(B_of_H(H_of_B(bt)), bt, atol=1e-12)

    # --- 13.2 / 13.3: curve parameters and saturation ---
    Q("mu_r at H = 0 (1 + B_s/(mu0 H_s))", float(mu_diff(0.0)) / MU0, "1593")
    Q("mu0 M_s = B_s [T]", BS, "1.6")
    Q("M_s = B_s/mu0 [A/m]", BS / MU0, "1.27e6")
    s = saturation_table()
    Q("H for B = 1.5 T [A/m]", s[1.5]["H"], "1.37e3")
    Q("H for B = 1.6 T [A/m]", s[1.6]["H"], "2.73e3")
    Q("H for B = 1.7 T [A/m]", s[1.7]["H"], "7.96e4")
    Q("MMF over 0.2 m at 1.5 T [A]", s[1.5]["mmf"], "273")
    Q("MMF over 0.2 m at 1.6 T [A]", s[1.6]["mmf"], "547")
    Q("MMF over 0.2 m at 1.7 T [A]", s[1.7]["mmf"], "1.59e4")
    Q("1.7 T: H ratio vs 1.6 T", s[1.7]["H"] / s[1.6]["H"], "29")
    Q("mu_r,diff at 1.5 T", s[1.5]["mur_diff"], "197")
    Q("mu_r,diff at 1.6 T", s[1.6]["mur_diff"], "7.8")
    Q("mu_r,diff at 1.7 T", s[1.7]["mur_diff"], "1.00")
    Q("mu_r,sec at 1.7 T", s[1.7]["mur_sec"], "17.0")
    Q("mu0 M at 1.7 T [T]", s[1.7]["mu0M"], "1.600")

    # --- 13.4: one operating point, three permeabilities ---
    t = three_permeabilities()
    Q("H0 [A/m] = 2 H_s", H0 / HS, "2")
    Q("B0 at H0 [T]", t["B0"], "1.544")
    Q("mu_r,sec at H0", t["mur_sec"], "768")
    Q("mu_r,diff at H0", t["mur_diff"], "113")
    Q("mu_sec/mu_diff at H0", t["ratio"], "6.8")
    Q("mu0 M share of B0 [%]", 100 * t["mu0M_share"], "99.9")
    Q("dB parallel for dH = 10 A/m [mT]", 1e3 * t["dB_par"], "1.43")
    Q("dB parallel (finite difference) [mT]", 1e3 * t["dB_par_exact"], "1.41")
    Q("dB parallel if mu_sec used [mT]", 1e3 * t["dB_par_if_sec"], "9.65")
    Q("dB transverse for dH = 10 A/m [mT]", 1e3 * t["dB_perp"], "9.65")
    Q("dB transverse (exact rotation) [mT]", 1e3 * t["dB_perp_exact"], "9.65")
    Q("rotation angle of B [mrad]", 1e3 * t["angle"], "6.25")
    Q("rotation angle of H [mrad]", 1e3 * t["angleH"], "6.25")
    # incremental tensor: eigenvalues 1/mu_diff along B, 1/mu_sec across
    nu, ns, nd = incremental_reluctivity([t["B0"], 0.0, 0.0])
    assert np.isclose(nu[0, 0], 1 / (t["mur_diff"] * MU0)) and np.isclose(nu[1, 1], 1 / (t["mur_sec"] * MU0))
    # finite-difference check of the tensor in a rotated frame
    Bv = t["B0"] * np.array([0.6, 0.8, 0.0])
    nu, _, _ = incremental_reluctivity(Bv)
    Hfun = lambda B: nu_sec(np.linalg.norm(B)) * B
    eps = 1e-7
    J = np.column_stack([(Hfun(Bv + eps * e) - Hfun(Bv - eps * e)) / (2 * eps) for e in np.eye(3)])
    assert np.allclose(J, nu, rtol=1e-5)

    # --- 13.5: hysteresis ---
    L = soft_loop_numbers()
    Q("soft loop B_r [T]", L["Br"], "0.739")
    Q("soft loop H_cJ [A/m]", L["HcJ"], "400")
    Q("soft loop H_cB [A/m]", L["HcB"], "399.7")
    Q("soft loop H_cJ - H_cB [A/m]", L["HcJ"] - L["HcB"], "0.3")
    Q("soft loop area [J/m^3 per cycle]", L["area"], "2.56e3")
    Q("rectangular estimate 4 H_c B_s [J/m^3]", L["rect"], "2.56e3")
    Q("recoil-line intercept B_r/(mu0 mu_rec) [kA/m]", HC_LINE / 1e3, "909")
    hi, lo = hard_numbers(HCJ_HIGH), hard_numbers(HCJ_LOW)
    Q("high-H_cJ material: H_cB [kA/m]", hi["HcB"] / 1e3, "909")
    Q("high-H_cJ material: H_cJ [kA/m]", hi["HcJ"] / 1e3, "1500")
    Q("low-H_cJ material: H_cB [kA/m]", lo["HcB"] / 1e3, "734")
    Q("low-H_cJ material: H_cJ [kA/m]", lo["HcJ"] / 1e3, "800")
    ks, kk = knee_recoil(P_SAFE), knee_recoil(P_KNEE)
    Q("P = 1: operating B [T]", ks["Bp"], "0.585")
    Q("P = 1: remanence after recoil [T]", ks["Br_new"], "1.200")
    Q("P = 0.2: straight-line operating B [T]", kk["Bi"], "0.192")
    Q("P = 0.2: straight-line operating H [kA/m]", kk["Hi"] / 1e3, "-764")
    Q("P = 0.2: knee operating H [kA/m]", kk["Hp"] / 1e3, "-705")
    Q("P = 0.2: knee operating B [T]", kk["Bp"], "0.177")
    Q("P = 0.2: remanence after recoil [T]", kk["Br_new"], "1.11")
    Q("P = 0.2: remanence lost [%]", 100 * kk["loss"], "7.7")
    assert kk["Br_new"] < BR and abs(ks["Br_new"] - BR) < 1e-4
    # --- 13.6: vector use ---
    c45 = componentwise(1.5, 45.0)
    c30 = componentwise(1.5, 30.0)
    c0 = componentwise(1.5, 0.0)
    Q("|H| correct at 1.5 T [A/m]", c45["Hmag_ok"], "1367")
    Q("|H| componentwise, B along x [A/m]", c0["Hmag_cw"], "1367")
    Q("|H| componentwise, B at 45 deg [A/m]", c45["Hmag_cw"], "902")
    Q("componentwise / correct at 45 deg", c45["ratio"], "0.66")
    Q("componentwise H direction for B at 30 deg [deg]", c30["ang_cw"], "24.2")
    Q("nu_sec(0) = 1/mu_diff(0) continuity", float(nu_sec(1e-9) * mu_diff(0.0)), "1.000")

    # --- 13.7: data import ---
    u = units_example()
    Q("15 kG [T]", u["B"], "1.5")
    Q("10 Oe [A/m]", u["H"], "795.8")
    Q("1 Oe [A/m]", OERSTED, "79.58")
    Q("mu_r from correct units", u["mur"], "1500")
    Q("mu_r with Oe read as A/m", u["mur_wrong"], "1.19e5")
    Q("factor of the unit error", u["factor"], "79.6")
    Q("component of 1.5 T at 45 deg [T]", float(c45["B"][0]), "1.06")
    Q("relative increase 1.6 -> 1.7 T [%]", 100 * (1.7 / 1.6 - 1), "6")
    Q("number of samples", len(SAMPLES_H), "7")
    I = interpolation_numbers()
    Q("spline: negative slope from [A/m]", I["neg_lo"], "3.95e3")
    Q("spline: negative slope to [A/m]", I["neg_hi"], "7.97e3")
    Q("spline: most negative mu_r,diff", I["min_mur"], "-9.8")
    Q("spline: local peak at H [A/m]", I["H_peak"], "3.95e3")
    Q("spline: local dip at H [A/m]", I["H_dip"], "7.97e3")
    Q("sample value at 5000 A/m [T]", float(B_of_H(5000.0)), "1.606")
    Q("true mu_r,diff at 3000 A/m", I["true_3000"], "4.5")
    Q("spline mu_r,diff at 3000 A/m", I["sp_3000"], "11.4")
    Q("PCHIP mu_r,diff at 3000 A/m", I["pc_3000"], "8.3")
    Q("spline: local peak B [T]", I["B_peak"], "1.612")
    Q("spline: local dip B [T]", I["B_dip"], "1.579")
    Q("true mu_r,diff at 6000 A/m", I["true_6000"], "1.00")
    Q("spline mu_r,diff at 6000 A/m", I["sp_6000"], "-9.8")
    Q("PCHIP slope at the last sample (mu_r)", I["pc_end"], "0.0")
    assert I["pc_monotone"] and I["pc_min"] >= 0.0

    # --- self-check answers ---
    Q("self-check: mu_r,sec at 5000 A/m", float(mu_sec(5000.0)) / MU0, "256")
    Q("self-check: mu_r,diff at 5000 A/m", float(mu_diff(5000.0)) / MU0, "1.02")
    global HC0
    keep = HC0
    HC0 = 200.0
    half = soft_loop_numbers()
    HC0 = keep
    Q("self-check: loop area with H_c = 200 A/m [J/m^3]", half["area"], "1.28e3")
    kh = knee_recoil(P_KNEE, HcJ=HCJ_HIGH)
    Q("self-check: high-H_cJ material at P = 0.2, B_m [T]", kh["Bp"], "0.192")
    Q("self-check: high-H_cJ material at P = 0.2, B_r after [T]", kh["Br_new"], "1.200")

    if verbose:
        print("\n".join(log))
        print(f"\nall {len(log)} quoted values check out")
    return log


if __name__ == "__main__":
    run_checks()
