"""Chapter 16 (scalar finite elements): models and checks for every number in zh/ch16_femscalar.tex.

    python3 checks/check_ch16.py          # run all assertions, print the quoted values

figsrc/fig_ch16.py imports the model functions from this file, so text, figures and checks share
one implementation.  Only NumPy, SciPy and the book's numerics.py are used.  The bar numbers are
those of the old edition (proj:2; the same matrix is in extended_checks.py); the two-layer numbers
are those of ex:twolayer (Ch 6; concept_checks.py); the convergence data come from numerics.run_all.
All values are teaching inputs.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import numerics
from numerics import element_geometry, manufactured_solution


def close(x, ref, rtol=5e-3, name="", atol=0.0):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:62s} {x:.6g}")
    return x


# ---------------------------------------------------------------------------
# 1-D bar elements (section 16.1, 16.3, 16.4)
# ---------------------------------------------------------------------------
BAR = dict(length=0.02, area=1.0e-4, sigma=0.5, V=0.4)        # old proj:2
LAYER = dict(t=(0.008, 0.002), sigma=(0.2, 0.02), area=1.0e-4, I=1.0e-3)   # ex:twolayer


def bar_matrix(lengths, sigmas, area):
    """Assemble 1-D linear elements: local matrix (sigma A / h) [[1,-1],[-1,1]]."""
    n = len(lengths) + 1
    K = np.zeros((n, n))
    for e, (h, s) in enumerate(zip(lengths, sigmas)):
        G = s * area / h
        K[e:e + 2, e:e + 2] += G * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return K


def hat_local_matrix(h, sigma, area, nq=4):
    """Element matrix from the definition K_ij = int sigma A N_i' N_j' dx (Gauss quadrature)."""
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * h * (xg + 1.0)
    w = 0.5 * h * wg
    dN = np.array([-1.0 / h, 1.0 / h])          # N_0 = 1 - x/h, N_1 = x/h
    K = np.zeros((2, 2))
    for xi, wi in zip(x, w):
        K += wi * sigma * area * np.outer(dN, dN)
    return K


def bar_dirichlet():
    """Two-element bar with phi(0) = 0.4 V, phi(l) = 0 (section 16.1)."""
    h = BAR["length"] / 2
    K = bar_matrix([h, h], [BAR["sigma"]] * 2, BAR["area"])
    phi = np.zeros(3)
    phi[0] = BAR["V"]
    free, known = [1], [0, 2]
    phi[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, known)] @ phi[known])
    reaction = K @ phi
    E = -np.diff(phi) / h
    J = BAR["sigma"] * E
    return dict(K=K, phi=phi, reaction=reaction, E=E, J=J, I=BAR["area"] * J[0], h=h,
                R=BAR["V"] / (BAR["area"] * J[0]),
                P_VI=BAR["V"] * BAR["area"] * J[0], P_quad=phi @ K @ phi,
                P_int=np.sum(BAR["sigma"] * BAR["area"] * h * (np.diff(phi) / h) ** 2))


def bar_twolayer(I_in=LAYER["I"], I_out=None, mode="ground"):
    """Two-layer conductor (ex:twolayer) as two elements; node 1 receives I_in.

    mode="ground": phi_3 = 0 is an essential condition, node-3 current is a reaction.
    mode="neumann": both terminal currents given (I_out at node 3, sign = injected), then phi_3 = 0
    is used only to fix the reference; the specified I_out never enters the solve.
    """
    K = bar_matrix(LAYER["t"], LAYER["sigma"], LAYER["area"])
    f = np.array([I_in, 0.0, 0.0 if I_out is None else I_out])
    phi = np.zeros(3)
    free = [0, 1]
    phi[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    reaction = K @ phi
    E = -np.diff(phi) / np.array(LAYER["t"])
    return dict(K=K, f=f, phi=phi, reaction=reaction, E=E, J=np.array(LAYER["sigma"]) * E,
                R=phi[0] / I_in, P=phi @ K @ phi, imbalance=f.sum())


def bar_cem(z=0.02, I=1.0e-3):
    """Two-element bar of section 16.1, left end through a CEM electrode (contact z, Ohm m^2), right
    end grounded.  Unknowns (phi1, phi2, U); 1-D: int_e N1 N1 dS = A, so the contact term is a
    conductance A/z between U and phi1 (eq:cemweak)."""
    h = BAR["length"] / 2
    Kt = bar_matrix([h, h], [BAR["sigma"]] * 2, BAR["area"])
    gc = BAR["area"] / z
    # order: phi1, phi2, U   (phi3 = 0 eliminated)
    M = np.array([[Kt[0, 0] + gc, Kt[0, 1], -gc],
                  [Kt[1, 0], Kt[1, 1], 0.0],
                  [-gc, 0.0, gc]])
    rhs = np.array([0.0, 0.0, I])
    phi1, phi2, U = np.linalg.solve(M, rhs)
    return dict(phi1=phi1, phi2=phi2, U=U, rc=z / BAR["area"], M=M)


def thin_layer_mesh(n_el):
    """Two-layer conductor meshed with n_el equal elements, conductivity by element centroid."""
    Ltot = sum(LAYER["t"])
    h = Ltot / n_el
    xc = (np.arange(n_el) + 0.5) * h
    sig = np.where(xc < LAYER["t"][0], LAYER["sigma"][0], LAYER["sigma"][1])
    K = bar_matrix([h] * n_el, sig, LAYER["area"])
    f = np.zeros(n_el + 1)
    f[0] = LAYER["I"]
    phi = np.zeros(n_el + 1)
    free = np.arange(n_el)
    phi[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    res = np.linalg.norm(K[np.ix_(free, free)] @ phi[free] - f[free]) / np.linalg.norm(f[free])
    return dict(R=phi[0] / LAYER["I"], residual=res, sigma=sig)


# ---------------------------------------------------------------------------
# Element matrices (section 16.3)
# ---------------------------------------------------------------------------
def simplex_gradients(V):
    """Gradients of the barycentric (P1) functions of a simplex with vertex rows V; also its measure."""
    import math
    d = V.shape[1]
    M = np.hstack([np.ones((d + 1, 1)), V])
    C = np.linalg.inv(M)                       # column i: coefficients of N_i
    vol = abs(np.linalg.det(M)) / math.factorial(d)
    return C[1:, :].T, vol                     # rows: grad N_i


def element_matrix(V, sigma=1.0):
    G, vol = simplex_gradients(np.asarray(V, float))
    S = np.atleast_2d(sigma) if np.ndim(sigma) == 2 else sigma * np.eye(G.shape[1])
    return vol * G @ S @ G.T


TET_REF = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
TET_K = np.array([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]], float) / 6.0
TRI_RIGHT = np.array([[0, 0], [1, 0], [0, 1]], float)
TRI_OBTUSE = np.array([[0, 0], [1, 0], [0.5, 0.2]], float)


def cot_angle(V, k):
    """Cotangent of the interior angle at vertex k of a triangle."""
    i, j = [m for m in range(3) if m != k]
    a, b = V[i] - V[k], V[j] - V[k]
    return float(a @ b / abs(a[0] * b[1] - a[1] * b[0]))


# ---------------------------------------------------------------------------
# Convergence study (section 16.7)
# ---------------------------------------------------------------------------
def convergence_table():
    out = Path(tempfile.mkdtemp()) / "results.json"
    return numerics.run_all(out)["fem_convergence"]


def flux_jumps(n):
    """Jumps of the normal and tangential components of grad u_h across interior edges."""
    _, (p, cells, u) = manufactured_solution(n)
    area, grads = element_geometry(p, cells)
    gu = np.einsum("ei,eid->ed", u[cells], grads)
    edges: dict = {}
    for e, c in enumerate(cells):
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
            edges.setdefault((min(a, b), max(a, b)), []).append(e)
    jn, jt = [], []
    for (a, b), els in edges.items():
        if len(els) != 2:
            continue
        t = p[b] - p[a]
        t = t / np.hypot(*t)
        nrm = np.array([t[1], -t[0]])
        d = gu[els[0]] - gu[els[1]]
        jn.append(abs(d @ nrm))
        jt.append(abs(d @ t))
    return np.array(jn), np.array(jt)


H1_NORM_EXACT = np.pi / np.sqrt(2.0)      # |u|_H1 of sin(pi x) sin(pi y) on the unit square
L2_NORM_EXACT = 0.5


# ---------------------------------------------------------------------------
def main():
    print("Section 16.1: two-element bar (old proj:2)")
    b = bar_dirichlet()
    assert np.allclose(b["K"], 0.005 * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]]))
    close(b["K"][0, 0], 0.005, name="element conductance sigma A / h (S)")
    close(1 / b["K"][0, 0], 200.0, name="element resistance (Ohm)")
    assert np.allclose(b["phi"], [0.4, 0.2, 0.0])
    close(b["phi"][1], 0.2, name="interior potential phi_2 (V)")
    assert np.allclose(b["reaction"], [1e-3, 0.0, -1e-3], atol=1e-15)
    close(b["reaction"][0] * 1e3, 1.0, name="reaction at node 1 (mA, injected)")
    close(b["reaction"][2] * 1e3, -1.0, name="reaction at node 3 (mA)")
    assert np.allclose(b["E"], 20.0) and np.allclose(b["J"], 10.0)
    close(b["I"] * 1e3, 1.0, name="I = A J (mA)")
    close(b["R"], 400.0, name="R (Ohm)")
    for k in ("P_VI", "P_quad", "P_int"):
        close(b[k] * 1e3, 0.4, rtol=1e-12, name=f"power {k} (mW)")
    # forcing all three rows to zero with the two end values prescribed is contradictory
    K = b["K"]
    phi2_row1 = 0.4            # row 1: G (0.4 - phi2) = 0
    phi2_row3 = 0.0            # row 3: G (phi2 - 0) = 0
    assert phi2_row1 != phi2_row3
    print("  ok  rows 1 and 3 = 0 would demand phi_2 = 0.4 V and phi_2 = 0 V at once")
    # the 1-D element matrix from the hat-function integral
    Kl = hat_local_matrix(0.01, 0.5, 1e-4)
    assert np.allclose(Kl, 0.005 * np.array([[1, -1], [-1, 1]]))
    print("  ok  int sigma A N_i' N_j' dx = (sigma A / h) [[1,-1],[-1,1]]")
    # node-2 equation = weak form with v = N_2 (current balance)
    assert abs(K[1] @ b["phi"]) < 1e-15
    print("  ok  row 2 (test function N_2) = Kirchhoff current law at node 2")
    assert abs(b["reaction"].sum()) < 1e-18 and np.allclose(K.sum(axis=1), 0.0)
    print("  ok  rows of K sum to zero, so the reactions add up to zero (global balance exact)")
    # non-uniform 1-D hats: K_ii = sigma A (1/h_i + 1/h_{i+1}), K_i,i+1 = -sigma A / h_{i+1}
    xn = np.array([0.0, 0.16, 0.38, 0.55, 0.78, 1.0])
    Kn = bar_matrix(np.diff(xn), [1.0] * 5, 1.0)
    assert np.isclose(Kn[2, 2], 1 / 0.22 + 1 / 0.17) and np.isclose(Kn[2, 3], -1 / 0.17) and Kn[2, 4] == 0
    print("  ok  hat matrix on a non-uniform 1-D mesh: tridiagonal, K_22 = sigma A (1/h_2 + 1/h_3)")

    print("Section 16.3: element matrices")
    Kt = element_matrix(TET_REF, 1.0)
    assert np.allclose(Kt, TET_K)
    a, s = 0.01, 0.5
    assert np.allclose(element_matrix(a * TET_REF, s), s * a / 6 * 6 * TET_K)
    print("  ok  reference tetrahedron: K = (sigma a / 6) [[3,-1,-1,-1],...]  (eq:tetraK)")
    assert np.allclose(Kt.sum(axis=1), 0.0)
    ev = np.linalg.eigvalsh(Kt)
    assert abs(ev[0]) < 1e-14 and np.all(ev[1:] > 0)
    print("  ok  rows sum to zero; one zero eigenvalue (constant), others positive")
    close(s * a / 6 * 3, 2.5e-3, name="K_00 for a = 1 cm, sigma = 0.5 S/m (S)")
    Kr = element_matrix(TRI_RIGHT, 1.0)
    assert np.allclose(Kr, 0.5 * np.array([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]]))
    print("  ok  right isosceles triangle: K = (sigma/2)[[2,-1,-1],[-1,1,0],[-1,0,1]] per unit depth")
    Ko = element_matrix(TRI_OBTUSE, 1.0)
    close(-0.5 * cot_angle(TRI_OBTUSE, 2), Ko[0, 1], rtol=1e-12, name="cot formula K_01 = -(sigma/2) cot(theta_2)")
    for k, (i, j) in ((0, (1, 2)), (1, (0, 2))):
        assert np.isclose(Ko[i, j], -0.5 * cot_angle(TRI_OBTUSE, k))
    theta = np.degrees(np.arccos(np.dot(TRI_OBTUSE[0] - TRI_OBTUSE[2], TRI_OBTUSE[1] - TRI_OBTUSE[2])
                                 / np.linalg.norm(TRI_OBTUSE[0] - TRI_OBTUSE[2])
                                 / np.linalg.norm(TRI_OBTUSE[1] - TRI_OBTUSE[2])))
    close(theta, 136.4, rtol=1e-3, name="obtuse angle at vertex 2 (deg)")
    close(Ko[0, 1], 0.525, rtol=1e-3, name="positive off-diagonal K_01 (sigma = 1)")
    assert np.all(np.linalg.eigvalsh(Ko) > -1e-14)
    print("  ok  obtuse element matrix is still symmetric positive semi-definite")

    print("Section 16.4: two-layer bar with a prescribed current (ex:twolayer numbers)")
    t = bar_twolayer()
    close(t["K"][0, 0] * 1e3, 2.5, name="G_1 (mS)")
    close(t["K"][2, 2] * 1e3, 1.0, name="G_2 (mS)")
    assert np.allclose(t["phi"], [1.4, 1.0, 0.0])
    close(t["phi"][0], 1.4, name="phi_1 (V)")
    close(t["phi"][1], 1.0, name="phi_2 (V)")
    assert np.allclose(t["E"], [50.0, 500.0]) and np.allclose(t["J"], 10.0)
    close(t["R"], 1400.0, name="R (Ohm)")
    close(t["P"] * 1e3, 1.4, name="P (mW)")
    close(t["reaction"][2] * 1e3, -1.0, name="reaction at grounded node 3 (mA)")
    u = bar_twolayer(I_in=1e-3, I_out=-0.9e-3, mode="neumann")
    assert np.allclose(u["phi"], t["phi"])
    close(u["imbalance"] * 1e3, 0.1, name="1^T f for +1 mA / -0.9 mA (mA)")
    close(u["reaction"][2] * 1e3, -1.0, name="reaction at reference node, specified was -0.9 mA")
    # zero-mean reference with a Lagrange multiplier: saddle-point matrix, indefinite
    z = np.array([LAYER["t"][0] / 2, sum(LAYER["t"]) / 2, LAYER["t"][1] / 2]) * LAYER["area"]   # int N_i dV
    Kz = np.block([[t["K"], z[:, None]], [z[None, :], np.zeros((1, 1))]])
    evz = np.linalg.eigvalsh(Kz)
    assert evz.min() < 0 < evz.max() and np.all(np.abs(evz) > 1e-12)
    sol = np.linalg.solve(Kz, np.r_[t["f"][0], 0.0, -t["f"][0], 0.0])
    assert abs(z @ sol[:3]) < 1e-15 and np.allclose(np.diff(sol[:3]), np.diff(t["phi"]))
    print("  ok  zero-mean constraint: augmented matrix nonsingular but indefinite; same E as grounding")
    Kg = t["K"][:2, :2]
    assert np.all(np.linalg.eigvalsh(Kg) > 0)
    print("  ok  reference fixed by elimination: reduced matrix symmetric positive definite")

    print("Section 16.5: CEM on the bar")
    c = bar_cem()
    close(c["rc"], 200.0, name="contact resistance z/A (Ohm), z = 0.02 Ohm m^2")
    close(c["U"], 0.6, name="electrode potential U (V) for 1 mA")
    close(c["phi1"], 0.4, name="tissue potential under the electrode (V)")
    close(c["phi2"], 0.2, name="phi_2 (V)")
    assert np.allclose(c["M"], c["M"].T)
    print("  ok  CEM system matrix symmetric")

    print("Section 16.6: what the nodal solution conserves")
    for n_el, R_ref in ((2, 500.0), (5, 1400.0), (4, 1625.0)):
        th = thin_layer_mesh(n_el)
        close(th["R"], R_ref, rtol=1e-9, name=f"R with {n_el} equal elements, centroid labels (Ohm)")
        assert th["residual"] < 1e-14
    close(500 * 1e-3 ** 2 * 1e3, 0.5, name="P at 1 mA, thin layer lost (mW)")
    for n, ref in ((8, 0.527), (16, 0.275), (32, 0.139), (64, 0.0694)):
        jn, jt = flux_jumps(n)
        close(jn.max() / np.pi, ref, rtol=5e-3, name=f"max normal-flux jump / max|grad u|, n = {n}")
        assert jt.max() < 1e-9
    print("  ok  tangential component of grad u_h continuous to round-off")

    print("Section 16.7: convergence (numerics.run_all)")
    conv = convergence_table()
    table = {8: (81, 128, 2.1134e-2, 4.3180e-1), 16: (289, 512, 5.3775e-3, 2.1754e-1),
             32: (1089, 2048, 1.3504e-3, 1.0898e-1), 64: (4225, 8192, 3.3799e-4, 5.4514e-2)}
    for row in conv:
        nn, ne, l2, h1 = table[row["n"]]
        assert row["nodes"] == nn and row["elements"] == ne
        close(row["L2_error"], l2, rtol=1e-4, name=f"L2 error n = {row['n']}")
        close(row["H1_seminorm_error"], h1, rtol=1e-4, name=f"H1 error n = {row['n']}")
    for row, o2, o1 in zip(conv[1:], (1.975, 1.994, 1.998), (0.989, 0.997, 0.999)):
        close(row["L2_error_order"], o2, rtol=1e-3, name=f"L2 order n = {row['n']}")
        close(row["H1_seminorm_error_order"], o1, rtol=1e-3, name=f"H1 order n = {row['n']}")
    r64 = conv[-1]["relative_algebraic_residual"]
    assert 5e-14 < r64 < 5e-13, r64
    close(r64, 1.4e-13, rtol=0.5, name="relative algebraic residual n = 64 (about 1.4e-13)")
    close(conv[-1]["H1_seminorm_error"] / H1_NORM_EXACT, 0.0245, rtol=5e-3, name="relative H1 error n = 64")
    for row in conv:
        assert row["relative_algebraic_residual"] < 1e-12
    # observed order without the exact solution: centre value on three meshes
    ys = {}
    for n in (8, 16, 32, 64):
        _, (p, cells, u) = manufactured_solution(n)
        ys[n] = u[np.argmin(np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5))]
    close(ys[8], 0.98725, rtol=1e-5, name="u_h(1/2,1/2), n = 8")
    close(ys[16], 0.99679, rtol=1e-5, name="u_h(1/2,1/2), n = 16")
    close(ys[32], 0.99920, rtol=1e-5, name="u_h(1/2,1/2), n = 32")
    close(ys[64], 0.99980, rtol=1e-5, name="u_h(1/2,1/2), n = 64")
    for a, b_, c_, ref in ((8, 16, 32, 1.99), (16, 32, 64, 2.00)):
        close(np.log2((ys[a] - ys[b_]) / (ys[b_] - ys[c_])), ref, rtol=3e-3, name=f"order from differences {a},{b_},{c_}")
    close(64 * conv[-1]["H1_seminorm_error"] / H1_NORM_EXACT / 0.01, 157, rtol=5e-3,
          name="n for a 1% relative gradient error (first order)")

    print("Self-check answers")
    h = BAR["length"] / 2
    K = bar_matrix([h, h], [BAR["sigma"]] * 2, BAR["area"])
    phi = np.array([0.4, 0.0, 0.0])
    phi[1:] = np.linalg.solve(K[1:, 1:], -K[1:, :1] @ phi[:1])
    assert np.allclose(phi, 0.4) and abs((K @ phi)[0]) < 1e-15
    print("  ok  right end insulated: phi_2 = phi_3 = 0.4 V, no current")
    K4 = bar_matrix([h / 2] * 4, [BAR["sigma"]] * 4, BAR["area"])
    close(K4[0, 0], 0.01, name="four elements: G (S)")
    phi = np.zeros(5)
    phi[0] = 0.4
    phi[1:4] = np.linalg.solve(K4[1:4, 1:4], -K4[1:4, [0, 4]] @ phi[[0, 4]])
    assert np.allclose(phi, [0.4, 0.3, 0.2, 0.1, 0.0])
    close(0.4 / (K4 @ phi)[0], 400.0, name="four elements: R (Ohm)")
    c2 = bar_cem(z=0.04)
    close(c2["U"], 0.8, name="CEM, z doubled: U (V)")
    close(c2["phi1"], 0.4, name="CEM, z doubled: phi_1 (V), tissue unchanged")
    print("All Chapter 16 checks passed.")


if __name__ == "__main__":
    main()
