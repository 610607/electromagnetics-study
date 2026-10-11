"""Chapter 17 (vector finite elements, nonlinear solution): models and checks for every number in
zh/ch17_femvector.tex.

    python3 checks/check_ch17.py          # run all assertions, print the quoted values

figsrc/fig_ch17.py imports the model functions from this file.  Only NumPy, SciPy and numerics.py
are used.  The nonlinear circuit is the single-loop magnet / gap / yoke circuit of Ch 14
(eq:nonlinearmagcircuit, numerics.magnetic_circuit) with the synthetic B-H curve of Ch 13
(eq:syntheticbh).  All values are teaching inputs.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from numerics import MU0, magnetic_circuit, synthetic_B

BS, HS = 1.6, 800.0


def close(x, ref, rtol=5e-3, name="", atol=0.0):
    assert np.isclose(x, ref, rtol=rtol, atol=atol), f"{name}: {x} != {ref}"
    print(f"  ok  {name:62s} {x:.6g}")
    return x


# ---------------------------------------------------------------------------
# Whitney (lowest-order Nedelec) functions on a simplex
# ---------------------------------------------------------------------------
def bary_grads(V):
    """Rows: gradients of the barycentric functions; also the coefficient matrix C (N = [1,x] @ C)."""
    V = np.asarray(V, float)
    M = np.hstack([np.ones((len(V), 1)), V])
    C = np.linalg.inv(M)
    return C[1:, :].T, C


def bary(V, x):
    _, C = bary_grads(V)
    x = np.atleast_2d(x)
    return np.hstack([np.ones((len(x), 1)), x]) @ C


def whitney(V, i, j, x):
    """W_ij = N_i grad N_j - N_j grad N_i at points x (eq:edgebasis)."""
    G, _ = bary_grads(V)
    N = bary(V, x)
    return N[:, [i]] * G[j] - N[:, [j]] * G[i]


def line_integral(fun, a, b, nq=6):
    xg, wg = np.polynomial.legendre.leggauss(nq)
    s = 0.5 * (xg + 1)
    pts = a + s[:, None] * (b - a)
    vals = fun(pts)
    return float(0.5 * np.sum(wg * (vals @ (b - a))))


TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.85]])       # triangle of fig. edge_vs_nodal
TRI_EDGES = [(0, 1), (1, 2), (2, 0)]
TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TET_FACES = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]     # face k is opposite vertex k


T2 = np.array([[0.0, 0.0], [1.2, 0.0], [0.4, 0.75], [0.95, -0.65]])   # triangles (0,1,2) above, (1,0,3) below
T2_TRIS = [(0, 1, 2), (1, 0, 3)]
SIG_RATIO = 4.0                                   # sigma_2 / sigma_1 (lower / upper triangle)
E_UP = np.array([0.6, 1.0])                       # (x = tangential, y = normal) field in the upper triangle
E_LO = np.array([0.6, 1.0 / SIG_RATIO])           # lower triangle: sigma1 E1n = sigma2 E2n, E1t = E2t


def interface_field(x):
    """Exact piecewise-constant interface field on T2 (the shared edge lies on y = 0)."""
    x = np.atleast_2d(x)
    return np.where(x[:, [1]] >= 0, E_UP, E_LO)


def nodal_interface(x):
    """Continuous nodal (P1 vector) interpolant: one vector per vertex, shared vertices take the
    average of the two sides (the best a single nodal value can do)."""
    x = np.atleast_2d(x)
    vals = {0: 0.5 * (E_UP + E_LO), 1: 0.5 * (E_UP + E_LO), 2: E_UP, 3: E_LO}
    out = np.zeros((len(x), 2))
    for k, p in enumerate(x):
        tri = T2_TRIS[0] if p[1] >= 0 else T2_TRIS[1]
        lam = bary(T2[list(tri)], p)[0]
        out[k] = sum(l * vals[v] for l, v in zip(lam, tri))
    return out


def edge_interface(x):
    """Edge (Whitney) representation of the interface field: degrees of freedom are the line
    integrals of the field along the edges, the field is sum_e a_e W_e."""
    x = np.atleast_2d(x)
    out = np.zeros((len(x), 2))
    for k, p in enumerate(x):
        tri = T2_TRIS[0] if p[1] >= 0 else T2_TRIS[1]
        V = T2[list(tri)]
        for i, j in TRI_EDGES:
            a = line_integral(lambda y: interface_field(y), V[i], V[j])
            out[k] += a * whitney(V, i, j, p)[0]
    return out


def incidence_G(edges, nv):
    G = np.zeros((len(edges), nv))
    for e, (a, b) in enumerate(edges):
        G[e, a], G[e, b] = -1.0, 1.0
    return G


def incidence_C(faces, edges):
    """Face-edge incidence: +1 if the edge runs along the face's cyclic orientation (a->b->c->a)."""
    idx = {e: k for k, e in enumerate(edges)}
    C = np.zeros((len(faces), len(edges)))
    for f, (a, b, c) in enumerate(faces):
        for p, q in ((a, b), (b, c), (c, a)):
            if (p, q) in idx:
                C[f, idx[(p, q)]] += 1.0
            else:
                C[f, idx[(q, p)]] -= 1.0
    return C


def tet_curlcurl(V, nu=1.0, edges=TET_EDGES):
    """Element curl-curl matrix with constant curls 2 grad N_i x grad N_j."""
    import math
    G, _ = bary_grads(V)
    vol = abs(np.linalg.det(np.hstack([np.ones((4, 1)), V]))) / math.factorial(3)
    curls = np.array([2 * np.cross(G[i], G[j]) for i, j in edges])
    return vol * nu * curls @ curls.T, curls, vol


# ---------------------------------------------------------------------------
# A small structured tetrahedral mesh of the unit cube (each cube -> 6 tetrahedra)
# ---------------------------------------------------------------------------
def cube_mesh(m):
    g = np.linspace(0, 1, m + 1)
    P = np.array([[x, y, z] for z in g for y in g for x in g])

    def vid(i, j, k):
        return i + (m + 1) * (j + (m + 1) * k)

    tets = []
    for i, j, k in itertools.product(range(m), repeat=3):
        c = [vid(i + a, j + b, k + d) for d in (0, 1) for b in (0, 1) for a in (0, 1)]   # x fastest
        v0, v7 = c[0], c[7]
        for perm in itertools.permutations((0, 1, 2)):            # Kuhn (Freudenthal) subdivision
            cur = [0, 0, 0]
            path = [v0]
            for ax in perm:
                cur[ax] = 1
                path.append(c[cur[0] + 2 * cur[1] + 4 * cur[2]])
            tets.append(path)
    tets = np.array(tets)
    edges = sorted({tuple(sorted((t[a], t[b]))) for t in tets for a, b in TET_EDGES})
    faces = sorted({tuple(sorted(f)) for t in tets for f in ([t[1], t[2], t[3]], [t[0], t[2], t[3]],
                                                               [t[0], t[1], t[3]], [t[0], t[1], t[2]])})
    return P, tets, edges, faces


def global_curlcurl(P, tets, edges):
    idx = {e: k for k, e in enumerate(edges)}
    K = np.zeros((len(edges), len(edges)))
    for t in tets:
        Ke, _, _ = tet_curlcurl(P[t])
        gl, sg = [], []
        for a, b in TET_EDGES:
            ga, gb = t[a], t[b]
            gl.append(idx[(min(ga, gb), max(ga, gb))])
            sg.append(1.0 if ga < gb else -1.0)      # global edge orientation: low -> high index
        sg = np.array(sg)
        K[np.ix_(gl, gl)] += sg[:, None] * Ke * sg[None, :]
    return K


def spanning_tree(nv, edges):
    """Breadth-first spanning tree of the edge graph; returns indices of tree edges."""
    adj = {v: [] for v in range(nv)}
    for k, (a, b) in enumerate(edges):
        adj[a].append((b, k))
        adj[b].append((a, k))
    seen, tree, queue = {0}, [], [0]
    while queue:
        v = queue.pop(0)
        for w, k in adj[v]:
            if w not in seen:
                seen.add(w)
                tree.append(k)
                queue.append(w)
    return tree


# ---------------------------------------------------------------------------
# Incremental reluctivity of an isotropic curve (eq:inctensor)
# ---------------------------------------------------------------------------
def mu_diff(H):
    return MU0 + BS / HS / np.cosh(np.clip(np.asarray(H, float) / HS, -300, 300)) ** 2


def h_of_b(b):
    """Inverse of eq:syntheticbh by safeguarded Newton iteration (b >= 0)."""
    H, lo, hi = 0.0, -1e9, 1e9
    for _ in range(300):
        r = synthetic_B(H) - b
        if r > 0:
            hi = min(hi, H)
        else:
            lo = max(lo, H)
        Hn = H - r / mu_diff(H)
        if not (lo < Hn < hi):
            Hn = 0.5 * (lo + hi)
        if abs(Hn - H) <= 1e-15 * max(1.0, abs(H)):
            return Hn
        H = Hn
    return H


def H_of_B(Bvec):
    """eq:HofB: H = h(|B|) B/|B|."""
    Bvec = np.asarray(Bvec, float)
    b = np.linalg.norm(Bvec)
    if b == 0:
        return Bvec / mu_diff(0.0)
    return h_of_b(b) * Bvec / b


def nu_inc(Bvec):
    Bvec = np.asarray(Bvec, float)
    b = np.linalg.norm(Bvec)
    bh = Bvec / b
    H = h_of_b(b)
    P = np.outer(bh, bh)
    return H / b * (np.eye(len(Bvec)) - P) + 1.0 / mu_diff(H) * P


# ---------------------------------------------------------------------------
# Nonlinear magnet circuit (eq:nonlinearmagcircuit), flux Phi as unknown (A-formulation analogue)
# ---------------------------------------------------------------------------
PM = dict(Br=1.2, mu_rec=1.05, lm=0.05, g=0.02, ly=0.40, Am=0.002, Ag=0.002, Ay=0.001)


def pm_parts(p=PM):
    Rm = p["lm"] / (MU0 * p["mu_rec"] * p["Am"])
    Rg = p["g"] / (MU0 * p["Ag"])
    Fc = p["Br"] * p["lm"] / (MU0 * p["mu_rec"])
    return Rm, Rg, Fc


def pm_residual(Phi, F_extra=0.0, Fc_scale=1.0, p=PM):
    Rm, Rg, Fc = pm_parts(p)
    return (Rm + Rg) * Phi + p["ly"] * h_of_b(Phi / p["Ay"]) - Fc_scale * Fc - F_extra


def yoke_reluctances(Phi, p=PM):
    b = Phi / p["Ay"]
    H = h_of_b(b)
    R_sec = p["ly"] * (H / b) / p["Ay"] if b != 0 else p["ly"] / (mu_diff(0.0) * p["Ay"])
    R_diff = p["ly"] / (mu_diff(H) * p["Ay"])
    return R_sec, R_diff


def newton(F_extra=0.0, Phi0=0.0, iters=8, Fc_scale=1.0, p=PM):
    Rm, Rg, _ = pm_parts(p)
    Phi, hist = Phi0, []
    for _ in range(iters):
        r = pm_residual(Phi, F_extra, Fc_scale, p)
        hist.append((Phi, r))
        _, Rd = yoke_reluctances(Phi, p)
        Phi = Phi - r / (Rm + Rg + Rd)
    hist.append((Phi, pm_residual(Phi, F_extra, Fc_scale, p)))
    return Phi, hist


def newton_solve(F_extra=0.0, Fc_scale=1.0, p=PM):
    Phi, _ = newton(F_extra, 0.0, 40, Fc_scale, p)
    return Phi


def picard(omega=1.0, Phi0=0.0, iters=40, Fc_scale=1.0, p=PM):
    """Secant-reluctivity (Picard) iteration with optional under-relaxation omega."""
    Rm, Rg, Fc = pm_parts(p)
    Phi, hist = Phi0, []
    for _ in range(iters):
        r = pm_residual(Phi, 0.0, Fc_scale, p)
        Rs, _ = yoke_reluctances(Phi, p)
        Phi_lin = Fc_scale * Fc / (Rm + Rg + Rs)      # linear solve with frozen secant reluctivity
        new = Phi + omega * (Phi_lin - Phi)
        hist.append((Phi, r, abs(new - Phi) / max(abs(new), 1e-30)))
        Phi = new
    return Phi, hist


def picard_factor(Phi, p=PM):
    """Local error factor of undamped Picard: 1 - (R0 + R_diff)/(R0 + R_sec)."""
    Rm, Rg, _ = pm_parts(p)
    Rs, Rd = yoke_reluctances(Phi, p)
    return 1.0 - (Rm + Rg + Rd) / (Rm + Rg + Rs)


# ---------------------------------------------------------------------------
def main():
    print("Section 17.1: Whitney functions on a triangle")
    Gt, _ = bary_grads(TRI)
    area = 0.5 * abs(np.linalg.det(np.hstack([np.ones((3, 1)), TRI])))
    for e, (i, j) in enumerate(TRI_EDGES):
        for f, (a, b) in enumerate(TRI_EDGES):
            val = line_integral(lambda x: whitney(TRI, i, j, x), TRI[a], TRI[b])
            assert np.isclose(val, 1.0 if e == f else 0.0, atol=1e-13), (e, f, val)
        curl = 2 * (Gt[i][0] * Gt[j][1] - Gt[i][1] * Gt[j][0])
        assert np.isclose(curl * area, 1.0)
    print("  ok  int_e W_f . dl = delta_ef on all three edges; curl W = 1/area (counter-clockwise edges)")
    rng = np.random.default_rng(1)
    pts = bary(TRI, np.zeros((1, 2)))  # noqa: F841 (shape check)
    w = rng.dirichlet(np.ones(3), 50)
    X = w @ TRI
    for i in range(3):
        s = sum(whitney(TRI, j, i, X) for j in range(3) if j != i)
        assert np.allclose(s, Gt[i])
    print("  ok  grad N_i = sum_j W_ji (gradients lie in the edge space)")
    # tangential continuity of an edge function across a shared edge, normal jump
    mid = 0.5 * (T2[0] + T2[1])
    Wa = whitney(T2[[0, 1, 2]], 0, 1, mid)[0]
    Wb = whitney(T2[[0, 1, 3]], 0, 1, mid)[0]
    t = (T2[1] - T2[0]) / np.linalg.norm(T2[1] - T2[0])
    n = np.array([-t[1], t[0]])
    assert np.isclose(Wa @ t, Wb @ t) and not np.isclose(Wa @ n, Wb @ n)
    print("  ok  shared-edge function: tangential component equal, normal component different")
    # interface field E with sigma_2 = 4 sigma_1 (eq:einterface): exactly representable by edges
    E1 = np.array([1.0, 0.6])            # in the frame (normal, tangential) of the shared edge
    E2 = np.array([0.25, 0.6])
    assert np.isclose(1.0 * E1[0], 4.0 * E2[0]) and np.isclose(E1[1], E2[1])
    print("  ok  interface example: sigma1 E1n = sigma2 E2n, E1t = E2t (sigma2/sigma1 = 4)")
    # the same field on the two-triangle patch T2 (fig. edge_vs_nodal)
    rng2 = np.random.default_rng(5)
    pts = np.vstack([rng2.dirichlet(np.ones(3), 40) @ T2[list(t)] for t in T2_TRIS])
    assert np.allclose(edge_interface(pts), interface_field(pts), atol=1e-12)
    print("  ok  edge (Whitney) representation reproduces the piecewise-constant interface field exactly")
    on_edge = np.array([[0.2, 0.0], [0.6, 0.0], [1.0, 0.0]])
    nod_up = nodal_interface(on_edge + [0, 1e-12])[:, 1]
    nod_lo = nodal_interface(on_edge - [0, 1e-12])[:, 1]
    assert np.allclose(nod_up, nod_lo)
    close(nod_up[1], 0.625, rtol=1e-12, name="nodal interpolant: normal component on the interface")
    close(E_UP[1], 1.0, rtol=1e-12, name="exact normal component, low-sigma side")
    close(E_LO[1], 0.25, rtol=1e-12, name="exact normal component, high-sigma side")

    print("Section 17.1: one tetrahedron")
    V = np.array([[0, 0, 0], [1.6, 0, 0], [0.4, 1.5, 0], [0.55, 0.5, 1.4]], float)   # fig_3d tetra
    G = incidence_G(TET_EDGES, 4)
    C = incidence_C([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)], TET_EDGES)        # outward orientation
    D = np.ones((1, 4))
    assert np.allclose(C @ G, 0) and np.allclose(D @ C, 0)
    print("  ok  C G = 0 and D C = 0 on a tetrahedron (outward-oriented faces)")
    Ke, curls, vol = tet_curlcurl(V)
    assert np.allclose(Ke @ G, 0, atol=1e-12)
    rank = np.linalg.matrix_rank(Ke, tol=1e-10)
    assert rank == 3
    print("  ok  element curl-curl matrix: 6 x 6, rank 3, null space = range(G) (gradients)")
    # face fluxes of curl W equal C (Stokes): flux of curl W_e through face f
    Gb, _ = bary_grads(V)
    for f, (a, b, c) in enumerate([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]):
        nA = 0.5 * np.cross(V[b] - V[a], V[c] - V[a])
        flux = curls @ nA
        assert np.allclose(flux, C[f]), (f, flux, C[f])
    print("  ok  flux of curl W_e through each face = face-edge incidence (exact discrete Stokes)")

    print("Section 17.3: gauge null space on a cube mesh")
    for m in (2, 3):
        P, tets, edges, faces = cube_mesh(m)
        Nv, Ne, Nf, Nt = len(P), len(edges), len(faces), len(tets)
        assert Nv - Ne + Nf - Nt == 1
        K = global_curlcurl(P, tets, edges)
        ev = np.linalg.eigvalsh(K)
        nullity = int(np.sum(ev < 1e-9 * ev.max()))
        assert nullity == Nv - 1, (nullity, Nv)
        Gg = incidence_G(edges, Nv)
        assert np.allclose(K @ Gg, 0, atol=1e-10)
        tree = spanning_tree(Nv, edges)
        assert len(tree) == Nv - 1
        keep = np.setdiff1d(np.arange(Ne), tree)
        evr = np.linalg.eigvalsh(K[np.ix_(keep, keep)])
        assert evr.min() > 1e-9 * evr.max()
        print(f"  ok  m = {m}: nodes {Nv}, edges {Ne}, faces {Nf}, tets {Nt}; nullity {nullity} = Nv - 1; "
              f"tree gauge ({len(tree)} edges) -> nonsingular")
        # essential condition n x A = 0 on the whole boundary: remove the boundary edges
        onb = np.any((P == 0) | (P == 1), axis=1)
        bedge = np.array([onb[a] and onb[b] and np.any((P[a] == P[b]) & ((P[a] == 0) | (P[a] == 1)))
                          for a, b in edges])
        inner = np.flatnonzero(~bedge)
        evi = np.linalg.eigvalsh(K[np.ix_(inner, inner)])
        null_i = int(np.sum(evi < 1e-9 * evi.max()))
        assert null_i == int(np.sum(~onb)), (null_i, int(np.sum(~onb)))
        print(f"  ok  m = {m}, n x A = 0 on the boundary: nullity {null_i} = number of interior nodes")
        if m == 2:
            Q = dict(Nv=Nv, Ne=Ne, Nf=Nf, Nt=Nt, nullity=nullity, null_bc=null_i)
    close(Q["Nv"], 27, name="m = 2 cube mesh: nodes")
    close(Q["Ne"], 98, name="m = 2 cube mesh: edges")
    close(Q["Nt"], 48, name="m = 2 cube mesh: tetrahedra")
    close(Q["nullity"], 26, name="m = 2 cube mesh: dimension of the null space")
    # compatibility: (G^T f)_j = int grad N_j . J_s ; zero for J_s = curl T_h with n x T = 0
    P, tets, edges, faces = cube_mesh(2)
    idx = {e: k for k, e in enumerate(edges)}
    on_bnd = np.any((P == 0) | (P == 1), axis=1)
    bnd_edge = np.array([on_bnd[a] and on_bnd[b] and np.any((P[a] == P[b]) & ((P[a] == 0) | (P[a] == 1)))
                         for a, b in edges])
    rng = np.random.default_rng(7)
    tcoef = np.where(bnd_edge, 0.0, rng.normal(size=len(edges)))
    Gg = incidence_G(edges, len(P))

    def load(Jfun):
        f = np.zeros(len(edges))
        for t in tets:
            Gb_, _ = bary_grads(P[t])
            import math
            vol = abs(np.linalg.det(np.hstack([np.ones((4, 1)), P[t]]))) / math.factorial(3)
            J = Jfun(t, Gb_)
            for a, b in TET_EDGES:
                ga, gb = t[a], t[b]
                s = 1.0 if ga < gb else -1.0
                intW = vol / 4 * (Gb_[b] - Gb_[a])            # int_T W_ab dV
                f[idx[(min(ga, gb), max(ga, gb))]] += s * intW @ J
        return f

    def curlT(t, Gb_):
        J = np.zeros(3)
        for a, b in TET_EDGES:
            ga, gb = t[a], t[b]
            s = 1.0 if ga < gb else -1.0
            J += s * tcoef[idx[(min(ga, gb), max(ga, gb))]] * 2 * np.cross(Gb_[a], Gb_[b])
        return J

    f1 = load(curlT)
    f2 = load(lambda t, Gb_: np.array([1.0, 0.0, 0.0]))
    assert np.abs(Gg.T @ f1).max() < 1e-12 * np.abs(f1).max()
    assert np.abs(Gg.T @ f2).max() > 1e-3
    print("  ok  G^T f = 0 for a discretely divergence-free source (curl of an edge field); not for J = const")
    # tree gauge: solve on the co-tree edges only, then look at ALL equations (like Ch 16 reactions)
    K = global_curlcurl(P, tets, edges)
    tree = spanning_tree(len(P), edges)
    keep = np.setdiff1d(np.arange(len(edges)), tree)
    for f, compatible in ((f1, True), (f2, False)):
        a = np.zeros(len(edges))
        a[keep] = np.linalg.solve(K[np.ix_(keep, keep)], f[keep])
        res = np.linalg.norm(K @ a - f) / np.linalg.norm(f)
        assert (res < 1e-10) == compatible, (compatible, res)
        if not compatible:
            assert res > 0.1
            print(f"  ok  tree gauge, J = const: relative residual of the full system {res:.2f} (not ~0)")
    print("  ok  tree gauge: compatible source satisfies all rows; J = const leaves the tree rows unsatisfied")
    # 2-D A_z with nodal elements: B = grad A_z x e_z, so B.n across an edge = tangential derivative of A_z
    import check_ch16 as C16
    for n in (8, 32):
        jn, jt = C16.flux_jumps(n)
        assert jt.max() < 1e-9 and jn.max() > 1e-2
    print("  ok  nodal A_z: normal B continuous across every edge (tangential B jumps, like normal E in Ch 16)")

    print("Section 17.5: incremental reluctivity tensor")
    H0 = 1600.0
    B0 = synthetic_B(H0)
    close(B0, 1.544, rtol=1e-3, name="B0 at H0 = 1600 A/m (T)")
    mus, mud = B0 / H0 / MU0, mu_diff(H0) / MU0
    close(mus, 768, rtol=2e-3, name="mu_sec / mu0")
    close(mud, 113, rtol=5e-3, name="mu_diff / mu0")
    Bv = B0 * np.array([np.cos(0.4), np.sin(0.4), 0.0])
    Nu = nu_inc(Bv)
    eps = 1e-7
    J = np.column_stack([(H_of_B(Bv + eps * e) - H_of_B(Bv - eps * e)) / (2 * eps) for e in np.eye(3)])
    assert np.allclose(J, Nu, rtol=1e-5, atol=1e-6 * np.abs(Nu).max())
    print("  ok  eq:inctensor equals the finite-difference Jacobian of eq:HofB")
    ev = np.sort(np.linalg.eigvalsh(Nu))
    assert np.allclose(ev, np.sort([1 / (mus * MU0), 1 / (mus * MU0), 1 / (mud * MU0)]), rtol=1e-6)
    print("  ok  eigenvalues: 1/mu_sec (twice, transverse), 1/mu_diff (along B)")
    bh = Bv / B0
    alt = (1 / (mus * MU0)) * np.eye(3) + (1 / (mud * MU0) - 1 / (mus * MU0)) * np.outer(bh, bh)
    assert np.allclose(alt, Nu)
    print("  ok  same tensor as nu_sec I + (nu_diff - nu_sec) b b^T (Ch 13 form)")
    close(1 - mus / mud, -5.8, rtol=2e-2, name="secant tangent, error factor along B: 1 - mu_sec/mu_diff")
    close(1 - mud / mus, 0.853, rtol=2e-3, name="isotropic mu_diff tangent, transverse factor")

    print("Section 17.6: Picard and Newton on the magnet circuit")
    ref = magnetic_circuit()
    Rm, Rg, Fc = pm_parts()
    close(Rm, 1.89e7, rtol=3e-3, name="R_m (A/Wb)")
    close(Rg, 7.96e6, rtol=3e-3, name="R_g (A/Wb)")
    close(Fc, 4.55e4, rtol=3e-3, name="F_c (A)")
    Phi_N, histN = newton()
    close(Phi_N, ref["flux_Wb"], rtol=1e-12, name="Newton flux = numerics.magnetic_circuit (Wb)")
    close(Phi_N / PM["Ay"], 1.607023, rtol=1e-6, name="B_y (T)")
    rN = [abs(r) / Fc for _, r in histN]
    close(rN[1], 0.536, rtol=3e-3, name="Newton |f|/F_c after 1 step")
    close(rN[2], 1.9e-5, rtol=3e-2, name="Newton |f|/F_c after 2 steps")
    close(rN[3], 2.3e-10, rtol=5e-2, name="Newton |f|/F_c after 3 steps")
    assert rN[4] < 1e-14
    close(histN[1][0] / PM["Ay"], 1.678, rtol=1e-3, name="Newton first iterate B_y (T) = linear-yoke answer")
    Rs, Rd = yoke_reluctances(Phi_N)
    close(Rs, 1.39e6, rtol=3e-3, name="yoke secant reluctance at the solution (A/Wb)")
    close(Rd, 3.17e8, rtol=3e-3, name="yoke differential reluctance (A/Wb)")
    close(PM["ly"] / (Rs * PM["Ay"] * MU0), 229, rtol=3e-3, name="yoke mu_sec / mu0")
    close(PM["ly"] / (Rd * PM["Ay"] * MU0), 1.005, rtol=1e-3, name="yoke mu_diff / mu0")
    gfac = picard_factor(Phi_N)
    close(gfac, -11.1, rtol=5e-3, name="undamped Picard error factor 1 - (R0+Rd)/(R0+Rs)")
    close(2 / (1 - gfac), 0.165, rtol=5e-3, name="largest omega for convergence 2/(1-g')")
    _, histP = picard(1.0, iters=40)
    bP = [ph / PM["Ay"] for ph, _, _ in histP]
    close(bP[1], 1.678, rtol=1e-3, name="Picard iterate 1 B_y (T)")
    close(bP[2], 1.092, rtol=2e-3, name="Picard iterate 2 B_y (T)")
    close(bP[-1] if bP[-1] > 1.4 else bP[-2], 1.675, rtol=1e-3, name="Picard 2-cycle upper B_y (T)")
    close(bP[-1] if bP[-1] < 1.4 else bP[-2], 1.106, rtol=1e-3, name="Picard 2-cycle lower B_y (T)")
    rP = [abs(r) / Fc for _, r, _ in histP]
    assert min(rP[20:]) > 0.3
    close(max(rP[30:]), 0.515, rtol=5e-3, name="Picard cycle residual |f|/F_c (upper point)")
    close(min(rP[30:]), 0.340, rtol=5e-3, name="Picard cycle residual |f|/F_c (lower point)")
    # damped Picard
    _, h05 = picard(0.05, iters=120)
    r05 = np.array([abs(r) / Fc for _, r, _ in h05])
    k05 = int(np.argmax(r05 < 1e-6))
    close(k05, 77, rtol=0.03, name="omega = 0.05: iterations to |f|/F_c < 1e-6")
    _, h02 = picard(0.02, iters=120)
    k = 78
    close(h02[k][2], 5.1e-3, rtol=3e-2, name="omega = 0.02, iteration 78: relative update")
    close(abs(h02[k][1]) / Fc, 0.21, rtol=3e-2, name="omega = 0.02, iteration 78: |f|/F_c")
    # continuation: Picard converges at low load, fails beyond the critical yoke flux density
    from scipy.optimize import brentq
    bcrit = brentq(lambda b: picard_factor(b * PM["Ay"]) + 1.0, 1.0, 1.6)
    close(bcrit, 1.598, rtol=5e-4, name="yoke B_y where the undamped Picard factor reaches -1 (T)")
    close(picard_factor(1.55 * PM["Ay"]), -0.098, rtol=2e-2, name="Picard factor at B_y = 1.55 T")
    close(picard_factor(1.60 * PM["Ay"]), -1.45, rtol=1e-2, name="Picard factor at B_y = 1.60 T")
    for s, should in ((0.5, True), (0.8, True), (1.0, False)):
        Phi_s = newton_solve(Fc_scale=s)
        conv = abs(picard_factor(Phi_s)) < 1
        assert conv == should, (s, picard_factor(Phi_s))
        Pp, hp = picard(1.0, Phi0=0.0, iters=200, Fc_scale=s)
        ok = abs(Pp - Phi_s) / Phi_s < 1e-10
        assert ok == should, (s, Pp, Phi_s)
        if should:
            assert abs(picard_factor(Phi_s)) < 0.02
        print(f"  ok  B_r scaled by {s}: B_y = {Phi_s / PM['Ay']:.3f} T, Picard factor {picard_factor(Phi_s):+.2f},"
              f" converges = {ok}")

    print("Section 17.7: small-signal response about the bias")
    Rs, Rd = yoke_reluctances(Phi_N)
    tan = 1 / (Rm + Rg + Rd)
    sec = 1 / (Rm + Rg + Rs)
    close(tan, 2.91e-9, rtol=2e-3, name="tangent dPhi/dF (Wb/A)")
    close(sec, 3.53e-8, rtol=3e-3, name="frozen-secant dPhi/dF (Wb/A)")
    close(sec / tan, 12.1, rtol=5e-3, name="secant / tangent")
    errs = {}
    for eps_ in (1000.0, 100.0, 10.0, 1.0, 0.1, 1e-6):
        cd = (newton_solve(+eps_) - newton_solve(-eps_)) / (2 * eps_)
        errs[eps_] = abs(cd - tan) / tan
    close(errs[1000.0], 0.117, rtol=2e-2, name="central difference rel. error, eps = 1000 A")
    close(errs[100.0], 2.8e-4, rtol=3e-2, name="eps = 100 A")
    close(errs[10.0], 2.7e-6, rtol=5e-2, name="eps = 10 A")
    assert errs[0.1] < 1e-8
    assert errs[1.0] < 3e-6 and errs[10.0] < 3e-6
    print("  ok  plateau eps = 0.1 ... 10 A: relative difference <= 3e-6")
    print(f"  ok  eps = 0.1 A: {errs[0.1]:.1e} (< 1e-8)")
    assert 1e-7 < errs[1e-6] < 1e-3
    print(f"  ok  eps = 1e-6 A: {errs[1e-6]:.1e} (round-off grows again)")
    print("All Chapter 17 checks passed.")


if __name__ == "__main__":
    main()
