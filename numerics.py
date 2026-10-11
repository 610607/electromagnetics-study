"""Small, reproducible checks for the accompanying lecture.

Only NumPy and SciPy are required. All material values in the magnetic-circuit
example are synthetic teaching inputs, not specifications for a real device.
The 2-D FEM tests are nondimensional mathematical benchmarks, not head models.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import brentq

MU0 = 4.0e-7 * np.pi  # adequate approximation for these engineering examples
EPS0 = 8.8541878128e-12


def square_mesh(n: int) -> tuple[NDArray, NDArray]:
    if n < 2:
        raise ValueError("n must be at least 2")
    x = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    p = np.column_stack((xx.ravel(), yy.ravel()))
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    a = (j * (n + 1) + i).ravel()
    cells = np.vstack((np.column_stack((a, a + 1, a + n + 2)),
                       np.column_stack((a, a + n + 2, a + n + 1))))
    return p, cells


def element_geometry(p: NDArray, cells: NDArray) -> tuple[NDArray, NDArray]:
    vertices = p[cells]
    coeff = np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=2)
    area = np.abs(np.linalg.det(coeff)) / 2.0
    if np.any(area <= 0.0):
        raise ValueError("Degenerate triangle")
    # Column i of inv(coeff) contains [constant, x-coefficient, y-coefficient]
    # of the local nodal basis function N_i.
    gradients = np.linalg.inv(coeff)[:, 1:, :].transpose(0, 2, 1)
    return area, gradients


def stiffness(p: NDArray, cells: NDArray, sigma: NDArray | float = 1.0):
    area, gradients = element_geometry(p, cells)
    s = np.broadcast_to(np.asarray(sigma, dtype=float), area.shape)
    if np.any(s <= 0.0):
        raise ValueError("Conductivity must be positive")
    local = (area * s)[:, None, None] * np.einsum("eik,ejk->eij", gradients, gradients)
    rows = np.repeat(cells, 3, axis=1).ravel()
    cols = np.tile(cells, (1, 3)).ravel()
    K = coo_matrix((local.ravel(), (rows, cols)), shape=(len(p), len(p))).tocsr()
    return K, area, gradients


def triangle_quadrature():
    # Degree-five Dunavant quadrature; weights sum to one (multiply by area).
    a, b = 0.059715871789770, 0.470142064105115
    c, d = 0.797426985353087, 0.101286507323456
    bary = np.array([[1/3, 1/3, 1/3], [a,b,b], [b,a,b], [b,b,a],
                     [c,d,d], [d,c,d], [d,d,c]])
    weights = np.array([0.225] + [0.132394152788506]*3 + [0.125939180544827]*3)
    return bary, weights


def manufactured_solution(n: int):
    p, cells = square_mesh(n)
    K, area, gradients = stiffness(p, cells)
    bary, weights = triangle_quadrature()
    qp = np.einsum("qi,eid->eqd", bary, p[cells])
    x, y = qp[..., 0], qp[..., 1]
    exact = np.sin(np.pi*x) * np.sin(np.pi*y)
    forcing = 2 * np.pi**2 * exact  # -laplacian(u) = f
    local_rhs = area[:, None] * np.einsum("q,qi,eq->ei", weights, bary, forcing)
    rhs = np.zeros(len(p))
    np.add.at(rhs, cells.ravel(), local_rhs.ravel())
    boundary = np.any((p == 0.0) | (p == 1.0), axis=1)
    free = np.flatnonzero(~boundary)
    u = np.zeros(len(p))
    u[free] = spsolve(K[free][:, free], rhs[free])
    uq = np.einsum("qi,ei->eq", bary, u[cells])
    grad_u = np.einsum("ei,eid->ed", u[cells], gradients)
    grad_exact = np.stack((np.pi*np.cos(np.pi*x)*np.sin(np.pi*y),
                           np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)), axis=-1)
    l2 = np.sqrt(np.einsum("e,q,eq->", area, weights, (uq - exact)**2))
    h1 = np.sqrt(np.einsum("e,q,eq->", area, weights,
                          np.sum((grad_u[:, None, :] - grad_exact)**2, axis=-1)))
    residual = np.linalg.norm((K @ u - rhs)[free]) / np.linalg.norm(rhs[free])
    return {"n": n, "nodes": len(p), "elements": len(cells), "h": 1/n,
            "L2_error": float(l2), "H1_seminorm_error": float(h1),
            "relative_algebraic_residual": float(residual)}, (p, cells, u)


def induced_field_example(n: int = 64):
    """Insulated square conductor driven by a prescribed rotational primary E.

    The inclusion boundary is approximated by element labels, deliberately not
    advertised as a boundary-conforming or anatomically accurate mesh.
    """
    p, cells = square_mesh(n)
    center = p[cells].mean(axis=1)
    inclusion = np.sum((center - [0.62, 0.56])**2, axis=1) < 0.16**2
    sigma = np.where(inclusion, 10.0, 1.0)
    K, area, gradients = stiffness(p, cells, sigma)
    ep = np.column_stack((0.5*(center[:,1] - 0.5), -0.5*(center[:,0] - 0.5)))
    rhs_e = (area * sigma)[:, None] * np.einsum("eid,ed->ei", gradients, ep)
    rhs = np.zeros(len(p))
    np.add.at(rhs, cells.ravel(), rhs_e.ravel())
    free = np.arange(1, len(p))  # remove one constant-potential gauge degree of freedom
    phi = np.zeros(len(p))
    phi[free] = spsolve(K[free][:, free], rhs[free])
    grad_phi = np.einsum("ei,eid->ed", phi[cells], gradients)
    E = ep - grad_phi
    residual = np.linalg.norm(K @ phi - rhs) / np.linalg.norm(rhs)
    orthogonality = np.sum(area*sigma*np.einsum("ed,ed->e", E, grad_phi))
    return p, cells, phi, E, sigma, float(residual), float(orthogonality)


def synthetic_B(H):
    return MU0*np.asarray(H) + 1.6*np.tanh(np.asarray(H)/800.0)


def magnetic_circuit():
    Br, mu_rec, lm, gap, ly = 1.2, 1.05, 0.05, 0.02, 0.40
    Am, Ag, Ay = 0.002, 0.002, 0.001
    def residual(Hy):
        flux = Ay*synthetic_B(Hy)
        Bm, Bg = flux/Am, flux/Ag
        Hm = (Bm - Br)/(MU0*mu_rec)
        return Hm*lm + Hy*ly + Bg*gap/MU0
    Hy = brentq(residual, 0.0, 1.0e6, xtol=1.0e-9)
    flux = Ay*synthetic_B(Hy)
    Bg, Bm = flux/Ag, flux/Am
    ideal_Bg = Br*lm/(lm*Ag/Am + mu_rec*gap)
    return {"Br_T": Br, "mu_rec_relative": mu_rec, "magnet_length_m": lm,
            "gap_m": gap, "yoke_length_m": ly, "Am_m2": Am, "Ag_m2": Ag,
            "Ay_m2": Ay, "Hy_A_per_m": float(Hy), "By_T": float(synthetic_B(Hy)),
            "Bg_T": float(Bg), "Bm_T": float(Bm), "flux_Wb": float(flux),
            "Hm_A_per_m": float((Bm-Br)/(MU0*mu_rec)),
            "ideal_infinite_mu_yoke_Bg_T": float(ideal_Bg),
            "loop_residual_A": float(residual(Hy))}


def run_all(out: Path):
    convergence = [manufactured_solution(n)[0] for n in (8, 16, 32, 64)]
    for k in range(1, len(convergence)):
        for field in ("L2_error", "H1_seminorm_error"):
            convergence[k][field + "_order"] = float(np.log2(convergence[k-1][field]/convergence[k][field]))
    f = 3000.0
    numbers = {}
    for name, sigma, ell in (("tissue_illustration", 0.5, 0.2), ("copper", 5.8e7, 0.002)):
        omega = 2*np.pi*f
        numbers[name] = {"sigma_S_per_m": sigma, "length_m": ell,
                         "delta_m": float(np.sqrt(2/(omega*MU0*sigma))),
                         "magnetic_induction_number": float(omega*MU0*sigma*ell**2)}
    _, _, _, _, _, residual, ortho = induced_field_example()
    result = {"fem_convergence": convergence, "magnetic_circuit": magnetic_circuit(),
              "scales_at_3kHz": numbers, "induced_field_relative_residual": residual,
              "induced_field_orthogonality": ortho}
    out.write_text(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "results.json"
    print(json.dumps(run_all(path), indent=2))
