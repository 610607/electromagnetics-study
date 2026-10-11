"""Numerical checks for Chapter 2 (zh/ch02_calculus.tex) and its figures.

Run from the project root:  python3 checks/check_ch02.py
Every statement checked here is a mathematical identity or a worked example of the chapter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figsrc"))
H = 1e-5


def div2(F, x, y):
    return ((F(x + H, y)[0] - F(x - H, y)[0]) + (F(x, y + H)[1] - F(x, y - H)[1])) / (2 * H)


def curl2(F, x, y):
    return ((F(x + H, y)[1] - F(x - H, y)[1]) - (F(x, y + H)[0] - F(x, y - H)[0])) / (2 * H)


def main() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, size=(50, 2))
    fields = {
        "source (x,y)": (lambda x, y: (x, y), 2.0, 0.0),
        "vortex (-y,x)": (lambda x, y: (-y, x), 0.0, 2.0),
        "uniform (1,0)": (lambda x, y: (1.0 + 0 * x, 0 * y), 0.0, 0.0),
        "saddle (x,-y)": (lambda x, y: (x, -y), 0.0, 0.0),
        "shear (y,0)": (lambda x, y: (y, 0 * x), 0.0, -1.0),
    }
    for name, (F, d0, c0) in fields.items():
        for x, y in pts:
            assert abs(div2(F, x, y) - d0) < 1e-6 and abs(curl2(F, x, y) - c0) < 1e-6, name
        print(f"{name:15s}: div = {d0:g}, curl_z = {c0:g}")

    # wire-like field (-y, x)/(x^2+y^2): curl 0 away from the origin, but circulation 2*pi around it
    W = lambda x, y: (-y / (x * x + y * y), x / (x * x + y * y))
    for x, y in pts:
        if np.hypot(x, y) > 0.3:
            assert abs(curl2(W, x, y)) < 1e-4 and abs(div2(W, x, y)) < 1e-4
    t = np.linspace(0, 2 * np.pi, 20001)
    for R in (0.5, 1.7):
        fx, fy = W(R * np.cos(t), R * np.sin(t))
        circ = np.trapezoid(fx * (-R * np.sin(t)) + fy * (R * np.cos(t)), t)
        assert abs(circ - 2 * np.pi) < 1e-8
    print("wire field: curl = 0 off the axis, circulation = 2*pi for every circle around it")

    # divergence theorem example: F = (x, y, 0) on a cylinder (radius R, height h) about the z axis
    R, h = 0.3, 2.0
    th = np.linspace(0, 2 * np.pi, 20001)
    Fn = R                                   # F.n on the side wall = r = R; F.n = 0 on the end caps
    flux = np.trapezoid(np.full_like(th, Fn) * R * h, th)
    assert abs(flux - 2 * np.pi * R**2 * h) < 1e-10
    print(f"cylinder R={R}, h={h}: outward flux = {flux:.6f} = 2 * volume = {2*np.pi*R**2*h:.6f}")

    # Stokes example: F = (-y, x, 0) around a circle of radius R
    fx, fy = -R * np.sin(th), R * np.cos(th)
    circ = np.trapezoid(fx * (-R * np.sin(th)) + fy * (R * np.cos(th)), th)
    assert abs(circ - 2 * np.pi * R**2) < 1e-10
    print(f"circle R={R}: circulation = {circ:.6f} = 2 * area = {2*np.pi*R**2:.6f}")

    # V3: curl grad = 0 and div curl = 0 for smooth test functions (finite differences, 3-D)
    phi = lambda x, y, z: np.sin(x) * np.exp(0.3 * y) + x * y * z**2
    A = lambda x, y, z: np.array([y * z**2, np.cos(x * z), x**2 * np.sin(y)])
    def grad(f, p):
        e = np.eye(3) * 1e-4
        return np.array([(f(*(p + e[i])) - f(*(p - e[i]))) / 2e-4 for i in range(3)])
    def curl(Fv, p):
        e = np.eye(3) * 1e-4
        J = np.array([(Fv(*(p + e[i])) - Fv(*(p - e[i]))) / 2e-4 for i in range(3)])  # J[i, k] = d_i F_k
        return np.array([J[1, 2] - J[2, 1], J[2, 0] - J[0, 2], J[0, 1] - J[1, 0]])
    for p in rng.uniform(-1, 1, size=(10, 3)):
        cg = curl(lambda x, y, z: grad(phi, np.array([x, y, z])), p)
        e = np.eye(3) * 1e-3
        dc = sum((curl(A, p + e[i])[i] - curl(A, p - e[i])[i]) / 2e-3 for i in range(3))
        assert np.abs(cg).max() < 1e-4 and abs(dc) < 1e-4, (cg, dc)
    print("V3: curl(grad phi) = 0 and div(curl A) = 0 (numerically, random points)")

    # V4 worked example: sigma(x) = sigma0 (1 + x/L), uniform J -> div E = -(grad sigma . E)/sigma != 0
    s0, L, J0 = 0.2, 0.01, 10.0
    x = np.linspace(0, 0.01, 2001)
    sig = s0 * (1 + x / L)
    E = J0 / sig
    divE = np.gradient(E, x)
    pred = -(s0 / L) * E / sig
    assert np.max(np.abs(divE[2:-2] - pred[2:-2])) < 1e-3 * np.max(np.abs(pred))
    assert abs(E[0] / E[-1] - 2.0) < 1e-12                 # sigma doubles across the slab -> E halves
    print("V4 example: div(sigma E)=0 with varying sigma forces div E = -(grad sigma . E)/sigma; E halves when sigma doubles")

    # gradient theorem in figure gradient_contours: two paths, same answer
    import fig_ch02 as f
    I1 = f.line_integral_of_grad(f.path_straight)
    I2 = f.line_integral_of_grad(f.path_curved)
    d = float(f.landscape(*f.PB) - f.landscape(*f.PA))
    assert abs(I1 - d) < 1e-6 and abs(I2 - d) < 1e-6, (I1, I2, d)
    print(f"gradient theorem (figure): path 1 = {I1:.6f}, path 2 = {I2:.6f}, phi(b)-phi(a) = {d:.6f}")
    print("Chapter 2 checks passed.")


if __name__ == "__main__":
    main()
