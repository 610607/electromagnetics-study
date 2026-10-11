"""Numerical checks for Chapter 1 (zh/ch01_intro.tex) and its computed figure intro_scene.

Run from the project root:  python3 checks/check_ch01.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figsrc"))


def main() -> None:
    # --- constants quoted in Sec. 1.2 (old edition, 00_picture) -----------------
    eps0, mu0 = 8.8542e-12, 4e-7 * np.pi
    assert abs(mu0 - 1.2566e-6) / 1.2566e-6 < 1e-4
    print(f"eps0 = {eps0:.4e} F/m, mu0 = 4*pi*1e-7 = {mu0:.4e} H/m")

    # --- preview of eq:uniforminduction (old edition, 02_formulations sub 5) ----
    dBdt, r = 1000.0, 0.08          # T/s, m
    E = r / 2 * dBdt                # |E_theta| = (r/2)|dB/dt|
    assert abs(E - 40.0) < 1e-12
    print(f"uniform dB/dt = {dBdt:g} T/s at r = {r} m  ->  |E| = {E:g} V/m")
    # a static field (dB/dt = 0) induces nothing
    assert r / 2 * 0.0 == 0.0

    # --- intro_scene: the computed thin-disc picture obeys its own physics -------
    import fig_ch01 as f
    R = 1.45
    th = np.linspace(0, 2 * np.pi, 721, endpoint=False)
    rr = 0.999 * R
    X, Y = rr * np.cos(th)[None, :], rr * np.sin(th)[None, :]
    Epx, Epy, Ex, Ey, *_ = f.sheet_fields(X, Y, R=R, c=(0.42, 0.22), a=0.5, h=0.45)
    n_tot = np.abs(Ex * np.cos(th) + Ey * np.sin(th)).max()
    n_pri = np.abs(Epx * np.cos(th) + Epy * np.sin(th)).max()
    assert n_tot < 0.02 * n_pri, (n_tot, n_pri)          # no current leaves the insulated rim
    # total field is divergence-free inside (Laplace correction keeps div E_p = 0)
    g = np.linspace(-1.0, 1.0, 401)
    GX, GY = np.meshgrid(g, g)
    _, _, Ex2, Ey2, *_ = f.sheet_fields(GX, GY, R=R, c=(0.42, 0.22), a=0.5, h=0.45)
    h = g[1] - g[0]
    div = np.gradient(Ex2, h, axis=1) + np.gradient(Ey2, h, axis=0)
    scale = np.hypot(Ex2, Ey2).max() / 1.0
    assert np.abs(div[5:-5, 5:-5]).max() < 2e-2 * scale, np.abs(div).max()
    print(f"intro_scene: max |E.n| on rim = {n_tot:.2e} (primary {n_pri:.2e}); interior div E ~ 0")
    print("Chapter 1 checks passed.")


if __name__ == "__main__":
    main()
