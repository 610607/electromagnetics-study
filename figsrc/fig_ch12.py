"""Figures for Chapter 12: eddy currents and coupled MQS.

eddy_oneway_vs_coupled  cross-sections of a long coil with (left) a weakly conducting body inside, whose
                        currents do not feed back on B, and (right) a thin conducting shell outside, whose
                        eddy currents oppose the change; below each, the computed field in the bore for a
                        trapezoidal coil current.  The right-hand curve is the thin-shell model derived in
                        the chapter:  tau dK_s/dt + K_s = -kappa tau dK_c/dt,  H_bore = K_c + K_s,
                        tau = mu0 sigma Delta R / 2,  kappa = a^2/R^2.
coil_voltage            V = R I + L dI/dt for a trapezoidal current with the old-edition teaching values
                        L = 20 uH, R = 0.1 ohm, I = 5000 A, dI/dt = 50 A/us.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Wedge

from style import (ACCENT, AMBER, CURR, ELEC, FIG_W, INK, IRON_T, MAG, MAG_T, MATL, MATL_T, MUTED, POT, T, arrow,
                   clean_axes, curved_arrow, figure, halo, into_page, label, out_of_page, panel_title, plt, rbox,
                   tint)

COPPER = "#B87333"
# worked example of Sec. eddy-diffusion (teaching values): copper shell R = 0.3 m, Delta = 2 mm; coil radius a = 0.25 m
SHELL_SIGMA, SHELL_DELTA, SHELL_R, COIL_A = 5.8e7, 2e-3, 0.3, 0.25
SHELL_KAPPA = (COIL_A / SHELL_R) ** 2


# ---------------------------------------------------------------------------
# Thin-shell model (normalised): time in units of tau, coil surface current K_c in units of its plateau.
# ---------------------------------------------------------------------------
def trapezoid(t, tr, tf):
    """Coil current: ramp 0->1 in tr, flat for tf, ramp 1->0 in tr."""
    return np.clip(np.minimum(t / tr, (2 * tr + tf - t) / tr), 0.0, 1.0)


def shell_response(t, kappa, tr, tf):
    """Exact solution of tau dKs/dt + Ks = -kappa tau dKc/dt (tau = 1) for the trapezoid, Ks(0) = 0."""
    Ks = np.zeros_like(t)
    rate = 1.0 / tr
    # superpose ramps: +rate starting at 0, -rate at tr, -rate at tr+tf, +rate at 2tr+tf
    for t0, r in ((0.0, rate), (tr, -rate), (tr + tf, -rate), (2 * tr + tf, rate)):
        s = np.clip(t - t0, 0.0, None)
        Ks += -kappa * r * (1.0 - np.exp(-s))
    return Ks


def shell_tau(sigma, delta, R, mu0=4e-7 * np.pi):
    return mu0 * sigma * delta * R / 2.0


@figure("eddy_oneway_vs_coupled")
def eddy_oneway_vs_coupled():
    fig = plt.figure(figsize=(FIG_W, 3.45))
    kappa, tr, tf = SHELL_KAPPA, 0.06, 2.4          # kappa of the worked example; ramp much shorter than tau
    t = np.linspace(-0.2, 5.5, 2000)
    Kc = trapezoid(t, tr, tf)
    Ks = shell_response(t, kappa, tr, tf)
    for col, coupled in enumerate((False, True)):
        x0 = 0.02 + 0.5 * col
        ax = fig.add_axes([x0 + 0.03, 0.43, 0.40, 0.56])
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-1.6, 1.62)
        ax.set_aspect("equal")
        ax.axis("off")
        a = 0.85
        th = np.linspace(0, 2 * np.pi, 300)
        if coupled:
            Rs = 1.3
            ax.add_patch(Circle((0, 0), Rs + 0.06, fc=tint(COPPER, 0.55), ec=COPPER, lw=0.8, zorder=1))
            ax.add_patch(Circle((0, 0), Rs - 0.06, fc="white", ec=COPPER, lw=0.8, zorder=1.1))
            # eddy current opposite to the coil current while it rises (clockwise)
            for ang in (np.deg2rad(60), np.deg2rad(180), np.deg2rad(300)):
                p = Rs * np.array([np.cos(ang), np.sin(ang)])
                v = 0.09 * np.array([np.sin(ang), -np.cos(ang)])
                arrow(ax, tuple(p - v), tuple(p + v), color=COPPER, lw=2.0, ms=11, z=6)
            label(ax, 2.0, -1.5, T("铜屏蔽中的涡流", "shield eddy current"), size=8.3, color=COPPER, ha="right")
            # in the bore: coil field (out of page) and the opposing eddy-current field (into the page)
            out_of_page(ax, -0.36, 0.12, color=MAG, r=0.08, lw=0.9)
            into_page(ax, 0.36, 0.12, color=tint(MAG, 0.6), r=0.08, lw=0.9)
            label(ax, -0.36, -0.14, T("线圈场", "coil"), size=7.6, color=MAG)
            label(ax, 0.36, -0.14, T("涡流场", "eddy"), size=7.6, color=tint(MAG, 0.75))
        else:
            ax.add_patch(Circle((0, 0), 0.62, fc=MATL_T, ec=MATL, lw=0.8, zorder=1))
            # weak induced currents in the body (clockwise while I rises)
            for r in (0.25, 0.48):
                ax.plot(r * np.cos(th), r * np.sin(th), color=tint(CURR, 0.55), lw=0.9, zorder=2)
                ang = np.deg2rad(40)
                p = r * np.array([np.cos(ang), np.sin(ang)])
                v = 0.05 * np.array([np.sin(ang), -np.cos(ang)])
                arrow(ax, tuple(p - v), tuple(p + v), color=tint(CURR, 0.75), lw=1.1, ms=7, z=3)
            for (x, y) in ((-0.3, -0.12), (0.3, -0.12), (0.0, 0.2)):
                out_of_page(ax, x, y, color=MAG, r=0.07, lw=0.9)
            label(ax, 0.0, -0.36, T("组织", "tissue"), size=8.2, color=INK)
            label(ax, 0.0, -1.18, T("组织电流太弱，它们的磁场可忽略", "body currents too weak to change B"), size=8.3, color=MUTED, ha="center")
        # coil winding (current counter-clockwise, rising)
        ax.plot(a * np.cos(th), a * np.sin(th), color=CURR, lw=3.0, zorder=4)
        for ang in (np.deg2rad(110), np.deg2rad(230), np.deg2rad(350)):
            p = a * np.array([np.cos(ang), np.sin(ang)])
            v = 0.06 * np.array([-np.sin(ang), np.cos(ang)])
            arrow(ax, tuple(p - v), tuple(p + v), color=CURR, lw=2.0, ms=11, z=5)
        label(ax, -1.42, 1.18, T("线圈 $I(t)$", "coil $I(t)$"), size=8.6, color=CURR, ha="center")
        ttl = T("单向：感应电流不改写磁场", "one-way: induced currents leave B unchanged") if not coupled else \
            T("耦合：涡流反过来改写磁场", "coupled: eddy currents rewrite B")
        label(ax, 0.0, 1.55, ttl, size=9.4, weight="bold", color=INK)
        # ---- time response
        bx = fig.add_axes([x0 + 0.07, 0.175, 0.38, 0.22])
        clean_axes(bx)
        bx.axhline(0, color=MUTED, lw=0.5)
        bx.plot(t, Kc, color=CURR, lw=1.3, ls=(0, (4, 2)), zorder=3)
        B = Kc + (Ks if coupled else 0.0)
        bx.plot(t, B, color=MAG, lw=1.9, zorder=4)
        if coupled:
            bx.plot(t, Ks, color=COPPER, lw=1.1, zorder=2)
            halo(bx.text(3.0, -0.42, T("屏蔽电流", "shield current"), fontsize=7.8, color=COPPER), 2)
            bx.text(3.0, 0.72, T("场滞后：以 $\\tau$\n爬升、以 $\\tau$ 衰减", "field lags: rises\nand decays with $\\tau$"), fontsize=7.6,
                    color=MAG, linespacing=1.2)
        else:
            bx.text(0.3, 0.38, T("场与电流同步\n（两线重合）", "field follows\nthe current"), fontsize=7.6, color=MAG,
                    linespacing=1.2)
        bx.set_ylim(-0.75, 1.25)
        bx.set_xlim(t[0], t[-1])
        bx.set_yticks([0, 1])
        bx.set_xticks([0, 1, 2, 3, 4, 5])
        bx.set_xlabel(T(r"时间 $t/\tau$", r"time $t/\tau$"), fontsize=8.6, labelpad=1)
        bx.tick_params(labelsize=7.8)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=CURR, lw=1.3, ls=(0, (4, 2))), Line2D([], [], color=MAG, lw=1.9)],
               [T("线圈电流（归一化）", "coil current (normalized)"), T("孔内磁场（归一化）", "field in the bore (normalized)")],
               loc="lower center", ncol=2, fontsize=8.3, bbox_to_anchor=(0.5, -0.01), handlelength=2.2)
    return fig


# ---------------------------------------------------------------------------
L_COIL, R_COIL, I_PEAK, SLEW = 20e-6, 0.1, 5000.0, 50e6      # H, ohm, A, A/s (old-edition teaching values)


def coil_waveform(t, t_flat=300e-6):
    tr = I_PEAK / SLEW
    I = I_PEAK * trapezoid(t, tr, t_flat)
    dI = np.where((t > 0) & (t < tr), SLEW, 0.0) + np.where((t > tr + t_flat) & (t < 2 * tr + t_flat), -SLEW, 0.0)
    return I, dI


@figure("coil_voltage")
def coil_voltage():
    t = np.linspace(-50e-6, 600e-6, 6501)
    I, dI = coil_waveform(t)
    VR = R_COIL * I
    VL = L_COIL * dI
    V = VR + VL
    tu = t * 1e6
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(FIG_W, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.55]))
    fig.subplots_adjust(left=0.1, right=0.78, top=0.92, bottom=0.13, hspace=0.3)
    clean_axes(ax)
    clean_axes(bx)
    ax.plot(tu, I / 1000, color=CURR, lw=1.9)
    ax.set_ylim(-0.3, 6)
    ax.set_yticks([0, 5])
    ax.text(0.0, 1.05, T("线圈电流 $I$ (kA)", "coil current $I$ (kA)"), transform=ax.transAxes, fontsize=8.8, color=CURR)
    bx.axhline(0, color=MUTED, lw=0.6)
    bx.plot(tu, VR, color=tint(POT, 0.55), lw=1.4, ls=(0, (4, 2)), zorder=3)
    bx.plot(tu, VL, color=ELEC, lw=1.3, ls=(0, (1.5, 1.5)), zorder=3)
    bx.plot(tu, V, color=POT, lw=2.0, zorder=4)
    bx.set_ylim(-1250, 1800)
    bx.set_yticks([-1000, -500, 0, 500, 1000, 1500])
    bx.text(0.0, 1.03, T("端电压 (V)", "terminal voltage (V)"), transform=bx.transAxes, fontsize=8.8, color=POT)
    bx.set_xlabel(T(r"时间 ($\mu$s)", r"time ($\mu$s)"))
    bx.set_xlim(tu[0], tu[-1])
    # annotations
    bx.annotate(T("1500 V = 500 + 1000", "1500 V = 500 + 1000"), xy=(100, 1500), xytext=(150, 1580), fontsize=8.0,
                color=POT, arrowprops=dict(arrowstyle="-", color=POT, lw=0.6), va="center")
    fig.text(0.80, 0.73, T("$I$ 爬升 100 μs\n$\\mathrm{d}I/\\mathrm{d}t=50\\,$A/μs", "ramp 100 μs\n$\\mathrm{d}I/\\mathrm{d}t=50\\,$A/μs"),
             fontsize=8.2, color=CURR, va="center", linespacing=1.3)
    fig.text(0.80, 0.53, T("—— 总电压 $V$", "—— total $V$"), fontsize=8.4, color=POT, va="center")
    fig.text(0.80, 0.46, T("- - 电阻压降 $RI$\n      平台期 500 V", "- - resistive $RI$\n      500 V on the flat top"), fontsize=8.2,
             color=tint(POT, 0.8), va="top", linespacing=1.3)
    fig.text(0.80, 0.33, T("··· 电感压降 $L\\,\\mathrm{d}I/\\mathrm{d}t$\n      爬升时 ±1000 V", "··· inductive $L\\,\\mathrm{d}I/\\mathrm{d}t$\n      ±1000 V on ramps"),
             fontsize=8.2, color=ELEC, va="top", linespacing=1.3)
    fig.text(0.80, 0.16, T("平台期储能\n$\\frac{1}{2}LI^2=250\\,$J", "stored on the flat top\n$\\frac{1}{2}LI^2=250\\,$J"), fontsize=8.2,
             color=INK, va="center", linespacing=1.3)
    return fig
