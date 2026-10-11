"""Numerical checks for the physical-atlas examples. No external data required."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

MU0 = 4.0e-7 * np.pi  # teaching approximation, not an exact modern-SI value
EPS0 = 8.8541878128e-12

def run_checks(destination: Path | None = None) -> dict:
    """Evaluate examples and assert gauge/flux and energy consistency."""
    # The two explicit gauges for a spatially uniform B(t).
    B, dB, radius = 0.004, 1.7, 0.037
    theta = np.linspace(0.0, 2.0*np.pi, 16385)
    x, y = radius*np.cos(theta), radius*np.sin(theta)
    dx, dy = -radius*np.sin(theta), radius*np.cos(theta)
    A_sym = np.stack((-B*y/2.0, B*x/2.0), axis=-1)
    A_alt = np.stack((np.zeros_like(x), B*x), axis=-1)
    flux_sym = float(np.trapezoid(A_sym[:,0]*dx+A_sym[:,1]*dy, theta))
    flux_alt = float(np.trapezoid(A_alt[:,0]*dx+A_alt[:,1]*dy, theta))
    exact_flux = float(np.pi*radius**2*B)
    E_sym = np.stack((dB*y/2.0, -dB*x/2.0), axis=-1)
    E_alt = np.stack((np.zeros_like(x), -dB*x), axis=-1)
    minus_grad_phi_alt = np.stack((dB*y/2.0, dB*x/2.0), axis=-1)
    gauge_error = float(np.max(np.abs(E_alt+minus_grad_phi_alt-E_sym)))
    np.testing.assert_allclose([flux_sym,flux_alt], exact_flux, rtol=1e-12)
    assert gauge_error < 1e-14

    # Ideal dielectric-filled capacitor, fixed free charge.
    area, d, er, Q = 0.01, 0.001, 4.0, 1e-9
    D = Q/area; E = D/(EPS0*er); voltage = E*d
    C = EPS0*er*area/d; P = D-EPS0*E
    energy = 0.5*C*voltage**2
    np.testing.assert_allclose(C*voltage, Q, rtol=1e-14)
    np.testing.assert_allclose(energy, 0.5*E*D*area*d, rtol=1e-14)

    # Two conducting layers in series under prescribed current.
    current, cross_section = 1e-3, 1e-4
    thicknesses = np.array([0.008, 0.002]); sigma = np.array([0.2,0.02])
    J = current/cross_section; fields = J/sigma
    layer_voltage = fields*thicknesses
    v_total = float(layer_voltage.sum()); resistance = v_total/current
    power = float(np.sum(sigma*fields**2*cross_section*thicknesses))
    np.testing.assert_allclose(power,current*v_total,rtol=1e-14)

    # Ideal long solenoid, leading-order magnetoquasistatic fields.
    turns_per_m, a, I, dI, r, tissue_sigma = 1000.0,0.02,5.0,1e6,0.01,0.2
    Bc = MU0*turns_per_m*I; dBc = MU0*turns_per_m*dI
    Atheta = Bc*r/2.0; Etheta = -dBc*r/2.0
    result = {
        "gauge_and_loop_flux": {
            "exact_flux_Wb":exact_flux,"symmetric_gauge_flux_Wb":flux_sym,
            "alternative_gauge_flux_Wb":flux_alt,"electric_field_gauge_error_V_per_m":gauge_error,
        },
        "capacitor": {
            "area_m2":area,"gap_m":d,"relative_permittivity":er,"charge_C":Q,
            "D_C_per_m2":D,"E_V_per_m":E,"voltage_V":voltage,
            "P_C_per_m2":P,"capacitance_F":C,"energy_J":energy,
        },
        "two_layer_conductor": {
            "current_A":current,"J_A_per_m2":J,"E_V_per_m":fields.tolist(),
            "layer_voltage_V":layer_voltage.tolist(),"voltage_V":v_total,
            "resistance_ohm":resistance,"dissipated_power_W":power,
        },
        "long_solenoid": {
            "B_T":Bc,"dB_dt_T_per_s":dBc,"sample_radius_m":r,
            "A_theta_Wb_per_m":Atheta,"E_theta_V_per_m":Etheta,
            "Ohmic_J_theta_A_per_m2":tissue_sigma*Etheta,
        },
        "status":"All numerical consistency assertions passed.",
    }
    if destination is not None:
        destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result

if __name__ == "__main__":
    result = run_checks(Path(__file__).with_name("concept_results.json"))
    print(result["status"])
    print(json.dumps(result,indent=2))
