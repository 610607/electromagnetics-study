"""Executable checks for the learning projects. Synthetic models, not safety validation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def run_checks():
    sigma,area,h=.5,1e-4,.01
    K=sigma*area/h*np.array([[1.,-1.,0.],[-1.,2.,-1.],[0.,-1.,1.]])
    phi=np.array([.4,.2,0.]);reaction=K@phi;E=-np.diff(phi)/h;J=sigma*E
    P_terminal=phi@reaction;P_volume=np.sum(sigma*area*h*E**2)
    assert np.allclose(reaction,[.001,0,-.001],atol=1e-14)
    assert np.allclose(E,20.) and np.allclose(J,10.)
    assert np.isclose(P_volume,.0004) and np.isclose(P_terminal,P_volume)
    # Explicit e^{+j omega t} convention: phasor (b,jb) rotates clockwise.
    b=1.7;bx,by=b,1j*b;co=(bx-1j*by)/2;counter=(bx+1j*by)/2
    assert np.isclose(co,b) and np.isclose(counter,0)
    theta=np.linspace(0,2*np.pi,101);physical=np.real(np.array([bx,by])[:,None]*np.exp(1j*theta))
    assert np.max(abs(physical-np.vstack((b*np.cos(theta),-b*np.sin(theta)))))<1e-12
    # Q in W/kg for two equal, real, peak electric-field basis vectors.
    sig=.5;rho=1000.;peak=100.;Q=sig*peak**2/(2*rho)*np.ones((2,2))
    q_same=float(np.array([1.,1.])@Q@np.array([1.,1.]))
    q_opposite=float(np.array([1.,-1.])@Q@np.array([1.,-1.]))
    assert np.all(np.linalg.eigvalsh(Q)>=-1e-12) and q_same==10 and q_opposite==0
    pulse_sar=sig*peak**2/(2*rho);train_sar=pulse_sar*(.001/.01)
    assert pulse_sar==2.5 and train_sar==.25
    # A source-map conversion test: 1 A/us = 10^6 A/s.
    assert np.isclose(2e-5*1e6,20.)
    return {'two_element_bar':{'K_S':K.tolist(),'phi_V':phi.tolist(),'inward_reactions_A':reaction.tolist(),'E_V_per_m':E.tolist(),'J_A_per_m2':J.tolist(),'terminal_power_W':P_terminal,'volume_power_W':P_volume},
    'rotating_phasor':{'amplitude':b,'co_rotating':float(co.real),'counter_rotating':float(counter.real),'maximum_time_reconstruction_error':float(np.max(abs(physical-np.vstack((b*np.cos(theta),-b*np.sin(theta))))))},
    'sar':{'Q_W_per_kg':Q.tolist(),'eigenvalues':np.linalg.eigvalsh(Q).tolist(),'two_equal_in_phase_W_per_kg':q_same,'two_equal_opposed_W_per_kg':q_opposite,'during_pulse_W_per_kg':pulse_sar,'pulse_train_W_per_kg':train_sar},
    'interpretation':'Teaching inputs. These checks validate algebra, conventions and implementation, not a human exposure model.'}

if __name__=='__main__':
    results=run_checks();p=Path(__file__).resolve().parent/'extended_results.json';p.write_text(json.dumps(results,indent=2));print('All extended checks passed:',p)
