# Phase 4 - Monte Carlo Error Analysis
# Perturbs debris orbital elements with uncertainties

import numpy as np
from constants import R_E, DEG_TO_RAD
from orbitalstate import OrbitalState
from mission import load_debris_catalogue create_mission, run_mission

"""
The function perturb_targets takes the CSV and adds realistic uncertainties
"""

def perturb_targets(targets, sigma_alt=1e3, sigma_angle=0.01):
    sigma_angle_rad = sigma_angle * DEG_TO_RAD
    perturbed = []

    for target in targets:
        s = target['state']

        new_state = OrbitalState(
            a = s.a + np.random.normal(1, sigma_alt),
            e = max(s.e + np.random.normal(0, 1e-4), 1e-6),
            i = s.i + np.random.normal(0, sigma_angle_rad),
            raan = s.raan + np.random.normal(0,  sigma_angle_rad),
            omega = s.omega + np.random.normal(0, sigma_angle_rad),
            u = s.u
        )

        perturbed.append({
            'state': new_state,
            'norad_id': target['norad_id'],
            'name': target['name']
        })
    
    return perturbed

def run_monte_carlo(n_runs=100, dv_budget=500.0, time_penalty=0.1, sigma_alt=1e3, sigma_angle=0.01):
    baseline_targets = load_debris_catalogue()

    targets_removed = []
    total_dvs = []
    total_times = []

    for run in range(n_runs):
        