# Phase 4 - Monte Carlo Error Analysis
# Perturbs debris orbital elements with uncertainties

import numpy as np
from constants import R_E, DEG_TO_RAD
from orbitalstate import OrbitalState
from mission import load_debris_catalogue, create_mission, run_mission

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
        perturbed = perturb_targets(baseline_targets, sigma_alt, sigma_angle)
        mission = create_mission(dv_budget=dv_budget)

        import io, sys
        quiet = io.StringIO()
        sys.stdout = quiet

        result = run_mission(mission, perturbed, time_penalty=time_penalty)

        sys.stdout = sys.__stdout__

        n_removed = len(result['visited'])
        dv_spent = result['dv_spent']
        drift_days = sum(leg['drift_time_days'] for leg in result['log'])

        targets_removed.append(n_removed)
        total_dvs.append(dv_spent)
        total_times.append(drift_days)

        if (run + 1) % 10 == 0:
            print(f"Run {run + 1 }/{n_runs} complete")
    
    return {
        'targets_removed': np.array(targets_removed),
        'total_dvs': np.array(total_dvs),
        'total_times': np.array(total_times),
        'n_runs': n_runs
    }

def print_statistics(results):
    """Print summary statistics from Monte Carlo results."""
    removed = results['targets_removed']
    dvs = results['total_dvs']
    times = results['total_times']
    n = results['n_runs']

    print(f"\n{'='*50}")
    print(f"Monte Carlo Results ({n} runs)")
    print(f"{'='*50}")

    print(f"\nTargets removed:")
    print(f"  Mean:   {np.mean(removed):.1f}")
    print(f"  Std:    {np.std(removed):.1f}")
    print(f"  Min:    {np.min(removed)}")
    print(f"  Max:    {np.max(removed)}")
    print(f"  5th %%:  {np.percentile(removed, 5):.0f}")
    print(f"  95th %%: {np.percentile(removed, 95):.0f}")

    print(f"\nTotal delta-V spent (m/s):")
    print(f"  Mean:   {np.mean(dvs):.1f}")
    print(f"  Std:    {np.std(dvs):.1f}")
    print(f"  Min:    {np.min(dvs):.1f}")
    print(f"  Max:    {np.max(dvs):.1f}")

    print(f"\nTotal mission drift time (days):")
    print(f"  Mean:   {np.mean(times):.0f}")
    print(f"  Std:    {np.std(times):.0f}")
    print(f"  Min:    {np.min(times):.0f}")
    print(f"  Max:    {np.max(times):.0f}")

if __name__ == "__main__":
    print("Starting Monte Carlo analysis...\n")

    results = run_monte_carlo(
        n_runs=100,
        dv_budget=500.0,
        time_penalty=0.1,
        sigma_alt=1e3,
        sigma_angle=0.01
    )

    print_statistics(results)