# Phase 4 - Monte Carlo Error Analysis
# Perturbs debris orbital elements with catalogue-level uncertainties drawn
# from Flohrer, Krag and Klinkrad (2008).

import numpy as np
from constants import R_E, DEG_TO_RAD
from orbitalstate import OrbitalState
from mission import load_debris_catalogue, create_mission, run_mission

"""
1) Flohrer, Krag and Klinkrad (2008) report TLE position residuals for the
SSN catalogue in the LVLH frame, broken down by orbit class. The class that
matches the SSO debris population used here - perigee altitude below 800 km,
inclination above 60 degrees, eccentricity below 0.1 - has standard
deviations of 115 m in the radial direction, 517 m in the along-track
direction, and 137 m in the cross-track direction.

2) These three position-space sigmas are mapped to orbital-element sigmas
below. The conversion uses the target's own semi-major axis and inclination
rather than a fixed reference value, so each debris object is perturbed with
its own angular sigma.

3) The eccentricity sigma is not reported by Flohrer; the 1e-4 value used
here is representative of the near-circular catalogue fits characterised by
Geul, Mooij and Noomen (2017).
"""

SIGMA_R = 115.0  # m, radial
SIGMA_T = 517.0  # m, along-track
SIGMA_N = 137.0  # m, cross-track
SIGMA_E = 1e-4   # eccentricity

"""
1) For a near-circular orbit the radial position error at a point is
dominated by the semi-major axis perturbation, so sigma_a is set equal to
the radial position sigma. The cross-track component splits between the two
plane-orientation angles: both inclination and RAAN are assigned the full
cross-track contribution, which is conservative and avoids having to pick
an arbitrary projection angle.

2) The along-track position error is folded into the argument of latitude
u = omega + theta rather than into omega alone. For near-circular orbits
the individual argument of perigee and true anomaly are ill-posed - a small
change in eccentricity can swap them - and the sequencer only consumes u
anyway, so omega is left at its catalogue value.
"""

def perturb_targets(targets):
    """Perturb each target's Keplerian elements with Flohrer-derived sigmas."""
    perturbed = []

    for target in targets:
        s = target['state']

        # Map LVLH position sigmas to per-target angular sigmas
        sigma_i    = SIGMA_N / s.a
        sigma_raan = SIGMA_N / (s.a * np.sin(s.i))
        sigma_u    = SIGMA_T / s.a

        new_state = OrbitalState(
            a = s.a + np.random.normal(0, SIGMA_R),
            e = max(s.e + np.random.normal(0, SIGMA_E), 1e-6),
            i = s.i + np.random.normal(0, sigma_i),
            raan = s.raan + np.random.normal(0, sigma_raan),
            omega = s.omega, # ill-defined for near-circular orbits; left unperturbed
            u = s.u + np.random.normal(0, sigma_u)
        )

        perturbed.append({
            'state': new_state,
            'norad_id': target['norad_id'],
            'name': target['name']
        })

    return perturbed

def run_monte_carlo(n_runs=100, dv_budget=500.0, time_penalty=0.1):
    baseline_targets = load_debris_catalogue()

    targets_removed = []
    total_dvs = []
    total_times = []

    for run in range(n_runs):
        perturbed = perturb_targets(baseline_targets)
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

    print(f"Monte Carlo Results ({n} runs)")

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
        time_penalty=0.1
    )

    print_statistics(results)