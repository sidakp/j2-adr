# Phase 3 + 4 - Mission Leg Optimisation
# Greedy approach - Always picks the cheapest (Delta-V) next target

import numpy as np
import pandas as pd
from constants import MU, R_E, J2, DEG_TO_RAD, DAY_TO_SEC
from orbitalstate import OrbitalState
from transferleg import TransferLeg

"""
1) Converts each row into an OrbitalState — the CSV stores angles in degrees and semi-major axis in km, while
OrbitalState expects radians and metres

2) Handle the fact that the CSV gives us MEAN_ANOMALY and ARG_OF_PERICENTER separately, but OrbitalState
uses argument of latitude u = omega + theta. For near-circular orbits (which debris in SSO mostly are), 
mean anomaly ≈ true anomaly, so u ≈ omega + M is a reasonable approximation.

3) Return a list of OrbitalState objects alongside the debris names/IDs so we can track which is which
"""

def load_debris_catalogue(csv_path="debris-catalogue.csv"):
    """Load debris targets from CSV and convert to OrbitalState objects."""
    df = pd.read_csv(csv_path)

    targets = []
    for _, row in df.iterrows():
        # CSV: km to m, deg to rad
        a = row['SEMIMAJOR_AXIS'] * 1e3
        e = row['ECCENTRICITY']
        i = row['INCLINATION'] * DEG_TO_RAD
        raan = row['RA_OF_ASC_NODE'] * DEG_TO_RAD
        omega = row['ARG_OF_PERICENTER'] * DEG_TO_RAD
        M = row['MEAN_ANOMALY'] * DEG_TO_RAD
        u = omega + M

        state = OrbitalState(a=a, e=e, i=i, raan=raan, omega=omega, u=u)

        targets.append({
            'state' : state,
            'norad_id' : row['NORAD_CAT_ID'],
            'name' : row['OBJECT_NAME']
        })  

    return targets

"""
1) Set the mission constraints - Define where the mothership starts and how much fuel it has (expressed as
a total Delta-V budget in m/s). This function just bundles those together so the sequencing loop has
everything it needs.

2) Using a function as allows for experimentation with different starting conditions and fuel budgets 
without changing the sequencing logic.
"""

def create_mission(dv_budget, a=R_E + 680e3, e=0.0001, i_deg=98.0,
                   raan_deg=45.0, omega=0.0, u=0.0):
    """"Define the mothership's initial state and Delta-V budget."""
    mothership = OrbitalState(
        a=a, e=e, i=np.radians(i_deg), raan=np.radians(raan_deg),
        omega=omega, u=u
    )

    return { 
        'mothership': mothership, # Current orbital state, which gets updated after each transfer leg
        'dv_budget': dv_budget, # Total fuel available in m/s
        'dv_spent': 0.0, # Cumulative Delta-V spent, starting at 0
        'visited' : [], # Array to accumulate targets removed in order
        'log' : [] # Array to record details of each transfer leg
    }

"""
The compute_all_costs function takes the mothership's current state and the list of unvisited targets,
then uses the TransferLeg class to price out every possible next transfer. It returns the results sorted
cheapest-first, so the greedy algorithm picks the top one.
"""

def compute_all_costs(mothership, targets):
    """Compute the Delta-V cost to transfer from the mothership's current state to each target."""
    costs = []

    for target in targets:
        leg = TransferLeg(mothership, target['state'])
        result = leg.optimise()

        if result['feasible']:
            costs.append({
                'target': target,
                'result': result
            })

    costs.sort(key=lambda x: x['result']['total_dv']) # Sort by total Delta-V cost

    return costs

""" 
The run_mission function ties everything together. It repeatedly finds the cheapest target, check if
the fuel budget allows for the transfer, visit the target, update the mothership, and repeat until
we're out of fuel or targets.
"""

def run_mission(mission, targets):
    """Greedy sequencing loop: always picks the cheapest next target"""
    remaining = list(targets) # Creates a copy to preserve original
    step = 0

    print(f"Mission Start: {len(remaining)} targets, "
          f"{mission['dv_budget']:.1f} m/s Delta-V budget")
    
    while remaining:
        # Compute costs to all remaining targets
        ranked = compute_all_costs(mission['mothership'], remaining)

        if not ranked:
            print("No feasible targets remain.")
            break

        # Greedy pick: cheapest first
        best = ranked[0]
        dv_cost = best['result']['total_dv']
        dv_remaining = mission['dv_budget'] - mission['dv_spent']

        # Is it affordable?
        if dv_cost > dv_remaining:
            print(f"No more Delta-V remaining. Next target costs {dv_cost:.1f} m/s "
                  f"but only {dv_remaining:.1f} m/s remains.")
            break

        # Visit the target
        step += 1
        mission['dv_spent'] += dv_cost
        mission['visited'].append(best['target'])
        mission['log'].append(best['result'])

        # Update mothership state to the target orbit
        mission['mothership'] = best['target']['state'].copy()

        # Remove from remaining
        remaining.remove(best['target'])

        # Print progress
        print(f"Step {step}: -> {best['target']['name']} "
              f"(NORAD {best['target']['norad_id']})")
        print(f"  Delta-V: {dv_cost:.1f} m/s | "
              f"Drift: {best['result']['drift_time_days']:.0f} days | "
              f"Spent: {mission['dv_spent']:.1f} / "
              f"{mission['dv_budget']:.1f} m/s")

    # Summary
    print(f"\n{'='*50}")
    print(f"Mission complete: {len(mission['visited'])} targets removed")
    print(f"Delta-V spent: {mission['dv_spent']:.1f} m/s")
    print(f"Budget remaining: "
          f"{mission['dv_budget'] - mission['dv_spent']:.1f} m/s remaining")
    
    return mission
