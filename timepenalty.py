# Phase 4 - Time Penalty Sweep
# Runs the greedy sequencer of mission.py across a set of time-penalty
# coefficients lambda to expose the fuel / duration trade-off.

import io
import sys

from mission import load_debris_catalogue, create_mission, run_mission

"""
1) The lambda sweep is expressed as a list of coefficients rather than a
single value so that a single invocation of this module produces the full
summary table consumed by Section 4.2 of the dissertation. Each lambda in
the list is run independently with a freshly-loaded catalogue and a
freshly-created mission state, so no information leaks between runs.

2) lambda = 0 reproduces the fuel-minimum baseline of mission.py's default
run. Larger values bias the search toward shorter drifts at the expense of
fuel; values above roughly 1 m/s/day make drift time prohibitively
expensive and collapse most legs onto direct transfers.

3) The dv_budget is held fixed at the mission.py default of 500 m/s across
the sweep; sweeping both budget and lambda is out of scope for this module.
"""

LAMBDAS = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
DV_BUDGET = 500.0

"""
1) sweep_time_penalty loops over the supplied lambda list and captures the
headline mission outcome for each run: number of targets removed, total
Delta-V spent, and total drift time summed across legs. Per-leg detail is
discarded, since the purpose of the sweep is the aggregate trade-off
rather than the sequence itself.

2) mission.py's run_mission prints progress to stdout on every leg. Those
prints are redirected through io.StringIO for the duration of each run to
keep the sweep's own output legible; the pattern matches that already used
in montecarlo.py.
"""

def sweep_time_penalty(lambdas=LAMBDAS, dv_budget=DV_BUDGET):
    """Run the greedy sequencer once per lambda and collect headline stats."""
    results = []

    for lam in lambdas:
        # Fresh catalogue and mission state each iteration avoids carry-over
        targets = load_debris_catalogue()
        mission = create_mission(dv_budget=dv_budget)

        # Silence run_mission's per-leg print output
        quiet = io.StringIO()
        saved_stdout = sys.stdout
        sys.stdout = quiet
        try:
            run_mission(mission, targets, time_penalty=lam)
        finally:
            sys.stdout = saved_stdout

        n_removed = len(mission['visited'])
        total_dv = mission['dv_spent']
        total_drift = sum(leg['drift_time_days'] for leg in mission['log'])

        results.append({
            'lambda': lam,
            'n_removed': n_removed,
            'total_dv': total_dv,
            'total_drift': total_drift
        })

        print(f"lambda={lam:>5.2f}: removed {n_removed:>3}, "
              f"DV={total_dv:>6.1f} m/s, drift={total_drift:>5.0f} days")

    return results

"""
1) print_summary_table formats the sweep results as a plain-text table
suitable for direct transcription into the LaTeX source. Columns match the
layout used in Section 4.2: time penalty lambda, targets removed, total
Delta-V spent, total drift time, and mission duration in years.
"""

def print_summary_table(results):
    """Print a formatted summary table of the sweep results."""
    print()
    print(f"{'lambda':>8}  {'removed':>8}  {'DV (m/s)':>10}  "
          f"{'drift (d)':>10}  {'duration (yr)':>14}")
    print("-" * 60)
    for r in results:
        years = r['total_drift'] / 365.25
        print(f"{r['lambda']:>8.2f}  {r['n_removed']:>8}  "
              f"{r['total_dv']:>10.1f}  {r['total_drift']:>10.0f}  "
              f"{years:>14.2f}")


if __name__ == "__main__":
    print(f"Time-penalty sweep: {len(LAMBDAS)} values, "
          f"dv_budget={DV_BUDGET} m/s\n")

    results = sweep_time_penalty()
    print_summary_table(results)
