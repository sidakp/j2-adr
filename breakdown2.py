# Verbose-breakdown variant with a non-zero time penalty.
# Identical to breakdown.py except for the lambda coefficient passed into
# TransferLeg.optimise via compute_all_costs. A non-zero lambda biases the
# grid search toward shorter drifts, which should prevent the optimiser
# from always closing the RAAN gap exactly and therefore expose a non-zero
# RAAN-residual component in the Delta-V breakdown — the test case for
# whether the residual term is ever actually paid.

import csv
import sys

import numpy as np

from constants import MU, R_E, J2, DEG_TO_RAD, DAY_TO_SEC
from orbitalstate import OrbitalState
from mission import load_debris_catalogue, create_mission, compute_all_costs


# Default taken from the lower end of timepenalty.py's sweep range; large
# enough to shorten drifts noticeably without collapsing every leg onto a
# direct transfer. Override from the command line with: python breakdown2.py 1.25
DEFAULT_LAMBDA = 1.5


def print_leg_breakdown(step, target, result, mothership, dv_spent,
                        dv_budget, time_penalty):
    """Per-leg verbose report. Mirrors breakdown.py but also prints the
    time-penalty contribution so the headline cost can be reconciled
    against the sum of the breakdown components."""
    bd = result['breakdown']
    name = target['name']
    norad = target['norad_id']
    t_state = target['state']

    dv_total = bd['total']
    time_cost = time_penalty * result['drift_time_days']
    ranked_cost = dv_total + time_cost

    ruler = "-" * 64
    print(f"\n{ruler}")
    print(f"Step {step}: -> {name}  (NORAD {norad})")
    print(ruler)

    print("Mothership parking state (constant across legs):")
    print(f"  a     = {mothership.a / 1e3:9.2f} km   "
          f"(alt {mothership.altitude / 1e3:6.2f} km)")
    print(f"  i     = {np.degrees(mothership.i):9.4f} deg")
    print(f"  RAAN  = {np.degrees(mothership.raan):9.4f} deg")

    print("Target state:")
    print(f"  a     = {t_state.a / 1e3:9.2f} km   "
          f"(alt {t_state.altitude / 1e3:6.2f} km)")
    print(f"  i     = {np.degrees(t_state.i):9.4f} deg")
    print(f"  RAAN  = {np.degrees(t_state.raan):9.4f} deg")

    print("Chosen drift geometry:")
    print(f"  drift altitude      = {result['drift_alt_km']:9.2f} km")
    print(f"  drift inclination   = {result['drift_inc_deg']:9.4f} deg")
    print(f"  drift time          = {result['drift_time_days']:9.2f} days")
    print(f"  RAAN gap (initial)  = {result['raan_gap_initial_deg']:9.4f} deg")
    print(f"  RAAN residual       = {result['raan_residual_deg']:9.4f} deg")

    print("Delta-V breakdown:")
    print(f"  Hohmann to drift    = {bd['hohmann_to_drift']:9.3f} m/s")
    print(f"  Inc to drift        = {bd['inc_to_drift']:9.3f} m/s")
    print(f"  Hohmann return      = {bd['hohmann_return']:9.3f} m/s")
    print(f"  Inc return          = {bd['inc_return']:9.3f} m/s")
    print(f"  RAAN residual       = {bd['raan_residual']:9.3f} m/s")
    print(f"  {'-' * 34}")
    print(f"  TOTAL leg Delta-V   = {dv_total:9.3f} m/s")

    print("Ranking cost (Delta-V plus time penalty):")
    print(f"  time penalty lambda = {time_penalty:9.3f} m/s per day")
    print(f"  time cost           = {time_cost:9.3f} m/s-equiv")
    print(f"  ranked cost         = {ranked_cost:9.3f} m/s-equiv")
    print("  (only the Delta-V portion is charged to the budget; "
          "the time term shapes the sequence, not the fuel spend)")

    print("Budget state:")
    print(f"  Leg cost            = {dv_total:9.3f} m/s")
    print(f"  Spent cumulative    = {dv_spent:9.3f} / {dv_budget:.3f} m/s")
    print(f"  Remaining           = {dv_budget - dv_spent:9.3f} m/s")


def run_mission_verbose(mission, targets, time_penalty):
    """Greedy sequencing loop with per-leg breakdown. Note that the
    budget still tracks pure Delta-V — only the ranking inside
    compute_all_costs uses the time-penalised cost."""
    remaining = list(targets)
    step = 0

    m0 = mission['mothership']
    print(f"Mission start: {len(remaining)} targets, "
          f"{mission['dv_budget']:.1f} m/s Delta-V budget, "
          f"lambda={time_penalty:.3f} m/s/day")
    print(f"Mothership parking orbit: "
          f"alt={m0.altitude / 1e3:.2f} km, "
          f"i={np.degrees(m0.i):.3f} deg, "
          f"RAAN={np.degrees(m0.raan):.3f} deg")

    while remaining:
        ranked = compute_all_costs(mission['mothership'], remaining,
                                    time_penalty)

        if not ranked:
            print("\nNo feasible targets remain.")
            break

        best = ranked[0]
        dv_cost = best['result']['total_dv']
        dv_remaining = mission['dv_budget'] - mission['dv_spent']

        if dv_cost > dv_remaining:
            print(f"\nBudget exhausted. Cheapest remaining leg costs "
                  f"{dv_cost:.3f} m/s but only {dv_remaining:.3f} m/s remains.")
            break

        step += 1
        mission['dv_spent'] += dv_cost
        mission['visited'].append(best['target'])
        mission['log'].append(best['result'])

        print_leg_breakdown(
            step=step,
            target=best['target'],
            result=best['result'],
            mothership=mission['mothership'],
            dv_spent=mission['dv_spent'],
            dv_budget=mission['dv_budget'],
            time_penalty=time_penalty,
        )

        # Advance catalogue by the committed drift duration (see mission.py)
        leg_dt_s = best['result']['drift_time_days'] * DAY_TO_SEC
        mission['mothership'].raan += (
            mission['mothership'].raan_dot_j2 * leg_dt_s
        )
        for tgt in remaining:
            tgt['state'].raan += tgt['state'].raan_dot_j2 * leg_dt_s

        remaining.remove(best['target'])

    print(f"\n{'=' * 64}")
    print(f"Mission complete (lambda={time_penalty:.3f} m/s/day)")
    print(f"Targets removed:  {len(mission['visited'])}")
    print(f"Delta-V spent:    {mission['dv_spent']:.3f} m/s")
    print(f"Budget remaining: {mission['dv_budget'] - mission['dv_spent']:.3f} m/s")

    if mission['log']:
        totals = {
            'hohmann_to_drift': 0.0,
            'inc_to_drift':     0.0,
            'hohmann_return':   0.0,
            'inc_return':       0.0,
            'raan_residual':    0.0,
        }
        for r in mission['log']:
            for k in totals:
                totals[k] += r['breakdown'][k]

        total_drift = sum(r['drift_time_days'] for r in mission['log'])

        print(f"\nAggregate Delta-V by component (all legs):")
        for k, v in totals.items():
            print(f"  {k:<20s} = {v:9.3f} m/s")
        print(f"  {'total':<20s} = {sum(totals.values()):9.3f} m/s")
        print(f"\nTotal drift time across all legs: "
              f"{total_drift:.1f} days "
              f"({total_drift / 365.25:.2f} years)")

        # Explicit sanity check on the RAAN-residual claim
        n_nonzero = sum(1 for r in mission['log']
                        if r['breakdown']['raan_residual'] > 1e-6)
        print(f"Legs with RAAN residual > 1e-6 m/s: "
              f"{n_nonzero} / {len(mission['log'])}")

    return mission


def write_leg_csv(mission, path="tables/leg_breakdown_lambda.csv"):
    """Dump the per-leg log as CSV, including the RAAN-residual column."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "target", "total_dv_ms",
            "hohmann_sum_ms", "inc_sum_ms", "raan_residual_ms",
            "drift_alt_km", "drift_inc_deg", "drift_time_days",
            "raan_gap_initial_deg", "raan_residual_deg",
            "cumulative_spent_ms",
        ])
        cum = 0.0
        for k, (target, result) in enumerate(
                zip(mission['visited'], mission['log']), start=1):
            bd = result['breakdown']
            cum += bd['total']
            w.writerow([
                k, target['name'], f"{bd['total']:.3f}",
                f"{bd['hohmann_to_drift'] + bd['hohmann_return']:.3f}",
                f"{bd['inc_to_drift'] + bd['inc_return']:.3f}",
                f"{bd['raan_residual']:.3f}",
                f"{result['drift_alt_km']:.2f}",
                f"{result['drift_inc_deg']:.4f}",
                f"{result['drift_time_days']:.2f}",
                f"{result['raan_gap_initial_deg']:.4f}",
                f"{result['raan_residual_deg']:.4f}",
                f"{cum:.3f}",
            ])


if __name__ == "__main__":
    lam = DEFAULT_LAMBDA
    if len(sys.argv) > 1:
        try:
            lam = float(sys.argv[1])
        except ValueError:
            print(f"Could not parse lambda from '{sys.argv[1]}', "
                  f"using default {DEFAULT_LAMBDA}")

    targets = load_debris_catalogue()
    mission = create_mission(dv_budget=500.0)
    run_mission_verbose(mission, targets, time_penalty=lam)

    csv_path = f"tables/leg_breakdown_lambda_{lam:g}.csv"
    write_leg_csv(mission, csv_path)
    print(f"\nAlso wrote: {csv_path}")
