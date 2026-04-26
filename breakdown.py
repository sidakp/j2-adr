# Verbose-breakdown variant of mission.py
# Runs the same greedy sequencer but prints the full per-leg Delta-V
# breakdown (as returned by manoeuvre.compute_leg_dv) at every step,
# and an aggregate breakdown at the end.
#
# The mothership returns to its parking orbit after each leg, so its
# state is constant across iterations — the per-leg report reflects
# that architecture.

import csv

import numpy as np

from constants import MU, R_E, J2, DEG_TO_RAD, DAY_TO_SEC
from orbitalstate import OrbitalState
from mission import load_debris_catalogue, create_mission, compute_all_costs


def print_leg_breakdown(step, target, result, mothership, dv_spent, dv_budget):
    """Per-leg verbose report, styled after manoeuvre.py's __main__ output."""
    bd = result['breakdown']
    name = target['name']
    norad = target['norad_id']
    t_state = target['state']

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
    print(f"  TOTAL leg Delta-V   = {bd['total']:9.3f} m/s")

    print("Budget state:")
    print(f"  Leg cost            = {bd['total']:9.3f} m/s")
    print(f"  Spent cumulative    = {dv_spent:9.3f} / {dv_budget:.3f} m/s")
    print(f"  Remaining           = {dv_budget - dv_spent:9.3f} m/s")


def run_mission_verbose(mission, targets, time_penalty=0):
    """Greedy sequencing loop with a full per-leg breakdown at each step."""
    remaining = list(targets)
    step = 0

    m0 = mission['mothership']
    print(f"Mission start: {len(remaining)} targets, "
          f"{mission['dv_budget']:.1f} m/s Delta-V budget")
    print(f"Mothership parking orbit: "
          f"alt={m0.altitude / 1e3:.2f} km, "
          f"i={np.degrees(m0.i):.3f} deg, "
          f"RAAN={np.degrees(m0.raan):.3f} deg")

    while remaining:
        ranked = compute_all_costs(mission['mothership'], remaining, time_penalty)

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
    print(f"Mission complete: {len(mission['visited'])} targets removed")
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

        print(f"\nAggregate Delta-V by component (all legs):")
        for k, v in totals.items():
            print(f"  {k:<20s} = {v:9.3f} m/s")
        print(f"  {'total':<20s} = {sum(totals.values()):9.3f} m/s")

    return mission


def write_leg_csv(mission, path="tables/leg_breakdown.csv"):
    """Dump the per-leg log as CSV for downstream analysis."""
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


def write_leg_tabular_tex(mission, path="tables/leg_breakdown_tabular.tex"):
    """Emit the per-leg table as a booktabs tabular, ready for \\input.
    Only the rows + totals line — the caller wraps it in a table env."""
    lines = []
    cum = 0.0
    sum_total = 0.0
    sum_hoh = 0.0
    sum_inc = 0.0
    for k, (target, result) in enumerate(
            zip(mission['visited'], mission['log']), start=1):
        bd = result['breakdown']
        hoh = bd['hohmann_to_drift'] + bd['hohmann_return']
        inc = bd['inc_to_drift'] + bd['inc_return']
        cum += bd['total']
        sum_total += bd['total']
        sum_hoh   += hoh
        sum_inc   += inc
        name = target['name'].replace("&", r"\&")
        lines.append(
            f"{k} & {name} & "
            f"{bd['total']:.2f} & {hoh:.2f} & {inc:.2f} & "
            f"{result['drift_alt_km']:.2f} & "
            f"{result['drift_time_days']:.1f} & "
            f"{cum:.2f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{2}}{{r}}{{\\textit{{Totals}}}} & "
        f"{sum_total:.2f} & {sum_hoh:.2f} & {sum_inc:.2f} & "
        f"\\multicolumn{{3}}{{c}}{{}} \\\\"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    targets = load_debris_catalogue()
    mission = create_mission(dv_budget=500.0)
    run_mission_verbose(mission, targets, time_penalty=0)

    write_leg_csv(mission, "tables/leg_breakdown.csv")
    write_leg_tabular_tex(mission, "tables/leg_breakdown_tabular.tex")
    print("\nAlso wrote: tables/leg_breakdown.csv, tables/leg_breakdown_tabular.tex")
