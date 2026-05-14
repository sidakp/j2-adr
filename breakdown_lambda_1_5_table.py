# Lambda = 1.5 final-solution table generator.
# This mirrors the mission loop used by breakdown.py, but prints a compact
# LaTeX-ready table for the selected time-penalised mission case.

import csv

from constants import DAY_TO_SEC
from mission import compute_all_costs, create_mission, load_debris_catalogue


LAMBDA = 1.5
CSV_PATH = "tables/leg_breakdown_lambda_1.5_with_norad.csv"
TABULAR_PATH = "tables/leg_breakdown_lambda_1.5_tabular.tex"


def escape_latex(text):
    """Escape the small subset of characters expected in object names."""
    return str(text).replace("&", r"\&")


def run_mission_for_lambda(mission, targets, time_penalty=LAMBDA):
    """Run the greedy sequencer with a fixed time penalty."""
    remaining = list(targets)

    while remaining:
        ranked = compute_all_costs(
            mission["mothership"], remaining, time_penalty
        )

        if not ranked:
            break

        best = ranked[0]
        dv_cost = best["result"]["total_dv"]
        dv_remaining = mission["dv_budget"] - mission["dv_spent"]

        if dv_cost > dv_remaining:
            break

        mission["dv_spent"] += dv_cost
        mission["visited"].append(best["target"])
        mission["log"].append(best["result"])

        # Advance RAAN geometry in the same way as mission.py.
        leg_dt_s = best["result"]["drift_time_days"] * DAY_TO_SEC
        mission["mothership"].raan += (
            mission["mothership"].raan_dot_j2 * leg_dt_s
        )
        for target in remaining:
            target["state"].raan += target["state"].raan_dot_j2 * leg_dt_s

        remaining.remove(best["target"])

    return mission


def iter_table_rows(mission):
    """Yield rows for the compact final-solution table."""
    cumulative_dv = 0.0
    for step, (target, result) in enumerate(
        zip(mission["visited"], mission["log"]), start=1
    ):
        total_dv = result["breakdown"]["total"]
        cumulative_dv += total_dv
        yield {
            "step": step,
            "target": target["name"],
            "norad_id": int(target["norad_id"]),
            "total_dv_ms": total_dv,
            "drift_time_days": result["drift_time_days"],
            "cumulative_spent_ms": cumulative_dv,
        }


def write_csv(rows, path=CSV_PATH):
    """Write the final-solution table as CSV for checking."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "target",
                "norad_id",
                "total_dv_ms",
                "drift_time_days",
                "cumulative_spent_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "step": row["step"],
                "target": row["target"],
                "norad_id": row["norad_id"],
                "total_dv_ms": f"{row['total_dv_ms']:.3f}",
                "drift_time_days": f"{row['drift_time_days']:.2f}",
                "cumulative_spent_ms": f"{row['cumulative_spent_ms']:.3f}",
            })


def build_tabular_rows(rows):
    """Return booktabs table rows suitable for direct LaTeX insertion."""
    lines = []
    total_drift = 0.0
    total_dv = 0.0

    for row in rows:
        total_drift += row["drift_time_days"]
        total_dv = row["cumulative_spent_ms"]
        lines.append(
            f"{row['step']:>2} & {escape_latex(row['target'])} & "
            f"{row['norad_id']} & "
            f"{row['total_dv_ms']:.1f} & "
            f"{row['drift_time_days']:.0f} & "
            f"{row['cumulative_spent_ms']:.1f} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{3}{r}{\textit{Totals}} & "
        f"{total_dv:.1f} & {total_drift:.0f} & {total_dv:.1f} \\\\"
    )
    return lines


def print_table(rows):
    """Print a complete LaTeX table environment for Chapter 4."""
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(
        r"\caption{Selected mission sequence for $\lambda = 1.5$ "
        r"m\,s\textsuperscript{$-$1}\,day\textsuperscript{$-$1}. "
        r"Each row corresponds to one committed transfer leg; the NORAD "
        r"identifier distinguishes debris objects with the same catalogue "
        r"name. All $\Delta V$ in m/s; drift time in days.}"
    )
    print(r"\label{tab:lambda-1p5-solution}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\begin{tabular}{@{}c l c c c c@{}}")
    print(r"\toprule")
    print(
        r"\multicolumn{1}{c}{Step} & Object & NORAD & "
        r"$\Delta V_{\text{leg}}$ & $t_{\text{drift}}$ & "
        r"$\sum\Delta V$ \\"
    )
    print(r"\midrule")
    for line in build_tabular_rows(rows):
        print(line)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def write_tabular(rows, path=TABULAR_PATH):
    """Write only the tabular rows, matching the style of breakdown.py."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(build_tabular_rows(rows)) + "\n")


if __name__ == "__main__":
    targets = load_debris_catalogue()
    mission = create_mission(dv_budget=500.0)
    run_mission_for_lambda(mission, targets, LAMBDA)

    rows = list(iter_table_rows(mission))
    write_csv(rows)
    write_tabular(rows)
    print_table(rows)

    total_drift = sum(row["drift_time_days"] for row in rows)
    print()
    print(f"Targets removed: {len(rows)}")
    print(f"Delta-V spent: {mission['dv_spent']:.3f} m/s")
    print(f"Budget remaining: {500.0 - mission['dv_spent']:.3f} m/s")
    print(f"Total drift: {total_drift:.1f} days")
    print(f"Duration: {total_drift / 365.25:.2f} years")
    print(f"Wrote: {CSV_PATH}")
    print(f"Wrote: {TABULAR_PATH}")
