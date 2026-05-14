# Phase 2 - Transfer leg
# Finds the optimal drift orbit to reach a debris target

import numpy as np
from constants import MU, R_E, J2, DAY_TO_SEC
from orbitalstate import OrbitalState
from propagator import GVEPropagator
from manoeuvre import compute_leg_dv

"""
1) TransferLeg prices one possible mothership-to-target leg. For each target,
the class searches over drift altitude and drift duration, then solves for
the drift inclination that gives the required J2 RAAN precession rate.

2) The returned total_dv is the Delta-V charged to the mission budget. The
ranked_cost adds the optional time penalty, so the same optimiser can be used
for the fuel-minimum baseline and the lambda sweep.

3) The mothership is assumed to return to its parking-orbit shape after each
leg. The RAAN state is advanced by mission.py once the selected drift time is
committed.
"""

class TransferLeg:
    def __init__(self, mothership_state, debris_state, min_drift_alt=550e3, max_drift_alt=750e3,
                 min_drift_days=5, max_drift_days=300, time_penalty=0):
        self.mothership_state = mothership_state.copy()
        self.debris_state = debris_state.copy()
        self.min_drift_alt = min_drift_alt
        self.max_drift_alt = max_drift_alt
        self.min_drift_days = min_drift_days
        self.max_drift_days = max_drift_days
        self.time_penalty = time_penalty  # Penalty weight in m/s per drift day
        self.propagator = GVEPropagator()

    """
    1) For a chosen drift duration, compute the RAAN rate the mothership
    must have while drifting so that it arrives in the target plane at the
    same time as the target.
    """

    def _required_raan_rate(self, drift_time_s):
        """What RAAN precession rate does the drift orbit need to close the gap
        with the debris in the given time?"""
        raan_gap = self.debris_state.raan - self.mothership_state.raan

        debris_raan_rate = self.debris_state.raan_dot_j2

        required_rate = debris_raan_rate + raan_gap / drift_time_s

        return required_rate, raan_gap
    
    """
    1) Rearrange the J2 RAAN-rate equation to find the inclination that
    produces the required drift rate at the chosen altitude.

    2) If the required cosine lies outside [-1, 1], that altitude-time pair
    cannot close the RAAN gap and is rejected by returning None.
    """

    def _drift_inclination(self, a_drift, required_raan_rate):
        """What inclination makes the drift orbit precess at the required rate?"""
        p_drift = a_drift  # Assumes circular orbit (e ≈ 0), where p = a
        n_drift = np.sqrt(MU / a_drift**3)

        cos_i = required_raan_rate / (-1.5 * n_drift * J2 * (R_E / p_drift)**2)

        if abs(cos_i) > 1.0:
            return None  # No solution, geometry is impossible. No inclination at that altitude can achieve the required RAAN rate. 
        
        return np.arccos(cos_i)
    
    """
    1) For one candidate drift altitude and drift time, solve the required
    drift inclination, reject infeasible geometries, and price the resulting
    parking-drift-parking transfer.

    2) The returned cost is the composite ranking cost. The Delta-V-only cost
    remains inside the breakdown and is the value charged to the mission
    budget by mission.py.
    """

    def compute_cost(self, drift_alt, drift_time_days):
        """For a specific drift altitude and time, compute the total delta-V cost of the transfer
        and the composite ranking cost."""
        drift_time_s = drift_time_days * DAY_TO_SEC # Seconds
        a_drift = R_E + drift_alt

        required_rate, raan_gap = self._required_raan_rate(drift_time_s)

        i_drift = self._drift_inclination(a_drift, required_rate)
        if i_drift is None:
            return np.inf, None  # No solution, geometry is impossible. Return infinite cost to exclude this option.
        
        drift_inc_deg = np.degrees(i_drift)
        if drift_inc_deg < 96.0 or drift_inc_deg > 100.0:
            return np.inf, None  # Exclude drift orbits outside the filtered SSO inclination band.
        
        mothership_raan_after = self.mothership_state.raan + required_rate * drift_time_s
        debris_raan_after = self.debris_state.raan + self.debris_state.raan_dot_j2 * drift_time_s

        delta_raan_residual = debris_raan_after - mothership_raan_after
        delta_raan_residual = (delta_raan_residual + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

        # Price out the full leg using manoeuvre.py
        total_dv, breakdown = compute_leg_dv(
            a_mothership=self.mothership_state.a,
            i_mothership=self.mothership_state.i,
            a_drift=a_drift,
            i_drift=i_drift,
            a_target=self.debris_state.a,
            i_target=self.debris_state.i,
            delta_raan_residual=delta_raan_residual
        )

        # Add metadata to the breakdown
        breakdown['drift_alt_km'] = drift_alt / 1e3
        breakdown['drift_inc_deg'] = np.degrees(i_drift)
        breakdown['drift_time_days'] = drift_time_days
        breakdown['raan_gap_initial_deg'] = np.degrees(raan_gap)
        breakdown['raan_residual_deg'] = np.degrees(delta_raan_residual)

        # Composite ranking cost used to compare candidate drift solutions for
        # this target. Only the delta-V component is charged to the mission
        # budget; the time-penalty term is a sequencing preference.
        total_cost = total_dv + self.time_penalty * drift_time_days

        return total_cost, breakdown
    
    def optimise(self, n_time_samples=30, n_alt_samples=20):
        """
        Search over drift altitudes and drift times to find the
        cheapest transfer. Returns a dict with the optimal parameters.
        """
        drift_times = np.linspace(self.min_drift_days, self.max_drift_days, n_time_samples)
        altitudes = np.linspace(self.min_drift_alt, self.max_drift_alt, n_alt_samples)

        best_cost = np.inf
        best_breakdown = None

        for t_days in drift_times:
            for alt in altitudes:
                cost, breakdown = self.compute_cost(alt, t_days)
                if cost < best_cost and breakdown is not None:
                    best_cost = cost
                    best_breakdown = breakdown

        if best_breakdown is None:
            return {'feasible': False}

        return {
            'feasible': True,
            'drift_alt_km': best_breakdown['drift_alt_km'],
            'drift_inc_deg': best_breakdown['drift_inc_deg'],
            'drift_time_days': best_breakdown['drift_time_days'],
            'total_dv': best_breakdown['total'],
            'ranked_cost': best_cost,
            'raan_gap_initial_deg': best_breakdown['raan_gap_initial_deg'],
            'raan_residual_deg': best_breakdown['raan_residual_deg'],
            'breakdown': best_breakdown
        }
    
if __name__ == "__main__":
    # Mothership at 680 km
    mothership = OrbitalState(
        a=R_E + 680e3, e=0.0001, i=np.radians(98.0),
        raan=np.radians(45.0), omega=0.0, u=0.0
    )

    # Debris at 650 km, 3 degrees ahead in RAAN
    debris = OrbitalState(
        a=R_E + 650e3, e=0.0001, i=np.radians(97.8),
        raan=np.radians(48.0), omega=0.0, u=0.0
    )

    print(f"Mothership: {mothership.altitude/1e3:.0f} km, "
          f"i={np.degrees(mothership.i):.2f} deg, "
          f"RAAN={np.degrees(mothership.raan):.2f} deg")
    print(f"Debris:     {debris.altitude/1e3:.0f} km, "
          f"i={np.degrees(debris.i):.2f} deg, "
          f"RAAN={np.degrees(debris.raan):.2f} deg")
    print(f"RAAN gap:   {np.degrees(debris.raan - mothership.raan):.2f} deg")

    # Fair direct baseline (no J2):
    # Match the J2-assisted leg's endpoint (mothership returns to its parking
    # orbit). The direct equivalent is therefore two combined plane changes at
    # the parking altitude: the first rotates the mothership onto the debris
    # plane so that mothership and debris are momentarily coplanar, the second
    # rotates it back to the original parking orientation. Altitude is not
    # changed in either manoeuvre, because the J2 leg does not transit to the
    # target's altitude either. Both manoeuvres are computed at parking-orbit
    # velocity, and both have the same angular magnitude, so the total is
    # simply twice the single-plane-change cost.
    from manoeuvre import combined_plane_change_dv
    delta_i = debris.i - mothership.i
    delta_raan = debris.raan - mothership.raan
    dv_one_plane_change, _ = combined_plane_change_dv(
        mothership.v_circular, delta_i, delta_raan, mothership.i
    )
    dv_direct = 2.0 * dv_one_plane_change

    print(f"\nDirect baseline (no J2, round-trip plane change): "
          f"{dv_direct:.2f} m/s")
    print(f"  Single plane change: {dv_one_plane_change:.2f} m/s "
          f"(forward and reverse at parking altitude)")

    # Optimise the J2-enhanced transfer
    print(f"\nOptimising J2-enhanced transfer...")
    leg = TransferLeg(mothership, debris)
    result = leg.optimise()

    if result['feasible']:
        print(f"\nOptimal solution:")
        print(f"  Drift altitude:    {result['drift_alt_km']:.1f} km")
        print(f"  Drift inclination: {result['drift_inc_deg']:.4f} deg")
        print(f"  Drift time:        {result['drift_time_days']:.1f} days")
        print(f"  Total delta-V:     {result['total_dv']:.2f} m/s")
        print(f"  RAAN residual:     {result['raan_residual_deg']:.4f} deg")

        savings = (1 - result['total_dv'] / dv_direct) * 100
        print(f"\n  Delta-V savings vs direct: {savings:.1f}%")

        bd = result['breakdown']
        print(f"\n  Breakdown:")
        print(f"    Hohmann to drift:     {bd['hohmann_to_drift']:.2f} m/s")
        print(f"    Inc to drift:         {bd['inc_to_drift']:.2f} m/s")
        print(f"    Hohmann return:       {bd['hohmann_return']:.2f} m/s")
        print(f"    Inc return:           {bd['inc_return']:.2f} m/s")
        print(f"    RAAN residual:        {bd['raan_residual']:.2f} m/s")
    else:
        print("  No feasible solution found")
