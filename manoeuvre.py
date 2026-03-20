# Parsing Delta-V calculations here
# Equations to show fuel cost to perform orbital manoeuvres

import numpy as np
from constants import MU, R_E, J2

def hohmann_delta_v(r1, r2):
    """Calculate the delta-V for a Hohmann transfer between two circular orbits."""
    # Calculate the semi-major axis of the transfer orbit
    a_transfer = (r1 + r2) / 2
    
    # Vis-Viva equation for a circular orbit. It gives the velocity of the mothership in its initial circular orbit.
    v1_circular = np.sqrt(MU / r1)
    # Full Vis-Viva equation. Gives the velocity needed at radius r1 to enter the transfer orbit.
    v1_transfer = np.sqrt(MU * (2 / r1 - 1 / a_transfer))
    
    dv1 = abs(v1_transfer - v1_circular)  # Delta-V for the first burn

    v2_circular = np.sqrt(MU / r2)  # Velocity of the target in its circular orbit
    v2_transfer = np.sqrt(MU * (2 / r2 - 1 / a_transfer))  # Velocity needed at radius r2 to enter the transfer orbit
    dv2 = abs(v2_circular - v2_transfer)  # Delta-V for the second burn

    # The value I'm interested in is the sum. The breakdown is helpful for debugging
    return dv1, dv2, dv1 + dv2
    
def hohmann_transfer_time(r1, r2):
    """Calculate the time of flight for a Hohmann transfer between two circular orbits."""
    # A Hohmann transfer traverses exactly half the transfer ellipse, so the flight time is half the orbital period
    a_transfer = (r1 + r2) / 2
    period_transfer = 2 * np.pi * np.sqrt(a_transfer**3 / MU)
    return period_transfer / 2  # Time to go from r1 to r2 is half the period of the transfer orbit

# Pure Inclination Change - I am defining pure here as a change in inclination using a single burn at a node.
# Potentially including in the report to justify the use of J2 perturbations
def pure_inclination_dv(v_circular, delta_i):
    """Calculate the delta-V for a pure inclination change at a node."""
    return abs(2 * v_circular * np.sin(delta_i / 2))

# Pure RAAN Change
def pure_raan_change_dv(v_circular, delta_raan, inclination):
    return abs(v_circular * delta_raan * np.sin(inclination))

# Combined plane change (RAAN + inclination)
def combined_plane_change_dv(v_circular, delta_i, delta_raan, inclination):
    """Calculate the delta-V for a combined plane cha    nge at a node."""
    # This is an approximation that assumes small angles. For larger angles, the geometry becomes more complex.
    plane_change_angle = np.sqrt(delta_i**2 + (delta_raan * np.sin(inclination))**2)
    dv = abs(v_circular * plane_change_angle)

    if abs (delta_i) > 1e-14:
        u_star = np.arctan2(delta_raan * np.sin(inclination), delta_i)
    else:
        u_star = np.pi / 2 if delta_raan > 0 else -np.pi / 2 
        # The nested if statement prevents division by 0 in cases there may be 0 inclination change

    # Return both the cost and the optimal burn location
    return dv, u_star

def compute_leg_dv(a_mothership, i_mothership, a_drift, i_drift, a_target,
                   i_target, delta_raan_residual):
    # The _, _, ignores the individual burns and just computes the total
    # 1) Hohmann transfer to drift altitude
    _, _, dv_hohmann_up = hohmann_delta_v(a_mothership, a_drift)
    # 2) Inclination change to drift value
    v_at_drift = np.sqrt(MU / a_drift)
    dv_inc_to_drift = pure_inclination_dv(v_at_drift, i_drift - i_mothership)

    # Hohmann transfer from drift to target                                      
    _, _, dv_hohmann_down = hohmann_delta_v(a_drift, a_target)

    # Inclination change from drift to target
    v_at_target = np.sqrt(MU / a_target)
    dv_inc_to_target = pure_inclination_dv(v_at_target, i_target - i_drift)

    # Residual RAAN - This is the cost of the RAAN gap the J2 drift didn't close
    dv_residual = pure_raan_change_dv(v_at_target, delta_raan_residual, i_target)

    total = (dv_hohmann_up + dv_inc_to_drift + dv_hohmann_down + dv_inc_to_target + dv_residual)

    breakdown = {
        'hohmann_to_drift' : dv_hohmann_up,
        'inc_to_drift' : dv_inc_to_drift,
        'hohmann_to_target' : dv_hohmann_down,
        'inc_to_target' : dv_inc_to_target,
        'raan_residual' : dv_residual,
        'total' : total
    }

    # The breakdown allows to see where the fuel is allocated
    return total, breakdown

if __name__ == "__main__":
    # Test: Hohmann transfer from 680 km to 650 km
    r1 = R_E + 680e3
    r2 = R_E + 650e3
    dv1, dv2, total = hohmann_delta_v(r1, r2)
    tof = hohmann_transfer_time(r1, r2)
    
    print("Hohmann transfer: 680 km -> 650 km")
    print(f"Burn 1: {dv1:.2f} m/s")
    print(f"Burn 2: {dv2:.2f} m/s")
    print(f"Total:  {total:.2f} m/s")
    print(f"Transfer time: {tof/60:.1f} minutes")
    
    # Test: Direct plane change cost WITHOUT J2 drift
    v = np.sqrt(MU / r2)  # velocity at 650 km
    delta_raan = np.radians(3.0)  # 3 degree RAAN gap
    delta_i = np.radians(0.2)     # 0.2 degree inclination difference
    
    dv_raan = pure_raan_change_dv(v, delta_raan, np.radians(98.0))
    dv_inc = pure_inclination_dv(v, delta_i)
    dv_combined, u_star = combined_plane_change_dv(v, delta_i, delta_raan, np.radians(98.0))
    
    print(f"\nPlane change costs at 650 km (3° RAAN, 0.2° inc):")
    print(f"Pure RAAN change:       {dv_raan:.2f} m/s")
    print(f"Pure inclination change: {dv_inc:.2f} m/s")
    print(f"Sequential (sum):       {dv_raan + dv_inc:.2f} m/s")
    print(f"Combined (single burn): {dv_combined:.2f} m/s")
    print(f"Savings from combining: {(dv_raan + dv_inc) - dv_combined:.2f} m/s")
    print(f"Optimal burn location:  {np.degrees(u_star):.2f} deg")
    
    # Test: Full leg cost
    print(f"\nFull leg cost (680 km -> drift at 720 km -> target at 650 km):")
    total, bd = compute_leg_dv(
        a_mothership=R_E + 680e3,
        i_mothership=np.radians(98.0),
        a_drift=R_E + 720e3,
        i_drift=np.radians(97.9),
        a_target=R_E + 650e3,
        i_target=np.radians(97.8),
        delta_raan_residual=np.radians(0.5)  # 0.5 deg residual
    )
    
    print(f"Hohmann to drift:    {bd['hohmann_to_drift']:.2f} m/s")
    print(f"Inc to drift:        {bd['inc_to_drift']:.2f} m/s")
    print(f"Hohmann to target:   {bd['hohmann_to_target']:.2f} m/s")
    print(f"Inc to target:       {bd['inc_to_target']:.2f} m/s")
    print(f"RAAN residual:       {bd['raan_residual']:.2f} m/s")
    print(f"TOTAL:               {bd['total']:.2f} m/s")