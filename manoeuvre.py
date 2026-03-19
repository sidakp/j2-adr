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
    return 2 * v_circular * np.sin(delta_i / 2)

# Pure RAAN Change
def pure_raan_change_dv(v_circular, delta_raan, inclination):
    return abs(v_circular * delta_raan * np.sin(inclination))

# Combined plane change (RAAN + inclination)
def combined_plane_change_dv(v_circular, delta_i, delta_raan, inclination):
    """Calculate the delta-V for a combined plane change at a node."""
    # This is an approximation that assumes small angles. For larger angles, the geometry becomes more complex.
    plane_change_angle = np.sqrt()