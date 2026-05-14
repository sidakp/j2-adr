# Shared constants
# Physical constants used throughout the ADR mission model.
# All values are stored in SI units unless the name says otherwise.

import numpy as np

"""
1) Define the Earth and gravity constants that are reused by the orbital
state, propagator, manoeuvre, transfer-leg, and mission-sequencing modules.

2) Keep the conversion factors in one place so the rest of the code can
convert between catalogue units (km and degrees) and propagation units
(metres and radians) without duplicating constants.
"""

# Standard gravitational parameter of Earth
MU = 3.986004418e14  # m^3/s^2
R_E = 6378.137e3  # m (WGS-84 equatorial radius)
J2 = 1.08263e-3  # Dimensionless

# Unit conversion factors
DEG_TO_RAD = np.pi / 180.0  
RAD_TO_DEG = 180.0 / np.pi
DAY_TO_SEC = 86400.0
SEC_TO_DAY = 1.0 / DAY_TO_SEC
KM_TO_M = 1000.0
M_TO_KM = 0.001
