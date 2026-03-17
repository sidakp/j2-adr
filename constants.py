# Physical constants used throughout the ADR mission framework
# All values are in SI units
# Conversion factors present to assist with unit conversions when necessary

import numpy as np

# Gravitational constant
MU = 3.986004418e14  # m^3/s^2
R_E = 6371e3  # m
J2 = 1.08263e-3  # Dimensionless

# Conversion Factors
DEG_TO_RAD = np.pi / 180.0  
RAD_TO_DEG = 180.0 / np.pi
DAY_TO_SEC = 86400.0
SEC_TO_DAY = 1.0 / DAY_TO_SEC
KM_TO_M = 1000.0
M_TO_KM = 0.001