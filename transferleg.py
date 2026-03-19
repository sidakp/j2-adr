# Phase 2 - Transfer leg
# Finds the optimal drift orbit to reach a debris target

import numpy as np
from constants import MU, R_E, J2, DAY_TO_SEC
from orbitalstate import OrbitalState
from propagator import GVEPropagator
from manoeuvre import compute_leg_dv

