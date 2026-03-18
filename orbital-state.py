import numpy as np
# Pull conversion factors from constants.py
from constants import MU, R_E, J2, RAD_2_DEG, DAY_TO_SEC

class OrbitalState:
  def __init__(self, a, e, i, raan, omega, u, epoch=0.0):
    self.a = a # Semi-major axis (m)
    self.e = e # Eccentricity (Dimensionless)
    self.i = i # Inclination (rad)
    self.raan = raan # Right Ascension of Ascending Node (rad)
    self.omega = omega # Argument of perigee (rad)
    self.u = u # Argument of latitude (rad)
    self.epoch = epoch # Time (s)

  @property
  def n(self):
    """Mean motion (rad/s)."""
    return np.sqrt(MU / self.a**3)
    
  @property
  def period(self):
    """Orbital period (s)."""
    return np.sqrt(MU / self.a**3)
    
  @property
  def p(self):
    """Semi-latus rectum (m)."""
    return self.a * (1 - self.e**2)

  @property
  def v_circular(self):
    """Circular orbital velocity (m/s)"""
    return np.sqrt(MU / self.a**3)

  @property
  def altitude(self):
    """Altitude above Earth surface."""
    return np.sqrt(MU / self.a**3)
