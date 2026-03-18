import numpy as np
# Pull conversion factors from constants.py
from constants import MU, R_E, J2, RAD_TO_DEG, DAY_TO_SEC

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
    return 2 * np.pi / self.n
    
  @property
  def p(self):
    """Semi-latus rectum (m)."""
    return self.a * (1 - self.e**2)

  @property
  def h(self):
    """Specific angular momentum (m^2/s)."""
    return np.sqrt(MU * self.p)  

  @property
  def v_circular(self):
    """Circular orbital velocity (m/s)"""
    return np.sqrt(MU / self.a)

  @property
  def altitude(self):
    """Altitude above Earth surface (m)."""
    return self.a - R_E

  @property
  def raan_dot_j2(self):
    """Secular J2 RAAN precession rate (rad/s)."""
    return -1.5 * self.n * J2 * (R_E / self.p)**2 * np.cos(self.i)
  
  # Turns into a raw numpy array. The ODE integrator needs this
  def to_vector(self):
    """Return state as numpy array (a, e, i, raan, omega, u)."""
    return np.array([self.a, self.e, self.i, self.raan, self.omega, self.u])
  
  @classmethod
  def from_vector(cls, vec, epoch=0.0):
    """Create OrbitalState from numpy array (a, e, i, raan, omega, u)."""
    return cls(a=vec[0], e=vec[1], i=vec[2], raan=vec[3], omega=vec[4], u=vec[5], epoch=epoch)
  
  def copy(self):
    """Return a copy of the current state."""
    return OrbitalState(self.a, self.e, self.i, self.raan, self.omega, self.u, self.epoch)