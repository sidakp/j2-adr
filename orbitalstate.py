import numpy as np
from constants import MU, R_E, J2, RAD_TO_DEG, DAY_TO_SEC

"""
1) OrbitalState is the common container for the six classical orbital
elements used throughout the project. Angles are stored in radians and
semi-major axis is stored in metres.

2) The class also provides derived quantities such as mean motion, orbital
period, circular velocity, and the secular J2 RAAN precession rate. This
keeps the equations in the transfer and mission modules shorter and reduces
the chance of unit mistakes.

3) The to_vector and from_vector helpers exist because scipy's ODE solver
expects raw numpy arrays, while the rest of the code is easier to read when
using named orbital-state fields.
"""

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
  
  # Turns the state into the raw array format expected by the ODE integrator
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
