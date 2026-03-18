# Phase 1 - Orbital Propagator using J2 perturbation

import numpy as np
# ODE solver
from scipy.integrate import solve_ivp
# Import the constants defined in constants.py
from constants import MU, R_E, J2, RAD_TO_DEG, DAY_TO_SEC
# Import the class defined in orbitalstate.py
from orbitalstate import OrbitalState

class GVEPropagator:
    def __init__(self, rtol=1e-10, atol=1e-12, max_step=300):
        self.rtol = rtol # Relative tolerance
        self.atol = atol # Absolute tolerance
        self.max_step = max_step # Maximum step size (s)
    
    def _gve_rhs(self, t, y, thrust_func):
        a, e, i, raan, omega, u = y

        # Formulae
        p = a * (1 - e**2) # Perigee
        n = np.sqrt(MU / a**3) # Mean motion
        h = np.sqrt(MU * p) # Specific angular momentum 
        theta = u - omega  # True anomaly
        r = p / (1 + e * np.cos(theta))  # Radius

        # Thrust accelerations. Should be 0
        if thrust_func is not None:
            f_r, f_c, f_n = thrust_func(t, y)  # Radial, tangential, normal thrust components
        else:
            f_r, f_c, f_n = 0.0, 0.0, 0.0
        
        sin_i = np.sin(i)
        cos_i = np.cos(i)
        sin_u = np.sin(u)
        cos_u = np.cos(u)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Eq (1): da/dt
        da = (2 * a**2 / h) * (e * sin_theta * f_r + (p / r) * f_c)

        # Eq (2): de/dt
        de = (1 / h) * (p * sin_theta * f_r
                        + ((p + r) * cos_theta + r * e) * f_c)

        # Eq (3): di/dt
        di = (r * cos_u / h) * f_n

        # Eq (4): dΩ/dt
        draan = (-3 * R_E**2 * J2 * n / (2 * p**2)) * cos_i
        if abs(sin_i) > 1e-12:
            draan += (r * sin_u / (h * sin_i)) * f_n

        # Eq (5): dω/dt
        domega = (3 * R_E**2 * J2 * n / (4 * p**2)) * (4 - 5 * sin_i**2)
        if abs(e) > 1e-10:
            domega += (p / (h * e)) * (
                (1 + r / p) * sin_theta * f_c - cos_theta * f_r
            )
        if abs(np.tan(i)) > 1e-12:
            domega -= (r * sin_u / (h * np.tan(i))) * f_n

        # Eq (6): du/dt
        e_factor = np.sqrt(1 - e**2)
        du = h / r**2 \
             - (3 * R_E**2 * n * J2 / (4 * p**2)) * (
                 sin_i**2 * (5 - 3 * e_factor)
                 + (2 * e_factor - 4))
        if abs(np.tan(i)) > 1e-12:
            du -= (r * sin_u / (h * np.tan(i))) * f_n

        return np.array([da, de, di, draan, domega, du])
    
    # Propagate method
    def propagate(self, initial_state, duration, thrust_func=None):
        y0 = initial_state.to_vector()

        sol = solve_ivp(
            fun=lambda t, y: self._gve_rhs(t, y, thrust_func),
            t_span=(0.0, duration),
            y0=y0,
            method='DOP853', # 8th order method
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            dense_output=True
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        final_state = OrbitalState.from_vector(
            sol.y[:, -1],
            epoch=initial_state.epoch + duration
        )

        return final_state, sol
    
if __name__ == "__main__":
    # Define a sun-synchronous orbit at 700 km
    state = OrbitalState(
        a=R_E + 700e3,
        e=0.0001,
        i=np.radians(98.19),
        raan=np.radians(45.0),
        omega=0.0,
        u=0.0
    )

    print(f"Initial orbit: {state.altitude/1e3:.0f} km, "
          f"i = {np.degrees(state.i):.2f} deg")
    print(f"Analytical RAAN rate: "
          f"{state.raan_dot_j2 * RAD_TO_DEG * DAY_TO_SEC:.6f} deg/day")

    # Propagate 30 days under J2 only (no thrust)
    prop = GVEPropagator()
    final, sol = prop.propagate(state, 30 * DAY_TO_SEC)

    # Compare numerical result to analytical prediction
    numerical = np.degrees(final.raan - state.raan)
    analytical = np.degrees(state.raan_dot_j2) * 30 * DAY_TO_SEC

    print(f"\n30-day RAAN drift:")
    print(f"Numerical (GVE): {numerical:.6f} deg")
    print(f"Analytical (secular): {analytical:.6f} deg")
    print(f"Difference: {abs(numerical - analytical):.8f} deg")

    # Check for 0 change in a, e, i
    print(f"\nChecks for 0 thrust propagation:")
    print(f"Delta-a: {(final.a - state.a)/1e3:.6f} km")
    print(f"Delta-e: {final.e - state.e:.8f}")
    print(f"Delta-i: {np.degrees(final.i - state.i):.8f} deg")