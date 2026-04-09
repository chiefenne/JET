"""Keller BOX discretization scheme.

Reference: Cebeci & Bradshaw, "Physical and Computational Aspects
of Convective Heat Transfer", Chapter 13-14.
"""

import numpy as np

from .base import Discretization


class KellerBox(Discretization):
    """Keller BOX scheme for parabolic boundary layer equations."""

    name = 'keller_box'

    def build_residual_function(self, state_old, b, e, b_old, e_old,
                                 nx, gsi, deta, etamax):
        f_o = state_old['f'].copy()
        u_o = state_old['u'].copy()
        v_o = state_old['v'].copy()
        g_o = state_old['g'].copy()
        p_o = state_old['p'].copy()

        def F(unknowns):
            f, u, v, g, p = self.unpack(unknowns, etamax)

            alpha = (3.0 / 2.0 * (gsi[nx] + gsi[nx - 1]) /
                     (gsi[nx] - gsi[nx - 1]))

            eq1 = np.zeros(etamax)
            eq2 = np.zeros(etamax)
            eq3 = np.zeros(etamax)
            eq4 = np.zeros(etamax)
            eq5 = np.zeros(etamax)

            # boundary conditions at jet center (symmetry)
            f[0] = 0.0
            v[0] = 0.0
            p[0] = 0.0
            f_o[0] = 0.0
            v_o[0] = 0.0
            p_o[0] = 0.0

            # boundary conditions at jet outer boundary (ambient)
            u[-1] = 0.0
            g[-1] = 0.0
            u_o[-1] = 0.0
            g_o[-1] = 0.0

            # ODE: f' = u
            eq1[1:] = (1.0 / deta * (f[1:] - f[:-1]) -
                        0.5 * (u[1:] + u[:-1]))

            # ODE: u' = v
            eq2[1:] = (1.0 / deta * (u[1:] - u[:-1]) -
                        0.5 * (v[1:] + v[:-1]))

            # ODE: g' = p
            eq3[1:] = (1.0 / deta * (g[1:] - g[:-1]) -
                        0.5 * (p[1:] + p[:-1]))

            # PDE: Momentum equation
            m1 = 1.0 / deta * (b[1:] * v[1:] - b[:-1] * v[:-1])
            m2 = (1.0 - alpha) * 0.5 * (u[1:]**2 + u[:-1]**2)
            m3 = (1.0 + alpha) * 0.5 * (f[1:] * v[1:] + f[:-1] * v[:-1])
            m4 = alpha * 0.25 * ((v_o[1:] + v_o[:-1]) * (f[1:] + f[:-1]) -
                                  (v[1:] + v[:-1]) * (f_o[1:] + f_o[:-1]))
            m5 = 1.0 / deta * (b_old[1:] * v_o[1:] - b_old[:-1] * v_o[:-1])
            m6 = (1.0 + alpha) * 0.5 * (u_o[1:]**2 + u_o[:-1]**2)
            m7 = (1.0 - alpha) * 0.5 * (f_o[1:] * v_o[1:] +
                                          f_o[:-1] * v_o[:-1])
            eq4[1:] = m1 + m2 + m3 + m4 + m5 + m6 + m7

            # PDE: Energy equation
            e1 = 1.0 / deta * (e[1:] * p[1:] - e[:-1] * p[:-1])
            e2 = (1.0 + alpha) * 0.5 * (f[1:] * p[1:] + f[:-1] * p[:-1])
            e3 = alpha * (
                0.5 * (u[1:] * g[1:] + u[:-1] * g[:-1]) +
                0.25 * ((u_o[1:] + u_o[:-1]) * (g[1:] + g[:-1]) -
                         (u[1:] + u[:-1]) * (g_o[1:] + g_o[:-1]) +
                         (p[1:] + p[:-1]) * (f_o[1:] + f_o[:-1]) -
                         (p_o[1:] + p_o[:-1]) * (f[1:] + f[:-1])))
            e4 = 1.0 / deta * (e_old[1:] * p_o[1:] -
                               e_old[:-1] * p_o[:-1])
            e5 = ((1.0 - alpha) * 0.5 *
                  (f_o[1:] * p_o[1:] + f_o[:-1] * p_o[:-1]))
            e6 = alpha * 0.5 * (u_o[1:] * g_o[1:] + u_o[:-1] * g_o[:-1])
            eq5[1:] = e1 + e2 - e3 + e4 + e5 + e6

            # boundary conditions as 0-th element equations
            eq1[0] = f[0]
            eq2[0] = v[0]
            eq3[0] = p[0]
            eq4[0] = u[-1]
            eq5[0] = g[-1]

            return np.concatenate([eq1, eq2, eq3, eq4, eq5])

        guess = self.pack(f_o, u_o, v_o, g_o, p_o)
        return F, guess


# Registry of available discretization schemes
DISCRETIZATIONS = {
    'keller_box': KellerBox,
}


def get_discretization(name: str) -> Discretization:
    """Get a discretization scheme by name."""
    if name not in DISCRETIZATIONS:
        raise ValueError(
            f"Unknown discretization '{name}'. "
            f"Available: {list(DISCRETIZATIONS.keys())}")
    return DISCRETIZATIONS[name]()
