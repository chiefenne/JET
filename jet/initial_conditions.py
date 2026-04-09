"""Initial and boundary conditions for the jet flow."""

import numpy as np
from numpy import sqrt, tanh

from .mesh import Mesh


class InitialConditions:
    """Compute initial velocity and temperature profiles.

    Implements Equation (14.31) from Cebeci & Bradshaw.
    """

    @staticmethod
    def compute(mesh: Mesh, Reynolds: float, Prandtl: float, nx: int = 0):
        """Compute initial profiles at station nx.

        Returns:
            dict with arrays: f, u, v, g, p
        """
        etamax = mesh.etamax
        gsi0 = mesh.gsi[nx]

        f = np.zeros(etamax)
        u = np.zeros(etamax)
        v = np.zeros(etamax)
        g = np.zeros(etamax)
        p = np.zeros(etamax)

        beta = 27.855
        etac = 1.0
        term = beta * 3.0 * gsi0**(2. / 3.) / sqrt(Reynolds)

        for j in range(etamax):
            tanf = tanh(term * (mesh.eta[j] - etac))
            u[j] = 3.0 / 2.0 * gsi0**(1. / 3.) * (1.0 - tanf)
            v[j] = (-term * 3.0 / 2.0 * gsi0**(1. / 3.) *
                     (1.0 - tanf**2))
            g[j] = u[j] / 3.0 * gsi0**(1. / 3.)
            p[j] = v[j] / 3.0 * gsi0**(1. / 3.)
            f[j] = f[j - 1] + mesh.a[j] * (u[j] + u[j - 1])

        # symmetry at jet center
        f[0] = 0.0
        v[0] = 0.0
        p[0] = 0.0

        # ambient at jet boundary
        u[-1] = 0.0
        g[-1] = 0.0

        # smooth derivatives at jet center
        for _ in range(5):
            v[1:-1] = 0.5 * (v[2:] + v[:-2])
            p[1:-1] = 0.5 * (p[2:] + p[:-2])

        return {'f': f, 'u': u, 'v': v, 'g': g, 'p': p}
