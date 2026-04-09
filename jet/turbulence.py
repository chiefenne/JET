"""Turbulence models."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TurbulenceModel:
    """Eddy viscosity turbulence model for the free jet.

    Based on Cebeci & Bradshaw algebraic model.
    """

    def __init__(self, Prandtl: float, Prandtl_turb: float, turbulent: bool):
        self.Prandtl = Prandtl
        self.Prandtl_turb = Prandtl_turb
        self.turbulent = turbulent
        self.urel_visc = 0.0

    def compute(self, u, eta, gsi_val, nx, Reynolds, etamax):
        """Compute eddy viscosity and diffusivity arrays.

        Args:
            u: Velocity profile array (etamax,)
            eta: Eta coordinate array
            gsi_val: Current gsi value
            nx: Current station index
            Reynolds: Reynolds number
            etamax: Number of eta grid points

        Returns:
            b: Effective viscosity array (etamax,)
            e: Effective diffusivity array (etamax,)
        """
        umaxh = 0.5 * u[0]

        # find half-width location
        index = np.where(u <= umaxh)
        if len(index[0]) > 0:
            j = index[0][0]
            etab = (eta[j - 1] + (eta[j] - eta[j - 1]) /
                    (u[j] - u[j - 1]) * (umaxh - u[j - 1]))
        else:
            etab = eta[-1]

        if self.turbulent:
            eddy_viscosity = (0.037 * etab * u[0] *
                              np.sqrt(Reynolds) * gsi_val**(1. / 3.))
        else:
            eddy_viscosity = 0.0

        # under-relaxation
        self.urel_visc = np.tanh(gsi_val - 0.4)
        eddy_viscosity *= self.urel_visc

        logger.info('  Eddy viscosity under-relaxation = %.3f', self.urel_visc)

        b = np.full(etamax, 1.0 + eddy_viscosity)
        e = np.full(etamax, 1.0 / self.Prandtl +
                    eddy_viscosity / self.Prandtl_turb)

        return b, e
