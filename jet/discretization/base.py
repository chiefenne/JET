"""Abstract base class for discretization schemes."""

from abc import ABC, abstractmethod

import numpy as np


class Discretization(ABC):
    """Base class for spatial discretization schemes.

    Subclasses must implement build_residual_function() which returns
    a callable F(unknowns) -> residuals for the nonlinear solver.
    """

    @abstractmethod
    def build_residual_function(self, state_old, b, e, b_old, e_old,
                                 nx, gsi, deta, etamax):
        """Build the residual function for the nonlinear system.

        Args:
            state_old: dict with previous station profiles (f, u, v, g, p)
            b, e: Current viscosity/diffusivity arrays
            b_old, e_old: Previous station viscosity/diffusivity
            nx: Current station index
            gsi: gsi coordinate array
            deta: eta spacing array
            etamax: Number of eta grid points

        Returns:
            F: callable(unknowns_1d) -> residuals_1d
            guess: initial guess as 1d array
        """
        pass

    @staticmethod
    def pack(f, u, v, g, p):
        """Pack field arrays into a single 1D vector."""
        return np.concatenate([f, u, v, g, p])

    @staticmethod
    def unpack(vector, etamax):
        """Unpack a 1D vector into field arrays."""
        f = vector[0 * etamax:1 * etamax]
        u = vector[1 * etamax:2 * etamax]
        v = vector[2 * etamax:3 * etamax]
        g = vector[3 * etamax:4 * etamax]
        p = vector[4 * etamax:5 * etamax]
        return f, u, v, g, p
