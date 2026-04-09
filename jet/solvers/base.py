"""Abstract base class for nonlinear solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SolverResult:
    """Result from a nonlinear solver."""
    solution: np.ndarray
    message: str
    success: bool = True


class Solver(ABC):
    """Base class for nonlinear equation solvers."""

    name: str = ''

    @abstractmethod
    def solve(self, F, guess, **kwargs) -> SolverResult:
        """Solve F(x) = 0.

        Args:
            F: Residual function callable(x) -> residuals
            guess: Initial guess vector

        Returns:
            SolverResult with solution vector and status
        """
        pass
