"""SciPy-based nonlinear solvers."""

import logging

from scipy import optimize

from .base import Solver, SolverResult

logger = logging.getLogger(__name__)


class FsolveSolver(Solver):
    name = 'fsolve'

    def solve(self, F, guess, tolerance=1e-06, **kwargs):
        result = optimize.fsolve(F, guess, full_output=True, xtol=tolerance)
        solution = result[0]
        message = result[3]
        logger.info('  Solver: %s', message)
        return SolverResult(solution=solution, message=message)


class NewtonKrylovSolver(Solver):
    name = 'newton_krylov'

    def solve(self, F, guess, iterations=100, **kwargs):
        solution = optimize.newton_krylov(
            F, guess, method='lgmres', verbose=1, iter=iterations)
        return SolverResult(
            solution=solution, message='Newton-Krylov converged')


class Broyden1Solver(Solver):
    name = 'broyden1'

    def solve(self, F, guess, tolerance=1e-06, iterations=100, **kwargs):
        solution = optimize.broyden1(
            F, guess, f_tol=tolerance, iter=iterations)
        return SolverResult(
            solution=solution, message='Broyden1 converged')


# Registry of available solvers
SOLVERS = {
    'fsolve': FsolveSolver,
    'newton_krylov': NewtonKrylovSolver,
    'broyden1': Broyden1Solver,
}


def get_solver(name: str) -> Solver:
    """Get a solver by name."""
    if name not in SOLVERS:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {list(SOLVERS.keys())}")
    return SOLVERS[name]()
