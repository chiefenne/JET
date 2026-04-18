"""Main simulation orchestrator."""

import os
import logging

from .config import SimulationConfig
from .case_resolution import resolve_case
from .mesh import Mesh
from .initial_conditions import InitialConditions
from .turbulence import TurbulenceModel
from .discretization import get_discretization
from .solvers import get_solver
from .results import StationResult, SimulationResults
from .plotting import (plot_profiles, plot_dimensional_profiles,
                       plot_dimensionless_summary,
                       plot_dimensional_summary)

logger = logging.getLogger(__name__)


class Simulation:
    """Orchestrates a 2D turbulent heated free jet simulation.

    Usage:
        config = SimulationConfig()
        sim = Simulation(config)
        sim.run()
        sim.save()
        sim.plot()
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._setup_logging()
        self._print_banner()

        # resolve the user-facing inputs into one internal dimensionless case
        resolved_case = resolve_case(config)
        self.case_mode = resolved_case.mode
        self.fluid_name = resolved_case.fluid_name
        self.Reynolds = resolved_case.Reynolds
        self.Prandtl = resolved_case.Prandtl
        self.Prandtl_turb = resolved_case.Prandtl_turb
        self.turbulent = resolved_case.turbulent
        self.reference_conditions = resolved_case.reference_conditions

        # build mesh
        self.mesh = Mesh(config.mesh)

        # initialize components
        self.turbulence = TurbulenceModel(
            self.Prandtl, self.Prandtl_turb, self.turbulent)
        self.discretization = get_discretization(config.discretization)
        self.solver = get_solver(config.solver.solver_type)

        # results
        self.results = SimulationResults(
            reference_conditions=self.reference_conditions)

    def _setup_logging(self):
        level = {0: logging.WARNING, 1: logging.INFO,
                 2: logging.DEBUG}.get(self.config.output.verbosity,
                                       logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(message)s',
            force=True)

    def _print_banner(self):
        logger.info('')
        logger.info('*' * 50)
        logger.info('*' * 50)
        logger.info(' 2D TURBULENT HEATED FREE JET '.center(50, '*'))
        logger.info('*' * 50)
        logger.info('*' * 50)

    def _print_flow_info(self):
        number_label = 'input' if self.case_mode == 'dimensionless' else 'derived'

        logger.info('')
        logger.info('*' * 30)
        logger.info(' FLOW PROPERTIES '.center(30, '*'))
        logger.info('*' * 30)
        logger.info(' CASE MODE = %s', self.case_mode.upper())
        if self.case_mode == 'dimensionless':
            logger.info(' DIMENSIONAL (PHYSICAL) INPUTS = not used')
        else:
            logger.info(' DIMENSIONLESS GROUPS = derived from dimensional '
                        'physical inputs')
        logger.info(' REYNOLDS (%s) = %s', number_label, self.Reynolds)
        logger.info(' PRANDTL (%s) = %s', number_label, self.Prandtl)
        logger.info(' PRANDTL turbulent = %s', self.Prandtl_turb)
        if self.reference_conditions is not None:
            ref = self.reference_conditions
            logger.info(' FLUID = %s', self.fluid_name)
            logger.info(' NOZZLE WIDTH [m] = %s', 2.0 * ref.half_width_m)
            logger.info(' NOZZLE HALF-WIDTH [m] = %s', ref.half_width_m)
            logger.info(' NOZZLE VELOCITY [m/s] = %s',
                        ref.nozzle_velocity_m_s)
            logger.info(' NOZZLE TEMPERATURE [C] = %s',
                        ref.nozzle_temperature_C)
            logger.info(' AMBIENT TEMPERATURE [C] = %s',
                        ref.ambient_temperature_C)
            logger.info(' FLUID PROPERTIES EVALUATED AT [C] = %s',
                        ref.nozzle_temperature_C)
            logger.info(' NOZZLE-TO-AMBIENT DELTA T [K] = %s',
                        ref.reference_temperature_difference_K)
        logger.info(' TURBULENCE = %s',
                     'ON' if self.turbulent else 'OFF')

    def _has_dimensional_reference(self):
        return self.reference_conditions is not None

    def _log_dimensional_skip(self, action: str):
        logger.info('Skipping %s: dimensional outputs require physical mode.',
                    action)

    def _print_station_header(self, nx):
        text = (f' Jet propagation: GSI = {self.mesh.gsi[nx]} '
                f'at stage {nx}')
        text += (' - TURBULENT Flow' if self.turbulent
                 else ' - LAMINAR Flow')
        logger.info('')
        logger.info('')
        logger.info('*' * len(text))
        logger.info(text)
        if nx == 0:
            logger.info(' Initial velocity profile')
        logger.info('*' * len(text))
        logger.info('')

    def _print_station_result(self, state, b, e):
        logger.info('  Viscosity (B)   = % .4e', b[0])
        logger.info('  Diffusivity (E) = % .4e', e[0])
        logger.info('')
        logger.info('   ' + '=' * 67)
        logger.info('  %2s %6s  %10s  %10s  %10s  %10s  %10s',
                     'J', 'ETA', 'F', 'U', 'V', 'G', 'P')
        logger.info('   ' + '=' * 67)
        for j in range(self.mesh.etamax):
            logger.info(
                '%4d %5.2f % .4e % .4e % .4e % .4e % .4e',
                j, self.mesh.eta[j],
                state['f'][j], state['u'][j], state['v'][j],
                state['g'][j], state['p'][j])

    def _store_result(self, nx, state, b, e, solver_message):
        result = StationResult(
            nx=nx,
            gsi=self.mesh.gsi[nx],
            eta=self.mesh.eta.copy(),
            f=state['f'].copy(),
            u=state['u'].copy(),
            v=state['v'].copy(),
            g=state['g'].copy(),
            p=state['p'].copy(),
            b=b.copy(),
            e=e.copy(),
            urel_visc=self.turbulence.urel_visc,
            solver_message=solver_message or '')
        self.results.append(result)

    def run(self):
        """Run the full simulation."""
        self._print_flow_info()

        # initial station
        nx = 0
        self._print_station_header(nx)

        state = InitialConditions.compute(
            self.mesh, self.Reynolds, self.Prandtl, nx)

        # compute turbulence for initial profile
        b, e = self.turbulence.compute(
            state['u'], self.mesh.eta, self.mesh.gsi[nx],
            nx, self.Reynolds, self.mesh.etamax)

        self._store_result(nx, state, b, e, 'Initial profile')
        self._print_station_result(state, b, e)

        # march downstream
        for nx in range(1, self.mesh.gsimax):
            self._print_station_header(nx)

            # update turbulence
            b_new, e_new = self.turbulence.compute(
                state['u'], self.mesh.eta, self.mesh.gsi[nx],
                nx, self.Reynolds, self.mesh.etamax)

            # build and solve the discrete system
            F, guess = self.discretization.build_residual_function(
                state, b_new, e_new, b, e,
                nx, self.mesh.gsi, self.mesh.deta, self.mesh.etamax)

            result = self.solver.solve(
                F, guess,
                tolerance=self.config.solver.tolerance,
                iterations=self.config.solver.iterations)

            # unpack solution into new state
            f, u, v, g, p = self.discretization.unpack(
                result.solution, self.mesh.etamax)
            state = {'f': f, 'u': u, 'v': v, 'g': g, 'p': p}

            b = b_new.copy()
            e = e_new.copy()

            self._store_result(nx, state, b, e, result.message)
            self._print_station_result(state, b, e)

    def save(self, filename=None):
        """Save results to file."""
        filename = filename or self.config.output.result_filename
        filepath = os.path.join(self.config.output.result_folder, filename)
        self.results.save(
            filepath,
            self.Reynolds, self.Prandtl, self.Prandtl_turb,
            self.turbulent,
            self.mesh.gsi[0], self.mesh.dgsi, self.mesh.deta[0],
            self.mesh.etae, self.mesh.eta[-1], self.mesh.stretch)
        if self.config.output.save_dimensional_results and self._has_dimensional_reference():
            dimensional_filepath = os.path.join(
                self.config.output.result_folder,
                self.config.output.dimensional_result_filename)
            self.results.save_dimensional(
                dimensional_filepath,
                self.Reynolds, self.Prandtl, self.Prandtl_turb,
                self.turbulent,
                self.mesh.gsi[0], self.mesh.dgsi, self.mesh.deta[0],
                self.mesh.etae, self.mesh.eta[-1], self.mesh.stretch)
        elif self.config.output.save_dimensional_results:
            self._log_dimensional_skip('dimensional result export')

    def plot(self, steps=None):
        """Generate profile plots."""
        if steps is None:
            steps = range(len(self.results))
        steps = list(steps)
        if self.config.output.plot_dimensionless:
            plot_profiles(
                self.results, steps, self.config.output.plot_folder,
                self.Reynolds, self.Prandtl, self.Prandtl_turb,
                self.turbulent,
                self.mesh.gsi[0], self.mesh.dgsi, self.mesh.deta[0],
                self.mesh.etae, self.mesh.stretch)
        if self.config.output.plot_dimensionless_summary:
            plot_dimensionless_summary(
                self.results, self.config.output.plot_folder,
                self.turbulent)
        if self.config.output.plot_dimensional and self._has_dimensional_reference():
            plot_dimensional_profiles(
                self.results, steps,
                self.config.output.dimensional_plot_folder,
                self.turbulent)
        elif self.config.output.plot_dimensional:
            self._log_dimensional_skip('dimensional profile plots')
        if (self.config.output.plot_dimensional_summary and
                self._has_dimensional_reference()):
            plot_dimensional_summary(
                self.results,
                self.config.output.dimensional_plot_folder,
                self.turbulent)
        elif self.config.output.plot_dimensional_summary:
            self._log_dimensional_skip('dimensional summary plot')
