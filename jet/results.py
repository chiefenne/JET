"""Result storage and file I/O."""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .postprocessing import ReferenceConditions, dimensionalize_results

logger = logging.getLogger(__name__)


@dataclass
class StationResult:
    """Result at a single gsi station."""
    nx: int
    gsi: float
    eta: np.ndarray
    f: np.ndarray
    u: np.ndarray
    v: np.ndarray
    g: np.ndarray
    p: np.ndarray
    b: np.ndarray
    e: np.ndarray
    urel_visc: float
    solver_message: str


@dataclass
class SimulationResults:
    """Collection of all station results."""
    stations: List[StationResult] = field(default_factory=list)
    reference_conditions: Optional[ReferenceConditions] = None
    scale_min: float = 1.e10
    scale_max: float = -1.e10

    def append(self, result: StationResult):
        self.stations.append(result)
        for arr in (result.f, result.u, result.v, result.g, result.p):
            self.scale_min = min(self.scale_min, arr.min())
            self.scale_max = max(self.scale_max, arr.max())

    def __len__(self):
        return len(self.stations)

    def __getitem__(self, idx):
        return self.stations[idx]

    def dimensionalize(self):
        """Convert the stored stations into dimensional values."""
        if self.reference_conditions is None:
            raise ValueError(
                'Missing reference conditions for dimensional postprocessing.')
        return dimensionalize_results(self, self.reference_conditions)

    def save(self, filepath, Reynolds, Prandtl, Prandtl_turb, turbulent,
             gsi0, dgsi, deta0, etae, eta_max, stretch):
        """Save results to a text file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        turb_text = ' - TURBULENT Flow' if turbulent else ' - LAMINAR Flow'

        with open(filepath, 'w') as fh:
            fh.write('#\n')
            fh.write('# 2D TURBULENT HEATED FREE JET\n')
            fh.write('#\n')
            fh.write(f' REYNOLDS = {int(Reynolds):d}\n')
            fh.write(f' PRANDTL = {Prandtl}\n')
            fh.write(f' PRANDTL turbulent = {Prandtl_turb}\n')
            fh.write(f' gsi_0={gsi0:f}\n')
            fh.write(f' dgsi={dgsi:f}\n')
            fh.write(f' deta={deta0:.5f}\n')
            fh.write(f' eta_e={etae:.5f}\n')
            fh.write(f' eta_max={eta_max:.5f}\n')
            fh.write(f' stretch factor={stretch}\n')

            for r in self.stations:
                fh.write('\n\n')
                fh.write('*' * 61 + '\n')
                msg = (f' Jet propagation: GSI = {r.gsi:6.3} '
                       f'at stage {r.nx}' + turb_text + '\n')
                fh.write(msg)
                fh.write('*' * 61 + '\n\n')
                fh.write(f'  Eddy viscosity under-relaxation = '
                         f'{r.urel_visc:4.2f}\n')
                fh.write(f'  Solver: {r.solver_message}\n')
                fh.write(f'  Viscosity (B)   = {r.b[0]: .4e}\n')
                fh.write(f'  Diffusivity (E) = {r.e[0]: .4e}\n\n')
                fh.write('   ' + '=' * 67 + '\n')
                fh.write('{:>4}{:^8}{:^11}{:^13}{:^12}{:^12}{:^12}\n'
                         .format('J', 'ETA', 'F', 'U', 'V', 'G', 'P'))
                fh.write('   ' + '=' * 67 + '\n')
                for j in range(len(r.eta)):
                    fh.write(
                        f'{j:4d} {r.eta[j]:5.2f} {r.f[j]: .4e} '
                        f'{r.u[j]: .4e} {r.v[j]: .4e} '
                        f'{r.g[j]: .4e} {r.p[j]: .4e}\n')

        logger.info('Results saved to: %s', os.path.abspath(filepath))

    def save_dimensional(self, filepath, turbulent):
        """Save dimensionalized results to a text file."""
        dimensional = self.dimensionalize()
        ref = dimensional.reference_conditions
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        turb_text = ' - TURBULENT Flow' if turbulent else ' - LAMINAR Flow'

        with open(filepath, 'w') as fh:
            fh.write('#\n')
            fh.write('# 2D TURBULENT HEATED FREE JET - DIMENSIONAL RESULTS\n')
            fh.write('#\n')
            fh.write(f' HALF_WIDTH [m] = {ref.half_width_m:.6e}\n')
            fh.write(
                f' NOZZLE VELOCITY [m/s] = {ref.nozzle_velocity_m_s:.6e}\n')
            fh.write(
                f' KINEMATIC VISCOSITY [m2/s] = '
                f'{ref.kinematic_viscosity_m2_s:.6e}\n')
            fh.write(
                f' AMBIENT TEMPERATURE [C] = '
                f'{ref.ambient_temperature_C:.6e}\n')
            fh.write(
                f' NOZZLE-TO-AMBIENT DELTA T [K] = '
                f'{ref.reference_temperature_difference_K:.6e}\n')
            fh.write(
                f' NOZZLE TEMPERATURE [C] = '
                f'{ref.nozzle_temperature_C:.6e}\n')

            for station in dimensional.stations:
                fh.write('\n\n')
                fh.write('*' * 79 + '\n')
                fh.write(
                    f' Jet propagation: XI = {station.xi:6.3f} '
                    f'X = {station.x_m: .6e} m '
                    f'DELTA = {station.delta_m: .6e} m'
                    f'{turb_text}\n')
                fh.write('*' * 79 + '\n\n')
                fh.write(
                    f'  Centerline U/U0 = '
                    f'{station.normalized_axial_velocity[0]: .6e}\n')
                fh.write(
                    f'  Centerline Theta = '
                    f'{station.normalized_temperature[0]: .6e}\n')
                fh.write(
                    f'  Centerline Ux [m/s] = '
                    f'{station.axial_velocity_m_s[0]: .6e}\n')
                fh.write(
                    f'  Centerline T [C] = '
                    f'{station.temperature_C[0]: .6e}\n')
                fh.write(
                    f'  Half-jet width ETA_1/2 = '
                    f'{station.half_jet_width_eta: .6e}\n')
                fh.write(
                    f'  Half-jet width Y_1/2 [m] = '
                    f'{station.half_jet_width_y_m: .6e}\n\n')
                fh.write(
                    '{:>4}{:>11}{:>14}{:>14}{:>14}{:>12}{:>12}{:>14}'
                    '{:>14}{:>14}{:>14}{:>14}\n'.format(
                        'J', 'ETA', 'Y [m]', 'Ux [m/s]', 'Vy [m/s]',
                        'Theta', 'dT [K]', 'T [C]',
                        'dU/dy [1/s]', 'dT/dy [K/m]',
                        'mu_eff', 'alpha_eff'))
                fh.write('   ' + '=' * 140 + '\n')
                for j in range(len(station.eta)):
                    fh.write(
                        f'{j:4d}'
                        f'{station.eta[j]:11.4f}'
                        f'{station.y_m[j]:14.6e}'
                        f'{station.axial_velocity_m_s[j]:14.6e}'
                        f'{station.transverse_velocity_m_s[j]:14.6e}'
                        f'{station.normalized_temperature[j]:12.4f}'
                        f'{station.temperature_excess_K[j]:12.4f}'
                        f'{station.temperature_C[j]:14.6e}'
                        f'{station.du_dy_1_s[j]:14.6e}'
                        f'{station.dT_dy_K_m[j]:14.6e}'
                        f'{station.effective_dynamic_viscosity_Pa_s[j]:14.6e}'
                        f'{station.effective_thermal_diffusivity_m2_s[j]:14.6e}\n'
                    )

        logger.info('Dimensional results saved to: %s',
                    os.path.abspath(filepath))
