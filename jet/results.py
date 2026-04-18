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

    def _write_header(self, fh, title, Reynolds, Prandtl, Prandtl_turb,
                      gsi0, dgsi, deta0, etae, eta_max, stretch):
        fh.write('#\n')
        fh.write(f'# {title}\n')
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

    def _write_dimensionless_station(self, fh, station, turb_text):
        fh.write('\n\n')
        fh.write('*' * 61 + '\n')
        msg = (f' Jet propagation: GSI = {station.gsi:6.3f} '
               f'at stage {station.nx}' + turb_text + '\n')
        fh.write(msg)
        fh.write('*' * 61 + '\n\n')
        fh.write(f'  Eddy viscosity under-relaxation = '
                 f'{station.urel_visc:4.2f}\n')
        fh.write(f'  Solver: {station.solver_message}\n')
        fh.write(f'  Viscosity (B)   = {station.b[0]: .4e}\n')
        fh.write(f'  Diffusivity (E) = {station.e[0]: .4e}\n\n')
        fh.write('   ' + '=' * 67 + '\n')
        fh.write('{:>4}{:^8}{:^11}{:^13}{:^12}{:^12}{:^12}\n'
                 .format('J', 'ETA', 'F', 'U', 'V', 'G', 'P'))
        fh.write('   ' + '=' * 67 + '\n')
        for j in range(len(station.eta)):
            fh.write(
                f'{j:4d} {station.eta[j]:5.2f} {station.f[j]: .4e} '
                f'{station.u[j]: .4e} {station.v[j]: .4e} '
                f'{station.g[j]: .4e} {station.p[j]: .4e}\n')

    def _write_reference_conditions(self, fh, reference):
        fh.write(f' HALF_WIDTH [m] = {reference.half_width_m:.6e}\n')
        fh.write(f' NOZZLE VELOCITY [m/s] = '
                 f'{reference.nozzle_velocity_m_s:.6e}\n')
        fh.write(f' KINEMATIC VISCOSITY [m2/s] = '
                 f'{reference.kinematic_viscosity_m2_s:.6e}\n')
        fh.write(f' AMBIENT TEMPERATURE [C] = '
                 f'{reference.ambient_temperature_C:.6e}\n')
        fh.write(f' NOZZLE-TO-AMBIENT DELTA T [K] = '
                 f'{reference.reference_temperature_difference_K:.6e}\n')
        fh.write(f' NOZZLE TEMPERATURE [C] = '
                 f'{reference.nozzle_temperature_C:.6e}\n')

    def _write_dimensional_station(self, fh, station):
        fh.write('\n')
        fh.write('   ' + '=' * 143 + '\n')
        fh.write('  DIMENSIONAL ADD-ON\n')
        fh.write(f'  X [m] = {station.x_m: .6e}\n')
        fh.write(f'  DELTA [m] = {station.delta_m: .6e}\n')
        fh.write(f'  Centerline U/U0 = '
                 f'{station.normalized_axial_velocity[0]: .6e}\n')
        fh.write(f'  Centerline Theta = '
                 f'{station.normalized_temperature[0]: .6e}\n')
        fh.write(f'  Centerline Ux [m/s] = '
                 f'{station.axial_velocity_m_s[0]: .6e}\n')
        fh.write(f'  Centerline T [C] = {station.temperature_C[0]: .6e}\n')
        fh.write(f'  Half-jet width ETA_1/2 = '
                 f'{station.half_jet_width_eta: .6e}\n')
        fh.write(f'  Half-jet width Y_1/2 [m] = '
                 f'{station.half_jet_width_y_m: .6e}\n\n')
        fh.write('{:>4}{:>9}{:>14}{:>14}{:>14}{:>12}{:>12}{:>14}'
                 '{:>14}{:>14}{:>14}{:>14}\n'.format(
                     'J', 'ETA', 'Y [m]', 'Ux [m/s]', 'Vy [m/s]',
                     'Theta', 'dT [K]', 'T [C]',
                     'dU/dy [1/s]', 'dT/dy [K/m]',
                     'mu_eff', 'alpha_eff'))
        fh.write('   ' + '=' * 143 + '\n')
        for j in range(len(station.eta)):
            fh.write(
                f'{j:4d}'
                f'{station.eta[j]:9.2f}'
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

    def save(self, filepath, Reynolds, Prandtl, Prandtl_turb, turbulent,
             gsi0, dgsi, deta0, etae, eta_max, stretch):
        """Save results to a text file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        turb_text = ' - TURBULENT Flow' if turbulent else ' - LAMINAR Flow'

        with open(filepath, 'w') as fh:
            self._write_header(
                fh, '2D TURBULENT HEATED FREE JET', Reynolds, Prandtl,
                Prandtl_turb, gsi0, dgsi, deta0, etae, eta_max, stretch)

            for station in self.stations:
                self._write_dimensionless_station(fh, station, turb_text)

        logger.info('Results saved to: %s', os.path.abspath(filepath))

    def save_dimensional(self, filepath, Reynolds, Prandtl, Prandtl_turb,
                         turbulent, gsi0, dgsi, deta0, etae, eta_max, stretch):
        """Save dimensionalized results to a text file."""
        dimensional = self.dimensionalize()
        ref = dimensional.reference_conditions
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        turb_text = ' - TURBULENT Flow' if turbulent else ' - LAMINAR Flow'

        with open(filepath, 'w') as fh:
            self._write_header(
                fh, '2D TURBULENT HEATED FREE JET - DIMENSIONAL ADD-ON',
                Reynolds, Prandtl, Prandtl_turb, gsi0, dgsi, deta0, etae,
                eta_max, stretch)
            self._write_reference_conditions(fh, ref)

            for result_station, dimensional_station in zip(
                    self.stations, dimensional.stations):
                self._write_dimensionless_station(
                    fh, result_station, turb_text)
                self._write_dimensional_station(fh, dimensional_station)

        logger.info('Dimensional results saved to: %s',
                    os.path.abspath(filepath))
