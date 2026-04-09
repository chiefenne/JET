import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from jet.postprocessing import ReferenceConditions
from jet.results import SimulationResults, StationResult


class PostprocessingTests(unittest.TestCase):
    def test_dimensionalization_reconstructs_expected_centerline_values(self):
        reference = ReferenceConditions(
            half_width_m=0.05,
            nozzle_velocity_m_s=20.0,
            nozzle_temperature_C=25.0,
            kinematic_viscosity_m2_s=1.0e-6,
            dynamic_viscosity_Pa_s=1.0e-3,
            density_kg_m3=1000.0,
            specific_heat_J_kg_K=4180.0,
            thermal_conductivity_W_m_K=0.6,
            ambient_temperature_C=20.0,
        )
        station = StationResult(
            nx=0,
            gsi=1.0,
            eta=np.array([0.0, 1.0, 2.0]),
            f=np.array([0.0, 1.0, 2.0]),
            u=np.array([3.0, 1.5, 0.03]),
            v=np.array([0.0, 0.1, 0.2]),
            g=np.array([1.0, 0.5, 0.0]),
            p=np.array([0.0, 0.2, 0.4]),
            b=np.array([1.2, 1.2, 1.2]),
            e=np.array([2.0, 2.0, 2.0]),
            urel_visc=0.5,
            solver_message="test",
        )

        results = SimulationResults(reference_conditions=reference)
        results.append(station)

        dimensional = results.dimensionalize()
        station_dim = dimensional[0]

        self.assertAlmostEqual(station_dim.x_m, 0.05)
        self.assertAlmostEqual(station_dim.delta_m, 0.00015)
        self.assertAlmostEqual(station_dim.normalized_axial_velocity[0], 1.0)
        self.assertAlmostEqual(station_dim.normalized_axial_velocity[1], 0.5)
        self.assertAlmostEqual(station_dim.axial_velocity_m_s[0], 20.0)
        self.assertAlmostEqual(station_dim.temperature_excess_K[1], 2.5)
        self.assertAlmostEqual(station_dim.temperature_C[0], 25.0)
        self.assertAlmostEqual(station_dim.half_jet_width_eta, 1.0)
        self.assertAlmostEqual(station_dim.half_jet_width_y_m, 0.00015)
        self.assertTrue(np.all(np.isfinite(station_dim.transverse_velocity_m_s)))


if __name__ == "__main__":
    unittest.main()
