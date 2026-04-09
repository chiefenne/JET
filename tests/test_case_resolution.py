import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from jet.case_resolution import resolve_case
from jet.config import (DimensionlessCaseConfig, FluidConfig,
                        GeometryConfig, InitialConditionConfig,
                        ScalingConfig, SimulationConfig,
                        TurbulenceConfig)
from jet.config_loader import load_simulation_config
from jet.fluid import FluidProperties


class CaseResolutionTests(unittest.TestCase):
    def test_simulation_config_defaults_to_scaling_based_physical_mode(self):
        config = SimulationConfig()

        self.assertEqual(config.mode, "physical")
        self.assertIsInstance(config.dimensionless_case, DimensionlessCaseConfig)
        self.assertIsInstance(config.scaling, ScalingConfig)
        self.assertIsInstance(config.turbulence, TurbulenceConfig)

    def test_dimensionless_mode_recalculates_velocity_from_reynolds(self):
        config = SimulationConfig(
            mode="dimensionless",
            dimensionless_case=DimensionlessCaseConfig(
                Reynolds=12345.0,
                Prandtl=0.83,
            ),
            scaling=ScalingConfig(
                fluid=FluidConfig(name="air"),
                geometry=GeometryConfig(nozzle_width=0.1),
                initial=InitialConditionConfig(
                    velocity=None,
                    nozzle_temperature_C=25.0,
                    ambient_temperature_C=20.0,
                ),
                recalculate="velocity",
            ),
            turbulence=TurbulenceConfig(
                Prandtl_turb=0.91,
                turbulent=False,
            ),
        )

        resolved = resolve_case(config)
        fluid = FluidProperties("air", 25.0)
        expected_velocity = 12345.0 * fluid.kinematic_viscosity / 0.05

        self.assertEqual(resolved.mode, "dimensionless")
        self.assertEqual(resolved.scaling_recalculated, "velocity")
        self.assertEqual(resolved.Reynolds, 12345.0)
        self.assertEqual(resolved.Prandtl, 0.83)
        self.assertEqual(resolved.Prandtl_turb, 0.91)
        self.assertFalse(resolved.turbulent)
        self.assertAlmostEqual(
            resolved.reference_conditions.nozzle_velocity_m_s,
            expected_velocity)
        self.assertAlmostEqual(
            resolved.reference_conditions.Reynolds,
            12345.0)

    def test_dimensionless_mode_recalculates_nozzle_width_from_reynolds(self):
        config = SimulationConfig(
            mode="dimensionless",
            dimensionless_case=DimensionlessCaseConfig(
                Reynolds=12345.0,
                Prandtl=0.83,
            ),
            scaling=ScalingConfig(
                fluid=FluidConfig(name="air"),
                geometry=GeometryConfig(nozzle_width=None),
                initial=InitialConditionConfig(
                    velocity=20.0,
                    nozzle_temperature_C=25.0,
                    ambient_temperature_C=20.0,
                ),
                recalculate="nozzle_width",
            ),
        )

        resolved = resolve_case(config)
        fluid = FluidProperties("air", 25.0)
        expected_half_width = 12345.0 * fluid.kinematic_viscosity / 20.0

        self.assertEqual(resolved.scaling_recalculated, "nozzle_width")
        self.assertAlmostEqual(
            resolved.reference_conditions.half_width_m,
            expected_half_width)
        self.assertAlmostEqual(
            resolved.reference_conditions.Reynolds,
            12345.0)

    def test_dimensionless_mode_rejects_inconsistent_scaling_when_fixed(self):
        config = SimulationConfig(
            mode="dimensionless",
            dimensionless_case=DimensionlessCaseConfig(
                Reynolds=12345.0,
                Prandtl=0.83,
            ),
            scaling=ScalingConfig(
                fluid=FluidConfig(name="air"),
                geometry=GeometryConfig(nozzle_width=0.1),
                initial=InitialConditionConfig(
                    velocity=20.0,
                    nozzle_temperature_C=25.0,
                    ambient_temperature_C=20.0,
                ),
                recalculate="none",
            ),
        )

        with self.assertRaises(ValueError):
            resolve_case(config)

    def test_dimensionless_mode_accepts_consistent_fixed_scaling(self):
        fluid = FluidProperties("water", 25.0)
        reynolds = fluid.Reynolds(20.0, 0.05)
        config = SimulationConfig(
            mode="dimensionless",
            dimensionless_case=DimensionlessCaseConfig(
                Reynolds=reynolds,
                Prandtl=fluid.Prandtl,
            ),
            scaling=ScalingConfig(
                fluid=FluidConfig(name="water"),
                geometry=GeometryConfig(nozzle_width=0.1),
                initial=InitialConditionConfig(
                    velocity=20.0,
                    nozzle_temperature_C=25.0,
                    ambient_temperature_C=20.0,
                ),
                recalculate="none",
            ),
        )

        resolved = resolve_case(config)

        self.assertEqual(resolved.scaling_recalculated, "none")
        self.assertAlmostEqual(
            resolved.reference_conditions.Reynolds,
            reynolds)

    def test_physical_mode_derives_reynolds_and_prandtl_from_scaling(self):
        config = SimulationConfig(
            mode="physical",
            scaling=ScalingConfig(
                fluid=FluidConfig(name="air"),
                geometry=GeometryConfig(nozzle_width=0.1),
                initial=InitialConditionConfig(
                    velocity=20.0,
                    nozzle_temperature_C=25.0,
                    ambient_temperature_C=20.0,
                ),
                recalculate="none",
            ),
            turbulence=TurbulenceConfig(
                Prandtl_turb=0.87,
                turbulent=True,
            ),
        )

        resolved = resolve_case(config)
        fluid = FluidProperties("air", 25.0)
        expected_reynolds = fluid.Reynolds(20.0, 0.05)

        self.assertEqual(resolved.mode, "physical")
        self.assertAlmostEqual(resolved.Reynolds, expected_reynolds)
        self.assertAlmostEqual(resolved.Prandtl, fluid.Prandtl)
        self.assertEqual(resolved.Prandtl_turb, 0.87)
        self.assertTrue(resolved.turbulent)

    def test_ini_loader_uses_dimensionless_mode_and_auto_velocity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.ini"
            config_path.write_text(
                "[run]\n"
                "mode = dimensionless\n"
                "discretization = keller_box\n\n"
                "[dimensionless_case]\n"
                "Reynolds = 12345.0\n"
                "Prandtl = 0.83\n\n"
                "[scaling]\n"
                "fluid = air\n"
                "recalculate = velocity\n"
                "nozzle_width = 0.1\n"
                "velocity = auto\n"
                "nozzle_temperature_C = 25.0\n"
                "ambient_temperature_C = 20.0\n\n"
                "[turbulence]\n"
                "Prandtl_turb = 0.92\n"
                "turbulent = true\n",
                encoding="utf-8",
            )

            config = load_simulation_config(config_path)

        self.assertEqual(config.mode, "dimensionless")
        self.assertEqual(config.dimensionless_case.Reynolds, 12345.0)
        self.assertEqual(config.scaling.recalculate, "velocity")
        self.assertEqual(config.scaling.geometry.nozzle_width, 0.1)
        self.assertIsNone(config.scaling.initial.velocity)
        self.assertEqual(config.turbulence.Prandtl_turb, 0.92)

    def test_ini_loader_uses_physical_mode_and_scaling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.ini"
            config_path.write_text(
                "[run]\n"
                "mode = physical\n\n"
                "[dimensionless_case]\n"
                "Reynolds = 1.0\n"
                "Prandtl = 1.0\n\n"
                "[scaling]\n"
                "fluid = water\n"
                "recalculate = none\n"
                "nozzle_width = 0.2\n"
                "velocity = 10.0\n"
                "nozzle_temperature_C = 30.0\n"
                "ambient_temperature_C = 15.0\n\n"
                "[turbulence]\n"
                "Prandtl_turb = 0.7\n"
                "turbulent = false\n",
                encoding="utf-8",
            )

            config = load_simulation_config(config_path)

        self.assertEqual(config.mode, "physical")
        self.assertEqual(config.scaling.fluid.name, "water")
        self.assertEqual(config.scaling.geometry.nozzle_width, 0.2)
        self.assertEqual(config.scaling.initial.velocity, 10.0)
        self.assertEqual(config.turbulence.Prandtl_turb, 0.7)
        self.assertFalse(config.turbulence.turbulent)


if __name__ == "__main__":
    unittest.main()
