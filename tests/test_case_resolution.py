import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from jet.case_resolution import resolve_case
from jet.config import (DimensionlessConfig, FluidConfig,
                        GeometryConfig, InitialConditionConfig,
                        ScalingConfig, SimulationConfig,
                        TurbulenceConfig)
from jet.config_loader import load_simulation_config
from jet.fluid import FluidProperties


class CaseResolutionTests(unittest.TestCase):
    def test_simulation_config_defaults_to_physical_mode(self):
        config = SimulationConfig()

        self.assertEqual(config.mode, "physical")
        self.assertIsInstance(config.dimensionless, DimensionlessConfig)
        self.assertIsInstance(config.scaling, ScalingConfig)
        self.assertIsInstance(config.turbulence, TurbulenceConfig)
        self.assertFalse(config.output.plot_dimensional)
        self.assertFalse(config.output.plot_dimensionless_summary)
        self.assertFalse(config.output.plot_dimensional_summary)
        self.assertFalse(config.output.save_dimensional_results)

    def test_dimensionless_mode_uses_only_dimensionless_inputs(self):
        config = SimulationConfig(
            mode="dimensionless",
            dimensionless=DimensionlessConfig(
                Reynolds=12345.0,
                Prandtl=0.83,
            ),
            scaling=ScalingConfig(
                fluid=FluidConfig(name="air"),
                geometry=GeometryConfig(nozzle_width=0.1),
                initial=InitialConditionConfig(
                    velocity=None,
                    nozzle_temperature_C=80.0,
                    ambient_temperature_C=5.0,
                ),
            ),
            turbulence=TurbulenceConfig(
                Prandtl_turb=0.91,
                turbulent=False,
            ),
        )

        resolved = resolve_case(config)

        self.assertEqual(resolved.mode, "dimensionless")
        self.assertIsNone(resolved.fluid_name)
        self.assertEqual(resolved.Reynolds, 12345.0)
        self.assertEqual(resolved.Prandtl, 0.83)
        self.assertEqual(resolved.Prandtl_turb, 0.91)
        self.assertFalse(resolved.turbulent)
        self.assertIsNone(resolved.reference_conditions)

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

    def test_physical_mode_requires_physical_inputs(self):
        config = SimulationConfig(
            mode="physical",
            scaling=ScalingConfig(
                fluid=FluidConfig(name="air"),
                geometry=GeometryConfig(nozzle_width=None),
                initial=InitialConditionConfig(
                    velocity=None,
                    nozzle_temperature_C=25.0,
                    ambient_temperature_C=20.0,
                ),
            ),
        )

        with self.assertRaises(ValueError):
            resolve_case(config)

    def test_ini_loader_allows_dimensionless_mode_without_physical_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.ini"
            config_path.write_text(
                "[run]\n"
                "mode = dimensionless\n"
                "discretization = keller_box\n\n"
                "[dimensionless]\n"
                "Reynolds = 12345.0\n"
                "Prandtl = 0.83\n\n"
                "[turbulence]\n"
                "Prandtl_turb = 0.92\n"
                "turbulent = true\n",
                encoding="utf-8",
            )

            config = load_simulation_config(config_path)

        self.assertEqual(config.mode, "dimensionless")
        self.assertEqual(config.dimensionless.Reynolds, 12345.0)
        self.assertEqual(config.turbulence.Prandtl_turb, 0.92)
        self.assertFalse(config.output.plot_dimensional)
        self.assertFalse(config.output.plot_dimensionless_summary)
        self.assertFalse(config.output.plot_dimensional_summary)
        self.assertFalse(config.output.save_dimensional_results)
        resolved = resolve_case(config)
        self.assertIsNone(resolved.reference_conditions)

    def test_ini_loader_requires_physical_section_for_physical_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.ini"
            config_path.write_text(
                "[run]\n"
                "mode = physical\n\n"
                "[dimensionless]\n"
                "Reynolds = 1.0\n"
                "Prandtl = 1.0\n\n"
                "[turbulence]\n"
                "Prandtl_turb = 0.7\n"
                "turbulent = false\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_simulation_config(config_path)

    def test_ini_loader_uses_physical_mode_and_physical_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.ini"
            config_path.write_text(
                "[run]\n"
                "mode = physical\n\n"
                "[dimensionless]\n"
                "Reynolds = 1.0\n"
                "Prandtl = 1.0\n\n"
                "[dimensional (physical)]\n"
                "fluid = water\n"
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
