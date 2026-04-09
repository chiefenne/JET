import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from jet.turbulence import TurbulenceModel


class TurbulenceModelTests(unittest.TestCase):
    def test_turbulent_prandtl_is_used_directly_in_diffusivity(self):
        model = TurbulenceModel(Prandtl=0.7, Prandtl_turb=0.9, turbulent=True)
        u = np.array([2.0, 0.9, 0.1])
        eta = np.array([0.0, 1.0, 2.0])

        b, e = model.compute(
            u=u, eta=eta, gsi_val=1.0, nx=0, Reynolds=1000.0, etamax=3)

        eddy_viscosity = b[0] - 1.0

        self.assertGreater(eddy_viscosity, 0.0)
        self.assertAlmostEqual(e[0], 1.0 / 0.7 + eddy_viscosity / 0.9)
        self.assertTrue(np.allclose(b, b[0]))
        self.assertTrue(np.allclose(e, e[0]))

    def test_laminar_mode_keeps_molecular_coefficients(self):
        model = TurbulenceModel(Prandtl=0.7, Prandtl_turb=0.9, turbulent=False)
        u = np.array([2.0, 0.9, 0.1])
        eta = np.array([0.0, 1.0, 2.0])

        b, e = model.compute(
            u=u, eta=eta, gsi_val=1.0, nx=0, Reynolds=1000.0, etamax=3)

        self.assertTrue(np.allclose(b, np.ones(3)))
        self.assertTrue(np.allclose(e, np.full(3, 1.0 / 0.7)))


if __name__ == "__main__":
    unittest.main()
