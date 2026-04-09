"""JET - Numerical simulation of a 2D turbulent heated free jet.

Usage:
    from jet import Simulation, SimulationConfig

    config = SimulationConfig()
    sim = Simulation(config)
    sim.run()
    sim.save()
    sim.plot()
"""

from .config import (SimulationConfig, FluidConfig,
                     InitialConditionConfig, GeometryConfig,
                     ScalingConfig, DimensionlessCaseConfig,
                     TurbulenceConfig, MeshConfig, SolverConfig,
                     OutputConfig)
from .simulation import Simulation
from .fluid import FluidProperties
from .config_loader import load_simulation_config
from .discretization import DISCRETIZATIONS
from .solvers import SOLVERS
from .postprocessing import ReferenceConditions

__version__ = '2.0'
__author__ = 'Andreas Ennemoser'
__license__ = 'MIT'

__all__ = [
    'Simulation', 'SimulationConfig',
    'FluidConfig', 'InitialConditionConfig',
    'GeometryConfig',
    'ScalingConfig', 'DimensionlessCaseConfig',
    'TurbulenceConfig',
    'MeshConfig', 'SolverConfig', 'OutputConfig',
    'FluidProperties', 'ReferenceConditions',
    'load_simulation_config',
    'DISCRETIZATIONS', 'SOLVERS',
]
