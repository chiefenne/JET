"""Configuration dataclasses for the JET simulation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FluidConfig:
    """Fluid selection."""
    name: str = 'water'


@dataclass
class InitialConditionConfig:
    """Nozzle-exit and ambient values defining the thermal setup."""
    velocity: Optional[float] = 20.0   # [m/s], may be calculated in scaling
    nozzle_temperature_C: float = 25.0  # [C]
    ambient_temperature_C: float = 20.0  # [C]

    @property
    def reference_temperature_difference_K(self) -> float:
        return self.nozzle_temperature_C - self.ambient_temperature_C


@dataclass
class GeometryConfig:
    """Nozzle geometry."""
    nozzle_width: Optional[float] = 0.1  # [m], may be calculated in scaling


@dataclass
class ScalingConfig:
    """Physical scaling used for SI interpretation or physical-mode inputs."""
    fluid: FluidConfig = field(default_factory=FluidConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    initial: InitialConditionConfig = field(
        default_factory=InitialConditionConfig)
    recalculate: str = 'none'  # 'none', 'velocity', 'nozzle_width'


@dataclass
class DimensionlessCaseConfig:
    """Dimensionless input case for the transformed solver."""
    Reynolds: float = 30000.0
    Prandtl: float = 0.7


@dataclass
class TurbulenceConfig:
    """Turbulence-model settings independent of case definition."""
    Prandtl_turb: float = 0.9
    turbulent: bool = True


@dataclass
class MeshConfig:
    """Mesh generation parameters."""
    gsimax: int = 30
    dgsi: float = 0.01
    etae: float = 25.0
    deta1: float = 0.01
    stretch: float = 1.1


@dataclass
class SolverConfig:
    """Solver selection and parameters."""
    solver_type: str = 'fsolve'
    iterations: int = 100
    tolerance: float = 1e-06


@dataclass
class OutputConfig:
    """Output paths and options."""
    plot_folder: str = 'PLOTS'
    dimensional_plot_folder: str = 'PLOTS_DIMENSIONAL'
    result_folder: str = 'RESULTS'
    result_filename: str = 'results.dat'
    dimensional_result_filename: str = 'results_dimensional.dat'
    plot_dimensionless: bool = True
    plot_dimensional: bool = True
    plot_dimensionless_summary: bool = True
    plot_dimensional_summary: bool = True
    save_dimensional_results: bool = True
    verbosity: int = 1  # 0=quiet, 1=normal, 2=debug


@dataclass
class SimulationConfig:
    """Top-level configuration combining all sub-configs."""
    mode: str = 'physical'  # 'physical' or 'dimensionless'
    dimensionless_case: DimensionlessCaseConfig = field(
        default_factory=DimensionlessCaseConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    discretization: str = 'keller_box'
