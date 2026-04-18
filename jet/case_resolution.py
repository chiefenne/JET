"""Resolve user-facing inputs into the dimensionless solver parameters."""

from dataclasses import dataclass
from typing import Optional

from .config import SimulationConfig, ScalingConfig
from .fluid import FluidProperties
from .postprocessing import ReferenceConditions


@dataclass(frozen=True)
class ResolvedDimensionlessCase:
    """Single internal representation consumed by the solver."""
    mode: str
    fluid_name: Optional[str]
    Reynolds: float
    Prandtl: float
    Prandtl_turb: float
    turbulent: bool
    reference_conditions: Optional[ReferenceConditions]


def _build_reference_conditions(fluid: FluidProperties,
                                scaling: ScalingConfig,
                                nozzle_width: float,
                                velocity: float) -> ReferenceConditions:
    half_width = nozzle_width / 2.0
    return ReferenceConditions(
        half_width_m=half_width,
        nozzle_velocity_m_s=velocity,
        nozzle_temperature_C=scaling.initial.nozzle_temperature_C,
        kinematic_viscosity_m2_s=fluid.kinematic_viscosity,
        dynamic_viscosity_Pa_s=fluid.viscosity,
        density_kg_m3=fluid.density,
        specific_heat_J_kg_K=fluid.specific_heat,
        thermal_conductivity_W_m_K=fluid.thermal_conductivity,
        ambient_temperature_C=scaling.initial.ambient_temperature_C)


def _get_fluid(scaling: ScalingConfig) -> FluidProperties:
    return FluidProperties(
        scaling.fluid.name,
        scaling.initial.nozzle_temperature_C)


def resolve_case(config: SimulationConfig) -> ResolvedDimensionlessCase:
    """Convert the user-facing configuration into solver inputs."""
    mode = config.mode.strip().lower()

    if mode == 'physical':
        fluid = _get_fluid(config.scaling)
        nozzle_width = config.scaling.geometry.nozzle_width
        velocity = config.scaling.initial.velocity
        if nozzle_width is None or velocity is None:
            raise ValueError(
                'Physical mode requires both '
                '[dimensional (physical)] nozzle_width and velocity.')
        reference_conditions = _build_reference_conditions(
            fluid, config.scaling, float(nozzle_width), float(velocity))
        Reynolds = fluid.Reynolds(
            reference_conditions.nozzle_velocity_m_s,
            reference_conditions.half_width_m)
        Prandtl = fluid.Prandtl
        return ResolvedDimensionlessCase(
            mode='physical',
            fluid_name=config.scaling.fluid.name,
            Reynolds=Reynolds,
            Prandtl=Prandtl,
            Prandtl_turb=config.turbulence.Prandtl_turb,
            turbulent=config.turbulence.turbulent,
            reference_conditions=reference_conditions)

    if mode == 'dimensionless':
        return ResolvedDimensionlessCase(
            mode='dimensionless',
            fluid_name=None,
            Reynolds=config.dimensionless_case.Reynolds,
            Prandtl=config.dimensionless_case.Prandtl,
            Prandtl_turb=config.turbulence.Prandtl_turb,
            turbulent=config.turbulence.turbulent,
            reference_conditions=None)

    raise ValueError("Invalid mode. Expected 'physical' or 'dimensionless'.")
