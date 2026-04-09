"""Resolve user-facing inputs into the dimensionless solver parameters."""

from dataclasses import dataclass

from .config import SimulationConfig, ScalingConfig
from .fluid import FluidProperties
from .postprocessing import ReferenceConditions

REYNOLDS_REL_TOL = 1e-6


@dataclass(frozen=True)
class ResolvedDimensionlessCase:
    """Single internal representation consumed by the solver."""
    mode: str
    fluid_name: str
    Reynolds: float
    Prandtl: float
    Prandtl_turb: float
    turbulent: bool
    scaling_recalculated: str
    reference_conditions: ReferenceConditions


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


def _resolve_dimensionless_scaling(config: SimulationConfig,
                                   fluid: FluidProperties):
    scaling = config.scaling
    Reynolds = config.dimensionless_case.Reynolds
    recalculate = scaling.recalculate.strip().lower()

    nozzle_width = scaling.geometry.nozzle_width
    velocity = scaling.initial.velocity

    if recalculate == 'velocity':
        if nozzle_width is None:
            raise ValueError(
                'Dimensionless mode with recalculate=velocity requires '
                'scaling.nozzle_width.')
        half_width = nozzle_width / 2.0
        velocity = Reynolds * fluid.kinematic_viscosity / half_width
    elif recalculate == 'nozzle_width':
        if velocity is None:
            raise ValueError(
                'Dimensionless mode with recalculate=nozzle_width requires '
                'scaling.velocity.')
        half_width = Reynolds * fluid.kinematic_viscosity / velocity
        nozzle_width = 2.0 * half_width
    elif recalculate == 'none':
        if nozzle_width is None or velocity is None:
            raise ValueError(
                'Dimensionless mode with recalculate=none requires both '
                'scaling.nozzle_width and scaling.velocity.')
        implied_reynolds = fluid.Reynolds(velocity, nozzle_width / 2.0)
        rel_error = abs(implied_reynolds - Reynolds) / max(abs(Reynolds), 1.0)
        if rel_error > REYNOLDS_REL_TOL:
            raise ValueError(
                'Dimensionless scaling is inconsistent with the specified '
                'Reynolds number. Set scaling.recalculate to velocity or '
                'nozzle_width, or provide consistent scaling values.')
    else:
        raise ValueError(
            "Invalid scaling.recalculate. Expected 'none', 'velocity', "
            "or 'nozzle_width'.")

    return float(nozzle_width), float(velocity), recalculate


def resolve_case(config: SimulationConfig) -> ResolvedDimensionlessCase:
    """Convert the user-facing configuration into solver inputs."""
    mode = config.mode.strip().lower()
    fluid = _get_fluid(config.scaling)

    if mode == 'physical':
        nozzle_width = config.scaling.geometry.nozzle_width
        velocity = config.scaling.initial.velocity
        if nozzle_width is None or velocity is None:
            raise ValueError(
                'Physical mode requires both scaling.nozzle_width and '
                'scaling.velocity.')
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
            scaling_recalculated='none',
            reference_conditions=reference_conditions)

    if mode == 'dimensionless':
        nozzle_width, velocity, recalculate = _resolve_dimensionless_scaling(
            config, fluid)
        reference_conditions = _build_reference_conditions(
            fluid, config.scaling, nozzle_width, velocity)
        return ResolvedDimensionlessCase(
            mode='dimensionless',
            fluid_name=config.scaling.fluid.name,
            Reynolds=config.dimensionless_case.Reynolds,
            Prandtl=config.dimensionless_case.Prandtl,
            Prandtl_turb=config.turbulence.Prandtl_turb,
            turbulent=config.turbulence.turbulent,
            scaling_recalculated=recalculate,
            reference_conditions=reference_conditions)

    raise ValueError("Invalid mode. Expected 'physical' or 'dimensionless'.")
