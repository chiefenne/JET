"""INI-based configuration loading for the JET solver."""

from __future__ import annotations

import configparser
from pathlib import Path

from .config import (DimensionlessCaseConfig, FluidConfig, GeometryConfig,
                     InitialConditionConfig, MeshConfig, OutputConfig,
                     ScalingConfig, SimulationConfig, SolverConfig,
                     TurbulenceConfig)

DIMENSIONAL_SECTION = 'dimensional (physical)'
LEGACY_SCALING_SECTION = 'scaling'


def _require_section(parser: configparser.ConfigParser, section: str):
    if not parser.has_section(section):
        raise ValueError(f"Missing required section [{section}] in config file.")


def _read_optional_float(parser: configparser.ConfigParser, section: str,
                         option: str):
    value = parser.get(section, option)
    if value.strip().lower() == 'auto':
        return None
    return float(value)


def _get_dimensional_section(parser: configparser.ConfigParser) -> str:
    if parser.has_section(DIMENSIONAL_SECTION):
        return DIMENSIONAL_SECTION
    if parser.has_section(LEGACY_SCALING_SECTION):
        return LEGACY_SCALING_SECTION
    raise ValueError(
        'Missing required section '
        f'[{DIMENSIONAL_SECTION}] in config file.')


def _read_scaling(parser: configparser.ConfigParser) -> ScalingConfig:
    section = _get_dimensional_section(parser)
    return ScalingConfig(
        fluid=FluidConfig(name=parser.get(section, 'fluid')),
        geometry=GeometryConfig(
            nozzle_width=_read_optional_float(parser, section, 'nozzle_width')),
        initial=InitialConditionConfig(
            velocity=_read_optional_float(parser, section, 'velocity'),
            nozzle_temperature_C=parser.getfloat(
                section, 'nozzle_temperature_C'),
            ambient_temperature_C=parser.getfloat(
                section, 'ambient_temperature_C'),
        ),
    )


def load_simulation_config(config_path: str | Path = 'config.ini'
                           ) -> SimulationConfig:
    """Load a simulation config from an INI file."""
    parser = configparser.ConfigParser()
    parser.optionxform = str

    path = Path(config_path)
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read config file: {path}")

    _require_section(parser, 'run')
    _require_section(parser, 'dimensionless_case')
    _require_section(parser, 'turbulence')
    mode = parser.get('run', 'mode', fallback='physical').strip().lower()
    if mode not in ('physical', 'dimensionless'):
        raise ValueError("Invalid mode. Expected 'physical' or 'dimensionless'.")

    if mode == 'physical':
        _get_dimensional_section(parser)

    mesh_section = 'mesh' if parser.has_section('mesh') else None
    solver_section = 'solver' if parser.has_section('solver') else None
    output_section = 'output' if parser.has_section('output') else None

    mesh = MeshConfig(
        gsimax=(parser.getint(mesh_section, 'gsimax', fallback=30)
                if mesh_section else 30),
        dgsi=(parser.getfloat(mesh_section, 'dgsi', fallback=0.01)
              if mesh_section else 0.01),
        etae=(parser.getfloat(mesh_section, 'etae', fallback=25.0)
              if mesh_section else 25.0),
        deta1=(parser.getfloat(mesh_section, 'deta1', fallback=0.01)
               if mesh_section else 0.01),
        stretch=(parser.getfloat(mesh_section, 'stretch', fallback=1.1)
                 if mesh_section else 1.1),
    )

    solver = SolverConfig(
        solver_type=(parser.get(solver_section, 'solver_type',
                                fallback='fsolve')
                     if solver_section else 'fsolve'),
        iterations=(parser.getint(solver_section, 'iterations', fallback=100)
                    if solver_section else 100),
        tolerance=(parser.getfloat(solver_section, 'tolerance',
                                   fallback=1e-6)
                   if solver_section else 1e-6),
    )

    output = OutputConfig(
        plot_folder=(parser.get(output_section, 'plot_folder',
                                fallback='PLOTS')
                     if output_section else 'PLOTS'),
        dimensional_plot_folder=(parser.get(
            output_section, 'dimensional_plot_folder',
            fallback='PLOTS_DIMENSIONAL')
            if output_section else 'PLOTS_DIMENSIONAL'),
        result_folder=(parser.get(output_section, 'result_folder',
                                  fallback='RESULTS')
                       if output_section else 'RESULTS'),
        result_filename=(parser.get(output_section, 'result_filename',
                                    fallback='results.dat')
                         if output_section else 'results.dat'),
        dimensional_result_filename=(parser.get(
            output_section, 'dimensional_result_filename',
            fallback='results_dimensional.dat')
            if output_section else 'results_dimensional.dat'),
        plot_dimensionless=(parser.getboolean(
            output_section, 'plot_dimensionless', fallback=True)
            if output_section else True),
        plot_dimensional=(parser.getboolean(
            output_section, 'plot_dimensional', fallback=False)
            if output_section else False),
        plot_dimensionless_summary=(parser.getboolean(
            output_section, 'plot_dimensionless_summary', fallback=False)
            if output_section else False),
        plot_dimensional_summary=(parser.getboolean(
            output_section, 'plot_dimensional_summary', fallback=False)
            if output_section else False),
        save_dimensional_results=(parser.getboolean(
            output_section, 'save_dimensional_results', fallback=False)
            if output_section else False),
        verbosity=(parser.getint(output_section, 'verbosity', fallback=1)
                   if output_section else 1),
    )

    return SimulationConfig(
        mode=mode,
        dimensionless_case=DimensionlessCaseConfig(
            Reynolds=parser.getfloat('dimensionless_case', 'Reynolds'),
            Prandtl=parser.getfloat('dimensionless_case', 'Prandtl'),
        ),
        scaling=(_read_scaling(parser)
                 if mode == 'physical' else ScalingConfig()),
        turbulence=TurbulenceConfig(
            Prandtl_turb=parser.getfloat('turbulence', 'Prandtl_turb'),
            turbulent=parser.getboolean('turbulence', 'turbulent'),
        ),
        mesh=mesh,
        solver=solver,
        output=output,
        discretization=parser.get('run', 'discretization',
                                  fallback='keller_box'),
    )
