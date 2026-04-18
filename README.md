# JET

Numerical solution of a 2D turbulent heated free jet exiting from a nozzle into still surroundings.

The solver follows Keller's BOX method after Cebeci and Bradshaw. The refactored code in `jet/` is the active implementation; `VERSION_2024/` keeps the older monolithic reference code.

## Current State

- The numerical solve itself uses only the transformed, dimensionless variables `f, u, v, g, p`.
- Conversion back to dimensional physical quantities is handled strictly in postprocessing.
- The user can run either `physical` mode or `dimensionless` mode through two strict case definitions.
- In physical mode, `Reynolds` and `Prandtl` are derived from the dimensional physical inputs.
- In dimensionless mode, `Reynolds` and `Prandtl` are prescribed directly.
- Dimensionless outputs are always available.
- Dimensional result files and dimensional plots are available only in physical mode.

## Repository Layout

- `jet/`: refactored solver package
- `jet.py`: runnable example configuration
- `jet.ipynb`: derivation notebook
- `VERSION_2024/`: archived pre-refactor implementation

## Running

From the repository root:

```bash
python jet.py
```

This reads [`config.ini`](config.ini) by default.

You can also pass a different config file:

```bash
python jet.py my_case.ini
```

The shipped [`config.ini`](config.ini) currently defaults to a strict dimensionless example while keeping a physical reference block in the same file for easy mode switching.

## Tests

Run the unit tests from the repository root with:

```bash
python -m unittest discover -s tests -v
```

## Configuration

The main configuration is assembled from dataclasses in [`jet/config.py`](jet/config.py).

- `DimensionlessCaseConfig`: directly specifies `Reynolds` and `Prandtl`.
- `ScalingConfig`: physical inputs used by the physical case definition.
- `TurbulenceConfig`: specifies `Prandtl_turb` and whether the run is turbulent or laminar.
- `FluidConfig`: selects the fluid model.
- `InitialConditionConfig`: nozzle-exit velocity, nozzle-exit temperature, and ambient temperature.
- `GeometryConfig`: nozzle width only.
- `MeshConfig`: transformed grid definition in `xi` and `eta`.
- `SolverConfig`: nonlinear solver selection and tolerances.
- `OutputConfig`: result folders and plot switches.

### INI Configuration

The entry point loads an INI file through [`jet/config_loader.py`](jet/config_loader.py).

The expected sections are:

- `[run]`: selects `mode = physical` or `mode = dimensionless`
- `[dimensionless_case]`: `Reynolds`, `Prandtl`
- `[dimensional (physical)]`: fluid, temperatures, nozzle width, velocity
- `[turbulence]`: `Prandtl_turb`, `turbulent`
- `[mesh]`
- `[solver]`
- `[output]`

The `[run]` mode toggle decides whether the solver uses the prescribed dimensionless groups or derives them from the physical inputs. In `dimensionless` mode, the physical block is ignored.

### Dimensionless Solver

Internally the solver always uses only:

- `Reynolds`
- `Prandtl`
- `Prandtl_turb`
- the transformed variables `f, u, v, g, p`

The two run modes only differ in how `Reynolds` and `Prandtl` are obtained.

### Dimensionless Case

In `dimensionless` mode, the solve is prescribed directly through:

- `Reynolds`
- `Prandtl`

The dimensionless case does not require any dimensional physical inputs. The solver uses only the prescribed dimensionless groups together with `Prandtl_turb` and the laminar/turbulent choice.

### Physical Case

In `physical` mode, the `[dimensional (physical)]` block defines:

- nozzle-exit velocity `u0`
- nozzle-exit temperature `T0`
- ambient temperature `T_inf`
- nozzle width
- fluid properties evaluated at the nozzle-exit temperature

From those physical inputs, the code computes the dimensionless solver inputs:

```text
Re = u0 * (nozzle_width / 2) / nu(T0)
Pr = mu(T0) * cp(T0) / k(T0)
```

The temperature reference difference is no longer a user input. It is derived internally during postprocessing as:

```text
Delta T0 = T0 - T_inf
```

This keeps the input model closer to the actual experiment or boundary-value problem.

### Example Modes

The default [`config.ini`](config.ini) contains both:

- a strict dimensionless example in `[dimensionless_case]`
- a strict physical reference block in `[dimensional (physical)]`

This makes it easy to switch modes while keeping each case definition explicit.

## Outputs

In physical mode the run can write:

- `RESULTS/results.dat`: dimensionless station data
- `RESULTS/results_dimensional.dat`: the same station-by-station layout with an added dimensional postprocessing block
- `PLOTS/`: legacy dimensionless station profiles
- `PLOTS_DIMENSIONAL/`: optional dimensional station profiles

The summary plots are optional extras controlled in `[output]` and are off by default so the legacy profile layout stays unchanged unless explicitly enabled.

The dimensional output is reconstructed only in physical mode after the solve from the stored dimensionless solution.

## Dimensional Postprocessing

The postprocessing layer is implemented in [`jet/postprocessing.py`](jet/postprocessing.py).

It reconstructs:

- `x [m]`
- `y [m]`
- `u_x [m/s]`
- `v_y [m/s]`
- `T [C]`
- `Delta T [K]`
- `dU/dy [1/s]`
- `dT/dy [K/m]`
- effective viscosity and thermal diffusivity

It also evaluates useful summary quantities:

- centerline velocity
- centerline temperature
- half-jet width from `u = 0.5 u_c`

Because the computed domain covers only one half of the jet by symmetry, the reported `eta`- and `y`-based width is the half-jet width.

## Turbulent Prandtl Number

`Prandtl_turb` remains a user input in both case modes.

This is consistent with the transformed model and the notebook definition of the turbulent Prandtl number. In the transformed energy equation it enters directly through the effective thermal-diffusion coefficient, so no additional transformation is applied when switching between physical and dimensionless case definitions.

## Numerical Notes

- The archived `VERSION_2024/` solver was used as a reference during refactoring.
- A direct like-for-like comparison against `VERSION_2024/jet.py` with the same physical water case reproduces the same `Reynolds`, `Prandtl`, and matching stored profiles for the compared stages.
- The energy-equation `e4` term had the same typo in both the archived solver and the refactored solver and has been corrected.
- Pre-fix thermal results should therefore be treated as outdated and regenerated.

## Literature

T. Cebeci, P. Bradshaw, *Physical and Computational Aspects of Convective Heat Transfer*, Springer, 1984.
