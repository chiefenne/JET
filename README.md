# JET

Numerical solution of a 2D turbulent heated free jet exiting from a nozzle into still surroundings.

The solver follows Keller's BOX method after Cebeci and Bradshaw. The refactored code in `jet/` is the active implementation; `VERSION_2024/` keeps the older monolithic reference code.

## Current State

- The numerical solve itself uses only the transformed, dimensionless variables `f, u, v, g, p`.
- Conversion back to dimensional physical quantities is handled strictly in postprocessing.
- The user can run either `physical` mode or `dimensionless` mode through a single shared scaling definition.
- In physical mode, `Reynolds` and `Prandtl` are derived from the scaling inputs.
- In dimensionless mode, `Reynolds` and `Prandtl` are prescribed directly and the scaling can recalculate either nozzle width or exit velocity to remain Reynolds-consistent.
- The code writes both dimensionless and dimensional result files.
- The code generates both station-wise profile plots and summary plots for centerline decay and half-jet width.

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

The shipped [`config.ini`](config.ini) defaults to the legacy 2024 physical water case so the out-of-the-box run matches the earlier reference setup.

## Tests

Run the unit tests from the repository root with:

```bash
python -m unittest discover -s tests -v
```

## Configuration

The main configuration is assembled from dataclasses in [`jet/config.py`](jet/config.py).

- `DimensionlessCaseConfig`: directly specifies `Reynolds` and `Prandtl`.
- `ScalingConfig`: shared physical scaling definition used for dimensional interpretation and for physical-mode property evaluation.
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
- `[scaling]`: fluid, temperatures, nozzle width, velocity, and optional Reynolds-consistency recalculation target
- `[turbulence]`: `Prandtl_turb`, `turbulent`
- `[mesh]`
- `[solver]`
- `[output]`

The `[dimensionless_case]` and `[scaling]` sections remain active and non-commented in the file. The `[run]` mode toggle decides whether the solver uses the prescribed dimensionless groups or derives them from the scaling inputs.

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

The shared `scaling` block does not change the dimensionless equations. It is used for:

- dimensional result reconstruction
- dimensional plots
- SI-unit interpretation of the dimensionless solution

If desired, the scaling block can also recalculate either:

- the exit velocity from the specified nozzle width and `Reynolds`
- the nozzle width from the specified exit velocity and `Reynolds`

This avoids inconsistent physical scaling for a given dimensionless case. The Reynolds number uses the half nozzle width by definition:

```text
Re = u0 * (nozzle_width / 2) / nu(T0)
```

### Physical Case

In `physical` mode, the scaling block defines:

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

- a physical water case matching the 2024 legacy script
- a dimensionless case with matching `Reynolds` and `Prandtl`

This makes it easy to switch modes without rewriting inputs.

## Outputs

By default the run writes:

- `RESULTS/results.dat`: dimensionless station data
- `RESULTS/results_dimensional.dat`: dimensional station data in SI units
- `PLOTS/`: dimensionless station profiles plus `summary_dimensionless.png`
- `PLOTS_DIMENSIONAL/`: dimensional station profiles plus `summary_dimensional.png`

The dimensional output is reconstructed only after the solve from the stored dimensionless solution.

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
