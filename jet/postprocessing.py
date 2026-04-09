"""Postprocessing helpers for dimensional quantities."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class ReferenceConditions:
    """Physical reference values used to dimensionalize the solution."""
    half_width_m: float
    nozzle_velocity_m_s: float
    nozzle_temperature_C: float
    kinematic_viscosity_m2_s: float
    dynamic_viscosity_Pa_s: float
    density_kg_m3: float
    specific_heat_J_kg_K: float
    thermal_conductivity_W_m_K: float
    ambient_temperature_C: float

    @property
    def Reynolds(self) -> float:
        return (self.nozzle_velocity_m_s * self.half_width_m /
                self.kinematic_viscosity_m2_s)

    @property
    def reference_temperature_difference_K(self) -> float:
        return self.nozzle_temperature_C - self.ambient_temperature_C

    @property
    def molecular_thermal_diffusivity_m2_s(self) -> float:
        return (self.thermal_conductivity_W_m_K /
                (self.density_kg_m3 * self.specific_heat_J_kg_K))


@dataclass
class DimensionalStation:
    """Dimensional quantities derived from a single station result."""
    nx: int
    xi: float
    x_m: float
    eta: np.ndarray
    y_m: np.ndarray
    delta_m: float
    streamfunction_m2_s: np.ndarray
    normalized_axial_velocity: np.ndarray
    axial_velocity_m_s: np.ndarray
    transverse_velocity_m_s: np.ndarray
    normalized_temperature: np.ndarray
    temperature_excess_K: np.ndarray
    temperature_C: np.ndarray
    du_dy_1_s: np.ndarray
    dT_dy_K_m: np.ndarray
    effective_kinematic_viscosity_m2_s: np.ndarray
    eddy_kinematic_viscosity_m2_s: np.ndarray
    effective_dynamic_viscosity_Pa_s: np.ndarray
    effective_thermal_diffusivity_m2_s: np.ndarray
    eddy_thermal_diffusivity_m2_s: np.ndarray
    effective_thermal_conductivity_W_m_K: np.ndarray
    half_jet_width_eta: float
    half_jet_width_y_m: float


@dataclass
class DimensionalResults:
    """Collection of dimensionalized station results."""
    stations: List[DimensionalStation]
    reference_conditions: ReferenceConditions

    def __len__(self):
        return len(self.stations)

    def __getitem__(self, idx):
        return self.stations[idx]


def _compute_df_dxi(stations) -> np.ndarray:
    if not stations:
        return np.empty((0, 0))

    xis = np.array([station.gsi for station in stations], dtype=float)
    f_values = np.stack([station.f for station in stations], axis=0)

    if len(stations) == 1:
        return np.zeros_like(f_values)

    edge_order = 2 if len(stations) >= 3 else 1
    return np.gradient(f_values, xis, axis=0, edge_order=edge_order)


def _find_relative_width(eta: np.ndarray, profile: np.ndarray,
                         fraction: float) -> float:
    """Return eta where the profile drops to a fraction of centerline."""
    if profile.size == 0 or profile[0] == 0.0:
        return np.nan

    threshold = fraction * profile[0]
    indices = np.where(profile <= threshold)[0]
    if len(indices) == 0:
        return np.nan

    j = int(indices[0])
    if j == 0:
        return float(eta[0])

    denom = profile[j] - profile[j - 1]
    if denom == 0.0:
        return float(eta[j])

    return float(
        eta[j - 1] + (eta[j] - eta[j - 1]) *
        (threshold - profile[j - 1]) / denom)


def dimensionalize_station(station, reference: ReferenceConditions,
                           df_dxi: np.ndarray) -> DimensionalStation:
    """Convert a single transformed station into dimensional values."""
    xi = station.gsi
    eta = station.eta.copy()
    sqrt_re = np.sqrt(reference.Reynolds)
    xi_one_third = xi**(1.0 / 3.0)
    xi_two_thirds = xi**(2.0 / 3.0)

    x_m = reference.half_width_m * xi
    delta_m = (3.0 * reference.half_width_m * xi_two_thirds / sqrt_re)
    y_m = eta * delta_m

    normalized_axial_velocity = station.u / (3.0 * xi_one_third)
    axial_velocity_m_s = (reference.nozzle_velocity_m_s * station.u /
                          (3.0 * xi_one_third))
    streamfunction_m2_s = (
        np.sqrt(reference.nozzle_velocity_m_s *
                reference.kinematic_viscosity_m2_s *
                reference.half_width_m) *
        xi_one_third * station.f)

    normalized_temperature = station.g / xi_one_third
    temperature_excess_K = (reference.reference_temperature_difference_K *
                            normalized_temperature)
    temperature_C = reference.ambient_temperature_C + temperature_excess_K

    du_dy_1_s = (reference.nozzle_velocity_m_s * sqrt_re * station.v /
                 (9.0 * reference.half_width_m * xi))
    dT_dy_K_m = (reference.reference_temperature_difference_K *
                 sqrt_re * station.p / (3.0 * reference.half_width_m * xi))

    transverse_velocity_m_s = -(
        reference.nozzle_velocity_m_s / sqrt_re) * (
            xi_one_third * df_dxi +
            (station.f - 2.0 * eta * station.u) /
            (3.0 * xi_two_thirds))

    effective_kinematic_viscosity_m2_s = (
        reference.kinematic_viscosity_m2_s * station.b)
    eddy_kinematic_viscosity_m2_s = (
        effective_kinematic_viscosity_m2_s -
        reference.kinematic_viscosity_m2_s)
    effective_dynamic_viscosity_Pa_s = (
        reference.dynamic_viscosity_Pa_s * station.b)

    effective_thermal_diffusivity_m2_s = (
        reference.kinematic_viscosity_m2_s * station.e)
    eddy_thermal_diffusivity_m2_s = (
        effective_thermal_diffusivity_m2_s -
        reference.molecular_thermal_diffusivity_m2_s)
    effective_thermal_conductivity_W_m_K = (
        reference.density_kg_m3 *
        reference.specific_heat_J_kg_K *
        effective_thermal_diffusivity_m2_s)

    half_jet_width_eta = _find_relative_width(
        eta, normalized_axial_velocity, 0.5)
    half_jet_width_y_m = half_jet_width_eta * delta_m

    return DimensionalStation(
        nx=station.nx,
        xi=xi,
        x_m=x_m,
        eta=eta,
        y_m=y_m,
        delta_m=delta_m,
        streamfunction_m2_s=streamfunction_m2_s,
        normalized_axial_velocity=normalized_axial_velocity,
        axial_velocity_m_s=axial_velocity_m_s,
        transverse_velocity_m_s=transverse_velocity_m_s,
        normalized_temperature=normalized_temperature,
        temperature_excess_K=temperature_excess_K,
        temperature_C=temperature_C,
        du_dy_1_s=du_dy_1_s,
        dT_dy_K_m=dT_dy_K_m,
        effective_kinematic_viscosity_m2_s=
        effective_kinematic_viscosity_m2_s,
        eddy_kinematic_viscosity_m2_s=eddy_kinematic_viscosity_m2_s,
        effective_dynamic_viscosity_Pa_s=effective_dynamic_viscosity_Pa_s,
        effective_thermal_diffusivity_m2_s=
        effective_thermal_diffusivity_m2_s,
        eddy_thermal_diffusivity_m2_s=eddy_thermal_diffusivity_m2_s,
        effective_thermal_conductivity_W_m_K=
        effective_thermal_conductivity_W_m_K,
        half_jet_width_eta=half_jet_width_eta,
        half_jet_width_y_m=half_jet_width_y_m)


def dimensionalize_results(results, reference: ReferenceConditions
                           ) -> DimensionalResults:
    """Convert all stored stations into dimensional results."""
    df_dxi_all = _compute_df_dxi(results.stations)
    stations = [
        dimensionalize_station(station, reference, df_dxi_all[idx])
        for idx, station in enumerate(results.stations)
    ]
    return DimensionalResults(
        stations=stations,
        reference_conditions=reference)
