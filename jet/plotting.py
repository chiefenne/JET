"""Plotting module for jet simulation results."""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from .results import SimulationResults

logger = logging.getLogger(__name__)


def plot_profiles(results: SimulationResults, steps, plot_folder,
                  Reynolds, Prandtl, Prandtl_turb, turbulent,
                  gsi0, dgsi, deta0, etae, stretch):
    """Create profile plots for selected stations.

    Args:
        results: SimulationResults object
        steps: Iterable of station indices to plot
        plot_folder: Output directory for plot images
        Reynolds, Prandtl, ...: Flow parameters for annotation
    """
    os.makedirs(plot_folder, exist_ok=True)
    logger.info('Creating plots in: %s', os.path.abspath(plot_folder))

    props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                 edgecolor='grey')

    for step in steps:
        if step >= len(results):
            logger.warning('Step %d not in results (max: %d)',
                          step, len(results) - 1)
            continue

        r = results[step]
        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xlim(0.0, r.eta[-1])
        ax.set_ylim(results.scale_min, results.scale_max)
        ax.plot(r.eta, r.f, label='F')
        ax.plot(r.eta, r.u, label='U')
        ax.plot(r.eta, r.v, label='V')
        ax.plot(r.eta, r.g, label='G')
        ax.plot(r.eta, r.p, label='P')

        text1 = '\n'.join((
            r'$Step={:03d}$'.format(r.nx),
            r'$\xi={:.2g}$'.format(r.gsi),
            r'$\eta_m={:.2g}$'.format(r.eta[-1]),
            r'$Reynolds={:d}$'.format(int(Reynolds)),
            r'$Prandtl={:.2g}$'.format(Prandtl),
            r'$Prandtl_t={:.2g}$'.format(Prandtl_turb)))

        text2 = '\n'.join((
            r'$\xi_0={:.2g}$'.format(gsi0),
            r'$d\xi={:.2g}$'.format(dgsi),
            r'$d\eta_0={:.2g}$'.format(deta0),
            r'$\eta_e={:.2g}$'.format(etae),
            r'$stretch\;factor={:.2g}$'.format(stretch)))

        ax.text(0.86, 0.41, text1, color='grey',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        ax.text(0.86, 0.18, text2, color='grey',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        ax.legend(loc='upper right', fontsize=12, frameon=True,
                  fancybox=True, framealpha=1.0, edgecolor='grey',
                  prop={'size': 12}, bbox_to_anchor=(1.0, 0.95))

        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$F, U, V, G, P$')
        title = ('2D Turbulent Heated Free Jet' if turbulent
                 else '2D Laminar Heated Free Jet')
        ax.set_title(title, fontsize=16)

        ax.set_xticks(np.arange(0, r.eta[-1], 1.0))
        step_size = 0.5
        ticks_pos = np.arange(0, results.scale_max + step_size, step_size)
        ticks_neg = np.arange(0, results.scale_min - step_size, -step_size)
        ax.set_yticks(np.unique(np.concatenate((ticks_neg, ticks_pos))))
        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.grid(True)

        figname = os.path.join(plot_folder, f'profiles_{step:04d}.png')
        logger.info('  %s', figname)
        plt.savefig(figname, dpi=150)
        plt.close()


def plot_dimensional_profiles(results: SimulationResults, steps, plot_folder,
                              turbulent):
    """Create dimensional profile plots for selected stations."""
    dimensional = results.dimensionalize()
    reference = dimensional.reference_conditions

    os.makedirs(plot_folder, exist_ok=True)
    logger.info('Creating dimensional plots in: %s',
                os.path.abspath(plot_folder))

    props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                 edgecolor='grey')

    for step in steps:
        if step >= len(dimensional):
            logger.warning('Step %d not in dimensional results (max: %d)',
                           step, len(dimensional) - 1)
            continue

        station = dimensional[step]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].plot(station.y_m, station.axial_velocity_m_s,
                     color='tab:blue', lw=2)
        axes[0].set_xlabel('y [m]')
        axes[0].set_ylabel('u_x [m/s]')
        axes[0].set_title('Axial Velocity')
        axes[0].grid(True)

        axes[1].plot(station.y_m, station.temperature_C,
                     color='tab:red', lw=2)
        axes[1].set_xlabel('y [m]')
        axes[1].set_ylabel('T [C]')
        axes[1].set_title('Temperature')
        axes[1].grid(True)

        text = '\n'.join((
            r'$Step={:03d}$'.format(station.nx),
            r'$\xi={:.3f}$'.format(station.xi),
            r'$x={:.4f}\,\mathrm{{m}}$'.format(station.x_m),
            r'$\delta={:.4e}\,\mathrm{{m}}$'.format(station.delta_m),
            r'$u_0={:.2f}\,\mathrm{{m/s}}$'.format(
                reference.nozzle_velocity_m_s),
            r'$T_0={:.1f}^\circ\mathrm{{C}}$'.format(
                reference.nozzle_temperature_C),
            r'$T_\infty={:.1f}^\circ\mathrm{{C}}$'.format(
                reference.ambient_temperature_C),
            r'$\Delta T_0={:.1f}\,\mathrm{{K}}$'.format(
                reference.reference_temperature_difference_K)))

        axes[1].text(0.97, 0.97, text, color='grey',
                     transform=axes[1].transAxes, fontsize=11,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)

        title = ('2D Turbulent Heated Free Jet - Dimensional'
                 if turbulent else
                 '2D Laminar Heated Free Jet - Dimensional')
        fig.suptitle(title, fontsize=15)
        fig.tight_layout()

        figname = os.path.join(plot_folder, f'profiles_dimensional_{step:04d}.png')
        logger.info('  %s', figname)
        plt.savefig(figname, dpi=150)
        plt.close()


def plot_dimensionless_summary(results: SimulationResults, plot_folder,
                               turbulent):
    """Create a summary plot for dimensionless centerline values."""
    dimensional = results.dimensionalize()
    os.makedirs(plot_folder, exist_ok=True)
    logger.info('Creating dimensionless summary plot in: %s',
                os.path.abspath(plot_folder))

    xi = np.array([station.xi for station in dimensional.stations])
    u_center = np.array([station.normalized_axial_velocity[0]
                         for station in dimensional.stations])
    theta_center = np.array([station.normalized_temperature[0]
                             for station in dimensional.stations])
    eta_half = np.array([station.half_jet_width_eta
                         for station in dimensional.stations])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(xi, u_center, color='tab:blue', lw=2)
    axes[0].set_ylabel(r'$U_c=u_{x,c}/u_0$')
    axes[0].set_title('Centerline Velocity')
    axes[0].grid(True)

    axes[1].plot(xi, theta_center, color='tab:red', lw=2)
    axes[1].set_ylabel(r'$\Theta_c$')
    axes[1].set_title('Centerline Temperature')
    axes[1].grid(True)

    axes[2].plot(xi, eta_half, color='tab:green', lw=2,
                 label=r'$\eta_{1/2}$')
    axes[2].set_xlabel(r'$\xi$')
    axes[2].set_ylabel(r'Half-jet width in $\eta$')
    axes[2].set_title('Half-Jet Width')
    axes[2].legend()
    axes[2].grid(True)

    title = ('2D Turbulent Heated Free Jet - Dimensionless Summary'
             if turbulent else
             '2D Laminar Heated Free Jet - Dimensionless Summary')
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()

    figname = os.path.join(plot_folder, 'summary_dimensionless.png')
    logger.info('  %s', figname)
    plt.savefig(figname, dpi=150)
    plt.close()


def plot_dimensional_summary(results: SimulationResults, plot_folder,
                             turbulent):
    """Create a summary plot for dimensional centerline values."""
    dimensional = results.dimensionalize()
    reference = dimensional.reference_conditions
    os.makedirs(plot_folder, exist_ok=True)
    logger.info('Creating dimensional summary plot in: %s',
                os.path.abspath(plot_folder))

    x = np.array([station.x_m for station in dimensional.stations])
    u_center = np.array([station.axial_velocity_m_s[0]
                         for station in dimensional.stations])
    temperature_center = np.array([station.temperature_C[0]
                                   for station in dimensional.stations])
    y_half = np.array([station.half_jet_width_y_m
                       for station in dimensional.stations])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(x, u_center, color='tab:blue', lw=2)
    axes[0].set_ylabel(r'$u_{x,c}$ [m/s]')
    axes[0].set_title('Centerline Velocity')
    axes[0].grid(True)

    axes[1].plot(x, temperature_center, color='tab:red', lw=2,
                 label=r'$T_c$')
    axes[1].axhline(reference.ambient_temperature_C, color='tab:gray',
                    lw=1.5, linestyle='--', label=r'$T_\infty$')
    axes[1].set_ylabel(r'$T$ [$^\circ$C]')
    axes[1].set_title('Centerline Temperature')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(x, y_half, color='tab:green', lw=2,
                 label=r'$y_{1/2}$')
    axes[2].set_xlabel(r'$x$ [m]')
    axes[2].set_ylabel('Half-jet width [m]')
    axes[2].set_title('Half-Jet Width')
    axes[2].legend()
    axes[2].grid(True)

    title = ('2D Turbulent Heated Free Jet - Dimensional Summary'
             if turbulent else
             '2D Laminar Heated Free Jet - Dimensional Summary')
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()

    figname = os.path.join(plot_folder, 'summary_dimensional.png')
    logger.info('  %s', figname)
    plt.savefig(figname, dpi=150)
    plt.close()
