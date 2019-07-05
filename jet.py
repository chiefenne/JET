#!bin/env python

import os
from functools import partial
import copy

import numpy as np
from numpy import sqrt, tanh
from scipy import optimize
import matplotlib.pyplot as plt

__author__ = 'Andreas Ennemoser'
__copyright__ = 'Copyright 2015'
__license__ = 'MIT'
__version__ = '0.95'
__email__ = 'andreas.ennemoser@aon.at'
__status__ = 'Development'

DEBUG = False


class Jet():
    """Numerical analysis of a 2D, heated, turbulent free jet

    Literature: T. CEBECI, P. BRADSHAW
    "Physical and computational aspects of convective heat transfer"
    Springer 198
    """

    def __init__(self):
        """Summary"""

        self.Reynolds = 10000.0
        self.Prandtl = 0.72
        self.Prandtl_turb = 0.9
        self.turbulent = False

        self.results = []
        self.solver_message = None

        self.nx = 0

        # centerline boundary condition for the energy equation (dp0=0)
        # switch between temperature or heat flux boundary type
        self.alfa0 = 0.0
        self.alfa1 = 1.0

        # plot scaling limits
        self.scale_min = 1.e10
        self.scale_max = -1.e10

        print('')
        print('**************************************************')
        print('**************************************************')
        print('********** 2D TURBULENT HEATED FREE JET **********')
        print('**************************************************')
        print('**************************************************')

    def set_Reynolds(self, Reynolds):
        self.Reynolds = Reynolds

    def set_Prandtl(self, Prandtl):
        self.Prandtl = Prandtl

    def set_Prandtl_turb(self, Prandtl_turb):
        self.Prandtl_turb = Prandtl_turb

    def set_FluidProperties(self, Reynolds, Prandtl, Prandtl_turb):
        self.set_Reynolds(Reynolds)
        self.set_Prandtl(Prandtl)
        self.set_Prandtl_turb(Prandtl_turb)

    def print_FluidAndFlowInformation(self):
        print(' ')
        print('***************************')
        print('***** FLOW PROPERTIES *****')
        print('***************************')
        print(' REYNOLDS = %s' % (self.Reynolds))
        print(' PRANDTL = %s' % (self.Prandtl))
        print(' PRANDTL turbulent = %s' % (self.Prandtl_turb))
        if self.turbulent:
            print(' TURBULENCE = ON')
        else:
            print(' TURBULENCE = OFF')

    def mesh(self, gsimax=10, dgsi=0.01, etae=8, deta1=0.01, stretch=1.12):
        """Mesh generation for 2D rectangular transformed grid

        Args:
            gsimax (int, optional): Description
            dgsi (float, optional): Description
            etae (int, optional): Description
            deta1 (float, optional): Description
            stretch (float, optional): Description
        """

        self.dgsi = dgsi
        self.etae = etae
        self.stretch = stretch

        # calculation of grid points from spacing parameters
        if stretch < 1.0001:
            etamax = etae / deta1
        else:
            # equation (13.49, page 400)
            etamax = np.log(1.0 + (stretch - 1.0) * etae / deta1) / \
                np.log(stretch)

        self.etamax = int(etamax)
        self.gsimax = gsimax + 1

        # initialize arrays
        # done here, as etamax first time available here
        self.f = np.zeros([self.etamax, 2])
        self.u = np.zeros([self.etamax, 2])
        self.v = np.zeros([self.etamax, 2])
        self.g = np.zeros([self.etamax, 2])
        self.p = np.zeros([self.etamax, 2])
        self.b = np.zeros([self.etamax, 2])
        self.e = np.zeros([self.etamax, 2])

        # initialize grid arrays
        self.gsi = np.zeros([self.gsimax + 1])
        self.eta = np.zeros([self.etamax])
        self.deta = np.zeros([self.etamax - 1])
        self.a = np.zeros([self.etamax])

        self.gsi[0] = 1.0
        self.deta[0] = deta1

        for i in range(1, self.gsimax + 1):
            self.gsi[i] = self.gsi[i - 1] + dgsi

        for j in range(1, self.etamax - 1):
            self.deta[j] = stretch * self.deta[j - 1]

        for j in range(1, self.etamax):
            self.a[j] = 0.5 * self.deta[j - 1]
            self.eta[j] = self.eta[j - 1] + self.deta[j - 1]

        print(' ')
        print('***************************')
        print('***** MESH PROPERTIES *****')
        print('***************************')
        print(' GSI sstart: ', self.gsi[0])
        print(' GSI spacing: ', dgsi)
        print(' ETA spacing (initial): ', deta1)
        print(' ETA growth rate: ', stretch)
        print(' ETA boundary (ETAE given): ', etae)
        print(' ETA boundary (ETAE calculated): ', self.eta[-1])
        print(' Number of grid points in GSI direction: ', self.gsimax)
        print(' Number of grid points in ETA direction: ', self.etamax)

    def boundary_conditions(self):
        """Initial velocity and temperature profiles.
        Equation (14.31), page 445
        """

        gsi0 = self.gsi[self.nx]

        beta = 27.855
        etac = 1.0
        term = beta * 3.0 * gsi0**(2. / 3.) / sqrt(self.Reynolds)

        for j in range(self.etamax):

            tanf = tanh(term * (self.eta[j] - etac))

            self.u[j, 0] = 3.0 / 2.0 * gsi0**(1. / 3.) * (1.0 - tanf)
            self.v[j, 0] = - term * 3.0 / 2.0 * gsi0**(1. / 3.) * \
                (1.0 - tanf**2)
            self.g[j, 0] = self.u[j, 0] / 3.0 * gsi0**(1. / 3.)
            self.p[j, 0] = self.v[j, 0] / 3.0 * gsi0**(1. / 3.)
            self.b[j, 0] = 1.0
            self.e[j, 0] = 1.0 / self.Prandtl

            self.f[j, 0] = self.f[j - 1, 0] + self.a[j] * \
                (self.u[j, 0] + self.u[j - 1, 0])

        # boundary conditions at jet center (symmetry)
        self.f[0, 0] = 0.0
        self.v[0, 0] = 0.0
        self.p[0, 0] = 0.0

        # boundary conditions at jet boundary (ambient)
        self.u[-1, 0] = 0.0
        self.g[-1, 0] = 0.0

        self.turbulence()

        # at first gsi stage make turbulence equal to second step
        self.b[:, 0] = copy.copy(self.b[:, 1])
        self.e[:, 0] = copy.copy(self.e[:, 1])

        # smooth initial profiles in order to avoid non-smooth
        # places, e.g. at jet center for derivatives
        iter = 5
        for k in range(iter):
            # self.f[1:-1] = 0.5 * (self.f[2:] + self.f[0:-2])
            # self.u[1:-1] = 0.5 * (self.u[2:] + self.u[0:-2])
            self.v[1:-1] = 0.5 * (self.v[2:] + self.v[0:-2])
            # self.g[1:-1] = 0.5 * (self.g[2:] + self.g[0:-2])
            self.p[1:-1] = 0.5 * (self.p[2:] + self.p[0:-2])

    def turbulence(self):

        umaxh = 0.5 * self.u[0, 0]

        # Find slice (i.e. index range) where u less than umaxh
        index = np.where(self.u[:, 0] <= umaxh)

        if index:
            # index of u where u first time is less than umaxh
            j = index[0][0]
            #
            etab = self.eta[j - 1] + (self.eta[j] - self.eta[j - 1]) / \
                (self.u[j, 0] - self.u[j - 1, 0]) * (umaxh - self.u[j - 1, 0])
        else:
            etab = self.eta[self.etamax]

        if self.turbulent:
            # compute turbulence viscosity
            eddy_viscosity = 0.037 * etab * self.u[0, 0] * \
                np.sqrt(self.Reynolds) * self.gsi[self.nx]**(1. / 3.)
        else:
            # zero eddy viscosity for laminar flow
            eddy_viscosity = 0.0

        # under-relaxation of eddy_viscosity
        self.urel_visc = np.tanh(self.gsi[self.nx] - 0.4)
        eddy_viscosity = eddy_viscosity * self.urel_visc
        print('  Eddy viscosity under-relaxation = {:5.3f}'.
              format(self.urel_visc))

        for j in range(self.etamax):
            self.b[j, 1] = 1.0 + eddy_viscosity
            self.e[j, 1] = 1.0 / self.Prandtl + \
                eddy_viscosity / self.Prandtl_turb

    def solver(self, solver_type, iterations):

        # knowns
        f_o = copy.copy(self.f[:, 0])
        u_o = copy.copy(self.u[:, 0])
        v_o = copy.copy(self.v[:, 0])
        g_o = copy.copy(self.g[:, 0])
        p_o = copy.copy(self.p[:, 0])

        self.turbulence()

        # extra parameters for 'function'
        # are later wrapped with partial
        b = copy.copy(self.b[:, 1])
        e = copy.copy(self.e[:, 1])
        b_o = copy.copy(self.b[:, 0])
        e_o = copy.copy(self.e[:, 0])
        nx = copy.copy(self.nx)
        gsi = copy.copy(self.gsi)
        deta = copy.copy(self.deta)
        etamax = copy.copy(self.etamax)

        #
        def F(unknowns, F_args=[nx, gsi, deta, etamax, b, e, b_o, e_o]):
            """Summary

            Args:
                unknowns (np.array): f, u, v, g, p

            Returns:
                TYPE: Description
            """

            # unknowns
            if DEBUG:
                print('unknowns', unknowns)
                print('nx', nx)
                print('gsi', gsi)
            # (f, u, v, g, p) = unknowns
            f = unknowns[0: etamax]
            u = unknowns[etamax:2 * etamax]
            v = unknowns[2 * etamax:3 * etamax]
            g = unknowns[3 * etamax:4 * etamax]
            p = unknowns[4 * etamax:5 * etamax]

            alpha = 3.0 / 2.0 * (gsi[nx] + gsi[nx - 1]) / \
                (gsi[nx] - gsi[nx - 1])

            eq1 = np.zeros_like(f)
            eq2 = np.zeros_like(f)
            eq3 = np.zeros_like(f)
            eq4 = np.zeros_like(f)
            eq5 = np.zeros_like(f)

            # boundary conditions at jet center (symmetry)
            f[0] = 0.0
            v[0] = 0.0
            p[0] = 0.0
            f_o[0] = 0.0
            v_o[0] = 0.0
            p_o[0] = 0.0

            # boundary conditions at jet outer boundary (ambient)
            u[-1] = 0.0
            g[-1] = 0.0
            u_o[-1] = 0.0
            g_o[-1] = 0.0

            # array slicing for index j
            # [1:] means index j
            # [:-1] means index j-1

            # ODE: f' = u
            eq1[1:] = 1.0 / deta * (f[1:] - f[:-1]) - 0.5 * (u[1:] + u[:-1])

            # ODE: u' = v
            eq2[1:] = 1.0 / deta * (u[1:] - u[:-1]) - 0.5 * (v[1:] + v[:-1])

            # ODE: g' = p
            eq3[1:] = 1.0 / deta * (g[1:] - g[:-1]) - 0.5 * (p[1:] + p[:-1])

            # PDE: Momentum
            mom1 = 1.0 / deta * (b[1:] * v[1:] - b[:-1] * v[:-1])
            mom2 = (1.0 - alpha) * 0.5 * (u[1:]**2 + u[:-1]**2)
            mom3 = (1.0 + alpha) * 0.5 * (f[1:] * v[1:] + f[:-1] * v[:-1])
            mom4 = alpha * 0.25 * ((v_o[1:] + v_o[:-1]) * (f[1:] + f[:-1]) -
                                   (v[1:] + v[:-1]) * (f_o[1:] + f_o[:-1]))
            mom5 = 1.0 / deta * (b_o[1:] * v_o[1:] - b_o[:-1] * v_o[:-1])
            mom6 = (1.0 + alpha) * 0.5 * (u_o[1:]**2 + u_o[:-1]**2)
            mom7 = (1.0 - alpha) * 0.5 * (f_o[1:] * v_o[1:] +
                                          f_o[:-1] * v_o[:-1])

            eq4[1:] = mom1 + mom2 + mom3 + mom4 + mom5 + mom6 + mom7

            # PDE: Energy
            ene1 = 1.0 / deta * (e[1:] * p[1:] - e[:-1] * p[:-1])
            ene2 = (1.0 + alpha) * 0.5 * (f[1:] * p[1:] + f[:-1] * p[:-1])
            ene3 = alpha * (0.5 * (u[1:] * g[1:] + u[:-1] * g[:-1]) +
                            0.25 * ((u_o[1:] + u_o[:-1]) * (g[1:] + g[:-1]) -
                                    (u[1:] + u[:-1]) * (g_o[1:] + g_o[:-1]) +
                                    (p[1:] + p[:-1]) * (f_o[1:] + f_o[:-1]) -
                                    (p_o[1:] + p_o[:-1]) * (f[1:] + f[:-1])))
            ene4 = 1.0 / deta * (e_o[1:] * p_o[1:] - e_o[1:] * p_o[1:])
            ene5 = (1.0 - alpha) * 0.5 * \
                (f_o[1:] * p_o[1:] + f_o[:-1] * p_o[:-1])
            ene6 = alpha * 0.5 * (u_o[1:] * g_o[1:] + u_o[:-1] * g_o[:-1])

            eq5[1:] = ene1 + ene2 - ene3 + ene4 + ene5 + ene6

            # boundary condintions make up another 5 equations
            # put them on the 0-th element of all 5 equations
            eq1[0] = f[0]
            eq2[0] = v[0]
            eq3[0] = p[0]
            eq4[0] = u[-1]
            eq5[0] = g[-1]

            return np.array([eq1, eq2, eq3, eq4, eq5]).ravel()

        # initial guess
        guess = np.array([f_o, u_o, v_o, g_o, p_o]).ravel()

        if solver_type == 'newton_krylov':
            # partial used to be able to send extra arguments
            # to the newton_krylov solver
            # those have to be the initial arguments in the function call F
            F_partial = partial(F,
                                F_args=[nx, gsi, deta, etamax, b, e, b_o, e_o])
            solution = optimize.newton_krylov(F_partial, guess,
                                              method='lgmres',
                                              verbose=1,
                                              iter=iterations)
        elif solver_type == 'fsolve':
            F_partial = partial(F,
                                F_args=[nx, gsi, deta, etamax, b, e, b_o, e_o])
            solution = optimize.fsolve(F_partial, guess,
                                       full_output=True, xtol=1e-06)
            solver_message = solution[3]
            print('  Solver: {}'.format(solver_message))
        elif solver_type == 'broyden1':
            pass

        return solution, solver_message

    def shift_profiles(self, solution):

        if DEBUG:
            print('solution', solution)
            print('len(solution)', len(solution))

        self.f[:, 0] = copy.copy(solution[0][0 * self.etamax:1 * self.etamax])
        self.u[:, 0] = copy.copy(solution[0][1 * self.etamax:2 * self.etamax])
        self.v[:, 0] = copy.copy(solution[0][2 * self.etamax:3 * self.etamax])
        self.g[:, 0] = copy.copy(solution[0][3 * self.etamax:4 * self.etamax])
        self.p[:, 0] = copy.copy(solution[0][4 * self.etamax:5 * self.etamax])
        self.b[:, 0] = copy.copy(self.b[:, 1])
        self.e[:, 0] = copy.copy(self.e[:, 1])

    def print_stage_header(self):

        text = ' Jet propagation: GSI = %s at stage %s' % \
            (self.gsi[self.nx], self.nx)

        if self.turbulent:
            text = text + ' - TURBULENT Flow'
        else:
            text = text + ' - LAMINAR Flow'

        textwidth = len(text)

        print(' ')
        print(' ')
        print('*' * textwidth)
        print(text)
        if self.nx == 0:
            print(' Initial velocity profile')
        print('*' * textwidth)
        print(' ')

    def print_result(self):
        print('  Viscosity (B)   = {: .4e}'.format(self.b[0, 1]))
        print('  Diffusivity (E) = {: .4e}'.format(self.e[0, 1]))
        print(' ')
        print('   ' + '=' * 67)
        print('  {:>2} {:^6}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}'
              .format('J', 'ETA', 'F', 'U', 'V', 'G', 'P'))
        print('   ' + '=' * 67)

        for j in range(self.etamax):

            print('{:4d} {:5.2f} {: .4e} {: .4e} {: .4e} {: .4e} {: .4e}'
                  .format(j, self.eta[j], self.f[j, 0], self.u[j, 0],
                          self.v[j, 0], self.g[j, 0], self.p[j, 0]))

    def store_result(self):
        nx = copy.copy(self.nx)
        gsi = copy.copy(self.gsi[self.nx])
        eta = copy.copy(self.eta)
        f = copy.copy(self.f[:, 0])
        u = copy.copy(self.u[:, 0])
        v = copy.copy(self.v[:, 0])
        g = copy.copy(self.g[:, 0])
        p = copy.copy(self.p[:, 0])
        b = copy.copy(self.b[:, 0])
        e = copy.copy(self.e[:, 0])
        self.results.append([nx, gsi, eta, f, u, v, g, p, b, e,
                             self.urel_visc, self.solver_message])

        # capture min, max values for scaling plots later
        for value in (f, u, v, g, p):
            self.scale_min = min(self.scale_min, min(value))
            self.scale_max = max(self.scale_max, max(value))

    def save_result(self, filename='results.dat'):

        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)

        filename = os.path.join(self.result_folder, filename)

        with open(filename, 'w') as f:
            f.write('#\n')
            f.write('# 2D TURBULENT HEATED FREE JET\n')
            f.write('#\n')
            f.write(' REYNOLDS = {:d}\n'.format(int(self.Reynolds)))
            f.write(' PRANDTL = {}\n'.format(self.Prandtl))
            f.write(' PRANDTL turbulent = {}\n'.format(self.Prandtl_turb))
            f.write(' gsi_0={:f}\n'.format(self.gsi[0]))
            f.write(' dgsi={:f}\n'.format(self.dgsi))
            f.write(' deta={:2.5f}\n'.format(self.deta[0]))
            f.write(' eta_e={:2.5f}\n'.format(self.etae))
            f.write(' eta_max={:2.5f}\n'.format(self.eta[-1]))
            f.write(' strech factor={}\n'.format(self.stretch))

            for result in self.results:
                f.write('\n\n')
                f.write(' Jet propagation: GSI = {:6.3} at stage {}\n'.
                        format(result[1], result[0]))
                f.write('  Solver: {}\n'.
                        format(result[11]))
                f.write('  Eddy viscosity under-relaxation = {:4.2f}\n'.
                        format(result[10]))
                f.write('  Viscosity (B)   = {: .4e}\n'.format(result[8][0]))
                f.write('  Diffusivity (E) = {: .4e}\n\n'.format(result[9][0]))
                f.write('   ' + '=' * 67 + '\n')
                f.write('{:>2}{:^8}{:^12}{:^12}{:^12}{:^12}{:^12}\n'
                        .format('J', 'ETA', 'F', 'U', 'V', 'G', 'P'))
                f.write('   ' + '=' * 67 + '\n')
                for j in range(self.etamax):
                    f.write('{:4d}{:5.2f}{:{f}}{:{f}}{:{f}}{:{f}}{:{f}}\n'.
                            format(j, result[2][j], result[3][j], result[4][j],
                                   result[5][j], result[6][j], result[7][j],
                                   f=' .4e'))

    def plot(self, steps=[]):

        if not os.path.exists(self.plot_folder):
            os.mkdir(self.plot_folder)

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)

        for step in steps:
            fig, ax = plt.subplots(figsize=(16, 8))

            try:
                result = self.results[step]
            except IndexError:
                print('Step {} not stored in results!'. format(step))
                continue

            ax.set_xlim(0.0, self.eta[-1])
            ax.set_ylim(self.scale_min, self.scale_max)
            ax.plot(result[2], result[3], label='F')
            ax.plot(result[2], result[4], label='U')
            ax.plot(result[2], result[5], label='V')
            ax.plot(result[2], result[6], label='G')
            ax.plot(result[2], result[7], label='P')

            text1 = '\n'.join((
                r'$Step={:03d}$'.format(result[0]),
                r'$\xi={:5.3f}$'.format(result[1]),
                r'$\eta_m={:2.3f}$'.format(self.eta[-1]),
                r'$Reynolds={:d}$'.format(int(self.Reynolds)),
                r'$Prandtl={}$'.format(self.Prandtl),
                r'$Prandtl_t={}$'.format(self.Prandtl_turb)))

            text2 = '\n'.join((
                r'$\xi_0={:1.3f}$'.format(self.gsi[0]),
                r'$d\xi={:5.3f}$'.format(self.dgsi),
                r'$d\eta_0={:2.3f}$'.format(self.deta[0]),
                r'$\eta_e={:2.2f}$'.format(self.etae),
                r'$stretch factor={}$'.format(self.stretch)))

            # place text boxes
            ax.text(0.86, 0.24, text1, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

            ax.text(0.1, 0.24, text2, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            plt.legend(loc='upper right')
            figname = os.path.join(self.plot_folder,
                                   'profiles_{:04d}.png'.format(step))
            print('Creating {}'.format(figname))
            plt.savefig(figname, dpi=150)
            plt.close()

    def main(self, solver_type='newton_krylov', iterations=None):

        # print thermo-physical fluid properties
        self.print_FluidAndFlowInformation()

        # initial velocity profile
        self.print_stage_header()
        self.boundary_conditions()
        self.store_result()
        self.print_result()

        for self.nx in range(1, self.gsimax):
            self.print_stage_header()
            solution, self.solver_message = self.solver(solver_type,
                                                        iterations)
            self.shift_profiles(solution)
            self.store_result()
            self.print_result()


if __name__ == "__main__":

    jet = Jet()

    jet.plot_folder = 'PLOTS'
    jet.result_folder = 'RESULTS'

    # make mesh
    jet.mesh(gsimax=30, dgsi=0.03, etae=13, deta1=0.03, stretch=1.1)

    # define fluid properties
    RE = 30000.0
    PR = 0.70
    PRt = 0.9
    jet.set_FluidProperties(RE, PR, PRt)

    # specifiy flow type (False=laminar, True=turbulent)
    jet.turbulent = True

    # run main program
    # solver types ('newton_krylov', 'broyden1', 'fsolve')
    jet.main(solver_type='fsolve')

    # print(jet.results)

    # save results as text file
    jet.save_result(filename='results.dat')

    # plot results
    jet.plot(steps=range(len(jet.results)))
