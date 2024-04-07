#!bin/env python

import os
from functools import partial
import copy

import numpy as np
from numpy import sqrt, tanh
from scipy import optimize
import matplotlib.pyplot as plt

__author__ = 'Andreas Ennemoser'
__copyright__ = 'Copyright 2024'
__license__ = 'MIT'
__version__ = '1.0'
__email__ = 'andreas.ennemoser@aon.at'
__status__ = 'Development'

DEBUG = False


class Jet():
    """
    Numerical analysis of a 2D, heated, turbulent free jet.

    This class implements a numerical solver for analyzing the flow and heat transfer in a 2D turbulent free jet. The solver is based on the work of T. Cebeci and P. Bradshaw, as described in their book "Physical and Computational Aspects of Convective Heat Transfer".

    The solver takes into account various parameters such as Reynolds number, Prandtl number, turbulent Prandtl number, and whether the flow is turbulent or not. It uses a transformed grid approach for mesh generation and solves the governing equations for velocity, temperature, and turbulence variables.

    The class provides methods for setting the fluid properties, printing the fluid and flow information, generating the mesh, setting the boundary conditions, solving the equations, and plotting the results.

    Example usage:
    ```
    jet = Jet(Reynolds=10000, Prandtl=0.7, Prandtl_turb=0.9, turbulent=True)
    jet.set_FluidProperties(Reynolds=20000, Prandtl=0.8, Prandtl_turb=1.0)
    jet.print_FluidAndFlowInformation()
    jet.mesh(gsimax=10, dgsi=0.01, etae=8, deta1=0.01, stretch=1.12)
    jet.boundary_conditions()
    jet.solver(solver_type='newton_krylov', iterations=100)
    jet.plot(steps=[0, 10, 20, 30, 40, 50])
    ```
    """

    def __init__(self, Reynolds, Prandtl, Prandtl_turb, turbulent):
        """
        Initialize the Jet object.

        Args:
            Reynolds (float): The Reynolds number.
            Prandtl (float): The Prandtl number.
            Prandtl_turb (float): The turbulent Prandtl number.
            turbulent (bool): Flag indicating whether the flow is turbulent or not.
        """

        self.Reynolds = Reynolds
        self.Prandtl = Prandtl
        self.Prandtl_turb = Prandtl_turb
        self.turbulent = turbulent
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
        print('*' * 50)
        print('*' * 50)
        print(' 2D TURBULENT HEATED FREE JET '.center(50, '*'))
        print('*' * 50)
        print('*' * 50)

    def set_Reynolds(self, Reynolds):
        """
        Set the Reynolds number.

        Args:
            Reynolds (float): The Reynolds number.
        """
        self.Reynolds = Reynolds

    def set_Prandtl(self, Prandtl):
        """
        Set the Prandtl number.

        Args:
            Prandtl (float): The Prandtl number.
        """
        self.Prandtl = Prandtl

    def set_Prandtl_turb(self, Prandtl_turb):
        """
        Set the turbulent Prandtl number.

        Args:
            Prandtl_turb (float): The turbulent Prandtl number.
        """
        self.Prandtl_turb = Prandtl_turb

    def set_FluidProperties(self, Reynolds, Prandtl, Prandtl_turb):
        """
        Set the fluid properties.

        Args:
            Reynolds (float): The Reynolds number.
            Prandtl (float): The Prandtl number.
            Prandtl_turb (float): The turbulent Prandtl number.
        """
        self.set_Reynolds(Reynolds)
        self.set_Prandtl(Prandtl)
        self.set_Prandtl_turb(Prandtl_turb)

    def print_FluidAndFlowInformation(self):
        """
        Print the fluid and flow information.
        """
        print('')
        print('*' * 30)
        print(' FLOW PROPERTIES '.center(30, '*'))
        print('*' * 30)
        print(' REYNOLDS = %s' % (self.Reynolds))
        print(' PRANDTL = %s' % (self.Prandtl))
        print(' PRANDTL turbulent = %s' % (self.Prandtl_turb))
        if self.turbulent:
            print(' TURBULENCE = ON')
        else:
            print(' TURBULENCE = OFF')

    def mesh(self, gsimax=10, dgsi=0.01, etae=8, deta1=0.01, stretch=1.12):
        """Generate a 2D rectangular transformed grid for mesh generation.

        Args:
            gsimax (int, optional): The maximum number of grid points in the GSI direction.
            dgsi (float, optional): The spacing between grid points in the GSI direction.
            etae (int, optional): The boundary value for the ETA direction.
            deta1 (float, optional): The initial spacing between grid points in the ETA direction.
            stretch (float, optional): The growth rate of the spacing between grid points in the ETA direction.
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
        print('*' * 30)
        print(' MESH PROPERTIES '.center(30, '*'))
        print('*' * 30)
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
        This method calculates the initial velocity and temperature profiles for the jet flow. It implements Equation (14.31) from the book "Physical and Computational Aspects of Convective Heat Transfer" by T. Cebeci and P. Bradshaw.

        The method uses the current values of the grid points and boundary conditions to calculate the initial profiles. It takes into account the Reynolds number, Prandtl number, turbulent Prandtl number, and whether the flow is turbulent or not.

        The calculated profiles are stored in the `u` and `g` arrays, which represent the velocity and temperature profiles respectively.

        Args:
            None

        Returns:
            None
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
        """
        Calculate the turbulence variables.

        This method calculates the turbulence variables, such as eddy viscosity and turbulent diffusivity, based on the current flow conditions and fluid properties. It takes into account the Reynolds number, Prandtl number, turbulent Prandtl number, and whether the flow is turbulent or not.

        The calculated turbulence variables are stored in the `b` and `e` arrays, which represent the eddy viscosity and turbulent diffusivity respectively.

        Args:
            None

        Returns:
            None
        """
        # Implementation of turbulence calculations
        # ...
        pass

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
        """
        Solve the equations.

        This method solves the governing equations for velocity, temperature, and turbulence variables using the specified solver type and number of iterations.

        Args:
            solver_type (str): The type of solver to use.
            iterations (int): The number of iterations to perform.

        Returns:
            tuple: A tuple containing the solution and solver message.
        """

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
            """
            Calculate the residual of the equations.

            This method calculates the residual of the governing equations for velocity, temperature, and turbulence variables. It takes the unknowns (f, u, v, g, p) as input and returns the residual.

            Args:
            unknowns (np.array): The unknowns (f, u, v, g, p).

            Returns:
            np.array: The residual of the equations.
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
            momentum_1 = 1.0 / deta * (b[1:] * v[1:] - b[:-1] * v[:-1])
            momentum_2 = (1.0 - alpha) * 0.5 * (u[1:]**2 + u[:-1]**2)
            momentum_3 = (1.0 + alpha) * 0.5 * (f[1:] * v[1:] + f[:-1] * v[:-1])
            momentum_4 = alpha * 0.25 * ((v_o[1:] + v_o[:-1]) * (f[1:] + f[:-1]) -
                                   (v[1:] + v[:-1]) * (f_o[1:] + f_o[:-1]))
            momentum_5 = 1.0 / deta * (b_o[1:] * v_o[1:] - b_o[:-1] * v_o[:-1])
            momentum_6 = (1.0 + alpha) * 0.5 * (u_o[1:]**2 + u_o[:-1]**2)
            momentum_7 = (1.0 - alpha) * 0.5 * (f_o[1:] * v_o[1:] +
                                          f_o[:-1] * v_o[:-1])

            eq4[1:] = momentum_1 + momentum_2 + momentum_3 + momentum_4 + \
                      momentum_5 + momentum_6 + momentum_7

            # PDE: Energy
            energy_1 = 1.0 / deta * (e[1:] * p[1:] - e[:-1] * p[:-1])
            energy_2 = (1.0 + alpha) * 0.5 * (f[1:] * p[1:] + f[:-1] * p[:-1])
            energy_3 = alpha * (0.5 * (u[1:] * g[1:] + u[:-1] * g[:-1]) +
                            0.25 * ((u_o[1:] + u_o[:-1]) * (g[1:] + g[:-1]) -
                                    (u[1:] + u[:-1]) * (g_o[1:] + g_o[:-1]) +
                                    (p[1:] + p[:-1]) * (f_o[1:] + f_o[:-1]) -
                                    (p_o[1:] + p_o[:-1]) * (f[1:] + f[:-1])))
            energy_4 = 1.0 / deta * (e_o[1:] * p_o[1:] - e_o[1:] * p_o[1:])
            energy_5 = (1.0 - alpha) * 0.5 * \
                (f_o[1:] * p_o[1:] + f_o[:-1] * p_o[:-1])
            energy_6 = alpha * 0.5 * (u_o[1:] * g_o[1:] + u_o[:-1] * g_o[:-1])

            eq5[1:] = energy_1 + energy_2 - energy_3 + energy_4 + energy_5 + \
                      energy_6

            # boundary conditions make up another 5 equations
            # put them on the 0-th element of all 5 equations
            eq1[0] = f[0]
            eq2[0] = v[0]
            eq3[0] = p[0]
            eq4[0] = u[-1]
            eq5[0] = g[-1]

            return np.array([eq1, eq2, eq3, eq4, eq5]).ravel()

        # initial guess
        guess = np.array([f_o, u_o, v_o, g_o, p_o]).ravel()

        # partial used to be able to send extra arguments to the solver
        # those have to be the initial arguments in the function call F
        F_partial = partial(F, F_args=[nx, gsi, deta, etamax, b, e, b_o, e_o])

        if solver_type == 'newton_krylov':
            solution = optimize.newton_krylov(F_partial, guess,
                                              method='lgmres',
                                              verbose=1,
                                              iter=iterations)
        elif solver_type == 'fsolve':
            solution = optimize.fsolve(F_partial, guess,
                                       full_output=True, xtol=1e-06)
            solver_message = solution[3]
            print('  Solver: {}'.format(solver_message))
        elif solver_type == 'broyden1':
            solution = optimize.broyden1(F_partial, guess,
                                         f_tol=1e-06, iter=iterations)
            solver_message = 'Broyden1 solver'

        return solution, solver_message

    def shift_profiles(self, solution):
        """
        Shifts the profiles of various variables based on the given solution.

        Args:
            solution (list): A list containing the solution values.

        Returns:
            None
        """

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
        """
        Prints the stage header for the Jet propagation.

        The stage header includes the GSI value and the stage number. If the flow is turbulent,
        it also indicates that. Additionally, it prints the initial velocity profile if the stage
        number is 0.

        Args:
            None

        Returns:
            None
        """
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
        """
        Prints the result of the JET calculation.

        This method prints the viscosity, diffusivity, and other calculated values
        for each iteration of the JET calculation to the terminal.

        Args:
            None

        Returns:
            None
        """
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
        """
        Stores the current state of the JET object's attributes into the results list.

        This method creates copies of the relevant attributes and appends them to the results list.
        It also captures the minimum and maximum values for scaling plots later.

        Args:
            None

        Returns:
            None
        """
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
        """
        Save the results of the 2D turbulent heated free jet simulation to a file.

        Args:
            filename (str, optional): The name of the file to save the results to. Defaults to 'results.dat'.
        """

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
            f.write(' stretch factor={}\n'.format(self.stretch))

            # text for turbulence
            if self.turbulent:
                text = ' - TURBULENT Flow'
            else:
                text = ' - LAMINAR Flow'

            for result in self.results:
                f.write('\n\n')
                f.write('*' * 61 + '\n')
                msg = f' Jet propagation: GSI = {result[1]:6.3} at stage {result[0]}' + text + '\n'
                f.write(msg)
                f.write('*' * 61 + '\n\n')
                f.write('  Eddy viscosity under-relaxation = {:4.2f}\n'.
                        format(result[10]))
                f.write('  Solver: {}\n'.
                        format(result[11]))
                f.write('  Viscosity (B)   = {: .4e}\n'.format(result[8][0]))
                f.write('  Diffusivity (E) = {: .4e}\n\n'.format(result[9][0]))
                f.write('   ' + '=' * 67 + '\n')
                f.write('{:>4}{:^8}{:^11}{:^13}{:^12}{:^12}{:^12}\n'
                        .format('J', 'ETA', 'F', 'U', 'V', 'G', 'P'))
                f.write('   ' + '=' * 67 + '\n')
                for j in range(self.etamax):
                    f.write('{:4d} {:5.2f} {: .4e} {: .4e} {: .4e} {: .4e} {: .4e}\n'.
                            format(j, result[2][j], result[3][j], result[4][j],
                                   result[5][j], result[6][j], result[7][j]))
                    
        print('\n')
        print('*' * 60)
        print('{} {}'.format('   Results saved to:', os.path.abspath(filename)))
        print('*' * 60)
        print('')

    def plot(self, steps=[]):
        """
        Create and save plots for each step in the given list of steps.

        Parameters:
        - steps (list): A list of step numbers for which plots will be created.

        Returns:
        - None

        This method creates plots for each step in the given list of steps. The plots
        show the values of F, U, V, G, and P as a function of xi. The plots are saved
        as PNG images in the plot_folder directory.

        If the plot_folder directory does not exist, it will be created before saving
        the plots.

        Note: The method assumes that the necessary data is available in the results
        attribute of the object.

        Example usage:
        jet = JET()
        jet.plot([1, 2, 3])
        """

        # message to user
        print('*' * 60)
        print('{} {}'.format('   Creating plots in folder:', os.path.abspath(self.plot_folder)))
        print('*' * 60)

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
                r'$stretch\;factor={}$'.format(self.stretch)))

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
        """
        Main method for running the simulation.

        Args:
            solver_type (str, optional): Type of solver to use. Defaults to 'newton_krylov'.
            iterations (int, optional): Number of iterations to perform. Defaults to None.
        """

        # print thermo-physical fluid properties
        self.print_FluidAndFlowInformation()

        # initial velocity profile
        self.print_stage_header()
        self.boundary_conditions()
        self.store_result()
        self.print_result()

        for self.nx in range(1, self.gsimax):
            self.print_stage_header()
            solution, self.solver_message = self.solver(solver_type, iterations)
            self.shift_profiles(solution)
            self.store_result()
            self.print_result()


if __name__ == "__main__":

    # define fluid properties
    Reynolds = 30000.0
    Prandtl = 0.70
    Prandtl_turb = 0.9
    turbulent = True

    jet = Jet(Reynolds, Prandtl, Prandtl_turb, turbulent)

    jet.plot_folder = 'PLOTS'
    jet.result_folder = 'RESULTS'

    # make mesh
    jet.mesh(gsimax=30, dgsi=0.03, etae=13, deta1=0.03, stretch=1.1)

    # run main program
    # solver types ('newton_krylov', 'broyden1', 'fsolve')
    jet.main(solver_type='fsolve')

    # save results as text file
    jet.save_result(filename='results.dat')

    # plot results
    jet.plot(steps=range(len(jet.results)))

