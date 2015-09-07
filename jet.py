#!c:/python27/python.exe

import sys

import numpy as np
from numpy import sqrt, tanh
import scipy.optimize as opt
import toyplot
import toyplot.browser

__author__ = 'Andreas Ennemoser'
__copyright__ = 'Copyright 2015'
__license__ = 'MIT'
__version__ = '0.9'
__email__ = 'andreas.ennemoser@gmail.com'
__status__ = 'Development'


class Jet(object):
    """Numerical analysis of a 2D, heated, turbulent free jet

    Literature: T. CEBECI, P. BRADSHAW
    "Physical and computational aspects of convective heat transfer"
    Springer 1984

    """

    def __init__(self):
        """Summary"""

        self.Reynolds = 10000.0
        self.Prandtl = 0.72
        self.Prandtl_turb = 0.9
        self.turbulent = True

        # initialize arrays
        self.ini_arrays(1000)

        self.nx = 1
        self.etae = 8
        self.epsilon = 0.0001
        self.max_iterations = 20

        # centerline boundary condition for the energy equation (dp0=0)
        # switch between temperature or heat flux boundary type
        self.alfa0 = 0.0
        self.alfa1 = 1.0

        print ''
        print '**************************************************'
        print '**************************************************'
        print '********** 2D TURBULENT HEATED FREE JET **********'
        print '**************************************************'
        print '**************************************************'

    def ini_arrays(self, size):
        self.f = np.zeros([size, 2])
        self.u = np.zeros([size, 2])
        self.v = np.zeros([size, 2])
        self.g = np.zeros([size, 2])
        self.p = np.zeros([size, 2])
        self.b = np.zeros([size, 2])
        self.e = np.zeros([size, 2])

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

    def print_FluidProperties(self):
        print ' '
        print '***************************'
        print '***** FLOW PROPERTIES *****'
        print '***************************'
        print ' REYNOLDS = %s' % (self.Reynolds)
        print ' PRANDTL = %s' % (self.Prandtl)
        print ' PRANDTL turbulent = %s' % (self.Prandtl_turb)

    def mesh(self, gsimax=1000, dgsi=0.01, etae=8, deta1=0.01, stretch=1.12):
        """Mesh generation for 2D rectangular transformed grid

        Args:
            gsimax (int, optional): Description
            dgsi (float, optional): Description
            etae (int, optional): Description
            deta1 (float, optional): Description
            stretch (float, optional): Description
        """

        self.etae = etae

        # calculation of grid points from spacing parameters
        if stretch < 1.0001:
            etamax = etae / deta1
        else:
            # equation (13.49, page 400)
            etamax = np.log(1.0 + (stretch - 1.0) * etae / deta1) / \
                     np.log(stretch)

        self.etamax = int(etamax)
        self.gsimax = gsimax

        # initialize grid arrays
        self.gsi = np.zeros([self.gsimax+1])
        self.eta = np.zeros([self.etamax+1])
        self.deta = np.zeros([self.etamax+1])
        self.a = np.zeros([self.etamax+1])

        self.gsi[0] = 0.0000001
        self.deta[0] = deta1
        stretch = 1.12

        for i in range(1, self.gsimax+1):
            self.gsi[i] = self.gsi[i-1] + dgsi

        for j in range(1, self.etamax):
            self.deta[j] = stretch * self.deta[j-1]
            self.a[j] = 0.5 * self.deta[j-1]
            self.eta[j] = self.eta[j-1] + self.deta[j-1]

        print ' '
        print '***************************'
        print '***** MESH PROPERTIES *****'
        print '***************************'
        print ' GSI spacing: ', dgsi
        print ' ETA spacing (initial): ', deta1
        print ' ETA growth rate: ', stretch
        print ' ETA boundary (ETAE): ', etae
        print ' Number of grid points in GSI direction: ', self.gsimax
        print ' Number of grid points in ETA direction: ', self.etamax

    def ivpl(self):
        """Initial velocity and temperature profiles.
        Equation (14.31), page 445
        """

        gsi0 = self.gsi[self.nx]

        # boundary condition at jet center (symmetry)
        self.f[0, 1] = 0.0

        beta = 27.855
        etac = 1.0
        term = beta * 3.0 * gsi0**(2./3.) / sqrt(self.Reynolds)

        for j in range(1, self.etamax+1):

            tanf = tanh(term * (self.eta[j] - etac))

            self.u[j, 1] = 3.0/2.0 * gsi0**(1./3.) * (1.0 - tanf)
            self.v[j, 1] = - term * 3.0/2.0 * gsi0**(1./3.) * (1.0 - tanf**2)
            self.g[j, 1] = self.u[j, 1] / 3.0 * gsi0**(1./3.)
            self.p[j, 1] = self.v[j, 1] / 3.0 * gsi0**(1./3.)
            self.b[j, 1] = 1.0
            self.e[j, 1] = 1.0 / self.Prandtl

            # check this
            # check this
            # check this
            if j > 0:
                self.f[j, 1] = self.f[j-1, 1] + self.a[j] * \
                    (self.u[j, 1] + self.u[j-1, 1])

        if self.turbulent:
            self.turbulence()

    def turbulence(self):

        umaxh = 0.5 * self.u[1, 1]

        for j in range(1, self.etamax+1):
            if self.u[j, 1] > umaxh:
                etab = self.eta[self.etamax]
            else:
                etab = self.eta[j-1] + (self.eta[j] - self.eta[j-1]) / \
                    (self.u[j, 1] - self.u[j-1, 1]) * (umaxh - self.u[j-1, 1])

        # compute turbulence viscosity
        edv = 0.037 * etab * self.u[0, 1] * np.sqrt(self.Reynolds) * \
            self.gsi[self.nx]**(1./3.)

        for j in range(self.etamax):
            self.b[j, 1] = 1.0 + edv
            self.e[j, 1] = 1.0 / self.Prandtl + edv / self.Prandtl_turb

    def solver(self):

        def residual(self, variables):
            """Summary

            Args:
                variables (TYPE): Description

            Returns:
                TYPE: Description
            """
            # All 5 unknowns
            (f, u, v, g, p, b, e) = variables
            deta = self.deta
            gsi = self.gsi
            nx = self.nx
            alpha = 3.0 / 2.0 * (gsi[nx] + gsi[nx-1]) / (gsi[nx] - gsi[nx-1])

            if self.turbulent:
                self.turbulence()

            # array slicing for indices j and n
            # index j   means [1:, ]  in first array column
            # index j-1 means [:-1, ] in first array column
            # index n   means [, 1] in second array column
            # index n-1 means [, 0] in second array column

            # boundary conditions at jet center (symmetry)
            f[0, :] = 0.0
            v[0, :] = 0.0
            p[0, :] = 0.0

            # boundary conditions at jet outer boundary (ambient)
            u[-1, :] = 0.0
            g[-1, :] = 0.0

            # ODE: f' = u
            eq1 = 1.0/deta*(f[1:, 1] - f[:-1, 1]) - 0.5*(u[2:, 1] + u[1:-1, 1])

            # ODE: u' = v
            eq2 = 1.0/deta*(u[1:, 1] - u[:-1, 1]) - 0.5*(v[2:, 1] + v[1:-1, 1])

            # ODE: g' = p
            eq3 = 1.0/deta*(g[1:, 1] - g[:-1, 1]) - 0.5*(p[2:, 1] + p[1:-1, 1])

            # PDE: Momentum
            eq4 = 1.0/deta * (b[1:, 1]*v[1:, 1] - b[:-1, 1]*v[:-1, 1]) + \
                (1.0-alpha)*0.5*(u[1:, 1]**2 - u[:-1, 1]**2) + \
                (1.0+alpha)*0.5*(f[1:, 1]*v[1:, 1] - f[:-1, 1]*v[:-1, 1]) + \
                alpha*0.25*((v[1:, 0]+v[:-1, 0])*(f[1:, 1]+f[:-1, 1]) -
                            (f[1:, 0]+f[:-1, 0])*(v[1:, 1]+v[:-1, 1])) + \
                1.0/deta * (b[1:, 0]*v[1:, 0] - b[:-1, 0]*v[:-1, 0]) + \
                (1.0+alpha)*0.5*(u[1:, 0]**2 - u[:-1, 0]**2) + \
                (1.0-alpha)*0.5*(f[1:, 0]*v[1:, 0] - f[:-1, 0]*v[:-1, 0])

            # PDE: Energy
            eq5 = 1.0/deta * (e[1:, 1]*p[1:, 1] - e[:-1, 1]*p[:-1, 1]) + \
                (1.0+alpha)*0.5*(f[1:, 1]*p[1:, 1] - f[:-1, 1]*p[:-1, 1]) - \
                alpha*(0.5*(u[1:, 1]*g[1:, 1] + u[:-1, 1]*g[:-1, 1]) +
                       0.25*((u[1:, 0]+u[:-1, 0])*(g[1:, 1]+g[:-1, 1]) -
                       (g[1:, 0]+g[:-1, 0])*(u[1:, 1]+u[:-1, 1]) +
                       (p[1:, 1]+p[:-1, 1])*(f[1:, 0]+f[:-1, 0]) -
                       (p[1:, 0]+p[:-1, 0])*(f[1:, 1]+f[:-1, 1]))) + \
                1.0/deta * (e[1:, 0]*p[1:, 0] - e[:-1, 0]*p[:-1, 0]) + \
                (1.0-alpha)*0.5*(f[1:, 0]*p[1:, 0] - f[:-1, 0]*p[:-1, 0]) + \
                alpha*0.5*(u[1:, 0]*u[1:, 0] - g[:-1, 0]*g[:-1, 0])

            return [eq1, eq2, eq3, eq4, eq5]

        def log_iterations(x, f):
            """This function is called on every iteration as callback(x, f)
            where x is the current solution and f the corresponding residual.

            Args:
                x (TYPE): Description
                f (TYPE): Description

            Returns:
                TYPE: Description
            """
            # current solution
            print x
            # current residual
            print f

        # run solver
        solution = opt.newton_krylov(residual, guess,
                                     method='lgmres',
                                     verbose=1,
                                     callback=log_iterations)

        return solution

    def print_stage_header(self):

        text = ' Jet propagation: GSI = %s at stage %s' % \
            (self.gsi[self.nx], self.nx)

        if self.nx == 1:
            text = text + ' - Initial velocity profile'

        textwidth = len(text)

        print ' '
        print ' '
        print '*' * textwidth
        print text
        print '*' * textwidth
        print ' '

    def print_results(self):
        print '  {:>2} {:^6}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}' \
            .format('J', 'ETA', 'F', 'U', 'V', 'G', 'P', 'B', 'E')

        for j in range(1, self.etamax+1):

            print '{:4d} {: .2f} {: .4e} {: .4e} {: .4e} {: .4e} {: .4e} {: .4e} {: .4e}' \
                .format(j, self.eta[j], self.f[j, 2], self.u[j, 2],
                        self.v[j, 2], self.g[j, 2], self.p[j, 2], self.b[j, 2],
                        self.e[j, 2])

    def shift_profiles(self):
        for j in range(1, self.etamax+1):
            self.f[j, 0] = self.f[j, 1]
            self.u[j, 0] = self.u[j, 1]
            self.v[j, 0] = self.v[j, 1]
            self.g[j, 0] = self.g[j, 1]
            self.p[j, 0] = self.p[j, 1]
            self.e[j, 0] = self.e[j, 1]
            self.b[j, 0] = self.b[j, 1]

    def convergence_check(self, iteration):
        """Summary

        Args:
            iteration (INT): Description

        Returns:
            BOOL: True, if convergence achieved, else False
        """
        converged = False

        residual = np.abs(self.delu[1] / self.u[1, 2])

        print 'Iteration = {:2d}, EPSILON = {:.2e}, RESIDUAL = {:.4e}' \
            .format(iteration, self.epsilon, residual)

        if residual < self.epsilon:
            converged = True

        if converged:
            print ' '
            print '*** CONVERGED ***'
            print ' '
        else:
            if iteration == self.max_iterations:
                print ' '
                print '*** NOT CONVERGED ***'
                print ' '
                sys.exit('Execution stopped.')

        return converged

    def plot(self):
        canvas = toyplot.Canvas(width=900, height=900)
        axes = canvas.axes()
        x = self.eta[1:self.etamax]
        y = self.u[1:self.etamax, 2]
        mark = axes.plot(x, y)
        mark1 = axes.scatterplot(x, y)
        toyplot.browser.show(canvas)

    def main(self):

        # print thermo-physical fluid properties
        self.print_FluidProperties()

        # initial velocity profile
        self.print_stage_header()
        self.ivpl()
        self.print_results()
        self.shift_profiles()
        self.plot()

        for self.nx in range(2, self.gsimax):
            self.print_stage_header()

            solution = self.solver()
            print solution

            self.print_results()
            self.shift_profiles()
            self.plot()


if __name__ == "__main__":

    jet = Jet()

    # make mesh
    jet.mesh(gsimax=1000, dgsi=0.01, etae=8, deta1=0.01, stretch=1.15)

    # define properties
    jet.set_FluidProperties(5300.0, 0.72, 0.5)

    # run main program
    jet.main()
