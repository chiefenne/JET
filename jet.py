#!c:/python27/python.exe

import sys

import numpy as np
from numpy import sqrt, tanh
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
        self.max_iterations = 15

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
        self.f = np.zeros([size, 3])
        self.u = np.zeros([size, 3])
        self.v = np.zeros([size, 3])
        self.g = np.zeros([size, 3])
        self.p = np.zeros([size, 3])
        self.b = np.zeros([size, 3])
        self.e = np.zeros([size, 3])

        self.c = np.zeros([size, 3])
        self.d = np.zeros([size, 3])
        self.bc = np.zeros([size])

        self.delp = np.zeros([size])
        self.delv = np.zeros([size])
        self.delf = np.zeros([size])
        self.delg = np.zeros([size])
        self.delu = np.zeros([size])

        self.s1 = np.zeros([size])
        self.s2 = np.zeros([size])
        self.s3 = np.zeros([size])
        self.s4 = np.zeros([size])
        self.s5 = np.zeros([size])
        self.s6 = np.zeros([size])
        self.s7 = np.zeros([size])
        self.s8 = np.zeros([size])

        self.b1 = np.zeros([size])
        self.b2 = np.zeros([size])
        self.b3 = np.zeros([size])
        self.b4 = np.zeros([size])
        self.b5 = np.zeros([size])
        self.b6 = np.zeros([size])
        self.b7 = np.zeros([size])
        self.b8 = np.zeros([size])
        self.b9 = np.zeros([size])
        self.b10 = np.zeros([size])

        self.a11 = np.zeros([size])
        self.a12 = np.zeros([size])
        self.a13 = np.zeros([size])
        self.a14 = np.zeros([size])
        self.a15 = np.zeros([size])
        self.a21 = np.zeros([size])
        self.a22 = np.zeros([size])
        self.a23 = np.zeros([size])
        self.a24 = np.zeros([size])
        self.a25 = np.zeros([size])
        self.a31 = np.zeros([size])
        self.a32 = np.zeros([size])
        self.a33 = np.zeros([size])
        self.a34 = np.zeros([size])
        self.a35 = np.zeros([size])

        self.g11 = np.zeros([size])
        self.g12 = np.zeros([size])
        self.g13 = np.zeros([size])
        self.g14 = np.zeros([size])
        self.g15 = np.zeros([size])
        self.g21 = np.zeros([size])
        self.g22 = np.zeros([size])
        self.g23 = np.zeros([size])
        self.g24 = np.zeros([size])
        self.g25 = np.zeros([size])
        self.g31 = np.zeros([size])
        self.g32 = np.zeros([size])
        self.g33 = np.zeros([size])
        self.g34 = np.zeros([size])
        self.g35 = np.zeros([size])

        self.w1 = np.zeros([size])
        self.w2 = np.zeros([size])
        self.w3 = np.zeros([size])
        self.w4 = np.zeros([size])
        self.w5 = np.zeros([size])

        self.r = np.zeros([6, size])

    def set_Reynolds(self, Reynolds):
        self.Reynolds = Reynolds

    def set_Prandtl(self, Prandtl):
        self.Prandtl = Prandtl

    def set_Prandtl_turb(self, Prandtl_turb):
        self.Prandtl_turb = Prandtl_turb

    def set_Properties(self, Reynolds, Prandtl, Prandtl_turb):
        self.set_Reynolds(Reynolds)
        self.set_Prandtl(Prandtl)
        self.set_Prandtl_turb(Prandtl_turb)

    def print_Properties(self):
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

        self.gsi[1] = 0.0000001
        self.deta[1] = deta1
        stretch = 1.12

        for i in range(2, self.gsimax+1):
            self.gsi[i] = self.gsi[i-1] + dgsi

        for j in range(2, self.etamax+1):
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

        self.f[1, 2] = 0.0
        beta = 27.855
        etac = 1.0
        term = beta * 3.0 * gsi0**(2./3.) / sqrt(self.Reynolds)

        for j in range(1, self.etamax+1):

            tanf = tanh(term * (self.eta[j] - etac))

            self.u[j, 2] = 3.0/2.0 * gsi0**(1./3.) * (1.0 - tanf)
            self.v[j, 2] = - term * 3.0/2.0 * gsi0**(1./3.) * (1.0 - tanf**2)
            self.g[j, 2] = self.u[j, 2] / 3.0 * gsi0**(1./3.)
            self.p[j, 2] = self.v[j, 2] / 3.0 * gsi0**(1./3.)
            self.c[j, 2] = 1.0
            self.b[j, 2] = 1.0
            self.e[j, 2] = 1.0 / self.Prandtl
            self.d[j, 2] = 0.0

            self.c[j, 1] = 1.0
            self.d[j, 1] = 0.0
            self.bc[j] = 1.0

            if j > 1:
                self.f[j, 2] = self.f[j-1, 2] + self.a[j] * \
                    (self.u[j, 2] + self.u[j-1, 2])

        if self.turbulent:
            self.turbulence()

    def turbulence(self):

        umaxh = 0.5 * self.u[1, 2]

        for j in range(1, self.etamax+1):
            if self.u[j, 2] > umaxh:
                etab = self.eta[self.etamax]
            else:
                etab = self.eta[j-1] + (self.eta[j] - self.eta[j-1]) / \
                    (self.u[j, 2] - self.u[j-1, 2]) * (umaxh - self.u[j-1, 2])

        # compute turbulence viscosity
        edv = 0.037 * etab * self.u[1, 2] * np.sqrt(self.Reynolds) * \
            self.gsi[self.nx]**(1./3.)

        for j in range(1, self.etamax+1):
            self.e[j, 2] = edv / self.Prandtl_turb
            self.b[j, 2] = edv

    def coefficients(self):

        cel = 1.5 * (self.gsi[self.nx] + self.gsi[self.nx-1]) / \
                    (self.gsi[self.nx] - self.gsi[self.nx-1])

        # to be checked (page 447)
        p1p = 1.0 + cel
        p2p = -1.0 + cel

        for j in range(2, self.etamax+1):

            # current position
            usb = 0.5 * (self.u[j, 2]**2 + self.u[j-1, 2]**2)
            fvb = 0.5 * (self.f[j, 2] * self.v[j, 2] +
                         self.f[j-1, 2] * self.v[j-1, 2])
            fpb = 0.5 * (self.f[j, 2] * self.p[j, 2] +
                         self.f[j-1, 2] * self.p[j-1, 2])
            ugb = 0.5 * (self.u[j, 2] * self.g[j, 2] +
                         self.u[j-1, 2] * self.g[j-1, 2])
            fb = 0.5 * (self.f[j, 2] + self.f[j-1, 2])
            ub = 0.5 * (self.u[j, 2] + self.u[j-1, 2])
            vb = 0.5 * (self.v[j, 2] + self.v[j-1, 2])
            gb = 0.5 * (self.g[j, 2] + self.g[j-1, 2])
            pb = 0.5 * (self.p[j, 2] + self.p[j-1, 2])

            derbv = 0.5 * (self.b[j, 2] * self.v[j, 2]-self.b[j-1, 2] *
                           self.v[j-1, 2]) / self.deta[j-1]
            derep = 0.5 * (self.e[j, 2] * self.p[j, 2]-self.e[j-1, 2] *
                           self.p[j-1, 2]) / self.deta[j-1]

            # previous position
            cfb = 0.5 * (self.f[j, 1] + self.f[j-1, 1])
            cub = 0.5 * (self.u[j, 1] + self.u[j-1, 1])
            cvb = 0.5 * (self.v[j, 1] + self.v[j-1, 1])
            cgb = 0.5 * (self.g[j, 1] + self.g[j-1, 1])
            cpb = 0.5 * (self.p[j, 1] + self.p[j-1, 1])
            cusb = 0.5 * (self.u[j, 1] * self.u[j-1, 1]**2)
            cfvb = 0.5 * (self.f[j, 1] * self.v[j, 1] +
                          self.f[j-1, 1]*self.v[j-1, 1])
            cfpb = 0.5 * (self.f[j, 1] * self.p[j, 1] +
                          self.f[j-1, 1]*self.p[j-1, 1])
            cugb = 0.5 * (self.u[j, 1] * self.g[j, 1] +
                          self.u[j-1, 1]*self.g[j-1, 1])

            cderbv = (self.b[j, 1] * self.v[j, 1] - self.b[j-1, 1] *
                      self.v[j-1, 1]) / self.deta[j-1]
            cderep = (self.e[j, 1] * self.p[j, 1] - self.e[j-1, 1] *
                      self.p[j-1, 1]) / self.deta[j-1]

            clb = cderbv + cfvb + cusb
            crb = -clb - cel * cusb + cel * cfvb
            cmb = cderep + cfpb
            ctb = -cmb + cel * (cfpb - cugb)

            # coefficients of the differenced momentum equation

            self.s1[j] = self.b[j, 2] / self.deta[j-1] + \
                0.5*p1p*self.f[j, 2] - 0.5*cel*cfb
            self.s2[j] = -self.b[j-1, 2] / self.deta[j-1] + \
                0.5*p1p*self.f[j-1, 2] - 0.5*cel*cfb
            self.s3[j] = 0.5*(p1p*self.v[j, 2] + cel*cvb)
            self.s4[j] = 0.5*(p1p*self.v[j-1, 2] + cel*cvb)
            self.s5[j] = -p2p*self.u[j, 2]
            self.s6[j] = -p2p*self.u[j-1, 2]
            self.s7[j] = 0.0
            self.s8[j] = 0.0
            self.r[2, j] = crb - (derbv + p1p*fvb - p2p*usb +
                                  cel*(fb*cvb-vb*cfb))

            # coefficients of the differenced energy equation

            self.b1[j] = self.e[j, 2]/self.deta[j-1] + \
                0.5*p1p*self.f[j, 2] - 0.5*cel*cfb
            self.b2[j] = -self.e[j-1, 2]/self.deta[j-1] + \
                0.5*p1p*self.f[j-1, 2] - 0.5*cel*cfb
            self.b3[j] = 0.5*(p1p*self.p[j, 2] + cel*cpb)
            self.b4[j] = 0.5*(p1p*self.p[j-1, 2] + cel*cpb)
            self.b5[j] = -0.5*cel*(self.g[j, 2]-cgb)
            self.b6[j] = -0.5*cel*(self.g[j-1, 2]-cgb)
            self.b7[j] = -0.5*cel*(self.u[j, 2]+cub)
            self.b8[j] = -0.5*cel*(self.u[j-1, 2]+cub)
            self.b9[j] = 0.0
            self.b10[j] = 0.0
            self.r[3, j] = ctb - (derep + p1p*fpb - cel*(ugb-cgb*ub+cub*gb) +
                                  cel*(cpb*fb - cfb*pb))

            # right hand side of matrix
            self.r[1, j] = self.f[j-1, 2] - self.f[j, 2] + self.deta[j-1]*ub
            self.r[4, j-1] = self.u[j-1, 2] - self.u[j, 2] + self.deta[j-1]*vb
            self.r[5, j-1] = self.g[j-1, 2] - self.g[j, 2] + self.deta[j-1]*pb

        # boundary conditions at right hand side of matrix
        self.r[1, 1] = 0.0
        self.r[2, 1] = 0.0
        self.r[3, 1] = 0.0
        self.r[4, self.etamax] = 0.0
        self.r[5, self.etamax] = 0.0

    def solv5(self):

        # tridiagonal matrix elements

        self.a11[1] = 1.0
        self.a12[1] = 0.0
        self.a13[1] = 0.0
        self.a14[1] = 0.0
        self.a15[1] = 0.0
        self.a21[1] = 0.0
        self.a22[1] = 0.0
        self.a23[1] = 1.0
        self.a24[1] = 0.0
        self.a25[1] = 0.0
        self.a31[1] = 0.0
        self.a32[1] = 0.0
        self.a33[1] = 0.0
        self.a34[1] = self.alfa0
        self.a35[1] = self.alfa1

        # elements of the w-vector

        self.w1[1] = self.r[1, 1]
        self.w2[1] = self.r[2, 1]
        self.w3[1] = self.r[3, 1]
        self.w4[1] = self.r[4, 1]
        self.w5[1] = self.r[5, 1]

        # forward sweep

        for j in range(2, self.etamax+1):
            aa1 = self.a[j]*self.a24[j-1] - self.a25[j-1]
            aa2 = self.a[j]*self.a34[j-1] - self.a35[j-1]
            aa3 = self.a[j]*self.a12[j-1] - self.a13[j-1]
            aa4 = self.a[j]*self.a22[j-1] - self.a23[j-1]
            aa5 = self.a[j]*self.a32[j-1] - self.a33[j-1]
            aa6 = self.a[j]*self.a14[j-1] - self.a15[j-1]
            aa7 = self.a[j]*self.s6[j] - self.s2[j]
            aa8 = self.a[j]*self.s8[j]
            aa9 = self.a[j]*self.b6[j] - self.b10[j]
            aa10 = self.a[j]*self.b8[j] - self.b2[j]

            # elements of the tridiagonal matrix

            det = self.a11[j-1] * (aa4*aa2-aa1*aa5) - self.a21[j-1] * \
                (aa3*aa2-aa5*aa6) + self.a31[j-1]*(aa3*aa1-aa4*aa6)
            self.g11[j] = (-(aa4*aa2-aa5*aa1)+self.a[j]**2 *
                            (self.a21[j-1]*aa2-self.a31[j-1]*aa1)) / det
            self.g12[j] = ((aa3*aa2-aa5*aa6)-self.a[j]**2 *
                           (self.a11[j-1]*aa2-self.a31[j-1]*aa6)) / det
            self.g13[j] = (-(aa3*aa1-aa4*aa6)+self.a[j]**2 *
                            (self.a11[j-1]*aa1-self.a21[j-1]*aa6)) / det
            self.g14[j] = self.g11[j]*self.a12[j-1] + \
                self.g12[j]*self.a22[j-1] + self.g13[j]*self.a32[j-1] + \
                self.a[j]
            self.g15[j] = self.g11[j]*self.a14[j-1] + \
                self.g12[j]*self.a24[j-1] + self.g13[j]*self.a34[j-1]
            self.g21[j] = (self.s4[j]*(aa2*aa4-aa1*aa5) +
                           self.a31[j-1]*(aa1*aa7-aa4*aa8) +
                           self.a21[j-1]*(aa5*aa8-aa7*aa2))/det
            self.g22[j] = (self.a11[j-1]*(aa2*aa7-aa5*aa8) +
                           self.a31[j-1]*(aa3*aa8-aa6*aa7) +
                           self.s4[j]*(aa5*aa6-aa2*aa3))/det
            self.g23[j] = (self.a11[j-1]*(aa4*aa8-aa1*aa7) +
                           self.s4[j]*(aa3*aa1-aa4*aa6) +
                           self.a21[j-1]*(aa7*aa6-aa3*aa8))/det
            self.g24[j] = self.g21[j]*self.a12[j-1] + \
                self.g22[j]*self.a22[j-1] + self.g23[j]*self.a32[j-1] - \
                self.s6[j]
            self.g25[j] = self.g21[j]*self.a14[j-1] + \
                self.g22[j]*self.a24[j-1] + self.g23[j]*self.a34[j-1] - \
                self.s8[j]
            self.g31[j] = (self.b4[j]*(aa4*aa2-aa5*aa1) -
                           aa9*(self.a21[j-1]*aa2 - self.a31[j-1]*aa1) +
                           aa10*(self.a21[j-1]*aa5-self.a31[j-1]*aa4)) / det
            self.g32[j] = (-self.b4[j]*(aa3*aa2-aa5*aa6) +
                           aa9*(self.a11[j-1]*aa2 - self.a31[j-1]*aa6) -
                           aa10*(self.a11[j-1]*aa5-self.a31[j-1]*aa3)) / det
            self.g33[j] = (self.b4[j]*(aa3*aa1-aa4*aa6) -
                           aa9*(self.a11[j-1]*aa1 - self.a21[j-1]*aa6) +
                           aa10*(self.a11[j-1]*aa4-self.a21[j-1]*aa3)) / det
            self.g34[j] = self.g31[j]*self.a12[j-1] + \
                self.g32[j]*self.a22[j-1] + self.g33[j]*self.a32[j-1] - \
                self.b6[j]
            self.g35[j] = self.g31[j]*self.a14[j-1] + \
                self.g32[j]*self.a24[j-1] + self.g33[j]*self.a34[j-1] - \
                self.b8[j]

            # elements of the tridiagonal matrix glg.(13.31c)
            self.a11[j] = 1.0
            self.a12[j] = -self.a[j] - self.g14[j]
            self.a13[j] = self.a[j]*self.g14[j]
            self.a14[j] = -self.g15[j]
            self.a15[j] = self.a[j]*self.g15[j]
            self.a21[j] = self.s3[j]
            self.a22[j] = self.s5[j] - self.g24[j]
            self.a23[j] = self.s1[j] + self.a[j]*self.g24[j]
            self.a24[j] = -self.g25[j] + self.s7[j]
            self.a25[j] = self.a[j]*self.g25[j]
            self.a31[j] = self.b3[j]
            self.a32[j] = self.b5[j] - self.g34[j]
            self.a33[j] = self.b9[j] + self.a[j]*self.g34[j]
            self.a34[j] = self.b7[j] - self.g35[j]
            self.a35[j] = self.b1[j] + self.a[j]*self.g35[j]

            # elements of the w-vektor glg.(13.32b)

            self.w1[j] = self.r[1, j] - self.g11[j]*self.w1[j-1] - \
                self.g12[j]*self.w2[j-1] - self.g13[j]*self.w3[j-1] - \
                self.g14[j]*self.w4[j-1]-self.g15[j]*self.w5[j-1]
            self.w2[j] = self.r[2, j] - self.g21[j]*self.w1[j-1] - \
                self.g22[j]*self.w2[j-1] - self.g23[j]*self.w3[j-1] - \
                self.g24[j]*self.w4[j-1]-self.g25[j]*self.w5[j-1]
            self.w3[j] = self.r[3, j] - self.g31[j]*self.w1[j-1] - \
                self.g32[j]*self.w2[j-1] - self.g33[j]*self.w3[j-1] - \
                self.g34[j]*self.w4[j-1]-self.g35[j]*self.w5[j-1]
            self.w4[j] = self.r[4, j]
            self.w5[j] = self.r[5, j]

        # backward sweep

        j = self.etamax

        # definitions

        dp = -(self.a31[j]*(self.a13[j]*self.w2[j] - self.w1[j]*self.a23[j]) -
               self.a32[j]*(self.a11[j]*self.w2[j] - self.w1[j]*self.a21[j]) +
               self.w3[j]*(self.a11[j]*self.a23[j] - self.a13[j]*self.a21[j]))
        dv = -(self.a31[j]*(self.w1[j]*self.a25[j] - self.w2[j]*self.a15[j]) -
               self.w3[j]*(self.a11[j]*self.a25[j] - self.a15[j]*self.a21[j]) +
               self.a35[j]*(self.a11[j]*self.w2[j] - self.w1[j]*self.a21[j]))
        df = -(self.w3[j]*(self.a13[j]*self.a25[j] - self.a23[j]*self.a15[j]) -
               self.a33[j]*(self.w1[j]*self.a25[j] - self.a15[j]*self.w2[j]) +
               self.a35[j]*(self.w1[j]*self.a23[j] - self.a13[j]*self.w2[j]))
        d1 = -(self.a31[j]*(self.a13[j]*self.a25[j] -
               self.a23[j]*self.a15[j]) -
               self.a33[j]*(self.a11[j]*self.a25[j] -
               self.a21[j]*self.a15[j]) +
               self.a35[j]*(self.a11[j]*self.a23[j] -
               self.a21[j]*self.a13[j]))

        # elements of the delta vector for j=np glg.(13.33a)

        self.delp[j] = dp / d1
        self.delv[j] = dv / d1
        self.delf[j] = df / d1
        self.delg[j] = 0.0
        self.delu[j] = 0.0

        for j in range(self.etamax-1, 0, -1):

            # definitions

            bb1 = self.delu[j+1] - self.a[j+1]*self.delv[j+1] - self.w4[j]
            bb2 = self.delg[j+1] - self.a[j+1]*self.delp[j+1] - self.w5[j]
            cc1 = self.w1[j] - self.a12[j]*bb1 - self.a14[j]*bb2
            cc2 = self.w2[j] - self.a22[j]*bb1 - self.a24[j]*bb2
            cc3 = self.w3[j] - self.a32[j]*bb1 - self.a34[j]*bb2
            dd1 = self.a13[j] - self.a12[j]*self.a[j+1]
            dd2 = self.a23[j] - self.a22[j]*self.a[j+1]
            dd3 = self.a33[j] - self.a32[j]*self.a[j+1]
            ee1 = self.a15[j] - self.a14[j]*self.a[j+1]
            ee2 = self.a25[j] - self.a24[j]*self.a[j+1]
            ee3 = self.a35[j] - self.a34[j]*self.a[j+1]
            dett = self.a11[j]*dd2*ee3 + self.a21[j]*dd3*ee1 + \
                self.a31[j]*dd1*ee2 - self.a31[j]*dd2*ee1 - \
                self.a21[j]*dd1*ee3-self.a11[j]*dd3*ee2

            # elements of the delta vector for glg.(13.33b)

            self.delf[j] = (cc1*dd2*ee3 + cc2*dd3*ee1 + cc3*dd1*ee2 -
                            cc3*dd2*ee1 - cc2*dd1*ee3 - cc1*dd3*ee2) / dett
            self.delv[j] = (self.a11[j]*cc2*ee3 + self.a21[j]*cc3*ee1 +
                            self.a31[j]*cc1*ee2 - self.a31[j]*cc2*ee1 -
                            self.a21[j]*cc1*ee3-self.a11[j]*cc3*ee2)/dett
            self.delp[j] = (self.a11[j]*cc3*dd2 + self.a21[j]*cc1*dd3 +
                            self.a31[j]*cc2*dd1 - self.a31[j]*cc1*dd2 -
                            self.a21[j]*cc3*dd1-self.a11[j]*cc2*dd3)/dett
            self.delu[j] = bb1 - self.a[j+1]*self.delv[j]
            self.delg[j] = bb2 - self.a[j+1]*self.delp[j]

        # new values of f, u, v, g, p

        for j in range(1, self.etamax+1):
            self.f[j, 2] = self.f[j, 2] + self.delf[j]
            self.u[j, 2] = self.u[j, 2] + self.delu[j]
            self.v[j, 2] = self.v[j, 2] + self.delv[j]
            self.g[j, 2] = self.g[j, 2] + self.delg[j]
            self.p[j, 2] = self.p[j, 2] + self.delp[j]

        # boundary condition

        self.u[self.etamax, 2] = 0.0

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
            self.f[j, 1] = self.f[j, 2]
            self.u[j, 1] = self.u[j, 2]
            self.v[j, 1] = self.v[j, 2]
            self.g[j, 1] = self.g[j, 2]
            self.p[j, 1] = self.p[j, 2]
            self.e[j, 1] = self.e[j, 2]
            self.b[j, 1] = self.b[j, 2]

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

        # print thermo-physical properties
        self.print_Properties()

        # initial velocity profile
        self.print_stage_header()
        self.ivpl()
        self.plot()
        self.print_results()
        self.shift_profiles()

        for self.nx in range(2, self.gsimax):
            self.print_stage_header()

            for iteration in range(1, self.max_iterations+1):

                self.turbulence()
                self.coefficients()
                self.solv5()
                converged = self.convergence_check(iteration)
                if converged:
                    break

            self.print_results()
            self.plot()
            self.shift_profiles()


if __name__ == "__main__":

    jet = Jet()

    # make mesh
    jet.mesh(gsimax=1000, dgsi=0.01, etae=8, deta1=0.01, stretch=1.15)

    # define properties
    jet.set_Properties(5300.0, 0.72, 0.5)

    # run main program
    jet.main()
