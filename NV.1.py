import numpy as np
import scipy as sc
import scipy.sparse.csgraph as dop1
import scipy.special as dop2
import matplotlib.pyplot as plt
import scipy.sparse as spr
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy.sparse.linalg import bicgstab
from sympy.utilities.iterables import flatten

#--- Parameters
N_mode = 3
h_planck = 1.
omR = 0.01
omM = 0.9
omNV = 1.
delt = omM - omNV

#-Equation parameters
gam = 1.
dt = 0.01
tmax = 2.
nT = 100 #- number of time steps in output
nStep = np.int(tmax/dt)
dsT = np.int(nStep/nT) #- time discretization in output

#--- operators for 1 atom
iden = np.identity(2)
sigm = np.eye(2, k=-1)
sigp = np.eye(2, k=1)
sigz = np.array([[1, 0], [0, -1]])

class Operators_a:
    """
    Class which projects the system's operators from analytics.
    Methods:
        phiInit - returns the constant phi
                input -> n_phot (number of photons in the mode)
        modestate - returns the mode eigenvectors
                input -> n_phot (number of photons in the mode)
        eigenstates - (0 - |n, +> vector
                       1 - |n, -> vector)
                input -> n_phot (number of photons in the mode)
        operators_in_eigenbasis - returns system operators
                      (0 - S1 = <k'|x*a + y*sigma|k">|k'><k"|
                       1 - S2 = <k'|sigma_z|k">|k'><k"|)
                input -> mode_coupling (x in def)
                         TLS_coupling (y in def)
    """
    def __init__(self):
        self.nph_vec = np.zeros([N_mode, 1])
        self.nplus_vec = np.zeros([N_mode, 1])
        self.nminus_vec = np.zeros([N_mode, 1])
        self.s1 = np.zeros([2*N_mode, 2*N_mode])
        self.s2 = np.zeros([2*N_mode, 2*N_mode])
        self.cosphi = 0
        self.sinphi = 0

    def phiInit(self, n_phot):
        self.cosphi = np.sqrt((np.sqrt(omR**2 * (n_phot + 1) +\
                              (delt / 2.)**2) + delt / 2.) / 2. / \
                              np.sqrt(omR**2 * (n_phot + 1) + (delt / 2.)**2))
        self.sinphi = np.sqrt((np.sqrt(omR**2 * (n_phot + 1) +\
                              (delt / 2.)**2) - delt / 2.) / 2. / \
                              np.sqrt(omR**2 * (n_phot + 1) + (delt / 2.)**2))
        return self.cosphi, self.sinphi

    def modestate(self, n_phot):
        self.nph_vec = np.zeros([N_mode, 1])  # - Mode n state
        self.nph_vec[n_phot, 0] = 1.
        return self.nph_vec

    #--- collective operators for N atoms
    def eigenstates(self, n_phot):
        """Eigenstates in 2 * N_mode space:
                    n_phot -> number of photons in the mode
            (0 - |n, +> vector
             1 - |n, -> vector)"""
        if n_phot > N_mode - 1:
            return 'n_phot is bigger than dimension of the space'
        else:
            e_vec = np.array([[1], [0]])  # - TLS states
            g_vec = np.array([[0], [1]])
            nph_vec1 = self.modestate(n_phot)  # - Mode n state
            nph_vec2 = self.modestate(n_phot + 1)
            alpha, beta = self.phiInit(n_phot)
            self.nplus_vec = alpha * np.kron(e_vec, nph_vec1) + \
                        beta * np.kron(g_vec, nph_vec2)
            self.nminus_vec = - beta * np.kron(e_vec, nph_vec1) + \
                        alpha * np.kron(g_vec, nph_vec2)
            return self.nplus_vec, self.nminus_vec

    def operators_in_eigenbasis(self, mode_coupling, TLS_coupling):
        for i in range(1, N_mode - 2):
            n_phot = i
            alpha, beta = self.phiInit(n_phot)
            alpha1, beta1 = self.phiInit(n_phot + 1)
            a, b = self.eigenstates(n_phot)
            a1, b1 = self.eigenstates(n_phot + 1)
            self.s1 += mode_coupling * \
                       ((np.sqrt(n_phot + 1) * alpha * \
                         alpha1 + np.sqrt(n_phot + 2) * \
                         beta * beta1) * \
                         np.dot(a, np.transpose(a1)) + \
                            (- np.sqrt(n_phot + 1) * alpha * \
                            beta1 + np.sqrt(n_phot + 2) * \
                            beta * alpha1) * \
                            np.dot(a, np.transpose(b1)) + \
                        (- np.sqrt(n_phot + 1) * beta * \
                         alpha1 + np.sqrt(n_phot + 2) * \
                         alpha * beta1) * \
                         np.dot(b, np.transpose(a1)) + \
                            (np.sqrt(n_phot + 2) * alpha * \
                            alpha1 + np.sqrt(n_phot + 1) * \
                            beta * beta1) * \
                            np.dot(b, np.transpose(b1))) + \
                  TLS_coupling * (alpha * beta * \
                                  np.dot(a, np.transpose(a)) - \
                                  beta**2 * \
                                  np.dot(a, np.transpose(b)) + \
                                  alpha**2 * \
                                  np.dot(b, np.transpose(a)) - \
                                  alpha * beta * \
                                  np.dot(b, np.transpose(b)))
            self.s2 += (alpha**2 - beta**2) * np.dot(a, np.transpose(a)) - \
                        2 * alpha * beta * np.dot(a, np.transpose(b)) + \
                        (beta**2 - alpha**2) * np.dot(b, np.transpose(b)) - \
                        2 * alpha * beta * np.dot(b, np.transpose(a))
        return self.s1, self.s2


class Operators_c:
    """

    """
    def __init__(self):
        self.a_oper = np.zeros(2 * N_mode)
        self.ac_oper = np.zeros(2 * N_mode)
        self.sigp = np.zeros(2 * N_mode)
        self.sigm = np.zeros(2 * N_mode)
        self.sigz = np.zeros(2 * N_mode)
        self.ham = np.zeros(2 * N_mode)

    def mode_opers(self):
        self.a_oper = np.eye(N_mode, k=1)
        self.a_oper = np.array([np.sqrt(i + 1) * self.a_oper[i] \
                                for i in range(N_mode)])
        self.a_oper = np.kron(np.eye(2), self.a_oper)
        self.ac_oper = np.conjugate(np.transpose(self.a_oper))
        return self.a_oper, self.ac_oper

    def TLS_opers(self):
        sigm = np.eye(2, k=-1)
        sigp = np.eye(2, k=1)
        sigz = np.array([[1, 0], [0, -1]])
        self.sigm = np.kron(sigm, np.eye(N_mode))
        self.sigp = np.kron(sigp, np.eye(N_mode))
        self.sigz = np.kron(sigz, np.eye(N_mode))
        return self.sigm, self.sigp, self.sigz

    def hamiltonian(self):
        a_oper = self.mode_opers()[0]
        ac_oper = self.mode_opers()[1]
        sigm = self.TLS_opers()[0]
        sigp = self.TLS_opers()[1]
        sigz = self.TLS_opers()[2]
        self.ham = h_planck * omM * np.dot(ac_oper, a_oper) + \
                   h_planck * omNV * 0.5 * sigz + \
                   h_planck * omR * (np.dot(ac_oper, sigm) + \
                                     np.dot(sigp, a_oper))
        return self.ham
    



'#--- Initial conditions'


def incon(n_phot, ind):
    """
    Initial conditions vector x
    input:
            n_phot - number of photons in the mode
            ind - 0 -> |n_phot, +>
                - 1 -> |n_phot, ->
    """
    constructor = Operators_a()
    y = constructor.eigenstates(n_phot)
    x = np.transpose(y[ind])
    return x

#--- Deterministic part
def detpart(x, gamij):
    """
    Equation of dynamics for
    diagonal elements of density matrix
    p[i]'[t] = sum((-sum(g0[k]*gamij[k, i, j], j))*p[i][t] +
    sum(g0[k]*gamij[j, i]*p[j][t], j), k)
    k - number of the operator (0 or 1)
    input: x - initial vector
           gamij - |Sk|**2
           g0[k] - coupling constants for S1 and S2
    """
    sepmass = np.ones(2*N_mode)
    y = np.dot(- gamij[0], sepmass)*x + \
        np.dot(x, gamij[0]) + \
        np.dot(- gamij[1], sepmass)*x + \
        np.dot(x, gamij[1])
    return y


'#--- Scheme'


def scheme(x, gamij, alpha, beta):
    """
    Matsuno Scheme
    input:
        x - initial vector of variables
        alpha, beta - parameters of the scheme
    """
    e1 = detpart(x, gamij)
    yp = x + alpha*e1*dt                   #- predictor
    e2 = detpart(yp, gamij)
    y = x + beta*e2*dt + (1 - beta)*e1*dt  #- corrector
    return y