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
N_mode = 100
h_planck = 1.
T = 0.4  #temperature of radiation resevoir (S1)
         #[0.4 coincides with 300K if h_plank = 1.]
T_ph = 0.2  #temperature of dephasing reservoir (S2)
T_pump = - 23.96  #negative temperature of pumping (S3)
omR = 0.05
omM = 30.
omNV = 28.
delt = omM - omNV
g_s1 = np.array([1., 45.])  #- gammas for S1 (g_s1[0] * a + g_s1[1] * sigma)
g0 = np.array([1., 0.9, 0.01])  #- gammas for S1, S2, S3

#-Equation parameters
dt = 0.0001
tmax = 5.
nT = 100  #- number of time steps in output
nStep = np.int(tmax/dt)
dsT = np.int(nStep/nT)  #- time discretization in output

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
        self.nground_vec = np.zeros([2 * N_mode, 1])
        self.nplus_vec = np.zeros([2 * N_mode, 1])
        self.nminus_vec = np.zeros([2 * N_mode, 1])
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
        self.nph_vec[N_mode -1 - n_phot, 0] = 1.
        return self.nph_vec

    #--- collective operators for N atoms
    def eigenstates(self, n_phot):
        """Eigenstates in 2 * N_mode space:
                    n_phot -> number of photons in the mode
            (0 - |n, +> vector
             1 - |n, -> vector)"""
        if n_phot == -1:
            self.nground_vec = np.kron(np.array([[0], [1]]), self.modestate(0))
            return self.nground_vec, np.zeros([2 * N_mode, 1])
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
    Class which projects the system's operators numerically.
    Methods:
        mode_opers - returns a and a^+ operators
                0 - a
                1 - a^+
        TLS_opers - returns sigma, sigma^+ and sigma_z
                0 - sigma
                1 - sigma^+
                2 - sigma_z
        hamiltonian - returns Hamilton operator of the system

        eigensystem - returns eigenvalues and eigenvectors of Hamiltonian
                0 - eigenvalues (sorted from excited to ground)
                1 - eigenvectors (from excited to ground)
        opers_inbasis - returns system operators S1 and S2 in basis of
                        system's eigenstates
                0 - S1 = <k'|x*a + y*sigma|k">|k'><k"| - Radiation operator
                1 - S2 = <k'|sigma_z|k">|k'><k"| - Dephasing operator
                2 - S3 = <k'|sigma|k">|k'><k"| - Pumping operator
            input -> mode_coupling (x in def)
                     NV_coupling (y in def)
    """
    def __init__(self):
        self.a_oper = np.zeros(2 * N_mode)
        self.ac_oper = np.zeros(2 * N_mode)
        self.sigp = np.zeros(2 * N_mode)
        self.sigm = np.zeros(2 * N_mode)
        self.sigz = np.zeros(2 * N_mode)
        self.ham = np.zeros(2 * N_mode)
        self.w = 0
        self.v = 0
        self.s1 = np.zeros(2 * N_mode)
        self.s2 = np.zeros(2 * N_mode)
        self.s3 = np.zeros(2 * N_mode)

    def mode_opers(self):
        self.a_oper = np.eye(N_mode, k=-1)
        self.a_oper = np.array([np.sqrt(N_mode - i) * self.a_oper[i] \
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
        self.ham = h_planck * omM * (np.dot(ac_oper, a_oper)) + \
                   h_planck * omNV * 0.5 * sigz + \
                   h_planck * omR * (np.dot(ac_oper, sigm) + \
                                     np.dot(sigp, a_oper))
        return self.ham

    def eigensystem(self):
        ham = self.hamiltonian()
        self.w, self.v = np.linalg.eigh(ham)
        self.w = np.flip(self.w, 0)
        self.v = np.flip(self.v, 1)
        return self.w, self.v

    def opers_inbasis(self, M_coupling, NV_coupling):
        self.s1 = M_coupling * self.mode_opers()[0] + \
             NV_coupling * self.TLS_opers()[0]
        self.s2 = self.TLS_opers()[2]
        self.s3 = self.TLS_opers()[0]
        v = self.eigensystem()[1]
        self.s1 = np.dot(np.transpose(v), np.dot(self.s1, v))
        self.s2 = np.dot(np.transpose(v), np.dot(self.s2, v))
        self.s3 = np.dot(np.transpose(v), np.dot(self.s3, v))
        return self.s1, self.s2, self.s3



'#--- Initial conditions'


def incon_a(n_phot, ind):
    """
    Initial conditions - vector x
        from analytics
    input:
            n_phot - number of photons in the mode
            ind - 0 -> |n_phot, +>
                - 1 -> |n_phot, ->
    """
    constructor = Operators_a()
    y = constructor.eigenstates(n_phot)
    x = np.diag(np.dot(y[ind], np.transpose(y[ind])))
    return x

def incon_c(numb):
    """
    Initial conditions - vector x
        from numerical computations
    input:
            numb - number of state
    """
    constructor = Operators_c()
    y = constructor.eigensystem()[1]
    x = np.diag(np.dot(np.transpose(y[:, numb].reshape(1, 2 * N_mode)), \
                       y[:, numb].reshape(1, 2 * N_mode)))
    return x

def incon_ex(n_phot, ex):
    """
    Initial conditions - matrix x
        !!Use only with matrix equation with nondiagonal elements!!
    input:
            n_phot - number of photons
            ex - NV state (1 = |e>, 0 = |g>)
    """
    constructor = Operators_a()
    y = constructor.modestate(n_phot)
    if ex == 1:
        z = np.array([[1], [0]])
    else:
        z = np.array([[0], [1]])
    x = np.dot(np.kron(z, y), np.transpose(np.kron(z, y)))
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
    sepmass = np.ones(2 * N_mode)
    y = np.dot(- gamij[0], sepmass)*x + \
        np.dot(x, gamij[0]) + \
        np.dot(- gamij[1], sepmass)*x + \
        np.dot(x, gamij[1]) + \
        np.dot(- gamij[2], sepmass)*x + \
        np.dot(x, gamij[2])
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


'''Program'''

comp_cl = Operators_c()  #- initiation of the class
ham = comp_cl.hamiltonian()
w_values, v_vectors = comp_cl.eigensystem()
s1_oper, s2_oper, s3_oper = comp_cl.opers_inbasis(M_coupling=g_s1[0], \
                                                  NV_coupling=g_s1[1])

'''Transition matrix'''

gamij = np.zeros([3, 2*N_mode, 2*N_mode])
gamij[0] = g0[0] * np.abs(np.transpose(s1_oper))**2
gamij[1] = g0[1] * np.abs(np.transpose(s2_oper))**2
gamij[2] = g0[2] * np.abs(np.transpose(s3_oper))**2

#- Gibbs factor for S1 operator
gam_up1 = np.array([[gamij[0, i, j] * \
                                 np.exp(- (w_values[i] - w_values[j]) / T) \
                                 if i < j else \
                                 0 for i in range(len(w_values))] \
                                for j in range(len(w_values))])
gamij[0] = gamij[0] + gam_up1
#- Gibbs factor for S2 operator
gam_up2 = np.array([[gamij[1, i, j] * \
                                 np.exp(- (w_values[i] - w_values[j]) / T_ph) \
                                 if i < j else \
                                 0 for i in range(len(w_values))] \
                                for j in range(len(w_values))])
gamij[1] = gamij[1] + gam_up2
#- Gibbs factor for S3 operator
gam_up3 = np.array([[gamij[2, i, j] * \
                               np.exp(- (w_values[i] - w_values[j]) / T_pump) \
                                 if i < j else \
                                 0 for i in range(len(w_values))] \
                                for j in range(len(w_values))])
gamij[2] = gamij[2] + gam_up3


"""Variables for solution"""

pdiagT = np.zeros([len(v_vectors[:, 0]), nT])  #- Probabilities
DHamAv = np.zeros(nStep)  #- Derivative of hamiltonian


"""Solve Lindblad equation"""


#pdiag = incon(numState) #- initial conditions
pdiag = incon_c(3)

for i in range(nStep):
    print(i)
    pdiag = scheme(pdiag, gamij, alpha=0.5, beta=0.5)
    DHamAv[i] = - np.dot(detpart(pdiag, gamij), w_values)
    #--- Probabilities in every 20 step
    if np.mod(i, dsT) == 0:
        pdiagT[:, np.int(i/dsT)] = pdiag


'''Plotting'''

#- Plot3D of probabilities + DHamAv
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
X = np.zeros([len(pdiagT), nT])
Y = np.zeros([len(pdiagT), nT])
# Get data
for i in range(len(pdiagT)):
    for j in range(nT):
        X[i, j] = w_values[i]
        Y[i, j] = j*dt*dsT
# Give the first plot only wireframes of the type y = c
ax1.plot_wireframe(X, Y, pdiagT, rstride=0, cstride=1, linewidth=1)
#ax1.plot_surface(X, Y, pdiagT, rstride=1, cstride=1, cmap=cm.coolwarm,
#                       linewidth=0)
ax1.set_title("Probabilities(t, state), N_mode = %s" % N_mode)
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(DHamAv, 'r')
ax2.set_title("Derivative of H, N_mode = %s" % N_mode)

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(w_values, pdiagT[:, nT - 1], 'black')
ax3.plot(w_values, np.max(pdiagT[:, nT - 1]) /\
         np.max(np.exp(- w_values / T)) * \
         np.exp(- w_values / T), 'r')
ax3.set_title("Probabilities, t = tmax, N_mode = %s" % N_mode)


plt.show()
