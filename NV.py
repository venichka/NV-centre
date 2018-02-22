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
delt = 1.
omR = 1.

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
        self.phi = 0

    def phiInit(self, n_phot):
        self.phi = 1 / 2. * np.arctan(2 * omR * np.sqrt(n_phot + 1) / delt)
        return self.phi

    def modestate(self, n_phot):
        self.nph_vec = np.zeros([N_mode, 1])  # - Mode n state
        self.nph_vec[N_mode - 1 - n_phot, 0] = 1.
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
            phi = self.phiInit(n_phot)
            alpha = np.cos(phi)
            beta = np.sin(phi)
            self.nplus_vec = alpha * np.kron(e_vec, nph_vec1) + \
                        beta * np.kron(g_vec, nph_vec2)
            self.nminus_vec = - beta * np.kron(e_vec, nph_vec1) + \
                        alpha * np.kron(g_vec, nph_vec2)
            return self.nplus_vec, self.nminus_vec

    def operators_in_eigenbasis(self, mode_coupling, TLS_coupling):
        for i in range(1, N_mode - 1):
            n_phot = i
            phi = self.phiInit(n_phot)
            phi1 = self.phiInit(n_phot + 1)
            a, b = self.eigenstates(n_phot)
            a1, b1 = self.eigenstates(n_phot + 1)
            self.s1 += mode_coupling * \
                       ((np.sqrt(n_phot + 1) * np.cos(phi) * \
                         np.cos(phi1) + np.sqrt(n_phot + 2) * \
                         np.sin(phi) * np.sin(phi1)) * \
                         np.dot(a, np.transpose(a1)) + \
                            (- np.sqrt(n_phot + 1) * np.cos(phi) * \
                            np.sin(phi1) + np.sqrt(n_phot + 2) * \
                            np.sin(phi) * np.cos(phi1)) * \
                            np.dot(a, np.transpose(b1)) + \
                        (- np.sqrt(n_phot + 1) * np.sin(phi) * \
                         np.cos(phi1) + np.sqrt(n_phot + 2) * \
                         np.cos(phi) * np.sin(phi1)) * \
                         np.dot(b, np.transpose(a1)) + \
                            (np.sqrt(n_phot + 2) * np.cos(phi) * \
                            np.cos(phi1) + np.sqrt(n_phot + 1) * \
                            np.sin(phi) * np.sin(phi1)) * \
                            np.dot(b, np.transpose(b1))) + \
                  TLS_coupling * (np.cos(phi) * np.sin(phi) * \
                                  np.dot(a, np.transpose(a)) - \
                                  np.sin(phi)**2 * \
                                  np.dot(a, np.transpose(b)) + \
                                  np.cos(phi)**2 * \
                                  np.dot(b, np.transpose(a)) - \
                                  np.cos(phi) * np.sin(phi) * \
                                  np.dot(b, np.transpose(b)))
            self.s2 += (np.cos(phi)**2 - np.sin(phi)**2) * \
                   np.dot(a, np.transpose(a)) - \
                   2 * np.cos(phi) * np.sin(phi) * np.dot(a, np.transpose(b)) \
                   + (np.sin(phi)**2 - np.cos(phi)**2) * \
                   np.dot(b, np.transpose(b)) - \
                   2 * np.cos(phi) * np.sin(phi) * np.dot(b, np.transpose(a))
        return self.s1, self.s2


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


#--------------------------------- Program
#-------------------------------- Asymmetric
operators = oper() #- Collective operators
Jm = operators[3]
Jp = operators[4]
Jz = operators[5]
Jj = Jm + Jp
CosPhi = operators[6]

"""Dicke states"""
H_dicke = ham0()
#--- Eigensystem: Dicke states
wD, vD = np.linalg.eigh(H_dicke) #- For hermite matrix
wDfl = np.flip(wD, 0)        #- Flipped array of eigenvalues
vDfl = np.flip(vD, 1)        #- Flipped array of eigenvectors
vDD = np.zeros((N + 1, 2**N))
index = 0
vDD[index] = vDfl[:, 0]
for i in range(1, len(wDfl)):
    if wDfl[i-1] != wDfl[i]:
        index += 1
    vDD[index] += np.sqrt(1. / sc.special.factorial(N) * \
                          sc.special.factorial(N - index) * \
                          sc.special.factorial(index)) * vDfl[:, i]

omR = OmegaRabi_geom(l=0.01,d_abs=0.00)
Ham = hamiltonian(omR)
"""Eigensystem"""
if np.abs(np.max(omR)) == 0:
    indicator = 1
    lab = 'Dicke'
    w, v = np.linalg.eigh(Ham) #- For hermite matrix
    w = np.flip(w, 0)        #- Flipped array of eigenvalues
    wfl = np.zeros(N + 1)
    index = 0
    wfl[index] = w[0]
    for i in range(1, len(w)):
        if w[i-1] != w[i]:
            index += 1
        wfl[index] = w[i]
    vfl = np.transpose(vDD)    #- Flipped array of eigenvectors
else:
    indicator = 0
    lab = 'Interaction'
    w, v = np.linalg.eigh(Ham) #- For hermite matrix
    wfl = np.flip(w, 0)        #- Flipped array of eigenvalues
    vfl = np.flip(v, 1)        #- Flipped array of eigenvectors

#--- Calculating gammaij
numState = np.argwhere(wfl == N/2)[0, 0] #- number of the fully inverted state
gamij = np.triu(np.transpose(np.abs(np.dot(np.transpose(vfl), \
                                   np.dot(Jm, vfl)))**2), k=1)

"""Variables for solution"""
pdiagT = np.zeros([len(vfl[0]), nT]) #- Probabilities
DHamAv = np.zeros(nStep) #- Derivative of hamiltonian
tranM1 = np.dot(np.transpose(vfl), np.dot(Jj, vfl)) #Transition matrix
JAv = np.zeros(nStep) #- dipole moment
DisCos = np.zeros(nStep) #- CosPhi dispersion
Cos2_inBasis = np.dot(np.transpose(vfl), \
                        np.dot(np.dot((CosPhi[0] - CosPhi[N - 1]), \
                                      (CosPhi[0] - CosPhi[N - 1])), vfl))
Cos_inBasis = np.dot(np.transpose(vfl), \
                        np.dot((CosPhi[0] - CosPhi[N - 1]), vfl))

"""Solve Lindblad equation"""
numP = np.array([], 'int')
#pdiag = incon(numState) #- initial conditions
pdiag = incon(0)

for i in range(nStep):
    print(i, lab)
    pdiag = scheme(pdiag, gamij, alpha=0.5, beta=0.5)
    DHamAv[i] = - np.dot(detpart(pdiag, gamij), wfl)
    JAv[i] = np.trace(np.dot(np.diag(pdiag), tranM1))
    DisCos[i] = np.trace(np.dot(np.diag(pdiag), Cos2_inBasis)) - \
                np.trace(np.dot(np.diag(pdiag), Cos_inBasis))**2
    #--- Probabilities in every 20 step
    if np.mod(i, dsT) == 0:
        pdiagT[:, np.int(i/dsT)] = pdiag

"""Finding the sorting massive, numP"""
for i in range(len(pdiag)):
    if np.max(pdiagT[i]) > 0.02:
        numP = np.append(numP, i)
states = vfl[:, numP] #- States which take place in system's evolution

"""Transition matrices"""
"""Real matrix"""
tranM1 = np.dot(np.transpose(vfl), np.dot(Jj, vfl))
tranM = np.array([[tranM1[i, j] if tranM1[i, j] > 0.000001 else \
                   0 for i in range(len(vfl[0]))] \
                  for j in range(len(vfl[0]))])
for i in range(len(numP)): # mark evolution states
    tranM[numP[i], numP[i]] = 5.
tranMm = np.matrix(tranM)
Sp_tranM = spr.csr_matrix(tranMm) #- Sparse row format (CSR)
# Reverse-Cuthill McKee ordering
perm = spr.csgraph.reverse_cuthill_mckee(Sp_tranM, symmetric_mode=False)
ro_tranM = Sp_tranM[np.ix_(np.flip(perm, 0),np.flip(perm, 0))].A

"""Dicke"""
gamij_D = np.triu(np.transpose(np.abs(np.dot(np.transpose(vDfl), \
                                   np.dot(Jm, vDfl)))**2), k=1)
#--- Transition matrix
tranM1_D = np.dot(np.transpose(vDfl), np.dot(Jj, vDfl))
tranM_D = np.array([[tranM1_D[i, j] if np.abs(tranM1_D[i, j]) > 0.000001 else \
                   0 for i in range(2**N)] \
                  for j in range(2**N)])
tranMm_D = np.matrix(tranM_D)
Sp_tranM_D = spr.csr_matrix(tranMm_D) #- Sparse row format (CSR)
# Reverse-Cuthill McKee ordering
perm_D = spr.csgraph.reverse_cuthill_mckee(Sp_tranM_D, symmetric_mode=False)
ro_tranM_D = Sp_tranM_D[np.ix_(np.flip(perm_D, 0),np.flip(perm_D, 0))].A

#--- Interaction eigensystem
V_oper = hamInteraction(omR=omR)
wV, vV = np.linalg.eigh(V_oper) #- For hermite matrix
wVfl = np.flip(wV, 0)        #- Flipped array of eigenvalues
vVfl = np.flip(vV, 1)        #- Flipped array of eigenvectors

#--- Scalar product of states
#overlap = np.abs(np.dot(vDD, states))
overlap2 = np.abs(np.dot(vDD, vfl))
overlapV = np.abs(np.dot(vDD, vVfl))
cond = overlapV > 0.6
ovl_part = 'hyu' if len(np.extract(cond, overlapV)) >= N else 0.

print(numP)
print(len(np.extract(cond, overlapV)))
print(np.max(np.abs(omR)))
print('max or not:', 1 if np.argmax(DHamAv) > 0 else 0)



#--- Plotting

#- Plot3D of probabilities + DHamAv
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
X = np.zeros([len(pdiagT), nT])
Y = np.zeros([len(pdiagT), nT])
# Get data
for i in range(len(pdiagT)):
    for j in range(nT):
        X[i, j] = wfl[i]
        Y[i, j] = j*dt*dsT
# Give the first plot only wireframes of the type y = c
ax1.plot_wireframe(X, Y, pdiagT, rstride=0, cstride=1, cmap=cm.coolwarm,
                       linewidth=1)
#ax1.plot_surface(X, Y, pdiagT, rstride=1, cstride=1, cmap=cm.coolwarm,
#                       linewidth=0)
ax1.set_title("Probabilities(t, state), N = %s" % N)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(DHamAv, 'r')
ax2.set_title("Derivative of H, N = %s" % N)

#Overlap
colors = ['r', 'g', 'c', 'r', 'g', 'c', 'r', 'g', 'c', 'r', 'g', 'c']
figbb = plt.figure(figsize=(16, 9))
axbb1 = figbb.add_subplot(121, projection='3d')
yticks1 = np.arange(len(overlap2))
xx1 = np.arange(len(overlap2[1]))
for c, k in zip(colors, yticks1):
    cs = [c] * (len(xx1) + 1)
    axbb1.bar(xx1, overlap2[k], zs=k, zdir='y', color=cs,\
            alpha=0.7, width=10)
axbb1.set_xlabel('States of Hamiltonian')
axbb1.set_ylabel('Dicke states')
axbb1.set_zlabel('Overlap')
axbb1.set_yticks(yticks1)
axbb1.set_title("Full Hamiltonian overlap, N = %s" % N)
axbb2 = figbb.add_subplot(122, projection='3d')
yticks2 = np.arange(len(overlapV))
xx2 = np.arange(len(overlapV[1]))
for c, k in zip(colors, yticks2):
    cs = [c] * (len(xx2) + 1)
    axbb2.bar(xx2, overlapV[k], zs=k, zdir='y', color=cs,\
            alpha=0.6, width=10)
axbb2.set_xlabel('States of interaction operator')
axbb2.set_ylabel('Dicke states')
axbb2.set_zlabel('Overlap')
axbb2.set_yticks(yticks2)
axbb2.set_title("Interaction overlap, N = %s" % N)

figl = plt.figure(figsize=(16, 8))
axl1 = figl.add_subplot(1, 3, 1)
axl1.plot(DHamAv, 'r')
axl1.set_title("Derivative of H, N = %s" % N)
axl2 = figl.add_subplot(1, 3, 2)
axl2.plot(DisCos, 'b')
axl2.set_title("(Cos(phi[0]) - Cos[phi[N - 1]])^2, N = %s" % N)
axl3 = figl.add_subplot(1, 3, 3)
axl3.plot(JAv, 'orange')
axl3.set_title("Dipole moment, N = %s" % N)
plt.show()
