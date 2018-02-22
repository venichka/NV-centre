N_mode = 4
ass = Operators_a()
a = ass.phiInit(2)
b = ass.modestate(0)
d = ass.eigenstates(0)
c = ass.operators_in_eigenbasis(1., 1.)

print(a, b, c, d)


gammaij = np.abs(c)**2


x = incon(3, 1)

y = detpart(x, gammaij)

x = np.ones([2, 3, 3])
x[1, 2, 1] = 3
x[0, 2, 1] = 2

x[0]
x[1]

g = np.zeros([1,2,2])
g[0,0,0] = 2
g[0,1,0] = 1
np.dot(g, x)

np.shape(sepmass)

g = np.arange(6*6).reshape(6, 6)
g
np.dot(np.transpose(incon(2, 0)), g)

s1 = ass.operators_in_eigenbasis(1., 1.)[0]
ass.modestate(0)
ass.eigenstates(0)[0]
ass.eigenstates(2)[1]

np.dot(s1, ass.eigenstates(2)[0])
np.kron(np.eye(2), sigz)
np.dot(np.kron(np.eye(2), sigz), np.kron(np.array([1, 0]), np.array([0, 1])))

a_oper = np.eye(N_mode, k=1)
a_oper = np.array([np.sqrt(i + 1 ) * a_oper[i] for i in range(N_mode)])
a_oper[1]**2

sss = Operators_c()
h_planck = 1. 
h = sss.hamiltonian()

h - np.transpose(h)

wD, vD = np.linalg.eigh(h)