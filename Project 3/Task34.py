from classes import Explicit_Problem_2nd, Explicit_ODE_2nd, Newmark, HHT_alpha
import numpy as np
import matplotlib.pyplot as mpl

def f(t):
	return np.zeros((2,))

M = np.diag([1, 1])
C = np.diag([0, 0])
K = np.diag([2, 4])

u0 = np.ones(2,)
up0 = np.ones(2,)

t0 = 0
tf = 5

testeq = Explicit_Problem_2nd(M, C, K, f, u0, up0, t0)

sim = Newmark(testeq)
t, u = sim.simulate(tf)

mpl.plot(t, u)
mpl.show()