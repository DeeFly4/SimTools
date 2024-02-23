from classes import Explicit_Problem_2nd, Explicit_ODE_2nd, Newmark
import numpy as np
import matplotlib.pyplot as mpl

def f(t):
	return np.zeros((2,))

M = np.array([1, 2])
C = np.array([0, 0])
K = np.array([1, 1])

u0 = np.array([1, 1])
up0 = np.array([0, 0])

testeq = Explicit_Problem_2nd(M, C, K, f, u0, up0, 0)

sim = Newmark(testeq)
t, u = sim.simulate(1)

mpl.plot(t, u)
mpl.show()