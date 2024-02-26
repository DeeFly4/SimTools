from classes import Explicit_Problem_2nd
from assimulo.solvers import CVode
import numpy as np
import matplotlib.pyplot as mpl

def f(t):
	return 0

M = np.array([1, 1])
C = np.array([3, 4])
K = np.array([2, 4])

u0 = np.ones(2,)
up0 = np.ones(2,)

t0 = 0
tf = 5

model = Explicit_Problem_2nd(M, C, K, f, u0, up0, t0)
sim = CVode(model)

t, u_all = sim.simulate(tf)
u = [states[0:len(states)//2] for states in u_all]

mpl.plot(t, u)

mpl.show()