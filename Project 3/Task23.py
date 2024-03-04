from classes import Explicit_Problem_2nd, Explicit_ODE_2nd, Newmark, HHT_alpha
from assimulo.solvers import CVode, RungeKutta4
import numpy as np
import matplotlib.pyplot as mpl

omega = np.pi / .3

def f1(t):
    return np.zeros(2,)

def f2(t):
	return np.array([0, 0, 50*np.sin(omega*t)]) if t < np.pi / omega else np.zeros(3,)

M1 = np.diag([1, 1])
K1 = np.array([[1, -2], [3, -4]])
C1 = np.array([[-2, 0], [-1, 0]])

M2 = np.diag([10, 20, 30])
K2 = 1e3 * np.array([[45, -20, -15], [-20, 45, -25], [-15, -25, 40]])
C2 = 3e-2 * K2

u0 = np.ones(2,)
up0 = np.zeros(2,)

t0 = 0
tf = 1

model = Explicit_Problem_2nd(M1, C1, K1, f1, u0, up0, t0)
sim = Newmark(model)
sim.h = 1e-3

t, u_all = sim.simulate(tf)
u = [states[0:len(states)//2] for states in u_all]

mpl.plot(t, u_all, lw=2)
mpl.hlines(0, 0, tf, ls='--', colors='k')
# mpl.axis([0, tf, -5, 5])

mpl.title('Second-order equation system using Newmark', fontsize=14)
mpl.xlabel('Time [s]', fontsize=14)

mpl.grid(True)

mpl.show()