from assimulo.explicit_ode import Explicit_ODE
from assimulo.problem import Explicit_Problem
from assimulo.ode import *
import numpy as np
from numpy import hstack
from numpy.linalg import inv
from scipy.linalg import solve

import matplotlib.pyplot as mpl

k = 1000
h = 6e-2

def lambdafunc(x,y):
	hyp = np.hypot(x,y)
	return k*(hyp-1)/hyp

def step(t, u, up, upp, h):
	u_next = u + up * h + upp * h**2/2
	upp_next = np.array([0, -1]) - lambdafunc(u_next[0], u_next[1]) * u_next
	up_next = up + (upp + upp_next)*h/2
	
	return t+h, u_next, up_next, upp_next

t0 = 0
tf = 5

u0 = np.array([.5, -1])
up0 = np.zeros(2,)

upp0 = np.array([0, -1]) - lambdafunc(u0[0], u0[1]) * u0

tres = []
ures = []

t, u, up, upp = t0, u0, up0, upp0

while t < tf:
	t, u, up, upp = step(t, u, up, upp, h)
	
	tres.append(t)
	ures.append(u.copy())
	
	h = min(h, abs(tf-t))

x = [states[0] for states in ures]
y = [states[1] for states in ures]

mpl.plot(x, y, 'o', ls='--', lw=1, ms=4, markevery=[0,-1])
mpl.plot([0, u0[0]], [0, u0[1]], ls='-', lw=1)
mpl.annotate('t=0', xy=(x[0], y[0]), xytext=(x[0]+.05, y[0]))
mpl.annotate('t=%i'%tf, xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] - .07))

mpl.xlabel('x', fontsize=14)
mpl.ylabel('y', fontsize=14)

mpl.title('Elastic pendulum for %i s, k=%i' %(tf, k), fontsize=16)
mpl.axis([-1, 1, -1.5, 0])
mpl.grid(True)

mpl.show()
