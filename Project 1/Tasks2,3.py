from assimulo.problem import Explicit_Problem
from assimulo.solvers import ExplicitEuler
import numpy as np
import matplotlib.pyplot as mpl
from classes import BDF_2, BDF_3, BDF_4

def rhs(t,y):
	global k
	root = np.sqrt(y[0]**2 + y[1]**2)
	temp = k*(root - 1.)/root

	y1dot = y[2]
	y2dot = y[3]
	y3dot = -y[0]*temp
	y4dot = -y[1]*temp - 1.
 
	return np.array([y1dot, y2dot, y3dot, y4dot])

# Spring constant, change as you like
k = 1500.

# Initial conditions
y0 = np.array([.5, -1, 0., 0.])
t0 = 0.
tf = 5.

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Elastic Pendulum'

sim = BDF_4(model) # Create a BDF solver of choice
EE_sim = ExplicitEuler(model)
t, y = sim.simulate(tf)

x = [states[0] for states in y]
y = [states[1] for states in y]

# plot commands
mpl.plot(x, y, 'o', ls='--', lw=1, ms=4, markevery=[0,-1])
mpl.plot([0, y0[0]], [0, y0[1]], ls='-', lw=1)
mpl.annotate('t=0', xy=(x[0], y[0]), xytext=(x[0]+.05, y[0]))
mpl.annotate('t=%i'%tf, xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] - .07))

mpl.xlabel('x', fontsize=14)
mpl.ylabel('y', fontsize=14)

mpl.title('Elastic pendulum for %i s, k=%i' %(tf, k), fontsize=16)
mpl.axis([-1, 1, -1.5, 0])
mpl.grid(True)

mpl.show()