from assimulo.explicit_ode import Explicit_ODE
from assimulo.problem import Explicit_Problem
from assimulo.ode import *
from assimulo.solvers import CVode
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL

# Define the rhs function
def rhs(t,y):
	global k
	root = np.sqrt(y[0]**2 + y[1]**2)
	temp = k*(root - 1)/root

	y1dot = y[2]
	y2dot = y[3]
	y3dot = -y[0]*temp
	y4dot = -y[1]*temp - 1
 
	return np.array([y1dot, y2dot, y3dot, y4dot])

# Spring constant
k = 2

# Initial conditions
y0 = np.array([0.1, 0, np.pi/4, 0])
t0 = 0

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Elastic Pendulum'

sim = CVode(model)

tfinal = 10
t, y = sim.simulate(tfinal)
sim.plot()