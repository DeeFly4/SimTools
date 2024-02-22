from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import numpy as np
from numpy import multiply, divide, reciprocal, hstack

import matplotlib.pyplot as mpl

class Explicit_Problem_2nd(Explicit_Problem):
	def __init__(self, M, C, K, f, u0, udot0, t0):
		self.a = -divide(K,M)
		self.b = -divide(C,M)
		self.c = reciprocal(M)
		self.f = f
		Explicit_Problem.__init__(self, self.rhs, hstack((u0, udot0)), t0)

	def rhs(self, t, y):
		u = y[0:len(y)//2]
		udot = y[len(y)//2:len(y)]
  
		y1dot = udot
		y2dot = multiply(self.a, u) + multiply(self.b, udot) + multiply(self.c, self.f(t))
  
		return hstack((y1dot, y2dot))

def f(t):
    return 0.

testeq = Explicit_Problem_2nd(1., 3., .2, f, 1., 0., 0.)
sim = CVode(testeq)
sim.simulate(5)

sim.plot()