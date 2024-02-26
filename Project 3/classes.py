from assimulo.explicit_ode import Explicit_ODE
from assimulo.problem import Explicit_Problem
from assimulo.ode import *
import numpy as np
from numpy import hstack
from numpy.linalg import inv
from scipy.linalg import solve

import matplotlib.pyplot as mpl

class Explicit_Problem_2nd(Explicit_Problem):
	def __init__(self, M, C, K, f, u0, up0, t0):
		self.u0 = u0
		self.up0 = up0
		self.t0 = t0

		self.M = M
		self.C = C
		self.K = K
		self.f = f
		Explicit_Problem.__init__(self, self.rhs, hstack((self.u0, self.up0)), t0)

	def rhs(self, t, y):
		u = y[0:len(y)//2]
		up = y[len(y)//2:len(y)]

		y1dot = up
		y2dot = solve(self.M, self.K@u) + solve(self.M, self.C@up) + solve(self.M, self.f(t))

		return hstack((y1dot, y2dot))

class Explicit_ODE_2nd(Explicit_ODE):
	def __init__(self, problem):
		Explicit_ODE.__init__(self, problem)
		self.M = problem.M
		self.C = problem.C
		self.K = problem.K
		self.f = problem.f
		self.u0 = problem.u0
		self.up0 = problem.up0
		self.t0 = problem.t0

class Newmark(Explicit_ODE_2nd):
	Beta = 1/4
	gamma = 1/2
	h = 1e-3
	
	def __init__(self, problem):
		Explicit_ODE_2nd.__init__(self, problem)
		self.up = self.up0
		
		self.A = self.M / (self.Beta*self.h**2) + self.gamma*self.C / (self.Beta*self.h) + self.K
 
	def integrate(self, t, u, up, tf, opts):
		h = min(self.h, abs(tf-t))
		upp = solve(self.M, self.f(0) - self.K@u)
		if self.C.any() != 0 and self.Beta != 0:
			upp -= solve(self.M, self.C@up)

		tres = []
		ures = []
  
		while t < tf:
			if self.C.all() == 0 and self.Beta == 0:
				t, u, up, upp = self.explicit_step(t, u, up, upp, h)
			else:
				t, u, up, upp = self.implicit_step(t, u, up, upp, h)

			tres.append(t)
			ures.append(u.copy())

			h = min(self.h, abs(tf-t))

		return ID_PY_OK, tres, ures
	
	def simulate(self, tf):
		flag, t, u = self.integrate(self.t0, self.u0, self.up0, tf, opts=None)
		return t, u
	
	def explicit_step(self, t, u, up, upp, h):
		u_next = u + up*h + upp*h**2/2
		upp_next = solve(self.M, self.f(t) - self.K@u)
		up_next = up + upp*h*(1-self.gamma) + self.gamma*upp_next*h

		return t+h, u_next, up_next, upp_next
	
	def implicit_step(self, t, u, up, upp, h):
		bh = self.Beta*h
		bh2 = self.Beta*h**2
		inv2bmo = 1/(2*self.Beta) - 1
		omgb = 1 - self.gamma/self.Beta
		omg2b = 1 - self.gamma/(2*self.Beta)
  
		t_next = t+h

		Bn = self.f(t_next) + self.M @ (u/bh2 + up/bh + upp*inv2bmo) + self.C @ (self.gamma*u/bh - up*omgb - h*upp*omg2b)
		
		u_next = solve(self.A, Bn)
		up_next = self.gamma*(u_next - u)/bh + up*omgb + h*upp*omg2b
		upp_next = (u_next - u)/bh2 - up/bh - upp*inv2bmo
  
		return t_next, u_next, up_next, upp_next

class HHT_alpha(Explicit_ODE_2nd):
	alpha = -0.2
	Beta = (1-alpha)**2/4
	gamma = 1/2 - alpha
	h = 1e-3

	def __init__(self, problem):
		Explicit_ODE_2nd.__init__(self, problem)
  
		self.up = self.up0
		
		self.A = self.M / (self.Beta*self.h**2) + self.gamma*self.C / (self.Beta*self.h) + (1+self.alpha)*self.K

	def step(self, t, u, up, upp, h):
		bh = self.Beta*h
		bh2 = self.Beta*h**2
		inv2bmo = 1/(2*self.Beta) - 1
		omgb = 1 - self.gamma/self.Beta
		omg2b = 1 - self.gamma/(2*self.Beta)
  
		t_next = t+h

		Bn = self.f(t_next) + self.M @ (u/bh2 + up/bh + upp*inv2bmo) + self.C @ (self.gamma*u/bh - up*omgb - h*upp*omg2b) + self.alpha*self.K@u
		
		u_next = solve(self.A, Bn)
		up_next = self.gamma*(u_next - u)/bh + up*omgb + h*upp*omg2b
		upp_next = (u_next - u)/bh2 - up/bh - upp*inv2bmo
  
		return t_next, u_next, up_next, upp_next

	def integrate(self, t, u, up, tf, opts):
		h = min(self.h, abs(tf-t))
		upp = solve(self.M, self.f(0) - self.K@u - self.C@up)

		tres = []
		ures = []
  
		while t < tf:
			t, u, up, upp = self.step(t, u, up, upp, h)

			tres.append(t)
			ures.append(u.copy())

			h = min(self.h, abs(tf-t))

		return ID_PY_OK, tres, ures
	
	def simulate(self, tf):
		flag, t, u = self.integrate(self.t0, self.u0, self.up0, tf, opts=None)
		return t, u