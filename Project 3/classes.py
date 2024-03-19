from assimulo.explicit_ode import Explicit_ODE
from assimulo.problem import Explicit_Problem
from assimulo.ode import *
from numpy import hstack
from scipy.linalg import solve
import scipy.sparse as ssp
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as mpl

import time

class Explicit_Problem_2nd(Explicit_Problem):
	def __init__(self, M, C, K, f, u0, up0, t0):
		self.u0 = u0
		self.up0 = up0
		self.t0 = t0

		if ssp.issparse(M) and ssp.issparse(C) and ssp.issparse(K):
			self.M = M
			self.C = C
			self.K = K
		else:
			self.M = ssp.csc_matrix(M)
			self.C = ssp.csc_matrix(C)
			self.K = ssp.csc_matrix(K)

		self.f = f
		Explicit_Problem.__init__(self, self.rhs, hstack((self.u0, self.up0)), t0)

	def rhs(self, t, y):
		u = y[0:len(y)//2]
		up = y[len(y)//2:len(y)]

		y1dot = up
		y2dot = spsolve(self.M, -self.K@u - self.C@up + self.f(t))

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
		
		#Solver options
		self.options["h"] = 1e-3
		
		#Statistics
		self.statistics["nsteps"] = 0
  
	def _set_h(self,h):
		self.options["h"] = float(h)

	def _get_h(self):
		return self.options["h"]

	h = property(_get_h,_set_h)

	def _integrate(self, t, u, up, upp, tf, step):
		h = self.options["h"]
		h = min(self.h, abs(tf-t))

		tres = []
		ures = []
  
		while t < tf:
			self.statistics["nsteps"] += 1
			t, u, up, upp = step(t, u, up, upp, h)

			tres.append(t)
			ures.append(u.copy())

			h = min(self.h, abs(tf-t))

		return ID_PY_OK, tres, ures


	def print_statistics(self, verbose=NORMAL):
		self.log_message('\nFinal Run Statistics          : {name} \n'.format(name=self.problem.name), verbose)
		self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
		self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]), verbose)               

		self.log_message('\nSolver options:\n', verbose)
		self.log_message(' Solver type : Fixed step', verbose)
		self.log_message(' Solver      : '+self.__class__.__name__, verbose)

	def simulate(self, tf):
		t = time.perf_counter()

		_, tt, u = self.integrate(self.t0, self.u0, self.up0, tf)

		self.print_statistics()
		self.log_message('Simulation interval	  : ' + str(self.t0) + ' - ' + str(tf) + ' seconds.', NORMAL)
		self.log_message('Elapsed simulation time : ' + str(time.perf_counter() - t) + ' seconds.', NORMAL)

		return tt, u

class Newmark(Explicit_ODE_2nd):
	gamma = 1/2
	Beta = 1/4
	
	def __init__(self, problem):
		Explicit_ODE_2nd.__init__(self, problem)
		self.up = self.up0
		
		self.A = self.M / (self.Beta*self.h**2) + self.gamma*self.C / (self.Beta*self.h) + self.K
		
		self.options["gamma"] = self.gamma
		self.options["Beta"] = self.Beta
 
	def integrate(self, t0, u0, up0, tf):
		upp0 = spsolve(self.M, self.f(0) - self.K@u0)
		
		if self.C.nnz == 0 and self.Beta == 0:
			step = self.explicit_step
		else:
			upp0 -= spsolve(self.M, self.C@up0)
			step = self.implicit_step

		return self._integrate(t0, u0, up0, upp0, tf, step)
	
	def explicit_step(self, t, u, up, upp, h):
		u_next = u + up*h + upp*h**2/2
		upp_next = spsolve(self.M, self.f(t) - self.K@u)
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
		
		u_next = spsolve(self.A, Bn)
		up_next = self.gamma*(u_next - u)/bh + up*omgb + h*upp*omg2b
		upp_next = (u_next - u)/bh2 - up/bh - upp*inv2bmo
  
		return t_next, u_next, up_next, upp_next

	def print_statistics(self, verbose=NORMAL):
		super().print_statistics(verbose)
		self.log_message(' gamma, Beta : ' + str(self.options["gamma"]) + ', ' + str(self.options["Beta"]) + '\n', verbose)		

class HHT_alpha(Explicit_ODE_2nd):
	alpha = -1/3
	Beta = (1 - alpha)**2/4
	gamma = 1/2 - alpha

	def __init__(self, problem):
		Explicit_ODE_2nd.__init__(self, problem)
  
		self.options["alpha"] = self.alpha
  
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
		
		u_next = spsolve(self.A, Bn)
		up_next = self.gamma*(u_next - u)/bh + up*omgb + h*upp*omg2b
		upp_next = (u_next - u)/bh2 - up/bh - upp*inv2bmo
  
		return t_next, u_next, up_next, upp_next

	def integrate(self, t0, u0, up0, tf):
		upp0 = spsolve(self.M, self.f(0) - self.K@u0 - self.C@up0)
		return self._integrate(t0, u0, up0, upp0, tf, self.step)

	def print_statistics(self, verbose=NORMAL):
		super().print_statistics(verbose)
		self.log_message(' alpha       : '+str(self.options["alpha"])+'\n', verbose)