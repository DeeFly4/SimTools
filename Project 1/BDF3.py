from assimulo.explicit_ode import Explicit_ODE, Explicit_ODE_Exception
from assimulo.problem import Explicit_Problem
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from scipy.optimize import fsolve

class BDF_3(Explicit_ODE):
	"""
	BDF-3
	"""
	tol=1.e-8     
	maxit=100     
	maxsteps=500
	alpha = [11./6., -3., 1.5, -1./3.]
	
	def __init__(self, problem):
		Explicit_ODE.__init__(self, problem) #Calls the base class
		
		#Solver options
		self.options["h"] = 0.01
		
		#Statistics
		self.statistics["nsteps"] = 0
		self.statistics["nfcns"] = 0
	
	def _set_h(self,h):
			self.options["h"] = float(h)

	def _get_h(self):
		return self.options["h"]
		
	h=property(_get_h,_set_h)
		
	def integrate(self, t, y, tf, opts):
		"""
		_integrates (t,y) values until t > tf
		"""
		h = self.options["h"]
		h = min(h, abs(tf-t))
		
		#Lists for storing the result
		tres = []
		yres = []
		
		t_nm2, t_nm1 = 0., h
		y_nm2 = y
  
		for i in range(self.maxsteps):
			if t >= tf:
				break
			self.statistics["nsteps"] += 1
			
			if i == 0:  # initial steps
				t_np1, y_np1 = self.step_EE(t, y, h)
				t = t_np1
				y = y_np1
				y_nm1 = y
			if i == 1:
				t_np1, y_np1 = self.step_EE(t, y, h)
				t = t_np1
				y = y_np1
			else:   
				t_np1, y_np1 = self.step_BDF3([t, t_nm1, t_nm2], [y, y_nm1, y_nm2], h)
				t, t_nm1, t_nm2 = t_np1, t, t_nm1
				y, y_nm1, y_nm2 = y_np1, y, y_nm1
			
			tres.append(t)
			yres.append(y.copy())
		
			h=min(self.h, np.abs(tf - t))
		else:
			raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
		
		return ID_PY_OK, tres, yres
	
	def step_EE(self, t, y, h):
		"""
		This calculates the next step in the integration with explicit Euler.
		"""
		self.statistics["nfcns"] += 1
		
		f = self.problem.rhs
		return t + h, y + h*f(t, y)
 
	def step_BDF3(self, T, Y, h):
		"""
		BDF-3: Backward differentiation formula
		y_np1 = 1/11 * [18y_n - 9y_nm1 + 2y_nm2 + 6hf(t_np1, y_np1)]

		F(y_np1) = 11/6 * y_np1 - 3y_n + 1.5y_nm1 - 1/3 * y_nm2 - hf(t_np1, y_np1)
		Find F(y_np1) = 0 with Newton-Raphson iteration
		y_np1_ip1 = y_np1_i - J(y_np1_i)^-1 * F(y_np1_i)
		"""
		f=self.problem.rhs

		t_np1 = T[0] + h
  
		def F(y_np1):
			return self.alpha[0]*y_np1 - self.alpha[1]*Y[0] + self.alpha[2]*Y[1] - self.alpha[3]*Y[2] - h*f(t_np1, y_np1)
  
		y_np1, infodict, ier, _ = fsolve(func=F, x0=Y[0], full_output=True, xtol=self.tol, maxfev=self.maxit)
		self.statistics["nfcns"] = infodict.get('nfev')
  
		if ier == 1:
			return t_np1, y_np1
		else:
			raise Explicit_ODE_Exception('Corrector could not converge within %s iterations' % self.maxit)
			
	def print_statistics(self, verbose=NORMAL):
		self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
		self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
		self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
		self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
			
		self.log_message('\nSolver options:\n',                                    verbose)
		self.log_message(' Solver            : BDF3',                     verbose)
		self.log_message(' Solver type       : Fixed step\n',                      verbose)

def rhs(t,y):
	global k
	root = np.sqrt(y[0]**2 + y[1]**2)
	temp = k*(root - 1.)/root

	y1dot = y[2]
	y2dot = y[3]
	y3dot = -y[0]*temp
	y4dot = -y[1]*temp - 1.
 
	return np.array([y1dot, y2dot, y3dot, y4dot])

# Spring constant
k = 2.

# Initial conditions
y0 = np.array([0.1, 0., np.pi/4, 0.])
t0 = 0.
tf = 1.

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Elastic Pendulum'

exp_sim = BDF_3(model) # Create a BDF solver
t, y = exp_sim.simulate(tf)
exp_sim.plot()
mpl.show()