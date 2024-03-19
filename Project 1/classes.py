from assimulo.explicit_ode import Explicit_ODE, Explicit_ODE_Exception
from assimulo.ode import *
import numpy as np
from scipy.optimize import fsolve

class BDF(Explicit_ODE):
	tol = 1.e-8
	maxit = 100
	maxsteps = 10000
 
	alpha2 = [3./2, -2., 1./2]
	alpha3 = [11./6., -3., 1.5, -1./3.]
	alpha4 = [25./12., -4., 3., -4./3., .25]

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
		
	h = property(_get_h,_set_h)

	def step_IE(self, t, y, h):
		"""
		This calculates the next step in the integration with implicit Euler.
		"""
		
		t_np1 = t + h
  
		def F(y_np1):
			return y + h*self.problem.rhs(t_np1, y_np1) - y_np1

		y_np1, infodict, ier, _ = fsolve(func=F, x0=y, full_output=True, xtol=self.tol, maxfev=self.maxit)

		self.statistics["nfcns"] += infodict.get('nfev')
  
		if ier == 1:
			return t_np1, y_np1
		else:
			raise Explicit_ODE_Exception('Corrector could not converge within %i iterations' %self.maxit)

	def step_BDF2(self, T, Y, h):
		"""
		BDF-2: Backward differentiation formula
		y_np1 = 1/3 * [4y_n - y_nm1 + 2hf(t_np1, y_np1)]
		
		F(y_np1) = 3/2*y_np1 - 2y_n + 1/2*y_nm1 - h*f(t_np1, y_np1)
		Find F(y_np1) = 0 with Newton-Raphson iteration
  		y_np1_ip1 = y_np1_i - J(y_np1_i)^-1 * F(y_np1_i)
		"""

		f = self.problem.rhs
		
		t_np1 = T[0] + h

		def F(y_np1):
			return self.alpha2[0]*y_np1 + self.alpha2[1]*Y[0] + self.alpha2[2]*Y[1] - h*f(t_np1, y_np1)

		y_np1, infodict, ier, _ = fsolve(func=F, x0=Y[0], full_output=True, xtol=self.tol, maxfev=self.maxit)
		self.statistics["nfcns"] += infodict.get('nfev')
  
		if ier == 1:
			return t_np1, y_np1
		else:
			raise Explicit_ODE_Exception('Corrector could not converge within %i iterations' %self.maxit)
	
	def step_BDF3(self, T, Y, h):
		"""
		BDF-3: Backward differentiation formula
		y_np1 = 1/11 * [18y_n - 9y_nm1 + 2y_nm2 + 6hf(t_np1, y_np1)]

		F(y_np1) = 11/6 * y_np1 - 3y_n + 1.5y_nm1 - 1/3 * y_nm2 - hf(t_np1, y_np1)
		Find F(y_np1) = 0 with Newton-Raphson iteration
		y_np1_ip1 = y_np1_i - J(y_np1_i)^-1 * F(y_np1_i)
		"""
		f = self.problem.rhs

		t_np1 = T[0] + h
  
		def F(y_np1):
			return self.alpha3[0]*y_np1 + self.alpha3[1]*Y[0] + self.alpha3[2]*Y[1] + self.alpha3[3]*Y[2] - h*f(t_np1, y_np1)
  
		y_np1, infodict, ier, _ = fsolve(func=F, x0=Y[0], full_output=True, xtol=self.tol, maxfev=self.maxit)
		self.statistics["nfcns"] += infodict.get('nfev')
  
		if ier == 1:
			return t_np1, y_np1
		else:
			raise Explicit_ODE_Exception('Corrector could not converge within %i iterations' %self.maxit)

	def step_BDF4(self, T, Y, h):
		"""
		BDF-4: Backward differentiation formula
		y_np1 = 1/25 * [48y_n -36y_nm1 + 16y_nm2 -3y_nm3 + 12hf(t_np1, y_np1)]

		F(y_np1) = alpha*[y_np1, Y] - hf(t_np1, y_np1)
		Find F(y_np1) = 0 with fsolve()
		"""
		f = self.problem.rhs

		t_np1 = T[0] + h

		def F(y_np1):
			return self.alpha4[0]*y_np1 + self.alpha4[1]*Y[0] + self.alpha4[2]*Y[1] + self.alpha4[3]*Y[2] + self.alpha4[4]*Y[3] - h*f(t_np1, y_np1)

		y_np1, infodict, ier, _ = fsolve(func=F, x0=Y[0], full_output=True, xtol=self.tol, maxfev=self.maxit)
		self.statistics["nfcns"] += infodict.get('nfev')
  
		if ier == 1:
			return t_np1, y_np1
		else:
			raise Explicit_ODE_Exception('Corrector could not converge within %i iterations' %self.maxit)

	def print_statistics(self, verbose=NORMAL):
		self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name), verbose)
		self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
		self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]), verbose)               
		self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]), verbose)
			
		self.log_message('\nSolver options:\n', verbose)
		self.log_message(' Solver type : Fixed step', verbose)
		self.log_message(' Solver      : '+self.__class__.__name__+'\n', verbose)

class BDF_2(BDF):
	"""
	BDF-2
	"""
	
	def __init__(self, problem):
		BDF.__init__(self, problem) #Calls the base class
		
	def integrate(self, t, y, tf, opts):
		"""
		integrates (t,y) values until t > tf. First step is implicit Euler.
		"""
		h = self.options["h"]
		h = min(h, abs(tf-t))
		
		#Lists for storing the result
		tres = []
		yres = []
  
		t_nm1 = 0
		y_nm1 = y
		
		for i in range(self.maxsteps):
			if t >= tf:
				break
			self.statistics["nsteps"] += 1
			
			if i==0:  # initial step
				t_np1, y_np1 = self.step_IE(t, y, h)
			else:   
				t_np1, y_np1 = self.step_BDF2([t, t_nm1], [y, y_nm1], h)
			t, t_nm1 = t_np1, t
			y, y_nm1 = y_np1, y
			
			tres.append(t)
			yres.append(y.copy())
		
			h = min(self.h, np.abs(tf-t))
		else:
			raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
		
		return ID_PY_OK, tres, yres
	
class BDF_3(BDF):
	"""
	BDF-3
	"""
	
	def __init__(self, problem):
		BDF.__init__(self, problem) #Calls the base class
		
	def integrate(self, t, y, tf, opts):
		"""
		integrates (t,y) values until t > tf. First step is implicit Euler, second step is BDF2.
		"""
		h = self.options["h"]
		h = min(h, abs(tf-t))
		
		#Lists for storing the result
		tres = []
		yres = []
  
		t_nm2 = 0
		y_nm2 = y
  
		for i in range(self.maxsteps+1):
			if t >= tf:
				break
			self.statistics["nsteps"] += 1
			
			if i == 0:  # initial steps
				t_np1, y_np1 = self.step_IE(t, y, h)
				t, t_nm1 = t_np1, t
				y, y_nm1 = y_np1, y
			elif i == 1:
				t_np1, y_np1 = self.step_BDF2([t, t_nm1], [y, y_nm1], h)
				t, t_nm1 = t_np1, t
				y, y_nm1 = y_np1, y
			else:   
				t_np1, y_np1 = self.step_BDF3([t, t_nm1, t_nm2], [y, y_nm1, y_nm2], h)
				t, t_nm1, t_nm2 = t_np1, t, t_nm1
				y, y_nm1, y_nm2 = y_np1, y, y_nm1
			
			tres.append(t)
			yres.append(y.copy())
		
			h = min(self.h, np.abs(tf - t))
		else:
			raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
		
		return ID_PY_OK, tres, yres	

class BDF_4(BDF):
	"""
	BDF-4
	"""
	
	def __init__(self, problem):
		BDF.__init__(self, problem) #Calls the base class
		
	def integrate(self, t, y, tf, opts):
		"""
		integrates (t,y) values until t > tf. First step is implicit Euler, second step is BDF2, third step is BDF3
		"""
		h = self.options["h"]
		h = min(h, abs(tf-t))
		
		#Lists for storing the result
		tres = []
		yres = []
		
		t_nm3 = 0
		y_nm3 = y
  
		for i in range(self.maxsteps+1):
			if t >= tf:
				break
			self.statistics["nsteps"] += 1
			
			if i == 0:  # initial steps
				t_np1, y_np1 = self.step_IE(t, y, h)
				t_nm2, y_nm2 = t, y # for step_BDF3
				t, t_nm1 = t_np1, t
				y, y_nm1 = y_np1, y
			elif i == 1:
				t_np1, y_np1 = self.step_BDF2([t, t_nm1], [y, y_nm1], h)
				t, t_nm1 = t_np1, t
				y, y_nm1 = y_np1, y
			elif i == 2:
				t_np1, y_np1 = self.step_BDF3([t, t_nm1, t_nm2], [y, y_nm1, y_nm2], h)
				t, t_nm1, t_nm2 = t_np1, t, t_nm1
				y, y_nm1, y_nm2 = y_np1, y, y_nm1
			else:
				t_np1, y_np1 = self.step_BDF4([t, t_nm1, t_nm2, t_nm3], [y, y_nm1, y_nm2, y_nm3], h)
				t, t_nm1, t_nm2, t_nm3 = t_np1, t, t_nm1, t_nm2
				y, y_nm1, y_nm2, y_nm3 = y_np1, y, y_nm1, y_nm2
			
			tres.append(t)
			yres.append(y.copy())
		
			h = min(self.h, np.abs(tf - t))
		else:
			raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
		
		return ID_PY_OK, tres, yres