from assimulo.explicit_ode import Explicit_ODE, Explicit_ODE_Exception
import numpy as np
import scipy.linalg as SL
from scipy.optimize import fsolve

class BDF_2(Explicit_ODE):
	"""
	BDF-2   (Example of how to set-up own integrators for Assimulo)
	"""
	tol = 1.e-8     
	maxit = 100
	maxsteps = 1000
 
	alpha=[3./2., -2., 1./2]
	
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
		
		for i in range(self.maxsteps):
			if t >= tf:
				break
			self.statistics["nsteps"] += 1
			
			if i==0:  # initial step
				t_np1,y_np1 = self.step_EE(t,y, h)
			else:   
				t_np1, y_np1 = self.step_BDF2([t,t_nm1], [y,y_nm1], h)
			t,t_nm1=t_np1,t
			y,y_nm1=y_np1,y
			
			tres.append(t)
			yres.append(y.copy())
		
			h=min(self.h,np.abs(tf-t))
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
		
	def step_BDF2(self, T, Y, h):
		"""
		BDF-2 with Fixed Point Iteration and Zero order predictor
		
		alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
		alpha=[3/2,-2,1/2]
		"""

		f = self.problem.rhs
		
		t_n, t_nm1 = T
		y_n, y_nm1 = Y
		# predictor
		t_np1 = t_n+h
		y_np1_i = y_n   # zero order predictor
		# corrector with fixed point iteration
		for i in range(self.maxit):
			self.statistics["nfcns"] += 1
			
			y_np1_ip1 = -(self.alpha[1]*y_n + self.alpha[2]*y_nm1) + h*f(t_np1,y_np1_i) / self.alpha[0]
			if SL.norm(y_np1_ip1 - y_np1_i) < self.tol:
				return t_np1, y_np1_ip1
			y_np1_i = y_np1_ip1
		else:
			raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)
			
	def print_statistics(self, verbose=NORMAL):
		self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
		self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
		self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
		self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
			
		self.log_message('\nSolver options:\n',                                    verbose)
		self.log_message(' Solver            : BDF2',                     verbose)
		self.log_message(' Solver type       : Fixed step\n',                      verbose)

class BDF_3(Explicit_ODE):
	"""
	BDF-3
	"""
	tol = 1.e-8
	maxit = 1000
	maxsteps = 1000
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
  
		for i in range(self.maxsteps+1):
			if t >= tf:
				break
			self.statistics["nsteps"] += 1
			
			if i == 0:  # initial steps
				t_np1, y_np1 = self.step_EE(t, y, h)
				t = t_np1
				y = y_np1
				y_nm1 = y
			elif i == 1:
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
			return self.alpha[0]*y_np1 + self.alpha[1]*Y[0] + self.alpha[2]*Y[1] + self.alpha[3]*Y[2] - h*f(t_np1, y_np1)
  
		y_np1, infodict, ier, _ = fsolve(func=F, x0=Y[0], full_output=True, xtol=self.tol, maxfev=self.maxit)
		self.statistics["nfcns"] += infodict.get('nfev')
  
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

class BDF_4(Explicit_ODE):
	"""
	BDF-4
	"""
	tol = 1.e-8
	maxit = 100
	maxsteps = 1000
	alpha = [25./12., -4., 3., -4./3., .25]
	
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
		
		t_nm3, t_nm2, t_nm1 = 0., h, 2.*h
		y_nm3 = y
  
		for i in range(self.maxsteps+1):
			if t >= tf:
				break
			self.statistics["nsteps"] += 1
			
			if i == 0:  # initial steps
				t_np1, y_np1 = self.step_EE(t, y, h)
				t = t_np1
				y = y_np1
				y_nm2 = y
			elif i == 1:
				t_np1, y_np1 = self.step_EE(t, y, h)
				t = t_np1
				y = y_np1
				y_nm1 = y
			elif i == 2:
				t_np1, y_np1 = self.step_EE(t, y, h)
				t = t_np1
				y = y_np1
			else:
				t_np1, y_np1 = self.step_BDF4([t, t_nm1, t_nm2, t_nm3], [y, y_nm1, y_nm2, y_nm3], h)
				t, t_nm1, t_nm2, t_nm3 = t_np1, t, t_nm1, t_nm2
				y, y_nm1, y_nm2, y_nm3 = y_np1, y, y_nm1, y_nm2
			
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
 
	def step_BDF4(self, T, Y, h):
		"""
		BDF-4: Backward differentiation formula
		y_np1 = 1/25 * [48y_n -36y_nm1 + 16y_nm2 -3y_nm3 + 12hf(t_np1, y_np1)]

		F(y_np1) = alpha*[y_np1, Y] - hf(t_np1, y_np1)
		Find F(y_np1) = 0 with fsolve()
		"""
		f=self.problem.rhs

		t_np1 = T[0] + h
  
		def F(y_np1):
			return self.alpha[0]*y_np1 + self.alpha[1]*Y[0] + self.alpha[2]*Y[1] + self.alpha[3]*Y[2] + self.alpha[4]*Y[3] - h*f(t_np1, y_np1)
  
		y_np1, infodict, ier, _ = fsolve(func=F, x0=Y[0], full_output=True, xtol=self.tol, maxfev=self.maxit)
		self.statistics["nfcns"] += infodict.get('nfev')
  
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
		self.log_message(' Solver            : BDF4',                     verbose)
		self.log_message(' Solver type       : Fixed step\n',                      verbose)
