from classes import Explicit_Problem_2nd, Explicit_ODE_2nd, Newmark, HHT_alpha
from assimulo.solvers import CVode, RungeKutta4, Dopri5
import numpy as np
import matplotlib.pyplot as mpl

omega = np.pi / .3

def f1(t):
    return np.zeros(2,)

def f2(t):
	return np.array([0, 0, 50*np.sin(omega*t)]) if t < np.pi / omega else np.zeros(3,)

# simple 2x2 system for testing
M1 = np.diag([1, 1])
K1 = np.array([[1, 1], [2, -3]])
C1 = np.array([[2, 0], [-1, 0]])

# 3x3 system from the source
M2 = np.diag([10, 20, 30])
K2 = 1e3 * np.array([[45, -20, -15], [-20, 45, -25], [-15, -25, 40]])
C2 = 3e-2 * K2

# initial conditions, change as you like
u0 = np.zeros(3,)
up0 = np.zeros(3,)

t0 = 0
tf = 3

model = Explicit_Problem_2nd(M2, C2, K2, f2, u0, up0, t0)
model.name = 'Test problem'
cvode = CVode(model)
sim = Newmark(model) # can be Newmark or HHT_alpha
sim.h = 1e-3

t1, u_all = cvode.simulate(tf)
t2, u2 = sim.simulate(tf)
u1 = [states[0:len(states)//2] for states in u_all]

# plot commands
fig, (ax1, ax2) = mpl.subplots(1,2)

ax1.plot(t1, u1, lw=2)
ax2.plot(t2, u2, lw=2)
ax1.hlines(0, 0, tf, ls='--', colors='k')
ax2.hlines(0, 0, tf, ls='--', colors='k')

ax1.set_title('CVode', fontsize=16)
ax2.set_title('Newmark', fontsize=16)
ax1.set_xlabel('Time [s]', fontsize=14)
ax2.set_xlabel('Time [s]', fontsize=14)

ax1.axis([0, tf, -0.005, 0.01])
ax2.axis([0, tf, -0.005, 0.01])

ax1.grid(True)
ax2.grid(True)

mpl.show()