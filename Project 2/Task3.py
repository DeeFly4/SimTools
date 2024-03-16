from __future__ import division
import matplotlib.pyplot as mpl
from scipy.linalg import norm
from scipy.optimize import fsolve
from numpy import zeros, sin, cos, array

# Geometry
xa, ya = -.06934, -.00227
xb, yb = -0.03635, .03273
d, e = 28.e-3, 2.e-2
rr = 7.e-3
ss = 35.e-3
u = 4.e-2
zf, zt = 2.e-2, 4.e-2

def g(q):
	Beta, gamma, Phi, delta, Omega, epsilon = q[0:6]

	gvec = zeros((6,))

	gvec[0] = rr*cos(Beta) - d*cos(Beta) - ss*sin(gamma) - xb
	gvec[1] = rr*sin(Beta) - d*sin(Beta) + ss*cos(gamma) - yb
	gvec[2] = rr*cos(Beta) - d*cos(Beta) - e*sin(Phi+delta) - zt*cos(delta) - xa
	gvec[3] = rr*sin(Beta) - d*sin(Beta) + e*cos(Phi+delta) - zt*sin(delta) - ya
	gvec[4] = rr*cos(Beta) - d*cos(Beta) - zf*cos(Omega+epsilon) - u*sin(epsilon) - xa
	gvec[5] = rr*sin(Beta) - d*sin(Beta) - zf*sin(Omega+epsilon) + u*cos(epsilon) - ya

	return gvec

y_1 = array([-0.0617138900142764496358948458001,  #  Beta
				0.455279819163070380255912382449,   # gamma
				0.222668390165885884674473185609,   # Phi
				0.487364979543842550225598953530,   # delta
				-0.222668390165885884674473185609,  # Omega
				1.23054744454982119249735015568])   # epsilon

# calculate initial conditions using fsolve
g0 = fsolve(g, x0=zeros((6,)))

print(g0 - y_1)
print(norm(g0 - y_1))