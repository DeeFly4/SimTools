# -*- coding: utf-8 -*-
from __future__ import division
from assimulo.problem import Explicit_Problem
from assimulo.solvers import RungeKutta4
import matplotlib.pyplot as mpl
from scipy.linalg import solve
from numpy import array,zeros,ones,block,hstack,sin,cos,sqrt,dot,pi

# Inertia data
m1,m2,m3,m4,m5,m6,m7=.04325,.00365,.02373,.00706,.07050,.00706,.05498
i1,i2,i3,i4,i5,i6,i7=2.194e-6,4.410e-7,5.255e-6,5.667e-7,1.169e-5,5.667e-7,1.912e-5

# Geometry
xa,ya=-.06934,-.00227
xb,yb=-0.03635,.03273
xc,yc=.014,.072
d,da,e,ea=28.e-3,115.e-4,2.e-2,1421.e-5
rr,ra=7.e-3,92.e-5
ss,sa,sb,sc,sd=35.e-3,1874.e-5,1043.e-5,18.e-3,2.e-2
ta,tb=2308.e-5,916.e-5
u,ua,ub=4.e-2,1228.e-5,449.e-5
zf,zt=2.e-2,4.e-2
fa=1421.e-5

# Driving torque
mom=0.033

# Spring data
c0=4530.
lo=0.07785

def init_squeezer():
	y_1 = array([-0.0617138900142764496358948458001,  #  Beta
			0.,                                 #  Theta
			0.455279819163070380255912382449,   # gamma
			0.222668390165885884674473185609,   # Phi
			0.487364979543842550225598953530,   # delta
			-0.222668390165885884674473185609,  # Omega
			1.23054744454982119249735015568])   # epsilon
	
	return hstack((y_1, zeros((7,))))

def rhs(t, y):
	# Initial computations and assignments
	beta, theta, gamma, phi, delta, omega, epsilon = y[0:7]
	bep, thp, gap, php, dep, omp, epp = y[7:14]
	sibe, sith, siga, siph, side, siom, siep = sin(y[0:7])
	cobe, coth, coga, coph, code, coom, coep = cos(y[0:7])
	sibeth = sin(beta+theta)
	cobeth = cos(beta+theta)
	siphde = sin(phi+delta)
	cophde = cos(phi+delta)
	siomep = sin(omega+epsilon)
	coomep = cos(omega+epsilon)

	# Mass matrix
	m = zeros((7,7))
	m[0,0] = m1*ra**2 + m2*(rr**2-2*da*rr*coth+da**2) + i1 + i2
	m[1,0] = m[0,1] = m2*(da**2-da*rr*coth) + i2
	m[1,1] = m2*da**2 + i2
	m[2,2] = m3*(sa**2+sb**2) + i3
	m[3,3] = m4*(e-ea)**2 + i4
	m[4,3] = m[3,4] = m4*((e-ea)**2+zt*(e-ea)*siph) + i4
	m[4,4] = m4*(zt**2+2*zt*(e-ea)*siph+(e-ea)**2) + m5*(ta**2+tb**2)+ i4 + i5
	m[5,5] = m6*(zf-fa)**2 + i6
	m[6,5] = m[5,6] = m6*((zf-fa)**2-u*(zf-fa)*siom) + i6
	m[6,6] = m6*((zf-fa)**2-2*u*(zf-fa)*siom+u**2) + m7*(ua**2+ub**2)+ i6 + i7

	#   Applied forces
	xd = sd*coga + sc*siga + xb
	yd = sd*siga - sc*coga + yb
	lang  = sqrt ((xd-xc)**2 + (yd-yc)**2)
	force = - c0 * (lang - lo)/lang
	fx = force * (xd-xc)
	fy = force * (yd-yc)
	ff = array([
	mom - m2*da*rr*thp*(thp+2*bep)*sith,
	m2*da*rr*bep**2*sith,
	fx*(sc*coga - sd*siga) + fy*(sd*coga + sc*siga),
	m4*zt*(e-ea)*dep**2*coph,
	- m4*zt*(e-ea)*php*(php+2*dep)*coph,
	- m6*u*(zf-fa)*epp**2*coom,
	m6*u*(zf-fa)*omp*(omp+2*epp)*coom])

	#  Jacobian matrix G(q)
	gp = zeros((6,7))

	gp[0,0] = - rr*sibe + d*sibeth
	gp[0,1] = d*sibeth
	gp[0,2] = - ss*coga
	gp[1,0] = rr*cobe - d*cobeth
	gp[1,1] = - d*cobeth
	gp[1,2] = - ss*siga
	gp[2,0] = - rr*sibe + d*sibeth
	gp[2,1] = d*sibeth
	gp[2,3] = - e*cophde
	gp[2,4] = - e*cophde + zt*side
	gp[3,0] = rr*cobe - d*cobeth
	gp[3,1] = - d*cobeth
	gp[3,3] = - e*siphde
	gp[3,4] = - e*siphde - zt*code
	gp[4,0] = - rr*sibe + d*sibeth
	gp[4,1] = d*sibeth
	gp[4,5] = zf*siomep
	gp[4,6] = zf*siomep - u*coep
	gp[5,0] = rr*cobe - d*cobeth
	gp[5,1] = - d*cobeth
	gp[5,5] = - zf*coomep
	gp[5,6] = - zf*coomep - u*siep
 
	# g_qq(q)(qdot, qdot) for index-1
	g_qq = zeros((6,))

	g_qq[0] = -rr*cobe*bep**2 + d*cobeth*(bep+thp)**2 + ss*siga*gap**2
	g_qq[1] = -rr*sibe*bep**2 + d*sibeth*(bep+thp)**2 - ss*coga*gap**2
	g_qq[2] = -rr*cobe*bep**2 + d*cobeth*(bep+thp)**2 + e*siphde*(php+dep)**2 + zt*code*dep**2
	g_qq[3] = -rr*sibe*bep**2 + d*sibeth*(bep+thp)**2 - e*cophde*(php+dep)**2 + zt*side*dep**2
	g_qq[4] = -rr*cobe*bep**2 + d*cobeth*(bep+thp)**2 + zf*coomep*(omp+epp)**2 + u*siep*epp**2
	g_qq[5] = -rr*sibe*bep**2 + d*sibeth*(bep+thp)**2 + zf*siomep*(omp+epp)**2 -  u*coep*epp**2

	A = block([[m, gp.T], [gp, zeros((6,6))]])
	b = block([[ff.T], [-g_qq.T]])

	wlam = solve(A, b)
	w = wlam[0:7]

	qdot = y[7:14]
	vdot = w
    
	return block([qdot, vdot])

y0 = init_squeezer()
tf = 0.03

model = Explicit_Problem(rhs, y0, 0)
rk4 = RungeKutta4(model)
rk4.h = 1e-3
t, y = rk4.simulate(tf)

angles = [states[0:7] for states in y]

mpl.plot(t, angles, lw=2)
mpl.hlines(0, 0, tf, ls='--', colors='k')

mpl.xlabel('Time [s]', fontsize=16)
mpl.xlabel('Angle [mod 2pi]', fontsize=16)

mpl.title('Squeezer angles', fontsize=1)

mpl.axis([0, tf, -1, 1])

mpl.grid(True)

mpl.show()