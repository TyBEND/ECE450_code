# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import exp, cos, sin, log
from math import pi, sqrt, exp, cos, sin
from cmath import exp

NN = 20000
dt = 0.01
TT = np.arange(0,dt*NN,dt)
time1 = np.zeros(NN)
time2 = np.zeros(NN)
time3 = np.zeros(NN)
time4 = np.zeros(NN)
ytime = np.zeros(NN)
a = 0.01
b = 0.02
c = 0.1
d = 0.05
e = 0.3
w1 = 2*pi*0.3
w2 = 2*pi*0.15


""" Matrix definitions """
A = np.matrix('0 1 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0.')
A[1, 0] = -e
A[1, 1] = -d
A[1, 3] = 1-b
A[2, 2] = -a
A[3, 3] = -b

B = np.matrix('0 0; 0 1; 1 0; 0 1.')

C = np.matrix('0 1 0 0.')
C[0,0] = c

x = np.matrix('0; 0; 0; 0.')
u = np.matrix('0; 0.')
x[0] = 0
x[1] = 0
x[2] = 0
x[3] = 0
""" Forcing functions """
f = np.zeros(NN)
g = np.zeros(NN)
for n in range(0, 2000):
    f[n] = sin(w1*(n*dt))
for n in range(2000, 4000):
    g[n] = sin(w2*(n*dt))

""" Begin simulation """

nsteps = NN

for i in range(nsteps):
    time1[i] = x[0]
    time2[i] = x[1]
    time3[i] = x[2]
    time4[i] = x[3]
    
    ytime[i] = C*x
    u[0] = f[i]
    u[1] = g[i]
    x = x + dt*A*x + dt*B*u

plt.figure(figsize=(12,9))
plt.subplot(3,1,1)
plt.plot(TT,f,'r-',label='f')
plt.plot(TT,g,'g-.',label='g')
plt.title('HW3 Solution 6.2.4')
plt.ylabel('Forcing Functions')
plt.legend()
plt.grid()
plt.axis([0, 50, -1, 1])
    
plt.subplot(3,1,2)
plt.plot(TT,time1,'r-',label='x1')
plt.plot(TT,time2,'b-.',label='x2')
plt.plot(TT,time3, 'g-',label='x3')
plt.plot(TT,time4, 'k-',label='x4')
plt.ylabel('State Variables')
plt.legend()
plt.grid()

plt.subplot(3,1,3)  
plt.plot(TT,ytime, 'k-',label='y')
plt.ylabel('Total Output')
plt.xlabel('sec')
plt.legend()
plt.grid()
plt.savefig('HW3-6-2-4.png', dpi=300)

