# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import exp, cos, sin, log
from math import pi, sqrt, exp, cos, sin
from cmath import exp

NN = 1000
dt = 0.01
TT = np.arange(0,dt*NN,dt)
time1 = np.zeros(NN)
time2 = np.zeros(NN)
time3 = np.zeros(NN)
ytime = np.zeros(NN)

""" Matrix definitions """
A = np.matrix('0 1 0; 0 0 1; -6 -16 -8')
B = np.matrix('0; 0; 1')
C = np.matrix('6 8 2')
x = np.matrix('0; 0; 0')
f = np.zeros(NN)

x[0] = 0
x[1] = 0
x[2] = 0
""" Forcing functions """
for n in range(200):
    f[n] = 1
    
## f = np.ones(NN) - f
""" Begin simulation """

nsteps = NN

for i in range(nsteps):
    time1[i] = x[0]
    time2[i] = x[1]
    time3[i] = x[2]
    
    ytime[i] = C*x
    x = x + dt*A*x + dt*B*f[i]
    
plt.figure(figsize=(12,9))
plt.subplot(3,1,1)
plt.plot(TT,f,'r-',label='f')
plt.title('HW3 Solution 6.2.2')
plt.ylabel('Forcing Function')
plt.legend()
plt.grid()

plt.subplot(3,1,2)
plt.plot(TT,time1,'r-',label='x1')
plt.plot(TT,time2,'b-.',label='x2')
plt.plot(TT,time3, 'g-',label='x3')
plt.ylabel('State Variables')
plt.legend()
plt.grid()

plt.subplot(3,1,3)  
plt.plot(TT,ytime, 'k-',label='y')
plt.ylabel('Total Output')
plt.xlabel('sec')
plt.legend()
plt.grid()
plt.savefig('HW3-6-2-2.png', dpi=300)
