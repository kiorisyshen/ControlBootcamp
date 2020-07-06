import numpy as np
import math
from scipy.integrate import solve_ivp
import control
from control.matlab import *
import matplotlib.pyplot as plt

from drawcartpend import drawcartpend
from cartpend import cartpendSystem

import taichi as ti
ti.init(debug=True)

control.use_numpy_matrix(flag=False)

m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = -9.8  # gravity  # simulate in stable state
d = 4.0  # dampping term

#############################
# Linearized system
A = np.array([
    [0, 1, 0, 0],
    [0, -d, -(g * m) / M, 0],
    [0, 0, 0, 1],
    [0, 0, (g * m + M * g) / (L * M), -0.1 * d]
], dtype=np.float)

B = np.array([[0], [1.0 / M], [0], [-1.0 / (L * M)]], dtype=np.float)

C = np.array([1, 0, 0, 0], dtype=np.float)  # only observable if x measured... because x can't be reconstructed

D = np.zeros((1, B.shape[1]), dtype=np.float)

Vd = 0.1*np.eye(4)  # disturbance covariance
Vn = 0.2            # noise covariance

BF = np.concatenate([B, Vd, 0.0*B], axis=1)  # augment inputs to include disturbance and noise


sysC = control.StateSpace(A, BF, C, np.array([0, 0, 0, 0, 0, Vn]))  # build big state space system... with single output
sysFullOutput = control.StateSpace(A, BF, np.eye(4), np.zeros(BF.shape))  # system with full state output, disturbance, no noise

#  Build Kalman filter
# Kf, P, E = control.lqe(A, Vd, C, Vd, Vn)  # design Kalman filter
Kf, _, _ = control.lqr(np.transpose(A), np.expand_dims(C, axis=1), Vd, Vn)   # alternatively, possible to design using "LQR" code
Kf = np.transpose(Kf)

sysKF = control.StateSpace(A-Kf*C, np.concatenate([B, Kf], axis=1), np.eye(4), 0*np.concatenate([B, Kf], axis=1))  # Kalman filter estimator

# Estimate linearized system in "down" position(Gantry crane)
dt = 0.01
tspan = [0, 10]
t = np.arange(tspan[0], tspan[1], dt)
y0 = np.array([0, 0, 0, 0], dtype=np.float)

uDIST = np.random.multivariate_normal(np.zeros(4), Vd, size=t.shape[0])
uDIST = np.transpose(uDIST)
uNOISE = np.random.normal(0.0, Vn, size=(1, t.shape[0]))
u = np.zeros([1, t.shape[0]], dtype=np.float)
u[:, 100: 110] = 50  # impulse
u[:, 400: 420] = -50  # impulse

uAUG = np.concatenate([u, uDIST, uNOISE])

yout, T1, xout1 = lsim(sysC, np.transpose(uAUG), t, X0=y0)
xtrue, T2, xout2 = lsim(sysFullOutput, np.transpose(uAUG), t, X0=y0)

ukal = np.concatenate([u, np.expand_dims(yout, axis=0)])
xk, T3, xout3 = lsim(sysKF, np.transpose(ukal), t)

plt.plot(T1, yout, "k-")
plt.plot(T2, xtrue[:, 0], "r--")
plt.plot(T2, xtrue[:, 1], "g--")
plt.plot(T2, xtrue[:, 2], "b--")
plt.plot(T2, xtrue[:, 3], "y--")
plt.plot(T3, xk[:, 0], "y-")
plt.plot(T3, xk[:, 1], "r-")
plt.plot(T3, xk[:, 2], "g-")
plt.plot(T3, xk[:, 3], "b-")
plt.show()

gui = ti.GUI('Sim cartpend', res=(512, 512), background_color=0xdddddd)
print(xtrue.shape)
t_curr = 0
while gui.running:
    drawcartpend(gui, xtrue[t_curr, :], m, M, L)
    gui.show()
    t_curr += 1
    if (t_curr > xtrue.shape[0] - 1):
        print(xtrue[t_curr - 1, :])
        t_curr = 0
