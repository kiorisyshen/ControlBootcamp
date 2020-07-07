import control
import numpy as np
import matplotlib.pyplot as plt

from drawcartpend import drawcartpend
from cartpend import cartpendSys_pyCtl
from cartpend import cartpendSysNoise_pyCtl

import taichi as ti
ti.init(debug=True)

m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = -9.8  # gravity
d = 4.0  # dampping term
Vd = 0.1 * np.eye(4)
Vn = 0.2
sysParams = {
    'm': m,  # mass of ball
    'M': M,  # mass of cart
    'L': L,  # pendulum length
    'g': g,  # gravity
    'd': d,   # dampping term
    'Vd': Vd,  # disterbance noise
    'Vn': Vn   # noise observe
}

tspan = [0, 30]  # time for simulation
dt = 0.03
y0 = [0.0, 0.0, 0.0, 0.1]  # initial state
xeq = [0.3, 0.0, 0.0, 0.0]  # final stable state

# Build non-linear standard system
sysFull = control.NonlinearIOSystem(cartpendSys_pyCtl, None, inputs=('u'),
                                    outputs=('x', 'xdot', 'theta', 'thetadot'),
                                    states=('x', 'xdot', 'theta', 'thetadot'),
                                    params=sysParams,
                                    name='sysFull')

C = np.array([1, 0, 0, 0], dtype=np.float)


def outfuncNoise(t, x, u, params):
    Vn = params.get('Vn')
    return x[0] + np.random.normal(0.0, Vn)


sysPartNoise = control.NonlinearIOSystem(cartpendSysNoise_pyCtl, outfuncNoise, inputs=('u'),
                                         outputs=('x'),
                                         states=('x', 'xdot', 'theta', 'thetadot'),
                                         params=sysParams,
                                         name='sysPartNoise')

# linearize on up position
sysLin = sysFull.linearize([0.0, 0.0, 0.0, 0.0], 0.0, params=sysParams)

# Kalman estimator
Kf, _, _ = control.lqr(np.transpose(sysLin.A), np.expand_dims(C, axis=1), Vd, Vn)   # alternatively, possible to design using "LQR" code
Kf = np.transpose(Kf)

# ssKF = control.StateSpace(sysLin.A - Kf * C, np.concatenate([sysLin.B, Kf], axis=1), np.eye(4), 0 * np.concatenate([sysLin.B, Kf], axis=1))  # Kalman filter estimator
# io_Estimator = control.LinearIOSystem(ssKF, inputs=('u', 's'),
#                                       outputs=('x_h', 'xdot_h', 'theta_h', 'thetadot_h'),
#                                       states=('x_h', 'xdot_h', 'theta_h', 'thetadot_h'),
#                                       name='kalmanEstimator')

K_A = sysLin.A - Kf * C
K_B = np.concatenate([sysLin.B, Kf], axis=1)


def kalmanStateUpdate(t, x, u, params):
    res = K_A @ np.array([x[0], x[1], x[2], x[3]]) + K_B @ np.array([u[0], u[1]])
    return res[0]


sysKF = control.NonlinearIOSystem(
    kalmanStateUpdate,
    None,
    inputs=('u', 's'), outputs=('x_h', 'xdot_h', 'theta_h', 'thetadot_h'),
    states=('x_h', 'xdot_h', 'theta_h', 'thetadot_h'),
    name='sysKF')

# test kalman estimator
# io_Estimator = control.InterconnectedSystem(
#     (sysPartNoise, sysKF),
#     connections=(
#         ('cartpendNoise.u', 'kalmanEstimator.u'),
#         ('kalmanEstimator.s', 'cartpendNoise.x')
#     ),
#     inplist=['kalmanEstimator.u'],
#     outlist=('kalmanEstimator.x_h', 'kalmanEstimator.xdot_h', 'kalmanEstimator.theta_h', 'kalmanEstimator.thetadot_h'),
#     states=('cartpendNoise.x', 'cartpendNoise.xdot', 'cartpendNoise.theta', 'cartpendNoise.thetadot',
#             'kalmanEstimator.x_h', 'kalmanEstimator.xdot_h', 'kalmanEstimator.theta_h', 'kalmanEstimator.thetadot_h')
# )


# LQR control
Q = np.array([[10, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 20]], dtype=np.float)
R = 0.01
K, _, _ = control.lqr(sysLin.A, sysLin.B, Q, R)
controlLQR = control.NonlinearIOSystem(
    None,
    lambda t, x, u, params: -K @ (np.array([u[0], u[1], u[2], u[3]]) - np.array(xeq)),
    inputs=('x', 'xdot', 'theta', 'thetadot'), outputs=('f'), name='controlLQR')

io_closed = control.InterconnectedSystem(
    (sysFull, sysKF, controlLQR),
    connections=(
        ('sysFull.u', 'controlLQR.f'),
        # ('sysPartNoise.u', 'controlLQR.f'),
        ('sysKF.u', 'controlLQR.f'),
        # ('sysKF.s', 'sysPartNoise.x'),
        ('sysKF.s', 'sysFull.x'),
        ('controlLQR.x', 'sysKF.x_h'),
        ('controlLQR.xdot', 'sysKF.xdot_h'),
        ('controlLQR.theta', 'sysKF.theta_h'),
        ('controlLQR.thetadot', 'sysKF.thetadot_h')
    ),
    inplist=[],
    outlist=('sysFull.x', 'sysFull.xdot', 'sysFull.theta', 'sysFull.thetadot',
             'sysKF.x_h', 'sysKF.xdot_h', 'sysKF.theta_h', 'sysKF.thetadot_h',
             'controlLQR.f')
)

T = np.arange(tspan[0], tspan[1], dt)
# u0 = np.zeros([1, T.shape[0]], dtype=np.float)
# u0[:, 100: 110] = 20  # impulse
# u0[:, 400: 420] = -20  # impulse

y0 = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
t0, yout0 = control.input_output_response(io_closed, T, X0=y0)
# t, yout = control.input_output_response(io_Estimator, np.arange(tspan[0], tspan[1], dt), X0=y0)
# t1, yout1 = control.input_output_response(sysFull, T, U=u0, X0=y0)
# t2, yout2 = control.input_output_response(sysPartNoise, T, U=u0, X0=y0)


# plt.plot(T, yout2, "k-")
plt.plot(T, yout0[0, :], "y-")
plt.plot(T, yout0[1, :], "r-")
plt.plot(T, yout0[2, :], "g-")
plt.plot(T, yout0[3, :], "b-")
plt.plot(T, yout0[4, :], "r--")
plt.plot(T, yout0[5, :], "g--")
plt.plot(T, yout0[6, :], "b--")
plt.plot(T, yout0[7, :], "y--")
plt.show()

##############################################
# GUI simulated animation
yout = yout0[:4, :]
gui = ti.GUI('Sim cartpend', res=(512, 512), background_color=0xdddddd)
t_curr = 0
while gui.running:
    drawcartpend(gui, yout[:, t_curr], m, M, L)
    gui.show()
    t_curr += 1
    if (t_curr > yout.shape[1] - 1):
        print(yout[:, t_curr - 1])
        t_curr = 0
