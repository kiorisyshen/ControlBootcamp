import control
import numpy as np
import matplotlib.pyplot as plt

from drawcartpend import drawcartpend
from cartpend import cartpendSys_pyCtl

import taichi as ti
ti.init(debug=True)

m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = 9.8  # gravity
d = 4.0  # dampping term
Vd = 0.001 * np.eye(4)
Vn = 0.001
sysParams = {
    'm': m,  # mass of ball
    'M': M,  # mass of cart
    'L': L,  # pendulum length
    'g': g,  # gravity
    'd': d,   # dampping term
}

tspan = [0, 10]  # time for simulation
dt = 0.01
yInit = [0.0, 0.0, 0.0, 0.1]  # initial state
xeq = [0.3, 0.0, 0.0, 0.0]  # final stable state

# Build non-linear standard system
sysFullNonLin = control.NonlinearIOSystem(cartpendSys_pyCtl, None, inputs=('u'),
                                          outputs=('x', 'xdot', 'theta', 'thetadot'),
                                          states=('x', 'xdot', 'theta', 'thetadot'),
                                          params=sysParams,
                                          name='sysFullNonLin')
sysLin = sysFullNonLin.linearize([0.0, 0.0, 0.0, 0.0], 0.0, params=sysParams)  # linearize on up position

A = sysLin.A
B = sysLin.B
BF = np.concatenate([B, Vd, 0.0 * B], axis=1)  # augment inputs to include disturbance and noise
C_full = sysLin.C
C_part = np.array([1, 0, 0, 0], dtype=np.float)


ssFull = control.StateSpace(A, BF, C_full, np.zeros(BF.shape))  # system with full state output, disturbance, no noise
sysFull = control.LinearIOSystem(ssFull, inputs=('u', 'vd0', 'vd1', 'vd2', 'vd3', 'vn'),
                                 outputs=('x', 'xdot', 'theta', 'thetadot'),
                                 states=('x', 'xdot', 'theta', 'thetadot'),
                                 name='sysFull')

ssPart = control.StateSpace(A, BF, C_part, np.array([0, 0, 0, 0, 0, Vn]))  # build big state space system... with single output

# Kalman estimator
# Kf, P, E = control.lqe(A, Vd, C, Vd, Vn)  # design Kalman filter
Kf, _, _ = control.lqr(np.transpose(A), np.expand_dims(C_part, axis=1), Vd, Vn)   # alternatively, possible to design using "LQR" code
Kf = np.transpose(Kf)

ssKF = control.StateSpace(A - Kf * C_part, np.concatenate([B, Kf], axis=1), np.eye(4), 0 * np.concatenate([B, Kf], axis=1))  # Kalman filter estimator

# Combine sysPart and sysKF
ssPartKF = control.connect(control.append(ssPart, ssKF), [[8, 1]], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5])
sysPartKF = control.LinearIOSystem(ssPartKF, inputs=('u', 'vd0', 'vd1', 'vd2', 'vd3', 'vn', 'uKF'),
                                   outputs=('x_n', 'x_h', 'xdot_h', 'theta_h', 'thetadot_h'),
                                   name='sysPartKF')


def compareKalmanAndSys():
    # test kalman estimator & compare non-linear system and linear system
    T = np.arange(tspan[0], tspan[1], dt)
    uDIST = np.random.multivariate_normal(np.zeros(4), Vd, size=T.shape[0])
    uDIST = np.transpose(uDIST)
    uNOISE = np.random.normal(0.0, Vn, size=(1, T.shape[0]))
    u0 = np.zeros([1, T.shape[0]], dtype=np.float)
    u0[:, 100: 110] = 20  # impulse
    u0[:, 400: 420] = -20  # impulse
    uAUG = np.concatenate([u0, uDIST, uNOISE, u0])

    y0 = []
    y0.extend(yInit)
    y0.extend(yInit)
    t0, yout0 = control.input_output_response(sysPartKF, T, U=uAUG, X0=y0)
    t1, yout1 = control.input_output_response(sysFullNonLin, T, U=u0, X0=yInit)
    t2, yout2 = control.input_output_response(sysFull, T, U=np.concatenate([u0, uDIST, uNOISE]), X0=yInit)

    plt.plot(T, yout0[0, :], "k-", label="noise observation of x")
    plt.plot(T, yout0[1, :], "r-", label="kalman estimation of x")
    plt.plot(T, yout0[2, :], "g-", label="kalman estimation of x_dot")
    plt.plot(T, yout0[3, :], "b-", label="kalman estimation of theta")
    plt.plot(T, yout0[4, :], "y-", label="kalman estimation of theta_dot")
    plt.plot(T, yout1[0, :], "y--", label="non-linear sys state of x")
    plt.plot(T, yout1[1, :], "r--", label="non-linear sys state of x_dot")
    plt.plot(T, yout1[2, :], "g--", label="non-linear sys state of theta")
    plt.plot(T, yout1[3, :], "b--", label="non-linear sys state of theta_dot")
    plt.plot(T, yout2[0, :], "y:", label="linear sys state of x")
    plt.plot(T, yout2[1, :], "r:", label="linear sys state of x_dot")
    plt.plot(T, yout2[2, :], "g:", label="linear sys state of theta")
    plt.plot(T, yout2[3, :], "b:", label="linear sys state of theta_dot")
    plt.title('Compare non-linear system, linear system and kalman estimator on noise linear system')
    plt.legend(bbox_to_anchor=(-0.05, -0.5, 1.05, 0.25), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)
    plt.subplots_adjust(bottom=0.3)
    plt.show()


# LQR control
Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 20]], dtype=np.float)
R = 0.001
K, _, _ = control.lqr(A, B, Q, R)
controlLQR = control.NonlinearIOSystem(
    None,
    lambda t, x, u, params: -K @ (np.array([u[0], u[1], u[2], u[3]]) - np.array(xeq)),
    inputs=('x', 'xdot', 'theta', 'thetadot'), outputs=('f'), name='controlLQR')

io_closed = control.InterconnectedSystem(
    # (sysFull, controlLQR),
    # (sysPartKF, controlLQR),
    # (sysFullNonLin, sysPartKF, controlLQR),
    (sysFull, sysPartKF, controlLQR),
    connections=(
        # ('sysFullNonLin.u', 'controlLQR.f'),
        ('sysPartKF.u', 'controlLQR.f'),
        ('sysPartKF.uKF', 'controlLQR.f'),
        ('controlLQR.x', 'sysPartKF.x_h'),
        ('controlLQR.xdot', 'sysPartKF.xdot_h'),
        ('controlLQR.theta', 'sysPartKF.theta_h'),
        ('controlLQR.thetadot', 'sysPartKF.thetadot_h'),
        ('sysFull.u', 'controlLQR.f'),
        # ('controlLQR.x', 'sysFull.x'),
        # ('controlLQR.xdot', 'sysFull.xdot'),
        # ('controlLQR.theta', 'sysFull.theta'),
        # ('controlLQR.thetadot', 'sysFull.thetadot')
    ),
    inplist=[
        'sysPartKF.vd0', 'sysPartKF.vd1', 'sysPartKF.vd2', 'sysPartKF.vd3', 'sysPartKF.vn',
        'sysFull.vd0', 'sysFull.vd1', 'sysFull.vd2', 'sysFull.vd3', 'sysFull.vn'],
    # outlist=('sysPartKF.x_h', 'sysPartKF.xdot_h', 'sysPartKF.theta_h', 'sysPartKF.thetadot_h')
    # outlist=('sysFull.x', 'sysFull.xdot', 'sysFull.theta', 'sysFull.thetadot')
    outlist=('sysFull.x', 'sysFull.xdot', 'sysFull.theta', 'sysFull.thetadot')
)


def evalClosedFeedback():
    T = np.arange(tspan[0], tspan[1], dt)
    uDIST = np.random.multivariate_normal(np.zeros(4), Vd, size=T.shape[0])
    uDIST = np.transpose(uDIST)
    uNOISE = np.random.normal(0.0, Vn, size=(1, T.shape[0]))

    y0 = []
    y0.extend(yInit)
    y0.extend(yInit)
    y0.extend(yInit)
    t0, yout0 = control.input_output_response(io_closed, T, U=np.concatenate([Vd@Vd@uDIST, Vn*uNOISE, Vd@Vd@uDIST, Vn*uNOISE]), X0=y0)

    # plt.plot(T, yout0[0, :], "k-", label="noise observation of x")
    # plt.plot(T, yout0[1, :], "r-", label="kalman estimation of x")
    # plt.plot(T, yout0[2, :], "g-", label="kalman estimation of x_dot")
    # plt.plot(T, yout0[3, :], "b-", label="kalman estimation of theta")
    # plt.plot(T, yout0[4, :], "y-", label="kalman estimation of theta_dot")
    # plt.plot(T, yout0[5, :], "y--", label="non-linear sys state of x")
    # plt.plot(T, yout0[6, :], "r--", label="non-linear sys state of x_dot")
    # plt.plot(T, yout0[7, :], "g--", label="non-linear sys state of theta")
    # plt.plot(T, yout0[8, :], "b--", label="non-linear sys state of theta_dot")
    # plt.plot(T, yout0[9, :], "k:", label="LQR control u")
    # plt.title('Close loop evaluation')
    # plt.legend(bbox_to_anchor=(-0.05, -0.4, 1.05, 0.25), loc='lower left',
    #            ncol=3, mode="expand", borderaxespad=0.)
    # plt.subplots_adjust(bottom=0.3)
    # plt.show()

    ##############################################
    # GUI simulated animation
    yout = yout0[:, :]
    gui = ti.GUI('Sim cartpend', res=(512, 512), background_color=0xdddddd)
    t_curr = 0
    while gui.running:
        drawcartpend(gui, yout[:, t_curr], m, M, L)
        gui.show()
        t_curr += 1
        if (t_curr > yout.shape[1] - 1):
            print(yout[:, t_curr - 1])
            t_curr = 0


evalClosedFeedback()
