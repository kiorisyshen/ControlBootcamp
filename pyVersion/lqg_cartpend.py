import control
import numpy as np
import matplotlib.pyplot as plt

from drawcartpend import drawcartpend
from cartpend import cartpendSys_pyCtl

import taichi as ti
ti.init(debug=True)

control.use_numpy_matrix(flag=False)

m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = 9.8  # gravity
d = 4.0  # dampping term
sysParams = {
    'm': m,  # mass of ball
    'M': M,  # mass of cart
    'L': L,  # pendulum length
    'g': g,  # gravity
    'd': d   # dampping term
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
                                    name='cartpendFull')

# linearize on up position
sysLin = sysFull.linearize([0.0, 0.0, 0.0, 0.0], 0.0, params=sysParams)
# LQR control
Q = np.array([[10, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 20]], dtype=np.float)
R = 0.01
K, _, _ = control.lqr(sysLin.A, sysLin.B, Q, R)
io_controller = control.NonlinearIOSystem(
    None,
    lambda t, x, u, params: -K @ (np.array([u[0], u[1], u[2], u[3]]) - np.array(xeq)),
    inputs=('x', 'xdot', 'theta', 'thetadot'), outputs=('f'), name='controlLQR')

io_closed = control.InterconnectedSystem(
    (sysFull, io_controller),
    connections=(
        ('cartpendFull.u', 'controlLQR.f'),
        ('controlLQR.x', 'cartpendFull.x'),
        ('controlLQR.xdot', 'cartpendFull.xdot'),
        ('controlLQR.theta', 'cartpendFull.theta'),
        ('controlLQR.thetadot', 'cartpendFull.thetadot')
    ),
    inplist=[],
    outlist=('cartpendFull.x', 'cartpendFull.xdot', 'cartpendFull.theta', 'cartpendFull.thetadot', 'controlLQR.f')
)

t, yout = control.input_output_response(io_closed, np.arange(tspan[0], tspan[1], dt), X0=y0)

plt.plot(t, yout[0, :], "y-")
plt.plot(t, yout[1, :], "r-")
plt.plot(t, yout[2, :], "g-")
plt.plot(t, yout[3, :], "b-")
plt.show()

##############################################
# GUI simulated animation
gui = ti.GUI('Sim cartpend', res=(512, 512), background_color=0xdddddd)
t_curr = 0
while gui.running:
    drawcartpend(gui, yout[:, t_curr], m, M, L)
    gui.show()
    t_curr += 1
    if (t_curr > yout.shape[1] - 1):
        print(yout[:, t_curr - 1])
        t_curr = 0
