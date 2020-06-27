import numpy as np
import math
from scipy.integrate import solve_ivp

from drawcartpend import drawcartpend
from cartpend import cartpendSystem

import taichi as ti
ti.init(debug=True)


def zeroForce(t, y):
    return 0.0


def uImplus(t, y):
    if (10.0 <= t < 10.3):
        return 12.0
    elif (14.0 <= t < 14.3):
        return -18.0
    else:
        return 0.0


m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = -10.0  # gravity
d = 4.0  # dampping term
sys = cartpendSystem(m, M, L, g, d, zeroForce)

tspan = [0, 30]  # time for simulation
dt = 0.03
y0 = np.array([0, 0, 0.5*math.pi, 0], dtype=np.float)

gui = ti.GUI('Sim cartpend', res=(512, 512), background_color=0xdddddd)

sol = solve_ivp(sys.cartpend, tspan, y0, t_eval=np.arange(tspan[0], tspan[1], dt))


t_curr = 0
while gui.running:
    drawcartpend(gui, sol.y[:, t_curr], m, M, L)
    gui.show()
    t_curr += 1
    if (t_curr > sol.y.shape[1] - 1):
        t_curr = 0
