import numpy as np
import math
from scipy.integrate import solve_ivp

from drawcartpend import drawcartpend
from cartpend import cartpend
from cartpend import setCartpendVars

import taichi as ti
ti.init(debug=True)

m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = 10.0  # gravity
d = 1.0  # dampping term

tspan = [0, 30]  # time for simulation
dt = 0.05
y0 = np.array([0, 0, -0.5 * math.pi, 0.1], dtype=np.float)


gui = ti.GUI('Sim cartpend', res=(512, 512), background_color=0xdddddd)

sol = solve_ivp(cartpend, tspan, y0, t_eval=np.arange(tspan[0], tspan[1], dt))


t_curr = 0
while gui.running:
    drawcartpend(gui, sol.y[:, t_curr], m, M, L)
    gui.show()
    t_curr += 1
    if (t_curr > sol.y.shape[1] - 1):
        t_curr = 0
