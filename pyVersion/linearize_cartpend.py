import numpy as np
import math
from scipy.integrate import solve_ivp
import control

from drawcartpend import drawcartpend
from cartpend import cartpendSystem

import taichi as ti
ti.init(debug=True)


m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = 9.8  # gravity
d = 4.0  # dampping term

#############################
# Linearized system and place pole
A = np.array([
    [0, 1, 0, 0],
    [0, -d, -(g * m) / M, 0],
    [0, 0, 0, 1],
    [0, 0, (g * m + M * g) / (L * M), -0.1 * d]
], dtype=np.float)

B = np.array([[0], [1.0 / M], [0], [-1.0 / (L * M)]], dtype=np.float)

print("eig A", np.linalg.eig(A))
C = control.ctrb(A, B)
print("ctrb: ", C)
print("rank(ctrb): ", np.linalg.matrix_rank(C))


def zeroForce(t, y):
    return 0.0
#############################
# sys = cartpendSystem(m, M, L, g, d, zeroForce)


def linearizedSys(t, y):
    B_tmp = np.array([B[0, 0], B[1, 0], B[2, 0], B[3, 0]], dtype=np.float)
    return A.dot(np.transpose(y)) + B_tmp * zeroForce(t, y)


tspan = [0, 30]  # time for simulation
dt = 0.03
y0 = np.array([0, 0, 0 + 0.001, 0], dtype=np.float)

gui = ti.GUI('Sim cartpend', res=(512, 512), background_color=0xdddddd)

# sol = solve_ivp(sys.cartpend, tspan, y0, t_eval=np.arange(tspan[0], tspan[1], dt))
sol = solve_ivp(linearizedSys, tspan, y0, t_eval=np.arange(tspan[0], tspan[1], dt))

t_curr = 0
while gui.running:
    drawcartpend(gui, sol.y[:, t_curr], m, M, L)
    gui.show()
    t_curr += 1
    if (t_curr > sol.y.shape[1] - 1):
        print(sol.y[:, t_curr - 1])
        t_curr = 0
