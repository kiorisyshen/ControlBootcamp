from drawcartpend import drawcartpend
import numpy as np
import math

import taichi as ti
ti.init(debug=True)

m = 1.0  # mass of ball
M = 5.0  # mass of cart
L = 2.0  # pendulum length
g = -10.0  # gravity
d = 1.0  # dampping term

tmax = 10.0  # max time for simulation
y0 = np.array([[0.5],
               [0],
               [0],
               [0.5]], dtype=np.float)

gui = ti.GUI('HW Mass Spring System', res=(512, 512), background_color=0xdddddd)


while gui.running:
    drawcartpend(gui, y0, m, M, L)
    gui.show()
