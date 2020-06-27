import math
import numpy as np


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
d = 1.0  # dampping term
u = uImplus  # force to the cart


def setCartpendVars(m_, M_, L_, g_, d_, u_):
    m = m_
    M = M_
    L = L_
    g = g_
    d = d_
    u = u_


# Sy = sin(y(3))
# Cy = cos(y(3))
# D = m*L*L*(M+m*(1-Cy ^ 2))

# dy(1, 1) = y(2)
# dy(2, 1) = (1/D)*(-m ^ 2*L ^ 2*g*Cy*Sy + m*L ^ 2*(m*L*y(4) ^ 2*Sy - d*y(2))) + m*L*L*(1/D)*u
# dy(3, 1) = y(4)
# dy(4, 1) = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*y(4) ^ 2*Sy - d*y(2))) - m*L*Cy*(1/D)*u + .01*randn

# def cartpend(t, y):
#     Sy = math.sin(y[2])
#     Cy = math.cos(y[2])
#     D = m*L*L*(M+m*(1.0-Cy*Cy))

#     dy = np.array([0, 0, 0, 0], dtype=np.float)
#     dy[0] = y[1]
#     dy[1] = (1.0/D)*(-m*m*L*L*g*Cy*Sy + m*L*L*(m*L*y[3]*y[3]*Sy - d*y[1])) + m*L*L*(1.0/D)*u(t, y)
#     dy[2] = y[3]
#     dy[3] = (1.0/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*y[3]*y[3]*Sy - d*y[1])) - m*L*Cy*(1.0/D)*u(t, y)

#     return dy


def cartpend(t, y):
    x_ddot = u(t, y) - m * L * y[3] * y[3] * np.cos(y[2]) + m * g * np.cos(y[2]) * np.sin(y[2])
    x_ddot = x_ddot / (M + m - m * np.sin(y[2]) * np.sin(y[2]))

    theta_ddot = -g / L * np.cos(y[2]) - 1. / L * np.sin(y[2]) * x_ddot

    damping_theta = -0.7 * d * y[3]
    damping_x = - d * y[1]

    dy = np.array([0, 0, 0, 0], dtype=np.float)
    dy[0] = y[1]
    dy[1] = x_ddot + damping_x
    dy[2] = y[3]
    dy[3] = theta_ddot + damping_theta

    return dy
