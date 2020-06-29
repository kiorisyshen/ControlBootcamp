import math
import numpy as np


class cartpendSystem:
    def __init__(self, m, M, L, g, d, u):
        self.m_ = m
        self.M_ = M
        self.L_ = L
        self.g_ = g
        self.d_ = d
        self.u_ = u

    def cartpend(self, t, y):
        m = self.m_
        M = self.M_
        L = self.L_
        g = self.g_
        d = self.d_
        u = self.u_

        SinTheta = math.sin(y[2])
        CosTheta = math.cos(y[2])

        x_dot = y[1]
        theta_dot = y[3]

        # x_ddot = u(t, y) + m * g * SinTheta * CosTheta + m * L * y[3] * y[3] * SinTheta
        # x_ddot = x_ddot / (M + m - m * CosTheta * CosTheta)

        x_ddot = (L * m * SinTheta * (theta_dot)**2 - g * m * CosTheta * SinTheta + u(t, y)) / (m * SinTheta**2 + M)

        theta_ddot = -(CosTheta * (x_ddot) - g * SinTheta) / L

        damping_theta = -0.1 * d * y[3]
        damping_x = -d * y[1]

        dy = np.array([0, 0, 0, 0], dtype=np.float)
        dy[0] = y[1]
        dy[1] = x_ddot + damping_x
        dy[2] = y[3]
        dy[3] = theta_ddot + damping_theta

        return dy
