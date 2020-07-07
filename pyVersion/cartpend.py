import math
import numpy as np
import control


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


def cartpendSys_pyCtl(t, x, u, params):
    # Parameter setup
    m = params.get('m', 1.0)
    M = params.get('M', 5.0)
    L = params.get('L', 2.0)
    g = params.get('g', 9.8)
    d = params.get('d', 4.0)

    SinTheta = math.sin(x[2])
    CosTheta = math.cos(x[2])

    x_dot = x[1]
    theta_dot = x[3]

    F_cart = u

    x_ddot = (L * m * SinTheta * (theta_dot)**2 - g * m * CosTheta * SinTheta + F_cart) / (m * SinTheta**2 + M)

    theta_ddot = -(CosTheta * (x_ddot) - g * SinTheta) / L

    damping_theta = -0.1 * d * x[3]
    damping_x = -d * x[1]

    return [x[1], x_ddot + damping_x, x[3], theta_ddot + damping_theta]


def cartpendSysNoise_pyCtl(t, x, u, params):
    # Parameter setup
    m = params.get('m', 1.0)
    M = params.get('M', 5.0)
    L = params.get('L', 2.0)
    g = params.get('g', 9.8)
    d = params.get('d', 4.0)
    Vd = params.get('Vd')

    SinTheta = math.sin(x[2])
    CosTheta = math.cos(x[2])

    x_dot = x[1]
    theta_dot = x[3]

    F_cart = u

    x_ddot = (L * m * SinTheta * (theta_dot)**2 - g * m * CosTheta * SinTheta + F_cart) / (m * SinTheta**2 + M)

    theta_ddot = -(CosTheta * (x_ddot) - g * SinTheta) / L

    damping_theta = -0.1 * d * x[3]
    damping_x = -d * x[1]

    nDIST = np.random.multivariate_normal(np.zeros(4), Vd)
    distD = Vd @ nDIST

    return [x[1] + distD[0], x_ddot + damping_x + distD[1], x[3] + distD[2], theta_ddot + damping_theta + distD[3]]
    # return [x[1], x_ddot + damping_x, x[3], theta_ddot + damping_theta]
