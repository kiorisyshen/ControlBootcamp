import numpy as np
import math
from scipy.integrate import solve_ivp
import control

from drawcartpend import drawcartpend
from cartpend import cartpendSystem

import taichi as ti
ti.init(debug=True)

control.use_numpy_matrix(flag=False)

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

C = np.array([1, 0, 0, 0], dtype=np.float)  # only observable if x measured... because x can't be reconstructed

O = control.obsv(A, C)
print("Obsv A,C: ", O)
print("rank(obsv): ", np.linalg.matrix_rank(O))

print('----------')
print('Which measurements are best if we omit "x"')
A = A[1:, 1:]
B = B[1:]

C = np.array([1, 0, 0], dtype=np.float)
print('C: ', C)
O = control.obsv(A, C)
print("Obsv A,C: ", O)
print("rank(obsv): ", np.linalg.matrix_rank(O))

C = np.array([0, 1, 0], dtype=np.float)
print('C: ', C)
O = control.obsv(A, C)
print("Obsv A,C: ", O)
print("rank(obsv): ", np.linalg.matrix_rank(O))

C = np.array([0, 0, 1], dtype=np.float)
print('C: ', C)
O = control.obsv(A, C)
print("Obsv A,C: ", O)
print("rank(obsv): ", np.linalg.matrix_rank(O))

# print('----------')
# print('sys gram')
# D = np.zeros((1, B.shape[1]), dtype=np.float)
# sys = control.StateSpace(A, B, C, D)
# control.gram(sys, 'o')
