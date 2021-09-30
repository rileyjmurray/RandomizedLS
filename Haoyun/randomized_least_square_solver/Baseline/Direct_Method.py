# This file contains the direct method to solve least square, and may be used as a baseline
# for tuning purpose.
import numpy as np
from numpy.linalg import inv


# This is the method that uses QR to solve least square.
def solve_ls_with_QR(A, b):
    Q, R = np.linalg.qr(A, mode='reduced')
    Qb = np.matmul(Q.T, b)
    x_qr = np.linalg.solve(R, Qb)
    return x_qr


# This is the method that uses normal equation to solve least square.
def solve_ls_with_normal_equation(A, b):
    ATb = np.matmul(A.T, b)
    x = np.linalg.solve(np.matmul(A.T, A), ATb)
    return x
