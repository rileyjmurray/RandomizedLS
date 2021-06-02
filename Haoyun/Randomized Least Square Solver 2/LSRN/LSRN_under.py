# I have not modified this function, and there are some bugs with this function.

# This algorithm is used when m << n, which is for under-determined system.
import math
import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import lsqr


def LSRN_under(A, b):
    m = A.shape[0]
    n = A.shape[1]
    # oversample factor gamma
    gamma = 2
    s = int(math.ceil(gamma * m))
    G = np.random.normal(0, 1, (n, s))
    A_new = A @ G
    U_new, s_new, Vh_new = svd(A_new, full_matrices=False)
    M = U_new @ np.linalg.inv(np.diag(s_new))
    x_new = lsqr(M.transpose() @ A, M.transpose() @ b)[0]
    return x_new
