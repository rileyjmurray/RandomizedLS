import numpy as np
import math


# This is Chebyshev semi-iterative (CS) method

def CS(A, b):
    epsilon = 10 ** -14
    m = A.shape[0]
    n = A.shape[1]
    U, s, Vh = np.linalg.svd(A)
    sigmaL = min([i for i in s if i > 0])
    sigmaU = max([i for i in s if i > 0])
    d = (sigmaU ** 2 + sigmaL ** 2) / 2
    c = (sigmaU ** 2 - sigmaL ** 2) / 2
    x = np.zeros(n)
    v = np.zeros(n)
    r = b

    iternum = 0
    beta, alpha = 0, 0
    for k in range(math.ceil((math.log(epsilon) - math.log(2)) / math.log((sigmaU - sigmaL) / (sigmaU + sigmaL))) + 1):
        # while np.linalg.norm(A @ x - b,ord =2) >= 1:
        if k == 0:
            beta = 0
            alpha = 1 / d
        elif k == 1:
            beta = 1 / 2 * (c / d) ** 2
            alpha = d - c ** 2 / (2 * d)
        else:
            alpha = 1 / (d - alpha * c ** 2 / 4)
            beta = (alpha * c / 2) ** 2

        v = beta * v + A.transpose() @ r
        x = x + alpha * v
        r = r - alpha * A @ v
        iternum = iternum + 1
    return x, iternum
