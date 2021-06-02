from scipy.sparse.linalg import aslinearoperator
import numpy as np
from numpy.linalg import norm
from math import sqrt


def LSQR(A, b, tol=1e-14, iter_lim=None):
    """
    A simple version of LSQR
    """

    A = aslinearoperator(A)
    m, n = A.shape

    eps = 32 * np.finfo(float).eps;  # slightly larger than eps

    if tol < eps:
        tol = eps
    elif tol >= 1:
        tol = 1 - eps

    u = b.squeeze().copy()
    beta = norm(u)
    if beta != 0:
        u /= beta

    v = A.rmatvec(u)
    alpha = norm(v)
    if alpha != 0:
        v /= alpha

    w = v.copy()

    x = np.zeros(n)

    phibar = beta
    rhobar = alpha

    nrm_a = 0.0
    cnd_a = 0.0
    sq_d = 0.0
    nrm_r = beta
    nrm_ar_0 = alpha * beta

    if nrm_ar_0 == 0:  # alpha == 0 || beta == 0
        return x, 0, 0

    nrm_x = 0
    sq_x = 0
    z = 0
    cs2 = -1
    sn2 = 0

    max_n_stag = 3
    stag = 0

    flag = -1
    if iter_lim is None:
        iter_lim = np.max([20, 2 * np.min([m, n])])

    for itn in range(int(iter_lim)):

        u = A.matvec(v) - alpha * u
        beta = norm(u)
        u /= beta

        # estimate of norm(A)
        nrm_a = sqrt(nrm_a ** 2 + alpha ** 2 + beta ** 2)

        v = A.rmatvec(u) - beta * v
        alpha = norm(v)
        v /= alpha

        rho = sqrt(rhobar ** 2 + beta ** 2)
        cs = rhobar / rho
        sn = beta / rho
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar

        x += (phi / rho) * w
        w = v - (theta / rho) * w

        # estimate of norm(r)
        nrm_r = phibar

        # estimate of norm(A'*r)
        nrm_ar = phibar * alpha * np.abs(cs)

        # check convergence
        if nrm_ar < tol * nrm_ar_0:
            flag = 0
            break

        if nrm_ar < eps * nrm_a * nrm_r:
            flag = 0
            break

        # estimate of cond(A)
        sq_w = np.dot(w, w)
        nrm_w = sqrt(sq_w)
        sq_d += sq_w / (rho ** 2)
        cnd_a = nrm_a * sqrt(sq_d)

        # check condition number
        if cnd_a > 1 / eps:
            flag = 1
            break

        # check stagnation
        if abs(phi / rho) * nrm_w < eps * nrm_x:
            stag += 1
        else:
            stag = 0
        if stag >= max_n_stag:
            flag = 1
            break

        # estimate of norm(x)
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        nrm_x = sqrt(sq_x + zbar ** 2)
        gamma = sqrt(gambar ** 2 + theta ** 2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        sq_x += z ** 2

    return x, flag, itn
