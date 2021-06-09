from math import ceil, sqrt

import numpy as np
from scipy.sparse.linalg import lsqr, LinearOperator
from numpy.linalg import svd


def LSRN_over(A, b, tol=1e-8, gamma=2, iter_lim=1000):
    """
    LSRN computes the min-length solution of linear least squares via LSQR with
    randomized preconditioning

    Parameters
    ----------

    A       : {matrix, sparse matrix, ndarray, LinearOperator} of size m-by-n

    b       : (m,) ndarray

    gamma : float (>1), oversampling factor

    tol : float, tolerance such that norm(A*x-A*x_opt)<tol*norm(A*x_opt)

    iter_lim : integer, the max iteration number

    rcond : float, reciprocal of the condition number

    Returns
    -------
    x      : (n,) ndarray, the min-length solution

    itn : int, iteration number

    r      : int, the rank of A

    flag : int,
    """
    m, n = A.shape

    # Incorporate the sketching method into the sketch.py

    if m > n:  # over-determined

        s = ceil(gamma * n)
        A_tilde = np.zeros([s, n])
        blk_sz = 128
        for i in range(int(ceil(1.0 * s / blk_sz))):
            blk_begin = i * blk_sz
            blk_end = np.min([(i + 1) * blk_sz, s])
            blk_len = blk_end - blk_begin
            G = np.random.randn(blk_len, m)
            A_tilde[blk_begin:blk_end, :] = G.dot(A)

        # print(np.linalg.cond(A_tilde))
        A_tilde = A_tilde * 1000
        # print(np.linalg.cond(A_tilde))

        U_tilde, Sigma_tilde, VH_tilde = svd(A_tilde, False)
        # t = U_tilde.dtype.char.lower()

        rcond = Sigma_tilde[0] * np.min(A.shape) * np.finfo(float).eps

        # rcond = np.min([m, n]) * np.finfo(float).eps

        # determine the rank
        # r = VH_tilde.shape[1]

        r_tol = rcond
        # print(r_tol)
        r = np.sum(Sigma_tilde > r_tol)
        # print(Sigma_tilde)
        # print('\t Dropped rank by %s' % (n - r))
        N = VH_tilde[:r, :].T / Sigma_tilde[:r]

        def LSRN_matvec(v):
            return A.dot(N.dot(v))

        def LSRN_rmatvec(v):
            return N.T.dot(A.T.dot(v))

        # reestimate gamma
        gamma_new = s / r
        # estimate the condition number of AN
        cond_AN = (sqrt(gamma_new) + 1) / (sqrt(gamma_new) - 1)

        AN = LinearOperator(shape=(m, r), matvec=LSRN_matvec, rmatvec=LSRN_rmatvec)
        result = lsqr(AN, b, atol=tol / cond_AN, btol=tol / cond_AN, iter_lim=iter_lim)

        y = result[0]
        flag = result[1]
        itn = result[2]
        # r1norm = result[3]
        # r2norm = result[4]
        # anorm = result[5]
        # acond = result[6]
        # arnorm = result[7]
        relative_residual_error_array_array = result[-1]
        relative_normal_equation_error_array = result[-2]
        x = N.dot(y)
    else:

        print("The under-determined case is not implemented.")

    # return x, itn, flag, r1norm, r2norm, anorm, acond, arnorm, r
    return x, itn, flag, r, relative_normal_equation_error_array, relative_residual_error_array_array
    # return x, itn, flag, r
