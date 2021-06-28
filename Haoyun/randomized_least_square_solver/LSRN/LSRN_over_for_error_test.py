from math import ceil, sqrt
import numpy as np
from scipy.sparse.linalg import LinearOperator
from numpy.linalg import svd
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy


def LSRN_over_for_error_test(A, b, tol=1e-8, gamma=2, iter_lim=1000):
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

    #####################################################
    # Incorporate the sketching method into the sketch.py
    #####################################################

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

        A_tilde = A_tilde * 1000

        U_tilde, Sigma_tilde, VH_tilde = svd(A_tilde, False)
        # t = U_tilde.dtype.char.lower()

        # determine the new rank
        rcond = Sigma_tilde[0] * np.min(A.shape) * np.finfo(float).eps
        r_tol = rcond

        r = np.sum(Sigma_tilde > r_tol)
        # print('\t Dropped rank by %s' % (n - r))

        N = VH_tilde[:r, :].T / Sigma_tilde[:r]

        def LSRN_matvec(v):
            return A.dot(N.dot(v))

        def LSRN_rmatvec(v):
            return N.T.dot(A.T.dot(v))

        # re-estimate gamma
        gamma_new = s / r
        # estimate the condition number of AN
        cond_AN = (sqrt(gamma_new) + 1) / (sqrt(gamma_new) - 1)

        AN = LinearOperator(shape=(m, r), matvec=LSRN_matvec, rmatvec=LSRN_rmatvec)
        result = lsqr_copy(AN, b, atol=tol / cond_AN, btol=tol / cond_AN, iter_lim=iter_lim)

        y = result[0]
        flag = result[1]
        itn = result[2]

        absolute_residual_error_array = result[-1]
        absolute_normal_equation_error_array = result[-2]
        relative_residual_error_array = result[-3]
        relative_normal_equation_error_array = result[-4]
        relative_error_array = result[-5]

        x = N.dot(y)
    else:

        print("The under-determined case is not implemented.")

    return x, itn, flag, r, relative_error_array, relative_normal_equation_error_array, relative_residual_error_array, \
           absolute_normal_equation_error_array, absolute_residual_error_array,
