from math import ceil
import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator
from numpy.linalg import svd, norm
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR_control_precision import lsqr_copy_control_precision


def LSRN_over_control_precision(A, b, tol=1e-8, gamma=2, iter_lim=1000):
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
    old_A, old_b = A, b
    A, b = np.single(A), np.single(b)

    # print("\tThe 2-norm difference of different precision A is:", norm(A - old_A, 2))
    # print("\tThe 2-norm difference of different precision b is:", norm(b - old_b, 2))

    m, n = A.shape

    if m > n:  # over-determined
        s = ceil(gamma * n)

        A_tilde = np.zeros([s, n]).astype('float32')
        blk_sz = 128
        for i in range(int(ceil(1.0 * s / blk_sz))):
            blk_begin = i * blk_sz
            blk_end = np.min([(i + 1) * blk_sz, s])
            blk_len = blk_end - blk_begin
            G = np.random.randn(blk_len, m).astype('float32')
            A_tilde[blk_begin:blk_end, :] = G.dot(A)
        # A_tilde = A_tilde * 1000

        # def apply_guassian_normal_distribution(mat):
        #     mat = np.random.normal(loc=0, scale=1, size=(s, m)) @ mat
        #     return np.double(mat)
        #
        # G = LinearOperator(shape=(s, m), matvec=apply_guassian_normal_distribution,
        #                    matmat=apply_guassian_normal_distribution)

        # A_tilde = G.dot(A)
        U_tilde, Sigma_tilde, VH_tilde = svd(A_tilde, False)
        # print(U_tilde.dtype)
        # print(Sigma_tilde.dtype)
        rcond = Sigma_tilde[0] * np.min(A.shape) * np.finfo('float32').eps
        r_tol = rcond
        r = np.sum(Sigma_tilde > r_tol)

        N = VH_tilde[:r, :].T / Sigma_tilde[:r]
        # print(N.dtype)
        def LSRN_matvec(v):
            v = np.single(v)
            return A.dot(N.dot(v))

        def LSRN_rmatvec(v):
            v = np.single(v)
            return N.T.dot(A.T.dot(v))

        # def p_mv(vec):
        #     # return y = inv(Sigma_tilde @ VH_tilde.T) @ vec
        #     return solve_triangular(VH_tilde.T, vec, lower=False)
        #
        # def p_rmv(vec):
        #     return solve_triangular(R, vec, 'T', lower=False)
        #
        # def mv(vec):
        #     return A @ (p_mv(vec))
        #
        # def rmv(vec):
        #     return p_rmv(A.T @ vec)

        # # reestimate gamma
        # gamma_new = s / r
        # # estimate the condition number of AN
        # cond_AN = (sqrt(gamma_new) + 1) / (sqrt(gamma_new) - 1)

        AN = LinearOperator(shape=(m, r), matvec=LSRN_matvec, rmatvec=LSRN_rmatvec)
        # print(AN.dtype)
        result = lsqr_copy_control_precision(AN, b, atol=tol, btol=tol, iter_lim=iter_lim)

        y = result[0]
        flag = result[1]
        itn = result[2]
        x = N.dot(y)
    else:

        print("The under-determined case is not implemented.")

    # print(x.dtype)
    return x, itn, flag, r
