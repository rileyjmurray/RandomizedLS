"""# Riley's Blendenpik, with LSQR as iterative solver"""

import numpy as np
from scipy.sparse import linalg as sparla
from scipy.linalg import solve_triangular

from Haoyun.randomized_least_square_solver.Test.Blendenpik import srct_approx_chol
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy


def blendenpik_srct_scipy_lsqr_for_error_test(A, b, d, tol, maxit):
    """
    WARNING: this is not yet tested (but its components are tested).
    Run preconditioned conjugate gradients to obtain an approximate solution to
        min{ || A @ x - b ||_2 : x in R^m }
    where A.shape = (n, m) has n >> m, so the problem is over-determined. We
    get the preconditioner by applying a subsampled randomized cosine transform
    (an SRCT) A. The SRCT operation effectively embeds the range of A (which is
    an m-dimensional subspace in R^n) into an m-dimensional subspace of R^d.
    Parameters
    ----------
    A : ndarray
        Data matrix with n rows and m columns. Columns are presumed linearly
        independent (for now).
    b : ndarray
        Right-hand-side b.shape = (n,).
    d : int
        The embedding dimension. Theory suggests taking d \approx m * log m
        to ensure the randomly-constructed preconditioner is very effective.
    tol : float
        Must be positive. Stopping criteria for running PCG.
    maxit : int
        Must be positive. Stopping criteria for running PCG.
    Returns
    -------
    x : ndarray
        Approximate solution from PCG. x.shape = (m,).
    residuals: ndarray
        Initialized to the vector of all -1's. Any nonnegative entries are the
        residuals of the least-squares normal equations, at the corresponding
        iteration of PCG.
    (r, e) : ndarrays
        Define the SRCT used when sketching the matrix A.
    """
    n, m = A.shape  # n >> m
    L = np.zeros((m, m))
    L, reg, (r, e) = srct_approx_chol(d, A.T, L, 0.0)

    def p_mv(vec):
        # return y = inv(L.T) @ vec
        return solve_triangular(L, vec, 'T', lower=True)

    def p_rmv(vec):
        return solve_triangular(L, vec, lower=True)

    def mv(vec):
        return A @ (p_mv(vec))

    def rmv(vec):
        return p_rmv(A.T @ vec)

    A_precond = sparla.LinearOperator(shape=(n, m), matvec=mv, rmatvec=rmv)
    cond_A_precond = np.linalg.cond(A_precond @ np.identity(A_precond.shape[1]))
    result = lsqr_copy(A_precond, b, atol=tol, btol=tol, iter_lim=maxit)
    x = p_mv(result[0])
    flag = result[1]
    iternum = result[2]

    absolute_residual_error_array = result[-1]
    absolute_normal_equation_error_array = result[-2]
    relative_residual_error_array = result[-3]
    relative_normal_equation_error_array = result[-4]
    relative_error_array = result[-5]

    # return x, flag, iternum, (r, e)
    return x, flag, iternum, (r, e), relative_error_array, relative_normal_equation_error_array, \
           relative_residual_error_array, absolute_normal_equation_error_array, absolute_residual_error_array,
