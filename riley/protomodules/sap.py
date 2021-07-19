"""
Sketch and precondition
"""
import numpy as np
from scipy.sparse import linalg as sparla
from scipy.linalg import solve_triangular
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy


def upper_tri_precond_lsqr(A, b, R, tol, maxit):
    """
    Run preconditioned LSQR to obtain an approximate solution to
        min{ || A @ x - b ||_2 : x in R^m }
    where A.shape = (n, m) has n >> m, so the problem is over-determined.

    This uses Haoyun's version of SciPy's LSQR.
    ----------
    A : ndarray
        Data matrix with n rows and m columns. Columns are presumed linearly
        independent (for now).
    b : ndarray
        Right-hand-side b.shape = (n,).
    R : int
        The upper-triangular preconditioner, has R.shape = (m, m).
    tol : float
        Must be positive. Stopping criteria for LSQR.
    maxit : int
        Must be positive. Stopping criteria for LSQR.
    Returns
    -------
    x
    flag
    iternum
    """
    n, m = A.shape  # n >> m

    def p_mv(vec):
        # return y = inv(R) @ vec
        return solve_triangular(R, vec, lower=False)

    def p_rmv(vec):
        return solve_triangular(R, vec, 'T', lower=False)

    def mv(vec):
        return A @ (p_mv(vec))

    def rmv(vec):
        return p_rmv(A.T @ vec)

    A_precond = sparla.LinearOperator(shape=(n, m), matvec=mv, rmatvec=rmv)

    result = lsqr_copy(A_precond, b, atol=tol, btol=tol, iter_lim=maxit)[:8]
    x = p_mv(result[0])
    flag = result[1]
    iternum = result[2]
    return x, flag, iternum
