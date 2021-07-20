"""
Sketch and precondition
"""
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import lsqr
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy
import riley.protomodules.preconditioners as pc
import numpy as np


def sketch_and_precond(A, b, S, tol, maxit):
    R, Q = pc.sketch_and_factor(S, A, reg=1e-8)
    b_ske = S @ b
    x_ske = solve_triangular(R, Q.T @ b_ske, lower=False)
    if np.linalg.norm(A @ x_ske - b) < np.linalg.norm(b):
        x0 = x_ske
    else:
        x0 = None
    x, flag, iternum = upper_tri_precond_lsqr(A, b, R, tol, maxit, x0=x0)
    return x, flag, iternum


def upper_tri_precond_lsqr(A, b, R, tol, maxit, x0=None):
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
    A_precond = pc.a_inv_r(A, R)
    if x0 is not None:
        x0 = (R @ x0).ravel()
        result = lsqr(A_precond, b, atol=tol, btol=tol, iter_lim=maxit, x0=x0)[:8]
    else:
        result = lsqr(A_precond, b, atol=tol, btol=tol, iter_lim=maxit)[:8]
    x = solve_triangular(R, result[0], lower=False)
    flag = result[1]
    iternum = result[2]
    return x, flag, iternum
