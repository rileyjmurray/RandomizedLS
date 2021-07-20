import numpy as np
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy
from scipy.sparse import linalg as sparla
from riley.protomodules.preconditioners import srct_precond
from scipy.linalg import solve_triangular


def blendenpik_srct(A, b, d, tol, maxit):
    """
    Run preconditioned LSQR to obtain an approximate solution to
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
        Must be positive. Stopping criteria for LSQR.
    maxit : int
        Must be positive. Stopping criteria for LSQR.
    Returns
    -------
    x : ndarray
        Approximate solution from LSQR. x.shape = (m,).
    residuals: ndarray
        Initialized to the vector of all -1's. Any nonnegative entries are the
        residuals of the least-squares normal equations, at the corresponding
        iteration of PCG.
    (r, e) : ndarrays
        Define the SRCT used when sketching the matrix A.
    """
    n, m = A.shape  # n >> m
    R, Q = srct_precond(A, d, 1e-6)
    # rhs = A.T @ b
    # x = np.zeros(m)
    # residuals = -np.ones(maxit)
    # At = np.ascontiguousarray(A.T)
    # dense_pcg(At, rhs, x, L, tol, maxit, residuals, 0.0)

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
