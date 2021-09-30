from time import perf_counter

import numpy as np
from scipy.sparse import linalg as sparla
import scipy as sp
from scipy.linalg import lapack
from scipy.linalg import solve_triangular, qr as qr_factorize
from scipy.fft import dct

# from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy


def apply_srct(r, e, mat, perm=None):
    """
    Apply a subsampled randomized cosine transform (SRCT) to the columns
    of the ndarray mat. The transform is defined by data (r, e).

    Parameters
    ----------
    r : ndarray
        The random restriction used in the SRCT. The entries of "r" must
        be unique integers between 0 and mat.shape[0] (exclusive).
    e : ndarray
        The vector of signs used in the SRCT; e.size == mat.shape[0].
    mat : ndarray
        The operand for the embedding. If mat.ndim == 1, then simply apply
        the SRCT to mat as a vector.
    perm : ndarray
        permutation of range(mat.shape[0]).

    Returns
    -------
    mat : ndarray
        The transformed input.
    """
    if mat.ndim > 1:
        if perm is not None:
            mat = mat[perm, :]
        mat = mat * e[:, None]
        mat = dct(mat, axis=0, norm='ortho')
        mat = mat[r, :]
    else:
        if perm is not None:
            mat = mat[perm]
        mat = mat * e
        mat = dct(mat, norm='ortho')
        mat = mat[r]
    return mat


def srct_precond(A, d, reg=1e-6):
    tic_srct_and_mult = perf_counter()
    n, m = A.shape
    r = np.random.choice(n, size=d, replace=False)
    e = np.random.rand(n)
    e[e > 0.5] = 1.0
    e[e != 1] = -1.0
    e *= np.sqrt(n / d)

    def srft(mat):
        return apply_srct(r, e, mat, None)
    # time_srct = perf_counter() - tic_srct

    # tic_mult = perf_counter()
    S = sparla.LinearOperator(shape=(d, n), matvec=srft, matmat=srft)
    Aske = S @ A

    time_srct_and_mult = perf_counter() - tic_srct_and_mult

    tic_qr = perf_counter()
    try:
        R = qr_factorize(Aske, mode='economic')[1]
    except sp.linalg.LinAlgError:
        Aske = np.row_stack((Aske, reg*np.eye(n)))
        R = qr_factorize(Aske, mode='economic')[1]
    time_qr = perf_counter() - tic_qr
    return R, (r, e), time_srct_and_mult, time_qr


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
    tic_all = perf_counter()

    timing = {'srct_and_mult': 0.0, 'qr': 0.0, 'iter': 0.0, 'all': 0.0}
    flops = {'srct_and_mult': 0.0, 'qr': 0.0, 'iter': 0.0, 'all': 0.0}

    n, m = A.shape  # n >> m
    R, (r, e), time_srct_and_mult, time_qr = srct_precond(A, d, 1e-6)

    timing['srct_and_mult'] = time_srct_and_mult
    timing['qr'] = time_qr
    # rhs = A.T @ b
    # x = np.zeros(m)
    # residuals = -np.ones(maxit)
    # At = np.ascontiguousarray(A.T)
    # dense_pcg(At, rhs, x, L, tol, maxit, residuals, 0.0)

    tic_iter = perf_counter()

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

    result = sparla.lsqr(A_precond, b, atol=tol, btol=tol, iter_lim=maxit)[:8]
    x = p_mv(result[0])
    flag = result[1]
    iternum = result[2]
    timing['iter'] += perf_counter() - tic_iter

    timing['all'] = perf_counter() - tic_all

    # Calculate the flops
    # The embedding dimension d does not appear here, because SRCT is naively doing
    # the subsampling on a vector of length n. Rather than doing the sampling as the part
    # of the discrete cosine transform. The result is a factor of log2(n) rather than a factor log2(d).
    flops['srct_and_mult'] = n * m * np.log2(n)

    # QR decomposition, referring to Section 9.3.2 of
    # https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/09-num-lin-alg-scribed.pdf
    # Instead of calling SciPy QR, which forms the Q explicitly, should call the LAPACK routine,
    # which returns an implicit Q.

    # Use the bottom cost when we switch to implicit Q.
    flops['qr'] = 2 * d * m ** 2 - 2 / 3 * m ** 3

    flops['iter'] = (iternum + 1) * (4 * m * n + 2 * m ** 2)
    flops['all'] = flops['srct_and_mult'] + flops['qr'] + flops['iter']
    return x, flag, iternum, (r, e), timing, flops
