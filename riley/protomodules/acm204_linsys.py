import numpy as np
from numba import jit
import numba as nb
from scipy.sparse import linalg as sparla
from scipy.fft import dct
import scipy as sp
from scipy.linalg import lapack
from scipy import sparse as spar


@jit(nopython=True,
     signature_or_function=nb.int64(nb.float64[:, :], nb.float64[:], nb.float64[:]),
     cache=True)
def py_cho_solve(L, rhs, soln):
    """
    Compute "soln" that solves "M @ soln = rhs", where "L" is the lower-
    triangular Cholesky factor of M.

    Parameters
    ----------
    L : ndarray
        L.shape = (n, n).  We only read the lower-triangular entries.
    rhs :ndarray
        rhs.shape = (n,). We DO NOT overwrite this value.
    soln : ndarray
        soln.shape = (n,).

    Returns
    -------
    0 (this is an artifact of using Numba).
    """
    n = rhs.size
    for i in range(n):
        num = rhs[i]
        for j in range(i):
            num -= L[i, j] * soln[j]
        soln[i] = num / L[i, i]
    for ic in range(1, n+1):
        i = n - ic
        num = soln[i]
        for jc in range(1, ic):
            num -= L[n - jc, i] * soln[n - jc]
        soln[i] = num / L[i, i]
    return 0


@jit(nopython=True,
     signature_or_function=nb.void(
         nb.float64[:, ::1], nb.float64[:], nb.float64[::1],
         nb.float64[:, ::1], nb.float64, nb.int64, nb.float64[:], nb.float64),
     cache=True)
def dense_pcg(At, rhs, y, L, tol, maxit, residuals, reg):
    """
    Solve the positive definite linear system of dimension "n"

        (G + reg * np.eye(n)) @ y = rhs

    where G = At @ At.T and At is wide (i.e., At.shape = (n, m) with n < m).

    The array L is lower-triangular and satisfies

        L @ L.T \approx G + reg * np.eye(n)

    we use L as a preconditioner.

    Parameters
    ----------
    At : ndarray
        At.shape = (n, m) and n < m (so At is a wide matrix).
    rhs : ndarray
        rhs.shape = (n,)
    y : ndarray
        y.shape = (n,). This argument is overwritten as the algorithm runs;
        it's initial value is the starting point of PCG.
    L : ndarray
        L.shape = (n, n). We only access the lower-triangular part.
    tol : float
        Must be positive. Stop if the residual of the candidate solution
        reaches or falls below this value.
    maxit : int
        Must be positive. Stop if we reach this number of iterations.
    residuals : ndarray
        residuals.shape = (maxit,). residuals[k] is equal to the 2-norm
        of rhs - G @ y at iteration k.
    reg : float
        Regularization parameter. Must be nonnegative; often zero.

    Returns
    -------
    nothing
    """
    # G = At @ At.T
    # L @ L.T  \approx G + reg * np.eye(G.shape[0])
    temp = y @ At  # At.T @ y;  At.T is not C-contiguous
    r = At @ temp  # At is C-contiguous
    if reg > 0:
        r += reg * y
    r -= rhs  # r = At @ (At.T @ y) - rhs
    z = np.zeros(y.size)
    _ = py_cho_solve(L, r, z)
    p = -z
    k = 0
    num = r @ z
    temp = np.zeros(At.shape[1])
    while k < maxit and np.sqrt(num) > tol:
        residuals[k] = np.sqrt(num)
        np.dot(At.T, p, out=temp)
        den = temp @ temp
        if reg > 0:
            den += reg * (p @ p)
        alpha = num / den
        y += alpha * p
        r += alpha * (At @ temp)
        if reg > 0:
            r += (alpha * reg) * p
        _ = py_cho_solve(L, r, z)
        newnum = r @ z
        beta = newnum / num
        p *= beta
        p -= z  # p = -z + beta * p
        k += 1
        num = newnum
    pass


def reg_cho_factor(mat, lowreg):
    """
    Compute a lower-triangular Cholesky factor of "mat", or a possibly
    regularized version of "mat".

    If "mat" is not positive definite, then we compute a Cholesky factor of
        "mat + reg * np.eye(n)"
    for a positive constant "reg". We start by attempting reg=lowreg and
    iteratively increase reg by a factor ten until factorization succeeds.

    Parameters
    ----------
    mat : ndarray
        mat.shape = (n, n).
    lowreg : float
        At least 1e-6.

    Returns
    -------
    L : ndarray
        L.shape = (n, n) is a lower-triangular Cholesky factor of
        mat + reg * np.eye(n).
    reg : float
        The regularization needed to obtain a valid Cholesky factor.
        Should be equal to zero if mat is symmetric positive definite.

    Notes
    -----
    The matrix "mat" is not modified. This function contains somewhat
    inefficient memory management / arithmetic.
    """
    lowreg = max(lowreg, 1e-6)
    reg = 0.0
    it = 0
    success = False
    while True:
        m = mat.shape[0]
        try:
            mat = sp.linalg.cho_factor(mat + np.eye(m) * reg, lower=True,
                                       overwrite_a=False, check_finite=False)[0]
            success = True
        except np.linalg.LinAlgError as err:
            reg = lowreg * 10.0**it
        if success:
            break
        else:
            it += 1
    return mat, reg


def sparse_pcg(At, rhs, y, L, tol, maxit):
    """
    WARNING. This implementation doesn't work when At is a SciPy sparse
    matrix (this is very easy to fix but I'm not going to do that right now).

    Use the lower-triangular part of L as a preconditioner to solve

        (At @ At.T) @ y = rhs

    by SciPy's built-in conjugate gradients implementation. Uses abstract
    LinearOperator objects to avoid applying the preconditioner explicitly.

    Parameters
    ----------
    At : ndarray or SciPy sparse matrix
        At.shape = (n, m) and n < m (so At is a wide matrix).
    rhs : ndarray
        rhs.shape = (n,). We do NOT overwrite this value.
    y : ndarray
        y.shape = (n,). The final solution.
    L : ndarray
        L.shape = (n, n). We only access the lower-triangular part.
    tol : float
        Must be positive. Stop if the residual of the candidate solution
        reaches or falls below this value.
    maxit : int
        Must be positive. Stop if we reach this number of iterations.

    Returns
    -------
    nothing
    """
    def apply_precond(z):
        x = sp.linalg.cho_solve((L, True), z)
        return x

    def apply_AtA(z):
        x = At.T @ z
        x = At @ x
        return x
    AtA = sparla.LinearOperator(shape=L.shape, matvec=apply_AtA)
    precond = sparla.LinearOperator(shape=L.shape, matvec=apply_precond)
    res, status = sparla.cg(AtA, rhs, tol=tol, maxiter=maxit, M=precond)
    y *= 0
    y += res
    pass


def apply_srct(r, e, mat):
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

    Returns
    -------
    mat : ndarray
        The transformed input.
    """
    if mat.ndim > 1:
        mat = mat * e[:, None]
        mat = dct(mat, axis=0, norm='ortho')
        mat = mat[r, :]
    else:
        mat = mat * e
        mat = dct(mat, norm='ortho')
        mat = mat[r]
    return mat


def srct_approx_chol(d, At, L, lowreg):
    """
    Find a good preconditioner for *regularized* versions of linear systems

        (At @ At.T) @ x == b.

    The regularization is only applied when At @ At.T is singular.

    The function works as follows. First, we construct an implicit random sketching
    operator "S" that maps a given tall matrix of shape (n, m) to another tall matrix
    of shape (d, m), where d < n. The operator S is a randomized subsampled discrete
    cosine transform (an SRCT). Once we have S, we construct Aske = S @ At.T. The last
    step is to compute the lower-triangular Cholesky factor of a regularized Gram matrix

        G = Aske.T @ Aske + reg * np.eye(m).

    The parameter "reg" is determined by "reg_cho_factor(Aske.T @ Aske, lowreg)".

    We return reg, the Cholesky factor of G, and data that implicitly defines S.

    Parameters
    ----------
    d : int
        The dimension of the embedding.
    At : ndarray
        At.shape = (m, n) has n >> m. We want a preconditioner for linear systems
        of the form (At @ At.T + reg * np.eye(n)) @ x == b.
    L : ndarray
        Use this as workspace in computing the Cholesky factor.
    lowreg : float
        Must be positive (forced to at least 1e-6).
        Refer to reg_cho_factor(mat, lowreg) for behavior.

    Returns
    -------
    L : ndarray
        L.shape = (m, m). The lower-triangular part of L gives the desired preconditioner.
    reg : float
        Nonnegative; often zero. If "d" is chosen appropriately (e.g., d \approx m*log(m))
        then the matrix "L" should be a very good preconditioner for linear systems
        of the form (At @ At.T + reg * np.eye(m)) @ x == b.
    (r, e) : tuple of ndarray
        Defines the SRCT "S" used for sketching. See "apply_srct(r, e, mat)" for meaning.

    Notes
    -----
    The whole point of calling this function is that forming At @ At.T is very expensive.
    In order for this kind of algorithm to be useful, it needs to run in far less time
    than it takes to compute At @ At.T. Right now, that will not be the case!

    Ideally, this function would exit with "L" as the desired preconditioner. This is
    not currently the case because of memory management in "reg_cho_factor" and the
    need for the Cholesky factor to be C-contiguous.
    """
    m, n = At.shape
    r = np.random.choice(n, size=d, replace=False)
    e = np.random.rand(n)
    e[e > 0.5] = 1.0
    e[e != 1] = -1.0
    e *= np.sqrt(n / d)

    def srft(mat):
        return apply_srct(r, e, mat)

    S = sparla.LinearOperator(shape=(d, n), matvec=srft, matmat=srft)
    Aske = S @ At.T  # Understand A = At.T and Aske = "A, sketched".
    np.dot(Aske.T, Aske, out=L)
    L, reg = reg_cho_factor(L, lowreg=lowreg)
    L = np.ascontiguousarray(L)
    return L, reg, (r, e)


def blendenpik_srct(A, b, d, tol, maxit):
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
    rhs = A.T @ b
    x = np.zeros(m)
    residuals = -np.ones(maxit)
    At = np.ascontiguousarray(A.T)
    dense_pcg(At, rhs, x, L, tol, maxit, residuals, 0.0)
    return x, residuals, (r, e)
