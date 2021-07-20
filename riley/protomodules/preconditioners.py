import numpy as np
from scipy.sparse import linalg as sparla
import scipy as sp
from scipy.linalg import lapack
from scipy.linalg import qr as qr_factorize
import warnings
from scipy.linalg import solve_triangular
from riley.protomodules.embeddings import srct_embedding, iid_sparse_embedding, fixed_sparse_embedding


def srct_precond(A, d, reg=1e-6):
    n, m = A.shape
    assert n > d > m
    S = srct_embedding(d, n)
    R, Q = sketch_and_factor(S, A, reg)
    return R, Q


def iid_sparse_precond(A, d, density, reg=1e-6):
    assert density > 0
    assert density <= 1
    n, m = A.shape
    assert n > d > m
    S = iid_sparse_embedding(d, n, density)
    R, Q = sketch_and_factor(S, A, reg)
    return R, Q


def fixed_sparse_precond(A, d, col_nnz, reg=1e-6):
    n, m = A.shape
    assert n > d > m
    S = fixed_sparse_embedding(d, n, col_nnz)
    R, Q = sketch_and_factor(S, A, reg)
    return R, Q


def sketch_and_factor(S, A, reg=1e-6):
    Aske = S @ A
    Q, R, flag = robust_qr(Aske, reg)
    if not flag:
        warnings.warn('Preconditioner is only valid for regularized system.')
    return R, Q


def robust_qr(mat, reg):
    # Return the upper triangular factor "R" in a QR factorization
    # of "mat". If that QR factorization fails, then try again
    # by QR factorizing [mat, reg * I] where "I" is the identity
    # matrix.
    n = mat.shape[1]
    success = True
    try:
        Q, R = qr_factorize(mat, mode='economic')
    except sp.linalg.LinAlgError:
        Aske = np.row_stack((mat, reg * np.eye(n)))
        Q, R = qr_factorize(Aske, mode='economic')
        success = False
    return Q, R, success


def a_inv_r(A, R):
    """Return a linear operator that represents A @ inv(R) """
    def p_mv(vec):
        # return y = inv(R) @ vec
        return solve_triangular(R, vec, lower=False)

    def p_rmv(vec):
        return solve_triangular(R, vec, 'T', lower=False)

    def mv(vec):
        return A @ (p_mv(vec))

    def rmv(vec):
        return p_rmv(A.T @ vec)

    A_precond = sparla.LinearOperator(shape=A.shape, matvec=mv, rmatvec=rmv)
    return A_precond
