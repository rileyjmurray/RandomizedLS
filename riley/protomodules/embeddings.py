import numpy as np
import scipy.sparse as spar
import scipy.sparse.linalg as sparla
from scipy.fft import dct


def fixed_sparse_embedding(n_rows, n_cols, col_nnz):
    """

    Parameters
    ----------
    n_rows : int
        number of rows of embedding operator
    n_cols : int
        number of columns of embedding operator
    col_nnz : int
        number of nonzeros in each column.

    Returns
    -------
    S : SciPy sparse matrix
    """
    # column and row indices
    row_vecs = []
    for i in range(n_cols):
        rows = np.random.choice(n_rows, col_nnz, replace=False)
        row_vecs.append(rows)
    rows = np.concatenate(row_vecs)
    cols = np.repeat(np.arange(n_cols), col_nnz)
    # values for each row and col
    vals = np.ones(n_cols * col_nnz)
    vals[np.random.rand(n_cols * col_nnz) <= 0.5] = -1
    vals /= np.sqrt(col_nnz)
    # wrap up
    S = spar.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    S = S.tocsc()
    return S


def iid_sparse_embedding(n_rows, n_cols, density):
    # get row indices and col indices
    nonzero_idxs = np.random.rand(n_rows * n_cols) < density
    attempt = 0
    while np.all(~nonzero_idxs):
        if attempt == 10:
            raise RuntimeError('Density too low.')
        nonzero_idxs = np.random.rand(n_rows * n_cols) < density
        attempt += 1
    nonzero_idxs = np.where(nonzero_idxs)[0]
    rows, cols = np.unravel_index(nonzero_idxs, (n_rows, n_cols))
    # get values for each row and col index
    nnz = rows.size
    vals = np.ones(nnz)
    vals[np.random.rand(vals.size) < 0.5] = -1
    vals /= np.sqrt(n_rows * density)
    # Wrap up
    S = spar.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    S = S.tocsr()
    return S


def srct_embedding(n_rows, n_cols):
    r = np.random.choice(n_cols, size=n_rows, replace=False)
    e = np.random.rand(n_cols)
    e[e > 0.5] = 1.0
    e[e != 1] = -1.0
    e *= np.sqrt(n_cols / n_rows)

    def srft(mat):
        return apply_srct(r, e, mat, None)

    S = sparla.LinearOperator(shape=(n_rows, n_cols), matvec=srft, matmat=srft)
    return S


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
