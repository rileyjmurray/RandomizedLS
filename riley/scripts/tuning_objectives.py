import numpy as np
from riley.protomodules.preconditioners import iid_sparse_precond
from riley.protomodules.preconditioners import fixed_sparse_precond
from riley.protomodules.sap import upper_tri_precond_lsqr


def sparse_sketch_tuning_objective(A, b, d, tol, maxit, sparsity_type='iid'):
    """
    This is a toy implementation that lets you see how parameters for
    sparse embeddings affect iteration count for the given least-squares
    problem. The parameters are "density" in the iid case and
    "number of nonzeros per column" in the fixed-sparsity case.

    We construct a sketch S, get the upper-triangular factor "R" from a
    QR decomposition of S @ A, and then run preconditioned LSQR for at
    most "maxit" iterations or until "tol" accuracy is reached.

    The returned function handle makes no attempt to manage random seeds
    in an intelligent way.
    """
    if d > A.shape[0]:
        msg = 'Embedding dimension d=%s cannot be greater than A.shape[0]=%s' % (str(d), str(A.shape[0]))
        raise ValueError(msg)
    if sparsity_type == 'iid':
        def objective(density):
            R = iid_sparse_precond(A, d, density)
            x, flag, iternum = upper_tri_precond_lsqr(A, b, R, tol, maxit)
            return iternum
        return objective
    elif sparsity_type == 'fixed':
        def objective(col_nnz):
            R = fixed_sparse_precond(A, d, col_nnz)
            x, flag, iternum = upper_tri_precond_lsqr(A, b, R, tol, maxit)
            return iternum
        return objective
    else:
        raise ValueError()
