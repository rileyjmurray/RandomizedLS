"""
RAndomized Linear Algebra Subroutines
"""
import numpy as np
import scipy.linalg as la


def powered_range_sketch_op(A, k, num_pass, sketch_op_gen, stabilizer, pps):
    """
    Use (num_pass - 1) passes over the matrix A to generate a matrix S whose
    range is (hopefully) closely aligned with the span of the top *right*
    singular vectors of A. This is useful for estimating the span of the top
    *left* singular vectors of A by evaluating Y = A @ S.

    We accomplish this roughly as follows:

        if num_pass is odd
            S = (A' A)^((num_pass - 1)//2) sketch_op_gen(n, k)
        if num_pass is even
            S = (A' A)^((num_pass - 2)//2) A' sketch_op_gen(m, k)

    That description is "rough" because repeated applications of A will cause
    floating point errors to rapidly accumulate. The parameter "pps" reads
    as "passes per stabilization": after "pps" applications of A or A.T, we
    call "stabilizer" on the working matrix to obtain a numerically well-
    behaved basis for its range. The most common choice of "stabilizer" is to
    return the factor Q from an (economic) QR factorization.
    """
    assert num_pass > 0
    if num_pass % 2 == 1:
        S = sketch_op_gen(A.shape[1], k)
        passes_done = 0
        q = (num_pass - 1) // 2
    else:
        S = A.T @ sketch_op_gen(A.shape[0], k)
        passes_done = 1
        if pps == 1:
            S = stabilizer(S)
        q = (num_pass - 2) // 2
    # q is an even integer; need to compute
    #   S := (A' A)^q S and
    # up to intermediate stabilization.
    while q > 0:
        S = A @ S
        passes_done += 1
        if passes_done % pps == 0:
            S = stabilizer(S)
        S = A.T @ S
        passes_done += 1
        if passes_done % pps == 0:
            S = stabilizer(S)
        q -= 1
    return S


def power_rangefinder(A, k, num_pass, sketch_op_gen=None, pps=1):
    """
    For an m-by-n input matrix A, return a randomly generated matrix Q with
    m rows and k << min(A.shape) orthonormal columns. We want the range of
    Q to be closely aligned with the span of the top left singular vectors
    of A. When building the matrix Q we are allowed to access A or A.T a total
    of num_pass times. See the function "power_rangefinder_sketch_op" for the
    meaning of the parameter "pps".

    sketch_op_gen is a function handle that accepts two positive integer arguments.
    The result of mat = sketch_op_gen(k1, k2) should be a k1-by-k2 random matrix.
    If no such function handle is provided, we define sketch_op_gen so that it
    generates a matrix with iid standard normal entries.
    """

    if sketch_op_gen is None:
        def sketch_op_gen(n_rows, n_cols):
            omega = np.random.standard_normal(size=(n_rows, n_cols))
            return omega

    def stabilizer(mat):
        return la.qr(mat, mode='economic')[0]

    S = powered_range_sketch_op(A, k, num_pass, sketch_op_gen, stabilizer, pps)
    Y = A @ S
    Q = la.qr(Y, mode='economic')[0]
    return Q
