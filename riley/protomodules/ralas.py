"""
RAndomized Linear Algebra Subroutines
"""
import numpy as np
import scipy.linalg as la


def raw_rangefinder(A, k, num_pass, sketch_op_gen, stabilizer, aps):
    """ 
    For an m-by-n input matrix A, return a randomly generated matrix Y with
    m rows and k << min(A.shape) columns. We want the range of Y to be
    closely aligned with the span of the top left singular vectors of A. When
    building the matrix Y we are allowed to access A or A.T a total of
    num_pass times.
    
    To do this, we ...
    
        1. Use (num_pass - 1) passes over the matrix A to generate a matrix
           S whose range is (hopefully) closely aligned with the span of the
           top *right* singular vectors of A. 
        
        2. Return Y = A @ S.

    Step 1 is accomplished roughly as follows: 
    
        if num_pass == 1,
            S = sketch_op_gen(n, k),
        if num_pass is odd and greater than 1,
            S = (A' A)^((num_pass - 1)//2) sketch_op_gen(n, k)
        if num_pass is even
            S = (A' A)^((num_pass - 2)//2) A' sketch_op_gen(m, k)
            
    That description is "rough" because repeated applications of A will cause
    floating point errors to rapidly accumulate. The parameter "aps" reads
    as "applications per stabilization": after "aps" applications of A or
    A.T, we call "stabilizer" on the working matrix to obtain a numerically
    well-behaved basis for its range. The most common choice of "stabilizer"
    is to return the factor Q from an (economic) QR factorization.
    
    We only perform stabilization while forming "S".
    """
    assert num_pass > 0
    if num_pass % 2 == 0:
        omega = A.T @ sketch_op_gen(A.shape[0], k)
        passes_done = 1
        if aps == 1:
            omega = stabilizer(omega)
        q = (num_pass - 2) // 2
    else:
        omega = sketch_op_gen(A.shape[1], k)
        passes_done = 0
        q = (num_pass - 1) // 2
    # q is an even integer; need to compute
    #   S = (A' A)^q omega and
    #   Y = A S
    # up to intermediate stabilization.
    Y = A @ omega
    while q > 0:
        if passes_done % aps == 0:
            Y = stabilizer(Y)
        Y = A.T @ Y
        passes_done += 1
        if passes_done % aps == 0:
            Y = stabilizer(Y)
        Y = A @ Y
        passes_done += 1
        q -= 1
    return Y


def rangefinder(A, k, num_pass, sketch_op_gen=None, aps=1):

    if sketch_op_gen is None:
        def sketch_op_gen(n_rows, n_cols):
            omega = np.random.standard_normal(size=(n_rows, n_cols))
            return omega

    def stabilizer(mat):
        return la.qr(mat, mode='economic')[0]

    Y = raw_rangefinder(A, k, num_pass, sketch_op_gen, stabilizer, aps)
    Q = la.qr(Y, mode='economic')[0]
    return Q
