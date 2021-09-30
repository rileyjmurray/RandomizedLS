from math import ceil
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from numpy.linalg import svd
# from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy
from mpi4py import MPI
from time import perf_counter
from zignor import randn

"""	
Lisandro Dalcin:

Below, an scalable point-to-point based implementation of barrier() 
with the sleep() trick you need. A standard implementation would just 
merge send() and recv() on a single sendrecv() call. Just may need to 
tweak the sleep interval, and perhaps use a different tag value to 
avoid previous on-going communication.
"""

import time


def barrier(comm, tag=0, sleep=0.01):
    size = comm.Get_size()
    if size == 1:
        return
    rank = comm.Get_rank()
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = comm.isend(None, dst, tag)
        while not comm.Iprobe(src, tag):
            time.sleep(sleep)
        comm.recv(None, src, tag)
        req.Wait()
        mask <<= 1

def LSRN_over(A, b, tol=1e-8, gamma=2, iter_lim=1000, comm=MPI.COMM_WORLD):
    """
    LSRN computes the min-length solution of linear least squares via LSQR with
    randomized preconditioning

    Parameters
    ----------

    A       : {matrix, sparse matrix, ndarray, LinearOperator} of size m-by-n

    b       : (m,) ndarray

    gamma : float (>1), oversampling factor

    tol : float, tolerance such that norm(A*x-A*x_opt)<tol*norm(A*x_opt)

    iter_lim : integer, the max iteration number

    rcond : float, reciprocal of the condition number

    comm : mpi communicator

    Returns
    -------
    x      : (n,) ndarray, the min-length solution

    itn : int, iteration number

    r      : int, the rank of A

    flag : int,
    """
    global r, s, itn, x, flag
    tic_all = perf_counter()

    m, n = A.shape

    timing = {'randn': 0.0, 'mult': 0.0, 'svd': 0.0, 'iter': 0.0, 'comm': 0.0, 'all': 0.0}
    flops = {'randn': 0.0, 'mult': 0.0, 'svd': 0.0, 'iter': 0.0, 'comm': 0.0, 'all': 0.0}

    rank = comm.Get_rank()
    size = comm.Get_size()

    if m > n:  # over-determined
        s = ceil(gamma * n)
        A_tilde = np.zeros((s, n))
        blk_sz = 128
        for i in range(int(ceil(1.0 * s / blk_sz))):
            blk_begin = i * blk_sz
            blk_end = np.min([(i + 1) * blk_sz, s])
            blk_len = blk_end - blk_begin

            tic_randn = perf_counter()
            G = randn(blk_len, m)
            timing['randn'] += perf_counter() - tic_randn

            tic_mult = perf_counter()
            A_tilde[blk_begin:blk_end, :] = np.matmul(G, A)
            timing['mult'] += perf_counter() - tic_mult

        tic_comm = perf_counter()
        A_tilde = comm.reduce(A_tilde)
        timing['comm'] += perf_counter() - tic_comm

        if rank == 0:
            tic_svd = perf_counter()
            U_tilde, Sigma_tilde, VH_tilde = svd(A_tilde, False)
            rcond = np.min([m, n]) * np.finfo(float).eps
            r_tol = Sigma_tilde[0] * rcond
            r = np.sum(Sigma_tilde > r_tol)
            N = VH_tilde[:r, :].T / Sigma_tilde[:r]
            timing['svd'] += perf_counter() - tic_svd
        else:
            N = None

        # U_tilde, Sigma_tilde, VH_tilde = svd(A_tilde, False)
        # rcond = np.min(A.shape) * np.finfo(float).eps
        # r_tol = Sigma_tilde[0] * rcond
        # r = np.sum(Sigma_tilde > r_tol)
        #
        # N = VH_tilde[:r, :].T / Sigma_tilde[:r]

        barrier(comm)

        tic_comm = perf_counter()
        N = comm.bcast(N)
        timing['comm'] += perf_counter() - tic_comm

        tic_iter = perf_counter()

        def LSRN_matvec(v):
            return A.dot(N.dot(v))

        def LSRN_rmatvec(v):
            return N.T.dot(A.T.dot(v))

        # def p_mv(vec):
        #     # return y = inv(Sigma_tilde @ VH_tilde.T) @ vec
        #     return solve_triangular(N_inv, vec, lower=False)
        #
        # def p_rmv(vec):
        #     return solve_triangular(N_inv, vec, 'T', lower=False)
        #
        # def mv(vec):
        #     return A @ (p_mv(vec))
        #
        # def rmv(vec):
        #     return p_rmv(A.T @ vec)

        # # re-estimate gamma
        # gamma_new = s / r
        # # estimate the condition number of AN
        # cond_AN = (sqrt(gamma_new) + 1) / (sqrt(gamma_new) - 1)

        AN = LinearOperator(shape=(m, r), matvec=LSRN_matvec, rmatvec=LSRN_rmatvec)
        # AN = LinearOperator(shape=(m, r), matvec=mv, rmatvec=rmv)

        result = lsqr(AN, b, atol=tol, btol=tol, iter_lim=iter_lim)

        y = result[0]
        flag = result[1]
        itn = result[2]
        x = N.dot(y)
        # x = p_mv(y)

        timing['iter'] += perf_counter() - tic_iter
    else:

        print("The under-determined case is not implemented.")

    timing['all'] = perf_counter() - tic_all

    # Calculate the flops

    # not sure about the first one
    # flops['randn'] = m * s

    # Matrix-Matrix Product Flops, referring to Section 19.1.3 of
    # https://www.stat.cmu.edu/~ryantibs/convexopt-F18/scribes/Lecture_19.pdf
    flops['mult'] = 2 * s * m * n

    # Following the LAPACK _gesdd routine, referring to Table 3.13 of
    # https://www.netlib.org/lapack/lug/node71.html.
    # This url says that n-by-n matrix svd decomposition costs 20/3 * n^3.
    # On Riley's recommendations, we are doing 20/3 * s * n^2 for s-by-n matrix, where s is greater
    # than n.
    flops['svd'] = 20/3 * s * n ** 2

    flops['iter'] = (itn + 1) * (4 * m * n + 4 * n * r)
    flops['all'] = flops['randn'] + flops['mult'] + flops['svd'] + flops['iter']

    return x, itn, flag, r, timing, flops
