import numpy as np
import scipy.linalg as la
import riley.protomodules.ralas as rs


def gaussian_sketch(n_rows, n_cols):
    omega = np.random.standard_normal(size=(n_rows, n_cols))
    return omega


def orth(mat):
    Q = la.qr(mat, mode='economic')[0]
    return Q


def rand_qb_b_fet(A, blk, tol, p, overwrite_A=False):
    if not overwrite_A:
        A = np.copy(A)
    Q = np.empty(shape=(A.shape[0], 0), dtype=float)
    B = np.empty(shape=(0, A.shape[1]), dtype=float)
    sqnorm_A = la.norm(A, ord='fro') ** 2
    sqtol = tol**2
    while True:
        # Step 1: powered rangefinder
        S = rs.powered_range_sketch_op(A, blk, p, gaussian_sketch, orth, pps=1)
        Y = A @ S
        Qi = la.qr(Y, mode='economic')[0]
        # Step 2: project onto orthogonal complement of span(Q), and
        # reorthogonalize (these steps not necessary in exact arithmetic)
        Qi = Qi - Q @ (Q.T @ Qi)
        Qi = la.qr(Qi, mode='economic')[0]
        # Step 3: complete this block's QB factorization
        Bi = Qi.T @ A
        # Step 4: update the full factorization
        Q = np.column_stack((Q, Qi))
        B = np.row_stack((B, Bi))
        A = A - Qi @ Bi
        sqnorm_A = sqnorm_A - la.norm(Bi, ord='fro')**2
        if sqnorm_A <= sqtol:
            break
    return Q, B


def rand_qb_b_pe_fet(A, blk, ell, tol, p):
    Q = np.empty(shape=(A.shape[0], 0), dtype=float)
    B = np.empty(shape=(0, A.shape[1]), dtype=float)
    sqnorm_A = la.norm(A, ord='fro') ** 2
    sqtol = tol**2

    S = rs.powered_range_sketch_op(A, ell, p, gaussian_sketch, orth, pps=1)
    G = A @ S
    H = A.T @ G
    for i in range(int(np.ceil(ell//blk))):  # we might stop early
        blk_start = i*blk
        blk_end = min((i+1)*blk, S.shape[1])
        Si = S[:, blk_start:blk_end]
        BSi = B @ Si
        Yi = G[:, blk_start:blk_end] - Q @ BSi
        Qi, Ri = la.qr(Yi, mode='economic', pivoting=False)
        Qi, Rihat = la.qr(Qi - Q @ (Q.T @ Qi), mode='economic', pivoting=False)
        Ri = Rihat @ Ri
        # Bi = R^{-T} H[:, blk_start:blk_end].T - Yi.T @ Q @ B - BSi.T @ B
        temp =  H[:, blk_start:blk_end].T - (Yi.T @ Q) @ B - BSi.T @ B
        Bi = la.solve_triangular(Ri, temp, trans='T')
        Q = np.column_stack((Q, Qi))
        B = np.row_stack((B, Bi))
        sqnorm_A = sqnorm_A - la.norm(Bi, ord='fro')**2
        if sqnorm_A <= sqtol:
            break
    return Q, B


def residual(Q, A):
    orthog = A - Q @ (Q.T @ A)
    return np.linalg.norm(orthog, ord=2)


def ex1():
    np.set_printoptions(precision=4)
    m, n = 1000, 200
    k = 50
    H = np.random.standard_normal(size=(m, k))
    H = H * (1 / np.arange(1, k+1))
    G = np.random.standard_normal(size=(k, n))
    A = H @ G

    max_pass = 7
    min_pass = 1

    min_blk = 3
    max_blk = 7
    num_pass = 3
    tols = np.logspace(2, -2, num=10) * la.norm(A, ord='fro')
    res = {
        'out_loop_err': np.zeros(shape=(tols.size, max_pass - min_pass)),
        'in_loop_err': np.zeros(shape=(tols.size, max_pass - min_pass)),
        'out_loop_rank': np.zeros(shape=(tols.size, max_pass - min_pass)),
        'in_loop_rank': np.zeros(shape=(tols.size, max_pass - min_pass)),
    }
    tols = np.logspace(2, -2, num=10) * la.norm(A, ord='fro')
    for i_, blk in enumerate(range(min_blk, max_blk)):
        for j_, curr_tol in enumerate(tols):
            np.random.seed(0)
            Q, B = rand_qb_b_pe_fet(A, blk, k+blk, curr_tol, p=num_pass)
            nrm = la.norm(A - Q @ B, ord=2)
            res['out_loop_err'][j_, i_] = nrm
            res['out_loop_rank'][j_, i_] = B.shape[0]
            np.random.seed(0)
            Q, B = rand_qb_b_fet(A, blk, curr_tol, p=num_pass)
            nrm = la.norm(A - Q @ B, ord=2)
            res['in_loop_err'][j_, i_] = nrm
            res['in_loop_rank'][j_, i_] = B.shape[0]

    print('\nOperator norm error')
    print('\tAccessing A in-loop (proper bounded fro error)')
    print(res['in_loop_err'])
    print('\tAccessing A out-of-loop ("pass efficient")')
    print(res['out_loop_err'])

    print('\nRank of approximation')
    print('\tAccessing A in-loop (proper bounded fro error)')
    print(res['in_loop_rank'])
    print('\tAccessing A out-of-loop ("pass efficient", rank bounded )')
    print(res['out_loop_rank'])
    pass


if __name__ == '__main__':
    ex1()
