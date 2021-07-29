import numpy as np
import scipy.linalg as la
import riley.protomodules.ralas as rs


def gaussian_sketch(n_rows, n_cols)
    omega = np.random.standard_normal(size=(n_rows, n_cols))
    return omega


def orth(mat):
    Q = la.qr(mat, mode='economic')[0]
    return Q

def rand_qb_b_fet(A, blk, tol, p, overwrite_A=True):
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
    assert ell/blk == ell//blk

    Q = np.empty(shape=(A.shape[0], 0), dtype=float)
    B = np.empty(shape=(0, A.shape[1]), dtype=float)
    sqnorm_A = la.norm(A, ord='fro') ** 2
    sqtol = tol**2

    S = rs.powered_range_sketch_op(A, ell, p, gaussian_sketch, orth, pps=1)
    G = A @ S
    H = A.T @ G
    for i in range(ell//blk):  # we might stop early
        blk_start = i*blk
        blk_end = (i+1)*blk
        Si = S[:, blk_start:blk_end]
        BSi = B @ Si
        Yi = G[:, blk_start:blk_end] - Q @ (BSi)
        Qi, Ri = la.qr(Yi, mode='economic', pivoting=False)
        Qi, Rihat = la.qr(Qi - Q @ (Q.T @ Qi), mode='economic', pivoting=False)
        Ri = Ri @ Rihat
        # Bi = R^{-T} H[:, blk_start:blk_end].T - Yi.T @ Q @ B - BSi.T @ B
        RinvTHT = la.solve_triangular(Ri, H[:, blk_start:blk_end].T, trans=True)
        Bi = RinvTHT - (Yi.T @ Q) @ B - BSi.T @ B
        Q = np.column_stack((Q, Qi))
        B = np.row_stack((B, Bi))
        sqnorm_A = sqnorm_A - la.norm(Bi, ord='fro')**2
        if sqnorm_A <= sqtol:
            break
    return Q, B


if __name__ == '__main__':

    print(0)
