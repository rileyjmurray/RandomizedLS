# This file contains the direct method to solve least square using LAPACK, and may be used as a baseline
# for tuning purpose.
import numpy as np
from scipy.linalg.lapack import dgeqrf, dormqr, dtrtrs, dposv, dgesv
from scipy.linalg.blas import dgemm, dgemv


# This is the method that uses QR to solve least square.
def LAPACK_solve_ls_with_QR(A, b):
    # Ref: https://stackoverflow.com/questions/21970510/solving-a-linear-system-with-lapacks-dgeqrf
    # The corresponding procedure in LAPACK is https://www.netlib.org/lapack/lug/node40.html
    qr, tau, work, info = dgeqrf(A)
    cq, work, info = dormqr('L', 'T', qr, tau, b, qr.shape[0])
    x_qr, info = dtrtrs(qr, cq)
    return x_qr[0:A.shape[1]]


# This is the method that uses QR to solve normal equation to solve least square.
def LAPACK_solve_ls_with_normal_equation(A, b):
    #  The corresponding procedure in LAPACK is https://www.netlib.org/lapack/lug/node27.html
    ATA = np.matmul(A.T, A)
    ATb = np.matmul(A.T, b)
    # ATA = dgemm(1, A.T, A)
    # ATb = dgemm(1, A.T, b)
    _, x, _ = dposv(ATA, ATb)
    # _, _, x, _ = dgesv(ATA, ATb)
    return x

# syrk
