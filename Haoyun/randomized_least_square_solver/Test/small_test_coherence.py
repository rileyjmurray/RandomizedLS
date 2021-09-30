from Haoyun.randomized_least_square_solver.Test.test_matrix_generator_diff_coherence import \
    overdetermined_ls_test_matrix_generator_diff_coherence
from time import perf_counter
from scipy.sparse.linalg import lsqr
from Haoyun.randomized_least_square_solver.Test.LSRN import LSRN_over_without_mpi
import numpy as np
from Haoyun.randomized_least_square_solver.Test.Blendenpik import blendenpik_srct_scipy_lsqr
from numpy.linalg import norm

# Small test for observing the behavior of different randomized least square solvers
# under different matrix coherence

LSRN_iternum_array

matrix_coherence_list = ["low", "medium", "high"]
matrix_num = 5
for matrix_coherence in matrix_coherence_list:
    for matrix in np.arange(matrix_num):
        A, x, b = overdetermined_ls_test_matrix_generator_diff_coherence(m=6000,
                                                                         n=300,
                                                                         theta=0,
                                                                         coherence=matrix_coherence,
                                                                         seednum=100 * (matrix + 1) + 10 * (
                                                                                 matrix + 2) + matrix + 3)
        x = x.ravel()
        b = b.ravel()

        t10 = perf_counter()
        x6, flag1, iternum2 = lsqr(A, b, atol=1e-14, btol=1e-14, iter_lim=1000)[:3]
        t11 = perf_counter() - t10
        r1 = b - A @ x6

print("\nNaive LSQR Without Preconditioning:")
print("\tNormal Equation Error:", norm(A.transpose() @ r1) / (norm(A) * norm(r1)))
print("\tResidual Error:", norm(r1) / norm(b))
# print("\tError:", norm(x6 - x) / norm(x))
print("\tComputational time (sec.):", t11)
print("\tThe iteration number is:", iternum2)
print("\tThe flag is:", flag1)

t2 = perf_counter()
x2, iternum3, flag2, r1norm1, r2norm1, anorm1, acond1, arnorm1 = LSRN_over_without_mpi(A, b, tol=1e-14)[:8]
# x2, iternum3, flag2 = LSRN_over(A, b, tol=1e-14)[:3]
t3 = perf_counter() - t2
r2 = b - A @ x2

print("\nLSRN algorithm:")
print("\tR1norm:", r1norm1)
print("\tR2norm:", r2norm1)
print("\tEstimated Anorm:", norm(A))
print("\tTrue Anorm:", anorm1)
# print("\tAcond:", acond1)
print("\tEstimated Arnorm:", norm(A.transpose() @ r2))
print("\tTrue Arnorm:", arnorm1)
print("\tEstimated Normal Equation Error:", norm(A.transpose() @ r2) / (norm(A) * norm(r2)))
print("\tTrue Normal Equation Error:", arnorm1 / (anorm1 * r1norm1))
print("\tEstimated Residual Error:", norm(r2) / norm(b))
print("\tTrue Residual Error:", r1norm1 / norm(b))
# print("\tRelative Error:", norm(x2 - x, ord=2) / norm(x, ord=2))
print("\tComputational time (sec.):", t3)
print("\tThe iteration number is:", iternum3)
print("\tThe flag is:", flag2)

# t4 = perf_counter()
# x3 = np.linalg.lstsq(A, b, rcond=None)[0]
# t5 = perf_counter() - t4
# r3 = b - A @ x3
# print("\nNumPy least-squares algorithm:")
# print("\tNormal Equation Error:", norm(A.transpose() @ r3) / (norm(A) * norm(r3)))
# print("\tResidual Error:", norm(r3) / norm(b))
# # print("\tError:", norm(x3 - x, ord=2) / norm(x, ord=2))
# print("\tComputational time (sec.):", t5)

# t6 = perf_counter()
# x4 = CS(A, b)
# t7 = perf_counter() - t6
# print("Naive CS Without Preconditioning:")
# print("Normal Equation:", np.linalg.norm(A.transpose() @ A @ x4 - A.transpose() @ b, ord=2))
# print("Residual (L2-norm):", np.linalg.norm(A @ x4 - b, ord=2))
# print("Error:", np.linalg.norm(x4 - x, ord=2) / np.linalg.norm(x, ord=2))
# print("Computational time (sec.):", t7)

# t8 = perf_counter()
# multiplier = 2
# d = multiplier * A.shape[1]
# tol = 1e-14
#
# x5, res, (r, e) = blendenpik_srct(A, b, d, tol, 1000)
# t9 = perf_counter() - t8
# r4 = b - A @ x5
#
# print("\nRiley's Blendenpik:")
# print("\tNormal Equation Error:", norm(A.transpose() @ r4) / (norm(A) * norm(r4)))
# print("\tResidual Error:", norm(r4) / norm(b))
# # print("\tError:", norm(x5 - x, ord=2) / norm(x, ord=2))
# print("\tComputational time (sec.):", t9)
# print("\tIteration number: ", np.count_nonzero(res > -1))

t8 = perf_counter()
multiplier = 2
d = multiplier * A.shape[1]
tol = 1e-14
x5, flag3, iternum4, (r, e) = blendenpik_srct_scipy_lsqr(A, b, d, tol, 1000)
t9 = perf_counter() - t8
r4 = b - A @ x5

print("\nRiley's Blendenpik:")
print("\tNormal Equation Error:", norm(A.transpose() @ r4) / (norm(A) * norm(r4)))
print("\tResidual Error:", norm(r4) / norm(b))
# print("\tError:", norm(x5 - x, ord=2) / norm(x, ord=2))
print("\tComputational time (sec.):", t9)
print("\tIteration number: ", iternum4)
print("\tThe flag is:", flag3)
