from test_matrix_generator import overdetermined_ls_test_matrix_generator
from time import perf_counter
from Haoyun.randomized_least_square_solver.Test.LSRN import LSRN_over
from Haoyun.randomized_least_square_solver.Test.Blendenpik.Riley_Blendenpik_new import blendenpik_srct
from numpy.linalg import norm

# Small Test For Choosing Condition Number of Test Matrix

# Set the tolerance to be 10 ** -10
tol = 1e-10

cond_num = 1e8
A, _, b = overdetermined_ls_test_matrix_generator(m=10000,
                                                  n=500,
                                                  theta=0,
                                                  seednum=345,
                                                  fill_diagonal_method='geometric',
                                                  condition_number=cond_num)
# x = x.ravel()
b = b.ravel()

# t1 = perf_counter()
# x1, flag1, iternum1 = lsqr_copy(A, b, atol=1e-14, btol=1e-14, iter_lim=1000)[:3]
# t2 = perf_counter() - t1
# r1 = b - A @ x1

# print("\nNaive LSQR Without Preconditioning:")
# print("\tRelative Normal Equation Error:", norm(A.transpose() @ r1) / (norm(A) * norm(A) * norm(x1)))
# print("\tRelative Residual Error:", norm(r1) / norm(b))
# print("\tComputational time (sec.):", t2)
# print("\tThe iteration number is:", iternum1)
# print("\tThe flag is:", flag1)

t3 = perf_counter()
x2, iternum2, flag2, _ = LSRN_over(A, b, tol=tol)
t4 = perf_counter() - t3
r2 = b - A @ x2

# print("\nLSRN With Original Scipy LSQR:")
print("\nLSRN:")
print("\tRelative Normal Equation Error:", norm(A.transpose() @ r2) / (norm(A) * norm(A) * norm(x2)))
print("\tRelative Residual Error:", norm(r2) / norm(b))
print("\tComputational time (sec.):", t4)
print("\tThe iteration number is:", iternum2)
print("\tThe flag is:", flag2)

# t5 = perf_counter()
# x3, iternum3, flag3, _ = LSRN_over_for_error_test(A, b, tol=1e-14)[:4]
# t6 = perf_counter() - t5
# r3 = b - A @ x3
#
# print("\nLSRN With Modified Scipy LSQR")
# print("\tRelative Normal Equation Error:", norm(A.transpose() @ r3) / (norm(A) * norm(A) * norm(x3)))
# print("\tRelative Residual Error:", norm(r3) / norm(b))
# print("\tComputational time (sec.):", t6)
# print("\tThe iteration number is:", iternum3)
# print("\tThe flag is:", flag3)

# Riley's Blendenpik
multiplier = 2
d = multiplier * A.shape[1]

t7 = perf_counter()
x4, flag4, iternum4, _ = blendenpik_srct(A, b, d, tol, 1000)
t8 = perf_counter() - t7
r4 = b - A @ x4

# print("\nRiley's Blendenpik With Original Scipy LSQR:")
print("\nRiley's Blendenpik:")
print("\tRelative Normal Equation Error:", norm(A.transpose() @ r4) / (norm(A) * norm(A) * norm(x4)))
print("\tRelative Residual Error:", norm(r4) / norm(b))
print("\tComputational time (sec.):", t8)
print("\tThe iteration number is:", iternum4)
print("\tThe flag is:", flag4)

# # Riley's Blendenpik
# multiplier = 2
# d = multiplier * A.shape[1]
# tol = 1e-14
#
# t9 = perf_counter()
# x5, flag5, iternum5, _ = blendenpik_srct_scipy_lsqr_for_error_test(A, b, d, tol, 1000)[:4]
# t10 = perf_counter() - t9
# r5 = b - A @ x5
#
# print("\nRiley's Blendenpik With Modified Scipy LSQR:")
# print("\tRelative Normal Equation Error:", norm(A.transpose() @ r5) / (norm(A) * norm(A) * norm(x5)))
# print("\tRelative Residual Error:", norm(r5) / norm(b))
# print("\tComputational time (sec.):", t10)
# print("\tIteration number: ", iternum5)
# print("\tThe flag is:", flag5)

# t6 = perf_counter()
# x4 = CS(A, b)
# t7 = perf_counter() - t6
# print("Naive CS Without Preconditioning:")
# print("Normal Equation:", np.linalg.norm(A.transpose() @ A @ x4 - A.transpose() @ b, ord=2))
# print("Residual (L2-norm):", np.linalg.norm(A @ x4 - b, ord=2))
# print("Error:", np.linalg.norm(x4 - x, ord=2) / np.linalg.norm(x, ord=2))
# print("Computational time (sec.):", t7)