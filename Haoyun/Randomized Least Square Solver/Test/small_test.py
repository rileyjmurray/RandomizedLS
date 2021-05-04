from Iter_Solver.CS import CS
from test_matrix_generator import overdetermined_ls_test_matrix_generator
from time import perf_counter
from scipy.sparse.linalg import lsqr
from LSRN.LSRN_over import LSRN_over
import numpy as np
from Blendenpik.Riley_Blendenpik import blendenpik_srct

# Small Test For Choosing Condition Number of Test Matrix

cond_num = 1000000
A, x, b = overdetermined_ls_test_matrix_generator(m=4000,
                                                  n=200,
                                                  theta=2**-1,
                                                  seednum=123,
                                                  fill_diagonal_method='arithmetic',
                                                  condition_number=cond_num)
x = x.ravel()
b = b.ravel()

# t0 = perf_counter()
# blendenpik = Blendenpik(A, b)
# x1, iternum1 = blendenpik.solve()
# t1 = perf_counter() - t0
# print("Blendenpik algorithm:")
# print("\tNormal Equation:", np.linalg.norm(A.transpose() @ A @ x1 - A.transpose() @ b,ord =2))
# print("\tResidual (L2-norm):", np.linalg.norm(A @ x1 - b,ord =2))
# print("\tError:", np.linalg.norm(x1 - x,ord =2)/np.linalg.norm(x,ord =2))
# print("\tComputational time (sec.):", t1)
# print("\tThe iteration number is:", iternum1)

t10 = perf_counter()
total2 = lsqr(A, b, atol=1e-14, btol=1e-14)
x6 = total2[0]
iternum2 = total2[2]
t11 = perf_counter() - t10
print("\nNaive LSQR Without Preconding:")
print("\tNormal Equation:", np.linalg.norm(A.transpose() @ A @ x6 - A.transpose() @ b, ord=2))
print("\tResidual (L2-norm):", np.linalg.norm(A @ x6 - b, ord=2))
print("\tError:", np.linalg.norm(x6 - x, ord=2) / np.linalg.norm(x, ord=2))
print("\tComputational time (sec.):", t11)
print("\tThe iteration number is:", iternum2)

t2 = perf_counter()
x2, iternum3 = LSRN_over(A, b, tol=1e-14)[:2]
t3 = perf_counter() - t2
print("\nLSRN algorithm:")
print("\tNormal Equation:", np.linalg.norm(A.transpose() @ A @ x2 - A.transpose() @ b, ord=2))
print("\tResidual (L2-norm):", np.linalg.norm(A @ x2 - b, ord=2))
print("\tError:", np.linalg.norm(x2 - x, ord=2) / np.linalg.norm(x, ord=2))
print("\tComputational time (sec.):", t3)
print("\tThe iteration number is:", iternum3)

# t4 = perf_counter()
# x3 = np.linalg.lstsq(A, b, rcond=None)[0]
# t5 = perf_counter() - t4
# print("\nNumPy least-squares algorithm:")
# print("\tNormal Equation:", np.linalg.norm(A.transpose() @ A @ x3 - A.transpose() @ b, ord=2))
# print("\tResidual (L2-norm):", np.linalg.norm(A @ x3 - b, ord=2))
# print("\tError:", np.linalg.norm(x3 - x, ord=2) / np.linalg.norm(x, ord=2))
# print("\tComputational time (sec.):", t5)

# t6 = perf_counter()
# x4 = CS(A, b)
# t7 = perf_counter() - t6
# print("Naive CS Without Preconditioning:")
# print("Normal Equation:", np.linalg.norm(A.transpose() @ A @ x4 - A.transpose() @ b, ord=2))
# print("Residual (L2-norm):", np.linalg.norm(A @ x4 - b, ord=2))
# print("Error:", np.linalg.norm(x4 - x, ord=2) / np.linalg.norm(x, ord=2))
# print("Computational time (sec.):", t7)

t8 = perf_counter()
multiplier = 2
d = multiplier * A.shape[1]
tol = 1e-14
x5, res, (r, e) = blendenpik_srct(A, b, d, tol, 1000)
t9 = perf_counter() - t8
print("\nRiley's Blendenpik:")
print("\tNormal Equation:", np.linalg.norm(A.transpose() @ A @ x5 - A.transpose() @ b, ord=2))
print("\tResidual (L2-norm):", np.linalg.norm(A @ x5 - b, ord=2))
print("\tError:", np.linalg.norm(x5 - x, ord=2) / np.linalg.norm(x, ord=2))
print("\tComputational time (sec.):", t9)
print("\tIteration number: ", np.count_nonzero(res > -1))
