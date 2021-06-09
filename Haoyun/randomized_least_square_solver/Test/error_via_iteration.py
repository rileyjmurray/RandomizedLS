from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr

from Haoyun.randomized_least_square_solver.LSRN.LSRN_over import LSRN_over
from test_matrix_generator import overdetermined_ls_test_matrix_generator

condition_number = 1e16
A, x, b = overdetermined_ls_test_matrix_generator(m=1000,
                                                  n=50,
                                                  theta=0,
                                                  seednum=123,
                                                  fill_diagonal_method='geometric',
                                                  condition_number=condition_number)
x = x.ravel()
b = b.ravel()

# Naive LSQR
t1 = perf_counter()

total1 = lsqr(A, b, atol=1e-14, btol=1e-14, iter_lim=1000)
x1 = total1[0]
flag1 = total1[1]
iternum1 = total1[2]
naive_lsqr_relative_residual_error_array_array = total1[-1]
naive_lsqr_relative_normal_equation_error_array = total1[-2]

t2 = perf_counter() - t1

print("\nNaive LSQR:")
print("\tThe flag is:", flag1)

# LSRN
t3 = perf_counter()

total2 = LSRN_over(A, b, tol=1e-14, iter_lim=2000)
x2 = total2[0]
iternum2 = total2[1]
flag2 = total2[2]
lsrn_relative_residual_error_array_array = total2[-1]
lsrn_relative_normal_equation_error_array = total2[-2]

t4 = perf_counter() - t3

print("\nLSRN:")
print("\tThe flag is:", flag2)

# Naive LSQR Error plots
iternum1_list = np.arange(1, iternum1 + 1)

plt.plot(iternum1_list, np.log10(naive_lsqr_relative_residual_error_array_array), 'g:', label='Relative Residual Error')
plt.plot(iternum1_list, np.log10(naive_lsqr_relative_normal_equation_error_array), 'b',
         label='Relative Normal Equation Error')
plt.xlabel('iteration number')
plt.ylabel('error')
plt.title('Error Via Iteration Naive LSQR Cond ' + str(condition_number))
plt.legend(loc='best', shadow=True)
plt.savefig('Error/Naive LSQR/Error(Cond ' + str(condition_number) + ').png')
plt.show()

# LSRN Error Plots
iternum2_list = np.arange(1, iternum2 + 1)

plt.plot(iternum2_list, np.log10(lsrn_relative_residual_error_array_array), 'g:', label='Relative Residual Error')
plt.plot(iternum2_list, np.log10(lsrn_relative_normal_equation_error_array), 'b',
         label='Relative Normal Equation Error')
plt.xlabel('iteration number')
plt.ylabel('error')
plt.title('Error Via Iteration LSRN Cond ' + str(condition_number))
plt.legend(loc='best', shadow=True)
plt.savefig('Error/LSRN/Error(Cond ' + str(condition_number) + ').png')
plt.show()

# # Riley Blendenpik Error Plots
# iternum2_list = np.arange(1, iternum2 + 1)
#
# plt.plot(iternum2_list, np.log10(lsrn_relative_residual_error_array_array), 'g:', label='Relative Residual Error')
# plt.plot(iternum2_list, np.log10(lsrn_relative_normal_equation_error_array), 'b',
#          label='Relative Normal Equation Error')
# plt.xlabel('iteration number')
# plt.ylabel('error')
# plt.title('Different Error Metrics Via Iteration Number(LSRN)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Error/LSRN/Error.png')
# plt.show()
