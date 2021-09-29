import numpy as np

from Haoyun.randomized_least_square_solver.LSRN.LSRN_over_control_precison import LSRN_over_control_precision
from test_matrix_generator import overdetermined_ls_test_matrix_generator
from time import perf_counter
from Haoyun.randomized_least_square_solver.LSRN.LSRN_over import LSRN_over
from numpy.linalg import norm

# Small Test For Testing LSRN With Different Precisions

# Choose some different randomized matrices with different random number seeds
seednum_array_length = 10
seednum_array = np.random.choice(1000, seednum_array_length, replace=False)
# Set the matrix coherence to be low
coherence_type = "low"
# # Set the tolerance to be 1e-10
# tol = 1e-10
# Set the condition number to be 1e8
cond_num = 1e8

without_control_precision_relative_normal_equation_error_list = []
without_control_precision_relative_residual_error_list = []
without_control_precision_computational_time_list = []
without_control_precision_iteration_number_list = []

with_control_precision_relative_normal_equation_error_list = []
with_control_precision_relative_residual_error_list = []
with_control_precision_computational_time_list = []
with_control_precision_iteration_number_list = []

difference_x_norm_diff_precision_list = []
for seednum in seednum_array:
    A, _, b = overdetermined_ls_test_matrix_generator(m=10000,
                                                      n=500,
                                                      theta=0,
                                                      seednum=seednum,
                                                      fill_diagonal_method='geometric',
                                                      condition_number=cond_num)

    # x = x.ravel()
    b = b.ravel()

    t1 = perf_counter()
    x1, iternum1, flag1, _ = LSRN_over(A, b, tol=1e-10)
    t2 = perf_counter() - t1
    r1 = b - A @ x1

    without_control_precision_relative_normal_equation_error_list.append(norm(A.transpose() @ r1) / (norm(A) * norm(A) * norm(x1)))
    without_control_precision_relative_residual_error_list.append(norm(r1) / norm(b))
    without_control_precision_computational_time_list.append(t2)
    without_control_precision_iteration_number_list.append(iternum1)

    t3 = perf_counter()
    x2, iternum2, flag2, _ = LSRN_over_control_precision(A, b, tol=1e-5)
    t4 = perf_counter() - t3
    r2 = b - A @ x2

    with_control_precision_relative_normal_equation_error_list.append(norm(A.transpose() @ r2) / (norm(A) * norm(A) * norm(x2)))
    with_control_precision_relative_residual_error_list.append(norm(r2) / norm(b))
    with_control_precision_computational_time_list.append(t4)
    with_control_precision_iteration_number_list.append(iternum2)

    difference_x_norm_diff_precision_list.append(norm(x1 - x2, 2))

print("\nLSRN precision 64 bit:")
# print("\tThe norm of computed x is:", norm(x1, 2))
print("\tAveraged relative normal equation error:", np.mean(without_control_precision_relative_normal_equation_error_list))
print("\tAveraged relative residual error:", np.mean(without_control_precision_relative_residual_error_list))
print("\tAveraged computational time (sec.):", np.mean(without_control_precision_computational_time_list))
print("\tAveraged iteration number is:", np.mean(without_control_precision_iteration_number_list))
# print("\tThe flag is:", flag1)

print("\nLSRN precision 32 bit:")
# print("\tThe norm of computed x is:", norm(x2, 2))
print("\tAveraged relative normal equation error:", np.mean(with_control_precision_relative_normal_equation_error_list))
print("\tAveraged relative residual error:", np.mean(with_control_precision_relative_residual_error_list))
print("\tAveraged computational time (sec.):", np.mean(with_control_precision_computational_time_list))
print("\tAveraged iteration number is:", np.mean(with_control_precision_iteration_number_list))
# print("\tThe flag is:", flag2)

print("\nAveraged difference of the computed x:", np.mean(difference_x_norm_diff_precision_list))
