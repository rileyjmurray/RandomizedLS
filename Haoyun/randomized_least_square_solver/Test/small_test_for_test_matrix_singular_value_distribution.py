from time import perf_counter
import numpy as np
from numpy.linalg import norm
from Haoyun.randomized_least_square_solver.Test.LSRN import LSRN_over_without_mpi
from Haoyun.randomized_least_square_solver.Test.test_matrix_generator import \
    overdetermined_ls_test_matrix_generator

# Small Test For Testing LSRN On Matrices Different Singular Value distributions

# Choose some different randomized matrices with different random number seeds
seednum_array_length = 10
seednum_array = np.random.choice(1000, seednum_array_length, replace=False)
# Set the matrix coherence to be low
coherence_type = "low"
# Set the tolerance to be 1e-10
tol = 1e-10
# Set the condition number to be 1e8
cond_num = 1e8

one_small_relative_normal_equation_error_list = []
one_small_relative_residual_error_list = []
one_small_computational_time_list = []
one_small_computational_iteration_number_list = []

geometric_relative_normal_equation_error_list = []
geometric_relative_residual_error_list = []
geometric_computational_time_list = []
geometric_computational_iteration_number_list = []

arithmetic_relative_normal_equation_error_list = []
arithmetic_relative_residual_error_list = []
arithmetic_computational_time_list = []
arithmetic_computational_iteration_number_list = []

for seednum in seednum_array:
    A_one_small, _, b_one_small = overdetermined_ls_test_matrix_generator(m=10000,
                                                                          n=500,
                                                                          coherence_type="low",
                                                                          added_row_count=1,
                                                                          theta=0,
                                                                          seednum=seednum,
                                                                          fill_diagonal_method='one small',
                                                                          condition_number=cond_num)

    # x = x.ravel()
    b_one_small = b_one_small.ravel()

    t1_one_small = perf_counter()
    x1_one_small, iternum1_one_small, flag1_one_small, _, _, _ = LSRN_over_without_mpi(A_one_small, b_one_small, tol=tol, gamma=2)
    t2_one_small = perf_counter() - t1_one_small
    r1_one_small = b_one_small - A_one_small @ x1_one_small

    one_small_relative_normal_equation_error_list.append(
        norm(A_one_small.transpose() @ r1_one_small) / (norm(A_one_small * norm(A_one_small) * norm(x1_one_small))))
    one_small_relative_residual_error_list.append(norm(r1_one_small) / norm(b_one_small))
    one_small_computational_time_list.append(t2_one_small)
    one_small_computational_iteration_number_list.append(iternum1_one_small)

    A_geometric, _, b_geometric = overdetermined_ls_test_matrix_generator(m=10000,
                                                                          n=500,
                                                                          coherence_type="low",
                                                                          added_row_count=1,
                                                                          theta=0,
                                                                          seednum=seednum,
                                                                          fill_diagonal_method='geometric',
                                                                          condition_number=cond_num)

    # x = x.ravel()
    b_geometric = b_geometric.ravel()

    t1_geometric = perf_counter()
    x1_geometric, iternum1_geometric, flag1_geometric, _, _, _ = LSRN_over_without_mpi(A_geometric, b_geometric, tol=tol, gamma=2)
    t2_geometric = perf_counter() - t1_one_small
    r1_geometric = b_geometric - A_geometric @ x1_geometric

    geometric_relative_normal_equation_error_list.append(
        norm(A_geometric.transpose() @ r1_geometric) / (norm(A_geometric * norm(A_geometric) * norm(x1_geometric))))
    geometric_relative_residual_error_list.append(norm(r1_geometric) / norm(b_geometric))
    geometric_computational_time_list.append(t2_geometric)
    geometric_computational_iteration_number_list.append(iternum1_geometric)

    A_arithmetic, _, b_arithmetic = overdetermined_ls_test_matrix_generator(m=10000,
                                                                            n=500,
                                                                            coherence_type="low",
                                                                            added_row_count=1,
                                                                            theta=0,
                                                                            seednum=seednum,
                                                                            fill_diagonal_method='arithmetic',
                                                                            condition_number=cond_num)

    # x = x.ravel()
    b_arithmetic = b_arithmetic.ravel()

    t1_arithmetic = perf_counter()
    x1_arithmetic, iternum1_arithmetic, flag1_arithmetic, _, _, _ = LSRN_over_without_mpi(A_arithmetic, b_arithmetic, tol=tol, gamma=2)
    t2_arithmetic = perf_counter() - t1_arithmetic
    r1_arithmetic = b_arithmetic - A_arithmetic @ x1_arithmetic

    arithmetic_relative_normal_equation_error_list.append(
        norm(A_arithmetic.transpose() @ r1_arithmetic) / (norm(A_arithmetic * norm(A_arithmetic) * norm(x1_arithmetic))))
    arithmetic_relative_residual_error_list.append(norm(r1_arithmetic) / norm(b_arithmetic))
    arithmetic_computational_time_list.append(t2_arithmetic)
    arithmetic_computational_iteration_number_list.append(iternum1_arithmetic)

print("\nLSRN With Singualar Value Distribution One Small Matrix:")
print("\tRelative Normal Equation Error:", np.mean(one_small_relative_normal_equation_error_list))
print("\tRelative Residual Error:", np.mean(one_small_relative_residual_error_list))
print("\tComputational time (sec.):", np.mean(one_small_computational_time_list))
print("\tThe iteration number is:", np.mean(one_small_computational_iteration_number_list))


print("\nLSRN With Singualar Value Distribution Geometric Matrix:")
print("\tRelative Normal Equation Error:", np.mean(geometric_relative_normal_equation_error_list))
print("\tRelative Residual Error:", np.mean(geometric_relative_residual_error_list))
print("\tComputational time (sec.):", np.mean(geometric_computational_time_list))
print("\tThe iteration number is:", np.mean(geometric_computational_iteration_number_list))

print("\nLSRN With Singualar Value Distribution Arithmetic Matrix:")
print("\tRelative Normal Equation Error:", np.mean(arithmetic_relative_normal_equation_error_list))
print("\tRelative Residual Error:", np.mean(arithmetic_relative_residual_error_list))
print("\tComputational time (sec.):", np.mean(arithmetic_computational_time_list))
print("\tThe iteration number is:", np.mean(arithmetic_computational_iteration_number_list))

