import numpy as np
from Haoyun.randomized_least_square_solver.Test.Blendenpik.Riley_Blendenpik_new import blendenpik_srct
from Haoyun.randomized_least_square_solver.Test.LSRN.LSRN_over_without_mpi import LSRN_over_without_mpi
from test_matrix_generator import overdetermined_ls_test_matrix_generator
from Haoyun.randomized_least_square_solver.Test.LSRN import LSRN_over
from numpy.linalg import norm

# Small Test For Choosing Condition Number of Test Matrix
# Set the tolerance to be 1e-10
tol = 1e-10
cond_num = 1e8
# seednum = 3578
# # Choose some different randomized matrices with different random number seeds
seednum_array_length = 5
seednum_array = np.random.choice(10000, seednum_array_length, replace=False)

lsrn_relative_normal_equation_error_list = []
lsrn_relative_residual_error_list = []
lsrn_total_computational_time_list = []
lsrn_generate_random_number_time_list = []
lsrn_matrix_multiply_time_list = []
lsrn_matrix_multiply_flops_list = []
lsrn_svd_decomposition_time_list = []
lsrn_svd_decomposition_flops_list = []
lsrn_comm_time_list = []
lsrn_iterative_solver_time_list = []
lsrn_iterative_solver_flops_list = []
lsrn_iteration_number_list = []

lsrn_parallel_relative_normal_equation_error_list = []
lsrn_parallel_relative_residual_error_list = []
lsrn_parallel_total_computational_time_list = []
lsrn_parallel_generate_random_number_and_matrix_multiply_time_list = []
lsrn_parallel_matrix_multiply_flops_list = []
lsrn_parallel_comm_time_list = []
lsrn_parallel_svd_decomposition_time_list = []
lsrn_parallel_svd_decomposition_flops_list = []
lsrn_parallel_iterative_solver_time_list = []
lsrn_parallel_iterative_solver_flops_list = []
lsrn_parallel_iteration_number_list = []


riley_blen_relative_normal_equation_list = []
riley_blen_relative_residual_list = []
riley_blen_total_computational_time_list = []
riley_blen_SRCT_matrix_multiply_time_list = []
riley_blen_SRCT_matrix_multiply_flops_list = []
riley_blen_QR_decomposition_time_list = []
riley_blen_QR_decomposition_flops_list = []
riley_blen_iterative_solver_time_list = []
riley_blen_iterative_solver_flops_list = []
riley_blen_iteration_number_list = []

for seednum in seednum_array:
    A, _, b = overdetermined_ls_test_matrix_generator(m=40000,
                                                      n=2000,
                                                      theta=0,
                                                      seednum=seednum,
                                                      fill_diagonal_method='geometric',
                                                      condition_number=cond_num)

    # x = x.ravel()
    b = b.ravel()

    # t1 = perf_counter()
    # x1, flag1, iternum1 = lsqr(A, b, atol=1e-14, btol=1e-14, iter_lim=1000)[:3]
    # t2 = perf_counter() - t1
    # r1 = b - A @ x1
    #
    # print("\nNaive LSQR Without Preconditioning:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r1) / (norm(A) * norm(A) * norm(x1)))
    # print("\tRelative Residual Error:", norm(r1) / norm(b))
    # print("\tComputational time (sec.):", t2)
    # print("\tThe iteration number is:", iternum1)
    # print("\tThe flag is:", flag1)

    x2, iternum2, flag2, _, timing2, flops2 = LSRN_over(A, b, tol=tol)
    r2 = b - A @ x2

    # lsrn_relative_normal_equation_error_list.append(norm(A.transpose() @ r2) / (norm(A) * norm(A) * norm(x2)))
    # lsrn_relative_residual_error_list.append(norm(r2) / norm(b))
    lsrn_total_computational_time_list.append(timing2['all'])
    lsrn_generate_random_number_time_list.append(timing2['randn'])
    lsrn_matrix_multiply_time_list.append(timing2['mult'])
    lsrn_matrix_multiply_flops_list.append(flops2['mult'])
    lsrn_comm_time_list.append(timing2['comm'])
    lsrn_svd_decomposition_time_list.append(timing2['svd'])
    lsrn_svd_decomposition_flops_list.append(flops2['svd'])
    lsrn_iterative_solver_time_list.append(timing2['iter'])
    lsrn_iterative_solver_flops_list.append(flops2['iter'])
    lsrn_iteration_number_list.append(iternum2)

    _, iternum3, flag3, _, timing3, flops3 = LSRN_over_without_mpi(A, b, tol=tol)

    # lsrn_parallel_relative_normal_equation_error_list.append(norm(A.transpose() @ r2) / (norm(A) * norm(A) * norm(x2)))
    # lsrn_parallel_relative_residual_error_list.append(norm(r2) / norm(b))
    lsrn_parallel_total_computational_time_list.append(timing3['all'])
    lsrn_parallel_generate_random_number_and_matrix_multiply_time_list.append(timing3['randn_and_mult'])
    lsrn_parallel_matrix_multiply_flops_list.append(flops3['mult'])
    lsrn_parallel_comm_time_list.append(timing3['comm'])
    lsrn_parallel_svd_decomposition_time_list.append(timing3['svd'])
    lsrn_parallel_svd_decomposition_flops_list.append(flops3['svd'])
    lsrn_parallel_iterative_solver_time_list.append(timing3['iter'])
    lsrn_parallel_iterative_solver_flops_list.append(flops3['iter'])
    lsrn_parallel_iteration_number_list.append(iternum3)

    # t3 = perf_counter()
    # x2, iternum2, flag2, _ = LSRN_over_diff_sketch(A, b, sketch="gaussian", tol=1e-14)
    # t4 = perf_counter() - t3
    # r2 = b - A @ x2
    #
    # print("\nLSRN With Gaussian Sketch:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r2) / (norm(A) * norm(A) * norm(x2)))
    # print("\tRelative Residual Error:", norm(r2) / norm(b))
    # print("\tComputational time (sec.):", t4)
    # print("\tThe iteration number is:", iternum2)
    # print("\tThe flag is:", flag2)
    #
    # t5 = perf_counter()
    # x3, iternum3, flag3, _ = LSRN_over_diff_sketch(A, b, sketch="srht", tol=1e-14)
    # t6 = perf_counter() - t5
    # r3 = b - A @ x3
    #
    # print("\nLSRN With Subsampled Randomized Hadamard Transform Sketch:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r3) / (norm(A) * norm(A) * norm(x3)))
    # print("\tRelative Residual Error:", norm(r3) / norm(b))
    # print("\tComputational time (sec.):", t6)
    # print("\tThe iteration number is:", iternum3)
    # print("\tThe flag is:", flag3)
    #
    # t7 = perf_counter()
    # x4, iternum4, flag4, _ = LSRN_over_diff_sketch(A, b, sketch="sparsejl", tol=1e-14)
    # t8 = perf_counter() - t7
    # r4 = b - A @ x4
    #
    # print("\nLSRN With Sparse Johnson Lindenstrauss Transforms Sketch:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r4) / (norm(A) * norm(A) * norm(x4)))
    # print("\tRelative Residual Error:", norm(r4) / norm(b))
    # print("\tComputational time (sec.):", t8)
    # print("\tThe iteration number is:", iternum4)
    # print("\tThe flag is:", flag4)

    # Riley's Blendenpik
    multiplier = 2
    d = multiplier * A.shape[1]
    tol = tol

    # t9 = perf_counter()
    x5, flag5, iternum5, _, timing, flops = blendenpik_srct(A, b, d, tol, 1000)
    # t10 = perf_counter() - t9
    r5 = b - A @ x5

    riley_blen_relative_normal_equation_list.append(norm(A.transpose() @ r5) / (norm(A) * norm(A) * norm(x5)))
    riley_blen_relative_residual_list.append(norm(r5) / norm(b))
    riley_blen_total_computational_time_list.append(timing['all'])
    riley_blen_SRCT_matrix_multiply_time_list.append(timing['srct_and_mult'])
    riley_blen_SRCT_matrix_multiply_flops_list.append(flops['srct_and_mult'])
    riley_blen_QR_decomposition_time_list.append(timing['qr'])
    riley_blen_QR_decomposition_flops_list.append(flops['qr'])
    riley_blen_iterative_solver_time_list.append(timing['iter'])
    riley_blen_iterative_solver_flops_list.append(flops['iter'])
    riley_blen_iteration_number_list.append(iternum5)

    # # Solve with QR
    # t11 = perf_counter()
    # x6 = solve_ls_with_QR(A, b)
    # t12 = perf_counter() - t11
    # r6 = b - A @ x6
    #
    # print("\nSolve with QR decomposition:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r6) / (norm(A) * norm(A) * norm(x6)))
    # print("\tRelative Residual Error:", norm(r6) / norm(b))
    # print("\tComputational time (sec.):", t12)
    #
    # # Solve with normal equation
    # t13 = perf_counter()
    # x7 = solve_ls_with_normal_equation(A, b)
    # t14 = perf_counter() - t13
    # r7 = b - A @ x7
    #
    # print("\nSolve with normal equation:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r7) / (norm(A) * norm(A) * norm(x7)))
    # print("\tRelative Residual Error:", norm(r7) / norm(b))
    # print("\tComputational time (sec.):", t14)
    #
    # # t6 = perf_counter()
    # # x4 = CS(A, b)
    # # t7 = perf_counter() - t6
    # # print("Naive CS Without Preconditioning:")
    # # print("Normal Equation:", np.linalg.norm(A.transpose() @ A @ x4 - A.transpose() @ b, ord=2))
    # # print("Residual (L2-norm):", np.linalg.norm(A @ x4 - b, ord=2))
    # # print("Error:", np.linalg.norm(x4 - x, ord=2) / np.linalg.norm(x, ord=2))
    # # print("Computational time (sec.):", t7)
    #
    # # Solve with QR
    # t15 = perf_counter()
    # x8 = LAPACK_solve_ls_with_QR(A, b)
    # t16 = perf_counter() - t15
    # r8 = b - A @ x8
    #
    # print("\nLAPACK scipy interface solve with QR:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r8) / (norm(A) * norm(A) * norm(x8)))
    # print("\tRelative Residual Error:", norm(r8) / norm(b))
    # print("\tComputational time (sec.):", t16)
    #
    # # Solve with normal equation
    # t17 = perf_counter()
    # x9 = LAPACK_solve_ls_with_normal_equation(A, b)
    # t18 = perf_counter() - t17
    # r9 = b - A @ x9
    #
    # print("\nLACPACK scipy interface solve with normal equation:")
    # print("\tRelative Normal Equation Error:", norm(A.transpose() @ r9) / (norm(A) * norm(A) * norm(x9)))
    # print("\tRelative Residual Error:", norm(r9) / norm(b))
    # print("\tComputational time (sec.):", t18)

print("\nLSRN:")
# print("\tRelative Normal Equation Error:", sum(lsrn_relative_normal_equation_error_list)/len(lsrn_relative_normal_equation_error_list))
# print("\tRelative Residual Error:", sum(lsrn_relative_residual_error_list)/len(lsrn_relative_residual_error_list))
print("\tTotal Computational time (sec.):", sum(lsrn_total_computational_time_list)/len(lsrn_total_computational_time_list))
# print("\tTotal Computational flops:", flops['all'])
print("\tGenerate random numbers time (sec.):", sum(lsrn_generate_random_number_time_list)/len(lsrn_generate_random_number_time_list))
# print("\tGenerate random numbers flops:", flops['randn'])
print("\tMatrix multiplication time (sec.):", sum(lsrn_matrix_multiply_time_list)/len(lsrn_matrix_multiply_time_list))
print("\tMatrix multiplication flops:", sum(lsrn_matrix_multiply_flops_list)/len(lsrn_matrix_multiply_flops_list))
print("\tComm time (sec.):", sum(lsrn_comm_time_list)/len(lsrn_comm_time_list))
print("\tSVD decomposition time (sec.):", sum(lsrn_svd_decomposition_time_list)/len(lsrn_svd_decomposition_time_list))
print("\tSVD decomposition flops:", sum(lsrn_svd_decomposition_flops_list)/len(lsrn_svd_decomposition_flops_list))
print("\tIterative solver time (sec.):", sum(lsrn_iterative_solver_time_list)/len(lsrn_iterative_solver_time_list))
print("\tIterative solver flops:", sum(lsrn_iterative_solver_flops_list)/len(lsrn_iterative_solver_flops_list))
print("\tThe iteration number is:", sum(lsrn_iteration_number_list)/len(lsrn_iteration_number_list))
# print("\tThe flag is:", flag2)

print("\nLSRN Parallelized:")
# print("\tRelative Normal Equation Error:", sum(lsrn_parallel_relative_normal_equation_error_list)/len(lsrn_parallel_relative_normal_equation_error_list))
# print("\tRelative Residual Error:", sum(lsrn_parallel_relative_residual_error_list)/len(lsrn_parallel_relative_residual_error_list))
print("\tTotal Computational time (sec.):", sum(lsrn_parallel_total_computational_time_list)/len(lsrn_parallel_total_computational_time_list))
# print("\tTotal Computational flops:", flops['all'])
print("\tGenerate random numbers and Matrix multiplication time (sec.):", sum(lsrn_parallel_generate_random_number_and_matrix_multiply_time_list)/len(lsrn_parallel_generate_random_number_and_matrix_multiply_time_list))
# print("\tGenerate random numbers flops:", flops['randn'])
print("\tMatrix multiplication flops:", sum(lsrn_parallel_matrix_multiply_flops_list)/len(lsrn_parallel_matrix_multiply_flops_list))
print("\tComm time (sec.):", sum(lsrn_parallel_comm_time_list)/len(lsrn_parallel_comm_time_list))
print("\tSVD decomposition time (sec.):", sum(lsrn_parallel_svd_decomposition_time_list)/len(lsrn_parallel_svd_decomposition_time_list))
print("\tSVD decomposition flops:", sum(lsrn_parallel_svd_decomposition_flops_list)/len(lsrn_parallel_svd_decomposition_flops_list))
print("\tIterative solver time (sec.):", sum(lsrn_parallel_iterative_solver_time_list)/len(lsrn_parallel_iterative_solver_time_list))
print("\tIterative solver flops:", sum(lsrn_parallel_iterative_solver_flops_list)/len(lsrn_parallel_iterative_solver_flops_list))
print("\tThe iteration number is:", sum(lsrn_parallel_iteration_number_list)/len(lsrn_parallel_iteration_number_list))

print("\nRiley's Blendenpik With Scipy LSQR:")
print("\tRelative Normal Equation Error:", sum(riley_blen_relative_normal_equation_list)/len(riley_blen_relative_normal_equation_list))
print("\tRelative Residual Error:", sum(riley_blen_relative_residual_list)/len(riley_blen_relative_residual_list))
print("\tTotal Computational time (sec.):", sum(riley_blen_total_computational_time_list)/len(riley_blen_total_computational_time_list))
# print("\tTotal Computational flops:", flops['all'])
print("\tSRCT & Matrix multiplication time (sec.):", sum(riley_blen_SRCT_matrix_multiply_time_list)/len(riley_blen_SRCT_matrix_multiply_time_list))
print("\tSRCT & Matrix multiplication flops:", sum(riley_blen_SRCT_matrix_multiply_flops_list)/len(riley_blen_SRCT_matrix_multiply_flops_list))
print("\tQR decomposition time (sec.):", sum(riley_blen_QR_decomposition_time_list)/len(riley_blen_QR_decomposition_time_list))
print("\tQR decomposition time flops:", sum(riley_blen_QR_decomposition_flops_list)/len(riley_blen_QR_decomposition_flops_list))
print("\tIterative solver time (sec.):", sum(riley_blen_iterative_solver_time_list)/len(riley_blen_iterative_solver_time_list))
print("\tIterative solver flops:", sum(riley_blen_iterative_solver_flops_list)/len(riley_blen_iterative_solver_flops_list))
print("\tIteration number:", sum(riley_blen_iteration_number_list)/len(riley_blen_iteration_number_list))
# print("\tThe flag is:", flag5)