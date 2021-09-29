import csv
import numpy as np
from matplotlib import pyplot as plt
from Haoyun.randomized_least_square_solver.Test.Blendenpik.Riley_Blendenpik_new import blendenpik_srct
from test_matrix_generator import overdetermined_ls_test_matrix_generator
from Haoyun.randomized_least_square_solver.Test.LSRN import LSRN_over

# Set the tolerance to be 1e-12
tol = 1e-12

# Set the condition number to be be low, medium and high, which is
# corresponding to 1e3, 1e8 and 1e13
cond_num_array = np.array([1e3, 1e8, 1e13])
cond_num_array_length = 3
condition_number_type_array = np.array(['Low', 'Medium', 'High'])
condition_number_type_array_length = 3
# # Choose some different randomized matrices with different random number seeds
seednum_array_length = 50
seednum_array = np.random.choice(10000, seednum_array_length, replace=False)

# Choose the range of oversampling factor to be 1.6 to 2.5, which is specified by the LSRN paper
oversampling_factor_array = np.arange(1.5, 10.5, 0.5)
oversampling_factor_array_length = len(oversampling_factor_array)


# # Create the matrix to store different batches of different condition condition number of LSRN and Blendenpik
# # lsrn_relative_normal_equation_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# # lsrn_relative_residual_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# lsrn_total_computational_time_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_total_computational_flops_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_time_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_time_without_rand_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_flops_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iterative_solver_time_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iterative_solver_flops_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iteration_number_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
#
# # riley_blen_relative_normal_equation_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# # riley_blen_relative_residual_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# riley_blen_total_computational_time_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_total_computational_flops_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_precondition_time_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_precondition_flops_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iterative_solver_time_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iterative_solver_flops_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iteration_number_diff_cond_mean_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
#
# # Create the matrix to store different batches of different condition condition number of LSRN and Blendenpik
# # lsrn_relative_normal_equation_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# # lsrn_relative_residual_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# lsrn_total_computational_time_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_total_computational_flops_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_time_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_time_without_rand_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_flops_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iterative_solver_time_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iterative_solver_flops_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iteration_number_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
#
# # riley_blen_relative_normal_equation_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# # riley_blen_relative_residual_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# riley_blen_total_computational_time_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_total_computational_flops_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_precondition_time_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_precondition_flops_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iterative_solver_time_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iterative_solver_flops_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iteration_number_diff_cond_5_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
#
# # lsrn_relative_normal_equation_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# # lsrn_relative_residual_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# lsrn_total_computational_time_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_total_computational_flops_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_time_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_time_without_rand_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_precondition_flops_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iterative_solver_time_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iterative_solver_flops_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# lsrn_iteration_number_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
#
# # riley_blen_relative_normal_equation_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# # riley_blen_relative_residual_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
# riley_blen_total_computational_time_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_total_computational_flops_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_precondition_time_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_precondition_flops_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iterative_solver_time_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iterative_solver_flops_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))
# riley_blen_iteration_number_diff_cond_95_percent_matrix = np.zeros((oversampling_factor_array_length, cond_num_array_length))

for cond_num_index in np.arange(cond_num_array_length):

    cond_num = cond_num_array[cond_num_index]

    # Create the matrix to store different batches of LSRN and Blendenpik
    # lsrn_relative_normal_equation_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    # lsrn_relative_residual_error_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_total_computational_time_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_total_computational_flops_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_precondition_time_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_precondition_time_without_rand_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_precondition_flops_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_iterative_solver_time_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_iterative_solver_flops_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    lsrn_iteration_number_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))

    # riley_blen_relative_normal_equation_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    # riley_blen_relative_residual_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    riley_blen_total_computational_time_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    riley_blen_total_computational_flops_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    riley_blen_precondition_time_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    riley_blen_precondition_flops_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    riley_blen_iterative_solver_time_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    riley_blen_iterative_solver_flops_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))
    riley_blen_iteration_number_matrix = np.zeros((oversampling_factor_array_length, seednum_array_length))

    for oversampling_factor_index in np.arange(oversampling_factor_array_length):
        oversampling_factor = oversampling_factor_array[oversampling_factor_index]
        for seednum_index in np.arange(seednum_array_length):
            seednum = seednum_array[seednum_index]
            A, _, b = overdetermined_ls_test_matrix_generator(m=5000,
                                                              n=250,
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

            x2, iternum2, flag2, _, timing2, flops2 = LSRN_over(A, b, gamma=oversampling_factor, tol=tol)
            r2 = b - A @ x2

            # lsrn_relative_normal_equation_error_matrix[oversampling_factor_index, seednum_index] = norm(A.transpose() @ r2) / (norm(A) * norm(A) * norm(x2))
            # lsrn_relative_residual_error_matrix[oversampling_factor_index, seednum_index] = norm(r2) / norm(b)
            lsrn_total_computational_time_matrix[oversampling_factor_index, seednum_index] = timing2['all']
            lsrn_total_computational_flops_matrix[oversampling_factor_index, seednum_index] = flops2['all']
            lsrn_precondition_time_matrix[oversampling_factor_index, seednum_index] = timing2['randn'] + timing2['mult'] + timing2['svd']
            lsrn_precondition_time_without_rand_matrix[oversampling_factor_index, seednum_index] = timing2['mult'] + timing2['svd']
            lsrn_precondition_flops_matrix[oversampling_factor_index, seednum_index] = flops2['randn'] + flops2['mult'] + flops2['svd']
            lsrn_iterative_solver_time_matrix[oversampling_factor_index, seednum_index] = timing2['iter']
            lsrn_iterative_solver_flops_matrix[oversampling_factor_index, seednum_index] = flops2['iter']
            lsrn_iteration_number_matrix[oversampling_factor_index, seednum_index] = iternum2

            # Riley's Blendenpik
            multiplier = oversampling_factor
            d = int(multiplier * A.shape[1])
            tol = tol

            x3, flag3, iternum3, _, timing3, flops3 = blendenpik_srct(A, b, d, tol, 1000)
            r3 = b - A @ x3

            # riley_blen_relative_normal_equation_matrix[oversampling_factor_index, seednum_index] = norm(A.transpose() @ r3) / (norm(A) * norm(A) * norm(x3))
            # riley_blen_relative_residual_matrix[oversampling_factor_index, seednum_index] = norm(r3) / norm(b)
            riley_blen_total_computational_time_matrix[oversampling_factor_index, seednum_index] = timing3['all']
            riley_blen_total_computational_flops_matrix[oversampling_factor_index, seednum_index] = flops3['all']
            riley_blen_precondition_time_matrix[oversampling_factor_index, seednum_index] = timing3['srct_and_mult'] + timing3['qr']
            riley_blen_precondition_flops_matrix[oversampling_factor_index, seednum_index] = flops3['srct_and_mult'] + flops3['qr']
            riley_blen_iterative_solver_time_matrix[oversampling_factor_index, seednum_index] = timing3['iter']
            riley_blen_iterative_solver_flops_matrix[oversampling_factor_index, seednum_index] = flops3['iter']
            riley_blen_iteration_number_matrix[oversampling_factor_index, seednum_index] = iternum3

    # Those data in the csv files, x-axis represents different oversampling factor,
    # while y-axis reprents different random number seed

    # Store the data of LSRN

    # name of csv file
    filename = "Time/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_total_computational_time.csv"

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the data rows
        csvwriter.writerows(lsrn_total_computational_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_precondition_time.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_precondition_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_precondition_time_without_rand.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_precondition_time_without_rand_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_iterative_solver_time.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_iterative_solver_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_iterative_solver_time.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_iterative_solver_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_iteration_number.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_iteration_number_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_total_computational_flops.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_total_computational_flops_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_precondition_flops.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_precondition_flops_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_iterative_solver_flops.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_iterative_solver_flops_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/LSRN/lsrn_iteration_number.csv"

    # Store the data of Blendenpik

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(riley_blen_total_computational_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_total_computational_time.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_total_computational_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_precondition_time.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(riley_blen_precondition_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_iterative_solver_time.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(riley_blen_iterative_solver_time_matrix)

    filename = "Time/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_iteration_number.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(riley_blen_iteration_number_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_total_computational_flops.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(riley_blen_total_computational_flops_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_precondition_flops.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(riley_blen_precondition_flops_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_iterative_solver_flops.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(lsrn_total_computational_flops_matrix)

    filename = "Flops Rate/" + condition_number_type_array[cond_num_index] + "/Blendenpik/riley_blen_iteration_number.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(riley_blen_iteration_number_matrix)

    # Calculate mean, five percentile and ninety five percentile of the original data
    lsrn_total_computational_time_mean_array = np.mean(lsrn_total_computational_time_matrix, axis=1)
    lsrn_total_computational_flops_mean_array = np.mean(lsrn_total_computational_flops_matrix, axis=1)
    lsrn_precondition_time_mean_array = np.mean(lsrn_precondition_time_matrix, axis=1)
    lsrn_precondition_time_without_rand_mean_array = np.mean(lsrn_precondition_time_without_rand_matrix, axis=1)
    lsrn_precondition_flops_mean_array = np.mean(lsrn_precondition_flops_matrix, axis=1)
    lsrn_iterative_solver_time_mean_array = np.mean(lsrn_iterative_solver_time_matrix, axis=1)
    lsrn_iterative_solver_flops_mean_array = np.mean(lsrn_iterative_solver_flops_matrix, axis=1)
    lsrn_iteration_number_mean_array = np.mean(lsrn_iteration_number_matrix, axis=1)

    lsrn_total_computational_time_5_percent_array = np.percentile(lsrn_total_computational_time_matrix, 5, axis=1)
    lsrn_total_computational_flops_5_percent_array = np.percentile(lsrn_total_computational_flops_matrix, 5, axis=1)
    lsrn_precondition_time_5_percent_array = np.percentile(lsrn_precondition_time_matrix, 5, axis=1)
    lsrn_precondition_time_without_rand_5_percent_array = np.percentile(lsrn_precondition_time_without_rand_matrix, 5, axis=1)
    lsrn_precondition_flops_5_percent_array = np.percentile(lsrn_precondition_flops_matrix, 5, axis=1)
    lsrn_iterative_solver_time_5_percent_array = np.percentile(lsrn_iterative_solver_time_matrix, 5, axis=1)
    lsrn_iterative_solver_flops_5_percent_array = np.percentile(lsrn_iterative_solver_flops_matrix, 5, axis=1)
    lsrn_iteration_number_5_percent_array = np.percentile(lsrn_iteration_number_matrix, 5, axis=1)

    lsrn_total_computational_time_95_percent_array = np.percentile(lsrn_total_computational_time_matrix, 95, axis=1)
    lsrn_total_computational_flops_95_percent_array = np.percentile(lsrn_total_computational_flops_matrix, 95, axis=1)
    lsrn_precondition_time_95_percent_array = np.percentile(lsrn_precondition_time_matrix, 95, axis=1)
    lsrn_precondition_time_without_rand_95_percent_array = np.percentile(lsrn_precondition_time_without_rand_matrix, 95, axis=1)
    lsrn_precondition_flops_95_percent_array = np.percentile(lsrn_precondition_flops_matrix, 95, axis=1)
    lsrn_iterative_solver_time_95_percent_array = np.percentile(lsrn_iterative_solver_time_matrix, 95, axis=1)
    lsrn_iterative_solver_flops_95_percent_array = np.percentile(lsrn_iterative_solver_flops_matrix, 95, axis=1)
    lsrn_iteration_number_95_percent_array = np.percentile(lsrn_iteration_number_matrix, 95, axis=1)

    riley_blen_total_computational_time_mean_array = np.mean(riley_blen_total_computational_time_matrix, axis=1)
    riley_blen_total_computational_flops_mean_array = np.mean(riley_blen_total_computational_flops_matrix, axis=1)
    riley_blen_precondition_time_mean_array = np.mean(riley_blen_precondition_time_matrix, axis=1)
    riley_blen_precondition_flops_mean_array = np.mean(riley_blen_precondition_flops_matrix, axis=1)
    riley_blen_iterative_solver_time_mean_array = np.mean(riley_blen_iterative_solver_time_matrix, axis=1)
    riley_blen_iterative_solver_flops_mean_array = np.mean(riley_blen_iterative_solver_flops_matrix, axis=1)
    riley_blen_iteration_number_mean_array = np.mean(riley_blen_iteration_number_matrix, axis=1)

    riley_blen_total_computational_time_5_percent_array = np.percentile(riley_blen_total_computational_time_matrix, 5, axis=1)
    riley_blen_total_computational_flops_5_percent_array = np.percentile(riley_blen_total_computational_flops_matrix, 5, axis=1)
    riley_blen_precondition_time_5_percent_array = np.percentile(riley_blen_precondition_time_matrix, 5, axis=1)
    riley_blen_precondition_flops_5_percent_array = np.percentile(riley_blen_precondition_flops_matrix, 5, axis=1)
    riley_blen_iterative_solver_time_5_percent_array = np.percentile(riley_blen_iterative_solver_time_matrix, 5, axis=1)
    riley_blen_iterative_solver_flops_5_percent_array = np.percentile(riley_blen_iterative_solver_flops_matrix, 5, axis=1)
    riley_blen_iteration_number_5_percent_array = np.percentile(riley_blen_iteration_number_matrix, 5, axis=1)

    riley_blen_total_computational_time_95_percent_array = np.percentile(riley_blen_total_computational_time_matrix, 95, axis=1)
    riley_blen_total_computational_flops_95_percent_array = np.percentile(riley_blen_total_computational_flops_matrix, 95, axis=1)
    riley_blen_precondition_time_95_percent_array = np.percentile(riley_blen_precondition_time_matrix, 95, axis=1)
    riley_blen_precondition_flops_95_percent_array = np.percentile(riley_blen_precondition_flops_matrix, 95, axis=1)
    riley_blen_iterative_solver_time_95_percent_array = np.percentile(riley_blen_iterative_solver_time_matrix, 95, axis=1)
    riley_blen_iterative_solver_flops_95_percent_array = np.percentile(riley_blen_iterative_solver_flops_matrix, 95, axis=1)
    riley_blen_iteration_number_95_percent_array = np.percentile(riley_blen_iteration_number_matrix, 95, axis=1)

    # Draw the 2D plots for running time of LSRN and Blendenpik
    condition_number_type = condition_number_type_array[cond_num_index]
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 32))
    st = fig.suptitle('Time ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch', fontsize=20)
    st.set_y(0.95)

    mean_array = np.array([[lsrn_precondition_time_mean_array, riley_blen_precondition_time_mean_array],
                           [lsrn_iterative_solver_time_mean_array, riley_blen_iterative_solver_time_mean_array],
                           [lsrn_total_computational_time_mean_array, riley_blen_total_computational_time_mean_array],
                           [lsrn_iteration_number_mean_array, riley_blen_iteration_number_mean_array]])

    five_percent_array = np.array([[lsrn_precondition_time_5_percent_array, riley_blen_precondition_time_5_percent_array],
                                   [lsrn_iterative_solver_time_5_percent_array, riley_blen_iterative_solver_time_5_percent_array],
                                   [lsrn_total_computational_time_5_percent_array, riley_blen_total_computational_time_5_percent_array],
                                   [lsrn_iteration_number_5_percent_array, riley_blen_iteration_number_5_percent_array]])

    ninety_percent_array = np.array([[lsrn_precondition_time_95_percent_array, riley_blen_precondition_time_95_percent_array],
                                     [lsrn_iterative_solver_time_95_percent_array, riley_blen_iterative_solver_time_95_percent_array],
                                     [lsrn_total_computational_time_95_percent_array, riley_blen_total_computational_time_95_percent_array],
                                     [lsrn_iteration_number_95_percent_array, riley_blen_iteration_number_95_percent_array]])

    time_array = ['Precondition', 'Iterative Step', 'Total', 'Iteration Number']
    # flops_array = ['Precondition', 'Iterative Step', 'Total']
    algo_array = ["LSRN", "Riley's Blendenpik"]

    for i in np.arange(len(time_array)):
        for j in np.arange(len(algo_array)):
            if i == 3:
                axes[i, j].plot(oversampling_factor_array, mean_array[i, j], label="mean")
                axes[i, j].plot(oversampling_factor_array, five_percent_array[i, j], label='five percent')
                axes[i, j].plot(oversampling_factor_array, ninety_percent_array[i, j], label='ninety five percent')
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('Iteration Number')
                axes[i, j].set_title(time_array[i] + ' of ' + algo_array[j])
                axes[i, j].legend()
            else:
                axes[i, j].plot(oversampling_factor_array, mean_array[i, j], label="mean")
                axes[i, j].plot(oversampling_factor_array, five_percent_array[i, j], label='five percent')
                axes[i, j].plot(oversampling_factor_array, ninety_percent_array[i, j], label='ninety five percent')
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('time(Sec.)')
                axes[i, j].set_title(time_array[i] + ' Time of ' + algo_array[j])
                axes[i, j].legend()

    plt.savefig('Time/Time ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch' '.png')
    # plt.show()

    # Draw the 2D plots for flops of LSRN and Blendenpik
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 32))
    st = fig.suptitle('Flops Rate ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch', fontsize=20)
    st.set_y(0.95)

    mean_array = np.array([[np.divide(lsrn_precondition_flops_mean_array, lsrn_precondition_time_without_rand_mean_array),
                            np.divide(riley_blen_precondition_flops_mean_array, riley_blen_precondition_time_mean_array)],
                           [np.divide(lsrn_iterative_solver_flops_mean_array, lsrn_iterative_solver_time_mean_array),
                            np.divide(riley_blen_iterative_solver_flops_mean_array, riley_blen_iterative_solver_time_mean_array)],
                           [np.divide(lsrn_total_computational_flops_mean_array, lsrn_total_computational_time_mean_array),
                            np.divide(riley_blen_total_computational_flops_mean_array, riley_blen_total_computational_time_mean_array)],
                           [lsrn_iteration_number_mean_array, riley_blen_iteration_number_mean_array]])

    five_percent_array = np.array([[np.divide(lsrn_precondition_flops_5_percent_array,
                                              lsrn_precondition_time_without_rand_5_percent_array),
                                    np.divide(riley_blen_precondition_flops_5_percent_array,
                                              riley_blen_precondition_time_5_percent_array)],
                                   [np.divide(lsrn_iterative_solver_flops_5_percent_array,
                                              lsrn_iterative_solver_time_mean_array),
                                    np.divide(riley_blen_iterative_solver_flops_5_percent_array,
                                              riley_blen_iterative_solver_time_5_percent_array)],
                                   [np.divide(lsrn_total_computational_flops_5_percent_array,
                                              lsrn_total_computational_time_5_percent_array),
                                    np.divide(riley_blen_total_computational_flops_5_percent_array,
                                              riley_blen_total_computational_time_5_percent_array)],
                                   [lsrn_iteration_number_5_percent_array, riley_blen_iteration_number_5_percent_array]])

    ninety_percent_array = np.array([[np.divide(lsrn_precondition_flops_95_percent_array,
                                                lsrn_precondition_time_without_rand_95_percent_array),
                                      np.divide(riley_blen_precondition_flops_95_percent_array,
                                                riley_blen_precondition_time_95_percent_array)],
                                     [np.divide(lsrn_iterative_solver_flops_95_percent_array,
                                                lsrn_iterative_solver_time_95_percent_array),
                                      np.divide(riley_blen_iterative_solver_flops_95_percent_array,
                                                riley_blen_iterative_solver_time_95_percent_array)],
                                     [np.divide(lsrn_total_computational_flops_95_percent_array,
                                                lsrn_total_computational_time_95_percent_array),
                                      np.divide(riley_blen_total_computational_flops_95_percent_array,
                                                riley_blen_total_computational_time_95_percent_array)],
                                     [lsrn_iteration_number_95_percent_array, riley_blen_iteration_number_95_percent_array]])

    flops_array = ['Precondition', 'Iterative Step', 'Total', 'Iteration Number']
    algo_array = ["LSRN", "Riley's Blendenpik"]

    for i in np.arange(len(time_array)):
        for j in np.arange(len(algo_array)):
            if i == 3:
                axes[i, j].plot(oversampling_factor_array, mean_array[i, j])
                axes[i, j].plot(oversampling_factor_array, five_percent_array[i, j], label='five percent')
                axes[i, j].plot(oversampling_factor_array, ninety_percent_array[i, j], label='ninety five percent')
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('Iteration Number')
                axes[i, j].set_title(time_array[i] + ' of ' + algo_array[j])
                axes[i, j].legend()
            else:
                axes[i, j].plot(oversampling_factor_array, mean_array[i, j])
                axes[i, j].plot(oversampling_factor_array, five_percent_array[i, j], label='five percent')
                axes[i, j].plot(oversampling_factor_array, ninety_percent_array[i, j], label='ninety five percent')
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('Flops/Sec.')
                axes[i, j].set_title(time_array[i] + ' Flops Rate of ' + algo_array[j])
                axes[i, j].legend()

    plt.savefig('Flops Rate/Flops Rate ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch' '.png')
    # plt.show()

    #
    # # store into mean diff cond matrix
    # lsrn_total_computational_time_diff_cond_mean_matrix[:, cond_num_index] = lsrn_total_computational_time_mean_array
    # lsrn_total_computational_flops_diff_cond_mean_matrix[:, cond_num_index] = lsrn_total_computational_flops_mean_array
    # lsrn_precondition_time_diff_cond_mean_matrix[:, cond_num_index] = lsrn_precondition_time_mean_array
    # lsrn_precondition_time_without_rand_diff_cond_mean_matrix[:, cond_num_index] = lsrn_precondition_time_without_rand_mean_array
    # lsrn_precondition_flops_diff_cond_mean_matrix[:, cond_num_index] = lsrn_precondition_flops_mean_array
    # lsrn_iterative_solver_time_diff_cond_mean_matrix[:, cond_num_index] = lsrn_iterative_solver_time_mean_array
    # lsrn_iterative_solver_flops_diff_cond_mean_matrix[:, cond_num_index] = lsrn_iterative_solver_flops_mean_array
    # lsrn_iteration_number_diff_cond_mean_matrix[:, cond_num_index] = lsrn_iteration_number_mean_array
    #
    # riley_blen_total_computational_time_diff_cond_mean_matrix[:, cond_num_index] = riley_blen_total_computational_time_mean_array
    # riley_blen_total_computational_flops_diff_cond_mean_matrix[:, cond_num_index] = riley_blen_total_computational_flops_mean_array
    # riley_blen_precondition_time_diff_cond_mean_matrix[:, cond_num_index] = riley_blen_precondition_time_mean_array
    # riley_blen_precondition_flops_diff_cond_mean_matrix[:, cond_num_index] = riley_blen_precondition_flops_mean_array
    # riley_blen_iterative_solver_time_diff_cond_mean_matrix[:, cond_num_index] = riley_blen_iterative_solver_time_mean_array
    # riley_blen_iterative_solver_flops_diff_cond_mean_matrix[:, cond_num_index] = riley_blen_iterative_solver_flops_mean_array
    # riley_blen_iteration_number_diff_cond_mean_matrix[:, cond_num_index] = riley_blen_iteration_number_mean_array
    #
    # lsrn_total_computational_time_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_total_computational_time_5_percent_array
    # lsrn_total_computational_flops_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_total_computational_flops_5_percent_array
    # lsrn_precondition_time_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_precondition_time_5_percent_array
    # lsrn_precondition_time_without_rand_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_precondition_time_without_rand_5_percent_array
    # lsrn_precondition_flops_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_precondition_flops_5_percent_array
    # lsrn_iterative_solver_time_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_iterative_solver_time_5_percent_array
    # lsrn_iterative_solver_flops_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_iterative_solver_flops_5_percent_array
    # lsrn_iteration_number_diff_cond_5_percent_matrix[:, cond_num_index] = lsrn_iteration_number_5_percent_array
    #
    # riley_blen_total_computational_time_diff_cond_5_percent_matrix[:, cond_num_index] = riley_blen_total_computational_time_5_percent_array
    # riley_blen_total_computational_flops_diff_cond_5_percent_matrix[:, cond_num_index] = riley_blen_total_computational_flops_5_percent_array
    # riley_blen_precondition_time_diff_cond_5_percent_matrix[:, cond_num_index] = riley_blen_precondition_time_5_percent_array
    # riley_blen_precondition_flops_diff_cond_5_percent_matrix[:, cond_num_index] = riley_blen_precondition_flops_5_percent_array
    # riley_blen_iterative_solver_time_diff_cond_5_percent_matrix[:, cond_num_index] = riley_blen_iterative_solver_time_5_percent_array
    # riley_blen_iterative_solver_flops_diff_cond_5_percent_matrix[:, cond_num_index] = riley_blen_iterative_solver_flops_5_percent_array
    # riley_blen_iteration_number_diff_cond_5_percent_matrix[:, cond_num_index] = riley_blen_iteration_number_5_percent_array
    #
    # lsrn_total_computational_time_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_total_computational_time_95_percent_array
    # lsrn_total_computational_flops_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_total_computational_flops_95_percent_array
    # lsrn_precondition_time_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_precondition_time_95_percent_array
    # lsrn_precondition_time_without_rand_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_precondition_time_without_rand_95_percent_array
    # lsrn_precondition_flops_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_precondition_flops_95_percent_array
    # lsrn_iterative_solver_time_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_iterative_solver_time_95_percent_array
    # lsrn_iterative_solver_flops_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_iterative_solver_flops_95_percent_array
    # lsrn_iteration_number_diff_cond_95_percent_matrix[:, cond_num_index] = lsrn_iteration_number_95_percent_array
    #
    # riley_blen_total_computational_time_diff_cond_95_percent_matrix[:, cond_num_index] = riley_blen_total_computational_time_95_percent_array
    # riley_blen_total_computational_flops_diff_cond_95_percent_matrix[:, cond_num_index] = riley_blen_total_computational_flops_95_percent_array
    # riley_blen_precondition_time_diff_cond_95_percent_matrix[:, cond_num_index] = riley_blen_precondition_time_95_percent_array
    # riley_blen_precondition_flops_diff_cond_95_percent_matrix[:, cond_num_index] = riley_blen_precondition_flops_95_percent_array
    # riley_blen_iterative_solver_time_diff_cond_95_percent_matrix[:, cond_num_index] = riley_blen_iterative_solver_time_95_percent_array
    # riley_blen_iterative_solver_flops_diff_cond_95_percent_matrix[:, cond_num_index] = riley_blen_iterative_solver_flops_95_percent_array
    # riley_blen_iteration_number_diff_cond_95_percent_matrix[:, cond_num_index] = riley_blen_iteration_number_95_percent_array



# Draw the Heat Maps for running time of LSRN and Blendenpik


# data_min = np.min([np.min(lsrn_precondition_time_diff_cond_mean_matrix), np.min(riley_blen_precondition_time_diff_cond_mean_matrix),
#                    np.min(lsrn_iterative_solver_time_diff_cond_mean_matrix), np.min(riley_blen_iterative_solver_time_diff_cond_mean_matrix)])
# data_max = np.max([np.max(lsrn_total_computational_time_diff_cond_mean_matrix), np.max(riley_blen_total_computational_time_diff_cond_mean_matrix)])
# fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 32))
# cmap = cm.get_cmap('viridis')
# normalizer = Normalize(data_min, data_max)
# im = cm.ScalarMappable(norm=normalizer)
#
# # iter_solver_array = ["Naive LSQR", "LSRN", "Riley's Blendenpik"]
# iter_solver_array = ["LSRN", "Riley's Blendenpik"]
# mean_array = np.array([[lsrn_precondition_time_diff_cond_mean_matrix, riley_blen_precondition_time_diff_cond_mean_matrix],
#                        [lsrn_iterative_solver_time_diff_cond_mean_matrix, riley_blen_iterative_solver_time_diff_cond_mean_matrix],
#                        [lsrn_total_computational_time_diff_cond_mean_matrix, riley_blen_total_computational_time_diff_cond_mean_matrix],
#                        [lsrn_iteration_number_diff_cond_mean_matrix, riley_blen_iteration_number_diff_cond_mean_matrix]])
#
# # five_percent_array = np.array([[lsrn_precondition_time_5_percent_array, riley_blen_precondition_time_5_percent_array],
# #                                [lsrn_iterative_solver_time_5_percent_array,
# #                                 riley_blen_iterative_solver_time_5_percent_array],
# #                                [lsrn_total_computational_time_5_percent_array,
# #                                 riley_blen_total_computational_time_5_percent_array]])
# #
# # ninety_percent_array = np.array(
# #     [[lsrn_precondition_time_95_percent_array, riley_blen_precondition_time_95_percent_array],
# #      [lsrn_iterative_solver_time_95_percent_array, riley_blen_iterative_solver_time_95_percent_array],
# #      [lsrn_total_computational_time_95_percent_array, riley_blen_total_computational_time_95_percent_array]])
# time_array = ['Precondition', 'Iterative Step', 'Total', 'Iteration Number']
# for i in np.arange(len(time_array)):
#     for j in np.arange(len(iter_solver_array)):
#         if i == 3:
#             axes[i, j].set_ylabel('oversampling factor')
#             axes[i, j].set_xlabel('log10(condition number)')
#             axes[i, j].set_yticks(np.arange(oversampling_factor_array_length))
#             axes[i, j].set_xticks(np.arange(cond_num_array_length))
#             axes[i, j].set_yticklabels(oversampling_factor_array)
#             axes[i, j].set_xticklabels(np.log10(cond_num_array).astype(int))
#             axes[i, j].set_title(time_array[i] + ' Time of ' + iter_solver_array[j])
#             axes[i, j].imshow(mean_array[i, j], cmap=cmap)
#         else:
#             axes[i, j].set_ylabel('oversampling factor')
#             axes[i, j].set_xlabel('log10(condition number)')
#             axes[i, j].set_yticks(np.arange(oversampling_factor_array_length))
#             axes[i, j].set_xticks(np.arange(cond_num_array_length))
#             axes[i, j].set_yticklabels(oversampling_factor_array)
#             axes[i, j].set_xticklabels(np.log10(cond_num_array).astype(int))
#             axes[i, j].set_title(time_array[i] + ' Time of ' + iter_solver_array[j])
#             axes[i, j].imshow(mean_array[i, j], cmap=cmap, norm=normalizer)
#             # for m in range(oversampling_factor_array_length):
#             #     for n in range(cond_num_array_length):
#             #         text = axes[m, n].text(n, m, mean_array[i, j],
#             #                        ha="center", va="center", color="w")
#
# # cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
# # fig.colorbar(im, cax=cbar_ax)
# fig.colorbar(im, ax=axes.ravel().tolist())
# plt.savefig('Time/Time 20 batch HeatMaps.png')
# plt.show()
#
#
# data_min = np.min([np.min(np.divide(lsrn_precondition_flops_diff_cond_mean_matrix, lsrn_precondition_time_without_rand_diff_cond_mean_matrix)), np.min(np.divide(riley_blen_precondition_flops_diff_cond_mean_matrix, riley_blen_precondition_time_diff_cond_mean_matrix)),
#                    np.min(np.divide(lsrn_iterative_solver_flops_diff_cond_mean_matrix, lsrn_iterative_solver_time_diff_cond_mean_matrix)), np.min(np.divide(riley_blen_iterative_solver_flops_diff_cond_mean_matrix, riley_blen_iterative_solver_time_diff_cond_mean_matrix))])
# data_max = np.max([np.max(np.divide(lsrn_total_computational_flops_diff_cond_mean_matrix, lsrn_total_computational_time_diff_cond_mean_matrix)), np.max(np.divide(riley_blen_total_computational_flops_diff_cond_mean_matrix, riley_blen_total_computational_time_diff_cond_mean_matrix))])
# fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 32))
# cmap = cm.get_cmap('viridis')
# normalizer = Normalize(data_min, data_max)
# im = cm.ScalarMappable(norm=normalizer)
#
# # iter_solver_array = ["Naive LSQR", "LSRN", "Riley's Blendenpik"]
# iter_solver_array = ["LSRN", "Riley's Blendenpik"]
# mean_array = np.array([[np.divide(lsrn_precondition_flops_diff_cond_mean_matrix, lsrn_precondition_time_without_rand_diff_cond_mean_matrix), np.divide(riley_blen_precondition_flops_diff_cond_mean_matrix, riley_blen_precondition_time_diff_cond_mean_matrix)],
#                        [np.divide(lsrn_iterative_solver_flops_diff_cond_mean_matrix, lsrn_iterative_solver_time_diff_cond_mean_matrix), np.divide(riley_blen_iterative_solver_flops_diff_cond_mean_matrix, riley_blen_iterative_solver_time_diff_cond_mean_matrix)],
#                        [np.divide(lsrn_total_computational_flops_diff_cond_mean_matrix, lsrn_total_computational_time_diff_cond_mean_matrix), np.divide(riley_blen_total_computational_flops_diff_cond_mean_matrix, riley_blen_total_computational_time_diff_cond_mean_matrix)],
#                        [lsrn_iteration_number_diff_cond_mean_matrix, riley_blen_iteration_number_diff_cond_mean_matrix]])
#
# # five_percent_array = np.array([[lsrn_precondition_time_5_percent_array, riley_blen_precondition_time_5_percent_array],
# #                                [lsrn_iterative_solver_time_5_percent_array,
# #                                 riley_blen_iterative_solver_time_5_percent_array],
# #                                [lsrn_total_computational_time_5_percent_array,
# #                                 riley_blen_total_computational_time_5_percent_array]])
# #
# # ninety_percent_array = np.array(
# #     [[lsrn_precondition_time_95_percent_array, riley_blen_precondition_time_95_percent_array],
# #      [lsrn_iterative_solver_time_95_percent_array, riley_blen_iterative_solver_time_95_percent_array],
# #      [lsrn_total_computational_time_95_percent_array, riley_blen_total_computational_time_95_percent_array]])
# time_array = ['Precondition', 'Iterative Step', 'Total', 'Iteration Number']
# for i in np.arange(len(time_array)):
#     for j in np.arange(len(iter_solver_array)):
#         if i == 3:
#             axes[i, j].set_ylabel('oversampling factor')
#             axes[i, j].set_xlabel('log10(condition number)')
#             axes[i, j].set_yticks(np.arange(oversampling_factor_array_length))
#             axes[i, j].set_xticks(np.arange(cond_num_array_length))
#             axes[i, j].set_yticklabels(oversampling_factor_array)
#             axes[i, j].set_xticklabels(np.log10(cond_num_array).astype(int))
#             axes[i, j].set_title(time_array[i] + ' Flops Rate of ' + iter_solver_array[j])
#             axes[i, j].imshow(mean_array[i, j], cmap=cmap)
#         else:
#             axes[i, j].set_ylabel('oversampling factor')
#             axes[i, j].set_xlabel('log10(condition number)')
#             axes[i, j].set_yticks(np.arange(oversampling_factor_array_length))
#             axes[i, j].set_xticks(np.arange(cond_num_array_length))
#             axes[i, j].set_yticklabels(oversampling_factor_array)
#             axes[i, j].set_xticklabels(np.log10(cond_num_array).astype(int))
#             axes[i, j].set_title(time_array[i] + ' Flops Rate of ' + iter_solver_array[j])
#             axes[i, j].imshow(mean_array[i, j], cmap=cmap, norm=normalizer)
#
# # cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
# # fig.colorbar(im, cax=cbar_ax)
# fig.colorbar(im, ax=axes.ravel().tolist())
# plt.savefig('Flops Rate/Flops Rate 20 batch HeatMaps small test.png')
# plt.show()
