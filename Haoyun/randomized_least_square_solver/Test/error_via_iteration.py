import numpy as np
from matplotlib import pyplot as plt

from Haoyun.randomized_least_square_solver.Blendenpik.Riley_Blen_Scipy_LSQR_for_error_test import \
    blendenpik_srct_scipy_lsqr_for_error_test
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy
from Haoyun.randomized_least_square_solver.LSRN.LSRN_over_for_error_test import LSRN_over_for_error_test
from test_matrix_generator import overdetermined_ls_test_matrix_generator

condition_number_array = 10 ** np.array([4, 8, 12, 16])
seednum_array = [123, 257, 396, 480]

for condition_number in condition_number_array:
    naive_lsqr_absolute_residual_error_array_list = []
    naive_lsqr_absolute_normal_equation_error_array_list = []
    naive_lsqr_relative_residual_error_array_list = []
    naive_lsqr_relative_normal_equation_error_array_list = []
    naive_lsqr_iternum_list = []

    lsrn_absolute_residual_error_array_list = []
    lsrn_absolute_normal_equation_error_array_list = []
    lsrn_relative_residual_error_array_list = []
    lsrn_relative_normal_equation_error_array_list = []
    lsrn_iternum_list = []

    Riley_blen_absolute_residual_error_array_list = []
    Riley_blen_absolute_normal_equation_error_array_list = []
    Riley_blen_relative_residual_error_array_list = []
    Riley_blen_relative_normal_equation_error_array_list = []
    Riley_blen_iternum_list = []
    for seednum in seednum_array:
        A, _, b = overdetermined_ls_test_matrix_generator(m=1000,
                                                          n=50,
                                                          theta=0,
                                                          seednum=seednum,
                                                          fill_diagonal_method='geometric',
                                                          condition_number=condition_number)
        # x = x.ravel()
        b = b.ravel()

        # Naive LSQR
        total1 = lsqr_copy(A, b, atol=1e-14, btol=1e-14, iter_lim=1000)

        iternum1 = total1[2]
        naive_lsqr_absolute_residual_error_array = total1[-1]
        naive_lsqr_absolute_normal_equation_error_array = total1[-2]
        naive_lsqr_relative_residual_error_array = total1[-3]
        naive_lsqr_relative_normal_equation_error_array = total1[-4]

        # Store the data into list
        naive_lsqr_absolute_residual_error_array_list.append(naive_lsqr_absolute_residual_error_array)
        naive_lsqr_absolute_normal_equation_error_array_list.append(naive_lsqr_absolute_normal_equation_error_array)
        naive_lsqr_relative_residual_error_array_list.append(naive_lsqr_relative_residual_error_array)
        naive_lsqr_relative_normal_equation_error_array_list.append(naive_lsqr_relative_normal_equation_error_array)
        naive_lsqr_iternum_list.append(iternum1)
        # LSRN
        total2 = LSRN_over_for_error_test(A, b, tol=1e-14, iter_lim=1000)

        iternum2 = total2[2]
        lsrn_absolute_residual_error_array = total2[-1]
        lsrn_absolute_normal_equation_error_array = total2[-2]
        lsrn_relative_residual_error_array = total2[-3]
        lsrn_relative_normal_equation_error_array = total2[-4]

        lsrn_absolute_residual_error_array_list.append(lsrn_absolute_residual_error_array)
        lsrn_absolute_normal_equation_error_array_list.append(lsrn_absolute_normal_equation_error_array)
        lsrn_relative_residual_error_array_list.append(lsrn_relative_residual_error_array)
        lsrn_relative_normal_equation_error_array_list.append(lsrn_relative_normal_equation_error_array)
        lsrn_iternum_list.append(iternum2)

        # Riley's Blendenpik
        multiplier = 2
        d = multiplier * A.shape[1]
        tol = 1e-14

        total3 = blendenpik_srct_scipy_lsqr_for_error_test(A, b, d, tol, 1000)

        iternum3 = total3[2]
        Riley_blen_absolute_residual_error_array = total3[-1]
        Riley_blen_absolute_normal_equation_error_array = total3[-2]
        Riley_blen_relative_residual_error_array = total3[-3]
        Riley_blen_relative_normal_equation_error_array = total3[-4]

        Riley_blen_absolute_residual_error_array_list.append(Riley_blen_absolute_residual_error_array)
        Riley_blen_absolute_normal_equation_error_array_list.append(Riley_blen_absolute_normal_equation_error_array)
        Riley_blen_relative_residual_error_array_list.append(Riley_blen_relative_residual_error_array)
        Riley_blen_relative_normal_equation_error_array_list.append(Riley_blen_relative_normal_equation_error_array)
        Riley_blen_iternum_list.append(iternum3)

    # Naive LSQR Error plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for i in np.arange(2):
        for j in np.arange(2):
            naive_lsqr_iterm_num_list = np.arange(1, naive_lsqr_iternum_list[i + j] + 1)
            naive_lsqr_relative_residual_error_array = naive_lsqr_absolute_residual_error_array_list[i + j]
            naive_lsqr_absolute_residual_error_array = naive_lsqr_absolute_residual_error_array_list[i + j]
            naive_lsqr_relative_normal_equation_error_array = naive_lsqr_relative_normal_equation_error_array_list[
                i + j]
            naive_lsqr_absolute_normal_equation_error_array = naive_lsqr_absolute_normal_equation_error_array_list[
                i + j]

            axes[i, j].set_xlabel('iteration number')
            axes[i, j].set_ylabel('error')
            axes[i, j].set_title('Error Via Iteration Naive LSQR Seed ' + str(seednum_array[i + j]))

            axes[i, j].plot(naive_lsqr_iterm_num_list, np.log10(naive_lsqr_relative_residual_error_array),
                            linestyle="-",
                            label='Relative Residual Error')
            axes[i, j].plot(naive_lsqr_iterm_num_list, np.log10(naive_lsqr_absolute_residual_error_array),
                            linestyle="--",
                            label='Absolute Residual Error')
            axes[i, j].plot(naive_lsqr_iterm_num_list, np.log10(naive_lsqr_relative_normal_equation_error_array),
                            linestyle="-.",
                            label='Relative Normal Equation Error')
            axes[i, j].plot(naive_lsqr_iterm_num_list, np.log10(naive_lsqr_absolute_normal_equation_error_array),
                            linestyle=":",
                            label='Absolute Normal Equation Error')
            axes[i, j].legend(loc='best', shadow=True)

    plt.savefig('Error/Naive LSQR/Error(Cond ' + str(condition_number) + ').png')
    plt.show()

    # LSRN Error Plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for i in np.arange(2):
        for j in np.arange(2):
            lsrn_iterm_num_list = np.arange(1, lsrn_iternum_list[i + j] + 1)
            lsrn_relative_residual_error_array = lsrn_absolute_residual_error_array_list[i + j]
            lsrn_absolute_residual_error_array = lsrn_absolute_residual_error_array_list[i + j]
            lsrn_relative_normal_equation_error_array = lsrn_relative_normal_equation_error_array_list[
                i + j]
            lsrn_absolute_normal_equation_error_array = lsrn_absolute_normal_equation_error_array_list[
                i + j]

            axes[i, j].set_xlabel('iteration number')
            axes[i, j].set_ylabel('error')
            axes[i, j].set_title('Error Via Iteration LSRN Seed ' + str(seednum_array[i + j]))

            axes[i, j].plot(lsrn_iterm_num_list, np.log10(lsrn_relative_residual_error_array),
                            linestyle="-",
                            label='Relative Residual Error')
            axes[i, j].plot(lsrn_iterm_num_list, np.log10(lsrn_absolute_residual_error_array),
                            linestyle="--",
                            label='Absolute Residual Error')
            axes[i, j].plot(lsrn_iterm_num_list, np.log10(lsrn_relative_normal_equation_error_array),
                            linestyle="-.",
                            label='Relative Normal Equation Error')
            axes[i, j].plot(lsrn_iterm_num_list, np.log10(lsrn_absolute_normal_equation_error_array),
                            linestyle=":",
                            label='Absolute Normal Equation Error')
            axes[i, j].legend(loc='best', shadow=True)

    plt.savefig('Error/LSRN/Error(Cond ' + str(condition_number) + ').png')
    plt.show()

    # Riley Blendenpik Error Plots

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for i in np.arange(2):
        for j in np.arange(2):
            Riley_blen_iterm_num_list = np.arange(1, Riley_blen_iternum_list[i + j] + 1)
            Riley_blen_relative_residual_error_array = Riley_blen_absolute_residual_error_array_list[i + j]
            Riley_blen_absolute_residual_error_array = Riley_blen_absolute_residual_error_array_list[i + j]
            Riley_blen_relative_normal_equation_error_array = Riley_blen_relative_normal_equation_error_array_list[
                i + j]
            Riley_blen_absolute_normal_equation_error_array = Riley_blen_absolute_normal_equation_error_array_list[
                i + j]

            axes[i, j].set_xlabel('iteration number')
            axes[i, j].set_ylabel('error')
            axes[i, j].set_title('Error Via Iteration Riley Blen Seed ' + str(seednum_array[i + j]))

            axes[i, j].plot(Riley_blen_iterm_num_list, np.log10(Riley_blen_relative_residual_error_array),
                            linestyle="-",
                            label='Relative Residual Error')
            axes[i, j].plot(Riley_blen_iterm_num_list, np.log10(Riley_blen_absolute_residual_error_array),
                            linestyle="--",
                            label='Absolute Residual Error')
            axes[i, j].plot(Riley_blen_iterm_num_list, np.log10(Riley_blen_relative_normal_equation_error_array),
                            linestyle="-.",
                            label='Relative Normal Equation Error')
            axes[i, j].plot(Riley_blen_iterm_num_list, np.log10(Riley_blen_absolute_normal_equation_error_array),
                            linestyle=":",
                            label='Absolute Normal Equation Error')
            axes[i, j].legend(loc='best', shadow=True)

    plt.savefig('Error/Riley Blen/Error(Cond ' + str(condition_number) + ').png')
    plt.show()
