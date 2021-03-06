import numpy as np
from matplotlib import pyplot as plt

from Haoyun.randomized_least_square_solver.Blendenpik.Riley_Blen_Scipy_LSQR_for_error_test import \
    blendenpik_srct_scipy_lsqr_for_error_test
from Haoyun.randomized_least_square_solver.Blendenpik.Riley_Blendenpik_new_for_error_test import \
    blendenpik_srct_new_for_error_test
from Haoyun.randomized_least_square_solver.Iter_Solver.Scipy_LSQR import lsqr_copy
from Haoyun.randomized_least_square_solver.LSRN.LSRN_over_for_error_test import LSRN_over_for_error_test
from test_matrix_generator import overdetermined_ls_test_matrix_generator

condition_number_array = 10 ** np.array([4, 8, 12, 16])
# condition_number_array = 10 ** np.array([8])
seednum_array_length = 100
seednum_array = np.random.choice(1000, seednum_array_length, replace=False)

# Set the tolerance to be 1e-10
tol = 1e-10

for condition_number in condition_number_array:
    # naive_lsqr_absolute_residual_error_array_list = []
    naive_lsqr_absolute_normal_equation_error_array_list = []
    naive_lsqr_relative_residual_error_array_list = []
    naive_lsqr_relative_normal_equation_error_array_list = []
    naive_lsqr_iternum_list = []
    naive_lsqr_relative_error_array_list = []

    # lsrn_absolute_residual_error_array_list = []
    lsrn_absolute_normal_equation_error_array_list = []
    lsrn_relative_residual_error_array_list = []
    lsrn_relative_normal_equation_error_array_list = []
    lsrn_iternum_list = []
    lsrn_relative_error_array_list = []

    # Riley_blen_absolute_residual_error_array_list = []
    Riley_blen_absolute_normal_equation_error_array_list = []
    Riley_blen_relative_residual_error_array_list = []
    Riley_blen_relative_normal_equation_error_array_list = []
    Riley_blen_iternum_list = []
    Riley_blen_relative_error_array_list = []
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
        total1 = lsqr_copy(A, b, atol=tol, btol=tol, iter_lim=1000)

        iternum1 = total1[2]
        # naive_lsqr_absolute_residual_error_array = total1[-1]
        naive_lsqr_absolute_normal_equation_error_array = total1[-2]
        naive_lsqr_relative_residual_error_array = total1[-3]
        naive_lsqr_relative_normal_equation_error_array = total1[-4]
        naive_lsqr_relative_error_array = total1[-5]

        # Store the data into list
        # naive_lsqr_absolute_residual_error_array_list.append(naive_lsqr_absolute_residual_error_array)
        naive_lsqr_absolute_normal_equation_error_array_list.append(naive_lsqr_absolute_normal_equation_error_array)
        naive_lsqr_relative_residual_error_array_list.append(naive_lsqr_relative_residual_error_array)
        naive_lsqr_relative_normal_equation_error_array_list.append(naive_lsqr_relative_normal_equation_error_array)
        naive_lsqr_relative_error_array_list.append(naive_lsqr_relative_error_array)
        naive_lsqr_iternum_list.append(iternum1)

        # LSRN
        total2 = LSRN_over_for_error_test(A, b, tol=tol, iter_lim=1000)

        iternum2 = total2[1]
        # lsrn_absolute_residual_error_array = total2[-1]
        lsrn_absolute_normal_equation_error_array = total2[-2]
        lsrn_relative_residual_error_array = total2[-3]
        lsrn_relative_normal_equation_error_array = total2[-4]
        lsrn_relative_error_array = total2[-5]

        # lsrn_absolute_residual_error_array_list.append(lsrn_absolute_residual_error_array)
        lsrn_absolute_normal_equation_error_array_list.append(lsrn_absolute_normal_equation_error_array)
        lsrn_relative_residual_error_array_list.append(lsrn_relative_residual_error_array)
        lsrn_relative_normal_equation_error_array_list.append(lsrn_relative_normal_equation_error_array)
        lsrn_relative_error_array_list.append(lsrn_relative_error_array)
        lsrn_iternum_list.append(iternum2)

        # Riley's Blendenpik
        multiplier = 2
        d = multiplier * A.shape[1]
        tol = tol

        total3 = blendenpik_srct_new_for_error_test(A, b, d, tol, 1000)

        iternum3 = total3[2]
        # Riley_blen_absolute_residual_error_array = total3[-1]
        Riley_blen_absolute_normal_equation_error_array = total3[-2]
        Riley_blen_relative_residual_error_array = total3[-3]
        Riley_blen_relative_normal_equation_error_array = total3[-4]
        Riley_blen_relative_error_array = total3[-5]

        # Riley_blen_absolute_residual_error_array_list.append(Riley_blen_absolute_residual_error_array)
        Riley_blen_absolute_normal_equation_error_array_list.append(Riley_blen_absolute_normal_equation_error_array)
        Riley_blen_relative_residual_error_array_list.append(Riley_blen_relative_residual_error_array)
        Riley_blen_relative_normal_equation_error_array_list.append(Riley_blen_relative_normal_equation_error_array)
        Riley_blen_relative_error_array_list.append(Riley_blen_relative_error_array)
        Riley_blen_iternum_list.append(iternum3)

    # Naive LSQR Error plots
    plt.xlabel('iteration number')
    plt.ylabel('log10(error)')
    plt.title('Error Via Iteration Naive LSQR(Cond Num' + "{:.0e}".format(condition_number) + ')')
    lines = []
    for i in np.arange(seednum_array_length):
        naive_lsqr_iterm_num_x_array = np.arange(1, naive_lsqr_iternum_list[i] + 1)
        naive_lsqr_relative_residual_error_array = naive_lsqr_relative_residual_error_array_list[i]
        # naive_lsqr_absolute_residual_error_array = naive_lsqr_absolute_residual_error_array_list[i]
        naive_lsqr_relative_normal_equation_error_array = naive_lsqr_relative_normal_equation_error_array_list[i]
        naive_lsqr_absolute_normal_equation_error_array = naive_lsqr_absolute_normal_equation_error_array_list[i]
        naive_lsqr_relative_error_array = naive_lsqr_relative_error_array_list[i]

        if i == 0:
            lines += plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_relative_residual_error_array), 'b',
                              label='Relative Residual Error', linewidth=0.4)
            # lines += plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_absolute_residual_error_array), "g",
            #                   label='Absolute Residual Error', linewidth=0.2)
            lines += plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_relative_normal_equation_error_array),
                              'r',
                              label='Relative Normal Equation Error', linewidth=0.4)
            lines += plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_absolute_normal_equation_error_array),
                              'k',
                              label='Absolute Normal Equation Error', linewidth=0.4)
            lines += plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_relative_error_array), 'g',
                              label='Relative Error', linewidth=0.4)
        else:
            plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_relative_residual_error_array), '-b',
                     label='Relative Residual Error', linewidth=0.4)
            # plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_absolute_residual_error_array), "g",
            #          label='Absolute Residual Error', linewidth=0.2)
            plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_relative_normal_equation_error_array), 'r',
                     label='Relative Normal Equation Error', linewidth=0.4)
            plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_absolute_normal_equation_error_array), 'k',
                     label='Absolute Normal Equation Error', linewidth=0.4)
            plt.plot(naive_lsqr_iterm_num_x_array, np.log10(naive_lsqr_relative_error_array), 'g',
                     label='Relative Error', linewidth=0.4)

    plt.legend(lines, ['Relative Residual Error', 'Relative Normal Equation Error', 'Absolute Normal Equation Error', 'Relative Error'], loc='best', shadow=True)
    plt.savefig('Error/Naive LSQR/Error(Cond Num' + "{:.0e}".format(condition_number) + ').png')
    plt.show()

    # LSRN Error Plots
    plt.xlabel('iteration number')
    plt.ylabel('log10(error)')
    plt.title('Error Via Iteration LSRN(Cond Num' + "{:.0e}".format(condition_number) + ')')
    lines = []

    for i in np.arange(seednum_array_length):
        lsrn_iterm_num_x_array = np.arange(1, lsrn_iternum_list[i] + 1)
        lsrn_relative_residual_error_array = lsrn_relative_residual_error_array_list[i]
        # lsrn_absolute_residual_error_array = lsrn_absolute_residual_error_array_list[i]
        lsrn_relative_normal_equation_error_array = lsrn_relative_normal_equation_error_array_list[i]
        lsrn_absolute_normal_equation_error_array = lsrn_absolute_normal_equation_error_array_list[i]
        lsrn_relative_error_array = lsrn_relative_error_array_list[i]

        if i == 0:
            lines += plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_relative_residual_error_array), 'b',
                              label='Relative Residual Error', linewidth=0.4)
            # lines += plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_absolute_residual_error_array), "g",
            #                   label='Absolute Residual Error')
            lines += plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_relative_normal_equation_error_array), 'r',
                              label='Relative Normal Equation Error', linewidth=0.4)
            lines += plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_absolute_normal_equation_error_array), 'k',
                              label='Absolute Normal Equation Error', linewidth=0.4)
            lines += plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_relative_error_array), 'g', label='Relative Error',
                              linewidth=0.4)

        else:
            plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_relative_residual_error_array), 'b',
                     label='Relative Residual Error', linewidth=0.4)
            # plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_absolute_residual_error_array), "g",
            #                   label='Absolute Residual Error')
            plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_relative_normal_equation_error_array), 'r',
                     label='Relative Normal Equation Error', linewidth=0.4)
            plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_absolute_normal_equation_error_array), 'k',
                     label='Absolute Normal Equation Error', linewidth=0.4)
            plt.plot(lsrn_iterm_num_x_array, np.log10(lsrn_relative_error_array), 'g', label='Relative Error',
                     linewidth=0.4)

    plt.legend(lines, ['Relative Residual Error', 'Relative Normal Equation Error', 'Absolute Normal Equation Error',
                       'Relative Error'], loc='best', shadow=True)

    plt.savefig('Error/LSRN/Error(Cond Num' + "{:.0e}".format(condition_number) + ').png')
    plt.show()

    # Riley Blendenpik Error Plots

    plt.xlabel('iteration number')
    plt.ylabel('log10(error)')
    plt.title('Error Via Iteration Riley Blendenpik(Cond Num' + "{:.0e}".format(condition_number) + ')')
    lines = []

    for i in np.arange(seednum_array_length):
        Riley_blen_iterm_num_x_array = np.arange(1, Riley_blen_iternum_list[i] + 1)
        Riley_blen_relative_residual_error_array = Riley_blen_relative_residual_error_array_list[i]
        # Riley_blen_absolute_residual_error_array = Riley_blen_absolute_residual_error_array_list[i]
        Riley_blen_relative_normal_equation_error_array = Riley_blen_relative_normal_equation_error_array_list[i]
        Riley_blen_absolute_normal_equation_error_array = Riley_blen_absolute_normal_equation_error_array_list[i]
        Riley_blen_relative_error_array = Riley_blen_relative_error_array_list[i]

        if i == 0:
            lines += plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_relative_residual_error_array), 'b',
                              label='Relative Residual Error', linewidth=0.4)
            # lines += plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_absolute_residual_error_array), "g",
            #                   label='Absolute Residual Error')
            lines += plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_relative_normal_equation_error_array),
                              'r', label='Relative Normal Equation Error', linewidth=0.4)
            lines += plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_absolute_normal_equation_error_array),
                              'k', label='Absolute Normal Equation Error', linewidth=0.4)
            lines += plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_relative_error_array), 'g',
                              label='Relative Error', linewidth=0.4)
        else:
            plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_relative_residual_error_array), 'b',
                     label='Relative Residual Error', linewidth=0.4)
            # plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_absolute_residual_error_array), "g",
            #                   label='Absolute Residual Error')
            plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_relative_normal_equation_error_array),
                     'r', label='Relative Normal Equation Error', linewidth=0.4)
            plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_absolute_normal_equation_error_array),
                     'k', label='Absolute Normal Equation Error', linewidth=0.4)
            plt.plot(Riley_blen_iterm_num_x_array, np.log10(Riley_blen_relative_error_array), 'g',
                     label='Relative Error', linewidth=0.4)

    plt.legend(lines, ['Relative Residual Error', 'Relative Normal Equation Error', 'Absolute Normal Equation Error',
                       'Relative Error'], loc='best', shadow=True)

    plt.savefig('Error/Riley Blen/Error(Cond Num' + "{:.0e}".format(condition_number) + ').png')
    plt.show()
