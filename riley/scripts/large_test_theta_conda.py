from time import perf_counter

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import lsqr

from Haoyun.randomized_least_square_solver.Test.Blendenpik import blendenpik_srct
from Haoyun.randomized_least_square_solver.Test.LSRN import LSRN_over
from Haoyun.randomized_least_square_solver.Test.test_matrix_generator import overdetermined_ls_test_matrix_generator

import seaborn as sns
import matplotlib.pyplot as plt

# cond_num_list = 10 ** np.arange(6)
cond_num_list = 10 ** np.linspace(6, 16, 3)
theta_list = 2 ** (-np.linspace(53, 0, 6))

cond_len = len(cond_num_list)
theta_list_len = len(theta_list)

# Blendenpik_iternum_matrix = np.zeros((cond_len, m_div_100_len))
# Blendenpik_time_matrix = np.zeros((cond_len, m_div_100_len))
# Blendenpik_normal_equation_error_matrix = np.zeros((cond_len, m_div_100_len))

LSRN_iternum_matrix = np.zeros((cond_len, theta_list_len))
LSRN_time_matrix = np.zeros((cond_len, theta_list_len))
LSRN_normal_equation_error_matrix = np.zeros((cond_len, theta_list_len))
LSRN_relative_error_matrix = np.zeros((cond_len, theta_list_len))

Naive_LSQR_iternum_matrix = np.zeros((cond_len, theta_list_len))
Naive_LSQR_time_matrix = np.zeros((cond_len, theta_list_len))
Naive_LSQR_normal_equation_error_matrix = np.zeros((cond_len, theta_list_len))
Naive_LSQR_relative_error_matrix = np.zeros((cond_len, theta_list_len))

Riley_Blen_iternum_matrix = np.zeros((cond_len, theta_list_len))
Riley_Blen_time_matrix = np.zeros((cond_len, theta_list_len))
Riley_Blen_normal_equation_error_matrix = np.zeros((cond_len, theta_list_len))
Riley_Blen_relative_error_matrix = np.zeros((cond_len, theta_list_len))

matrix_num = 1

for cond_num_index in np.arange(cond_len):
    cond_num = cond_num_list[cond_num_index]
    for theta_index in np.arange(theta_list_len):
        theta = theta_list[theta_index]
        for matrix in np.arange(matrix_num):
            A, x, b = overdetermined_ls_test_matrix_generator(m=6000,
                                                              n=300,
                                                              theta=theta,
                                                              seednum=100 * (matrix + 1) + 10 * (
                                                                      matrix + 2) + matrix + 3,
                                                              fill_diagonal_method='arithmetic',
                                                              condition_number=cond_num)
            x = x.ravel()
            b = b.ravel()

            # # My Blendenpik
            # t0_1 = perf_counter()
            # blendenpik1 = Blendenpik(A1, b1)
            # x1_1, iternum1_1 = blendenpik1.solve()
            # t1_1 = perf_counter() - t0_1

            # t0_2 = perf_counter()
            # blendenpik2 = Blendenpik(A2, b2)
            # x1_2, iternum1_2 = blendenpik2.solve()
            # t1_2 = perf_counter() - t0_2

            # t0_3 = perf_counter()
            # blendenpik3 = Blendenpik(A3, b3)
            # x1_3, iternum1_3 = blendenpik3.solve()
            # t1_3 = perf_counter() - t0_3

            # Blendenpik_iternum_list[index] = (iternum1_1 + iternum1_2 + iternum1_3) / 3 Blendenpik_time_list[index]
            # = (t1_1 + t1_2 + t1_3) / 3 Blendenpik_normal_equation_error_list[index] = (np.linalg.norm(A1.transpose(
            # ) @ A1 @ x1_1 - A1.transpose() @ b1,ord =2) + np.linalg.norm(A2.transpose() @ A2 @ x1_2 - A2.transpose(
            # ) @ b2,ord =2) + np.linalg.norm(A3.transpose() @ A3 @ x1_3 - A3.transpose() @ b3,ord =2)) / 3

            # print("Blendenpik algorithm:")
            # print("\tNormal Equation:", np.linalg.norm(A.T @ A @ x1 - A.T @ b,ord =2))
            # print("\tResidual (L2-norm):", np.linalg.norm(A @ x1 - b,ord =2))
            # print("\tError:", np.linalg.norm(x1 - x,ord =2)/np.linalg.norm(x,ord =2))
            # print("\tComputational time (sec.):", t1)
            # print("\tThe iteration number is:", iternum1)

            # Naive LSQR
            t10 = perf_counter()
            # total2 = LSQR(A, b, tol=1e-14)
            total2 = lsqr(A, b, atol=1e-14, btol=1e-14, iter_lim=2000)
            x6 = total2[0]
            iternum2 = total2[2]
            t11 = perf_counter() - t10

            Naive_LSQR_iternum_matrix[cond_num_index, theta_index] += iternum2
            Naive_LSQR_time_matrix[cond_num_index, theta_index] += t11
            Naive_LSQR_normal_equation_error_matrix[cond_num_index, theta_index] += np.linalg.norm(
                A.T @ (A @ x6) - A.T @ b, ord=2)
            Naive_LSQR_relative_error_matrix[cond_num_index, theta_index] += \
                np.linalg.norm(x6 - x, ord=2) / np.linalg.norm(x, ord=2)

            # LSRN
            t2 = perf_counter()
            x2, iternum3 = LSRN_over(A, b, tol=1e-14, iter_lim=2000)[:2]
            t3 = perf_counter() - t2

            LSRN_iternum_matrix[cond_num_index, theta_index] += iternum3
            LSRN_time_matrix[cond_num_index, theta_index] += t3
            LSRN_normal_equation_error_matrix[cond_num_index, theta_index] += np.linalg.norm(
                A.T @ (A @ x2) - A.T @ b, ord=2)
            LSRN_relative_error_matrix[cond_num_index, theta_index] += \
                np.linalg.norm(x2 - x, ord=2) / np.linalg.norm(x, ord=2)

            # Riley's Blendenpik
            multiplier = 2
            d = multiplier * A.shape[1]
            tol = 1e-14

            t8 = perf_counter()
            x5, res, (r, e) = blendenpik_srct(A, b, d, tol, 2000)
            t9 = perf_counter() - t8

            Riley_Blen_iternum_matrix[cond_num_index, theta_index] += np.count_nonzero(res > -1)
            Riley_Blen_time_matrix[cond_num_index, theta_index] += t9
            Riley_Blen_normal_equation_error_matrix[cond_num_index, theta_index] += np.linalg.norm(
                A.T @ (A @ x5) - A.T @ b, ord=2)
            Riley_Blen_relative_error_matrix[cond_num_index, theta_index] += \
                np.linalg.norm(x5 - x, ord=2) / np.linalg.norm(x, ord=2)

        Naive_LSQR_iternum_matrix[cond_num_index, theta_index] /= matrix_num
        Naive_LSQR_time_matrix[cond_num_index, theta_index] /= matrix_num
        Naive_LSQR_normal_equation_error_matrix[cond_num_index, theta_index] /= matrix_num
        Naive_LSQR_relative_error_matrix[cond_num_index, theta_index] /= matrix_num

        LSRN_iternum_matrix[cond_num_index, theta_index] /= matrix_num
        LSRN_time_matrix[cond_num_index, theta_index] /= matrix_num
        LSRN_normal_equation_error_matrix[cond_num_index, theta_index] /= matrix_num
        LSRN_relative_error_matrix[cond_num_index, theta_index] /= matrix_num

        Riley_Blen_iternum_matrix[cond_num_index, theta_index] /= matrix_num
        Riley_Blen_time_matrix[cond_num_index, theta_index] /= matrix_num
        Riley_Blen_normal_equation_error_matrix[cond_num_index, theta_index] /= matrix_num
        Riley_Blen_relative_error_matrix[cond_num_index, theta_index] /= matrix_num

# ##################
# #### Heatmap #####
# ##################

# The heat heap of averaged iteration number of Naive LSQR of five randomized 6000 * 300 matrix.
# log10(condition number) ranges from 1 to 6 and theta ranges from pi/2 * 2^[0, -5].
sns.heatmap(Naive_LSQR_iternum_matrix, xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of iteration number of Naive LSQR of 6000*300')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Iteration Number/Naive LSQR HeatMap Iteration Number.png')
plt.show()

# The heat heap of averaged iteration number of LSRN of five randomized 6000 * 300 matrix.
# log10(condition number) ranges from 1 to 6 and theta ranges from pi/2 * 2^[0, -5].
sns.heatmap(LSRN_iternum_matrix, xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of iteration number of LSRN of 6000*300')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Iteration Number/LSRN HeatMap of Iteration Number.png')
plt.show()

# The heat heap of averaged iteration number of Riley's Blendenpik of five randomized 6000 * 300 matrix.
# log10(condition number) ranges from 1 to 6 and theta ranges from pi/2 * 2^[0, -5].
sns.heatmap(Riley_Blen_iternum_matrix, xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of iteration number of Riley Blen of 6000*300')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Iteration Number/Riley Blendenpik HeatMap of Iteration Number.png')
plt.show()

# The heat heap of averaged normal equation error of Naive LSQR of five randomized 6000 * 300 matrix.
# log10(condition number) ranges from 1 to 6 and theta ranges from pi/2 * 2^[0, -5].
sns.heatmap(np.log10(Naive_LSQR_normal_equation_error_matrix), xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of normal equation error of Naive LSQR of 6000*300')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Normal Equation Error/Naive LSQR HeatMap of Normal Equation Error.png')
plt.show()

# The heat heap of averaged normal equation error of LSRN of five randomized 6000 * 300 matrix.
# log10(condition number) ranges from 1 to 6 and theta ranges from pi/2 * 2^[0, -5].
sns.heatmap(np.log10(LSRN_normal_equation_error_matrix), xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat heap of normal equation error of LSRN of 6000*300')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Normal Equation Error/LSRN HeatMap of Normal Equation Error.png')
plt.show()

# The heat heap of averaged normal equation error of Riley's Blendenpik of five randomized 6000 * 300 matrix.
# log10(condition number) ranges from 1 to 6 and theta ranges from pi/2 * 2^[0, -5].
sns.heatmap(np.log10(Riley_Blen_normal_equation_error_matrix), xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of normal equation error of Riley Blen of 6000*300')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Normal Equation Error/Riley Blendenpik HeatMap of Normal Equation Error.png')
plt.show()

# The heat heap of averaged relative error of Naive LSQR of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(np.log10(Naive_LSQR_relative_error_matrix), xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of relative error of Naive LSQR')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Residual Error/Naive LSQR HeatMap Residual Error.png')
plt.show()

# The heat heap of averaged relative error of LSRN of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(np.log10(LSRN_relative_error_matrix), xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of relative error of LSRN')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Residual Error/LSRN HeatMap Residual Error.png')
plt.show()

# The heat heap of averaged relative error of Riley's Blendenpik of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(np.log10(Riley_Blen_relative_error_matrix), xticklabels=np.round(np.log2(theta_list), 1),
            yticklabels=np.round(np.log10(cond_num_list), 1))
plt.title('Heat map of relative error of Riley Blendenpik')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Theta/Residual Error/Riley Blen HeatMap Residual Error.png')
plt.show()

##################
#### 2D Plot #####
##################

# Plot of iteration numbers of different algorithms versus log10(condition number)

log10_cond_num_list = np.log10(cond_num_list)
# plt.plot(log10_cond_num, Blendenpik_iternum_list,'r--', label = 'My Blendenpik')
plt.plot(log10_cond_num_list, Naive_LSQR_iternum_matrix[:, -1], 'g:', label='Naive LSQR')
plt.plot(log10_cond_num_list, Riley_Blen_iternum_matrix[:, -1], 'b', label='Riley Blendenpik')
plt.plot(log10_cond_num_list, LSRN_iternum_matrix[:, -1], 'k-.', label='LSRN')
plt.xlabel('log10(condition number)')
plt.ylabel('iteration number')
plt.title('Iteration Number of 6000 * 300 Randomized Matrices(Theta=pi/2-1)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Theta/Iteration Number/Fixed Dimension.png')
plt.show()

# Plot of iteration numbers of different algorithms versus log2(theta)

log2_theta_list = np.log2(theta_list)
# plt.plot(m_div_100_list, Blendenpik_iternum_list,'r--', label = 'My Blendenpik')
plt.plot(log2_theta_list, np.log10(Naive_LSQR_iternum_matrix[-1, :]), 'g:', label='Naive LSQR')
plt.plot(log2_theta_list, np.log10(Riley_Blen_iternum_matrix[-1, :]), 'b', label='Riley Blendenpik')
plt.plot(log2_theta_list, np.log10(LSRN_iternum_matrix[-1, :]), 'k-.', label='LSRN')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log(iteration number)')
plt.title('Iteration Number of 6000 * 300 Randomized Matrices(Cond 1e5)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Theta/Iteration Number/Fixed Condition number.png')
plt.show()

# Plots of normal equation error of different algorithms versus log10(condition number)

log10_cond_num_list = np.log10(cond_num_list)
# plt.plot(log10_cond_num, np.log10(Blendenpik_normal_equation_error_list),'r--', label = 'My Blendenpik')
plt.plot(log10_cond_num_list, np.log10(Naive_LSQR_normal_equation_error_matrix[:, -1]), 'g:', label='Naive LSQR')
plt.plot(log10_cond_num_list, np.log10(Riley_Blen_normal_equation_error_matrix[:, -1]), 'b', label='Riley Blendenpik')
plt.plot(log10_cond_num_list, np.log10(LSRN_normal_equation_error_matrix[:, -1]), 'k-.', label='LSRN')
plt.xlabel('log10(condition number)')
plt.ylabel('log10(normal equation error)')
plt.title('Normal Equation Error of 6000 * 300 Randomized Matrices(Theta=pi/2-1)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Theta/Normal Equation Error/Fixed Dimension.png')
plt.show()

# Plot of normal equation error of different algorithms versus log2(theta)

log2_theta_list = np.log2(theta_list)
# plt.plot(m_div_100_list, Blendenpik_iternum_list,'r--', label = 'My Blendenpik')
plt.plot(log2_theta_list, np.log10(Naive_LSQR_normal_equation_error_matrix[-1, :]), 'g:', label='Naive LSQR')
plt.plot(log2_theta_list, np.log10(Riley_Blen_normal_equation_error_matrix[-1, :]), 'b', label='Riley Blendenpik')
plt.plot(log2_theta_list, np.log10(LSRN_normal_equation_error_matrix[-1, :]), 'k-.', label='LSRN')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(normal equation error)')
plt.title('Normal Equation Error of 6000 * 300 Randomized Matrices(Cond 1e5)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Theta/Normal Equation Error/Fixed Condition number.png')
plt.show()

# Plot of relative error of different algorithms versus log2(theta)

log2_theta_list = np.log2(theta_list)
# plt.plot(m_div_100_list, Blendenpik_iternum_list,'r--', label = 'My Blendenpik')
plt.plot(log2_theta_list, np.log10(Naive_LSQR_relative_error_matrix[-1, :]), 'g:', label='Naive LSQR')
plt.plot(log2_theta_list, np.log10(Riley_Blen_relative_error_matrix[-1, :]), 'b', label='Riley Blendenpik')
plt.plot(log2_theta_list, np.log10(LSRN_relative_error_matrix[-1, :]), 'k-.', label='LSRN')
plt.xlabel('log2(pi/2-theta)')
plt.ylabel('log10(residual error)')
plt.title('Relative Error of 6000 * 300 Randomized Matrices(Cond 1e5)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Theta/Residual Error/Fixed Condition number.png')
plt.show()

# Plots of relative error of different algorithms versus log10(condition number)

log10_cond_num_list = np.log10(cond_num_list)
# plt.plot(log10_cond_num, np.log10(Blendenpik_normal_equation_error_list),'r--', label = 'My Blendenpik')
plt.plot(log10_cond_num_list, np.log10(Naive_LSQR_relative_error_matrix[:, -1]), 'g:', label='Naive LSQR')
plt.plot(log10_cond_num_list, np.log10(Riley_Blen_relative_error_matrix[:, -1]), 'b', label='Riley Blendenpik')
plt.plot(log10_cond_num_list, np.log10(LSRN_relative_error_matrix[:, -1]), 'k-.', label='LSRN')
plt.xlabel('log10(condition number)')
plt.ylabel('log10(residual error)')
plt.title('Relative Error of 6000 * 300 Randomized Matrices(Theta=pi/2-1)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Theta/Residual Error/Fixed Dimension.png')
plt.show()
