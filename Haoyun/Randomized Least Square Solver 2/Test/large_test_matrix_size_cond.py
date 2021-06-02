import math
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import lsqr

from Blendenpik.Riley_Blendenpik import blendenpik_srct
from LSRN.LSRN_over import LSRN_over
from Test.test_matrix_generator import overdetermined_ls_test_matrix_generator

import seaborn as sns
import matplotlib.pyplot as plt

cond_num_list = [1, 10, 100, 1000, 1e4, 1e5]

m_div_1000_list = np.arange(6) + 1

cond_len = len(cond_num_list)
m_div_1000_len = len(m_div_1000_list)

# Blendenpik_iternum_matrix = np.zeros((cond_len, m_div_100_len))
# Blendenpik_time_matrix = np.zeros((cond_len, m_div_100_len))
# Blendenpik_normal_equation_error_matrix = np.zeros((cond_len, m_div_100_len))

LSRN_iternum_matrix = np.zeros((cond_len, m_div_1000_len))
LSRN_time_matrix = np.zeros((cond_len, m_div_1000_len))
LSRN_normal_equation_error_matrix = np.zeros((cond_len, m_div_1000_len))

Naive_LSQR_iternum_matrix = np.zeros((cond_len, m_div_1000_len))
Naive_LSQR_time_matrix = np.zeros((cond_len, m_div_1000_len))
Naive_LSQR_normal_equation_error_matrix = np.zeros((cond_len, m_div_1000_len))

Riley_Blen_iternum_matrix = np.zeros((cond_len, m_div_1000_len))
Riley_Blen_time_matrix = np.zeros((cond_len, m_div_1000_len))
Riley_Blen_normal_equation_error_matrix = np.zeros((cond_len, m_div_1000_len))

matrix_num = 5

for cond_num_index in np.arange(cond_len):
    cond_num = cond_num_list[cond_num_index]
    for m_div_1000_index in np.arange(m_div_1000_len):
        m_div_1000 = m_div_1000_list[m_div_1000_index]
        for matrix in np.arange(matrix_num):
            A, x, b = overdetermined_ls_test_matrix_generator(m=m_div_1000 * 1000,
                                                              n=m_div_1000 * 50,
                                                              theta=math.pi / 1024,
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

            # Blendenpik_iternum_list[index] = (iternum1_1 + iternum1_2 + iternum1_3) / 3
            # Blendenpik_time_list[index] = (t1_1 + t1_2 + t1_3) / 3
            # Blendenpik_normal_equation_error_list[index] = (np.linalg.norm(A1.transpose() @ A1 @ x1_1 - A1.transpose() @ b1,ord =2) + np.linalg.norm(A2.transpose() @ A2 @ x1_2 - A2.transpose() @ b2,ord =2) + np.linalg.norm(A3.transpose() @ A3 @ x1_3 - A3.transpose() @ b3,ord =2)) / 3

            # print("Blendenpik algorithm:")
            # print("\tNormal Equation:", np.linalg.norm(A.transpose() @ A @ x1 - A.transpose() @ b,ord =2))
            # print("\tResidual (L2-norm):", np.linalg.norm(A @ x1 - b,ord =2))
            # print("\tError:", np.linalg.norm(x1 - x,ord =2)/np.linalg.norm(x,ord =2))
            # print("\tComputational time (sec.):", t1)
            # print("\tThe iteration number is:", iternum1)

            # Naive LSQR
            t10 = perf_counter()
            # total2 = LSQR(A, b, tol=1e-14)
            total2 = lsqr(A, b, atol=1e-14, btol=1e-14)
            x6 = total2[0]
            iternum2 = total2[2]
            t11 = perf_counter() - t10

            Naive_LSQR_iternum_matrix[cond_num_index, m_div_1000_index] += iternum2
            Naive_LSQR_time_matrix[cond_num_index, m_div_1000_index] += t11
            Naive_LSQR_normal_equation_error_matrix[cond_num_index, m_div_1000_index] += np.linalg.norm(
                A.transpose() @ A @ x6 - A.transpose() @ b, ord=2)

            # LSRN
            t2 = perf_counter()
            x2, iternum3 = LSRN_over(A, b, tol=1e-14)[:2]
            t3 = perf_counter() - t2

            LSRN_iternum_matrix[cond_num_index, m_div_1000_index] += iternum3
            LSRN_time_matrix[cond_num_index, m_div_1000_index] += t3
            LSRN_normal_equation_error_matrix[cond_num_index, m_div_1000_index] += np.linalg.norm(
                A.transpose() @ A @ x2 - A.transpose() @ b, ord=2)

            # Riley's Blendenpik
            multiplier = 2
            d = multiplier * A.shape[1]
            tol = 1e-14

            t8 = perf_counter()
            x5, res, (r, e) = blendenpik_srct(A, b, d, tol, 1000)
            t9 = perf_counter() - t8

            Riley_Blen_iternum_matrix[cond_num_index, m_div_1000_index] += np.count_nonzero(res > -1)
            Riley_Blen_time_matrix[cond_num_index, m_div_1000_index] += t9
            Riley_Blen_normal_equation_error_matrix[cond_num_index, m_div_1000_index] += np.linalg.norm(
                A.transpose() @ A @ x5 - A.transpose() @ b, ord=2)

        Naive_LSQR_iternum_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        Naive_LSQR_time_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        Naive_LSQR_normal_equation_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num

        LSRN_iternum_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        LSRN_time_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        LSRN_normal_equation_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num

        Riley_Blen_iternum_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        Riley_Blen_time_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        Riley_Blen_normal_equation_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num

##################
#### Heatmap #####
##################

# The heat heap of averaged iteration number of Naive LSQR of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(Naive_LSQR_iternum_matrix, xticklabels=m_div_1000_list, yticklabels=np.log10(cond_num_list))
plt.title('Heat map of iteration number of Naive LSQR')
plt.xlabel('m/1000')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Matrix Size/Iteration Number/Naive LSQR HeatMap Iteration Number.png')
plt.show()

# The heat heap of averaged iteration number of LSRN of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(LSRN_iternum_matrix, xticklabels=m_div_1000_list, yticklabels=np.log10(cond_num_list))
plt.title('Heat map of iteration number of LSRN')
plt.xlabel('m/1000')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Matrix Size/Iteration Number/LSRN HeatMap Iteration Number.png')
plt.show()

# The heat heap of averaged iteration number of Riley's Blendenpik of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(Riley_Blen_iternum_matrix, xticklabels=m_div_1000_list, yticklabels=np.log10(cond_num_list))
plt.title('Heat map of iteration number of Riley Blendenpik')
plt.xlabel('m/1000')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Matrix Size/Iteration Number/Riley Blen HeatMap Iteration Number.png')
plt.show()

# The heat heap of averaged normal equation error of Naive LSQR of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(np.log10(Naive_LSQR_normal_equation_error_matrix), xticklabels=m_div_1000_list,
            yticklabels=np.log10(cond_num_list))
plt.title('Heat map of normal equation error of Naive LSQR')
plt.xlabel('m/1000')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Matrix Size/Normal Equation Error/Naive LSQR HeatMap Normal Equation Error.png')
plt.show()

# The heat heap of averaged normal equation error of LSRN of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(np.log10(LSRN_normal_equation_error_matrix), xticklabels=m_div_1000_list,
            yticklabels=np.log10(cond_num_list))
plt.title('Heat map of normal equation error of LSRN')
plt.xlabel('m/1000')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Matrix Size/Normal Equation Error/LSRN HeatMap Normal Equation Error.png')
plt.show()

# The heat heap of averaged normal equation error of Riley's Blendenpik of five randomized m * m/20 matrix.
# log10(condition number) ranges from 1 to 6 and m/1000 ranges from 1 to 6.

sns.heatmap(np.log10(Riley_Blen_normal_equation_error_matrix), xticklabels=m_div_1000_list,
            yticklabels=np.log10(cond_num_list))
plt.title('Heat map of normal equation error of Riley Blendenpik')
plt.xlabel('m/1000')
plt.ylabel('log10(condition number)')
plt.savefig('HeatMap/Matrix Size/Normal Equation Error/Riley Blen HeatMap Normal Equation Error.png')
plt.show()

##################
#### 2D Plot #####
##################

# Plot of iternation numbers of different algorithms versus log10(condition number)

log10_cond_num_list = np.log10(cond_num_list)
# plt.plot(log10_cond_num, Blendenpik_iternum_list,'r--', label = 'My Blendenpik')
plt.plot(log10_cond_num_list, Naive_LSQR_iternum_matrix[:, -1], 'g:', label='Naive LSQR')
plt.plot(log10_cond_num_list, Riley_Blen_iternum_matrix[:, -1], 'b', label='Riley Blendenpik')
plt.plot(log10_cond_num_list, LSRN_iternum_matrix[:, -1], 'k-.', label='LSRN')
plt.xlabel('log10(condition number)')
plt.ylabel('Iteration Number')
plt.title('Iteration Number of 6000 * 300 Randomized Matrices')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Matrix Size/Iteration Number/Fixed Dimension.png')
plt.show()

# Plot of iternation numbers of different algorithms versus m/1000

# plt.plot(m_div_100_list, Blendenpik_iternum_list,'r--', label = 'My Blendenpik')
plt.plot(m_div_1000_list, Naive_LSQR_iternum_matrix[-1, :], 'g:', label='Naive LSQR')
plt.plot(m_div_1000_list, Riley_Blen_iternum_matrix[-1, :], 'b', label='Riley Blendenpik')
plt.plot(m_div_1000_list, LSRN_iternum_matrix[-1, :], 'k-.', label='LSRN')
plt.xlabel('m/1000')
plt.ylabel('Iteration Number')
plt.title('Iteration Number of m * m/20 Randomized Matrices(Cond 1e5)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Matrix Size/Iteration Number/Fixed Condition number.png')
plt.show()

# Plots of normal equation error of different algorithms versus log10(condition number)

log10_cond_num_list = np.log10(cond_num_list)
# plt.plot(log10_cond_num, np.log10(Blendenpik_normal_equation_error_list),'r--', label = 'My Blendenpik')
plt.plot(log10_cond_num_list, np.log10(Naive_LSQR_normal_equation_error_matrix[:, -1]), 'g:', label='Naive LSQR')
plt.plot(log10_cond_num_list, np.log10(Riley_Blen_normal_equation_error_matrix[:, -1]), 'b', label='Riley Blendenpik')
plt.plot(log10_cond_num_list, np.log10(LSRN_normal_equation_error_matrix[:, -1]), 'k-.', label='LSRN')
plt.xlabel('log10(condition number)')
plt.ylabel('log10(normal equation error)')
plt.title('Normal Equation Error of 6000 * 300 Randomized Matrices')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Matrix Size/Normal Equation Error/Fixed Dimension.png')
plt.show()

# Plot of normal equation error of different algorithms versus m/1000

# plt.plot(m_div_100_list, Blendenpik_iternum_list,'r--', label = 'My Blendenpik')
plt.plot(m_div_1000_list, np.log10(Naive_LSQR_normal_equation_error_matrix[-1, :]), 'g:', label='Naive LSQR')
plt.plot(m_div_1000_list, np.log10(Riley_Blen_normal_equation_error_matrix[-1, :]), 'b', label='Riley Blendenpik')
plt.plot(m_div_1000_list, np.log10(LSRN_normal_equation_error_matrix[-1, :]), 'k-.', label='LSRN')
plt.xlabel('m/1000')
plt.ylabel('log10(normal equation error)')
plt.title('Normal Equation Error of m * m/20 Randomized Matrices(Cond 1e5)')
plt.legend(loc='best', shadow=True)
plt.savefig('2D Plot/Matrix Size/Normal Equation Error/Fixed Condition number.png')
plt.show()
