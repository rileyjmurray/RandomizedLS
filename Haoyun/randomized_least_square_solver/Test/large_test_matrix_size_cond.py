from time import perf_counter
import numpy as np
from numpy.linalg import norm
from Haoyun.randomized_least_square_solver.Test.Blendenpik.Riley_Blendenpik_new import blendenpik_srct
from Haoyun.randomized_least_square_solver.Test.LSRN import LSRN_over
from Haoyun.randomized_least_square_solver.Test.test_matrix_generator import overdetermined_ls_test_matrix_generator
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Set the condition number to be ranging from 1 to 10^16.
# Set the matrix size to be ranging from 1000*50 to 10000*500

cond_num_list = 10 ** np.linspace(0, 12, 4)
# cond_num_list = 10 ** np.linspace(2, 6, 3)
m_div_1000_list = np.arange(2, 12, 2)
# m_div_1000_list = np.arange(1, 4, 1)

cond_len = len(cond_num_list)
m_div_1000_len = len(m_div_1000_list)

# Initialize the matrices to store the iteration number, time, relative normal equation error and relative
# residual error of different iterative solvers.

# Naive_LSQR_iternum_matrix = np.zeros((cond_len, m_div_1000_len))
# Naive_LSQR_time_matrix = np.zeros((cond_len, m_div_1000_len))
# Naive_LSQR_relative_normal_equation_error_matrix = np.zeros((cond_len, m_div_1000_len))
# Naive_LSQR_relative_residual_error_matrix = np.zeros((cond_len, m_div_1000_len))

LSRN_iternum_matrix = np.zeros((cond_len, m_div_1000_len))
LSRN_time_matrix = np.zeros((cond_len, m_div_1000_len))
LSRN_relative_normal_equation_error_matrix = np.zeros((cond_len, m_div_1000_len))
LSRN_relative_residual_error_matrix = np.zeros((cond_len, m_div_1000_len))

Riley_Blen_iternum_matrix = np.zeros((cond_len, m_div_1000_len))
Riley_Blen_time_matrix = np.zeros((cond_len, m_div_1000_len))
Riley_Blen_relative_normal_equation_error_matrix = np.zeros((cond_len, m_div_1000_len))
Riley_Blen_relative_residual_error_matrix = np.zeros((cond_len, m_div_1000_len))

# Choose five different randomized matrices
matrix_num = 5
# Set the matrix coherence to be low
coherence_type = "low"
# Set the tolerance to be 10^-10
tol = 1e-10

for cond_num_index in np.arange(cond_len):
    cond_num = cond_num_list[cond_num_index]
    for m_div_1000_index in np.arange(m_div_1000_len):
        m_div_1000 = m_div_1000_list[m_div_1000_index]
        for matrix in np.arange(matrix_num):
            A, _, b = overdetermined_ls_test_matrix_generator(m=m_div_1000 * 1000,
                                                              n=m_div_1000 * 50,
                                                              coherence_type=coherence_type,
                                                              added_row_count=m_div_1000 * 25,
                                                              theta=0,
                                                              seednum=100 * (matrix + 1) + 10 * (
                                                                      matrix + 2) + matrix + 3,
                                                              fill_diagonal_method='geometric',
                                                              condition_number=cond_num)
            # x = x.ravel()
            b = b.ravel()

            # ----------------------------------------------------------------------------------------------------------------------------
            # Remove Naive LSQR, which is an outlier
            # ----------------------------------------------------------------------------------------------------------------------------

            # # Naive LSQR
            # t1 = perf_counter()
            # total1 = lsqr(A, b, atol=1e-14, btol=1e-14, iter_lim=500)
            # x1 = total1[0]
            # iternum1 = total1[2]
            # t2 = perf_counter() - t1
            # r1 = b - A @ x1
            #
            # Naive_LSQR_iternum_matrix[cond_num_index, m_div_1000_index] += iternum1
            # Naive_LSQR_time_matrix[cond_num_index, m_div_1000_index] += t2
            # Naive_LSQR_relative_normal_equation_error_matrix[cond_num_index, m_div_1000_index] \
            #     += norm(A.transpose() @ r1) / (norm(A) * norm(A) * norm(x1))
            # Naive_LSQR_relative_residual_error_matrix[cond_num_index, m_div_1000_index] += norm(r1) / norm(b)

            # LSRN
            t3 = perf_counter()
            x2, iternum2 = LSRN_over(A, b, tol=tol, iter_lim=1000)[:2]
            t4 = perf_counter() - t3
            r2 = b - A @ x2

            LSRN_iternum_matrix[cond_num_index, m_div_1000_index] += iternum2
            LSRN_time_matrix[cond_num_index, m_div_1000_index] += t4
            LSRN_relative_normal_equation_error_matrix[cond_num_index, m_div_1000_index] \
                += norm(A.transpose() @ r2) / (norm(A) * norm(A) * norm(x2))
            LSRN_relative_residual_error_matrix[cond_num_index, m_div_1000_index] += norm(r2) / norm(b)

            # Riley's Blendenpik
            multiplier = 2
            d = multiplier * A.shape[1]

            t5 = perf_counter()
            x3, _, iternum3, _ = blendenpik_srct(A, b, d, tol, 1000)
            t6 = perf_counter() - t5
            r3 = b - A @ x3

            Riley_Blen_iternum_matrix[cond_num_index, m_div_1000_index] += iternum3
            Riley_Blen_time_matrix[cond_num_index, m_div_1000_index] += t6
            Riley_Blen_relative_normal_equation_error_matrix[cond_num_index, m_div_1000_index] \
                += norm(A.transpose() @ r3) / (norm(A) * norm(A) * norm(x3))
            Riley_Blen_relative_residual_error_matrix[cond_num_index, m_div_1000_index] += norm(r3) / norm(b)

        # Naive_LSQR_iternum_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        # Naive_LSQR_time_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        # Naive_LSQR_relative_normal_equation_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        # Naive_LSQR_relative_residual_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num

        LSRN_iternum_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        LSRN_time_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        LSRN_relative_normal_equation_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        LSRN_relative_residual_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num

        Riley_Blen_iternum_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        Riley_Blen_time_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        Riley_Blen_relative_normal_equation_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num
        Riley_Blen_relative_residual_error_matrix[cond_num_index, m_div_1000_index] /= matrix_num

##################
#### Heat Map ####
##################

# The heat heap of log10(averaged iteration number) of Naive LSQR, LSRN, and Riley's Blendenpik
# of five randomized m * m/20 matrix. log10(matrix condition number) ranges from 0 to 16
# with step size 4 and m/1000 ranges from 2 to 10 with step size 2.

# data_min = np.min([np.min(np.log10(Naive_LSQR_iternum_matrix)), np.log10(np.min(LSRN_iternum_matrix)),
#                    np.log10(np.min(Riley_Blen_iternum_matrix))])
# data_max = np.max([np.max(np.log10(Naive_LSQR_iternum_matrix)), np.log10(np.max(LSRN_iternum_matrix)),
#                    np.log10(np.max(Riley_Blen_iternum_matrix))])
data_min = np.min([np.log10(np.min(LSRN_iternum_matrix)), np.log10(np.min(Riley_Blen_iternum_matrix))])
data_max = np.max([np.log10(np.max(LSRN_iternum_matrix)), np.log10(np.max(Riley_Blen_iternum_matrix))])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
cmap = cm.get_cmap('viridis')
normalizer = Normalize(data_min, data_max)
im = cm.ScalarMappable(norm=normalizer)

# iter_solver_array = ["Naive LSQR", "LSRN", "Riley's Blendenpik"]
iter_solver_array = ["LSRN", "Riley's Blendenpik"]
# data_array = [np.log10(Naive_LSQR_iternum_matrix), np.log10(LSRN_iternum_matrix), np.log10(Riley_Blen_iternum_matrix)]
data_array = [np.log10(LSRN_iternum_matrix), np.log10(Riley_Blen_iternum_matrix)]
for i in np.arange(len(iter_solver_array)):
    axes[i].set_xlabel('m/1000')
    axes[i].set_ylabel('log10(condition number)')
    axes[i].set_xticks(np.arange(m_div_1000_len))
    axes[i].set_yticks(np.arange(cond_len))
    axes[i].set_xticklabels(m_div_1000_list)
    axes[i].set_yticklabels(np.log10(cond_num_list).astype(int))
    axes[i].set_title('log10(Iteration number) of ' + iter_solver_array[i])
    axes[i].imshow(data_array[i], cmap=cmap, norm=normalizer)


# cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
# fig.colorbar(im, cax=cbar_ax)
fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig('Matrix Size & Cond/Heat Map/log10(Iteration Number) ' + coherence_type + ' Coherence.png')
plt.show()

# The heat heap of averaged iteration number of Naive LSQR, LSRN, and Riley's Blendenpik
# of five randomized m * m/20 matrix. log10(matrix condition number) ranges from 0 to 16
# with step size 4 and m/1000 ranges from 2 to 10 with step size 2.

# data_min = np.min([np.min(np.log10(Naive_LSQR_iternum_matrix)), np.log10(np.min(LSRN_iternum_matrix)),
#                    np.log10(np.min(Riley_Blen_iternum_matrix))])
# data_max = np.max([np.max(np.log10(Naive_LSQR_iternum_matrix)), np.log10(np.max(LSRN_iternum_matrix)),
#                    np.log10(np.max(Riley_Blen_iternum_matrix))])
data_min = np.min([np.min(LSRN_iternum_matrix), np.min(Riley_Blen_iternum_matrix)])
data_max = np.max([np.max(LSRN_iternum_matrix), np.max(Riley_Blen_iternum_matrix)])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
cmap = cm.get_cmap('viridis')
normalizer = Normalize(data_min, data_max)
im = cm.ScalarMappable(norm=normalizer)

# iter_solver_array = ["Naive LSQR", "LSRN", "Riley's Blendenpik"]
iter_solver_array = ["LSRN", "Riley's Blendenpik"]
# data_array = [np.log10(Naive_LSQR_iternum_matrix), np.log10(LSRN_iternum_matrix), np.log10(Riley_Blen_iternum_matrix)]
data_array = [LSRN_iternum_matrix, Riley_Blen_iternum_matrix]
for i in np.arange(len(iter_solver_array)):
    axes[i].set_xlabel('m/1000')
    axes[i].set_ylabel('condition number')
    axes[i].set_xticks(np.arange(m_div_1000_len))
    axes[i].set_yticks(np.arange(cond_len))
    axes[i].set_xticklabels(m_div_1000_list)
    axes[i].set_yticklabels(np.log10(cond_num_list).astype(int))
    axes[i].set_title('Iteration number of ' + iter_solver_array[i])
    axes[i].imshow(data_array[i], cmap=cmap, norm=normalizer)


# cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
# fig.colorbar(im, cax=cbar_ax)
fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig('Matrix Size & Cond/Heat Map/Iteration Number ' + coherence_type + ' Coherence.png')
plt.show()

# # The heat heap of averaged running time of Naive LSQR, LSRN, and Riley's Blendenpik
# of five randomized m * m/20 matrix.log10(matrix condition number) ranges from 0 to 16
# with step size 4 and m/1000 ranges from 2 to 10 with step size 2.

# data_min = np.min([np.min(Naive_LSQR_time_matrix), np.min(LSRN_time_matrix), np.min(Riley_Blen_time_matrix)])
# data_max = np.max([np.max(Naive_LSQR_time_matrix), np.max(LSRN_time_matrix), np.max(Riley_Blen_time_matrix)])
data_min = np.min([np.min(LSRN_time_matrix), np.min(Riley_Blen_time_matrix)])
data_max = np.max([np.max(LSRN_time_matrix), np.max(Riley_Blen_time_matrix)])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
cmap = cm.get_cmap('viridis')
normalizer = Normalize(data_min, data_max)
im = cm.ScalarMappable(norm=normalizer)

iter_solver_array = ["LSRN", "Riley's Blendenpik"]
# data_array = [Naive_LSQR_time_matrix, LSRN_time_matrix, Riley_Blen_time_matrix]
data_array = [LSRN_time_matrix, Riley_Blen_time_matrix]
for i in np.arange(len(iter_solver_array)):
    axes[i].set_xlabel('m/1000')
    axes[i].set_ylabel('log10(condition number)')
    axes[i].set_xticks(np.arange(m_div_1000_len))
    axes[i].set_yticks(np.arange(cond_len))
    axes[i].set_xticklabels(m_div_1000_list)
    axes[i].set_yticklabels(np.log10(cond_num_list).astype(int))
    axes[i].set_title('Running time of ' + iter_solver_array[i])
    axes[i].imshow(data_array[i], cmap=cmap, norm=normalizer)

fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig('Matrix Size & Cond/Heat Map/Running Time ' + coherence_type + ' Coherence.png')
plt.show()

# The heat heap of log10(averaged relative normal equation error) of Naive LSQR, LSRN, and Riley's Blendenpik
# of five randomized m * m/20 matrix. log10(matrix condition number) ranges from 0 to 16 with step size 4
# and m/1000 ranges from 2 to 10 with step size 2.
# data_min = np.min([np.min(np.log10(Naive_LSQR_relative_normal_equation_error_matrix)),
#                    np.min(np.log10(LSRN_relative_normal_equation_error_matrix)),
#                    np.min(np.log10(Riley_Blen_relative_normal_equation_error_matrix))])
# data_max = np.max([np.max(np.log10(Naive_LSQR_relative_normal_equation_error_matrix)),
#                    np.max(np.log10(LSRN_relative_normal_equation_error_matrix)),
#                    np.max(np.log10(Riley_Blen_relative_normal_equation_error_matrix))])
data_min = np.min([np.min(np.log10(LSRN_relative_normal_equation_error_matrix)),
                   np.min(np.log10(Riley_Blen_relative_normal_equation_error_matrix))])
data_max = np.max([np.max(np.log10(LSRN_relative_normal_equation_error_matrix)),
                   np.max(np.log10(Riley_Blen_relative_normal_equation_error_matrix))])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
cmap = cm.get_cmap('viridis')
normalizer = Normalize(data_min, data_max)
im = cm.ScalarMappable(norm=normalizer)

# iter_solver_array = ["Naive LSQR", "LSRN", "Riley's Blen"]
iter_solver_array = ["LSRN", "Riley's Blen"]
# data_array = [np.log10(Naive_LSQR_relative_normal_equation_error_matrix),
#               np.log10(LSRN_relative_normal_equation_error_matrix),
#               np.log10(Riley_Blen_relative_normal_equation_error_matrix)]
data_array = [np.log10(LSRN_relative_normal_equation_error_matrix),
              np.log10(Riley_Blen_relative_normal_equation_error_matrix)]
for i in np.arange(len(iter_solver_array)):
    axes[i].set_xlabel('m/1000')
    axes[i].set_ylabel('log10(condition number)')
    axes[i].set_xticks(np.arange(m_div_1000_len))
    axes[i].set_yticks(np.arange(cond_len))
    axes[i].set_xticklabels(m_div_1000_list)
    axes[i].set_yticklabels(np.log10(cond_num_list).astype(int))
    axes[i].set_title('log10(rela normal equa error) ' + iter_solver_array[i])
    axes[i].imshow(data_array[i], cmap=cmap, norm=normalizer)

fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig('Matrix Size & Cond/Heat Map/log10(Relative Normal Equation Error) ' + coherence_type + ' Coherence.png')
plt.show()

# The heat heap of log10(averaged relative residual error) of Naive LSQR, LSRN, and Riley's Blendenpik
# of five randomized m * m/20 matrix.log10(matrix condition number) ranges from 0 to 16 with step size 4
# and m/1000 ranges from 2 to 10 with step size 2.
# data_min = np.min([np.min(np.log10(Naive_LSQR_relative_residual_error_matrix)),
#                    np.min(np.log10(LSRN_relative_residual_error_matrix)),
#                    np.min(np.log10(Riley_Blen_relative_residual_error_matrix))])
# data_max = np.max([np.max(np.log10(Naive_LSQR_relative_residual_error_matrix)),
#                    np.max(np.log10(LSRN_relative_residual_error_matrix)),
#                    np.max(np.log10(Riley_Blen_relative_residual_error_matrix))])
data_min = np.min([np.min(np.log10(LSRN_relative_residual_error_matrix)),
                   np.min(np.log10(Riley_Blen_relative_residual_error_matrix))])
data_max = np.max([np.max(np.log10(LSRN_relative_residual_error_matrix)),
                   np.max(np.log10(Riley_Blen_relative_residual_error_matrix))])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
cmap = cm.get_cmap('viridis')
normalizer = Normalize(data_min, data_max)
im = cm.ScalarMappable(norm=normalizer)

iter_solver_array = ["LSRN", "Riley's Blen"]
# data_array = [np.log10(Naive_LSQR_relative_residual_error_matrix),
#               np.log10(LSRN_relative_residual_error_matrix),
#               np.log10(Riley_Blen_relative_residual_error_matrix)]
data_array = [np.log10(LSRN_relative_residual_error_matrix),
              np.log10(Riley_Blen_relative_residual_error_matrix)]
for i in np.arange(len(iter_solver_array)):
    axes[i].set_xlabel('m/1000')
    axes[i].set_ylabel('log10(condition number)')
    axes[i].set_xticks(np.arange(m_div_1000_len))
    axes[i].set_yticks(np.arange(cond_len))
    axes[i].set_xticklabels(m_div_1000_list)
    axes[i].set_yticklabels(np.log10(cond_num_list).astype(int))
    axes[i].set_title('log10(relative residual error) ' + iter_solver_array[i])
    axes[i].imshow(data_array[i], cmap=cmap, norm=normalizer)

fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig('Matrix Size & Cond/Heat Map/log10(Relative Residual Error) ' + coherence_type + ' Coherence.png')
plt.show()


# ##################
# #### 2D Plot #####
# ##################
#
# # Plot of iteration numbers of different algorithms versus log10(condition number).
# # log10(matrix condition number) ranges from 0 to 16 with step size 4.
#
# log10_cond_num_list = np.log10(cond_num_list)
# # plt.plot(log10_cond_num_list, np.log10(Naive_LSQR_iternum_matrix[:, -1]), 'g:', label='Naive LSQR')
# plt.plot(log10_cond_num_list, np.log10(Riley_Blen_iternum_matrix[:, -1]), 'b', label='Riley Blendenpik')
# plt.plot(log10_cond_num_list, np.log10(LSRN_iternum_matrix[:, -1]), 'k-.', label='LSRN')
# plt.title('log10(Iteration number) of 10000 * 500 randomized matrices')
# plt.xlabel('log10(Condition Number)')
# plt.ylabel('log10(Iteration Number)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Iteration Number/Fixed Dimension ' + coherence_type + ' Coherence.png')
# plt.show()
#
# # Plot of iteration numbers of different algorithms versus m/1000.
# # m/1000 ranges from 2 to 10 with step size 2.
#
# # plt.plot(m_div_1000_list, np.log10(Naive_LSQR_iternum_matrix[2, :]), 'g:', label='Naive LSQR')
# plt.plot(m_div_1000_list, np.log10(Riley_Blen_iternum_matrix[2, :]), 'b', label='Riley Blendenpik')
# plt.plot(m_div_1000_list, np.log10(LSRN_iternum_matrix[2, :]), 'k-.', label='LSRN')
# plt.title('log10(Iteration number) of m * m/20 randomized matrices(cond 1e8)')
# plt.xlabel('m/1000')
# plt.ylabel('log10(Iteration Number)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Iteration Number/Fixed Condition Number ' + coherence_type + ' Coherence.png')
# plt.show()
#
# # Plot of running time of different algorithms versus log10(condition number).
# # log10(matrix condition number) ranges from 0 to 16 with step size 4.
#
# log10_cond_num_list = np.log10(cond_num_list)
# # plt.plot(log10_cond_num_list, Naive_LSQR_time_matrix[:, -1], 'g:', label='Naive LSQR')
# plt.plot(log10_cond_num_list, Riley_Blen_time_matrix[:, -1], 'b', label='Riley Blendenpik')
# plt.plot(log10_cond_num_list, LSRN_time_matrix[:, -1], 'k-.', label='LSRN')
# plt.title('Running time of 10000 * 500 randomized matrices')
# plt.xlabel('log10(Condition Number)')
# plt.ylabel('Running Time(Sec)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Time/Fixed Dimension ' + coherence_type + ' Coherence.png')
# plt.show()
#
# # Plot of running time of different algorithms versus m/1000.
# # m/1000 ranges from 2 to 10 with step size 2.
#
# # plt.plot(m_div_1000_list, np.log10(Naive_LSQR_time_matrix[2, :]), 'g:', label='Naive LSQR')
# plt.plot(m_div_1000_list, np.log10(Riley_Blen_time_matrix[2, :]), 'b', label='Riley Blendenpik')
# plt.plot(m_div_1000_list, np.log10(LSRN_time_matrix[2, :]), 'k-.', label='LSRN')
# plt.title('Running Time of m * m/20 randomized matrices(cond 1e8)')
# plt.xlabel('m/1000')
# plt.ylabel('Running Time(Sec.)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Time/Fixed Condition Number ' + coherence_type + ' Coherence.png')
# plt.show()
#
# # Plots of relative normal equation error of different algorithms versus log10(condition number).
# # log10(matrix condition number) ranges from 0 to 16 with step size 4.
#
# log10_cond_num_list = np.log10(cond_num_list)
# # plt.plot(log10_cond_num_list, np.log10(Naive_LSQR_relative_normal_equation_error_matrix[:, -1]), 'g:',
# #          label='Naive LSQR')
# plt.plot(log10_cond_num_list, np.log10(Riley_Blen_relative_normal_equation_error_matrix[:, -1]), 'b',
#          label='Riley Blendenpik')
# plt.plot(log10_cond_num_list, np.log10(LSRN_relative_normal_equation_error_matrix[:, -1]), 'k-.',
#          label='LSRN')
# plt.title('Relative normal equation error of 10000 * 500 randomized matrices')
# plt.xlabel('log10(Condition Number)')
# plt.ylabel('log10(Relative Normal Equation Error)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Relative Normal Equation Error/Fixed Dimension ' + coherence_type
#             + ' Coherence.png')
# plt.show()
#
# # Plot of relative normal equation error of different algorithms versus m/1000.
# # m/1000 ranges from 2 to 10 with step size 2.
#
# # plt.plot(m_div_1000_list, np.log10(Naive_LSQR_relative_normal_equation_error_matrix[2, :]), 'g:',
# #          label='Naive LSQR')
# plt.plot(m_div_1000_list, np.log10(Riley_Blen_relative_normal_equation_error_matrix[2, :]), 'b',
#          label='Riley Blendenpik')
# plt.plot(m_div_1000_list, np.log10(LSRN_relative_normal_equation_error_matrix[2, :]), 'k-.',
#          label='LSRN')
# plt.title('Relative normal equation error m * m/20 randomized matrices(cond 1e8)')
# plt.xlabel('m/1000')
# plt.ylabel('log10(Relative Normal Equation Error)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Relative Normal Equation Error/Fixed Condition Number ' + coherence_type
#             + ' Coherence.png')
# plt.show()
#
# # Plots of relative error of different algorithms versus log10(condition number).
# # log10(matrix condition number) ranges from 0 to 16 with step size 4.
#
# log10_cond_num_list = np.log10(cond_num_list)
# # plt.plot(log10_cond_num_list, np.log10(Naive_LSQR_relative_residual_error_matrix[:, -1]), 'g:', label='Naive LSQR')
# plt.plot(log10_cond_num_list, np.log10(Riley_Blen_relative_residual_error_matrix[:, -1]), 'b', label='Riley Blendenpik')
# plt.plot(log10_cond_num_list, np.log10(LSRN_relative_residual_error_matrix[:, -1]), 'k-.', label='LSRN')
# plt.title('Relative residual error of 10000 * 500 randomized matrices')
# plt.xlabel('log10(Condition Number)')
# plt.ylabel('log10(Relative Residual Error)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Relative Residual Error/Fixed Dimension ' + coherence_type
#             + ' Coherence.png')
# plt.show()
#
# # Plot of relative residual error of different algorithms versus m/1000.
# # m/1000 ranges from 2 to 10 with step size 2.
#
# # plt.plot(m_div_1000_list, np.log10(Naive_LSQR_relative_residual_error_matrix[2, :]), 'g:', label='Naive LSQR')
# plt.plot(m_div_1000_list, np.log10(Riley_Blen_relative_residual_error_matrix[2, :]), 'b', label='Riley Blendenpik')
# plt.plot(m_div_1000_list, np.log10(LSRN_relative_residual_error_matrix[2, :]), 'k-.', label='LSRN')
# plt.title('Relative residual error of m * m/20 randomized matrices(cond 1e8)')
# plt.xlabel('m/1000')
# plt.ylabel('log10(Relative Residual Rrror)')
# plt.legend(loc='best', shadow=True)
# plt.savefig('Matrix Size & Cond/2D Plot/Relative Residual Error/Fixed Condition Number ' + coherence_type
#             + ' Coherence.png')
# plt.show()
