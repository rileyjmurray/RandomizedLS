import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

condition_number_type_array = np.array(['Low', 'Medium', 'High'])
condition_number_type_array_length = 3


# # Choose some different randomized matrices with different random number seeds
seednum_array_length = 5

# Choose matrix size to be row_num * row_num/20
row_num = 5000
col_num = int(row_num/20)

# Choose the range of oversampling factor to be 1.6 to 2.5, which is specified by the LSRN paper
oversampling_factor_array = np.arange(1.5, 10.5, 0.25)

# Set the stopping criteria
stopping_criteria = "new"

for cond_number_type_index in np.arange(condition_number_type_array_length):
    # Those data in the csv files, x-axis represents different oversampling factor,
    # while y-axis represents different random number seed

    # Extract the data of LSRN

    # name of csv file
    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_total_computational_time_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_total_computational_time_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()


    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_precondition_time_Batch" + str(seednum_array_length) + '_Matrix_Size' + str(
        row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_precondition_time_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_precondition_time_without_rand_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_precondition_time_without_rand_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_iterative_solver_time_Batch" + str(seednum_array_length) + '_Matrix_Size' + str(
        row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_iterative_solver_time_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_iteration_number_Batch" + str(seednum_array_length) + '_Matrix_Size' + str(
        row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_iteration_number_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Flops Rate/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_total_computational_flops_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_total_computational_flops_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Flops Rate/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_precondition_flops_Batch" + str(seednum_array_length) + '_Matrix_Size' + str(
        row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_precondition_flops_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Flops Rate/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/LSRN/lsrn_iterative_solver_flops_Batch" + str(seednum_array_length) + '_Matrix_Size' + str(
        row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    lsrn_iterative_solver_flops_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()


    # filename = "Tuning/Oversampling Factor/Flops Rate/" + condition_number_type_array[
    #     cond_number_type_index] + "/LSRN/lsrn_iteration_number_Batch" + str(seednum_array_length) + '_Matrix_Size' + str(
    #     row_num) + '*' + str(col_num) + ".csv"
    #
    # lsrn_iteration_number_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    # Extract the data of Blendenpik

    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/Blendenpik/riley_blen_total_computational_time_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    riley_blen_total_computational_time_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/Blendenpik/riley_blen_precondition_time_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    riley_blen_precondition_time_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/Blendenpik/riley_blen_iterative_solver_time_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    riley_blen_iterative_solver_time_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Time/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/Blendenpik/riley_blen_iteration_number_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    riley_blen_iteration_number_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Flops Rate/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/Blendenpik/riley_blen_total_computational_flops_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    riley_blen_total_computational_flops_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Flops Rate/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/Blendenpik/riley_blen_precondition_flops_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    riley_blen_precondition_flops_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    filename = "Tuning/Oversampling Factor/Flops Rate/Matrix Cond/" + condition_number_type_array[
        cond_number_type_index] + "/Blendenpik/riley_blen_iterative_solver_flops_Batch" + str(
        seednum_array_length) + '_Matrix_Size' + str(row_num) + '_by_' + str(col_num) + stopping_criteria + ".csv"

    riley_blen_iterative_solver_flops_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()

    # filename = "Tuning/Oversampling Factor/Flops Rate/" + condition_number_type_array[
    #     cond_number_type_index] + "/Blendenpik/riley_blen_iteration_number_Batch" + str(
    #     seednum_array_length) + '_Matrix_Size' + str(row_num) + '*' + str(col_num) + ".csv"
    #
    # riley_blen_iteration_number_matrix = pd.read_csv(filename, sep=",", header=None).to_numpy()


    # Calculate mean, five percentile and ninety five percentile of the original data
    lsrn_total_computational_time_mean_array = np.mean(lsrn_total_computational_time_matrix, axis=1)
    lsrn_total_computational_flop_rate_mean_array = np.mean(np.divide(lsrn_total_computational_flops_matrix, lsrn_total_computational_time_matrix), axis=1)
    lsrn_precondition_time_mean_array = np.mean(lsrn_precondition_time_matrix, axis=1)
    # lsrn_precondition_time_without_rand_mean_array = np.mean(lsrn_precondition_time_without_rand_matrix, axis=1)
    lsrn_precondition_flop_rate_mean_array = np.mean(np.divide(lsrn_precondition_flops_matrix, lsrn_precondition_time_without_rand_matrix), axis=1)
    lsrn_iterative_solver_time_mean_array = np.mean(lsrn_iterative_solver_time_matrix, axis=1)
    lsrn_iterative_solver_flop_rate_mean_array = np.mean(np.divide(lsrn_iterative_solver_flops_matrix, lsrn_iterative_solver_time_matrix), axis=1)
    lsrn_iteration_number_mean_array = np.mean(lsrn_iteration_number_matrix, axis=1)

    lsrn_total_computational_time_5_percent_array = np.percentile(lsrn_total_computational_time_matrix, 5, axis=1)
    lsrn_total_computational_flop_rate_5_percent_array = np.percentile(np.divide(lsrn_total_computational_flops_matrix, lsrn_total_computational_time_matrix), 5, axis=1)
    lsrn_precondition_time_5_percent_array = np.percentile(lsrn_precondition_time_matrix, 5, axis=1)
    # lsrn_precondition_time_without_rand_5_percent_array = np.percentile(lsrn_precondition_time_without_rand_matrix, 5, axis=1)
    lsrn_precondition_flop_rate_5_percent_array = np.percentile(np.divide(lsrn_precondition_flops_matrix, lsrn_precondition_time_without_rand_matrix), 5, axis=1)
    lsrn_iterative_solver_time_5_percent_array = np.percentile(lsrn_iterative_solver_time_matrix, 5, axis=1)
    lsrn_iterative_solver_flop_rate_5_percent_array = np.percentile(np.divide(lsrn_iterative_solver_flops_matrix, lsrn_iterative_solver_time_matrix), 5, axis=1)
    lsrn_iteration_number_5_percent_array = np.percentile(lsrn_iteration_number_matrix, 5, axis=1)

    lsrn_total_computational_time_95_percent_array = np.percentile(lsrn_total_computational_time_matrix, 95, axis=1)
    lsrn_total_computational_flop_rate_95_percent_array = np.percentile(np.divide(lsrn_total_computational_flops_matrix, lsrn_total_computational_time_matrix), 95, axis=1)
    lsrn_precondition_time_95_percent_array = np.percentile(lsrn_precondition_time_matrix, 95, axis=1)
    # lsrn_precondition_time_without_rand_95_percent_array = np.percentile(lsrn_precondition_time_without_rand_matrix, 95, axis=1)
    lsrn_precondition_flop_rate_95_percent_array = np.percentile(np.divide(lsrn_precondition_flops_matrix, lsrn_precondition_time_without_rand_matrix), 95, axis=1)
    lsrn_iterative_solver_time_95_percent_array = np.percentile(lsrn_iterative_solver_time_matrix, 95, axis=1)
    lsrn_iterative_solver_flop_rate_95_percent_array = np.percentile(np.divide(lsrn_iterative_solver_flops_matrix, lsrn_iterative_solver_time_matrix), 95, axis=1)
    lsrn_iteration_number_95_percent_array = np.percentile(lsrn_iteration_number_matrix, 95, axis=1)

    riley_blen_total_computational_time_mean_array = np.mean(riley_blen_total_computational_time_matrix, axis=1)
    riley_blen_total_computational_flop_rate_mean_array = np.mean(np.divide(riley_blen_total_computational_flops_matrix, riley_blen_total_computational_time_matrix), axis=1)
    riley_blen_precondition_time_mean_array = np.mean(riley_blen_precondition_time_matrix, axis=1)
    riley_blen_precondition_flop_rate_mean_array = np.mean(np.divide(riley_blen_precondition_flops_matrix, riley_blen_precondition_time_matrix), axis=1)
    riley_blen_iterative_solver_time_mean_array = np.mean(riley_blen_iterative_solver_time_matrix, axis=1)
    riley_blen_iterative_solver_flop_rate_mean_array = np.mean(np.divide(riley_blen_iterative_solver_flops_matrix, riley_blen_iterative_solver_time_matrix), axis=1)
    riley_blen_iteration_number_mean_array = np.mean(riley_blen_iteration_number_matrix, axis=1)

    riley_blen_total_computational_time_5_percent_array = np.percentile(riley_blen_total_computational_time_matrix, 5, axis=1)
    riley_blen_total_computational_flop_rate_5_percent_array = np.percentile(np.divide(riley_blen_total_computational_flops_matrix, riley_blen_total_computational_time_matrix), 5, axis=1)
    riley_blen_precondition_time_5_percent_array = np.percentile(riley_blen_precondition_time_matrix, 5, axis=1)
    riley_blen_precondition_flop_rate_5_percent_array = np.percentile(np.divide(riley_blen_precondition_flops_matrix, riley_blen_precondition_time_matrix), 5, axis=1)
    riley_blen_iterative_solver_time_5_percent_array = np.percentile(riley_blen_iterative_solver_time_matrix, 5, axis=1)
    riley_blen_iterative_solver_flop_rate_5_percent_array = np.percentile(np.divide(riley_blen_iterative_solver_flops_matrix, riley_blen_iterative_solver_time_matrix), 5, axis=1)
    riley_blen_iteration_number_5_percent_array = np.percentile(riley_blen_iteration_number_matrix, 5, axis=1)

    riley_blen_total_computational_time_95_percent_array = np.percentile(riley_blen_total_computational_time_matrix, 95, axis=1)
    riley_blen_total_computational_flop_rate_95_percent_array = np.percentile(np.divide(riley_blen_total_computational_flops_matrix, riley_blen_total_computational_time_matrix), 95, axis=1)
    riley_blen_precondition_time_95_percent_array = np.percentile(riley_blen_precondition_time_matrix, 95, axis=1)
    riley_blen_precondition_flop_rate_95_percent_array = np.percentile(np.divide(riley_blen_precondition_flops_matrix, riley_blen_precondition_time_matrix), 95, axis=1)
    riley_blen_iterative_solver_time_95_percent_array = np.percentile(riley_blen_iterative_solver_time_matrix, 95, axis=1)
    riley_blen_iterative_solver_flop_rate_95_percent_array = np.percentile(np.divide(riley_blen_iterative_solver_flops_matrix, riley_blen_iterative_solver_time_matrix), 95, axis=1)
    riley_blen_iteration_number_95_percent_array = np.percentile(riley_blen_iteration_number_matrix, 95, axis=1)

    # Draw the 2D plots of running time of LSRN and Blendenpik
    condition_number_type = condition_number_type_array[cond_number_type_index]
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 32))
    st = fig.suptitle("New Stopping Criteria" + 'Time ' + str(row_num) + ' * ' + str(col_num) + ' Matrix ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch', fontsize=20)
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
                axes[i, j].set_ylim([0, np.max(ninety_percent_array[i, :])])
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('Iteration Number')
                axes[i, j].set_title(time_array[i] + ' of ' + algo_array[j])
                axes[i, j].legend()
            else:
                axes[i, j].plot(oversampling_factor_array, mean_array[i, j], label="mean")
                axes[i, j].plot(oversampling_factor_array, five_percent_array[i, j], label='five percent')
                axes[i, j].plot(oversampling_factor_array, ninety_percent_array[i, j], label='ninety five percent')
                axes[i, j].set_ylim([0, np.max(ninety_percent_array[i, :])])
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('time(sec.)')
                axes[i, j].set_title(time_array[i] + ' Time of ' + algo_array[j])
                axes[i, j].legend()

    plt.savefig('Tuning/Oversampling Factor/Time/Time ' + str(row_num) + '_by_' + str(col_num) + ' Matrix ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch' + stopping_criteria + '.png')
    # plt.show()

    # Draw the 2D plots of flops rate of LSRN and Blendenpik
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 32))
    st = fig.suptitle("New Stopping Criteria" + 'Flops Rate ' + str(row_num) + ' * ' + str(col_num) + ' Matrix ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch', fontsize=20)
    st.set_y(0.95)

    mean_array = np.array([[lsrn_precondition_flop_rate_mean_array, riley_blen_precondition_flop_rate_mean_array],
                           [lsrn_iterative_solver_flop_rate_mean_array, riley_blen_iterative_solver_flop_rate_mean_array],
                           [lsrn_total_computational_flop_rate_mean_array, riley_blen_total_computational_flop_rate_mean_array],
                           [lsrn_iteration_number_mean_array, riley_blen_iteration_number_mean_array]])

    five_percent_array = np.array([[lsrn_precondition_flop_rate_5_percent_array, riley_blen_precondition_flop_rate_5_percent_array],
                                   [lsrn_iterative_solver_flop_rate_5_percent_array, riley_blen_iterative_solver_flop_rate_5_percent_array,],
                                   [lsrn_total_computational_flop_rate_5_percent_array, riley_blen_total_computational_flop_rate_5_percent_array],
                                   [lsrn_iteration_number_5_percent_array, riley_blen_iteration_number_5_percent_array]])

    ninety_percent_array = np.array([[lsrn_precondition_flop_rate_95_percent_array, riley_blen_precondition_flop_rate_95_percent_array],
                                     [lsrn_iterative_solver_flop_rate_95_percent_array, riley_blen_iterative_solver_flop_rate_95_percent_array],
                                     [lsrn_total_computational_flop_rate_95_percent_array, riley_blen_total_computational_flop_rate_95_percent_array],
                                     [lsrn_iteration_number_95_percent_array, riley_blen_iteration_number_95_percent_array]])

    flops_array = ['Precondition', 'Iterative Step', 'Total', 'Iteration Number']
    algo_array = ["LSRN", "Riley's Blendenpik"]

    for i in np.arange(len(flops_array)):
        for j in np.arange(len(algo_array)):
            if i == 3:
                axes[i, j].plot(oversampling_factor_array, mean_array[i, j], label='mean')
                axes[i, j].plot(oversampling_factor_array, five_percent_array[i, j], label='five percent')
                axes[i, j].plot(oversampling_factor_array, ninety_percent_array[i, j], label='ninety five percent')
                axes[i, j].set_ylim([0, np.max(ninety_percent_array[i, :])])
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('Iteration Number')
                axes[i, j].set_title(time_array[i] + ' of ' + algo_array[j])
                axes[i, j].legend()
            else:
                axes[i, j].plot(oversampling_factor_array, mean_array[i, j], label='mean')
                axes[i, j].plot(oversampling_factor_array, five_percent_array[i, j], label='five percent')
                axes[i, j].plot(oversampling_factor_array, ninety_percent_array[i, j], label='ninety five percent')
                axes[i, j].set_ylim([0, np.max(ninety_percent_array[i, :])])
                axes[i, j].set_xlabel('oversampling factor')
                axes[i, j].set_ylabel('flops rate (flop/sec.)')
                axes[i, j].set_title(time_array[i] + ' Flops Rate of ' + algo_array[j])
                axes[i, j].legend()

    plt.savefig('Tuning/Oversampling Factor/Flops Rate/Flops Rate ' + str(row_num) + '_by_' + str(col_num) + ' Matrix ' + 'Condition Num ' + condition_number_type + ' ' + str(seednum_array_length) + ' batch' + stopping_criteria + '.png')
    # plt.show()

