from time import perf_counter

import numpy as np
from numpy.linalg import norm
from Haoyun.randomized_least_square_solver.LSRN.LSRN_over import LSRN_over
from Haoyun.randomized_least_square_solver.Test.test_matrix_generator import \
    overdetermined_ls_test_matrix_generator

cond_num = 1e14

A_low, _, b_low = overdetermined_ls_test_matrix_generator(m=10000,
                                                          n=500,
                                                          coherence_type="low",
                                                          added_row_count=1,
                                                          theta=0,
                                                          seednum=247,
                                                          fill_diagonal_method='geometric',
                                                          condition_number=cond_num)

# x = x.ravel()
b_low = b_low.ravel()

Q_low, _ = np.linalg.qr(A_low)
coherence_low = max(np.sum(np.multiply(Q_low, Q_low), 1))

t1_low = perf_counter()
x1_low, iternum1_low, flag1_low, _ = LSRN_over(A_low, b_low, tol=1e-14, gamma=1.05)
t2_low = perf_counter() - t1_low
r1_low = b_low - A_low @ x1_low

print("\nLSRN With Low Coherence Matrix:")
print("\tRelative Normal Equation Error:",
      norm(A_low.transpose() @ r1_low) / (norm(A_low * norm(A_low) * norm(x1_low))))
print("\tRelative Residual Error:", norm(r1_low) / norm(b_low))
print("\tComputational time (sec.):", t2_low)
print("\tThe iteration number is:", iternum1_low)
print("\tThe flag is:", flag1_low)
print("\tCoherence:", coherence_low)
print("\tCondition Number:", np.linalg.cond(A_low))

A_medium, _, b_medium = overdetermined_ls_test_matrix_generator(m=10000,
                                                                n=500,
                                                                coherence_type="medium",
                                                                added_row_count=2,
                                                                theta=0,
                                                                seednum=247,
                                                                fill_diagonal_method='geometric',
                                                                condition_number=cond_num)

# x = x.ravel()
b_medium = b_medium.ravel()

Q_medium, _ = np.linalg.qr(A_medium)
coherence_medium = max(np.sum(np.multiply(Q_medium, Q_medium), 1))

t1_medium = perf_counter()
x1_medium, iternum1_medium, flag1_medium, _ = LSRN_over(A_medium, b_medium, tol=1e-14, gamma=1.05)
t2_medium = perf_counter() - t1_medium
r1_medium = b_medium - A_medium @ x1_medium

print("\nLSRN With Medium Coherence Matrix:")
print("\tRelative Normal Equation Error:",
      norm(A_medium.transpose() @ r1_medium) / (norm(A_medium * norm(A_medium) * norm(x1_medium))))
print("\tRelative Residual Error:", norm(r1_medium) / norm(b_medium))
print("\tComputational time (sec.):", t2_medium)
print("\tThe iteration number is:", iternum1_medium)
print("\tThe flag is:", flag1_medium)
print("\tCoherence:", coherence_medium)
print("\tCondition Number:", np.linalg.cond(A_medium))

A_high, _, b_high = overdetermined_ls_test_matrix_generator(m=10000,
                                                            n=500,
                                                            coherence_type="high",
                                                            added_row_count=2,
                                                            theta=0,
                                                            seednum=247,
                                                            fill_diagonal_method='geometric',
                                                            condition_number=cond_num)

# x = x.ravel()
b_high = b_high.ravel()

Q_high, _ = np.linalg.qr(A_high)
coherence_high = max(np.sum(np.multiply(Q_high, Q_high), 1))

t1_high = perf_counter()
x1_high, iternum1_high, flag1_high, _ = LSRN_over(A_high, b_high, tol=1e-14, gamma=1.05)
t2_high = perf_counter() - t1_high
r1_high = b_high - A_high @ x1_high

print("\nLSRN With High Coherence Matrix:")
print("\tRelative Normal Equation Error:",
      norm(A_high.transpose() @ r1_high) / (norm(A_high * norm(A_high) * norm(x1_high))))
print("\tRelative Residual Error:", norm(r1_high) / norm(b_high))
print("\tComputational time (sec.):", t2_high)
print("\tThe iteration number is:", iternum1_high)
print("\tThe flag is:", flag1_high)
print("\tCoherence:", coherence_high)
print("\tCondition Number:", np.linalg.cond(A_high))