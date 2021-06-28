import numpy as np
from numpy.linalg import norm

from Haoyun.randomized_least_square_solver.Test.test_matrix_generator import overdetermined_ls_test_matrix_generator

seednum_array_length = 50
seednum_array = np.random.choice(1000, seednum_array_length, replace=False)
for seednum in seednum_array:
    A, _, b = overdetermined_ls_test_matrix_generator(m=1000,
                                                      n=50,
                                                      theta=0,
                                                      seednum=seednum,
                                                      fill_diagonal_method='geometric',
                                                      condition_number=1e8)
    # x = x.ravel()
    b = b.ravel()

    print("\nNorm(b) is: ", norm(b, 2))