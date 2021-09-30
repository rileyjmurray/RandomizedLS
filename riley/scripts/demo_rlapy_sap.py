import time
import rlapy as rla
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from riley.protomodules.demo_helper import make_demo_helper


def time_alg(A, b, alg, tol, seeds):
    """
    Run the randomized algorithm "alg" on the least squares problem (A, b)
    with different random seeds. Save a log for each run of the algorithm.

    Each log is a dict of the following form

        'time_sketch': float    # time to build S and compute S @ A.

        'time_factor': float    # time to factor S @ A.

        'time_presolve': float  # time to compute S @ b and initialize x.

        'time_iterate': float   # total time spent by LSQR

        'x' : ndarray           # the final solution vector

        'arnorms': ndarray

            A vector of preconditioned normal equation errors for the
            approximate solution "x". arnorms[0] is the error when x = 0.
            arnorms[i] for i >= 1 is the error at the start of iteration i
            of LSQR. Iteration 1 starts with the output of sketch-and-solve.

         'times': ndarray

            times[i] is the total time spent by the algorithm to produce
            the iterate with error equal to arnorms[i]. These times are
            produced under the model that each iteration of LSQR takes the
            same time as every other iteration.

            LSQR ran for (times.size - 1) iterations.

    """
    iter_lim = A.shape[1]
    rla_data = []
    print(seeds)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        alg(A, b, tol, iter_lim, rng, logging=True)
        rla_data.append(alg.log.copy())
    return rla_data


if __name__ == '__main__':
    #
    #   Make a parent random number generator.
    #   Generate one random seed for each run of the algorithm.
    #   (It's good for those other random seeds to be explicit.)
    #
    main_rng = np.random.default_rng(897389723094)
    num_runs = 5
    seeds = main_rng.integers(0, int(1e10), size=num_runs)

    #
    #   Construct problem data. I start by setting the dimensions of the data
    #   matrix and choosing its singular values in a deterministic way.
    #
    #   Then I call the "make_demo_helper" function to get a LSDemoHelper
    #   object. That object contains the matrix A and the vector b.
    #
    #       The make_demo_helper function samples the singular vectors of A
    #       from the Haar measure.
    #
    #       The target b is sampled from a Gaussian distribution and then
    #       modified so the angle between b and range(A) is arccos(0.95).
    #
    #   REMARK: LSDemoHelper objects can resample target vectors "b" so
    #   that you can run many experiments with the same data matrix. I
    #   don't illustrate that feature in this script.
    #
    n = 2000
    m = 100000
    kappa = 1e5  # condition number
    spec = np.linspace(kappa**0.5, kappa**-0.5, num=n)
    rng = np.random.default_rng(seeds[0])
    dh = make_demo_helper(m, n, spec, 0.95, rng)
    A, b = dh.A, dh.b

    #
    #   Build the algorithm object with appropriate parameters. I'm using an
    #   SJLT since that's what you're looking at for auto-tuning. I have to
    #   construct a function handle which produces SJLTs with specified
    #   number of nonzeros per column.
    #
    sampling_factor = 3.0
    k = 8  # number of nonzeros per column of S
    sketch_op_gen = lambda _d, _m, _rng: rla.sjlt_operator(_d, _m, _rng, k)
    sap = rla.SAP1(sketch_op_gen, sampling_factor)

    #
    #   Run repeated experiments with this algorithm configuration.
    #
    alg_logs = time_alg(A, b, sap, 1e-16, seeds)

    #
    #   Plot the normal equation error vs time for each run of the algorithm.
    #
    title = f'(m, n) = ({m}, {n}),   (d/n, k) = ({sampling_factor}, {k})'
    fig = plt.figure(dpi=300, figsize=(5, 5))
    for log in alg_logs:
        plt.semilogy(log['times'], log['arnorms'])
    plt.title(title)  # we use the same title later on
    plt.xlabel('Runtime in seconds')
    plt.ylabel('Preconditioned normal equation error')
    plt.tight_layout()
    plt.show()

    #
    #   Get a sense of how long is spent on each phase of the algorithm.
    #   I just show the average. A more detailed report might want to do
    #   box-and-whisker plot, but that would require a larger number of  runs.
    #
    big_log = {
        'time_sketch': [],
        'time_factor': [],
        'time_presolve': [],
        'time_iterate': []
    }
    for log in alg_logs:
        big_log['time_sketch'].append(log['time_sketch'])
        big_log['time_factor'].append(log['time_factor'])
        big_log['time_presolve'].append(log['time_presolve'])
        big_log['time_iterate'].append(log['time_iterate'])
    fig = plt.figure()
    phases = ['sketch', 'factor', 'presolve', 'iterate']
    times = [
        np.mean(big_log['time_sketch']),
        np.mean(big_log['time_factor']),
        np.mean(big_log['time_presolve']),
        np.mean(big_log['time_iterate'])
    ]
    plt.bar(phases, times)
    plt.title(title)
    plt.ylabel('time in seconds (averaged)')
    plt.xlabel('sketch-and-precondition algorithm phase')
    plt.show()

    run_lapack = False
    if run_lapack:
        for driver in ['gelss', 'gelsd']:  # could do gelsy
            tic = time.time()
            la.lstsq(A, b, lapack_driver=driver)
            toc = time.time()
            print('%s: %s' % (driver, str(toc - tic)))
        # run gels
        Af = np.asfortranarray(A)
        bf = np.asfortranarray(np.reshape(b, (-1, 1)))
        tic = time.time()
        la.lapack.dgels(Af, bf)
        toc = time.time()
        print('gels: %s' % str(toc - tic))
