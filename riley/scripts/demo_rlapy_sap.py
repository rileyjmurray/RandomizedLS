import time
import rlapy as rla
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from riley.protomodules.demo_helper import make_demo_helper


def time_alg(A, b, alg, tol, seeds):
    iter_lim = A.shape[1]
    rla_data = []
    print(seeds)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        alg(A, b, tol, iter_lim, rng, logging=True)
        rla_data.append(alg.log.copy())
    return rla_data


if __name__ == '__main__':
    main_rng = np.random.default_rng(897389723094)
    sap = rla.SAP2(rla.sjlt_operator, sampling_factor=3, smart_init=True)

    # linearly decaying spectrum
    num_runs = 5
    seeds = main_rng.integers(0, int(1e10), size=num_runs)
    dims = (100000, 1000)  # Try (100000, 2000)
    kappa = 1e5
    spec = np.linspace(kappa**0.5, kappa**-0.5, num=dims[1])

    rng = np.random.default_rng(seeds[0])
    dh = make_demo_helper(dims[0], dims[1], spec, 0.95, rng)
    A, b = dh.A, dh.b

    rlad = time_alg(A, b, sap, 1e-16, seeds)

    fig = plt.figure(dpi=300, figsize=(5, 5))
    for datum in rlad:
        plt.semilogy(datum['times'], datum['arnorms'])
        print((datum['time_sketch'], datum['time_factor'],
               datum['time_presolve'], datum['time_iterate']))
    plt.title('SJLT sketch -> SVD precond -> LSQR')
    plt.xlabel('Runtime in seconds')
    plt.ylabel('Preconditioned normal equation error')
    plt.tight_layout()
    plt.show()

    for driver in ['gelss', 'gelsd']:
        tic = time.time()
        la.lstsq(A, b, lapack_driver=driver)
        toc = time.time()
        print('%s: %s' % (driver, str(toc - tic)))
