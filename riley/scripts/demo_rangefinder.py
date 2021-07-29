from riley.protomodules.ralas import oblique_rangefinder, rangefinder
import numpy as np
import scipy.linalg as la


def residual(Q, A):
    orthog = A - Q @ (Q.T @ A)
    return np.linalg.norm(orthog, ord=2)


def run_rangefinders(max_pass, min_pass, max_k, min_k, A, aps=1):
    s = la.svdvals(A)
    results = np.zeros(shape=(max_k - min_k, max_pass - min_pass))
    for inp, num_pass in enumerate(range(min_pass, max_pass)):
        print('\nUsing %s passes' % str(num_pass))
        for ik, currk in enumerate(range(min_k, max_k)):
            np.random.seed(0)
            Q = rangefinder(A, k=currk, num_pass=num_pass, aps=aps)
            res = residual(Q, A)
            # Best rank "currk" approx of A would result in a residual
            # with operator norm of s[currk].
            print('\t%s' % str(res - s[currk]))
            results[ik, inp] = res
    return results


def ex1():
    np.set_printoptions(precision=4)
    m, n = 200, 50
    k = 10
    H = np.random.standard_normal(size=(m, k))
    H = H * np.logspace(1, -3, num=k)  # scale the columns of H.
    G = np.random.standard_normal(size=(k, n))
    A = H @ G

    max_pass = 7
    min_pass = 1
    max_k = k + 2
    min_k = k // 2

    print(50*'-' + '\nNever stabilize\n' + 50*'-')
    results_nostab = run_rangefinders(max_pass, min_pass, max_k, min_k, A, aps=np.inf)
    print(50 * '-' + '\nStabilize after each pass\n' + 50 * '-')
    results_allstab = run_rangefinders(max_pass, min_pass, max_k, min_k, A, aps=1)
    diff = results_nostab - results_allstab
    print('\nApprox error: (never stabilize) - (stabilize after each pass)')
    print('\trows: increasing rank\n\tcols: increasing number of passes')
    print(diff)
    pass


if __name__ == '__main__':
    ex1()
