import unittest
import numpy as np
from riley.protomodules.blendenpik import blendenpik_srct, srct_precond


class TestLinSys(unittest.TestCase):

    def test_blendenpik_lsqr(self):
        np.random.seed(0)
        n, m = 1000, 50
        d = 3*50
        A = np.random.randn(n, m)
        x0 = np.random.randn(m)
        b0 = A @ x0
        b = b0 + 0.05 * np.random.randn(n)
        x_bp, flag, iternum, (r, e, p) = blendenpik_srct(A, b, d, 1e-8, m)
        self.assertLessEqual(iternum / m, 0.85)
        x_np = np.linalg.lstsq(A, b)[0]
        self.assertAlmostEqual(np.linalg.norm(x_bp - x_np), 0.0, places=4)


class TestSketching(unittest.TestCase):

    def test_srct_approx_chol(self):
        np.random.seed(0)
        n = 1000
        m = 10
        d = int(10 * m * np.log(m))
        num_trials = 10
        errs = np.zeros(num_trials)
        for t in range(num_trials):
            A = np.random.randn(n, m)
            A *= 100 * np.random.rand(m)
            G = A.T @ A
            R, (r, e, p) = srct_precond(A, d, True, 1e-8)
            R[np.tril_indices(m, k=-1)] = 0.0
            Rinv = np.linalg.inv(R)
            Gprecond = Rinv.T @ G @ Rinv
            delta = Gprecond - np.eye(m)
            err = np.linalg.norm(delta, ord=2)  # operator norm
            errs[t] = err
        assert np.mean(errs) <= 0.7  # a 0.3 embedding
