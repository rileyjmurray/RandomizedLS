import unittest
import numpy as np
from riley.protomodules import acm204_linsys as linsys
from scipy.linalg import cho_factor


class TestLinSys(unittest.TestCase):

    def test_CG(self):
        np.random.seed(0)
        A = np.random.randn(3, 5)
        rhs = np.random.randn(3)

        G = A @ A.T
        x_star = np.linalg.solve(G, rhs)

        x = np.zeros(3)
        L = np.eye(3)
        residuals = np.zeros(3)
        linsys.dense_pcg(A, rhs, x, L, tol=1e-10, maxit=3, residuals=residuals, reg=0.0)

        delta = np.linalg.norm(x - x_star)
        self.assertLessEqual(delta, 1e-8)

    def test_cho_solve(self):
        np.random.seed(0)
        A = np.random.randn(4, 10)
        G = A @ A.T
        L = cho_factor(G, lower=True)[0]
        x_star = np.random.randn(4)
        h = G @ x_star

        x = np.zeros(4)
        _ = linsys.py_cho_solve(L, h, x)

        delta = np.linalg.norm(x - x_star)
        self.assertLessEqual(delta, 1e-8)

    def test_pcg(self):
        np.random.seed(0)
        A = np.random.randn(3, 5)
        rhs = np.random.randn(3)

        G = A @ A.T
        x_star = np.linalg.solve(G, rhs)
        L = np.linalg.cholesky(G)
        x = np.zeros(3)
        residuals = np.zeros(1)
        linsys.dense_pcg(A, rhs, x, L, maxit=1, tol=1e-10, residuals=residuals, reg=0.0)

        delta = np.linalg.norm(x - x_star)
        self.assertLessEqual(delta, 1e-8)

    def test_blendenpik(self):
        np.random.seed(0)
        n, m = 100, 10
        d = 20
        A = np.random.randn(n, m)
        x0 = np.random.randn(m)
        b0 = A @ x0
        b = b0 + 0.05 * np.random.randn(n)
        x_bp, residuals, (r, c) = linsys.blendenpik_srct(A, b, d, 1e-8, m//2)
        x_np = np.linalg.lstsq(A, b)
        self.assertAlmostEqual(np.linalg.norm(x_bp - x_bp), 0.0, 1e-4)


class TestSketching(unittest.TestCase):

    def test_srct_approx_chol(self):
        np.random.seed(0)
        n = 1000
        m = 10
        d = int(10 * m * np.log(m))
        num_trials = 10
        errs = np.zeros(num_trials)
        for t in range(num_trials):
            A = np.random.randn(m, n)
            A *= 1000 * np.random.rand(n)
            G = A @ A.T
            L = np.zeros(shape=(m, m))
            L, _, _ = linsys.srct_approx_chol(d, A, L, 0.0)
            L[np.triu_indices(m, k=1)] = 0.0
            Gprecond = np.linalg.inv(L) @ G @ np.linalg.inv(L.T)
            delta = Gprecond - np.eye(m)
            err = np.linalg.norm(delta, ord=2)  # operator norm
            errs[t] = err
        assert np.mean(errs) <= 0.9  # a 0.1 embedding
