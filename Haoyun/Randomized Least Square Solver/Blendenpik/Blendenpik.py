# My Blendenpik has some bugs, and I need some time to fix that.

import numpy as np
from scipy.sparse import coo_matrix, bmat
from scipy.fftpack import dctn, fftn
from scipy.sparse.linalg import lsqr
from math import ceil

# import xalglib
# !pip install sparseqr
# import sparseqr

class Blendenpik(object):
    use_xalglib = True

    # Constructor
    def __init__(self, A, b, gamma=2):
        self.A = A
        self.b = b
        self.gamma = gamma  # from 1.5 to 10

    def solve(self):

        self.m_tilde = ceil(self.A.shape[0] / 1000) * 1000

        if self.m_tilde > self.A.shape[0]:
            zero_mat = coo_matrix((self.m_tilde - self.A.shape[0], self.A.shape[1]))
            #             M = spmat.bmat([[self.A], [zero_mat]],format = "lil")
            M = bmat([[self.A], [zero_mat]]).toarray()
        else:
            M = self.A

        diag_D = np.random.choice([-1, 1], self.m_tilde)

        # Choose from DCT or FHT
        # FHT is recommended by the original paper. But my implementation with alglib
        # is not so fast.
        DM = np.zeros((self.m_tilde, self.A.shape[1]))
        for i in range(self.m_tilde):
            DM[i, :] = diag_D[i] * M[i, :]

        M = self.DCT2D(DM)

        # prob = self.gamma * self.A.shape[1] / self.m_tilde
        # diag_S = np.random.choice([1, 0], self.m_tilde, p = [prob, 1 - prob])
        # S = spmat.diags(diag_S).toarray()
        # sampledM = S.dot(M)

        sampled_rate = min(1, self.gamma * self.A.shape[1] / self.m_tilde)
        sampled_rows = np.random.choice(self.m_tilde, int(sampled_rate * self.m_tilde))
        sampledM = M[sampled_rows, :]

        # For QR decomposition,
        # numpy is the fastest followed by scipy, sparseqr.
        Q, R = np.linalg.qr(sampledM, mode='reduced')
        # x, iternum = bnp_lsqr(self.A, self.b, R)
        total = lsqr(A=self.A.dot(np.linalg.inv(R)), b=self.b, tol=1e-14)
        z = total[0]
        iternum = total[2]
        x = np.linalg.inv(R).dot(z)
        return x, iternum

    def DCT2D(self, X):
        #         Y = fft.fftn(x.tolist().toarray(), shape=x.shape).real
        Y = fftn(X, shape=X.shape).real
        # Y = dctn(X, shape = X.shape, norm='ortho')
        return Y

    # '''
    # This code requires Alglib for FHT.
    # '''
    # if use_xalglib:
    #     def FHT2D(self, x):
    #         '''
    #         FHT2D is FHT for a 2-d array, which is separable to
    #         the following FHTs for 1-d arrays.
    #         - First, FHT for each row of the 2-d array.
    #         - Second, FHT for each column of the 2-d array.
    #         We use Alglib FHT for 1-d array.
    #         '''
    #         n_row = x.shape[0]
    #         n_col = x.shape[1]
    #         for r in range(n_row):
    #             #y = self.fhtr1d(x[r,:].todense(), n_col)
    #             index = x[r,:].nonzero()[1]
    #             if len(index) > 0:
    #                 x[r,:] = self.fhtr1d(x[r,:].todense(), n_col)
    #             #print("x", x[r,:])

    #         for c in range(n_col):
    #             index = x[:,c].nonzero()[1]
    #             if len(index) > 0:
    #                 x[:,c].T = self.fhtr1d(x[:,c].todense(), n_row)

    #     def fhtr1d(self, x, n):
    #         '''
    #         This is a wrapper for fhtr1d in ALglib.
    #         Although xalglib.fhtr1d has python interface, the interface receives only a python list.
    #         This wrapper allows Alglib to receive and return a numpy array.
    #         '''
    #         _error_msg = ctypes.c_char_p(0)
    #         __c = ctypes.c_void_p(0)
    #         __x = xalglib.x_vector(cnt=n, datatype=xalglib.DT_REAL, owner=xalglib.OWN_CALLER,
    #                       last_action=0,ptr=xalglib.x_multiptr(p_ptr=x.ctypes.data))
    #         __n = xalglib.c_ptrint_t(n)

    #         xalglib._lib_alglib.alglib_fhtr1d(
    #             ctypes.byref(_error_msg),
    #             ctypes.byref(__x),
    #             ctypes.byref(__n))

    #         INTP = ctypes.POINTER(ctypes.c_double)
    #         ptr = ctypes.cast(__x.ptr.p_ptr, INTP)
    #         return np.fromiter(ptr, dtype=np.float, count=n)