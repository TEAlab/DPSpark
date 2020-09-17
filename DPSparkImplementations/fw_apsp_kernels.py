__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"


import numpy as np
import numba as nb
from scipy.sparse.csgraph import floyd_warshall as apsp_fw

'''
    Iterative kernels, Directly taken from Schoeneman's ICPP 2019 implementation
                                            (for the sake of experimental comparision):


    Min-plus matrix multiplication of matrices A*B = C. D_ returned is elementwise min of C and input D_.
    Compiled for better performance using numba.
    Input matrices A, B stored C, Fortran contiguous, respectively, for best performance.

'''

def funA_iter(x_block):
    return apsp_fw(x_block, directed = True, unweighted = False, overwrite = True)

def minmpmatmul(A, B, D_):
    return _minmpmatmul(np.ascontiguousarray(A), np.asfortranarray(B), D_)
@nb.jit(nopython=True)
def _minmpmatmul(A, B, D_):
    assert A.shape[1] == B.shape[0], 'Matrix dimension mismatch in minmpmatmul().'
    for k in range(A.shape[0]):
        for i in range(B.shape[1]):
            somesum = np.inf
            for j in range(A.shape[1]):
                D_[i, j] = min(D_[i, j], A[i, k] + B[k, j])
    return D_

def funcB_iter(x_block, u_block):
    return minmpmatmul(u_block, x_block, x_block)

def funcC_iter(x_block, v_block):
    return minmpmatmul(x_block, v_block, x_block)

def funcD_iter(x_block, u_block, v_block):
    return minmpmatmul(u_block, v_block, x_block)
