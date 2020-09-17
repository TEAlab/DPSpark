__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"

import numpy as np
import numba as nb

'''
    Iterative kernels
'''


def update_iter(u_block, x_block, n, I_, J_, K_):
    return _update_iter(np.ascontiguousarray(u_block), np.ascontiguousarray(x_block), n, I_, J_, K_)
@nb.jit(nopython=True)
def _update_iter(u_block, x_block, n, I_, J_, K_):
    # For testing purposes, rather than passing f_matrix_broadcast, we call this function
    def f_matrix(i, j):
        return float(i+j)
    for k in range(x_block.shape[0]-1, -1, -1):
        K = K_*x_block.shape[0]+k
        for j in range(x_block.shape[0]-1, -1, -1):
            J = J_*x_block.shape[0]+j
            for i in range(x_block.shape[0]-1, -1, -1):
                I = I_*x_block.shape[0]+i
                min1 = min(K-2, n-3)
                min2 = min(J-1, n-4)
                if ((K < n) and (K >= 3) and (J <= min1) and (J >= I+1) and (I <= min2)):
                    x_block[i, j] = max(x_block[i, j], u_block[j+1, k] + f_matrix(J+1, min(K, 2*J-I+1)))
    return x_block

def funcA_iter(block_info, n):
    ((I_, J_), x_block) = block_info
    return update_iter(x_block, x_block, n, I_, J_, I_)

def funcX_iter(block_info, u_block_info, n):
    ((I_, J_), x_block) = block_info
    ((UI_, UJ_), u_block) = u_block_info
    return update_iter(u_block, x_block, n, I_, J_, UJ_)
