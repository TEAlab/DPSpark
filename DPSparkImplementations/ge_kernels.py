__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"


import numba as nb
import numpy as np

'''
    Iterative kernels
'''
def update_iter(w_block, u_block, v_block, x_block, n, I_, J_, K_):
    return _update_iter(np.ascontiguousarray(w_block), np.ascontiguousarray(u_block), np.asfortranarray(v_block), x_block, n, I_, J_, K_)
@nb.jit(nopython=True)
def _update_iter(w_block, u_block, v_block, x_block, n, I_, J_, K_):
    for k in range(x_block.shape[0]):
        K = K_*x_block.shape[0]+k
        for i in range(x_block.shape[0]):
            I = I_*x_block.shape[0]+i
            for j in range(x_block.shape[0]):
                J = J_*x_block.shape[0]+j
                if K < (n-1) and I > K and J >= K:
                    x_block[i, j] -= (u_block[i, k] * v_block[k, j])/w_block[k, k]
    return x_block

def funcA_iter(x_block, n, I_, J_, K_): # all inputs are of size (b x b)
    return update_iter(x_block, x_block, x_block, x_block, n, I_, J_, K_)

def funcB_iter(x_block, u_block, n, I_, J_, K_): # all inputs are of size (b x b)
    return update_iter(u_block, u_block, x_block, x_block, n, I_, J_, K_)

def funcC_iter(x_block, v_block, n, I_, J_, K_): # all inputs are of size (b x b)
    return update_iter(v_block, x_block, v_block, x_block, n, I_, J_, K_)

def funcD_iter(x_block, u_block, v_block, w_block, n, I_, J_, K_): # all inputs are of size (b x b)
    return update_iter(w_block, u_block, v_block, x_block, n, I_, J_, K_)
