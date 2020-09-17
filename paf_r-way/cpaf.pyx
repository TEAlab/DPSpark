import numpy as np
cimport numpy as np

cdef extern from "paf_rdp_omp.h":
    void paf_B(double *X, double *U, int n_block, int N_total, int block_I, int block_J, int block_kk)
    void paf_A(double *X, int n_block, int N_total, int block_I, int block_J, int block_kk)
    # double timesTwo(double val)
    # void prdoubleName(const char *name)

# def prdouble_name(name: bytes) -> None:
# def prdouble_name(name):
#     prdoubleName(name)

# def fwB_(double[:, :] X, double[:, :] U, N):
# paf.pafB(x_block, u_block, x_block.shape[0], n, I_, J_, k)
def pafB(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=2, mode='c'] U, n, N, I, J, K):
    paf_B(<double *> np.PyArray_DATA(X), <double *> np.PyArray_DATA(U), n, N, I, J, K)
    return X


# def fwA_(double[:, :] X, N):
# paf.pafA(x_block, x_block.shape[0], n, I_, J_, I_)
def pafA(np.ndarray[double, ndim=2, mode='c'] X, n, N, I, J, K):
    # fw_A(&X[0, 0], N)
    paf_A(<double *> np.PyArray_DATA(X), n, N, I, J, K)
    return X

'''
def fwA(X, N):
    fwA_(X, N)
    return X
'''
'''
def fwD(X_, U, V, N):
    prdouble X_.flags
    X = memoryview(X_.copy())
    fwD_(X, U, V, N)
    return X
'''
'''
def fwC(X_, V, N):
    prdouble X_.flags
    X = memoryview(X_.copy())
    fwC_(X, V, N)
    return X
'''


'''
def fwB(X_, U, N):
    prdouble X_.flags
    X = memoryview(X_.copy())
    fwB_(X, U, N)
    return X
'''


# def times_two(name: bytes) -> None:
    # return timesTwo(value)
#     prdoubleName(name)
