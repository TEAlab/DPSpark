import numpy as np
cimport numpy as np

cdef extern from "ge_rdp_omp.h":
    void ge_D(double *X, double *U, double *V, double *W, int n_block, int N_total, int block_I, int block_J, int block_kk, int R)
    void ge_C(double *X, double *V, int n_block, int N_total, int block_I, int block_J, int block_kk, int R)
    void ge_B(double *X, double *U, int n_block, int N_total, int block_I, int block_J, int block_kk, int R)
    void ge_A(double *X, int n_block, int N_total, int block_I, int block_J, int block_kk, int R)
    # double timesTwo(double val)
    # void prdoubleName(const char *name)

# def prdouble_name(name: bytes) -> None:
# def prdouble_name(name):
#     prdoubleName(name)

# def fwD_(double[:, :] X, double[:, :] U, double[:, :] V, N):
def geD(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=2, mode='c'] U, 
    np.ndarray[double, ndim=2, mode='c'] V, np.ndarray[double, ndim=2, mode='c'] W, n, N, I, J, K, R):
    # fw_D(&X[0, 0], &U[0, 0], &V[0, 0], N)
    ge_D(<double *> np.PyArray_DATA(X), <double *> np.PyArray_DATA(U), <double *> np.PyArray_DATA(V), <double *> np.PyArray_DATA(W), n, N, I, J, K, R)
    return X


# def fwC_(double[:, :] X, double[:, :] V, N):
def geC(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=2, mode='c'] V, n, N, I, J, K, R):
    ge_C(<double *> np.PyArray_DATA(X), <double *> np.PyArray_DATA(V), n, N, I, J, K, R)
    return X


# def fwB_(double[:, :] X, double[:, :] U, N):
def geB(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=2, mode='c'] U, n, N, I, J, K, R):
    ge_B(<double *> np.PyArray_DATA(X), <double *> np.PyArray_DATA(U), n, N, I, J, K, R)
    return X


# def fwA_(double[:, :] X, N):
def geA(np.ndarray[double, ndim=2, mode='c'] X, n, N, I, J, K, R):
    # fw_A(&X[0, 0], N)
    ge_A(<double *> np.PyArray_DATA(X), n, N, I, J, K, R)
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
