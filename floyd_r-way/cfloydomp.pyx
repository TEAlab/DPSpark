import numpy as np
cimport numpy as np

cdef extern from "fw_rdp_omp.h":
    void fw_D(double *X, double *U, double *V, double N, double R)
    void fw_C(double *X, double *V, double N, double R)
    void fw_B(double *X, double *U, double N, double R) 
    void fw_A(double *X, double N, double R)
    # double timesTwo(double val)
    # void prdoubleName(const char *name)

# def prdouble_name(name: bytes) -> None:
# def prdouble_name(name):
#     prdoubleName(name)

# def fwD_(double[:, :] X, double[:, :] U, double[:, :] V, N):
def fwD(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=2, mode='c'] U, np.ndarray[double, ndim=2, mode='c'] V, N, R):
    # fw_D(&X[0, 0], &U[0, 0], &V[0, 0], N)
    fw_D(<double *> np.PyArray_DATA(X), <double *> np.PyArray_DATA(U), <double *> np.PyArray_DATA(V), N, R)
    return X


# def fwC_(double[:, :] X, double[:, :] V, N):
def fwC(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=2, mode='c'] V, N, R):
    fw_C(<double *> np.PyArray_DATA(X), <double *> np.PyArray_DATA(V), N, R)
    return X



# def fwB_(double[:, :] X, double[:, :] U, N):
def fwB(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=2, mode='c'] U, N, R):
    fw_B(<double *> np.PyArray_DATA(X), <double *> np.PyArray_DATA(U), N, R)
    return X


# def fwA_(double[:, :] X, N):
def fwA(np.ndarray[double, ndim=2, mode='c'] X, N, R):
    # fw_A(&X[0, 0], N)
    fw_A(<double *> np.PyArray_DATA(X), N, R)
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
