#ifndef GE_H
#define GE_H

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

# ifndef DATA_TYPE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%f "
# endif

//#ifndef RWAY
//    #define RWAY 2
//#endif

#ifndef BASESIZE
    #define BASESIZE 64
#endif

void ge_D(DATA_TYPE *X, DATA_TYPE *U, DATA_TYPE *V, DATA_TYPE *W, int n_block, int N_total, int block_I, int block_J, int block_kk, int R);

void ge_C(DATA_TYPE *X, DATA_TYPE *V, int n_block, int N_total, int block_I, int block_J, int block_kk, int R);

void ge_B(DATA_TYPE *X, DATA_TYPE *U, int n_block, int N_total, int block_I, int block_J, int block_kk, int R);

void ge_A(DATA_TYPE *X, int n_block, int N_total, int block_I, int block_J, int block_kk, int R);

#endif
