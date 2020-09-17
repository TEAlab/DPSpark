#ifndef PAF_H
#define PAF_H

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)
	
# ifndef DATA_TYPE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%lf "
# endif

#ifndef RWAY
    #define RWAY 2
#endif

#ifndef BASESIZE
    #define BASESIZE 32
#endif

void paf_B(DATA_TYPE *X, DATA_TYPE *U, int n_block, int N_total, int block_I, int block_J, int block_kk);

void paf_A(DATA_TYPE *X, int n_block, int N_total, int block_I, int block_J, int block_kk);

#endif
