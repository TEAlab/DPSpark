#ifndef FW_H
#define FW_H

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

# ifndef DATA_TYPE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%f "
# endif

/*
#ifndef RWAY
    #define RWAY 2
#endif
*/

#ifndef BASESIZE
    #define BASESIZE 64
#endif

void fw_D(DATA_TYPE *X, DATA_TYPE *U, DATA_TYPE *V, int N, int R);

void fw_C(DATA_TYPE *X, DATA_TYPE *V, int N, int R);

void fw_B(DATA_TYPE *X, DATA_TYPE *U, int N, int R);

void fw_A(DATA_TYPE *X, int N, int R);

#endif
