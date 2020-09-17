


/*
Copy: 
    scp fw_rdp.cpp zafahmad@stampede2.tacc.utexas.edu:/home1/05072/zafahmad/SC_2019_submission/

Compile:
    module load papi/5.5.1
    icc -DDEBUG -O3 -fopenmp -xhost -AVX512 fw_rdp_omp.cpp -o fw_rdp -I$TACC_PAPI_INC -Wl,-rpath,$TACC_PAPI_LIB -L$TACC_PAPI_LIB -lpapi
    icc -O3 -fopenmp -xhost -AVX512 fw_rdp_omp_poly_base.c -o exec -DDEBUG -DPOLYBENCH  polybench-c-3.2/utilities/polybench.c -DPOLYBENCH_TIME -DSMALL_DATASET -Ipolybench-c-3.2/utilities/ -I. -I$TACC_PAPI_INC -Wl,-rpath,$TACC_PAPI_LIB -L$TACC_PAPI_LIB -lpapi

    icc fw_rdp_omp.c -o fw_rdp -DDEBUG -DPOLYBENCH polybench-c-4.2/utilities/polybench.c -DPOLYBENCH_TIME -DPOLYBENCH_USE_RESTRICT -Ipolybench-c-4.2/utilities/ -I. -O2 -qopenmp -xKNL -qopt-prefetch=5 -xhost -AVX512 -lm 
Execute: 
    ./fw_rdp N B R P
    ./fw_rdp 1024 128 2 272


    export GOMP_CPU_AFFINITY='0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268'
    export OMP_NUM_THREADS=68
    export OMP_PROC_BIND=true
    # export OMP_NUM_THREADS=272
*/

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
// #include <iostream>

#include <omp.h>
#include "fw_rdp_omp.h"

// #include <cilk/cilk.h>
// #include <cilk/cilk_api.h>    
// #include "cilktime.h"
#ifdef USE_PAPI
#include <papi.h>    
#include "papilib.h"
#endif

#ifdef POLYBENCH
    #include <polybench.h>
#endif

// using namespace std;

// Compile command: gcc -o ge ge.c -lm

// For the followings, I have tested:
/*
TEST #1) ./ge 16 4 2
TEST #2) ./ge 512 64 32
TEST #3) ./ge 1024 64 32

*/
/*
#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

# ifndef DATA_TYPE
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
# endif
*/
// #define CACHE_OPTIMIZED
#define CACHE_OPTIMIZEDX
/*
    STEP 1) The input algorithm is the triply-nested for loop version passed to 
           PoCC to get parametric tiled version of the code
*/

int NN; // original size of the matrix

void fw(DATA_TYPE **D, int N) {
    int i, j, k;
    int counter = 0;
    for (k = 0; k < N; ++k) {
        for (i = 0; i < N; ++i) { 
            for (j = 0; j < N; ++j) {
                // if (i > k && j >= k) {
                    /*printf("D[%d][%d] -= (D[%d][%d] * D[%d][%d]) / D[%d][%d]\n",
                            i, j, i, k, k, j, k, k);*/
                    D[i][j] = min(D[i][j], (D[i][k] + D[k][j]));
                    counter++;
                // }
            }
        }
    }
    // printf("Total number of updates are: %d", counter);
}


/*
    STEP 5) Recursive but applying index set splitting AND having multiple functions

*/


// in function A, X = U = V = W --> The least parallel one
void fw_rec3_A(DATA_TYPE *X, int N, int R, int base_size,
            int k_lb, int k_ub, int i_lb, int i_ub, 
            int j_lb, int j_ub);

// in function B, X = V but X != U and X != W
void fw_rec3_B(DATA_TYPE *X, DATA_TYPE *U, DATA_TYPE *W,
               int N, int R, int base_size,
               int k_lb, int k_ub, int i_lb, int i_ub, 
               int j_lb, int j_ub);

// in function C, X = U but X != V and X != W
void fw_rec3_C(DATA_TYPE *X, DATA_TYPE *V, DATA_TYPE *W,
               int N, int R, int base_size,
               int k_lb, int k_ub, int i_lb, int i_ub, 
               int j_lb, int j_ub);

// in function D, X != U and X != V and X != W --> The most parallel one
void fw_rec3_D(DATA_TYPE *X, DATA_TYPE *U, DATA_TYPE *V, DATA_TYPE *W,
               int N, int R, int base_size,
               int k_lb, int k_ub, int i_lb, int i_ub, 
               int j_lb, int j_ub);


void fw_D(DATA_TYPE *X, DATA_TYPE *U, DATA_TYPE *V, int N, int R){
    NN = N;
    //int R = RWAY;
    int base_size = BASESIZE;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            fw_rec3_D(X, U, V, X, N, R, base_size, 0, N, 0, N, 0, N);
        }
    }
}

void fw_C(DATA_TYPE *X, DATA_TYPE *V, int N, int R){
    NN = N;
    //int R = RWAY;
    int base_size = BASESIZE; 
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            fw_rec3_C(X, V, X, N, R, base_size, 0, N, 0, N, 0, N);
        }
    }
}

void fw_B(DATA_TYPE *X, DATA_TYPE *U, int N, int R){
    NN = N;
    //int R = RWAY;
    int base_size = BASESIZE;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            fw_rec3_B(X, U, X, N, R, base_size, 0, N, 0, N, 0, N);
        }
    }
}

void fw_A(DATA_TYPE *X, int N, int R){
    NN = N;
    //int R = RWAY;
    int base_size = BASESIZE;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            printf("Number of threads: %d -----------------------------------------------------------", omp_get_num_threads());
            fw_rec3_A(X, N, R, base_size, 0, N, 0, N, 0, N);
        }
    }
}


// in function D, X != U and X != V and X != W --> The most parallel one
void fw_rec3_D(DATA_TYPE *X, DATA_TYPE *U, DATA_TYPE *V, DATA_TYPE *W,
               int N, int R, int base_size,
               int k_lb, int k_ub, int i_lb, int i_ub, 
               int j_lb, int j_ub) {

    if (k_lb >= NN || i_lb >= NN || j_lb >= NN)
        return ;

    // printf("N: %d NN: %d R: %d base: %d klb: %d kub: %d ilb: %d iub: %d jlb: %d jub: %d\n", N, NN, R, base_size, k_lb, k_ub,
    //         i_lb, i_ub, j_lb, j_ub);


    int i, j, k;
    // base case
    if ((k_ub - k_lb) <= base_size || N <= R) {
#ifdef USE_PAPI
    int id = tid();
    papi_for_thread(id);
    int retval = 0;
    if ( (retval=PAPI_start(EventSet[id])) != PAPI_OK)
        ERROR_RETURN(retval);
#endif    
#ifndef CACHE_OPTIMIZEDX
        for (k = k_lb; k < k_ub && k < NN; ++k) {
            for (i = i_lb; i < i_ub && i < NN; ++i) {
                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (U[i][k] * V[k][j])/W[k][k];
                    // X[i][j] = min(X[i][j], (U[i][k] + V[k][j]));
                    X[i*NN + j] = min(X[i*NN + j], (U[i*NN + k] + V[k*NN + j]));
                    // }
                }
            }
        }
#else
        DATA_TYPE u_col_major[base_size * base_size], v_row_major[base_size * base_size];
        DATA_TYPE x_row_major[base_size * base_size];
        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                x_row_major[(i-i_lb)*base_size + (j-j_lb)] = X[i*NN+j]; //X[i][j];

        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (k = k_lb; k < k_ub && k < NN; ++k)
                u_col_major[(k-k_lb)*base_size + (i-i_lb)] = U[i*NN+k]; //U[i][k];

        for (k = k_lb; k < k_ub && k < NN; ++k)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                v_row_major[(k-k_lb)*base_size + (j-j_lb)] = V[k*NN+j]; //V[k][j];

        for (k = k_lb; k < k_ub && k < NN; ++k) {
            DATA_TYPE w_kk = W[k*NN+k]; //W[k][k];

            for (i = i_lb; i < i_ub && i < NN; ++i) {

                // DATA_TYPE div_ik_kk = U[i][k]/w_kk;
                DATA_TYPE div_ik_kk = u_col_major[(k - k_lb) * base_size + (i - i_lb)]; ///w_kk;

                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (U[i][k] * V[k][j])/W[k][k];
                        // X[i][j] -= div_ik_kk * v_row_major[(k - k_lb) * base_size + (j - j_lb)];
                        // x_row_major[(i-i_lb)*base_size + (j-j_lb)] -= div_ik_kk * v_row_major[(k - k_lb) * base_size + (j - j_lb)];
                    x_row_major[(i-i_lb)*base_size + (j-j_lb)] = min(x_row_major[(i-i_lb)*base_size + (j-j_lb)], 
                                                                        (div_ik_kk + v_row_major[(k - k_lb) * base_size + (j - j_lb)]));
                    // }
                }
            }
        }

        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j){
                X[i*NN+j] = x_row_major[(i-i_lb)*base_size + (j-j_lb)]; //X[i][j]
            }
#endif


#ifdef USE_PAPI
    countMisses(id );
#endif      
        return;
    }
    int ii, jj, kk;
    int tile_size = N/R;
    for (kk = 0; kk < R && k_lb + kk * (tile_size) < NN; kk++) {
        
// All the following fw_rec3_D(...) functions can run in parallel.
// IN PARALLEL:

        // Only possible case is this as all the input/read tiles are different than output/write tile
        // #pragma omp parallel
        {
            // #pragma omp for collapse(2)
            for (ii = 0; ii < R && i_lb + ii * (tile_size) < NN; ii++) {
                for (jj = 0; jj < R && j_lb + jj * (tile_size) < NN; jj++) {
                    #pragma omp task
                    fw_rec3_D(X, U, V, W, tile_size, R, base_size,
                              k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                              i_lb + ii * (tile_size), i_lb + (ii + 1) * tile_size,
                              j_lb + jj * (tile_size), j_lb + (jj + 1) * tile_size); 
                }
            }
        }
        #pragma omp taskwait   
// JOIN - SYNC
    }
}


// in function C, X = U but X != V and X != W
void fw_rec3_C(DATA_TYPE *X, DATA_TYPE *V, DATA_TYPE *W,
               int N, int R, int base_size,
               int k_lb, int k_ub, int i_lb, int i_ub, 
               int j_lb, int j_ub) {
    int i, j, k;

    // printf("N: %d NN: %d R: %d base: %d klb: %d kub: %d ilb: %d iub: %d jlb: %d jub: %d\n", N, NN, R, base_size, k_lb, k_ub,
    //         i_lb, i_ub, j_lb, j_ub);

    if (k_lb >= NN || i_lb >= NN || j_lb >= NN)
        return ;

    // base case
    if ((k_ub - k_lb) <= base_size || N <= R) {
#ifdef USE_PAPI
    int id = tid();
    papi_for_thread(id);
    int retval = 0;
    if ( (retval=PAPI_start(EventSet[id])) != PAPI_OK)
        ERROR_RETURN(retval);
#endif     
#ifndef CACHE_OPTIMIZEDX
        for (k = k_lb; k < k_ub && k < NN; ++k) {
            for (i = i_lb; i < i_ub && i < NN; ++i) {
                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (X[i][k] * V[k][j])/W[k][k];
                    // X[i][j] = min(X[i][j], (X[i][k] + V[k][j])); ///W[k][k];
                    X[i*NN + j] = min(X[i*NN + j], (X[i*NN + k] + V[k*NN + j]));
                    // }
                }
            }
        }
#else
        DATA_TYPE v_row_major[base_size * base_size];
        DATA_TYPE x_row_major[base_size * base_size];
        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                x_row_major[(i-i_lb)*base_size + (j-j_lb)] = X[i*NN+j]; //X[i][j];

        for (k = k_lb; k < k_ub && k < NN; ++k)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                v_row_major[(k-k_lb)*base_size + (j-j_lb)] = V[k*NN+j]; //V[k][j];

        for (k = k_lb; k < k_ub && k < NN; ++k) {
            DATA_TYPE w_kk = W[k*NN+k]; //W[k][k];

            for (i = i_lb; i < i_ub && i < NN; ++i) {

                // DATA_TYPE div_ik_kk = U[i][k]/w_kk;

                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (U[i][k] * V[k][j])/W[k][k];
                        // X[i][j] -= div_ik_kk * v_row_major[(k - k_lb) * base_size + (j - j_lb)];
                        // x_row_major[(i-i_lb)*base_size + (j-j_lb)] -= x_row_major[(i-i_lb)*base_size + (k-k_lb)] 
                        //                                                 * v_row_major[(k - k_lb) * base_size + (j - j_lb)] / w_kk;
                        x_row_major[(i-i_lb)*base_size + (j-j_lb)] = min (x_row_major[(i-i_lb)*base_size + (j-j_lb)], (x_row_major[(i-i_lb)*base_size + (k-k_lb)] 
                                                                        + v_row_major[(k - k_lb) * base_size + (j - j_lb)])); // / w_kk;
                    // }
                }
            }
        }

        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j){
                X[i*NN+j] = x_row_major[(i-i_lb)*base_size + (j-j_lb)]; //X[i][j]
            }
#endif
#ifdef USE_PAPI
    countMisses(id );   
#endif      
        return;
    }
    int ii, jj, kk;
    int tile_size = N/R;
    for (kk = 0; kk < R  && k_lb + kk * (tile_size) < NN; kk++) {
        // Applying the same idea of index set Splitting

// All the following fw_rec3_C(...) functions can run in parallel.
// IN PARALLEL:

        // CASE 1: kk = jj 
        // #pragma omp parallel
        {
            // #pragma omp for
            for (ii = 0; ii < R && i_lb + ii * (tile_size) < NN; ii++) {
                #pragma omp task
                fw_rec3_C(X, V, W, tile_size, R, base_size,
                          k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                          i_lb + ii * (tile_size), i_lb + (ii + 1) * tile_size,
                          j_lb + kk * (tile_size), j_lb + (kk + 1) * tile_size);
            }
        }
        #pragma omp taskwait        
// JOIN - SYNC
// All the following fw_rec3_D(...) functions can run in parallel.
// IN PARALLEL:

        // else of CASE 1
        // CASE 2: kk != ii and kk != jj ==> Function D(X, U, V, ...)
        // #pragma omp parallel
        {
            // #pragma omp for collapse(2)
            for (ii = 0; ii < R && i_lb + ii * (tile_size) < NN; ii++) {
                for (jj = 0; jj < R && j_lb + jj * (tile_size) < NN; jj++) {
                    if (jj == kk) continue;
                    #pragma omp task
                    fw_rec3_D(X, X, V, W, tile_size, R, base_size,
                              k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                              i_lb + ii * (tile_size), i_lb + (ii + 1) * tile_size,
                              j_lb + jj * (tile_size), j_lb + (jj + 1) * tile_size); 
                }
            }
        }
        #pragma omp taskwait   
// JOIN - SYNC
    }   
}



// in function B, X = V but X != U and X != W
void fw_rec3_B(DATA_TYPE *X, DATA_TYPE *U, DATA_TYPE *W,
               int N, int R, int base_size,
               int k_lb, int k_ub, int i_lb, int i_ub, 
               int j_lb, int j_ub) {
    int i, j, k;
    if (k_lb >= NN || i_lb >= NN || j_lb >= NN)
        return ;

    // printf("N: %d NN: %d R: %d base: %d klb: %d kub: %d ilb: %d iub: %d jlb: %d jub: %d\n", N, NN, R, base_size, k_lb, k_ub,
    //         i_lb, i_ub, j_lb, j_ub);

    // base case
    if ((k_ub - k_lb) <= base_size || N <= R) {
#ifdef USE_PAPI
    int id = tid();
    papi_for_thread(id);
    int retval = 0;
    if ( (retval=PAPI_start(EventSet[id])) != PAPI_OK)
        ERROR_RETURN(retval);
#endif    
#ifndef CACHE_OPTIMIZEDX
        for (k = k_lb; k < k_ub && k < NN; ++k) {
            for (i = i_lb; i < i_ub && i < NN; ++i) {
                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (U[i][k] * X[k][j])/W[k][k];
                    X[i*NN + j] = min(X[i*NN + j], (U[i*NN + k] + X[k*NN + j])); ///W[k][k];
                    // }
                }
            }
        }
#else
        DATA_TYPE u_col_major[base_size * base_size];
        DATA_TYPE x_row_major[base_size * base_size];
        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                x_row_major[(i-i_lb)*base_size + (j-j_lb)] = X[i*NN+j]; //X[i][j];

        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (k = k_lb; k < k_ub && k < NN; ++k)
                u_col_major[(k-k_lb)*base_size + (i-i_lb)] = U[i*NN+k]; //U[i][k];

        for (k = k_lb; k < k_ub && k < NN; ++k) {
            DATA_TYPE w_kk = W[k*NN+k]; //W[k][k];

            for (i = i_lb; i < i_ub && i < NN; ++i) {

                // DATA_TYPE div_ik_kk = U[i][k]/w_kk;
                DATA_TYPE div_ik_kk = u_col_major[(k - k_lb) * base_size + (i - i_lb)]; ///w_kk;
                //int j_ub_min = min(j_ub, NN);
// #pragma GCC ivdep
                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (U[i][k] * V[k][j])/W[k][k];
                        // X[i][j] -= div_ik_kk * x_row_major[(k - k_lb) * base_size + (j - j_lb)];
                        // x_row_major[(i-i_lb)*base_size + (j-j_lb)] -= div_ik_kk * x_row_major[(k - k_lb) * base_size + (j - j_lb)];
                    x_row_major[(i-i_lb)*base_size + (j-j_lb)] = min (x_row_major[(i-i_lb)*base_size + (j-j_lb)], (div_ik_kk + x_row_major[(k - k_lb) * base_size + (j - j_lb)]));
                    // }
                }
            }
        }

        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                X[i*NN+j] = x_row_major[(i-i_lb)*base_size + (j-j_lb)]; //X[i][j]
            
#endif
#ifdef USE_PAPI
    countMisses(id );
#endif      
        return;
    }
    int ii, jj, kk;
    int tile_size = N/R;
    for (kk = 0; kk < R && k_lb + kk * (tile_size) < NN; kk++) {
        // Applying the same idea of index set Splitting

// All the following fw_rec3_B(...) functions can run in parallel.
// IN PARALLEL:

        // CASE 1: kk = ii but kk != jj ==> Function B(X, U, W, ...)
        // #pragma omp parallel
        {
            // #pragma omp for
            for (jj = 0; jj < R && j_lb + jj * (tile_size) < NN; jj++) {
                #pragma omp task
                fw_rec3_B(X, U, W, tile_size, R, base_size,
                          k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                          i_lb + kk * (tile_size), i_lb + (kk + 1) * tile_size,
                          j_lb + jj * (tile_size), j_lb + (jj + 1) * tile_size);
            }
        }
        #pragma omp taskwait   
// JOIN - SYNC
// All the following fw_rec3_D(...) functions can run in parallel.
// IN PARALLEL:
        // else of CASE 1
        // CASE 2: kk != ii ==> Function D(X, U, V, W, ...)
        // #pragma omp parallel
        {
            // #pragma omp for collapse(2)
            for (ii = 0; ii < R && i_lb + ii * (tile_size) < NN; ii++) {
                if (ii == kk) continue;
                for (jj = 0; jj < R && j_lb + jj * (tile_size) < NN; jj++) {
                    #pragma omp task
                    fw_rec3_D(X, U, X, W, tile_size, R, base_size,
                              k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                              i_lb + ii * (tile_size), i_lb + (ii + 1) * tile_size,
                              j_lb + jj * (tile_size), j_lb + (jj + 1) * tile_size); 
                }
            }
        }
        #pragma omp taskwait   
// JOIN - SYNC
    }
}

// in function A, X = U = V = W --> The least parallel one
void fw_rec3_A(DATA_TYPE *X, int N, int R, int base_size,
            int k_lb, int k_ub, int i_lb, int i_ub, 
            int j_lb, int j_ub) {

    if (k_lb >= NN || i_lb >= NN || j_lb >= NN)
        return ;

    // printf("N: %d NN: %d R: %d base: %d klb: %d kub: %d ilb: %d iub: %d jlb: %d jub: %d\n", N, NN, R, base_size, k_lb, k_ub,
    //         i_lb, i_ub, j_lb, j_ub);

    int i, j, k;
    // base case
    if ((k_ub - k_lb) <= base_size || N <= R) {
#ifdef USE_PAPI
    int id = tid();
    papi_for_thread(id);
    int retval = 0;
    if ( (retval=PAPI_start(EventSet[id])) != PAPI_OK)
        ERROR_RETURN(retval);
#endif
#ifndef CACHE_OPTIMIZEDX
        // printf("N: %d NN: %d R: %d base: %d klb: %d kub: %d ilb: %d iub: %d jlb: %d jub: %d\n", N, NN, R, base_size, k_lb, k_ub,
                    // i_lb, i_ub, j_lb, j_ub);
        for (k = k_lb; k < k_ub && k < NN; ++k) {
            for (i = i_lb; i < i_ub  && i < NN; ++i) {
                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (X[i][k] * X[k][j])/X[k][k];
                    X[i*NN + j] = min(X[i*NN + j], (X[i*NN + k] + X[k*NN + j])); ///X[k][k];
                        // (*counter)++;
                    // }
                }
            }
        }
#else
        DATA_TYPE x_row_major[base_size * base_size];
        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                x_row_major[(i-i_lb)*base_size + (j-j_lb)] = X[i*NN+j]; //X[i][j];

        for (k = k_lb; k < k_ub && k < NN; ++k) {
            for (i = i_lb; i < i_ub && i < NN; ++i) {
                for (j = j_lb; j < j_ub && j < NN; ++j) {
                    // if (i > k && j >= k) {
                        // X[i][j] -= (U[i][k] * V[k][j])/W[k][k];
                        // x_row_major[(i-i_lb)*base_size + (j-j_lb)] -= x_row_major[(i - i_lb)*base_size + (k - k_lb)] * 
                        //                         x_row_major[(k - k_lb) * base_size + (j - j_lb)] / 
                        //                         x_row_major[(k - k_lb) * base_size + (k - k_lb)];
                        x_row_major[(i-i_lb)*base_size + (j-j_lb)] = min (x_row_major[(i-i_lb)*base_size + (j-j_lb)], 
                                                x_row_major[(i - i_lb)*base_size + (k - k_lb)] + 
                                                x_row_major[(k - k_lb) * base_size + (j - j_lb)]);  
                                                // / x_row_major[(k - k_lb) * base_size + (k - k_lb)];
                    // }
                }
            }
        }

        for (i = i_lb; i < i_ub && i < NN; ++i)
            for (j = j_lb; j < j_ub && j < NN; ++j)
                X[i*NN+j] = x_row_major[(i-i_lb)*base_size + (j-j_lb)]; //X[i][j]
#endif

#ifdef USE_PAPI
    countMisses(id );
#endif      
        return;
    }
    int ii, jj, kk;
    int tile_size = N/R;
    for (kk = 0; kk < R && k_lb + kk * (tile_size) < NN; kk++) {
        // Applying the same idea of index set Splitting
        // CASE 1: kk = ii and kk = jj --> Function A(X, ...)
        fw_rec3_A(X, tile_size, R, base_size,
                  k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                  i_lb + kk * (tile_size), i_lb + (kk + 1) * tile_size,
                  j_lb + kk * (tile_size), j_lb + (kk + 1) * tile_size);
// following functions fw_rec3_B(...) can be in parallel with all the
// function calls fw_rec3_C(...) as they are writing to different tiles.
// IN PARALLEL:

        // CASE 2: kk = ii but kk != jj ==> Function B(X, U, ...)
        // #pragma omp parallel
        {
            // #pragma omp for
            for (jj = 0; jj < R && j_lb + jj * (tile_size) < NN; jj++) {
                if (jj == kk) continue;
                #pragma omp task
                fw_rec3_B(X, X, X, tile_size, R, base_size,
                          k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                          i_lb + kk * (tile_size), i_lb + (kk + 1) * tile_size,
                          j_lb + jj * (tile_size), j_lb + (jj + 1) * tile_size);
            }
        }

        #pragma omp taskwait   
        // CASE 3: kk = jj but kk != ii ==> Function C(X, V, ...)
        // #pragma omp parallel
        {
            // #pragma omp for
            for (ii = 0; ii < R && i_lb + ii * (tile_size) < NN; ii++) {
                if (ii == kk) continue;
                #pragma omp task
                fw_rec3_C(X, X, X, tile_size, R, base_size,
                          k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                          i_lb + ii * (tile_size), i_lb + (ii + 1) * tile_size,
                          j_lb + kk * (tile_size), j_lb + (kk + 1) * tile_size);
            }
        }
        #pragma omp taskwait   
// JOIN - SYNC
// All the following fw_rec3_D(...) functions can run in parallel.
// IN PARALLEL:
        // CASE 4: kk != ii and kk != jj ==> Function D(X, U, V, ...)
        // #pragma omp parallel
        {
            // #pragma omp for collapse(2)
            for (ii = 0; ii < R && i_lb + ii * (tile_size) < NN; ii++) {
                for (jj = 0; jj < R && j_lb + jj * (tile_size) < NN; jj++) {
                    if (ii == kk || jj == kk) continue;
                    #pragma omp task
                    fw_rec3_D(X, X, X, X, tile_size, R, base_size,
                              k_lb + kk * (tile_size), k_lb + (kk + 1) * tile_size,
                              i_lb + ii * (tile_size), i_lb + (ii + 1) * tile_size,
                              j_lb + jj * (tile_size), j_lb + (jj + 1) * tile_size); 
                }
            }
        }
        #pragma omp taskwait   
// JOIN - SYNC
    }
}

/*
void fw_rec_top_level3(DATA_TYPE **D, int N, int R, int base_size) {
    int ii, jj, kk;
    int i, j, k;
    int tile_size = N/R;

// #pragma scop
    // cout << "N: " << N << " R: " << R << " tile_size: " << tile_size << endl;
    for (kk = 0; kk < R && kk * (tile_size) < NN; kk++) {
        // cout << kk << " : before a" << endl; 
        // Applying index set Splitting
        // CASE 1: kk = ii and kk = jj --> Function A(X, ...)
        fw_rec3_A(D, tile_size, R, base_size,
                  kk * (tile_size), (kk + 1) * tile_size,
                  kk * (tile_size), (kk + 1) * tile_size,
                  kk * (tile_size), (kk + 1) * tile_size);

// following functions fw_rec3_B(...) can be in parallel with all the
// function calls fw_rec3_C(...) as they are writing to different tiles.
// IN PARALLEL:

        // cout << k << " : before b" << endl;
        // CASE 2: kk = ii but kk != jj ==> Function B(X, U, ...)
        // #pragma omp parallel
        {
            // #pragma omp for
            for (jj = 0; jj < R && jj * tile_size < NN; jj++) {
                if (jj == kk) continue;
                #pragma omp task
                fw_rec3_B(D, D, D, tile_size, R, base_size,
                          kk * (tile_size), (kk + 1) * tile_size,
                          kk * (tile_size), (kk + 1) * tile_size,
                          jj * (tile_size), (jj + 1) * tile_size);
            }
        }
        // cout << k << " : before c" << endl;
        // CASE 3: kk = jj but kk != ii ==> Function C(X, V, ...)

        // #pragma omp parallel
        {
            // #pragma omp for
            for (ii = 0; ii < R && ii * tile_size < NN; ii++) {
                if (ii == kk) continue;
                #pragma omp task
                fw_rec3_C(D, D, D, tile_size, R, base_size,
                          kk * (tile_size), (kk + 1) * tile_size,
                          ii * (tile_size), (ii + 1) * tile_size,
                          kk * (tile_size), (kk + 1) * tile_size);
            }
        }
        #pragma omp taskwait   
// JOIN - SYNC

        // cout << k << " : before d" << endl;
// All the following fw_rec3_D(...) functions can run in parallel.
// IN PARALLEL:
        // CASE 4: kk != ii and kk != jj ==> Function D(X, U, V, ...)
        // #pragma omp parallel
        {
            // #pragma omp for collapse(2)
            for (ii = 0; ii < R && ii * tile_size < NN; ii++) {
                // #pragma omp for
                for (jj = 0; jj < R && jj * tile_size < NN; jj++) {
                    if (ii == kk || jj == kk) continue;
                    #pragma omp task
                    fw_rec3_D(D, D, D, D, tile_size, R, base_size,
                           kk * (tile_size), (kk + 1) * tile_size,
                           ii * (tile_size), (ii + 1) * tile_size,
                           jj * (tile_size), (jj + 1) * tile_size);       
                }
            }
        }
        #pragma omp taskwait
// JOIN - SYNC
    }
// #pragma endscop
    // printf("Total number of updates are: %d\n", counter);
}

*/

void print_arr(DATA_TYPE **arr, int N){
    int i, j;
    printf("---------ARR----------\n");
    for (i=0; i < N; i++){
        for (j = 0; j < N; j++)
            printf("%d\t", arr[i][j]);
        printf("\n");
    }
}

/*
int main(int argc, char **argv) {
    int i, j;
    NN = 1024;
    if (argc > 1){
        NN = atoi(argv[1]);
    }
    int N = 2;
    while (N < NN)
        N = (N << 1);

    int B = 32;
    if (argc > 2)
        B = atoi(argv[2]);
    int base_size = B;

    int R = 2;
    if (argc > 3)
        R = atoi(argv[3]);

    // making sure virtual padding will give the desired base case sizes
    // only for power of 2 base case sizes
    // otherwise it should be commented
    int RR = 1;
    while (N / RR > B)
        RR *= R;
    N = RR * B;
    // End of extra virtual padding for base case

#ifdef USE_PAPI
    papi_init();
#endif

    if (argc > 4) {
        
        omp_set_num_threads(atoi(argv[4]));

        // if (0 != __cilkrts_set_param("nworkers", argv[3])) {
        //     printf("Failed to set worker count\n");
        //     return 1;
        // }
       
    }   
    // int P = __cilkrts_get_nworkers();
    // printf("%d,", __cilkrts_get_nworkers());  

    DATA_TYPE **D_serial = (DATA_TYPE **)malloc(NN * sizeof(DATA_TYPE *));
    DATA_TYPE **D_recursive3 = (DATA_TYPE **)malloc(NN * sizeof(DATA_TYPE *));
    for (i = 0; i < NN; ++i) {
        D_serial[i] = (DATA_TYPE *)malloc(NN * sizeof(DATA_TYPE));
        D_recursive3[i] = (DATA_TYPE *)malloc(NN * sizeof(DATA_TYPE));

        for (j = 0; j < NN; ++j) {
            // D_serial[i][j] = rand() % 100 + 1; 
            D_serial[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / N;
            D_recursive3[i][j] = D_serial[i][j];
        }
    }

    // printf("STEP 1:\n");
    
    // print_arr(D_serial, NN);

#ifdef DEBUG
    unsigned long long tstart_serial = time(NULL);
    fw(D_serial, NN);
    unsigned long long tend_serial = time(NULL);
    // // // // // // // // // cout << "serial: " << tend_serial - tstart_serial << endl;
#endif

    // printf("%d,", base_size);
    unsigned long long tstart = time(NULL);


    // printf("\n\nSTEP 5:\n");
    // printf("\nNOW HAVING MULTIPLE FUNCTIONS AS A RESULT OF INDEX SET SPLITTING\n\n");

#ifdef POLYBENCH
    // Start timer. 
    polybench_start_instruments;
#endif

    // print_arr(D_recursive3, NN);

    int P = 0;
    #pragma omp parallel
    {
        // P = omp_num_procs();
        P = omp_get_max_threads();
        #pragma omp single
        {
            #pragma omp task
            fw_rec_top_level3(D_recursive3, N, R, base_size);   
            // fw_rec3_A(D_recursive3, N, R, base_size, 0, N, 0, N, 0, N);    
        }
    }

#ifdef POLYBENCH
    // Stop and print timer. 
    polybench_stop_instruments;
    polybench_print_instruments;
#endif

    unsigned long long tend = time(NULL);
    // printf("%d,%f,", N, cilk_ticks_to_seconds(tend - tstart));
    // cout << R << "," << N << "," << B << "," 
    //  << P << "," << (tend - tstart);

    // printf("%d, %d, %d, %d, %lld\n", R, N, B, P, (tend - tstart));

#ifdef USE_PAPI        
    countTotalMiss(p);
    PAPI_shutdown();
    delete threadcounter;
    for (int i = 0; i < p; i++) delete l2miss[i];
    delete l2miss;
    delete errstring;
    delete EventSet;
    delete eventCode;
#endif    

    // print_arr(D_serial, NN);
    // print_arr(D_recursive3, NN);


    for (i = 0; i < NN; ++i) {
        for (j = 0; j < NN; ++j) {
#ifdef DEBUG
            if (D_serial[i][j] != D_recursive3[i][j]) {
                printf("WE HAVE ISSUE IN THE RECURSIVE PROGRAM 3\n");
            }
#endif
        }
        free(D_serial[i]);
        free(D_recursive3[i]);

    }
    free(D_serial);
    free(D_recursive3);
    // printf("\n");

    return 0;
}
*/

