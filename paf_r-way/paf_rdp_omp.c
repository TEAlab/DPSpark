/*
Copy: 
    scp fw_rdp.cpp zafahmad@stampede2.tacc.utexas.edu:/home1/05072/zafahmad/SC_2019_submission/

Compile:
    module load papi/5.5.1
    icc -DDEBUG -O3 -fopenmp -xhost -AVX512 fw_rdp_omp.cpp -o fw_rdp -I$TACC_PAPI_INC -Wl,-rpath,$TACC_PAPI_LIB -L$TACC_PAPI_LIB -lpapi
    icc -O3 -fopenmp -xhost -AVX512 fw_rdp_omp_poly_base.c -o exec -DDEBUG -DPOLYBENCH  polybench-c-3.2/utilities/polybench.c -DPOLYBENCH_TIME -DSMALL_DATASET -Ipolybench-c-3.2/utilities/ -I. -I$TACC_PAPI_INC -Wl,-rpath,$TACC_PAPI_LIB -L$TACC_PAPI_LIB -lpapi

    icc fw_rdp_omp.c -o fw_rdp -DDEBUG -DPOLYBENCH polybench-c-4.2/utilities/polybench.c -DPOLYBENCH_TIME -DPOLYBENCH_USE_RESTRICT -Ipolybench-c-4.2/utilities/ -I. -O2 -qopenmp -xKNL -qopt-prefetch=5 -xhost -AVX512 -lm 

    icc paf_rdp_omp.c -o paf_rdp -DPOLYBENCH polybench-c-4.2/utilities/polybench.c -DPOLYBENCH_TIME -DPOLYBENCH_USE_RESTRICT -Ipolybench-c-4.2/utilities/ -I. -O2 -qopenmp -xKNL -qopt-prefetch=5 -xhost -AVX512 -lm 
    
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
#include "paf_rdp_omp.h"

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


// #define CACHE_OPTIMIZED
// #define CACHE_OPTIMIZEDX
/*
    STEP 1) The input algorithm is the triply-nested for loop version passed to 
           PoCC to get parametric tiled version of the code
*/

int NN_orig; // original size of the matrix
int N_TOTAL, II, JJ, KK;

int paf(int **S, int **F, int N) {
    int i, j, k;
    for (k = N; k >= 0; --k) {
        for (j = N; j >= 0; --j) {
            for (i = N; i >= 0; --i) {
                if (k < N && k >= 3 && j <= min(k-2, N-3) &&
                    j >= 1 && i <= min(j-1,N-4)) {
                    S[i][j] = max(S[i][j], S[j+1][k] + F[j+1][min(k, 2*j-i+1)]);
                }
            }
        }
    }

    int result = 0;
    for (j = 0; j < N; ++j) {
        result = max(result, S[0][j]);
    }
    return result;  
}

void paf_rec3_B(DATA_TYPE *X, DATA_TYPE *U, int NN, int N, int R, int base_size,
                 int k_ub, int k_lb, int j_ub, int j_lb, int i_ub, int i_lb);

void paf_rec3_A(DATA_TYPE *X, int NN, int N, int R, int base_size,
                 int k_ub, int k_lb, int j_ub, int j_lb, int i_ub, int i_lb);

double f_matrix(double i, double j){
    return i+j;
}

// (x_block, uv_block, n, I_, J_, k)
void paf_B(DATA_TYPE *X, DATA_TYPE *U, int n_block, int N_total, int block_I, int block_J, int block_kk){
    NN_orig = n_block;
    N_TOTAL = N_total;
    int R = RWAY;
    int base_size = BASESIZE;

    II = block_I, JJ = block_J, KK = block_kk;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            paf_rec3_B(X, U, n_block, n_block, R, base_size, n_block, 0, n_block, 0, n_block, 0);
        }
    }
}

// (x_block, n, N, I_, J_, k)
void paf_A(DATA_TYPE *X, int n_block, int N_total, int block_I, int block_J, int block_kk){
    NN_orig = n_block;
    N_TOTAL = N_total;
    int R = RWAY;
    int base_size = BASESIZE;

    II = block_I, JJ = block_J, KK = block_kk;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            // printf("Number of threads: %d -----------------------------------------------------------", omp_get_num_threads());
            paf_rec3_A(X, n_block, n_block, R, base_size, n_block, 0, n_block, 0, n_block, 0);
        }
    }
}


// in function B, X != U --> The most parallel one
void paf_rec3_B(DATA_TYPE *X, DATA_TYPE *U, int NN, int N, int R, int base_size,
                 int k_ub, int k_lb, int j_ub, int j_lb, int i_ub, int i_lb) {


    if (k_lb >= NN || i_lb >= NN || j_lb >= NN)
        return ;

    // printf("NN: %d N: %d R: %d base: %d klb: %d kub: %d ilb: %d iub: %d jlb: %d jub: %d\n", NN, N, R, base_size, k_lb, k_ub,
    //                 i_lb, i_ub, j_lb, j_ub);

    int i, j, k;
    // base case

    if ((k_ub - k_lb) <= base_size || N <= R) {
        DATA_TYPE x_col_major[base_size * base_size];
#ifdef USE_PAPI
    int id = tid();
    papi_for_thread(id);
    int retval = 0;
    if ( (retval=PAPI_start(EventSet[id])) != PAPI_OK)
        ERROR_RETURN(retval);
#endif         
        for (i = i_lb; i < i_ub && i < NN_orig; i++)
            for (j = j_lb; j < j_ub && j < NN_orig; j++)
                x_col_major[(j-j_lb)*base_size + (i-i_lb)] = X[i*NN_orig + j];

        for (k = k_ub - 1; k >= k_lb; --k) {
            // if (k >= NN_orig || k < 3) continue;
            int K = KK*NN_orig + k;

            for (j = j_ub - 1; j >= j_lb; --j) {
                // if (j >= NN_orig || j > min(k-2, NN-3)) continue;
                DATA_TYPE u_jk = U[(j+1)*NN_orig + k];
                int J = JJ*NN_orig + j;

                for (i = i_ub - 1; i >= i_lb; --i) {
                    // if (i >= NN_orig || i > min(j-1,NN-4)) continue;
                    int I = II*NN_orig + i;
                    int min1 = min(K-2, N_TOTAL-3);
                    int min2 = min(J-1, N_TOTAL-4);
                    if ((K < N_TOTAL) && (K >= 3) && (J <= min1) && (J >= I+1) && (I <= min2)){
                    // if (k < NN && k >= 3 && j <= min(k-2, NN-3) && 
                    //     j >= 1 && i <= min(j-1,NN-4)) {
// J+1, min(K, 2*J-I+1)
                        x_col_major[(j-j_lb)*base_size + (i-i_lb)] = max(x_col_major[(j-j_lb)*base_size + (i-i_lb)], u_jk +  f_matrix(J+1, min(K, 2*J-I+1)));
                        // X[i][j] = max(X[i][j], u_jk + F[j+1][min(k, 2*j-i+1)]);
                    }
                }
            }
        }

        for (i = i_lb; i < i_ub && i < NN_orig; i++)
            for (j = j_lb; j < j_ub && j < NN_orig; j++)    
                X[i*NN_orig + j] = x_col_major[(j-j_lb)*base_size + (i-i_lb)];
#ifdef USE_PAPI
    countMisses(id );
#endif   
        return;
    }
    int ii, jj, kk;
    int tile_size = N/R;
    for (kk = R-1; kk >= 0; --kk) {
        if (k_lb + kk * tile_size >= NN_orig) continue;
//         // Applying index set Splitting
//         // Output/write tile S[ii,jj] is disjoint from input/read tile U[jj,kk]
// // IN PARALLEL:     
//         for (jj = kk; jj >= 0; --jj) {
//             for (ii = jj; ii >= 0; --ii) {
//                 if (i_lb + ii * tile_size >= NN_orig || j_lb + jj * tile_size >= NN_orig) continue;
//                 #pragma omp task
//                 paf_rec3_B(X, U, F, NN, tile_size, R, base_size,
//                          k_lb + (((kk+1)*tile_size) - 1), k_lb + (kk*tile_size),
//                          j_lb + (((jj+1)*tile_size) - 1), j_lb + (jj*tile_size),
//                          i_lb + (((ii+1)*tile_size) - 1), i_lb + (ii*tile_size));
//             }
//         }
        for (jj = R-1; jj >= 0; --jj) { // kk
            for (ii = R-1; ii >= 0; --ii) { // jj
                if (i_lb + ii * tile_size >= NN_orig || j_lb + jj * tile_size >= NN_orig) continue;
                #pragma omp task
                paf_rec3_B(X, U, NN, tile_size, R, base_size,
                         k_lb + ((kk+1)*tile_size), k_lb + (kk*tile_size),
                         j_lb + ((jj+1)*tile_size), j_lb + (jj*tile_size),
                         i_lb + ((ii+1)*tile_size), i_lb + (ii*tile_size));
            }
        }
        #pragma omp taskwait
// JOIN - SYNC      
    }   
}

// in function A, X = U --> The least parallel one
void paf_rec3_A(DATA_TYPE *X, int NN, int N, int R, int base_size,
                 int k_ub, int k_lb, int j_ub, int j_lb, int i_ub, int i_lb) {
    

    if (k_lb >= NN || i_lb >= NN || j_lb >= NN)
        return ;

    int i, j, k;
    // base case
    if ((k_ub - k_lb) <= base_size || N <= R) {
#ifndef CACHE_OPTIMIZEDX
#ifdef USE_PAPI
    int id = tid();
    papi_for_thread(id);
    int retval = 0;
    if ( (retval=PAPI_start(EventSet[id])) != PAPI_OK)
        ERROR_RETURN(retval);
#endif          
        // printf(">>>>>>>>>>>>>>>>> NN: %d N: %d KU: %d KL: %d JU: %d JL: %d IU: %d IL: %d\n", NN, N, k_ub, k_lb, j_ub, j_lb, i_ub, i_lb);
        for (k = k_ub - 1; k >= k_lb; --k) {
            // if (k >= NN_orig || k < 3) continue;
            int K = KK*NN_orig + k;
            for (j = j_ub - 1; j >= j_lb; --j) {
                // if (j >= NN_orig || j > min(k-2, NN-3)) continue;
                int J = JJ*NN_orig + j;
                DATA_TYPE x_jk = X[(j+1)*NN_orig + k];
                for (i = i_ub - 1; i >= i_lb; --i) {
                    // if (i >= NN_orig || i > min(j-1,NN-4)) continue;
                    int I = II*NN_orig + i;
                    int min1 = min(K-2, N_TOTAL-3);
                    int min2 = min(J-1, N_TOTAL-4);
                    // printf(">>>>>>>>>>>>>>>>> I: %d J: %d K: %d min1: %d min2: %d\n", I, J, K, min1, min2);
                    if ((K < N_TOTAL) && (K >= 3) && (J <= min1) && (J >= I+1) && (I <= min2)){
                    // if (k < NN && k >= 3 && j <= min(k-2, NN-3) && 
                    //     j >= 1 && i <= min(j-1,NN-4)) {
                        X[i*NN_orig + j] = max(X[i*NN_orig + j], x_jk + f_matrix(J+1, min(K, 2*J-I+1)));
                        // printf("---------------------------------------> HERE I AM\n");
                    }
                }
            }
        }
#else
    
#endif
#ifdef USE_PAPI
    countMisses(id );
#endif 
        return;
    }
    int ii, jj, kk;
    int tile_size = N/R;
    for (kk = R-1; kk >= 0; --kk) {
        if (k_lb + kk * tile_size >= NN_orig) continue;
        // // Applying index set Splitting
        // // CASE 1: ii == jj && jj == kk --> Function A(X, ...)
        // paf_rec3_A(X, F, N, tile_size, R, base_size,
        //          k_lb + (((kk+1)*tile_size) - 1), k_lb + (kk*tile_size),
        //          j_lb + (((kk+1)*tile_size) - 1), j_lb + (kk*tile_size),
        //          i_lb + (((kk+1)*tile_size) - 1), i_lb + (kk*tile_size));
        paf_rec3_A(X, N, tile_size, R, base_size,
                 k_lb + ((kk+1)*tile_size), k_lb + (kk*tile_size),
                 j_lb + ((kk+1)*tile_size), j_lb + (kk*tile_size),
                 i_lb + ((kk+1)*tile_size), i_lb + (kk*tile_size));

        // CASE 2: if only we have jj == kk --> Function B(X, U, ...)
// IN PARALLEL:
        // for (ii = kk - 1; ii >= 0; --ii) {
        //     if (i_lb + ii * tile_size >= NN_orig) continue;

        //     #pragma omp task
        //     paf_rec3_B(X, X, F, N, tile_size, R, base_size,
        //              k_lb + (((kk+1)*tile_size) - 1), k_lb + (kk*tile_size),
        //              j_lb + (((kk+1)*tile_size) - 1), j_lb + (kk*tile_size),
        //              i_lb + (((ii+1)*tile_size) - 1), i_lb + (ii*tile_size));
        // }
        for (ii = R - 1; ii >= 0; --ii) { // kk - 1
            if (i_lb + ii * tile_size >= NN_orig) continue;
            #pragma omp task
            paf_rec3_B(X, X, N, tile_size, R, base_size,
                     k_lb + ((kk+1)*tile_size), k_lb + (kk*tile_size),
                     j_lb + ((kk+1)*tile_size), j_lb + (kk*tile_size),
                     i_lb + ((ii+1)*tile_size), i_lb + (ii*tile_size));
        }
        #pragma omp taskwait
// JOIN - SYNC
// IN PARALLEL:
        // case 3: if none of them are equal. I.e., ii != jj && jj != kk --> Functions C(X, U, ...) and D(X, U, ...)
        // for (jj = kk - 1; jj >= 0; --jj) {
        //     for (ii = jj; ii >= 0; --ii) {
        //         if (i_lb + ii * tile_size >= NN_orig || j_lb + jj * tile_size >= NN_orig) continue;
        //         #pragma omp task
        //         paf_rec3_B(X, X, F, N, tile_size, R, base_size,
        //                  k_lb + (((kk+1)*tile_size) - 1), k_lb + (kk*tile_size),
        //                  j_lb + (((jj+1)*tile_size) - 1), j_lb + (jj*tile_size),
        //                  i_lb + (((ii+1)*tile_size) - 1), i_lb + (ii*tile_size));
        //     }
        // }
        for (jj = kk - 1; jj >= 0; --jj) {
            for (ii = jj; ii >= 0; --ii) {
                if (i_lb + ii * tile_size >= NN_orig || j_lb + jj * tile_size >= NN_orig) continue;
                #pragma omp task
                paf_rec3_B(X, X, N, tile_size, R, base_size,
                         k_lb + ((kk+1)*tile_size), k_lb + (kk*tile_size),
                         j_lb + ((jj+1)*tile_size), j_lb + (jj*tile_size),
                         i_lb + ((ii+1)*tile_size), i_lb + (ii*tile_size));
            }
        }
        #pragma omp taskwait
// JOIN - SYNC

    }
}


int paf_rec_top_level3(DATA_TYPE *S, DATA_TYPE *F, int N, int R, int base_size) {
    int ii, jj, kk;
    int i, j, k;
    int tile_size = N/R;
    // printf("tile_size: %d, N: %d, R: %d\n", tile_size, N, R);
    for (kk = R-1; kk >= 0; --kk) {
        // printf("Here 2: %d\n", kk);
        // Applying index set Splitting
        // CASE 1: ii == jj && jj == kk --> Function A(X, ...)
        if (kk * tile_size >= NN_orig)
            continue;
        // paf_rec3_A(S, F, N, tile_size, R, base_size,
        //          (((kk+1)*tile_size) - 1), (kk*tile_size),
        //          (((kk+1)*tile_size) - 1), (kk*tile_size),
        //          (((kk+1)*tile_size) - 1), (kk*tile_size));

        paf_rec3_A(S, N, tile_size, R, base_size,
                 ((kk+1)*tile_size), (kk*tile_size),
                 ((kk+1)*tile_size), (kk*tile_size),
                 ((kk+1)*tile_size), (kk*tile_size));

        // CASE 2: if only we have jj == kk --> Function B(X, U, ...)
// IN PARALLEL:
        // for (ii = kk - 1; ii >= 0; --ii) {
        //     if (ii * tile_size >= NN_orig) continue;

        //     #pragma omp task
        //     paf_rec3_B(S, S, F, N, tile_size, R, base_size,
        //              (((kk+1)*tile_size) - 1), (kk*tile_size),
        //              (((kk+1)*tile_size) - 1), (kk*tile_size),
        //              (((ii+1)*tile_size) - 1), (ii*tile_size));
        // }
        for (ii = R-1; ii >= 0; --ii) { // kk - 1
            if (ii * tile_size >= NN_orig) continue;
            #pragma omp task
            paf_rec3_B(S, S, N, tile_size, R, base_size,
                     ((kk+1)*tile_size), (kk*tile_size),
                     ((kk+1)*tile_size), (kk*tile_size),
                     ((ii+1)*tile_size), (ii*tile_size));
        }
// JOIN - SYNC
        #pragma omp taskwait

        // case 3: if none of them are equal. I.e., ii != jj && jj != kk --> Functions C(X, U, ...) and D(X, U, ...)
// IN PARALLEL:
        // for (jj = kk - 1; jj >= 0; --jj) {
        //     for (ii = jj ; ii >= 0; --ii) {
        //         if (jj * tile_size >= NN_orig || ii *tile_size >= NN_orig) continue;
        //         #pragma omp task
        //         paf_rec3_B(S, S, F, N, tile_size, R, base_size,
        //                  (((kk+1)*tile_size) - 1), (kk*tile_size),
        //                  (((jj+1)*tile_size) - 1), (jj*tile_size),
        //                  (((ii+1)*tile_size) - 1), (ii*tile_size));
        //     }
        // }
        for (jj = kk - 1; jj >= 0; --jj) {
            for (ii = jj ; ii >= 0; --ii) {
                if (jj * tile_size >= NN_orig || ii *tile_size >= NN_orig) continue;
                #pragma omp task
                paf_rec3_B(S, S, N, tile_size, R, base_size,
                         ((kk+1)*tile_size), (kk*tile_size),
                         ((jj+1)*tile_size), (jj*tile_size),
                         ((ii+1)*tile_size), (ii*tile_size));
            }
        }
        #pragma omp taskwait
// JOIN - SYNC
    }

    int result = 0;
    for (j = 0; j < N; ++j) {
        result = max(result, S[0*NN_orig + j]);
    }
    return result;
}


void print_arr(DATA_TYPE **arr, int N){
    int i, j;
    printf("---------ARR----------\n");
    for (i=0; i < N; i++){
        for (j = 0; j < N; j++)
            printf("%d\t", arr[i][j]);
        printf("\n");
    }
}

void make_protein(char *protein_seq, int N);
void init_F(DATA_TYPE **F, char *protein_seq, int N);

// int main(int argc, char *argv) {
//     int i, j;
//     NN_orig = 1024;
//     if (argc > 1){
//         NN_orig = atoi(argv[1]);
//     }
//     int N = 2;
//     while (N < NN_orig)
//         N = (N << 1);

//     int B = 32;
//     if (argc > 2)
//         B = atoi(argv[2]);
//     int base_size = B;

//     int R = 2;
//     if (argc > 3)
//         R = atoi(argv[3]);

//     // making sure virtual padding will give the desired base case sizes
//     // only for power of 2 base case sizes
//     // otherwise it should be commented
//     int RR = 1;
//     while (N / RR > B)
//         RR *= R;
//     N = RR * B;
//     // End of extra virtual padding for base case    

// #ifdef USE_PAPI
//     papi_init();
// #endif

//     if (argc > 4) {
        
//         omp_set_num_threads(atoi(argv[4]));
//         /*
//         if (0 != __cilkrts_set_param("nworkers", argv[3])) {
//             printf("Failed to set worker count\n");
//             return 1;
//         }*/
       
//     }   
//     // int P = __cilkrts_get_nworkers();
//     // printf("%d,", __cilkrts_get_nworkers());  

//     int *F = (int *)malloc(NN_orig * sizeof(int *));
//     char *protein_seq = (char*)malloc(NN_orig * sizeof(char));

//     DATA_TYPE *D_serial = (DATA_TYPE *)malloc(NN_orig * sizeof(DATA_TYPE *));
//     DATA_TYPE *D_recursive3 = (DATA_TYPE *)malloc(NN_orig * sizeof(DATA_TYPE *));
//     for (i = 0; i < NN_orig; ++i) {
//         D_serial[i] = (DATA_TYPE *)malloc(NN_orig * sizeof(DATA_TYPE));
//         D_recursive3[i] = (DATA_TYPE *)malloc(NN_orig * sizeof(DATA_TYPE));

//         F[i] = (int *)malloc(N * sizeof(int));
//         for (j = 0; j < NN_orig; ++j) {
//             // D_serial[i][j] = rand() % 100 + 1; 
//             D_serial[i][j] = 0;
//             D_recursive3[i][j] = D_serial[i][j];
//         }
//     }

//     // printf("STEP 1:\n");
    
//     // print_arr(D_serial, NN);

//     // Step 1 of Initialization: making protein sequence
//     make_protein(protein_seq, NN_orig);
//     // Step 2 of Initialization: initializing F[][] array
//     init_F(F, protein_seq, NN_orig);

// #ifdef DEBUG
//     unsigned long long tstart_serial = time(NULL);
//     paf(D_serial, F, NN_orig);
//     unsigned long long tend_serial = time(NULL);
//     // // // // // // // // // cout << "serial: " << tend_serial - tstart_serial << endl;
//     printf("serial: %lld\n", tend_serial - tstart_serial);
// #endif

//     // printf("%d,", base_size);
//     unsigned long long tstart = time(NULL);


//     // printf("\n\nSTEP 5:\n");
//     // printf("\nNOW HAVING MULTIPLE FUNCTIONS AS A RESULT OF INDEX SET SPLITTING\n\n");

// #ifdef POLYBENCH
//     /* Start timer. */
//     polybench_start_instruments;
// #endif

//     // print_arr(D_recursive3, NN);

//     int P = 0;
//     #pragma omp parallel
//     {
//         // P = omp_num_procs();
//         P = omp_get_max_threads();
//         #pragma omp single
//         {
//             #pragma omp task
//             paf_rec_top_level3(D_recursive3, F, N, R, base_size);   
//             // fw_rec3_A(D_recursive3, N, R, base_size, 0, N, 0, N, 0, N);    
//         }
//     }

// #ifdef POLYBENCH
//     /* Stop and print timer. */
//     polybench_stop_instruments;
//     polybench_print_instruments;
// #endif

//     unsigned long long tend = time(NULL);
//     // printf("%d,%f,", N, cilk_ticks_to_seconds(tend - tstart));
//     // cout << R << "," << N << "," << B << "," 
//     //  << P << "," << (tend - tstart);

//     // printf("%d, %d, %d, %d, %lld\n", R, N, B, P, (tend - tstart));

// #ifdef USE_PAPI      
//     countTotalMiss(p);
//     PAPI_shutdown();
//     delete threadcounter;
//     for (int i = 0; i < p; i++) delete l2miss[i];
//     delete l2miss;
//     delete errstring;
//     delete EventSet;
//     delete eventCode;
// #endif    

//     // print_arr(D_serial, NN);
//     // print_arr(D_recursive3, NN);


//     for (i = 0; i < NN_orig; ++i) {
//         for (j = 0; j < NN_orig; ++j) {
// #ifdef DEBUG
//             if (D_serial[i][j] != D_recursive3[i][j]) {
//                 printf("WE HAVE ISSUE IN THE RECURSIVE PROGRAM 3\n");
//             }
// #endif
//         }
//         free(F[i]);
//         free(D_serial[i]);
//         free(D_recursive3[i]);

//     }
//     free(protein_seq);
//     free(D_serial);
//     free(D_recursive3);
//     // printf("\n");

//     return 0;
// }


void make_protein(char *protein_seq, int N) {
    int i;
    for (i = 0; i < N; ++i) {
        int val = rand() % 10;
        if (val < 5) {
            protein_seq[i] = '1';
        }
        else {
            protein_seq[i] = '0';
        }
    }
}

void init_F(DATA_TYPE **F, char *protein_seq, int N) {
    int j, k;
    for(j = 1; j < (N-1); ++j) {
        for (k = (j + 2); k < N; ++k) {
            if (k > (2 * j)) {
                F[j][k] = F[j][2*j];
            }
            else {
                F[j][k] = F[j][k-1];
                if (((2*j-k-1) >= 0) && 
                    (protein_seq[2*j-k-1] == protein_seq[k]) && 
                    (protein_seq[k] == '1')) {

                    ++F[j][k];
                }
            }
        }
    }
}