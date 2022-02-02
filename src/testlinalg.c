//#include "linalg.h"
#include <stdio.h>
#include <stdexcept>


#include <cublas_v2.h>
#include <cuda_runtime.h>

// y <- alpha * A * x + beta * y when trans = 'N' or 'n'
// xgemv(cublasHandle_t handle,
// 		const cublasOperation_t trans, 
// 		const int m, const int n,
// 		const num *alpha, /* host or device pointer */ 
// 		const num *A, 
// 		const int lda,
// 		const num *x, 
// 		const int incx,
// 		const num *beta, /* host or device pointer */
// 		num *y, 
// 		const int incy);

#define m 2
#define n 2

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)




typedef double data_type;





int main(){
    printf("hello world\n");

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    // const int m = 2;
    // const int n = 2;
    const int lda = m;

    const double A[m*n] = {1,2,3,4};
    const double x[n] = {5,6};
    const double alpha = 1.0;
    const double beta = 0.0;
    double y[m] = {0,0};
    const int incx =1;
    const int incy =1;

    data_type *d_A = NULL;
    data_type *d_x = NULL;
    data_type *d_y = NULL;

    cublasOperation_t transa = CUBLAS_OP_N;
    

    printf("size of A: %d\n",sizeof(A));
    printf("size of alpha: %d\n",sizeof(alpha));

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

}
