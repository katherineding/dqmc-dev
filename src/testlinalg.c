#include "linalg.h"
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

// #define m 2
// #define n 2

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

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const data_type A[m*k] = {1,2,3,4};
    const data_type B[k*n] = {5,6,7,8};
    const data_type x[n] = {5,6};

    const data_type alpha = 1.0;
    const data_type beta = 0.0;
    data_type C[m*n] = {0};
    data_type y[m] = {0};
    const int incx =1;
    const int incy =1;

    data_type *d_A = NULL;
    data_type *d_x = NULL;
    data_type *d_y = NULL;
    data_type *d_B = NULL;
    data_type *d_C = NULL;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("size of A: %d\n",sizeof(A));
    printf("size of B: %d\n",sizeof(B));
    printf("size of alpha: %d\n",sizeof(alpha));

    /* step 1: create cublas handle, bind a stream */
    // What is the point of a stream?
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));


    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(A)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(B)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(C)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(x)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(y)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, &A, sizeof(A), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, &B, sizeof(B), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, &x, sizeof(x), cudaMemcpyHostToDevice, stream));

    /* step 3: compute */
    CUBLAS_CHECK(
        xgemv(cublasH, transa, m, n, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy));
    CUBLAS_CHECK(
        xgemm(cublasH, transa, transb, m, n, k, 
            &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(&y, d_y, sizeof(y), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&C, d_C, sizeof(C), cudaMemcpyDeviceToHost, stream));


    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    /* check results */

    printf("y result:\n");

    for (int i = 0;i < m; i++){
        printf("y[%d] = %f\n",i,y[i]);
    }

    printf("C result:\n"); //

    for (int i = 0;i < m; i++){
        for (int j = 0;j < n;j++) {
            printf("C[%d][%d] = %f\n",i,j,C[i + j*m]);
        }
    }

}
