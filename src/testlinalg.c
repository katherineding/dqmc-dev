#include <stdio.h>
#include <stdexcept>


#include <cuda_runtime.h>

#include "linalg.h"
//#include "util.h"

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
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


int main(){

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("This device supports CUDA compute version %d\n", prop.major * 10 + prop.minor);


    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    //what is a cudaStream_t under the hood?
    cudaStream_t stream = NULL;
    cudaStream_t stream2 = NULL;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    const cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
    const cublasOperation_t transa = CUBLAS_OP_N;
    const cublasOperation_t transb = CUBLAS_OP_N;

    const num A[m*k] = {1,3,2,4};
    const num B[k*n] = {5,7,6,8};
    const num x[n] = {5,6};

    const num alpha = 1.0;
    const num beta = 0.0;
    num C[m*n] = {0};
    num D[m*n] = {0};
    num y[m] = {0};

    const int incx =1;
    const int incy =1;

    num *d_A = NULL;
    num *d_x = NULL;
    num *d_y = NULL;
    num *d_B = NULL;
    num *d_C = NULL;
    num *d_D = NULL;

    //host side initialization
    num AA[3*3] = {1,4,7,2,5,8,3,6,10};
    num BB[3] = {1,2,3};
    int ipiv[3] = {0};
    int info = 0;
    int lwork = 0;

    //device pointers
    num *d_AA = NULL;
    num *d_BB = NULL;
    int *d_ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    num *d_work = NULL; /* device workspace for getrf */

    printf("size of A: %d\n",sizeof(A));
    printf("size of B: %d\n",sizeof(B));
    printf("size of alpha: %d\n",sizeof(alpha));
    printf("size of AA: %d\n",sizeof(AA));
    printf("size of BB: %d\n",sizeof(BB));


    /* step 1: create cublas handle, bind a stream */
    // What is the point of a stream?
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream2));


    /* step 2: copy data to device */
    /* Allocate memory for result arrays,*/
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(A)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(B)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(C)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D), sizeof(D)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(x)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(y)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, &A, sizeof(A), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, &B, sizeof(B), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, &x, sizeof(x), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_AA), sizeof(AA)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_BB), sizeof(BB)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ipiv), sizeof(ipiv)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(info)));
    CUDA_CHECK(cudaMemcpyAsync(d_AA, &AA, sizeof(AA), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_BB, &BB, sizeof(BB), cudaMemcpyHostToDevice, stream));

    /* step 2.5: query working space of getrf in lwork
    Note lwork is host side variable and its address is passed to bufferSize*/
    CUSOLVER_CHECK(xgetrf_bs(cusolverH, 3, 3, d_AA, 3, &lwork));
    printf("lwork = %d\n", lwork); //lwork is a pretty large multiple of n

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(num) * lwork));


    /* step 3: compute */
    CUBLAS_CHECK(
        xgemv(cublasH, transa, m, n, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy));
    CUBLAS_CHECK(
        xgemm(cublasH, transa, transb, m, n, k, 
            &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
    CUBLAS_CHECK(
        xtrmm(cublasH, side, uplo, transa, diag, m, n,  
            &alpha, d_A, lda, d_B, ldb, d_D, ldc));
    CUSOLVER_CHECK(
        xgetrf(cusolverH, 3, 3, d_AA, 3, d_work, d_ipiv, d_info));
    // CUBLAS_CHECK(
    //     xgetrf(handle, m, n, a, lda, work, ipiv, info);


    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(&y, d_y, sizeof(y), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&C, d_C, sizeof(C), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&D, d_D, sizeof(D), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&AA, d_AA, sizeof(AA), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&ipiv, d_ipiv, sizeof(ipiv), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(info), cudaMemcpyDeviceToHost, stream));


    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* check results */

    int major=-1,minor=-1,patch=-1;
    cusolverGetProperty(MAJOR_VERSION, &major);
    cusolverGetProperty(MINOR_VERSION, &minor);
    cusolverGetProperty(PATCH_LEVEL, &patch);
    printf("CUSOLVER Version (Major,Minor,PatchLevel): %d.%d.%d\n",
    major,minor,patch);

    major=-1,minor=-1,patch=-1;
    cublasGetProperty(MAJOR_VERSION, &major);
    cublasGetProperty(MINOR_VERSION, &minor);
    cublasGetProperty(PATCH_LEVEL, &patch);
    printf("CUBLAS Version (Major,Minor,PatchLevel): %d.%d.%d\n",
    major,minor,patch);

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

    printf("D result:\n"); //

    for (int i = 0;i < m; i++){
        for (int j = 0;j < n;j++) {
            printf("D[%d][%d] = %f\n",i,j,D[i + j*m]);
        }
    }

    printf("info = %d\n",info);

    printf("LU result:\n"); //

    for (int i = 0;i < 3; i++){
        for (int j = 0;j < 3;j++) {
            printf("LU[%d][%d] = %f\n",i,j,AA[i + j*3]);
        }
    }

    /* step n: solve using LU decomposition*/
    CUSOLVER_CHECK(xgetrs(cusolverH, CUBLAS_OP_N, 3, 1, /* nrhs */
                                        d_AA, 3, d_ipiv, d_BB, 3, d_info));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(info), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&BB, d_BB, sizeof(BB), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("info = %d\n",info);

    

    printf("A x = B solution x:\n");
    for (int i = 0;i < 3; i++){
        printf("x[%d] = %f\n",i,BB[i]);
    }

    /* free resources */

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

    CUDA_CHECK(cudaFree(d_AA));
    CUDA_CHECK(cudaFree(d_BB));
    CUDA_CHECK(cudaFree(d_ipiv));
    CUDA_CHECK(cudaFree(d_info));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaDeviceReset());

}
