#pragma once

/**	
 * Interface of DQMC code with CUBLAS and CUSOLVER library code.
 * Define away d
*/
#include "util.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef USE_CPLX
	#define cast(p) (cuDoubleComplex *)(p)
	#define ccast(p) (const cuDoubleComplex *)(p)
#else
	#define cast(p) (p)
	#define ccast(p) (p)
#endif

// // general matrix-matrix multiplication: Level 3 BLAS
// static inline cublasStatus_t xgemm(
// 		cublasHandle_t handle, 
// 		cublasOperation_t transa, 
// 		cublasOperation_t transb,
// 		const int m, const int n, const int k,
// 		const num *alpha, /* host or device pointer */
// 		const num *a, 
// 		const int lda,
// 		const num *b, 
// 		const int ldb,
// 		const num *beta, /* host or device pointer */
// 		num *c, 
// 		const int ldc)
// {
// #ifdef USE_CPLX
// 	cublasZgemm(
// #else
// 	cublasDgemm(
// #endif
// 	handle, transa, transb, m, n, k,
// 	alpha, a, lda, b, ldb, beta, c, ldc);
// }

// general matrix-vector product, Level 2 BLAS
// y <- alpha * A * x + beta * y when trans = 'N' or 'n'
static inline cublasStatus_t xgemv(cublasHandle_t handle,
		const cublasOperation_t trans, 
		const int m, const int n,
		const num *alpha, /* host or device pointer */ 
		const num *A, /* device */ 
		const int lda,
		const num *x, /* device */
		const int incx, 
		const num *beta, /* host or device pointer */
		num *y,       /* device */
		const int incy)
{
#ifdef USE_CPLX
	cublasZgemv(
#else
	cublasDgemv(
#endif
	handle, trans, m, n,
	alpha, A, lda, x, incx,
	beta, y, incy);
}

// // triangular matrix - general matrix product, level 3 BLAS
// static inline void xtrmm(cublasHandle_t handle, 
// 		cublasSideMode_t side, cublasFillMode_t uplo, 
// 		const char *transa, const char *diag,
// 		const int m, const int n,
// 		const num alpha, const num *a, const int lda,
// 		num *b, const int ldb)
// {
// #ifdef USE_CPLX
// 	ztrmm(
// #else
// 	dtrmm(
// #endif
// 	handle, side, uplo, transa, diag, &m, &n,
// 	ccast(&alpha), ccast(a), &lda, cast(b), &ldb);
// }

// // LAPACK
// // Compute LU factorization of general matrix using partial pivoting with row interchanges
// static inline void xgetrf(const int m, const int n, num* a,
// 		const int lda, int* ipiv, int* info)
// {
// #ifdef USE_CPLX
// 	zgetrf(
// #else
// 	dgetrf(
// #endif
// 	&m, &n, cast(a), &lda, ipiv, info);
// }

// // LAPACK
// // Compute inverse of general matrix using LU factorization provided by xgetrf
// static inline void xgetri(const int n, num* a, const int lda, const int* ipiv,
// 		num* work, const int lwork, int* info)
// {
// #ifdef USE_CPLX
// 	zgetri(
// #else
// 	dgetri(
// #endif
// 	&n, cast(a), &lda, ipiv, cast(work), &lwork, info);
// }

// // LAPACK
// // Solve linear equation with general matrix using LU factorization provided by zgetrf
// static inline void xgetrs(const char* trans, const int n, const int nrhs,
// 		const num* a, const int lda, const int* ipiv,
// 		num* b, const int ldb, int* info)
// {
// #ifdef USE_CPLX
// 	zgetrs(
// #else
// 	dgetrs(
// #endif
// 	trans, &n, &nrhs, ccast(a), &lda, ipiv, cast(b), &ldb, info);
// }

// // LAPACK
// // QR factorization of general matrix using column pivoting
// static inline void xgeqp3(const int m, const int n, num* a, const int lda, int* jpvt, num* tau,
// 		num* work, const int lwork, double* rwork, int* info)
// {
// #ifdef USE_CPLX
// 	zgeqp3(&m, &n, cast(a), &lda, jpvt, cast(tau),
// 	cast(work), &lwork, rwork, info);
// #else
// 	dgeqp3(&m, &n, cast(a), &lda, jpvt, cast(tau),
// 	cast(work), &lwork, info); // rwork not used
// #endif
// }


// // LAPACK
// // QR factorization of general matrix without pivoting
// static inline void xgeqrf(const int m, const int n, num* a, const int lda, num* tau,
// 		num* work, const int lwork, int* info)
// {
// #ifdef USE_CPLX
// 	zgeqrf(
// #else
// 	dgeqrf(
// #endif
// 	&m, &n, cast(a), &lda, cast(tau), cast(work), &lwork, info);
// }

// // LAPACK
// // Multiplies a real/complex matrix by the orthogonal/unitary matrix Q 
// // of the QR factorization formed by xgeqrf or xgeqp3
// static inline void xunmqr(const char* side, const char* trans,
// 		const int m, const int n, const int k, const num* a,
// 		const int lda, const num* tau, num* c,
// 		const int ldc, num* work, const int lwork, int* info)
// {
// #ifdef USE_CPLX
// 	zunmqr(side, trans,
// #else
// 	dormqr(side, trans[0] == 'C' ? "T" : trans,
// #endif
// 	&m, &n, &k, ccast(a), &lda, ccast(tau),
// 	cast(c), &ldc, cast(work), &lwork, info);
// }

// // LAPACK
// // Compute inverse of triangular matrix
// static inline void xtrtri(const char* uplo, const char* diag, const int n,
// 		num* a, const int lda, int* info)
// {
// #ifdef USE_CPLX
// 	ztrtri(
// #else
// 	dtrtri(
// #endif
// 	uplo, diag, &n, cast(a), &lda, info);
// }

#undef ccast
#undef cast
