#pragma once

/**	
 * Interface of DQMC code with CUBLAS and CUSOLVER library code.
 * Define away d (double) and z (double complex) prefixes into x
*/
#include "util.h"

#include <cusolverDn.h>
//#include <cublas_v2.h>
#include <cuda_runtime.h>

// TODO: handle proper casting to cuDoubleComplex type
// want to use C "double complex" or C++ complex<double> on host side
// and cuDoubleComplex on device side.
// Some stackoverflow posts claim that a simple reinterpret_cast
// between these produces the desired behavior as per C++0x standard,
// but need to check this.

/*=============================================
=            BLAS routine wrappers            =
=============================================*/

// general matrix-matrix multiplication: Level 3 BLAS
// C <- alpha * A * B + beta * C when transa = transb = 'N' or 'n'
static inline cublasStatus_t xgemm(
		cublasHandle_t handle, 
		cublasOperation_t transa, 
		cublasOperation_t transb,
		const int m, const int n, const int k,
		const num *alpha, /* host or device pointer */
		const num *A, /* device */
		const int lda,
		const num *B, /* device */
		const int ldb, 
		const num *beta, /* host or device pointer */
		num *C,  /* device */
		const int ldc)
{
#ifdef USE_CPLX
	cublasZgemm(
#else
	cublasDgemm(
#endif
	handle, transa, transb, m, n, k,
	alpha, A, lda, B, ldb, beta, C, ldc);
}

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


// TODO: should cublasHandle_t be const?
// triangular matrix - general matrix product, level 3 BLAS
// Assume uplo = "U" or 'u' so A is a upper triangular matrix. The entries in 
// 	the strictly lower triangular part of A is ignored
// Assume diag = 'N' or 'n' so A does not have unit diagonal
// Assume transa = "N" or 'n'
// If side = "L" or 'l' perform C <- A*B
// If side = "R" or 'r' perform C <- B*A
// Note deviation from BLAS API, result is written to new matrix C
//  instead of input matrix B
static inline cublasStatus_t xtrmm(cublasHandle_t handle, 
		const cublasSideMode_t side, 
		const cublasFillMode_t uplo, 
		const cublasOperation_t transa, 
		const cublasDiagType_t diag,
		const int m, 
		const int n,
		const num *alpha, /* host or device */
		const num *A, /* device */
		const int lda,
		const num *B, /* device */
		const int ldb,
		num *C,       /* device */
		const int ldc)
{
#ifdef USE_CPLX
	cublasDtrmm(
#else
	cublasDtrmm(
#endif
	handle, side, uplo, transa, diag, m, n,
	alpha, A, lda, B, ldb, C, ldc);
}

/*=============================================
=           LAPACK routine wrappers           =
=============================================*/

// LAPACK
// Compute LU factorization of general matrix A using partial pivoting 
// with row interchanges
// A = P*L*U
// A is overwritten by L (unit diagonal) and U
static inline cusolverStatus_t xgetrf(cusolverDnHandle_t handle, 
	const int m, 
	const int n, 
	num *a, /*device in/out*/
	const int lda, 
	num *work, /*device in/out*/
	int *ipiv, /*device out*/
	int *info  /*device out*/
	)
{
#ifdef USE_CPLX
	cusolverDnZgetrf(
#else
	cusolverDnDgetrf(
#endif
	handle, m, n, a, lda, work, ipiv, info);
}

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


// LAPACK
// Solve linear equations A * x = b, using LU factorization of A 
// 	provided by xgetrf, with multiple right hand sides
static inline cusolverStatus_t xgetrs(cusolverDnHandle_t handle, 
	cublasOperation_t trans, 
	const int n, 
	const int nrhs,
	const num* a, 
	const int lda, 
	const int* ipiv,
	num* b, 
	const int ldb, 
	int* info)
{
#ifdef USE_CPLX
	cusolverDnZgetrs(
#else
	cusolverDnDgetrs(
#endif
	handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}

// TODO: problem! No pivoted QR in CUSOLVER
// Is there some way to replace this functionality?
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

// LAPACK
// QR factorization of general matrix without column pivoting
// A = Q*R
// upper triangular part of A is R, Q represented implicitly
static inline cusolverStatus_t xgeqrf(cusolverDnHandle_t handle, 
	const int m, 
	const int n, 
	num* a, 
	const int lda, 
	num* tau,
	num* work, 
	const int lwork, 
	int* info)
{
#ifdef USE_CPLX
	cusolverDnZgeqrf(
#else
	cusolverDnDgeqrf(
#endif
	handle, m, n, a, lda, tau, work, lwork, info);
}


// LAPACK
// Multiplies a real/complex matrix C by the orthogonal/unitary matrix Q 
// [stored in A] of the QR factorization formed by xgeqrf
// If trans = 'C', use Q^T/Q^H
// If side = "L", C <- Q^T * C; if side = "R", C <- C * Q^T
static inline cusolverStatus_t xunmqr(cusolverDnHandle_t handle,
    const cublasSideMode_t side,
    const cublasOperation_t trans,
	const int m, 
	const int n, 
	const int k, 
	const num* a,
	const int lda,
	const num* tau, 
	num* c,
	const int ldc, 
	num* work, 
	const int lwork, 
	int* info)
{
#ifdef USE_CPLX
	cusolverDnZunmqr(handle, side, trans,
#else
	cusolverDnDormqr(handle, side, trans == CUBLAS_OP_C ? CUBLAS_OP_T : trans,
#endif
	m, n, k, a, lda, tau,
	c, ldc, work, lwork, info);
}

// TODO: problem! only cusolverDnXtrtri interface in CUBLAS
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

// #undef ccast
// #undef cast