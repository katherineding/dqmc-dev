#pragma once

/**	
 * Interface of DQMC code with CUBLAS and CUSOLVER library code.
 * Define away d (double) and z (double complex) prefixes into x
*/

#include <cusolverDn.h>
#include <cublas_v2.h>

#include "util.h"

// TODO: made function non-void return. Restore original void return?
// TODO: how to handle functions without cublas/cuSolve implementation?


// Pointer casting to and from Nvidia cuDoubleComplex type
#ifdef USE_CPLX
	//#include <complex.h>
	// here p is double _Complex * type
	#define cast(p)  reinterpret_cast<cuDoubleComplex *>(p)
	// here p is const double _Complex * type
	#define ccast(p)  reinterpret_cast<const cuDoubleComplex *>(p)
	// here p is cuDoubleComplex * type
	#define recast(p) reinterpret_cast<double _Complex *>(p)
#else
	// if using double, just leave p alone
	#define cast(p)  p
	#define ccast(p)  p
	#define recast(p)  p
#endif



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
	return cublasZgemm(
#else
	return cublasDgemm(
#endif
	handle, transa, transb, m, n, k,
	ccast(alpha), ccast(A), lda, ccast(B), ldb, ccast(beta), cast(C), ldc);
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
	return cublasZgemv(
#else
	return cublasDgemv(
#endif
	handle, trans, m, n,
	ccast(alpha), ccast(A), lda, ccast(x), incx,
	ccast(beta), cast(y), incy);
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
	return cublasZtrmm(
#else
	return cublasDtrmm(
#endif
	handle, side, uplo, transa, diag, m, n,
	ccast(alpha), ccast(A), lda, ccast(B), ldb, cast(C), ldc);
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
	return cusolverDnZgetrf(
#else
	return cusolverDnDgetrf(
#endif
	handle, m, n, cast(a), lda, cast(work), ipiv, info);
}

static inline cusolverStatus_t xgetrf_bs(
	cusolverDnHandle_t handle, 
	const int m, 
	const int n, 
	num *a, 
	const int lda, 
	int *lwork
	)
{
#ifdef USE_CPLX
	return cusolverDnZgetrf_bufferSize(
#else
	return cusolverDnDgetrf_bufferSize(
#endif
	handle, m, n, cast(a), lda, lwork);
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
	const num* a,    /* device in */
	const int lda, 
	const int* ipiv, /* device in */
	num* b,          /* device out*/
	const int ldb, 
	int* info)       /* device out*/
{
#ifdef USE_CPLX
	return cusolverDnZgetrs(
#else
	return cusolverDnDgetrs(
#endif
	handle, trans, n, nrhs, ccast(a), lda, ipiv, cast(b), ldb, info);
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
	return cusolverDnZgeqrf(
#else
	return cusolverDnDgeqrf(
#endif
	handle, m, n, cast(a), lda, cast(tau), cast(work), lwork, info);
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
	return cusolverDnZunmqr(handle, side, trans,
#else
	return cusolverDnDormqr(handle, side, trans == CUBLAS_OP_C ? CUBLAS_OP_T : trans,
#endif
	m, n, k, ccast(a), lda, ccast(tau),
	cast(c), ldc, cast(work), lwork, info);
}

// LAPACK
// Compute inverse of triangular matrix
static inline cusolverStatus_t xtrtri(
	cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	cublasDiagType_t diag,
	const int n,
	num* a, 
	const int lda, 
	num *dwork,
	size_t dwork_size,
	num *hwork,
	size_t hwork_size,
	int *info)
{
	return cusolverDnXtrtri(handle, uplo, diag, n, 
#ifdef USE_CPLX
	CUDA_C_64F, 
#else
	CUDA_R_64F, 
#endif
	cast(a), lda, dwork, dwork_size, hwork, hwork_size, info);

}

static inline cusolverStatus_t xtrtri_bs(
	cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	cublasDiagType_t diag,
	const int n,
	num* a, 
	const int lda, 
	size_t* dwork_size,
	size_t* hwork_size
	)
{
	return cusolverDnXtrtri_bufferSize(handle, uplo, diag, n, 
#ifdef USE_CPLX
	CUDA_C_64F, 
#else
	CUDA_R_64F, 
#endif
	cast(a), lda, dwork_size,hwork_size);

}

// avoid namespace pollution
#undef cast
#undef ccast
#undef recast