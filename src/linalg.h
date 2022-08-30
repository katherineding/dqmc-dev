#pragma once

#include <cblas.h>
#include <lapacke.h>
#include "util.h"

//#ifdef USE_CPLX
//	#define cast(p) (MKL_Complex16 *)(p)
//	#define ccast(p) (const MKL_Complex16 *)(p)
//#else
//	#define cast(p) (p)
//	#define ccast(p) (p)
//#endif


/*=============================================
=            BLAS routine wrappers            =
=============================================*/

// general matrix-matrix multiplication: Level 3 BLAS
// C <- alpha * A * B + beta * C when transa = transb = 'N' or 'n'
// Only trans = "N" or trans = "T" appears in the code.
static inline void xgemm(const char *transa, const char *transb,
		const int m, const int n, const int k,
		const num alpha, const num *a, const int lda,
		const num *b, const int ldb,
		const num beta, num *c, const int ldc)
{
        enum CBLAS_TRANSPOSE Transa = (*transa == 'N' || *transa == 'n') ? CblasNoTrans : CblasTrans;
        enum CBLAS_TRANSPOSE Transb = (*transb == 'N' || *transb == 'n') ? CblasNoTrans : CblasTrans;
#ifdef USE_CPLX
	cblas_zgemm(
#else
	cblas_dgemm(
#endif
	CblasColMajor, Transa, Transb, m, n, 
        k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// general matrix-vector product, Level 2 BLAS
// y <- alpha * A * x + beta * y when trans = 'N' or 'n'
static inline void xgemv(const char *trans, const int m, const int n,
		const num alpha, const num *a, const int lda,
		const num *x, const int incx,
		const num beta, num *y, const int incy)
{
        enum CBLAS_TRANSPOSE Trans = (*trans == 'N' || *trans == 'n') ? CblasNoTrans : CblasTrans;
#ifdef USE_CPLX
	cblas_zgemv(
#else
	cblas_dgemv(
#endif
	CblasColMajor, Trans, m, n,
	alpha, a, lda, x, incx,
	beta, y, incy);
}

// triangular matrix - general matrix product, level 3 BLAS
// Assume uplo = "U" or 'u' so A is a upper triangular matrix. The entries in 
// 	the strictly lower triangular part of A is ignored
// Assume diag = 'N' or 'n' so A does not have unit diagonal
// Assume transa = "N" or 'n'
// If side = "L" or 'l' perform C <- A*B
// If side = "R" or 'r' perform C <- B*A
static inline void xtrmm(const char *side, const char *uplo, const char *transa, const char *diag,
		const int m, const int n,
		const num alpha, const num *a, const int lda,
		num *b, const int ldb)
{
    enum CBLAS_SIDE Side = (*side == 'l' || *side == 'L') ? CblasLeft : CblasRight ;
    enum CBLAS_UPLO Uplo = (*uplo == 'u' || *uplo == 'U') ? CblasUpper : CblasLower ;
    enum CBLAS_TRANSPOSE Transa = (*transa == 'N' || *transa == 'n') ? CblasNoTrans : CblasTrans;
    enum CBLAS_DIAG Diag = (*diag == 'u' || *diag == 'U') ? CblasUnit : CblasNonUnit ; 

#ifdef USE_CPLX
	cblas_ztrmm(
#else
	cblas_dtrmm(
#endif
	CblasColMajor, Side, Uplo, Transa, Diag, m, n,
	alpha, a, lda, b, ldb);
}

/*=============================================
=           LAPACK routine wrappers           =
=============================================*/

// LAPACK
// Compute LU factorization of general matrix A using partial pivoting 
// with row interchanges
// A = P*L*U
// A is overwritten by L (unit diagonal) and U
static inline void xgetrf(const int m, const int n, num* a,
		const int lda, int* ipiv, int* info)
{
#ifdef USE_CPLX
        LAPACK_zgetrf(
#else
	LAPACK_dgetrf(
#endif
	&m, &n, a, &lda, ipiv, info);
}


// LAPACK
// Compute inverse of general matrix using LU factorization provided by xgetrf
static inline void xgetri(const int n, num* a, const int lda, const int* ipiv,
		num* work, const int lwork, int* info)
{
#ifdef USE_CPLX
	LAPACK_zgetri(
#else
	LAPACK_dgetri(
#endif
	&n, a, &lda, ipiv, work, &lwork, info);
}

// LAPACK
// Solve linear equations A * x = b, using LU factorization of A 
// 	provided by xgetrf, with multiple right hand sides
static inline void xgetrs(const char* trans, const int n, const int nrhs,
		const num* a, const int lda, const int* ipiv,
		num* b, const int ldb, int* info)
{
#ifdef USE_CPLX
	LAPACK_zgetrs(
#else
	LAPACK_dgetrs(
#endif
	trans, &n, &nrhs, a, &lda, ipiv, b, &ldb,info);
}

// LAPACK
// QR factorization of general matrix using column pivotin
static inline void xgeqp3(const int m, const int n, num* a, const int lda, int* jpvt, num* tau,
		num* work, const int lwork, double* rwork, int* info)
{
#ifdef USE_CPLX
	LAPACK_zgeqp3(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, info);
#else
	LAPACK_dgeqp3(&m, &n, a, &lda, jpvt, tau, work, &lwork, info); // rwork not used
#endif
}


// LAPACK
// QR factorization of general matrix without column pivoting
// A = Q*R
// upper triangular part of A is R, Q represented implicitly
static inline void xgeqrf(const int m, const int n, num* a, const int lda, num* tau,
		num* work, const int lwork, int* info)
{
#ifdef USE_CPLX
	LAPACK_zgeqrf(
#else
	LAPACK_dgeqrf(
#endif
	&m, &n, a, &lda, tau, work, &lwork, info);
}

// LAPACK
// Multiplies a real/complex matrix C by the orthogonal/unitary matrix Q 
// [stored in A] of the QR factorization formed by xgeqrf
// If trans = 'C', use Q^T/Q^H
// If side = "L", C <- Q^T * C; if side = "R", C <- C * Q^T
static inline void xunmqr(const char* side, const char* trans,
		const int m, const int n, const int k, const num* a,
		const int lda, const num* tau, num* c,
		const int ldc, num* work, const int lwork, int* info)
{
#ifdef USE_CPLX
	LAPACK_zunmqr(side, trans,
#else
	LAPACK_dormqr(side, trans[0] == 'C' ? "T" : trans,
#endif
	&m, &n, &k, a, &lda, tau,
	c, &ldc, work, &lwork, info);
}

// LAPACK
// Compute inverse of triangular matrix
static inline void xtrtri(const char* uplo, const char* diag, const int n,
		num* a, const int lda, int* info)
{
#ifdef USE_CPLX
	LAPACK_ztrtri(
#else
	LAPACK_dtrtri(
#endif
	uplo, diag, &n, a, &lda, info);
}

