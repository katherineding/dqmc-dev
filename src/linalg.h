#pragma once

//#include <blis.h>
#include <FLAME.h>
#include "util.h"

// #ifdef USE_CPLX
// 	#define cast(p) (dcomplex *)(p)
// 	#define ccast(p) (const dcomplex *)(p)
// #else
// 	#define cast(p) (p)
// 	#define ccast(p) (p)
// #endif

/*=============================================
=            BLAS routine wrappers            =
=============================================*/

// general matrix-matrix multiplication: Level 3 BLAS
// C <- alpha * A * B + beta * C when transa = transb = 'N' or 'n'
static inline void xgemm(
	char *transa, 
	char *transb,
	int m, 
	int n, 
	int k,
	num alpha, 
	num *a, 
	int lda,
	num *b, 
	int ldb,
	num beta, 
	num *c, 
	int ldc)
{
#ifdef USE_CPLX
	zgemm_(
#else
	dgemm_(
#endif
	transa, transb, &m, &n, &k,
	&alpha, a, &lda, b, &ldb,
	&beta, c, &ldc);
}

// general matrix-vector product, Level 2 BLAS
// y <- alpha * A * x + beta * y when trans = 'N' or 'n'
static inline void xgemv(char *trans, 
	int m, 
	int n,
	num alpha, 
	num *a, 
	int lda,
	num *x, 
	int incx,
	num beta,
	num *y, 
	int incy)
{
#ifdef USE_CPLX
	zgemv_(
#else
	dgemv_(
#endif
	trans, &m, &n,
	&alpha, a, &lda, x, &incx,
	&beta, y, &incy);
}

// triangular matrix - general matrix product, level 3 BLAS
// Assume uplo = "U" or 'u' so A is a upper triangular matrix. The entries in 
// 	the strictly lower triangular part of A is ignored
// Assume diag = 'N' or 'n' so A does not have unit diagonal
// Assume transa = "N" or 'n'
// If side = "L" or 'l' perform C <- A*B
// If side = "R" or 'r' perform C <- B*A
static inline void xtrmm(
	char *side, 
	char *uplo, 
	char *transa, 
	char *diag,
	int m, 
	int n,
	num alpha,
	num *a, 
	int lda,
	num *b, 
	int ldb)
{
#ifdef USE_CPLX
	ztrmm_(
#else
	dtrmm_(
#endif
	side, uplo, transa, diag, &m, &n,
	&alpha, a, &lda, b, &ldb);
}

/*=============================================
=           LAPACK routine wrappers           =
=============================================*/

// LAPACK
// Compute LU factorization of general matrix A using partial pivoting 
// with row interchanges
// A = P*L*U
// A is overwritten by L (unit diagonal) and U
static inline void xgetrf(
	int m, 
	int n, 
	num* a,
	int lda, 
	int* ipiv, 
	int* info)
{
#ifdef USE_CPLX
	zgetrf_(
#else
	dgetrf_(
#endif
	&m, &n, a, &lda, ipiv, info);
}


// LAPACK
// Compute inverse of general matrix using LU factorization provided by xgetrf
static inline void xgetri(
	int n, num* a, 
	int lda, 
	int* ipiv,
	num* work, 
	int lwork, 
	int* info)
{
#ifdef USE_CPLX
	zgetri_(
#else
	dgetri_(
#endif
	&n, a, &lda, ipiv, work, &lwork, info);
}

// LAPACK
// Solve linear equations A * x = b, using LU factorization of A 
// 	provided by xgetrf, with multiple right hand sides
static inline void xgetrs(
	char* trans, 
	int n, 
	int nrhs,
	num* a, 
	int lda, 
	int* ipiv,
	num* b, 
	int ldb, 
	int* info)
{
#ifdef USE_CPLX
	zgetrs_(
#else
	dgetrs_(
#endif
	trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

// LAPACK
// QR factorization of general matrix using column pivotin
static inline void xgeqp3(
	int m, 
	int n, 
	num* a, 
	int lda, 
	int* jpvt, num* tau,
	num* work, 
	int lwork, 
	double* rwork, 
	int* info)
{
#ifdef USE_CPLX
	zgeqp3_(&m, &n, a, &lda, jpvt, tau,
	work, &lwork, rwork, info);
#else
	dgeqp3_(&m, &n, a, &lda, jpvt, tau,
	work, &lwork, info); // rwork not used
#endif
}

// LAPACK
// QR factorization of general matrix without column pivoting
// A = Q*R
// upper triangular part of A is R, Q represented implicitly
static inline void xgeqrf(
	int m, 
	int n, 
	num* a, 
	int lda, 
	num* tau,
	num* work, 
	int lwork, 
	int* info)
{
#ifdef USE_CPLX
	zgeqrf_(
#else
	dgeqrf_(
#endif
	&m, &n, a, &lda, tau, work, &lwork, info);
}

// LAPACK
// Multiplies a real/complex matrix C by the orthogonal/unitary matrix Q 
// [stored in A] of the QR factorization formed by xgeqrf
// If trans = 'C', use Q^T/Q^H
// If side = "L", C <- Q^T * C; if side = "R", C <- C * Q^T
static inline void xunmqr(char* side, 
	char* trans,
	int m, 
	int n, 
	int k, 
	num* a,
	int lda, 
	num* tau, 
	num* c,
	int ldc, 
	num* work, 
	int lwork, 
	int* info)
{
#ifdef USE_CPLX
	zunmqr_(side, trans,
#else
	dormqr_(side, trans[0] == 'C' ? "T" : trans,
#endif
	&m, &n, &k, a, &lda, tau,
	c, &ldc, work, &lwork, info);
}

// LAPACK
// Compute inverse of triangular matrix
static inline void xtrtri(
	char* uplo, 
	char* diag, 
	int n,
	num* a, 
	int lda, 
	int* info)
{
#ifdef USE_CPLX
	ztrtri_(
#else
	dtrtri_(
#endif
	uplo, diag, &n, a, &lda, info);
}

// #undef ccast
// #undef cast
