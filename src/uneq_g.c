#include "uneq_g.h"
#include <mkl.h>
#include "prof.h"
#include "util.h"

#define G_BLK(i, j) (G + N*(i) + NL*N*(j))

int get_lwork_ue_g(const int N, const int L)
{
	double lwork;
	int info = 0;
	int max_lwork = N*N; // can be smaller if mul_seq doesn't use work

	const int NL = N*L;
	const int N2 = 2*N;

	dgeqrf(&N2, &N, NULL, &NL, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("L", "T", &N2, &N, &N, NULL, &NL, NULL, NULL, &NL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dgeqrf(&N2, &N2, NULL, &NL, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "T", &NL, &N2, &N2, NULL, &N2, NULL, NULL, &NL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "T", &NL, &N2, &N, NULL, &N2, NULL, NULL, &NL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}

static inline void mul_seq(const int N, const int stride, const int L,
		const int min, const int maxp1, const double *const restrict B,
		const double alpha, double *const restrict A, const int ldA,
		double *const restrict work)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(work);

	const int n_mul = (min == maxp1) ? L : (L + maxp1 - min) % L;
	if (n_mul == 1) {
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			A[i + ldA*j] = alpha*B[i + N*j + stride*min];
		return;
	}

	int l = min;
	if (n_mul % 2 == 0) {
		dgemm("N", "N", &N, &N, &N, &alpha, B + stride*((l + 1)%L),
		      &N, B + stride*l, &N, cdbl(0.0), A, &ldA);
		l = (l + 2) % L;
	} else {
		dgemm("N", "N", &N, &N, &N, &alpha, B + stride*((l + 1)%L),
		      &N, B + stride*l, &N, cdbl(0.0), work, &N);
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 2)%L),
		      &N, work, &N, cdbl(0.0), A, &ldA);
		l = (l + 3) % L;
	}

	for (; l != maxp1; l = (l + 2) % L) {
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*l,
		      &N, A, &ldA, cdbl(0.0), work, &N);
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 1)%L),
		      &N, work, &N, cdbl(0.0), A, &ldA);
	}
}

static void calc_o(const int N, const int stride, const int L, const int n_mul,
		const double *const restrict B, double *const restrict G,
		double *const restrict work)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(G);

	const int E = 1 + (L - 1) / n_mul;
	const int NE = N*E;

	for (int i = 0; i < NE * NE; i++) G[i] = 0.0;

	for (int e = 0; e < E - 1; e++) // subdiagonal blocks
		mul_seq(N, stride, L, e*n_mul, (e + 1)*n_mul, B, -1.0,
		        G + N*(e + 1) + NE*N*e, NE, work);
	for (int j = 0; j < N; j++) // top right corner
	for (int i = 0; i < N; i++)
		mul_seq(N, stride, L, (E - 1)*n_mul, 0, B, 1.0,
		        G + NE*N*(E - 1), NE, work);

	for (int i = 0; i < NE; i++) G[i + NE*i] += 1.0; // 1 on diagonal
}

static void bsofi(const int N, const int L,
		double *const restrict G, // input: O matrix, output: G = O^-1
		double *const restrict tau, // NL
		double *const restrict Q, // 2*N * 2*N
		double *const restrict work, const int lwork)
{
	_aa(G); _aa(tau); _aa(Q); _aa(work);

	const int NL = N*L;
	const int N2 = 2*N;
	int info;

	// block qr
	for (int l = 0; l < L - 2; l++) {
		dgeqrf(&N2, &N, G_BLK(l, l), &NL, tau + N*l, work, &lwork, &info);
		dormqr("L", "T", &N2, &N, &N, G_BLK(l, l), &NL, tau + N*l,
		       G_BLK(l, l + 1), &NL, work, &lwork, &info);
		dormqr("L", "T", &N2, &N, &N, G_BLK(l, l), &NL, tau + N*l,
		       G_BLK(l, L - 1), &NL, work, &lwork, &info);
	}
	dgeqrf(&N2, &N2, G_BLK(L - 2, L - 2), &NL, tau + N*(L - 2), work, &lwork, &info);

	// invert r
	if (L <= 2) {
		dtrtri("U", "N", &NL, G, &NL, &info);
	} else {
		dtrtri("U", "N", cint(3*N), G_BLK(L - 3, L - 3), &NL, &info);
		if (L > 3) {
			dtrmm("R", "U", "N", "N", cint(N*(L - 3)), &N, cdbl(1.0),
			      G_BLK(L - 1, L - 1), &NL, G_BLK(0, L - 1), &NL);
			for (int l = L - 4; l >= 0; l--) {
				dtrtri("U", "N", &N, G_BLK(l, l), &NL, &info);
				dtrmm("L", "U", "N", "N", &N, &N, cdbl(-1.0),
				      G_BLK(l, l), &NL, G_BLK(l, L - 1), &NL);
				dtrmm("L", "U", "N", "N", &N, &N, cdbl(-1.0),
				      G_BLK(l, l), &NL, G_BLK(l, l + 1), &NL);
				dgemm("N", "N", &N, cint(N*(L - l - 2)), &N, cdbl(1.0),
				      G_BLK(l, l + 1), &NL, G_BLK(l + 1, l + 2), &NL, cdbl(1.0),
				      G_BLK(l, l + 2), &NL);
				dtrmm("R", "U", "N", "N", &N, &N, cdbl(1.0),
				      G_BLK(l + 1, l + 1), &NL, G_BLK(l, l + 1), &NL);
			}
		}
	}

	// multiply by q inverse
	for (int i = 0; i < 4*N*N; i++) Q[i] = 0.0;

	for (int j = 0; j < N2; j++)
	for (int i = j + 1; i < N2; i++) {
		Q[i + N2*j] = G_BLK(L - 2, L - 2)[i + NL*j];
		G_BLK(L - 2, L - 2)[i + NL*j] = 0.0;
	}
	dormqr("R", "T", &NL, &N2, &N2, Q, &N2, tau + N*(L - 2),
	       G_BLK(0, L - 2), &NL, work, &lwork, &info);
	for (int l = L - 3; l >= 0; l--) {
		for (int j = 0; j < N; j++)
		for (int i = j + 1; i < N2; i++) {
			Q[i + N2*j] = G_BLK(l, l)[i + NL*j];
			G_BLK(l, l)[i + NL*j] = 0.0;
		}
		dormqr("R", "T", &NL, &N2, &N, Q, &N2, tau + N*l,
		       G_BLK(0, l), &NL, work, &lwork, &info);
	}
}

static void expand_g(const int N, const int stride, const int L, const int E, const int n_matmul,
		const double *const restrict B,
		const double *const restrict iB,
		const double *const restrict Gred,
		double *const restrict G)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(iB); _aa(Gred); _aa(G);

	const int NL = N*L;

	// copy Gred to G
	for (int f = 0; f < E; f++)
	for (int e = 0; e < E; e++) {
		const int l = f*n_matmul;
		const int k = e*n_matmul;
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			G_BLK(k, l)[i + NL*j] = Gred[(i + N*e) + N*E*(j + N*f)];
	}

	// number of steps to move in each direction
	// except for boundaries, when L % n_matmul != 0
	const int n_left = (n_matmul - 1)/2;
	const int n_right = n_matmul/2;
	const int n_up = n_left;
	const int n_down = n_right;

	// left and right
	for (int f = 0; f < E; f++)
	for (int e = 0; e < E; e++) {
		const int l = f*n_matmul;
		const int k = e*n_matmul;
		int lstop = l - n_left;
		if (f == 0) {
			lstop = (E - 1)*n_matmul + n_right;
			if (lstop >= L) lstop = L - 1;
			lstop = (lstop + 1) % L;
		}
		int rstop = l + n_right;
		if (rstop >= L) rstop = L - 1;
		for (int m = l; m != lstop;) {
			const int next = (m - 1 + L) % L;
			const double alpha = (m == 0) ? -1.0 : 1.0;
			dgemm("N", "N", &N, &N, &N, &alpha,
			      G_BLK(k, m), &NL, B + stride*next, &N, cdbl(0.0),
			      G_BLK(k, next), &NL);
			m = next;
		}
		for (int m = l; m != rstop;) {
			const int next = (m + 1) % L;
			const double alpha = (next == 0) ? -1.0 : 1.0;
			const double beta = (k == m) ? -alpha : 0.0;
			if (k == m) {
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					G_BLK(k, next)[i + NL*j] = iB[i + N*j + stride*m];
			}
			dgemm("N", "N", &N, &N, &N, &alpha,
			      G_BLK(k, m), &NL, iB + stride*m, &N, &beta,
			      G_BLK(k, next), &NL);
			m = next;
		}
	}

	// up and down
	for (int e = 0; e < E; e++)
	for (int l = 0; l < L; l++) {
		const int k = e*n_matmul;
		int ustop = k - n_up;
		if (e == 0) {
			ustop = (E - 1)*n_matmul + n_down;
			if (ustop >= L) ustop = L - 1;
			ustop = (ustop + 1) % L;
		}
		int dstop = k + n_down;
		if (dstop >= L) dstop = L - 1;
		for (int m = k; m != ustop;) {
			const int next = (m - 1 + L) % L;
			const double alpha = (m == 0) ? -1.0 : 1.0;
			const double beta = (m == l) ? -alpha : 0.0;
			if (m == l)
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					G_BLK(next, l)[i + NL*j] = iB[i + N*j + stride*next];
			dgemm("N", "N", &N, &N, &N, &alpha,
			      iB + stride*next, &N, G_BLK(m, l), &NL, &beta,
			      G_BLK(next, l), &NL);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			const double alpha = (next == 0) ? -1.0 : 1.0;
			dgemm("N", "N", &N, &N, &N, &alpha,
			      B + stride*m, &N, G_BLK(m, l), &NL, cdbl(0.0),
			      G_BLK(next, l), &NL);
			if (next == l)
				for (int i = 0; i < N; i++)
					G_BLK(next, l)[i + NL*i] += 1.0;
			m = next;
		}
	}
}

void calc_ue_g(const int N, const int stride, const int L, const int F, const int n_mul,
		const double *const restrict B, const double *const restrict iB,
		const double *const restrict C,
		double *const restrict G,
		double *const restrict Gred,
		double *const restrict tau,
		double *const restrict Q,
		double *const restrict work, const int lwork)
{
	const int E = 1 + (F - 1) / n_mul;

	profile_begin(calc_o);
	calc_o(N, stride, F, n_mul, C, Gred, work);
	profile_end(calc_o);

	profile_begin(bsofi);
	bsofi(N, E, Gred, tau, Q, work, lwork);
	profile_end(bsofi);

	profile_begin(expand_g);
	expand_g(N, stride, L, E, (L/F) * n_mul, B, iB, Gred, G);
	profile_end(expand_g);
}


// tests
// icc -D_TEST_ -std=gnu11 -Wall -Wextra -Ofast -xHost -DMKL_DIRECT_CALL_SEQ -mkl=sequential uneq_g.c -o test
#ifdef _TEST_

#include <math.h>
#include <stdio.h>
#include "rand.h"
#include "util.h"
static void test1(const int N, const int F)
{
	const int stride = N*N;
	const int NF = N*F;

	double *C = my_calloc(stride*F * sizeof(double));
	double *O = my_calloc(NF*NF * sizeof(double));
	double *G = my_calloc(NF*NF * sizeof(double));
	double *I = my_calloc(NF*NF * sizeof(double));
	double *tau = my_calloc(NF * sizeof(double));
	double *Q = my_calloc(2*N * 2*N * sizeof(double));
	const int lwork = get_lwork_ue_g(N, F);
	double *work = my_calloc(lwork * sizeof(double));

	uint64_t rng[17] = {0};
	rng[7] = 117;
	for (int i = 0; i < 1117; i++) rand_uint(rng);

	for (int i = 0; i < stride*F; i++) C[i] = 2.0*rand_doub(rng) - 1.0;
	calc_o(N, stride, F, 1, C, O, work);
	for (int i = 0; i < NF*NF; i++) G[i] = O[i];
	bsofi(N, F, G, tau, Q, work, lwork);

	double avg;

	dgemm("N", "N", &NF, &NF, &NF, cdbl(1.0), O, &NF, G, &NF, cdbl(0.0), I, &NF);
	avg = 0.0;
	for (int i = 0; i < NF*NF; i++) avg += fabs(I[i] - (i % (NF + 1) == 0));
	avg /= NF*NF;
	printf("%g\n", avg);

	dgemm("N", "N", &NF, &NF, &NF, cdbl(1.0), G, &NF, O, &NF, cdbl(0.0), I, &NF);
	avg = 0.0;
	for (int i = 0; i < NF*NF; i++) avg += fabs(I[i] - (i % (NF + 1) == 0));
	avg /= NF*NF;
	printf("%g\n", avg);

	my_free(work);
	my_free(Q);
	my_free(tau);
	my_free(I);
	my_free(G);
	my_free(O);
	my_free(C);
}

static void test2(const int N, const int L, const int F)
{
	if (L % F != 0) {
		printf("%d %% %d != 0\n", L, F);
		return;
	}
	const int n_matmul = L/F;
	const int stride = N*N;
	const int NF = N*F;
	const int NL = N*L;

	double *B = my_calloc(stride*L * sizeof(double));
	double *iB = my_calloc(stride*L * sizeof(double));
	double *C = my_calloc(stride*F * sizeof(double));
	double *Gred = my_calloc(NF*NF * sizeof(double));
	double *O = my_calloc(NL*NL * sizeof(double));
	double *G = my_calloc(NL*NL * sizeof(double));
	double *temp = my_calloc(NL*NL * sizeof(double));

	double *tau = my_calloc(NL * sizeof(double));
	double *Q = my_calloc(2*N * 2*N * sizeof(double));
	const int lwork = get_lwork_ue_g(N, L);
	double *work = my_calloc(lwork * sizeof(double));

	double avg;

	uint64_t rng[17] = {0};
	rng[7] = 117;
	for (int i = 0; i < 1117; i++) rand_uint(rng);

	for (int i = 0; i < stride*L; i++) B[i] = 2.0*rand_doub(rng) - 1.0;

	int info;
	int *piv = my_calloc(N * sizeof(int));
	for (int l = 0; l < L; l++) {
		for (int i = 0; i < N*N; i++) iB[i + stride*l] = B[i + stride*l];
		dgetrf(&N, &N, iB + stride*l, &N, piv, &info);
		dgetri(&N, iB + stride*l, &N, piv, work, &lwork, &info);
	}

	for (int f = 0; f < F; f++)
		mul_seq(N, stride, L, f*n_matmul, ((f+1)*n_matmul)%L, B, 1.0,
		        C + stride*f, N, temp);

	calc_ue_g(N, stride, L, F, 1, B, iB, C, G, Gred, tau, Q, work, lwork);

	calc_o(N, stride, L, 1, B, O, work);
	for (int i = 0; i < NL*NL; i++) temp[i] = O[i];
	bsofi(N, L, temp, tau, Q, work, lwork);

	avg = 0.0;
	for (int j = 0; j < NL; j++)
	for (int i = 0; i < NL; i++)
		avg += fabs(G[i + NL*j] - temp[i + NL*j]);
	avg /= NL * NL;
	printf("%g\n", avg);

	// for (int m = 0; m < L; m++)
	// for (int k = 0; k < L; k++) {
		// avg = 0.0;
		// for (int j = 0; j < N; j++)
		// for (int i = 0; i < N; i++)
			// avg += fabs(G[(i + k*N) + NL*(j + m*N)] - temp[(i + k*N) + NL*(j + m*N)]);
		// avg /= N*N;
		// printf("%d\t%d\t%g\n", k, m, avg);
	// }

	// dgemm("N", "N", &NL, &NL, &NL, cdbl(1.0), O, &NL, G, &NL, cdbl(0.0), temp, &NL);
	// avg = 0.0;
	// for (int i = 0; i < NL*NL; i++) avg += fabs(temp[i] - (i % (NL + 1) == 0));
	// avg /= NL*NL;
	// printf("%g\n", avg);

	// dgemm("N", "N", &NL, &NL, &NL, cdbl(1.0), G, &NL, O, &NL, cdbl(0.0), temp, &NL);
	// avg = 0.0;
	// for (int i = 0; i < NL*NL; i++) avg += fabs(temp[i] - (i % (NL + 1) == 0));
	// avg /= NL*NL;
	// printf("%g\n", avg);

	my_free(piv);
	my_free(work);
	my_free(Q);
	my_free(tau);
	my_free(temp);
	my_free(G);
	my_free(O);
	my_free(Gred);
	my_free(C);
	my_free(iB);
	my_free(B);
}

static void test3(const int N, const int L, const int F, const int n_mul)
{
	if (L % F != 0) {
		printf("%d %% %d != 0\n", L, F);
		return;
	}
	const int n_matmul = L/F;
	const int stride = N*N;
	const int NF = N*F;
	const int NL = N*L;

	double *B = my_calloc(stride*L * sizeof(double));
	double *iB = my_calloc(stride*L * sizeof(double));
	double *C = my_calloc(stride*F * sizeof(double));
	double *Gred = my_calloc(NF*NF * sizeof(double));
	double *O = my_calloc(NL*NL * sizeof(double));
	double *G = my_calloc(NL*NL * sizeof(double));
	double *temp = my_calloc(NL*NL * sizeof(double));

	double *tau = my_calloc(NL * sizeof(double));
	double *Q = my_calloc(2*N * 2*N * sizeof(double));
	const int lwork = get_lwork_ue_g(N, L);
	double *work = my_calloc(lwork * sizeof(double));

	double avg;

	uint64_t rng[17] = {0};
	rng[7] = 117;
	for (int i = 0; i < 1117; i++) rand_uint(rng);

	for (int i = 0; i < stride*L; i++) B[i] = 2.0*rand_doub(rng) - 1.0;

	int info;
	int *piv = my_calloc(N * sizeof(int));
	for (int l = 0; l < L; l++) {
		for (int i = 0; i < N*N; i++) iB[i + stride*l] = B[i + stride*l];
		dgetrf(&N, &N, iB + stride*l, &N, piv, &info);
		dgetri(&N, iB + stride*l, &N, piv, work, &lwork, &info);
	}

	for (int f = 0; f < F; f++)
		mul_seq(N, stride, L, f*n_matmul, ((f+1)*n_matmul)%L, B, 1.0,
		        C + stride*f, N, temp);

	calc_ue_g(N, stride, L, F, n_mul, B, iB, C, G, Gred, tau, Q, work, lwork);

	calc_o(N, stride, L, 1, B, O, work);
	for (int i = 0; i < NL*NL; i++) temp[i] = O[i];
	bsofi(N, L, temp, tau, Q, work, lwork);

	avg = 0.0;
	for (int j = 0; j < NL; j++)
	for (int i = 0; i < NL; i++)
		avg += fabs(G[i + NL*j] - temp[i + NL*j]);
	avg /= NL * NL;
	printf("%g\n", avg);

	my_free(piv);
	my_free(work);
	my_free(Q);
	my_free(tau);
	my_free(temp);
	my_free(G);
	my_free(O);
	my_free(Gred);
	my_free(C);
	my_free(iB);
	my_free(B);
}

int main(void)
{
	puts("test1");
	test1(1, 2);
	test1(1, 13);
	test1(8, 2);
	test1(8, 8);
	test1(17, 23);
	test1(32, 24);
	puts("test2");
	test2(4, 4, 2);
	test2(17, 16, 2);
	test2(17, 16, 8);
	test2(67, 35, 7);
	test2(7, 63, 7);
	test2(1, 63, 7);
	test2(10, 45, 9);
	puts("test3");
	test3(13, 21, 3, 1);
	test3(13, 21, 7, 2);
	test3(5, 30, 10, 5);
	test3(5, 33, 11, 5);
	test3(5, 36, 12, 5);
	test3(5, 39, 13, 5);
	test3(5, 42, 14, 5);
	test3(5, 45, 15, 5);
	test3(8, 30, 10, 3);
	test3(8, 33, 11, 3);
	test3(8, 36, 12, 3);
	test3(8, 39, 13, 3);
	test3(23, 20, 4, 2);
	test3(23, 20, 5, 2);
	test3(16, 27, 9, 2);
	test3(16, 8, 4, 2);
	return 0;
}

#endif // _TEST_
