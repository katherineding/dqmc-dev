#include "meas.h"
#include "data.h"

void measure_eqlt(const struct params *const restrict p, const int sign,
		const double *const restrict gu,
		const double *const restrict gd,
		struct meas_eqlt *const restrict m)
{
	m->n_sample++;
	m->sign += sign;
	const int N = p->N, num_i = p->num_i, num_ij = p->num_ij;

	// 1-site measurements
	for (int i = 0; i < N; i++) {
		const int r = p->map_i[i];
		const double pre = (double)sign / p->degen_i[r];
		const double guii = gu[i + i*N], gdii = gd[i + i*N];
		m->density[r] += pre*(2. - guii - gdii);
		m->double_occ[r] += pre*(1. - guii)*(1. - gdii);
	}

	// 2-site measurements
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int delta = (i == j);
		const int r = p->map_ij[i + j*N];
		const double pre = (double)sign / p->degen_ij[r];
		const double guii = gu[i + i*N], gdii = gd[i + i*N];
		const double guij = gu[i + j*N], gdij = gd[i + j*N];
		const double guji = gu[j + i*N], gdji = gd[j + i*N];
		const double gujj = gu[j + j*N], gdjj = gd[j + j*N];
		m->g00[r] += 0.5*pre*(guij + gdij);
		const double x = delta*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r] += 0.25*pre*(delta*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r] += pre*guij*gdij;
	}

	// 2-bond measurements
}

void measure_uneqlt(const struct params *const restrict p, const int sign,
		const double *const restrict Gu,
		const double *const restrict Gd,
		struct meas_uneqlt *const restrict m)
{
	m->n_sample++;
	m->sign += sign;
	const int N = p->N, L = p->L, num_i = p->num_i, num_ij = p->num_ij;
	const int NL = N*L;

	// 2-site measurements
	#pragma omp parallel for
	for (int t = 0; t < L; t++)
	for (int l = 0; l < L; l++)
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int k = (l + t) % L;
		const int T_sign = ((k >= l) ? 1.0 : -1.0); // for fermionic
		const int delta_t = (t == 0);
		const int r = p->map_ij[i + j*N];
		const int delta_tij = delta_t * (i == j);
		const double pre = (double)sign / p->degen_ij[r] / L;
		const double guii = Gu[(i + N*i) + N*N*(k + L*k)];
		const double guij = Gu[(i + N*j) + N*N*(k + L*l)];
		const double guji = Gu[(j + N*i) + N*N*(l + L*k)];
		const double gujj = Gu[(j + N*j) + N*N*(l + L*l)];
		const double gdii = Gd[(i + N*i) + N*N*(k + L*k)];
		const double gdij = Gd[(i + N*j) + N*N*(k + L*l)];
		const double gdji = Gd[(j + N*i) + N*N*(l + L*k)];
		const double gdjj = Gd[(j + N*j) + N*N*(l + L*l)];
		m->gt0[r + num_ij*t] += 0.5*T_sign*pre*(guij + gdij);
		const double x = delta_tij*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r + num_ij*t] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r + num_ij*t] += 0.25*pre*(delta_tij*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r + num_ij*t] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r + num_ij*t] += pre*guij*gdij;
	}
}
