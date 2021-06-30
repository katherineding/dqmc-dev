#include "meas.h"
#include "data.h"
#include "util.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

// number of types of bonds kept for 4-particle nematic correlators.
// 2 by default since these are slow measurerments
#define NEM_BONDS 2

// if complex numbers are being used, multiple some measurements by Peierls
// phases to preserve gauge invariance
#ifdef USE_CPLX
#define USE_PEIERLS
#else
// if not, these are equal to 1 anyway. multiplying by the variables costs a
// little performance, so #define them away at compile time
// TODO: exception: if using twisted boundaries, these are not always 1
#define pui0i1 1
#define pui1i0 1
#define pdi0i1 1
#define pdi1i0 1
#define puj0j1 1
#define puj1j0 1
#define pdj0j1 1
#define pdj1j0 1
// #define pui1i2 1
// #define pui2i1 1
// #define pdi1i2 1
// #define pdi2i1 1
// #define puj1j2 1
// #define puj2j1 1
// #define pdj1j2 1
// #define pdj2j1 1
#define ppui0i2 1
#define ppui2i0 1
#define ppdi0i2 1
#define ppdi2i0 1
#define ppuj0j2 1
#define ppuj2j0 1
#define ppdj0j2 1
#define ppdj2j0 1
#endif

#ifdef USE_CPLX
int approx_equal(num a, num b) {
	return fabs(creal(a)-creal(b)) < 1e-10 && fabs(cimag(a)-cimag(b)) < 1e-10;
}
#else
int approx_equal(num a, num b) {
	return fabs(a-b) < 1e-10;
}
#endif

void measure_eqlt(const struct params *const restrict p, const num phase,
		const num *const restrict gu,
		const num *const restrict gd,
		struct meas_eqlt *const restrict m)
{
	m->n_sample++;
	m->sign += phase;
	const int N = p->N, num_i = p->num_i, num_ij = p->num_ij;
	const int num_b = p->num_b, num_bs = p->num_bs, num_bb = p->num_bb;
	const int meas_energy_corr = p->meas_energy_corr;

	// 1 site measurements
	for (int i = 0; i < N; i++) {
		const int r = p->map_i[i];
		const num pre = phase / p->degen_i[r];
		const num guii = gu[i + i*N], gdii = gd[i + i*N];
		m->density[r] += pre*(2. - guii - gdii);
		m->double_occ[r] += pre*(1. - guii)*(1. - gdii);
	}

	// 2 site measurements
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int delta = (i == j);
		const int r = p->map_ij[i + j*N];
		const num pre = phase / p->degen_ij[r];
		const num guii = gu[i + i*N], gdii = gd[i + i*N];
		const num guij = gu[i + j*N], gdij = gd[i + j*N];
		const num guji = gu[j + i*N], gdji = gd[j + i*N];
		const num gujj = gu[j + j*N], gdjj = gd[j + j*N];
#ifdef USE_PEIERLS
		m->g00[r] += 0.5*pre*(guij*p->peierlsu[j + i*N] + gdij*p->peierlsd[j + i*N]);
#else
		m->g00[r] += 0.5*pre*(guij + gdij);
#endif
		const num x = delta*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r] += 0.25*pre*(delta*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r] += pre*guij*gdij;
		if (meas_energy_corr) {
			const double nuinuj = (1. - guii)*(1. - gujj) + (delta - guji)*guij;
			const double ndindj = (1. - gdii)*(1. - gdjj) + (delta - gdji)*gdij;
			m->vv[r] += pre*nuinuj*ndindj;
			m->vn[r] += pre*(nuinuj*(1. - gdii) + (1. - guii)*ndindj);
		}
	}

	if (!meas_energy_corr)
		return;

	// 1 bond 1 site measurements
	for (int j = 0; j < N; j++)
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bs = p->map_bs[b + num_b*j];
		const num pre = phase / p->degen_bs[bs];
		const int delta_i0i1 = 0;
		const int delta_i0j = (i0 == j);
		const int delta_i1j = (i1 == j);
		const num gui0j = gu[i0 + N*j];
		const num guji0 = gu[j + N*i0];
		const num gdi0j = gd[i0 + N*j];
		const num gdji0 = gd[j + N*i0];
		const num gui1j = gu[i1 + N*j];
		const num guji1 = gu[j + N*i1];
		const num gdi1j = gd[i1 + N*j];
		const num gdji1 = gd[j + N*i1];
		const num gui0i1 = gu[i0 + N*i1];
		const num gui1i0 = gu[i1 + N*i0];
		const num gdi0i1 = gd[i0 + N*i1];
		const num gdi1i0 = gd[i1 + N*i0];
		const num gujj = gu[j + N*j];
		const num gdjj = gd[j + N*j];

		const num ku = pui1i0*(delta_i0i1 - gui0i1) + pui0i1*(delta_i0i1 - gui1i0);
		const num kd = pdi1i0*(delta_i0i1 - gdi0i1) + pdi0i1*(delta_i0i1 - gdi1i0);
		const num xu = pui0i1*(delta_i0j - guji0)*gui1j + pui1i0*(delta_i1j - guji1)*gui0j;
		const num xd = pdi0i1*(delta_i0j - gdji0)*gdi1j + pdi1i0*(delta_i1j - gdji1)*gdi0j;
		m->kv[bs] += pre*((ku*(1. - gujj) + xu)*(1. - gdjj)
		                + (kd*(1. - gdjj) + xd)*(1. - gujj));
		m->kn[bs] += pre*((ku + kd)*(2. - gujj - gdjj) + xu + xd);
	}

	// 1 bond -- 1 bond measurements
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const int delta_i0j0 = (i0 == j0);
		const int delta_i1j0 = (i1 == j0);
		const int delta_i0j1 = (i0 == j1);
		const int delta_i1j1 = (i1 == j1);
		const num gui1i0 = gu[i1 + i0*N];
		const num gui0i1 = gu[i0 + i1*N];
		const num gui0j0 = gu[i0 + j0*N];
		const num gui1j0 = gu[i1 + j0*N];
		const num gui0j1 = gu[i0 + j1*N];
		const num gui1j1 = gu[i1 + j1*N];
		const num guj0i0 = gu[j0 + i0*N];
		const num guj1i0 = gu[j1 + i0*N];
		const num guj0i1 = gu[j0 + i1*N];
		const num guj1i1 = gu[j1 + i1*N];
		const num guj1j0 = gu[j1 + j0*N];
		const num guj0j1 = gu[j0 + j1*N];
		const num gdi1i0 = gd[i1 + i0*N];
		const num gdi0i1 = gd[i0 + i1*N];
		const num gdi0j0 = gd[i0 + j0*N];
		const num gdi1j0 = gd[i1 + j0*N];
		const num gdi0j1 = gd[i0 + j1*N];
		const num gdi1j1 = gd[i1 + j1*N];
		const num gdj0i0 = gd[j0 + i0*N];
		const num gdj1i0 = gd[j1 + i0*N];
		const num gdj0i1 = gd[j0 + i1*N];
		const num gdj1i1 = gd[j1 + i1*N];
		const num gdj1j0 = gd[j1 + j0*N];
		const num gdj0j1 = gd[j0 + j1*N];
		const num x = pui0i1*puj0j1*(delta_i0j1 - guj1i0)*gui1j0 + pui1i0*puj1j0*(delta_i1j0 - guj0i1)*gui0j1
		            + pdi0i1*pdj0j1*(delta_i0j1 - gdj1i0)*gdi1j0 + pdi1i0*pdj1j0*(delta_i1j0 - gdj0i1)*gdi0j1;
		const num y = pui0i1*puj1j0*(delta_i0j0 - guj0i0)*gui1j1 + pui1i0*puj0j1*(delta_i1j1 - guj1i1)*gui0j0
		            + pdi0i1*pdj1j0*(delta_i0j0 - gdj0i0)*gdi1j1 + pdi1i0*pdj0j1*(delta_i1j1 - gdj1i1)*gdi0j0;
		m->kk[bb] += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 + pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
		                 *(puj1j0*guj0j1 + puj0j1*guj1j0 + pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x + y);
	}
	}
}

void measure_uneqlt(const struct params *const restrict p, const num phase,
		const num *const Gu0t,
		const num *const Gutt,
		const num *const Gut0,
		const num *const Gd0t,
		const num *const Gdtt,
		const num *const Gdt0,
		struct meas_uneqlt *const restrict m)
{
	m->n_sample++;
	m->sign += phase;
	const int N = p->N, L = p->L, num_i = p->num_i, num_ij = p->num_ij;
	const int num_b = p->num_b, num_bs = p->num_bs, num_bb = p->num_bb;
	const int num_b2 = p->num_b2, num_b2b2 = p->num_b2b2, num_b2b = p->num_b2b, num_bb2 = p->num_bb2;
	// const int num_hop2 = p->num_hop2, num_hop2_hop2 = p->num_hop2_hop2,
	//           num_hop2_b = p->num_hop2_b, num_b_hop2 = p->num_b_hop2;
	const int meas_bond_corr = p->meas_bond_corr;
	const int meas_2bond_corr = p->meas_2bond_corr;
	// const int meas_hop2_corr = p->meas_hop2_corr;
	const int meas_energy_corr = p->meas_energy_corr;
	const int meas_nematic_corr = p->meas_nematic_corr;
	const int meas_thermal = p->meas_thermal;

	const num *const restrict Gu00 = Gutt;
	const num *const restrict Gd00 = Gdtt;

	// 2 site measurements
	#pragma omp parallel for
	for (int t = 0; t < L; t++) {
		const int delta_t = (t == 0);
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int r = p->map_ij[i + j*N];
		const int delta_tij = delta_t * (i == j);
		const num pre = phase / p->degen_ij[r];
		const num guii = Gutt_t[i + N*i];
		const num guij = Gut0_t[i + N*j];
		const num guji = Gu0t_t[j + N*i];
		const num gujj = Gu00[j + N*j];
		const num gdii = Gdtt_t[i + N*i];
		const num gdij = Gdt0_t[i + N*j];
		const num gdji = Gd0t_t[j + N*i];
		const num gdjj = Gd00[j + N*j];
#ifdef USE_PEIERLS
		m->gt0[r + num_ij*t] += 0.5*pre*(guij*p->peierlsu[j + i*N] + gdij*p->peierlsd[j + i*N]);
#else
		m->gt0[r + num_ij*t] += 0.5*pre*(guij + gdij);
#endif
		const num x = delta_tij*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r + num_ij*t] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r + num_ij*t] += 0.25*pre*(delta_tij*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r + num_ij*t] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r + num_ij*t] += pre*guij*gdij;
		if (meas_energy_corr) {
			const num nuinuj = (1. - guii)*(1. - gujj) + (delta_tij - guji)*guij;
			const num ndindj = (1. - gdii)*(1. - gdjj) + (delta_tij - gdji)*gdij;
			m->vv[r + num_ij*t] += pre*nuinuj*ndindj;
			m->vn[r + num_ij*t] += pre*(nuinuj*(1. - gdii) + (1. - guii)*ndindj);
		}
	}
	}

	// 1 bond 1 site measurements
	if (meas_energy_corr)
	#pragma omp parallel for
	for (int t = 0; t < L; t++) {
		const int delta_t = (t == 0);
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int j = 0; j < N; j++)
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bs = p->map_bs[b + num_b*j];
		const num pre = phase / p->degen_bs[bs];
		const int delta_i0i1 = 0;
		const int delta_i0j = delta_t*(i0 == j);
		const int delta_i1j = delta_t*(i1 == j);
		const num gui0j = Gut0_t[i0 + N*j];
		const num guji0 = Gu0t_t[j + N*i0];
		const num gdi0j = Gdt0_t[i0 + N*j];
		const num gdji0 = Gd0t_t[j + N*i0];
		const num gui1j = Gut0_t[i1 + N*j];
		const num guji1 = Gu0t_t[j + N*i1];
		const num gdi1j = Gdt0_t[i1 + N*j];
		const num gdji1 = Gd0t_t[j + N*i1];
		const num gui0i1 = Gutt_t[i0 + N*i1];
		const num gui1i0 = Gutt_t[i1 + N*i0];
		const num gdi0i1 = Gdtt_t[i0 + N*i1];
		const num gdi1i0 = Gdtt_t[i1 + N*i0];
		const num gujj = Gu00[j + N*j];
		const num gdjj = Gd00[j + N*j];

		const num ku = pui1i0*(delta_i0i1 - gui0i1) + pui0i1*(delta_i0i1 - gui1i0);
		const num kd = pdi1i0*(delta_i0i1 - gdi0i1) + pdi0i1*(delta_i0i1 - gdi1i0);
		const num xu = pui0i1*(delta_i0j - guji0)*gui1j + pui1i0*(delta_i1j - guji1)*gui0j;
		const num xd = pdi0i1*(delta_i0j - gdji0)*gdi1j + pdi1i0*(delta_i1j - gdji1)*gdi0j;
		m->kv[bs + num_bs*t] += pre*((ku*(1. - gujj) + xu)*(1. - gdjj)
		                           + (kd*(1. - gdjj) + xd)*(1. - gujj));
		m->kn[bs + num_bs*t] += pre*((ku + kd)*(2. - gujj - gdjj) + xu + xd);
	}
	}

	// Below includes:
	// * 1 bond <-> 1 bond correlators: pair_bb, jj, jsjs, kk, ksks (4 fermion, 2 phases)
	// * (1 site)1 bond <-> (1 site)1 bond correlator: jnjn (8 fermion, 2 phases)
	// * (1 site)1 bond <-> 1 bond correlators: jjn, jnj (6 fermion, 2 phases)
	// * 2 hop bond <-> 1 bond correlators: J2jn, jnJ2 (6 fermion, 3 phases) & J2j, jJ2 (4 fermion, 3 phases)
	// * 2 hop bond <-> 2 hop bond correlators: J2J2 (4 fermion, 4 phases)
	// * nematic correlators: nem_nnnn, nem_ssss (? fermions?)

	// minor optimization: handle t = 0 separately, since there are no delta
	// functions for t > 0. not really needed in 2-site measurements above
	// as those are fast anyway
	if (meas_bond_corr || meas_thermal)
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const int delta_i0j0 = (i0 == j0);
		const int delta_i1j0 = (i1 == j0);
		const int delta_i0j1 = (i0 == j1);
		const int delta_i1j1 = (i1 == j1);
		const int delta_i0i1 = 0;
		const int delta_j0j1 = 0;
		const num gui1i0 = Gu00[i1 + i0*N];
		const num gui0i1 = Gu00[i0 + i1*N];
		const num gui0j0 = Gu00[i0 + j0*N];
		const num gui1j0 = Gu00[i1 + j0*N];
		const num gui0j1 = Gu00[i0 + j1*N];
		const num gui1j1 = Gu00[i1 + j1*N];
		const num guj0i0 = Gu00[j0 + i0*N];
		const num guj1i0 = Gu00[j1 + i0*N];
		const num guj0i1 = Gu00[j0 + i1*N];
		const num guj1i1 = Gu00[j1 + i1*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num gdi1i0 = Gd00[i1 + i0*N];
		const num gdi0i1 = Gd00[i0 + i1*N];
		const num gdi0j0 = Gd00[i0 + j0*N];
		const num gdi1j0 = Gd00[i1 + j0*N];
		const num gdi0j1 = Gd00[i0 + j1*N];
		const num gdi1j1 = Gd00[i1 + j1*N];
		const num gdj0i0 = Gd00[j0 + i0*N];
		const num gdj1i0 = Gd00[j1 + i0*N];
		const num gdj0i1 = Gd00[j0 + i1*N];
		const num gdj1i1 = Gd00[j1 + i1*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		// 1 bond -- 1 bond correlator measurements, t = 0
		if (meas_bond_corr){
			m->pair_bb[bb] += 0.5*pre*(gui0j0*gdi1j1 + gui1j0*gdi0j1 + gui0j1*gdi1j0 + gui1j1*gdi0j0);
			const num x = pui0i1*puj0j1*(delta_i0j1 - guj1i0)*gui1j0 
			            + pui1i0*puj1j0*(delta_i1j0 - guj0i1)*gui0j1
			            + pdi0i1*pdj0j1*(delta_i0j1 - gdj1i0)*gdi1j0 
			            + pdi1i0*pdj1j0*(delta_i1j0 - gdj0i1)*gdi0j1;
			const num y = pui0i1*puj1j0*(delta_i0j0 - guj0i0)*gui1j1 
			            + pui1i0*puj0j1*(delta_i1j1 - guj1i1)*gui0j0
			            + pdi0i1*pdj1j0*(delta_i0j0 - gdj0i0)*gdi1j1 
			            + pdi1i0*pdj0j1*(delta_i1j1 - gdj1i1)*gdi0j0;
			m->jj[bb]   += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 + pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
			                   *(puj1j0*guj0j1 - puj0j1*guj1j0 + pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x - y);
			m->jsjs[bb] += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 - pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
			                   *(puj1j0*guj0j1 - puj0j1*guj1j0 - pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x - y);
			m->kk[bb]   += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 + pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
			                   *(puj1j0*guj0j1 + puj0j1*guj1j0 + pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x + y);
			m->ksks[bb] += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 - pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
			                   *(puj1j0*guj0j1 + puj0j1*guj1j0 - pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x + y);
		}
		// thermal: t = 0
		if (meas_thermal){
			const num gui0i0 = Gu00[i0 + i0*N];
			const num gui1i1 = Gu00[i1 + i1*N];
			const num guj0j0 = Gu00[j0 + j0*N];
			const num guj1j1 = Gu00[j1 + j1*N];
			const num gdi0i0 = Gd00[i0 + i0*N];
			const num gdi1i1 = Gd00[i1 + i1*N];
			const num gdj0j0 = Gd00[j0 + j0*N];
			const num gdj1j1 = Gd00[j1 + j1*N];

			//jn(i0i1)j(j0j1): 6 fermion product, 2 phases, t = 0
			//TODO: further group these expressions together?
			num _wick_jn = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
			 			   (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			num _wick_j = - puj1j0*guj0j1 + puj0j1*guj1j0 - pdj1j0*gdj0j1 + pdj0j1*gdj1j0;

			num t1 = ( (delta_i0j1 - guj1i0) * gui0j0 + (delta_i1j1 - guj1i1) * gui1j0 ) * 
				puj0j1 * (pdi1i0 * gdi0i1 - pdi0i1 * gdi1i0);
			num t2 = ( (delta_i0j0 - guj0i0) * gui0j1 + (delta_i1j0 - guj0i1) * gui1j1 ) * 
				puj1j0 * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1);
			num t3 = ( (delta_i0j1 - gdj1i0) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi1j0 ) * 
				pdj0j1 * (pui1i0 * gui0i1 - pui0i1 * gui1i0);
			num t4 = ( (delta_i0j0 - gdj0i0) * gdi0j1 + (delta_i1j0 - gdj0i1) * gdi1j1 ) * 
				pdj1j0 * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			num t5 = (2 - gui0i0 - gui1i1) * 
				(+pdi0i1 * pdj0j1 * (delta_i0j1 - gdj1i0) * gdi1j0 
				 -pdi0i1 * pdj1j0 * (delta_i0j0 - gdj0i0) * gdi1j1
				 -pdi1i0 * pdj0j1 * (delta_i1j1 - gdj1i1) * gdi0j0
				 +pdi1i0 * pdj1j0 * (delta_i1j0 - gdj0i1) * gdi0j1);
			num t6 = (2 - gdi0i0 - gdi1i1) *
				(+pui0i1 * puj0j1 * (delta_i0j1 - guj1i0) * gui1j0
				 -pui0i1 * puj1j0 * (delta_i0j0 - guj0i0) * gui1j1
				 -pui1i0 * puj0j1 * (delta_i1j1 - guj1i1) * gui0j0
				 +pui1i0 * puj1j0 * (delta_i1j0 - guj0i1) * gui0j1);

			// const num t13 =  pdi0i1 * pdj0j1 * (2-gui0i0-gui1i1) * (delta_i0j1 - gdj1i0) * gdi1j0;
			// const num t14 = -pdi0i1 * pdj1j0 * (2-gui0i0-gui1i1) * (delta_i0j0 - gdj0i0) * gdi1j1;
			// const num t23 = -pdi1i0 * pdj0j1 * (2-gui0i0-gui1i1) * (delta_i1j1 - gdj1i1) * gdi0j0;
			// const num t24 =  pdi1i0 * pdj1j0 * (2-gui0i0-gui1i1) * (delta_i1j0 - gdj0i1) * gdi0j1;

			// const num t31 =  pui0i1 * puj0j1 * (2-gdi0i0-gdi1i1) * (delta_i0j1 - guj1i0) * gui1j0;
			// const num t32 = -pui0i1 * puj1j0 * (2-gdi0i0-gdi1i1) * (delta_i0j0 - guj0i0) * gui1j1;
			// const num t41 = -pui1i0 * puj0j1 * (2-gdi0i0-gdi1i1) * (delta_i1j1 - guj1i1) * gui0j0;
			// const num t42 =  pui1i0 * puj1j0 * (2-gdi0i0-gdi1i1) * (delta_i1j0 - guj0i1) * gui0j1;

			// const num t11 =  pdi0i1 * puj0j1 * (-gdi1i0) * ( (delta_i0j1 - guj1i0) * gui0j0 + (delta_i1j1 - guj1i1) * gui1j0 );
			// const num t21 = -pdi1i0 * puj0j1 * (-gdi0i1) * ( (delta_i0j1 - guj1i0) * gui0j0 + (delta_i1j1 - guj1i1) * gui1j0 );
			// const num t22 =  pdi1i0 * puj1j0 * (-gdi0i1) * ( (delta_i0j0 - guj0i0) * gui0j1 + (delta_i1j0 - guj0i1) * gui1j1 );
			// const num t12 = -pdi0i1 * puj1j0 * (-gdi1i0) * ( (delta_i0j0 - guj0i0) * gui0j1 + (delta_i1j0 - guj0i1) * gui1j1 );

			// const num t33 =  pui0i1 * pdj0j1 * (-gui1i0) * ( (delta_i0j1 - gdj1i0) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi1j0 );
			// const num t43 = -pui1i0 * pdj0j1 * (-gui0i1) * ( (delta_i0j1 - gdj1i0) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi1j0 );
			// const num t44 =  pui1i0 * pdj1j0 * (-gui0i1) * ( (delta_i0j0 - gdj0i0) * gdi0j1 + (delta_i1j0 - gdj0i1) * gdi1j1 );
			// const num t34 = -pui0i1 * pdj1j0 * (-gui1i0) * ( (delta_i0j0 - gdj0i0) * gdi0j1 + (delta_i1j0 - gdj0i1) * gdi1j1 );

			m->new_jnj[bb] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);

			//j(i0i1)jn(j0j1), 6 fermion product, 2 phases, t = 0
			_wick_j = - pui1i0*gui0i1 + pui0i1*gui1i0 - pdi1i0*gdi0i1 + pdi0i1*gdi1i0;
			_wick_jn = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
			 		   (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

			t5 = (2 - gdj0j0 - gdj1j1) * 
				(+pui0i1 * puj0j1 * (delta_i0j1 - guj1i0) * gui1j0
				 -pui0i1 * puj1j0 * (delta_i0j0 - guj0i0) * gui1j1
				 -pui1i0 * puj0j1 * (delta_i1j1 - guj1i1) * gui0j0
				 +pui1i0 * puj1j0 * (delta_i1j0 - guj0i1) * gui0j1);

			t6 = (2 - guj0j0 - guj1j1) * 
				(+pdi0i1 * pdj0j1 * (delta_i0j1 - gdj1i0) * gdi1j0
			     -pdi0i1 * pdj1j0 * (delta_i0j0 - gdj0i0) * gdi1j1
				 -pdi1i0 * pdj0j1 * (delta_i1j1 - gdj1i1) * gdi0j0
				 +pdi1i0 * pdj1j0 * (delta_i1j0 - gdj0i1) * gdi0j1);

			// t1 = ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 ) * pui0i1 * pdj0j1 * (gdj0j1 - gdj1j0);
			// t2 = ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 ) * pui1i0 * pdj1j0 * (gdj1j0 - gdj0j1);
			// t3 = ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 ) * pdi0i1 * puj0j1 * (guj0j1 - guj1j0);
			// t4 = ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 ) * pdi1i0 * puj1j0 * (guj1j0 - guj0j1);

			t1 = ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 ) * 
				pui0i1 * (pdj1j0 * gdj0j1 - pdj0j1 * gdj1j0);
			t2 = ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 ) * 
				pui1i0 * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1);
			t3 = ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 ) * 
				pdi0i1 * (puj1j0 * guj0j1 - puj0j1 * guj1j0);
			t4 = ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 ) * 
				pdi1i0 * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

			// const num t13 = pui0i1 * puj0j1 * (2 - gdj0j0 - gdj1j1) * (delta_i0j1 - guj1i0) * gui1j0;
			// const num t14 =-pui0i1 * puj1j0 * (2 - gdj0j0 - gdj1j1) * (delta_i0j0 - guj0i0) * gui1j1;
			// const num t23 =-pui1i0 * puj0j1 * (2 - gdj0j0 - gdj1j1) * (delta_i1j1 - guj1i1) * gui0j0;
			// const num t24 = pui1i0 * puj1j0 * (2 - gdj0j0 - gdj1j1) * (delta_i1j0 - guj0i1) * gui0j1;

			// const num t31 = pdi0i1 * pdj0j1 * (2 - guj0j0 - guj1j1) * (delta_i0j1 - gdj1i0) * gdi1j0;
			// const num t32 =-pdi0i1 * pdj1j0 * (2 - guj0j0 - guj1j1) * (delta_i0j0 - gdj0i0) * gdi1j1;
			// const num t41 =-pdi1i0 * pdj0j1 * (2 - guj0j0 - guj1j1) * (delta_i1j1 - gdj1i1) * gdi0j0;
			// const num t42 = pdi1i0 * pdj1j0 * (2 - guj0j0 - guj1j1) * (delta_i1j0 - gdj0i1) * gdi0j1;

			// const num t11 = pui0i1 * pdj0j1 * (-gdj1j0) * ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 );
			// const num t12 =-pui0i1 * pdj1j0 * (-gdj0j1) * ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 );
			// const num t22 = pui1i0 * pdj1j0 * (-gdj0j1) * ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 );
			// const num t21 =-pui1i0 * pdj0j1 * (-gdj1j0) * ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 );

			// const num t33 = pdi0i1 * puj0j1 * (-guj1j0) * ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 );
			// const num t34 =-pdi0i1 * puj1j0 * (-guj0j1) * ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 );
			// const num t44 = pdi1i0 * puj1j0 * (-guj0j1) * ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 );
			// const num t43 =-pdi1i0 * puj0j1 * (-guj1j0) * ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 );

			m->new_jjn[bb] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
	
			//TODO simplify this expression for faster measurements
			//There are 16 possible phase product combinations among pui0i1, pdi0i1, pui1i0, pdi1i0
			//												         puj0j1, pdj0j1, puj1j0, pdj1j0
			//TODO: declare these constant earlier.
			const num _wick_jn_i = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
			 			   		   (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);

			const num _wick_jn_j = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
			 		   			   (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);


			const num c1 = ( (delta_i0j0-guj0i0)*gui0j0 + (delta_i1j0-guj0i1)*gui1j0 
				           + (delta_i0j1-guj1i0)*gui0j1 + (delta_i1j1-guj1i1)*gui1j1 ) *
				( pdi0i1*pdj0j1*( gdi1i0*gdj1j0 + (delta_i0j1-gdj1i0)*gdi1j0 ) 
				 -pdi0i1*pdj1j0*( gdi1i0*gdj0j1 + (delta_i0j0-gdj0i0)*gdi1j1 ) 
				 -pdi1i0*pdj0j1*( gdi0i1*gdj1j0 + (delta_i1j1-gdj1i1)*gdi0j0 ) 
				 +pdi1i0*pdj1j0*( gdi0i1*gdj0j1 + (delta_i1j0-gdj0i1)*gdi0j1 ));

			const num c2 = (2-gui0i0-gui1i1) * (2-guj0j0-guj1j1) *
				( pdi0i1*pdj0j1 * (delta_i0j1-gdj1i0)*gdi1j0 
				 -pdi0i1*pdj1j0 * (delta_i0j0-gdj0i0)*gdi1j1 
				 -pdi1i0*pdj0j1 * (delta_i1j1-gdj1i1)*gdi0j0 
				 +pdi1i0*pdj1j0 * (delta_i1j0-gdj0i1)*gdi0j1);


			// const num t11 =+pdi0i1*pdj0j1 * ( (delta_i0j0-guj0i0)*gui0j0 + (delta_i1j0-guj0i1)*gui1j0 + (delta_i0j1-guj1i0)*gui0j1 + (delta_i1j1-guj1i1)*gui1j1 ) * 
			// 								( gdi1i0*gdj1j0 + (delta_i0j1-gdj1i0)*gdi1j0 ) 
			// 			   +pdi0i1*pdj0j1 * (2-gui0i0-gui1i1) * (2-guj0j0-guj1j1) * (delta_i0j1-gdj1i0)*gdi1j0;

			// const num t12 =-pdi0i1*pdj1j0 * ( (delta_i0j0-guj0i0)*gui0j0 + (delta_i1j0-guj0i1)*gui1j0 + (delta_i0j1-guj1i0)*gui0j1 + (delta_i1j1-guj1i1)*gui1j1 ) * 
			// 							    ( gdi1i0*gdj0j1 + (delta_i0j0-gdj0i0)*gdi1j1 ) 
			// 			   -pdi0i1*pdj1j0 * (2-gui0i0-gui1i1) * (2-guj0j0-guj1j1) * (delta_i0j0-gdj0i0)*gdi1j1;

			// const num t21 =-pdi1i0*pdj0j1 * ( (delta_i0j0-guj0i0)*gui0j0 + (delta_i1j0-guj0i1)*gui1j0 + (delta_i0j1-guj1i0)*gui0j1 + (delta_i1j1-guj1i1)*gui1j1 ) * 
			// 								( gdi0i1*gdj1j0 + (delta_i1j1-gdj1i1)*gdi0j0 ) 
			// 			   -pdi1i0*pdj0j1 * (2-gui0i0-gui1i1) * (2-guj0j0-guj1j1) * (delta_i1j1-gdj1i1)*gdi0j0;

			// const num t22 =+pdi1i0*pdj1j0 * ( (delta_i0j0-guj0i0)*gui0j0 + (delta_i1j0-guj0i1)*gui1j0 + (delta_i0j1-guj1i0)*gui0j1 + (delta_i1j1-guj1i1)*gui1j1 ) * 
			// 								( gdi0i1*gdj0j1 + (delta_i1j0-gdj0i1)*gdi0j1 ) 
			// 			   +pdi1i0*pdj1j0 * (2-gui0i0-gui1i1) * (2-guj0j0-guj1j1) * (delta_i1j0-gdj0i1)*gdi0j1;


			const num t13 =+pdi0i1*puj0j1 * ( (delta_i0j1-guj1i0)*gui0j0 + (delta_i1j1-guj1i1)*gui1j0 ) * 
			                                ( (-gdi1i0)*(2-gdj0j0-gdj1j1) +   (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 )
			               +pdi0i1*puj0j1 * (2-gui0i0-gui1i1) * (-guj1j0) * ( (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 );
			const num t14 =-pdi0i1*puj1j0 * ( (delta_i0j0-guj0i0)*gui0j1 + (delta_i1j0-guj0i1)*gui1j1 ) * 
										    ( (-gdi1i0)*(2-gdj0j0-gdj1j1) +   (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 )
						   -pdi0i1*puj1j0 * (2-gui0i0-gui1i1) * (-guj0j1) * ( (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 );
			const num t23 =-pdi1i0*puj0j1 * ( (delta_i0j1-guj1i0)*gui0j0 + (delta_i1j1-guj1i1)*gui1j0 ) * 
											( (-gdi0i1)*(2-gdj0j0-gdj1j1) +   (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 )
						   -pdi1i0*puj0j1 * (2-gui0i0-gui1i1) * (-guj1j0) * ( (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 );
			const num t24 =+pdi1i0*puj1j0 * ( (delta_i0j0-guj0i0)*gui0j1 + (delta_i1j0-guj0i1)*gui1j1 ) *
											( (-gdi0i1)*(2-gdj0j0-gdj1j1) +   (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 )
						   +pdi1i0*puj1j0 * (2-gui0i0-gui1i1) * (-guj0j1) * ( (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 );

			const num t31 =+pui0i1*pdj0j1 * ( (delta_i0j1-gdj1i0)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi1j0 ) * 
											( (-gui1i0)*(2-guj0j0-guj1j1) +   (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 )
						   +pui0i1*pdj0j1 * (2-gdi0i0-gdi1i1) * (-gdj1j0) * ( (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 );
			const num t32 =-pui0i1*pdj1j0 * ( (delta_i0j0-gdj0i0)*gdi0j1 + (delta_i1j0-gdj0i1)*gdi1j1 ) *
											( (-gui1i0)*(2-guj0j0-guj1j1) +   (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 )
						   -pui0i1*pdj1j0 * (2-gdi0i0-gdi1i1) * (-gdj0j1) * ( (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 );
			const num t41 =-pui1i0*pdj0j1 * ( (delta_i0j1-gdj1i0)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi1j0 ) * 
											( (-gui0i1)*(2-guj0j0-guj1j1) +   (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 )
						   -pui1i0*pdj0j1 * (2-gdi0i0-gdi1i1) * (-gdj1j0) * ( (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 );
			const num t42 =+pui1i0*pdj1j0 * ( (delta_i0j0-gdj0i0)*gdi0j1 + (delta_i1j0-gdj0i1)*gdi1j1 ) *
											( (-gui0i1)*(2-guj0j0-guj1j1) +   (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 )
						   +pui1i0*pdj1j0 * (2-gdi0i0-gdi1i1) * (-gdj0j1) * ( (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 );


			const num c3 = ( (delta_i0j0-gdj0i0)*gdi0j0 + (delta_i1j0-gdj0i1)*gdi1j0 
				           + (delta_i0j1-gdj1i0)*gdi0j1 + (delta_i1j1-gdj1i1)*gdi1j1 ) * 
				(+pui0i1*puj0j1 * ( gui1i0*guj1j0 + (delta_i0j1-guj1i0)*gui1j0 )
				 -pui0i1*puj1j0 * ( gui1i0*guj0j1 + (delta_i0j0-guj0i0)*gui1j1 )
				 -pui1i0*puj0j1 * ( gui0i1*guj1j0 + (delta_i1j1-guj1i1)*gui0j0 )
				 +pui1i0*puj1j0 * ( gui0i1*guj0j1 + (delta_i1j0-guj0i1)*gui0j1 ));

			const num c4 = (2-gdi0i0-gdi1i1) * (2-gdj0j0-gdj1j1) * 
				(+pui0i1*puj0j1 * (delta_i0j1-guj1i0)*gui1j0
				 -pui0i1*puj1j0 * (delta_i0j0-guj0i0)*gui1j1
				 -pui1i0*puj0j1 * (delta_i1j1-guj1i1)*gui0j0
				 +pui1i0*puj1j0 * (delta_i1j0-guj0i1)*gui0j1);

			// const num t33 =+pui0i1*puj0j1 * ( (delta_i0j0-gdj0i0)*gdi0j0 + (delta_i1j0-gdj0i1)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi0j1 + (delta_i1j1-gdj1i1)*gdi1j1 ) *
			// 				 				( gui1i0*guj1j0 + (delta_i0j1-guj1i0)*gui1j0 )
			// 			   +pui0i1*puj0j1 * (2-gdi0i0-gdi1i1) * (2-gdj0j0-gdj1j1) * (delta_i0j1-guj1i0)*gui1j0;

			// const num t34 =-pui0i1*puj1j0 * ( (delta_i0j0-gdj0i0)*gdi0j0 + (delta_i1j0-gdj0i1)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi0j1 + (delta_i1j1-gdj1i1)*gdi1j1 ) *
			// 								( gui1i0*guj0j1 + (delta_i0j0-guj0i0)*gui1j1 )
			// 			   -pui0i1*puj1j0 * (2-gdi0i0-gdi1i1) * (2-gdj0j0-gdj1j1) * (delta_i0j0-guj0i0)*gui1j1;

			// const num t43 =-pui1i0*puj0j1 * ( (delta_i0j0-gdj0i0)*gdi0j0 + (delta_i1j0-gdj0i1)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi0j1 + (delta_i1j1-gdj1i1)*gdi1j1 ) *
			// 								( gui0i1*guj1j0 + (delta_i1j1-guj1i1)*gui0j0 )
			// 			   -pui1i0*puj0j1 * (2-gdi0i0-gdi1i1) * (2-gdj0j0-gdj1j1) * (delta_i1j1-guj1i1)*gui0j0;

			// const num t44 =+pui1i0*puj1j0 * ( (delta_i0j0-gdj0i0)*gdi0j0 + (delta_i1j0-gdj0i1)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi0j1 + (delta_i1j1-gdj1i1)*gdi1j1 ) *
			// 								( gui0i1*guj0j1 + (delta_i1j0-guj0i1)*gui0j1 )
			// 			   +pui1i0*puj1j0 * (2-gdi0i0-gdi1i1) * (2-gdj0j0-gdj1j1) * (delta_i1j0-guj0i1)*gui0j1;





			const num sum1= _wick_jn_i * _wick_jn_j + c1 + c2 + t13 + t14
														      + t23 + t24
														+ t31 + t32 + c3 + c4
														+ t41 + t42;

			m->jnjn[bb] += pre*(_wick_jn_i * _wick_jn_j + c1 + c2 + t13 + t14
														          + t23 + t24
														+ t31 + t32 + c3 + c4
														+ t41 + t42);

// 			const num tAA = pdi0i1*pdj0j1 * (+1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0j1-gdj1i0)*gdi1j0)
// +1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0j1-gdj1i0)*gdi1j0)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0j1-gdj1i0)*gdi1j0)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0j1-gdj1i0)*gdi1j0));
// 			const num tAB = pdi0i1*pdj1j0 * (-1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0j0-gdj0i0)*gdi1j1)
// -1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0j0-gdj0i0)*gdi1j1)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0j0-gdj0i0)*gdi1j1)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0j0-gdj0i0)*gdi1j1));
// 			const num tAC = pdi0i1*puj0j1 * (+1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0j0-gdj0i0)*gdi1j0)
// +1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0j1-gdj1i0)*gdi1j1)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0j0-gdj0i0)*gdi1j0)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0j1-gdj1i0)*gdi1j1));
// 			const num tAD = pdi0i1*puj1j0 * (-1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0j0-gdj0i0)*gdi1j0)
// -1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0j1-gdj1i0)*gdi1j1)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0j0-gdj0i0)*gdi1j0)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0j1-gdj1i0)*gdi1j1));

// 			const num tBA = pdi1i0*pdj0j1 * (-1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi0j0)
// -1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j1-gdj1i1)*gdi0j0)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i1j1-gdj1i1)*gdi0j0)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i1j1-guj1i1)*gui1j1*(delta_i1j1-gdj1i1)*gdi0j0));
// 			const num tBB = pdi1i0*pdj1j0 * (+1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j0-gdj0i1)*gdi0j1)
// +1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi0j1)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i1j0-guj0i1)*gui1j0*(delta_i1j0-gdj0i1)*gdi0j1)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i1j0-gdj0i1)*gdi0j1));
// 			const num tBC = pdi1i0*puj0j1 * (-1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i1j0-gdj0i1)*gdi0j0)
// -1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i0j1-guj1i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi0j1)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i1j0-gdj0i1)*gdi0j0)
// -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i1j1-guj1i1)*gui1j0*(delta_i1j1-gdj1i1)*gdi0j1));
// 			const num tBD = pdi1i0*puj1j0 * (+1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i0j0-guj0i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi0j0)
// +1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i1j1-gdj1i1)*gdi0j1)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i1j0-guj0i1)*gui1j1*(delta_i1j0-gdj0i1)*gdi0j0)
// +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i1j1-gdj1i1)*gdi0j1));

// 			const num tCA = pui0i1*pdj0j1 * (+1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0j0-guj0i0)*gui1j0)
// +1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0j1-guj1i0)*gui1j1)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0j0-guj0i0)*gui1j0)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0j1-guj1i0)*gui1j1));
// 			const num tCB = pui0i1*pdj1j0 * (-1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0j0-guj0i0)*gui1j0)
// -1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0j1-guj1i0)*gui1j1)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0j0-guj0i0)*gui1j0)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0j1-guj1i0)*gui1j1));
// 			const num tCC = pui0i1*puj0j1 * (+1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0j1-guj1i0)*gui1j0)
// +1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0j1-guj1i0)*gui1j0)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0j1-guj1i0)*gui1j0)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0j1-guj1i0)*gui1j0));
// 			const num tCD = pui0i1*puj1j0 * (-1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0j0-guj0i0)*gui1j1)
// -1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0j0-guj0i0)*gui1j1)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0j0-guj0i0)*gui1j1)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0j0-guj0i0)*gui1j1));

// 			const num tDA = pui1i0*pdj0j1 * (-1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j0-guj0i1)*gui0j0)
// -1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j1-guj1i1)*gui0j1)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i1j0-guj0i1)*gui0j0)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i1j1-guj1i1)*gui0j1));
// 			const num tDB = pui1i0*pdj1j0 * (+1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j0-guj0i1)*gui0j0)
// +1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j1-guj1i1)*gui0j1)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i1j0-guj0i1)*gui0j0)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i1j1-guj1i1)*gui0j1));
// 			const num tDC = pui1i0*puj0j1 * (-1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-guj1i1)*gui0j0)
// -1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j1-guj1i1)*gui0j0)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i1j1-guj1i1)*gui0j0)
// -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i1j1-guj1i1)*gui0j0));
// 			const num tDD = pui1i0*puj1j0 * (+1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j0-guj0i1)*gui0j1)
// +1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-guj0i1)*gui0j1)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i1j0-guj0i1)*gui0j1)
// +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i1j0-guj0i1)*gui0j1));

// 			const num sum2 = tAA+tAB+tAC+tAD
// 							+tBA+tBB+tBC+tBD
// 							+tCA+tCB+tCC+tCD
// 							+tDA+tDB+tDC+tDD;
			// #ifdef USE_CPLX
			// printf("my sum = %f + %fi, edwin sum = %f + %fi \n ", creal(sum1),creal(sum1),creal(sum2),creal(sum2) );
			// #else
			// printf("my sum = %f, edwin sum = %f \n ", sum1, sum2 );
			// #endif
			//assert(approx_equal(sum1,sum2));
			//
			// fflush(stdout); 


			// m->jnjn[bb] += pre*(tAA+tAB+tAC+tAD
			// 				+tBA+tBB+tBC+tBD
			// 				+tCA+tCB+tCC+tCD
			// 				+tDA+tDB+tDC+tDD);
		}
	}
	}

	// measurement of j2-j2: 4 fermion product, 4 phases, t = 0
	// this is the ``clever'' way to do it
	// TODO: implement pair_b2b2,js2js2,k2k2,ks2ks2
	const int b2ps = num_b2/N;
	if (meas_2bond_corr)
	for (int c = 0; c < num_b2; c++) {
		const int jtype = c / N;
		const int j = c % N;
#ifdef USE_PEIERLS
		const num ppuj0j2 = p->pp_u[ j + N*jtype];
		const num ppuj2j0 = p->ppr_u[j + N*jtype];
		const num ppdj0j2 = p->pp_d[ j + N*jtype];
		const num ppdj2j0 = p->ppr_d[j + N*jtype];
#endif
		// printf("c = %d, jtype = %d,j = %d, ", c, jtype,j );
		// printf("pj0j1*pj1j2 = %f \n", (double) ppuj0j2);
		// fflush(stdout); 
		const int j0 = p->bond2s[c];
		const int j2 = p->bond2s[c + num_b2];
	for (int b = 0; b < num_b2; b++) {
		const int itype = b / N;
		const int i = b % N;
#ifdef USE_PEIERLS
		const num ppui0i2 = p->pp_u[ i + N*itype];
		const num ppui2i0 = p->ppr_u[i + N*itype];
		const num ppdi0i2 = p->pp_d[ i + N*itype];
		const num ppdi2i0 = p->ppr_d[i + N*itype];
#endif
		const int i0 = p->bond2s[b];
		const int i2 = p->bond2s[b + num_b2];

		const int bb = p->map_b2b2[b + c*num_b2];
		const num pre = phase / p->degen_b2b2[bb];

		const int delta_i0j0 = (i0 == j0);
		const int delta_i2j0 = (i2 == j0);
		const int delta_i0j2 = (i0 == j2);
		const int delta_i2j2 = (i2 == j2);
		// const int delta_i0i1 = 0;
		// const int delta_j0j1 = 0;
		const num gui2i0 = Gu00[i2 + i0*N];
		const num gui0i2 = Gu00[i0 + i2*N];
		const num gui0j0 = Gu00[i0 + j0*N];
		const num gui2j0 = Gu00[i2 + j0*N];
		const num gui0j2 = Gu00[i0 + j2*N];
		const num gui2j2 = Gu00[i2 + j2*N];
		const num guj0i0 = Gu00[j0 + i0*N];
		const num guj2i0 = Gu00[j2 + i0*N];
		const num guj0i2 = Gu00[j0 + i2*N];
		const num guj2i2 = Gu00[j2 + i2*N];
		const num guj2j0 = Gu00[j2 + j0*N];
		const num guj0j2 = Gu00[j0 + j2*N];
		const num gdi2i0 = Gd00[i2 + i0*N];
		const num gdi0i2 = Gd00[i0 + i2*N];
		const num gdi0j0 = Gd00[i0 + j0*N];
		const num gdi2j0 = Gd00[i2 + j0*N];
		const num gdi0j2 = Gd00[i0 + j2*N];
		const num gdi2j2 = Gd00[i2 + j2*N];
		const num gdj0i0 = Gd00[j0 + i0*N];
		const num gdj2i0 = Gd00[j2 + i0*N];
		const num gdj0i2 = Gd00[j0 + i2*N];
		const num gdj2i2 = Gd00[j2 + i2*N];
		const num gdj2j0 = Gd00[j2 + j0*N];
		const num gdj0j2 = Gd00[j0 + j2*N];

		const num x = ppui0i2*ppuj0j2*(delta_i0j2 - guj2i0)*gui2j0 +
					  ppui2i0*ppuj2j0*(delta_i2j0 - guj0i2)*gui0j2 +
					  ppdi0i2*ppdj0j2*(delta_i0j2 - gdj2i0)*gdi2j0 +
					  ppdi2i0*ppdj2j0*(delta_i2j0 - gdj0i2)*gdi0j2;
		const num y = ppui0i2*ppuj2j0*(delta_i0j0 - guj0i0)*gui2j2 +
		              ppui2i0*ppuj0j2*(delta_i2j2 - guj2i2)*gui0j0 +
		              ppdi0i2*ppdj2j0*(delta_i0j0 - gdj0i0)*gdi2j2 +
		              ppdi2i0*ppdj0j2*(delta_i2j2 - gdj2i2)*gdi0j0;
		m->j2j2[bb] += pre*((ppui2i0*gui0i2 - ppui0i2*gui2i0 + ppdi2i0*gdi0i2 - ppdi0i2*gdi2i0)
		                   *(ppuj2j0*guj0j2 - ppuj0j2*guj2j0 + ppdj2j0*gdj0j2 - ppdj0j2*gdj2j0) 
		                   + x - y);

	}
	}

// 	// measurement of J2-J2: 4 fermion product, 4 phases, t = 0
// 	if (meas_hop2_corr)
// 	for (int c = 0; c < num_hop2; c++) {
// 		const int j0 = p->hop2s[c];
// 		const int j1 = p->hop2s[c + num_hop2];
// 		const int j2 = p->hop2s[c + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num puj0j1 = p->peierlsu[j0 + N*j1];
// 		const num puj1j0 = p->peierlsu[j1 + N*j0];
// 		const num pdj0j1 = p->peierlsd[j0 + N*j1];
// 		const num pdj1j0 = p->peierlsd[j1 + N*j0];
// 		const num puj1j2 = p->peierlsu[j1 + N*j2];
// 		const num puj2j1 = p->peierlsu[j2 + N*j1];
// 		const num pdj1j2 = p->peierlsd[j1 + N*j2];
// 		const num pdj2j1 = p->peierlsd[j2 + N*j1];
// #endif
// 	for (int b = 0; b < num_hop2; b++) {
// 		const int i0 = p->hop2s[b];
// 		const int i1 = p->hop2s[b + num_hop2];
// 		const int i2 = p->hop2s[b + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num pui0i1 = p->peierlsu[i0 + N*i1];
// 		const num pui1i0 = p->peierlsu[i1 + N*i0];
// 		const num pdi0i1 = p->peierlsd[i0 + N*i1];
// 		const num pdi1i0 = p->peierlsd[i1 + N*i0];
// 		const num pui1i2 = p->peierlsu[i1 + N*i2];
// 		const num pui2i1 = p->peierlsu[i2 + N*i1];
// 		const num pdi1i2 = p->peierlsd[i1 + N*i2];
// 		const num pdi2i1 = p->peierlsd[i2 + N*i1];
// #endif
// 		const int bb = p->map_hop2_hop2[b + c*num_hop2];
// 		const num pre = phase / p->degen_hop2_hop2[bb];
// 		const int delta_i0j0 = (i0 == j0);
// 		const int delta_i2j0 = (i2 == j0);
// 		const int delta_i0j2 = (i0 == j2);
// 		const int delta_i2j2 = (i2 == j2);
// 		// const int delta_i0i1 = 0;
// 		// const int delta_j0j1 = 0;
// 		const num gui2i0 = Gu00[i2 + i0*N];
// 		const num gui0i2 = Gu00[i0 + i2*N];
// 		const num gui0j0 = Gu00[i0 + j0*N];
// 		const num gui2j0 = Gu00[i2 + j0*N];
// 		const num gui0j2 = Gu00[i0 + j2*N];
// 		const num gui2j2 = Gu00[i2 + j2*N];
// 		const num guj0i0 = Gu00[j0 + i0*N];
// 		const num guj2i0 = Gu00[j2 + i0*N];
// 		const num guj0i2 = Gu00[j0 + i2*N];
// 		const num guj2i2 = Gu00[j2 + i2*N];
// 		const num guj2j0 = Gu00[j2 + j0*N];
// 		const num guj0j2 = Gu00[j0 + j2*N];
// 		const num gdi2i0 = Gd00[i2 + i0*N];
// 		const num gdi0i2 = Gd00[i0 + i2*N];
// 		const num gdi0j0 = Gd00[i0 + j0*N];
// 		const num gdi2j0 = Gd00[i2 + j0*N];
// 		const num gdi0j2 = Gd00[i0 + j2*N];
// 		const num gdi2j2 = Gd00[i2 + j2*N];
// 		const num gdj0i0 = Gd00[j0 + i0*N];
// 		const num gdj2i0 = Gd00[j2 + i0*N];
// 		const num gdj0i2 = Gd00[j0 + i2*N];
// 		const num gdj2i2 = Gd00[j2 + i2*N];
// 		const num gdj2j0 = Gd00[j2 + j0*N];
// 		const num gdj0j2 = Gd00[j0 + j2*N];

// 		const num x = pui0i1*pui1i2*puj0j1*puj1j2*(delta_i0j2 - guj2i0)*gui2j0 +
// 					  pui1i0*pui2i1*puj1j0*puj2j1*(delta_i2j0 - guj0i2)*gui0j2 +
// 					  pdi0i1*pdi1i2*pdj0j1*pdj1j2*(delta_i0j2 - gdj2i0)*gdi2j0 +
// 					  pdi1i0*pdi2i1*pdj1j0*pdj2j1*(delta_i2j0 - gdj0i2)*gdi0j2;
// 		const num y = pui0i1*pui1i2*puj1j0*puj2j1*(delta_i0j0 - guj0i0)*gui2j2 +
// 		              pui1i0*pui2i1*puj0j1*puj1j2*(delta_i2j2 - guj2i2)*gui0j0 +
// 		              pdi0i1*pdi1i2*pdj1j0*pdj2j1*(delta_i0j0 - gdj0i0)*gdi2j2 +
// 		              pdi1i0*pdi2i1*pdj0j1*pdj1j2*(delta_i2j2 - gdj2i2)*gdi0j0;
// 		m->J2J2[bb] += pre*((pui1i0*pui2i1*gui0i2 - pui0i1*pui1i2*gui2i0 + pdi1i0*pdi2i1*gdi0i2 - pdi0i1*pdi1i2*gdi2i0)
// 		                   *(puj1j0*puj2j1*guj0j2 - puj0j1*puj1j2*guj2j0 + pdj1j0*pdj2j1*gdj0j2 - pdj0j1*pdj1j2*gdj2j0) 
// 		                   + x - y);

// 	}
// 	}


	// measurement of jn(i0i1)-j2(j0j1j2): 6 fermion product, 3 phases, t = 0
	//                j(i0i1) -j2(j0j1j2): 4 fermion product, 3 phases, t = 0
	// i = i0 <-> i1
	// j = j0 <-> j1 <-> j2
	// Essentially matrix[j,i] = bond(i) x bond2(j)
	// This is "clever" way to do it
	if (meas_thermal || meas_2bond_corr) 
	for (int c = 0; c < num_b2; c++) {
		const int jtype = c / N;
		const int j = c % N;
#ifdef USE_PEIERLS
		const num ppuj0j2 = p->pp_u[ j + N*jtype];
		const num ppuj2j0 = p->ppr_u[j + N*jtype];
		const num ppdj0j2 = p->pp_d[ j + N*jtype];
		const num ppdj2j0 = p->ppr_d[j + N*jtype];
#endif
		const int j0 = p->bond2s[c];
		const int j2 = p->bond2s[c + num_b2];
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_b2b[b + c*num_b];
		const num pre = phase / p->degen_b2b[bb];

		const int delta_i0j0 = (i0 == j0);
		const int delta_i1j0 = (i1 == j0);
		const int delta_i0j2 = (i0 == j2);
		const int delta_i1j2 = (i1 == j2);

		const num gui1i0 = Gu00[i1 + i0*N];
		const num gui0i1 = Gu00[i0 + i1*N];
		const num gui0j0 = Gu00[i0 + j0*N];
		const num gui1j0 = Gu00[i1 + j0*N];
		const num gui0j2 = Gu00[i0 + j2*N];
		const num gui1j2 = Gu00[i1 + j2*N];
		const num guj0i0 = Gu00[j0 + i0*N];
		const num guj2i0 = Gu00[j2 + i0*N];
		const num guj0i1 = Gu00[j0 + i1*N];
		const num guj2i1 = Gu00[j2 + i1*N];
		const num guj2j0 = Gu00[j2 + j0*N];
		const num guj0j2 = Gu00[j0 + j2*N];
		const num gdi1i0 = Gd00[i1 + i0*N];
		const num gdi0i1 = Gd00[i0 + i1*N];
		const num gdi0j0 = Gd00[i0 + j0*N];
		const num gdi1j0 = Gd00[i1 + j0*N];
		const num gdi0j2 = Gd00[i0 + j2*N];
		const num gdi1j2 = Gd00[i1 + j2*N];
		const num gdj0i0 = Gd00[j0 + i0*N];
		const num gdj2i0 = Gd00[j2 + i0*N];
		const num gdj0i1 = Gd00[j0 + i1*N];
		const num gdj2i1 = Gd00[j2 + i1*N];
		const num gdj2j0 = Gd00[j2 + j0*N];
		const num gdj0j2 = Gd00[j0 + j2*N];

		if (meas_thermal){
			const num gui0i0 = Gu00[i0 + i0*N];
			const num gui1i1 = Gu00[i1 + i1*N];
			const num gdi0i0 = Gd00[i0 + i0*N];
			const num gdi1i1 = Gd00[i1 + i1*N];

			//jn(i0i1)-j2(j0j1j2): 6 fermion product, 3 phases, t = 0
			//TODO: further group these expressions together?
			const num _wick_jn = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
			 			         (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			const num _wick_j = - ppuj2j0 * guj0j2 + ppuj0j2 * guj2j0 
						        - ppdj2j0 * gdj0j2 + ppdj0j2 * gdj2j0;

			const num t1 = ( (delta_i0j2 - guj2i0) * gui0j0 + (delta_i1j2 - guj2i1) * gui1j0 ) * 
				ppuj0j2 * (pdi1i0 * gdi0i1 - pdi0i1 * gdi1i0);
			const num t2 = ( (delta_i0j0 - guj0i0) * gui0j2 + (delta_i1j0 - guj0i1) * gui1j2 ) * 
				ppuj2j0 * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1);
			const num t3 = ( (delta_i0j2 - gdj2i0) * gdi0j0 + (delta_i1j2 - gdj2i1) * gdi1j0 ) * 
				ppdj0j2 * (pui1i0 * gui0i1 - pui0i1 * gui1i0);
			const num t4 = ( (delta_i0j0 - gdj0i0) * gdi0j2 + (delta_i1j0 - gdj0i1) * gdi1j2 ) * 
				ppdj2j0 * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			const num t5 = (2 - gui0i0 - gui1i1) * 
				(+pdi0i1 * ppdj0j2 * (delta_i0j2 - gdj2i0) * gdi1j0 
				 -pdi0i1 * ppdj2j0 * (delta_i0j0 - gdj0i0) * gdi1j2
				 -pdi1i0 * ppdj0j2 * (delta_i1j2 - gdj2i1) * gdi0j0
				 +pdi1i0 * ppdj2j0 * (delta_i1j0 - gdj0i1) * gdi0j2);
			const num t6 = (2 - gdi0i0 - gdi1i1) *
				(+pui0i1 * ppuj0j2 * (delta_i0j2 - guj2i0) * gui1j0
				 -pui0i1 * ppuj2j0 * (delta_i0j0 - guj0i0) * gui1j2
				 -pui1i0 * ppuj0j2 * (delta_i1j2 - guj2i1) * gui0j0
				 +pui1i0 * ppuj2j0 * (delta_i1j0 - guj0i1) * gui0j2);

			m->jnj2[bb]   += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
		}
		if (meas_2bond_corr) {
			//j(i0i1) -j2(j0j1j2): 4 fermion product, 3 phases, t = 0
			const num x = pui0i1 * ppuj0j2 * (delta_i0j2 - guj2i0)*gui1j0 +
						  pui1i0 * ppuj2j0 * (delta_i1j0 - guj0i1)*gui0j2 +
						  pdi0i1 * ppdj0j2 * (delta_i0j2 - gdj2i0)*gdi1j0 +
						  pdi1i0 * ppdj2j0 * (delta_i1j0 - gdj0i1)*gdi0j2;
			const num y = pui0i1 * ppuj2j0 * (delta_i0j0 - guj0i0)*gui1j2 +
			              pui1i0 * ppuj0j2 * (delta_i1j2 - guj2i1)*gui0j0 +
			              pdi0i1 * ppdj2j0 * (delta_i0j0 - gdj0i0)*gdi1j2 +
			              pdi1i0 * ppdj0j2 * (delta_i1j2 - gdj2i1)*gdi0j0;
			m->jj2[bb]  += pre*((pui1i0 * gui0i1  - pui0i1 * gui1i0  + pdi1i0 * gdi0i1  - pdi0i1 * gdi1i0)
			                   *(ppuj2j0 * guj0j2 - ppuj0j2 * guj2j0 + ppdj2j0 * gdj0j2 - ppdj0j2 * gdj2j0) 
			                   + x - y);
		}
	}
	}

	// measurement of jn(i0i1)-J2(j0j1j2): 6 fermion product, 3 phases, t = 0
	//                j(i0i1) -J2(j0j1j2): 4 fermion product, 3 phases, t = 0
	// i = i0 <-> i1
	// j = j0 <-> j1 <-> j2
	// Essentially matrix[j,i] = bond(i) x hop2(j)
// 	if (meas_hop2_corr) 
// 	for (int c = 0; c < num_hop2; c++) {
// 		const int j0 = p->hop2s[c];
// 		const int j1 = p->hop2s[c + num_hop2];
// 		const int j2 = p->hop2s[c + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num puj0j1 = p->peierlsu[j0 + N*j1];
// 		const num puj1j0 = p->peierlsu[j1 + N*j0];
// 		const num pdj0j1 = p->peierlsd[j0 + N*j1];
// 		const num pdj1j0 = p->peierlsd[j1 + N*j0];
// 		const num puj1j2 = p->peierlsu[j1 + N*j2];
// 		const num puj2j1 = p->peierlsu[j2 + N*j1];
// 		const num pdj1j2 = p->peierlsd[j1 + N*j2];
// 		const num pdj2j1 = p->peierlsd[j2 + N*j1];
// #endif
// 	for (int b = 0; b < num_b; b++) {
// 		const int i0 = p->bonds[b];
// 		const int i1 = p->bonds[b + num_b];
// #ifdef USE_PEIERLS
// 		const num pui0i1 = p->peierlsu[i0 + N*i1];
// 		const num pui1i0 = p->peierlsu[i1 + N*i0];
// 		const num pdi0i1 = p->peierlsd[i0 + N*i1];
// 		const num pdi1i0 = p->peierlsd[i1 + N*i0];
// #endif
// 		const int bb = p->map_b_hop2[b + c*num_b];
// 		const num pre = phase / p->degen_b_hop2[bb];
// 		const int delta_i0j0 = (i0 == j0);
// 		const int delta_i1j0 = (i1 == j0);
// 		const int delta_i0j2 = (i0 == j2);
// 		const int delta_i1j2 = (i1 == j2);
// 		const num gui1i0 = Gu00[i1 + i0*N];
// 		const num gui0i1 = Gu00[i0 + i1*N];
// 		const num gui0j0 = Gu00[i0 + j0*N];
// 		const num gui1j0 = Gu00[i1 + j0*N];
// 		const num gui0j2 = Gu00[i0 + j2*N];
// 		const num gui1j2 = Gu00[i1 + j2*N];
// 		const num guj0i0 = Gu00[j0 + i0*N];
// 		const num guj2i0 = Gu00[j2 + i0*N];
// 		const num guj0i1 = Gu00[j0 + i1*N];
// 		const num guj2i1 = Gu00[j2 + i1*N];
// 		const num guj2j0 = Gu00[j2 + j0*N];
// 		const num guj0j2 = Gu00[j0 + j2*N];
// 		const num gdi1i0 = Gd00[i1 + i0*N];
// 		const num gdi0i1 = Gd00[i0 + i1*N];
// 		const num gdi0j0 = Gd00[i0 + j0*N];
// 		const num gdi1j0 = Gd00[i1 + j0*N];
// 		const num gdi0j2 = Gd00[i0 + j2*N];
// 		const num gdi1j2 = Gd00[i1 + j2*N];
// 		const num gdj0i0 = Gd00[j0 + i0*N];
// 		const num gdj2i0 = Gd00[j2 + i0*N];
// 		const num gdj0i1 = Gd00[j0 + i1*N];
// 		const num gdj2i1 = Gd00[j2 + i1*N];
// 		const num gdj2j0 = Gd00[j2 + j0*N];
// 		const num gdj0j2 = Gd00[j0 + j2*N];

// 		const num gui0i0 = Gu00[i0 + i0*N];
// 		const num gui1i1 = Gu00[i1 + i1*N];
// 		const num gdi0i0 = Gd00[i0 + i0*N];
// 		const num gdi1i1 = Gd00[i1 + i1*N];

// 		//jn(i0i1)-J2(j0j1j2): 6 fermion product, 3 phases, t = 0
// 		//TODO: further group these expressions together?
// 		const num _wick_jn = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
// 		 			         (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
// 		const num _wick_j = - puj1j0*puj2j1*guj0j2 + puj0j1*puj1j2*guj2j0 
// 					        - pdj1j0*pdj2j1*gdj0j2 + pdj0j1*pdj1j2*gdj2j0;

// 		const num t1 = ( (delta_i0j2 - guj2i0) * gui0j0 + (delta_i1j2 - guj2i1) * gui1j0 ) * 
// 			puj0j1 * puj1j2 * (pdi1i0 * gdi0i1 - pdi0i1 * gdi1i0);
// 		const num t2 = ( (delta_i0j0 - guj0i0) * gui0j2 + (delta_i1j0 - guj0i1) * gui1j2 ) * 
// 			puj1j0 * puj2j1 * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1);
// 		const num t3 = ( (delta_i0j2 - gdj2i0) * gdi0j0 + (delta_i1j2 - gdj2i1) * gdi1j0 ) * 
// 			pdj0j1 * pdj1j2 * (pui1i0 * gui0i1 - pui0i1 * gui1i0);
// 		const num t4 = ( (delta_i0j0 - gdj0i0) * gdi0j2 + (delta_i1j0 - gdj0i1) * gdi1j2 ) * 
// 			pdj1j0 * pdj2j1 * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
// 		const num t5 = (2 - gui0i0 - gui1i1) * 
// 			(+pdi0i1 * pdj0j1 * pdj1j2 * (delta_i0j2 - gdj2i0) * gdi1j0 
// 			 -pdi0i1 * pdj1j0 * pdj2j1 * (delta_i0j0 - gdj0i0) * gdi1j2
// 			 -pdi1i0 * pdj0j1 * pdj1j2 * (delta_i1j2 - gdj2i1) * gdi0j0
// 			 +pdi1i0 * pdj1j0 * pdj2j1 * (delta_i1j0 - gdj0i1) * gdi0j2);
// 		const num t6 = (2 - gdi0i0 - gdi1i1) *
// 			(+pui0i1 * puj0j1 * puj1j2 * (delta_i0j2 - guj2i0) * gui1j0
// 			 -pui0i1 * puj1j0 * puj2j1 * (delta_i0j0 - guj0i0) * gui1j2
// 			 -pui1i0 * puj0j1 * puj1j2 * (delta_i1j2 - guj2i1) * gui0j0
// 			 +pui1i0 * puj1j0 * puj2j1 * (delta_i1j0 - guj0i1) * gui0j2);

// 		m->jnJ2[bb]   += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
// 		//j(i0i1) -J2(j0j1j2): 4 fermion product, 3 phases, t = 0
// 		const num x = pui0i1*puj0j1*puj1j2*(delta_i0j2 - guj2i0)*gui1j0 +
// 					  pui1i0*puj1j0*puj2j1*(delta_i1j0 - guj0i1)*gui0j2 +
// 					  pdi0i1*pdj0j1*pdj1j2*(delta_i0j2 - gdj2i0)*gdi1j0 +
// 					  pdi1i0*pdj1j0*pdj2j1*(delta_i1j0 - gdj0i1)*gdi0j2;
// 		const num y = pui0i1*puj1j0*puj2j1*(delta_i0j0 - guj0i0)*gui1j2 +
// 		              pui1i0*puj0j1*puj1j2*(delta_i1j2 - guj2i1)*gui0j0 +
// 		              pdi0i1*pdj1j0*pdj2j1*(delta_i0j0 - gdj0i0)*gdi1j2 +
// 		              pdi1i0*pdj0j1*pdj1j2*(delta_i1j2 - gdj2i1)*gdi0j0;
// 		m->jJ2[bb]  += pre*((pui1i0*gui0i1        - pui0i1*gui1i0        + pdi1i0*gdi0i1        - pdi0i1*gdi1i0)
// 		                   *(puj1j0*puj2j1*guj0j2 - puj0j1*puj1j2*guj2j0 + pdj1j0*pdj2j1*gdj0j2 - pdj0j1*pdj1j2*gdj2j0) 
// 		                   + x - y);
// 	}
// 	}

    // measurement of j2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t = 0
	//                j2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t = 0
	// i = i0 <-> i1 <-> i2
	// j = j0 <-> j1 Is this the correct indexing?
	// Essentially matrix[j,i] = bond2(i) x bond(j)
	if (meas_thermal || meas_2bond_corr) 
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b2; b++) {
		const int itype = b / N;
		const int i = b % N;
#ifdef USE_PEIERLS
		const num ppui0i2 = p->pp_u[ i + N*itype];
		const num ppui2i0 = p->ppr_u[i + N*itype];
		const num ppdi0i2 = p->pp_d[ i + N*itype];
		const num ppdi2i0 = p->ppr_d[i + N*itype];
#endif
		const int i0 = p->bond2s[b];
		const int i2 = p->bond2s[b + num_b2];

		const int bb = p->map_bb2[b + c*num_b2];
		const num pre = phase / p->degen_bb2[bb];

		const int delta_i0j0 = (i0 == j0);
		const int delta_i2j0 = (i2 == j0);
		const int delta_i0j1 = (i0 == j1);
		const int delta_i2j1 = (i2 == j1);

		const num gui2i0 = Gu00[i2 + i0*N];
		const num gui0i2 = Gu00[i0 + i2*N];
		const num gui0j0 = Gu00[i0 + j0*N];
		const num gui2j0 = Gu00[i2 + j0*N];
		const num gui0j1 = Gu00[i0 + j1*N];
		const num gui2j1 = Gu00[i2 + j1*N];
		const num guj0i0 = Gu00[j0 + i0*N];
		const num guj1i0 = Gu00[j1 + i0*N];
		const num guj0i2 = Gu00[j0 + i2*N];
		const num guj1i2 = Gu00[j1 + i2*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num gdi2i0 = Gd00[i2 + i0*N];
		const num gdi0i2 = Gd00[i0 + i2*N];
		const num gdi0j0 = Gd00[i0 + j0*N];
		const num gdi2j0 = Gd00[i2 + j0*N];
		const num gdi0j1 = Gd00[i0 + j1*N];
		const num gdi2j1 = Gd00[i2 + j1*N];
		const num gdj0i0 = Gd00[j0 + i0*N];
		const num gdj1i0 = Gd00[j1 + i0*N];
		const num gdj0i2 = Gd00[j0 + i2*N];
		const num gdj1i2 = Gd00[j1 + i2*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		if (meas_thermal) {
			const num guj0j0 = Gu00[j0 + j0*N];
			const num guj1j1 = Gu00[j1 + j1*N];
			const num gdj0j0 = Gd00[j0 + j0*N];
			const num gdj1j1 = Gd00[j1 + j1*N];

			//j2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t = 0
			const num _wick_j = - ppui2i0 * gui0i2 + ppui0i2 * gui2i0 
			                    - ppdi2i0 * gdi0i2 + ppdi0i2 * gdi2i0;
			const num _wick_jn = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
			 		             (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

			const num t5 = (2 - gdj0j0 - gdj1j1) * 
				(+ppui0i2 * puj0j1 * (delta_i0j1 - guj1i0) * gui2j0
				 -ppui0i2 * puj1j0 * (delta_i0j0 - guj0i0) * gui2j1
				 -ppui2i0 * puj0j1 * (delta_i2j1 - guj1i2) * gui0j0
				 +ppui2i0 * puj1j0 * (delta_i2j0 - guj0i2) * gui0j1);

			const num t6 = (2 - guj0j0 - guj1j1) * 
				(+ppdi0i2 * pdj0j1 * (delta_i0j1 - gdj1i0) * gdi2j0
			     -ppdi0i2 * pdj1j0 * (delta_i0j0 - gdj0i0) * gdi2j1
				 -ppdi2i0 * pdj0j1 * (delta_i2j1 - gdj1i2) * gdi0j0
				 +ppdi2i0 * pdj1j0 * (delta_i2j0 - gdj0i2) * gdi0j1);

			const num t1 = ( (delta_i0j0 - guj0i0) * gui2j0 + (delta_i0j1 - guj1i0) * gui2j1 ) * 
				ppui0i2 * (pdj1j0 * gdj0j1 - pdj0j1 * gdj1j0);
			const num t2 = ( (delta_i2j0 - guj0i2) * gui0j0 + (delta_i2j1 - guj1i2) * gui0j1 ) * 
				ppui2i0 * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1);
			const num t3 = ( (delta_i0j0 - gdj0i0) * gdi2j0 + (delta_i0j1 - gdj1i0) * gdi2j1 ) * 
				ppdi0i2 * (puj1j0 * guj0j1 - puj0j1 * guj1j0);
			const num t4 = ( (delta_i2j0 - gdj0i2) * gdi0j0 + (delta_i2j1 - gdj1i2) * gdi0j1 ) * 
				ppdi2i0 * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

			m->j2jn[bb] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
		}
		if (meas_2bond_corr) {
			//j2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t = 0
			const num x = ppui0i2 * puj0j1*(delta_i0j1 - guj1i0)*gui2j0 +
						  ppui2i0 * puj1j0*(delta_i2j0 - guj0i2)*gui0j1 +
						  ppdi0i2 * pdj0j1*(delta_i0j1 - gdj1i0)*gdi2j0 +
						  ppdi2i0 * pdj1j0*(delta_i2j0 - gdj0i2)*gdi0j1;
			const num y = ppui0i2 * puj1j0*(delta_i0j0 - guj0i0)*gui2j1 +
			              ppui2i0 * puj0j1*(delta_i2j1 - guj1i2)*gui0j0 +
			              ppdi0i2 * pdj1j0*(delta_i0j0 - gdj0i0)*gdi2j1 +
			              ppdi2i0 * pdj0j1*(delta_i2j1 - gdj1i2)*gdi0j0;
			m->j2j[bb]  += pre*((ppui2i0 * gui0i2 - ppui0i2 * gui2i0 + ppdi2i0 * gdi0i2 - ppdi0i2 * gdi2i0)
			                   *( puj1j0 * guj0j1 -  puj0j1 * guj1j0 +  pdj1j0 * gdj0j1 -  pdj0j1 * gdj1j0) 
			                   + x - y);
		}
	}
	}


	// measurement of J2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t = 0
	//                J2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t = 0
	// i = i0 <-> i1 <-> i2
	// j = j0 <-> j1 Is this the correct indexing?
	// Essentially matrix[j,i] = hop2(i) x bond(j)
// 	if (meas_hop2_corr) 
// 	for (int c = 0; c < num_b; c++) {
// 		const int j0 = p->bonds[c];
// 		const int j1 = p->bonds[c + num_b];
// #ifdef USE_PEIERLS
// 		const num puj0j1 = p->peierlsu[j0 + N*j1];
// 		const num puj1j0 = p->peierlsu[j1 + N*j0];
// 		const num pdj0j1 = p->peierlsd[j0 + N*j1];
// 		const num pdj1j0 = p->peierlsd[j1 + N*j0];
// #endif
// 	for (int b = 0; b < num_hop2; b++) {
// 		const int i0 = p->hop2s[b];
// 		const int i1 = p->hop2s[b + num_hop2];
// 		const int i2 = p->hop2s[b + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num pui0i1 = p->peierlsu[i0 + N*i1];
// 		const num pui1i0 = p->peierlsu[i1 + N*i0];
// 		const num pdi0i1 = p->peierlsd[i0 + N*i1];
// 		const num pdi1i0 = p->peierlsd[i1 + N*i0];
// 		const num pui1i2 = p->peierlsu[i1 + N*i2];
// 		const num pui2i1 = p->peierlsu[i2 + N*i1];
// 		const num pdi1i2 = p->peierlsd[i1 + N*i2];
// 		const num pdi2i1 = p->peierlsd[i2 + N*i1];
// #endif
// 		const int bb = p->map_hop2_b[b + c*num_hop2];
// 		const num pre = phase / p->degen_hop2_b[bb];
// 		const int delta_i0j0 = (i0 == j0);
// 		const int delta_i2j0 = (i2 == j0);
// 		const int delta_i0j1 = (i0 == j1);
// 		const int delta_i2j1 = (i2 == j1);
// 		const num gui2i0 = Gu00[i2 + i0*N];
// 		const num gui0i2 = Gu00[i0 + i2*N];
// 		const num gui0j0 = Gu00[i0 + j0*N];
// 		const num gui2j0 = Gu00[i2 + j0*N];
// 		const num gui0j1 = Gu00[i0 + j1*N];
// 		const num gui2j1 = Gu00[i2 + j1*N];
// 		const num guj0i0 = Gu00[j0 + i0*N];
// 		const num guj1i0 = Gu00[j1 + i0*N];
// 		const num guj0i2 = Gu00[j0 + i2*N];
// 		const num guj1i2 = Gu00[j1 + i2*N];
// 		const num guj1j0 = Gu00[j1 + j0*N];
// 		const num guj0j1 = Gu00[j0 + j1*N];
// 		const num gdi2i0 = Gd00[i2 + i0*N];
// 		const num gdi0i2 = Gd00[i0 + i2*N];
// 		const num gdi0j0 = Gd00[i0 + j0*N];
// 		const num gdi2j0 = Gd00[i2 + j0*N];
// 		const num gdi0j1 = Gd00[i0 + j1*N];
// 		const num gdi2j1 = Gd00[i2 + j1*N];
// 		const num gdj0i0 = Gd00[j0 + i0*N];
// 		const num gdj1i0 = Gd00[j1 + i0*N];
// 		const num gdj0i2 = Gd00[j0 + i2*N];
// 		const num gdj1i2 = Gd00[j1 + i2*N];
// 		const num gdj1j0 = Gd00[j1 + j0*N];
// 		const num gdj0j1 = Gd00[j0 + j1*N];

// 		const num guj0j0 = Gu00[j0 + j0*N];
// 		const num guj1j1 = Gu00[j1 + j1*N];
// 		const num gdj0j0 = Gd00[j0 + j0*N];
// 		const num gdj1j1 = Gd00[j1 + j1*N];

// 		//J2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t = 0
// 		const num _wick_j = - pui1i0*pui2i1*gui0i2 + pui0i1*pui1i2*gui2i0 
// 		                    - pdi1i0*pdi2i1*gdi0i2 + pdi0i1*pdi1i2*gdi2i0;
// 		const num _wick_jn = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
// 		 		             (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

// 		const num t5 = (2 - gdj0j0 - gdj1j1) * 
// 			(+pui0i1 * pui1i2 *  puj0j1 * (delta_i0j1 - guj1i0) * gui2j0
// 			 -pui0i1 * pui1i2 *  puj1j0 * (delta_i0j0 - guj0i0) * gui2j1
// 			 -pui1i0 * pui2i1 *  puj0j1 * (delta_i2j1 - guj1i2) * gui0j0
// 			 +pui1i0 * pui2i1 *  puj1j0 * (delta_i2j0 - guj0i2) * gui0j1);

// 		const num t6 = (2 - guj0j0 - guj1j1) * 
// 			(+pdi0i1 * pdi1i2 *  pdj0j1 * (delta_i0j1 - gdj1i0) * gdi2j0
// 		     -pdi0i1 * pdi1i2 *  pdj1j0 * (delta_i0j0 - gdj0i0) * gdi2j1
// 			 -pdi1i0 * pdi2i1 *  pdj0j1 * (delta_i2j1 - gdj1i2) * gdi0j0
// 			 +pdi1i0 * pdi2i1 *  pdj1j0 * (delta_i2j0 - gdj0i2) * gdi0j1);

// 		const num t1 = ( (delta_i0j0 - guj0i0) * gui2j0 + (delta_i0j1 - guj1i0) * gui2j1 ) * 
// 			pui0i1 * pui1i2 *  (pdj1j0 * gdj0j1 - pdj0j1 * gdj1j0);
// 		const num t2 = ( (delta_i2j0 - guj0i2) * gui0j0 + (delta_i2j1 - guj1i2) * gui0j1 ) * 
// 			pui1i0 * pui2i1 *  (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1);
// 		const num t3 = ( (delta_i0j0 - gdj0i0) * gdi2j0 + (delta_i0j1 - gdj1i0) * gdi2j1 ) * 
// 			pdi0i1 * pdi1i2 *  (puj1j0 * guj0j1 - puj0j1 * guj1j0);
// 		const num t4 = ( (delta_i2j0 - gdj0i2) * gdi0j0 + (delta_i2j1 - gdj1i2) * gdi0j1 ) * 
// 			pdi1i0 * pdi2i1 *  (puj0j1 * guj1j0 - puj1j0 * guj0j1);

// 		m->J2jn[bb] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
// 		//J2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t = 0
// 		const num x = pui0i1*pui1i2*puj0j1*(delta_i0j1 - guj1i0)*gui2j0 +
// 					  pui1i0*pui2i1*puj1j0*(delta_i2j0 - guj0i2)*gui0j1 +
// 					  pdi0i1*pdi1i2*pdj0j1*(delta_i0j1 - gdj1i0)*gdi2j0 +
// 					  pdi1i0*pdi2i1*pdj1j0*(delta_i2j0 - gdj0i2)*gdi0j1;
// 		const num y = pui0i1*pui1i2*puj1j0*(delta_i0j0 - guj0i0)*gui2j1 +
// 		              pui1i0*pui2i1*puj0j1*(delta_i2j1 - guj1i2)*gui0j0 +
// 		              pdi0i1*pdi1i2*pdj1j0*(delta_i0j0 - gdj0i0)*gdi2j1 +
// 		              pdi1i0*pdi2i1*pdj0j1*(delta_i2j1 - gdj1i2)*gdi0j0;
// 		m->J2j[bb]  += pre*((pui1i0*pui2i1*gui0i2 - pui0i1*pui1i2*gui2i0 + pdi1i0*pdi2i1*gdi0i2 - pdi0i1*pdi1i2*gdi2i0)
// 		                   *(puj1j0*guj0j1        - puj0j1*guj1j0        + pdj1j0*gdj0j1        - pdj0j1*gdj1j0) 
// 		                   + x - y);
// 	}
// 	}


	// nematic correlator measurements, t = 0
	if (meas_nematic_corr)
	for (int c = 0; c < NEM_BONDS*N; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < NEM_BONDS*N; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const int delta_i0j0 = (i0 == j0);
		const int delta_i1j0 = (i1 == j0);
		const int delta_i0j1 = (i0 == j1);
		const int delta_i1j1 = (i1 == j1);
		const num gui0i0 = Gu00[i0 + i0*N];
		const num gui1i0 = Gu00[i1 + i0*N];
		const num gui0i1 = Gu00[i0 + i1*N];
		const num gui1i1 = Gu00[i1 + i1*N];
		const num gui0j0 = Gu00[i0 + j0*N];
		const num gui1j0 = Gu00[i1 + j0*N];
		const num gui0j1 = Gu00[i0 + j1*N];
		const num gui1j1 = Gu00[i1 + j1*N];
		const num guj0i0 = Gu00[j0 + i0*N];
		const num guj1i0 = Gu00[j1 + i0*N];
		const num guj0i1 = Gu00[j0 + i1*N];
		const num guj1i1 = Gu00[j1 + i1*N];
		const num guj0j0 = Gu00[j0 + j0*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num guj1j1 = Gu00[j1 + j1*N];
		const num gdi0i0 = Gd00[i0 + i0*N];
		const num gdi1i0 = Gd00[i1 + i0*N];
		const num gdi0i1 = Gd00[i0 + i1*N];
		const num gdi1i1 = Gd00[i1 + i1*N];
		const num gdi0j0 = Gd00[i0 + j0*N];
		const num gdi1j0 = Gd00[i1 + j0*N];
		const num gdi0j1 = Gd00[i0 + j1*N];
		const num gdi1j1 = Gd00[i1 + j1*N];
		const num gdj0i0 = Gd00[j0 + i0*N];
		const num gdj1i0 = Gd00[j1 + i0*N];
		const num gdj0i1 = Gd00[j0 + i1*N];
		const num gdj1i1 = Gd00[j1 + i1*N];
		const num gdj0j0 = Gd00[j0 + j0*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		const num gdj1j1 = Gd00[j1 + j1*N];
		const int delta_i0i1 = 0;
		const int delta_j0j1 = 0;
		const num uuuu = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-guj1j0)*guj0j1-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j1-guj1i1)*guj0j1+(delta_i0i1-gui1i0)*gui0j1*(delta_i1j0-guj0i1)*(delta_j0j1-guj1j0)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0i1*gui1j1*(delta_j0j1-guj1j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-guj1i1)*gui1j1-(delta_i0j0-guj0i0)*gui0j1*(1.-gui1i1)*(delta_j0j1-guj1j0)-(delta_i0j0-guj0i0)*gui0j1*(delta_i1j1-guj1i1)*gui1j0+(delta_i0j1-guj1i0)*gui0i1*gui1j0*guj0j1+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gui1i1)*guj0j1-(delta_i0j1-guj1i0)*gui0j0*(delta_i1j0-guj0i1)*gui1j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num uuud = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-gdj1j1);
		const num uudu = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num uudd = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-gdj1j0)*gdj0j1;
		const num uduu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0j1*(1.-gdi1i1)*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gdi1i1)*guj0j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num udud = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi1j1;
		const num uddu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		const num uddd = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0);
		const num duuu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0);
		const num duud = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num dudu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-guj1i1)*gui1j1;
		const num dudd = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gui1i1)*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gui1i1)*gdj0j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num dduu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-guj1j0)*guj0j1;
		const num ddud = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num dddu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-guj1j1);
		const num dddd = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-gdj1j0)*gdj0j1-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdj0j1+(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j0-gdj0i1)*(delta_j0j1-gdj1j0)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0i1*gdi1j1*(delta_j0j1-gdj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdi1j1-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gdi1i1)*(delta_j0j1-gdj1j0)-(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j1-gdj1i1)*gdi1j0+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j0*gdj0j1+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gdi1i1)*gdj0j1-(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j0-gdj0i1)*gdi1j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		m->nem_nnnn[bb] += pre*(uuuu + uuud + uudu + uudd
				      + uduu + udud + uddu + uddd
				      + duuu + duud + dudu + dudd
				      + dduu + ddud + dddu + dddd);
		m->nem_ssss[bb] += pre*(uuuu - uuud - uudu + uudd
				      - uduu + udud + uddu - uddd
				      - duuu + duud + dudu - dudd
				      + dduu - ddud - dddu + dddd);
	}
	}

	//=======================================================================
	// now handle t > 0 case: no delta functions here.

	if (meas_bond_corr || meas_thermal)
	#pragma omp parallel for
	for (int t = 1; t < L; t++) {
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const num gui1i0 = Gutt_t[i1 + i0*N];
		const num gui0i1 = Gutt_t[i0 + i1*N];
		const num gui0j0 = Gut0_t[i0 + j0*N];
		const num gui1j0 = Gut0_t[i1 + j0*N];
		const num gui0j1 = Gut0_t[i0 + j1*N];
		const num gui1j1 = Gut0_t[i1 + j1*N];
		const num guj0i0 = Gu0t_t[j0 + i0*N];
		const num guj1i0 = Gu0t_t[j1 + i0*N];
		const num guj0i1 = Gu0t_t[j0 + i1*N];
		const num guj1i1 = Gu0t_t[j1 + i1*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num gdi1i0 = Gdtt_t[i1 + i0*N];
		const num gdi0i1 = Gdtt_t[i0 + i1*N];
		const num gdi0j0 = Gdt0_t[i0 + j0*N];
		const num gdi1j0 = Gdt0_t[i1 + j0*N];
		const num gdi0j1 = Gdt0_t[i0 + j1*N];
		const num gdi1j1 = Gdt0_t[i1 + j1*N];
		const num gdj0i0 = Gd0t_t[j0 + i0*N];
		const num gdj1i0 = Gd0t_t[j1 + i0*N];
		const num gdj0i1 = Gd0t_t[j0 + i1*N];
		const num gdj1i1 = Gd0t_t[j1 + i1*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];

		const int delta_i0i1 = 0;
		const int delta_j0j1 = 0;
		const int delta_i0j0 = 0;
		const int delta_i0j1 = 0;
		const int delta_i1j0 = 0;
		const int delta_i1j1 = 0;
		// 1 bond -- 1 bond correlator measurements, t > 0
		if (meas_bond_corr) {
			m->pair_bb[bb + num_bb*t] += 0.5*pre*(gui0j0*gdi1j1 + gui1j0*gdi0j1 + gui0j1*gdi1j0 + gui1j1*gdi0j0);
			const num x = -pui0i1*puj0j1*guj1i0*gui1j0 - pui1i0*puj1j0*guj0i1*gui0j1
			             - pdi0i1*pdj0j1*gdj1i0*gdi1j0 - pdi1i0*pdj1j0*gdj0i1*gdi0j1;
			const num y = -pui0i1*puj1j0*guj0i0*gui1j1 - pui1i0*puj0j1*guj1i1*gui0j0
			             - pdi0i1*pdj1j0*gdj0i0*gdi1j1 - pdi1i0*pdj0j1*gdj1i1*gdi0j0;
			m->jj[bb + num_bb*t]   += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 + pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
			                              *(puj1j0*guj0j1 - puj0j1*guj1j0 + pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x - y);
			m->jsjs[bb + num_bb*t] += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 - pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
			                              *(puj1j0*guj0j1 - puj0j1*guj1j0 - pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x - y);
			m->kk[bb + num_bb*t]   += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 + pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
			                              *(puj1j0*guj0j1 + puj0j1*guj1j0 + pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x + y);
			m->ksks[bb + num_bb*t] += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 - pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
			                              *(puj1j0*guj0j1 + puj0j1*guj1j0 - pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x + y);
		}
		if (meas_thermal){
			const num gui0i0 = Gutt_t[i0 + i0*N];
			const num gdi0i0 = Gdtt_t[i0 + i0*N];
			const num gui1i1 = Gutt_t[i1 + i1*N];
			const num gdi1i1 = Gdtt_t[i1 + i1*N];
			const num guj0j0 = Gu00[j0 + j0*N];
			const num gdj0j0 = Gd00[j0 + j0*N];
			const num guj1j1 = Gu00[j1 + j1*N];
			const num gdj1j1 = Gd00[j1 + j1*N];

			//jn(i0i1)j(j0j1): 6 fermion product, 2 phases, t > 0
			num _wick_jn = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
			 			   (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			num _wick_j = - puj1j0*guj0j1 + puj0j1*guj1j0 - pdj1j0*gdj0j1 + pdj0j1*gdj1j0;

			num t1 = ( (delta_i0j1 - guj1i0) * gui0j0 + (delta_i1j1 - guj1i1) * gui1j0 ) * 
				puj0j1 * (pdi1i0 * gdi0i1 - pdi0i1 * gdi1i0);
			num t2 = ( (delta_i0j0 - guj0i0) * gui0j1 + (delta_i1j0 - guj0i1) * gui1j1 ) * 
				puj1j0 * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1);
			num t3 = ( (delta_i0j1 - gdj1i0) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi1j0 ) * 
				pdj0j1 * (pui1i0 * gui0i1 - pui0i1 * gui1i0);
			num t4 = ( (delta_i0j0 - gdj0i0) * gdi0j1 + (delta_i1j0 - gdj0i1) * gdi1j1 ) * 
				pdj1j0 * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			num t5 = (2-gui0i0-gui1i1) * 
				(+pdi0i1 * pdj0j1 * (delta_i0j1 - gdj1i0) * gdi1j0 
				 -pdi0i1 * pdj1j0 * (delta_i0j0 - gdj0i0) * gdi1j1
				 -pdi1i0 * pdj0j1 * (delta_i1j1 - gdj1i1) * gdi0j0
				 +pdi1i0 * pdj1j0 * (delta_i1j0 - gdj0i1) * gdi0j1);
			num t6 = (2-gdi0i0-gdi1i1) *
				(+pui0i1 * puj0j1 * (delta_i0j1 - guj1i0) * gui1j0
				 -pui0i1 * puj1j0 * (delta_i0j0 - guj0i0) * gui1j1
				 -pui1i0 * puj0j1 * (delta_i1j1 - guj1i1) * gui0j0
				 +pui1i0 * puj1j0 * (delta_i1j0 - guj0i1) * gui0j1);

			// const num t13 =  pdi0i1 * pdj0j1 * (2-gui0i0-gui1i1) * (delta_i0j1 - gdj1i0) * gdi1j0;
			// const num t14 = -pdi0i1 * pdj1j0 * (2-gui0i0-gui1i1) * (delta_i0j0 - gdj0i0) * gdi1j1;
			// const num t23 = -pdi1i0 * pdj0j1 * (2-gui0i0-gui1i1) * (delta_i1j1 - gdj1i1) * gdi0j0;
			// const num t24 =  pdi1i0 * pdj1j0 * (2-gui0i0-gui1i1) * (delta_i1j0 - gdj0i1) * gdi0j1;

			// const num t31 =  pui0i1 * puj0j1 * (2-gdi0i0-gdi1i1) * (delta_i0j1 - guj1i0) * gui1j0;
			// const num t32 = -pui0i1 * puj1j0 * (2-gdi0i0-gdi1i1) * (delta_i0j0 - guj0i0) * gui1j1;
			// const num t41 = -pui1i0 * puj0j1 * (2-gdi0i0-gdi1i1) * (delta_i1j1 - guj1i1) * gui0j0;
			// const num t42 =  pui1i0 * puj1j0 * (2-gdi0i0-gdi1i1) * (delta_i1j0 - guj0i1) * gui0j1;

			// const num t11 =  pdi0i1 * puj0j1 * (-gdi1i0) * ( (delta_i0j1 - guj1i0) * gui0j0 + (delta_i1j1 - guj1i1) * gui1j0 );
			// const num t21 = -pdi1i0 * puj0j1 * (-gdi0i1) * ( (delta_i0j1 - guj1i0) * gui0j0 + (delta_i1j1 - guj1i1) * gui1j0 );

			// const num t22 =  pdi1i0 * puj1j0 * (-gdi0i1) * ( (delta_i0j0 - guj0i0) * gui0j1 + (delta_i1j0 - guj0i1) * gui1j1 );
			// const num t12 = -pdi0i1 * puj1j0 * (-gdi1i0) * ( (delta_i0j0 - guj0i0) * gui0j1 + (delta_i1j0 - guj0i1) * gui1j1 );

			// const num t33 =  pui0i1 * pdj0j1 * (-gui1i0) * ( (delta_i0j1 - gdj1i0) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi1j0 );
			// const num t43 = -pui1i0 * pdj0j1 * (-gui0i1) * ( (delta_i0j1 - gdj1i0) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi1j0 );

			// const num t44 =  pui1i0 * pdj1j0 * (-gui0i1) * ( (delta_i0j0 - gdj0i0) * gdi0j1 + (delta_i1j0 - gdj0i1) * gdi1j1 );
			// const num t34 = -pui0i1 * pdj1j0 * (-gui1i0) * ( (delta_i0j0 - gdj0i0) * gdi0j1 + (delta_i1j0 - gdj0i1) * gdi1j1 );
			//j(i0i1)jn(j0j1), 6 fermion product, 2 phases, t > 0
			m->new_jnj[bb + num_bb*t] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);

			_wick_jn = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
			 		   (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);
			_wick_j = - pui1i0*gui0i1 + pui0i1*gui1i0 - pdi1i0*gdi0i1 + pdi0i1*gdi1i0;

			t5 = (2 - gdj0j0 - gdj1j1) * 
				(+pui0i1 * puj0j1 * (delta_i0j1 - guj1i0) * gui1j0
				 -pui0i1 * puj1j0 * (delta_i0j0 - guj0i0) * gui1j1
				 -pui1i0 * puj0j1 * (delta_i1j1 - guj1i1) * gui0j0
				 +pui1i0 * puj1j0 * (delta_i1j0 - guj0i1) * gui0j1);

			t6 = (2 - guj0j0 - guj1j1) * 
				(+pdi0i1 * pdj0j1 * (delta_i0j1 - gdj1i0) * gdi1j0
			     -pdi0i1 * pdj1j0 * (delta_i0j0 - gdj0i0) * gdi1j1
				 -pdi1i0 * pdj0j1 * (delta_i1j1 - gdj1i1) * gdi0j0
				 +pdi1i0 * pdj1j0 * (delta_i1j0 - gdj0i1) * gdi0j1);
			t1 = ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 ) * 
				pui0i1 * (pdj1j0 * gdj0j1 - pdj0j1 * gdj1j0);
			t2 = ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 ) * 
				pui1i0 * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1);
			t3 = ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 ) * 
				pdi0i1 * (puj1j0 * guj0j1 - puj0j1 * guj1j0);
			t4 = ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 ) * 
				pdi1i0 * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

			// t1 = ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 ) * pui0i1 * pdj0j1 * (gdj0j1 - gdj1j0);
			// t2 = ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 ) * pui1i0 * pdj1j0 * (gdj1j0 - gdj0j1);
			// t3 = ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 ) * pdi0i1 * puj0j1 * (guj0j1 - guj1j0);
			// t4 = ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 ) * pdi1i0 * puj1j0 * (guj1j0 - guj0j1);

			// const num t13 = pui0i1 * puj0j1 * (2 - gdj0j0 - gdj1j1) * (delta_i0j1 - guj1i0) * gui1j0;
			// const num t14 =-pui0i1 * puj1j0 * (2 - gdj0j0 - gdj1j1) * (delta_i0j0 - guj0i0) * gui1j1;
			// const num t23 =-pui1i0 * puj0j1 * (2 - gdj0j0 - gdj1j1) * (delta_i1j1 - guj1i1) * gui0j0;
			// const num t24 = pui1i0 * puj1j0 * (2 - gdj0j0 - gdj1j1) * (delta_i1j0 - guj0i1) * gui0j1;

			// const num t31 = pdi0i1 * pdj0j1 * (2 - guj0j0 - guj1j1) * (delta_i0j1 - gdj1i0) * gdi1j0;
			// const num t32 =-pdi0i1 * pdj1j0 * (2 - guj0j0 - guj1j1) * (delta_i0j0 - gdj0i0) * gdi1j1;
			// const num t41 =-pdi1i0 * pdj0j1 * (2 - guj0j0 - guj1j1) * (delta_i1j1 - gdj1i1) * gdi0j0;
			// const num t42 = pdi1i0 * pdj1j0 * (2 - guj0j0 - guj1j1) * (delta_i1j0 - gdj0i1) * gdi0j1;

			// const num t11 = pui0i1 * pdj0j1 * (-gdj1j0) * ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 );
			// const num t12 =-pui0i1 * pdj1j0 * (-gdj0j1) * ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 );
			// const num t21 =-pui1i0 * pdj0j1 * (-gdj1j0) * ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 );
			// const num t22 = pui1i0 * pdj1j0 * (-gdj0j1) * ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 );

			// const num t33 = pdi0i1 * puj0j1 * (-guj1j0) * ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 );
			// const num t34 =-pdi0i1 * puj1j0 * (-guj0j1) * ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 );
			// const num t43 =-pdi1i0 * puj0j1 * (-guj1j0) * ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 );
			// const num t44 = pdi1i0 * puj1j0 * (-guj0j1) * ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 );

			m->new_jjn[bb + num_bb*t] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);

			// thermal: jnjn, t > 0. TODO: simplify this expression for faster measurements
			const num tAA = pdi0i1*pdj0j1 * 
			(+1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0j1-gdj1i0)*gdi1j0)
			 +1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0j1-gdj1i0)*gdi1j0)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0j1-gdj1i0)*gdi1j0)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j0*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj1j0)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0j1-gdj1i0)*gdi1j0));
			const num tAB = pdi0i1*pdj1j0 * 
			(-1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0j0-gdj0i0)*gdi1j1)
			 -1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0j0-gdj0i0)*gdi1j1)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0j0-gdj0i0)*gdi1j1)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j1*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi1i0)*(delta_j0j1-gdj0j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0j0-gdj0i0)*gdi1j1));
			const num tAC = pdi0i1*puj0j1 * 
			(+1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0j0-gdj0i0)*gdi1j0)
			 +1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0j1-gdj1i0)*gdi1j1)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0j0-gdj0i0)*gdi1j0)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0j1-gdj1i0)*gdi1j1));
			const num tAD = pdi0i1*puj1j0 * 
			(-1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0j0-gdj0i0)*gdi1j0)
			 -1*(+(1.-gui0i0)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0j1-gdj1i0)*gdi1j1)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i0j0-gdj0i0)*gdi1j0*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi1i0)*(1.-gdj0j0)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0j0-gdj0i0)*gdi1j0)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i0j1-gdj1i0)*gdi1j1*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi1i0)*(1.-gdj1j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0j1-gdj1i0)*gdi1j1));

			const num tBA = pdi1i0*pdj0j1 * 
			(-1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi0j0)
			 -1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j1-gdj1i1)*gdi0j0)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i1j1-gdj1i1)*gdi0j0)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j0*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj1j0)+(delta_i1j1-guj1i1)*gui1j1*(delta_i1j1-gdj1i1)*gdi0j0));
			const num tBB = pdi1i0*pdj1j0 * 
			(+1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j0-gdj0i1)*gdi0j1)
			 +1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi0j1)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj0j0)+(delta_i1j0-guj0i1)*gui1j0*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i1j0-guj0i1)*gui1j0*(delta_i1j0-gdj0i1)*gdi0j1)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j1*(1.-guj1j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i0i1-gdi0i1)*(delta_j0j1-gdj0j1)+(delta_i1j1-guj1i1)*gui1j1*(delta_i1j0-gdj0i1)*gdi0j1));
			const num tBC = pdi1i0*puj0j1 * 
			(-1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i1j0-gdj0i1)*gdi0j0)
			 -1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i0j1-guj1i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi0j1)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i1j0-gdj0i1)*gdi0j0)
			 -1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj1j0)+(delta_i1j1-guj1i1)*gui1j0*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i1j1-guj1i1)*gui1j0*(delta_i1j1-gdj1i1)*gdi0j1));
			const num tBD = pdi1i0*puj1j0 * 
			(+1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i0j0-guj0i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi0j0)
			 +1*(+(1.-gui0i0)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj0j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j1*(delta_i1j1-gdj1i1)*gdi0j1)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i1j0-gdj0i1)*gdi0j0*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi0i1)*(1.-gdj0j0)+(delta_i1j0-guj0i1)*gui1j1*(delta_i1j0-gdj0i1)*gdi0j0)
			 +1*(+(1.-gui1i1)*(delta_i0i1-gdi0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gui1i1)*(delta_i1j1-gdj1i1)*gdi0j1*(delta_j0j1-guj0j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i0i1-gdi0i1)*(1.-gdj1j1)+(delta_i1j0-guj0i1)*gui1j1*(delta_i1j1-gdj1i1)*gdi0j1));

			const num tCA = pui0i1*pdj0j1 * 
			(+1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0j0-guj0i0)*gui1j0)
			 +1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0j1-guj1i0)*gui1j1)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0j0-guj0i0)*gui1j0)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0j1-guj1i0)*gui1j1));
			const num tCB = pui0i1*pdj1j0 * 
			(-1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0j0-guj0i0)*gui1j0)
			 -1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0j1-guj1i0)*gui1j1)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j0*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui1i0)*(1.-guj0j0)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0j0-guj0i0)*gui1j0)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j1*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui1i0)*(1.-guj1j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0j1-guj1i0)*gui1j1));
			const num tCC = pui0i1*puj0j1 * 
			(+1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0j1-guj1i0)*gui1j0)
			 +1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0j1-guj1i0)*gui1j0)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0j1-guj1i0)*gui1j0)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i0j1-guj1i0)*gui1j0*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj1j0)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0j1-guj1i0)*gui1j0));
			const num tCD = pui0i1*puj1j0 * 
			(-1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0j0-guj0i0)*gui1j1)
			 -1*(+(1.-gdi0i0)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0j0-guj0i0)*gui1j1)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0j0-guj0i0)*gui1j1)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui1i0)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i0j0-guj0i0)*gui1j1*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui1i0)*(delta_j0j1-guj0j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0j0-guj0i0)*gui1j1));

			const num tDA = pui1i0*pdj0j1 * 
			(-1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j0-guj0i1)*gui0j0)
			 -1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j1-guj1i1)*gui0j1)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i1j0-guj0i1)*gui0j0)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj1j0)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj1j0)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i1j1-gdj1i1)*gdi1j0*(delta_i1j1-guj1i1)*gui0j1));
			const num tDB = pui1i0*pdj1j0 * 
			(+1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j0-guj0i1)*gui0j0)
			 +1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj0j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j1-guj1i1)*gui0j1)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj0j0)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j0*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui0i1)*(1.-guj0j0)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i1j0-guj0i1)*gui0j0)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-guj1j1)*(delta_j0j1-gdj0j1)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j1*(delta_j0j1-gdj0j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i0i1-gui0i1)*(1.-guj1j1)+(delta_i1j0-gdj0i1)*gdi1j1*(delta_i1j1-guj1i1)*gui0j1));
			const num tDC = pui1i0*puj0j1 * 
			(-1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-guj1i1)*gui0j0)
			 -1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j1-guj1i1)*gui0j0)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i1j1-guj1i1)*gui0j0)
			 -1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj1j0)+(1.-gdi1i1)*(delta_i1j1-guj1i1)*gui0j0*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj1j0)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i1j1-guj1i1)*gui0j0));
			const num tDD = pui1i0*puj1j0 * 
			(+1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j0-guj0i1)*gui0j1)
			 +1*(+(1.-gdi0i0)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-guj0i1)*gui0j1)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj0j0)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj0j0)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i1j0-gdj0i1)*gdi1j0*(delta_i1j0-guj0i1)*gui0j1)
			 +1*(+(1.-gdi1i1)*(delta_i0i1-gui0i1)*(1.-gdj1j1)*(delta_j0j1-guj0j1)+(1.-gdi1i1)*(delta_i1j0-guj0i1)*gui0j1*(1.-gdj1j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i0i1-gui0i1)*(delta_j0j1-guj0j1)+(delta_i1j1-gdj1i1)*gdi1j1*(delta_i1j0-guj0i1)*gui0j1));

			m->jnjn[bb + num_bb*t]  += pre*(tAA+tAB+tAC+tAD
								+tBA+tBB+tBC+tBD
								+tCA+tCB+tCC+tCD
								+tDA+tDB+tDC+tDD);
		}

	}
	}
	}

	// measurement of j2-j2: 4 fermion product, 4 phases, t > 0
	// this is the ``clever'' way to do it
	// TODO: implement pair_b2b2,js2js2,k2k2,ks2ks2
	if (meas_2bond_corr)
	#pragma omp parallel for
	for (int t = 1; t < L; t++) {
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int c = 0; c < num_b2; c++) {
		const int jtype = c / N;
		const int j = c % N;
#ifdef USE_PEIERLS
		const num ppuj0j2 = p->pp_u[ j + N*jtype];
		const num ppuj2j0 = p->ppr_u[j + N*jtype];
		const num ppdj0j2 = p->pp_d[ j + N*jtype];
		const num ppdj2j0 = p->ppr_d[j + N*jtype];
#endif
		// printf("c = %d, jtype = %d,j = %d, ", c, jtype,j );
		// printf("pj0j1*pj1j2 = %f \n", (double) ppuj0j2);
		// fflush(stdout); 
		const int j0 = p->bond2s[c];
		const int j2 = p->bond2s[c + num_b2];
	for (int b = 0; b < num_b2; b++) {
		const int itype = b / N;
		const int i = b % N;
#ifdef USE_PEIERLS
		const num ppui0i2 = p->pp_u[ i + N*itype];
		const num ppui2i0 = p->ppr_u[i + N*itype];
		const num ppdi0i2 = p->pp_d[ i + N*itype];
		const num ppdi2i0 = p->ppr_d[i + N*itype];
#endif
		const int i0 = p->bond2s[b];
		const int i2 = p->bond2s[b + num_b2];

		const int bb = p->map_b2b2[b + c*num_b2];
		const num pre = phase / p->degen_b2b2[bb];

		const int delta_i0j0 = 0;
		const int delta_i2j0 = 0;
		const int delta_i0j2 = 0;
		const int delta_i2j2 = 0;
		// const int delta_i0i1 = 0;
		// const int delta_j0j1 = 0;
		const num gui2i0 = Gutt_t[i2 + i0*N];
		const num gui0i2 = Gutt_t[i0 + i2*N];
		const num gui0j0 = Gut0_t[i0 + j0*N];
		const num gui2j0 = Gut0_t[i2 + j0*N];
		const num gui0j2 = Gut0_t[i0 + j2*N];
		const num gui2j2 = Gut0_t[i2 + j2*N];
		const num guj0i0 = Gu0t_t[j0 + i0*N];
		const num guj2i0 = Gu0t_t[j2 + i0*N];
		const num guj0i2 = Gu0t_t[j0 + i2*N];
		const num guj2i2 = Gu0t_t[j2 + i2*N];
		const num guj2j0 = Gu00[j2 + j0*N];
		const num guj0j2 = Gu00[j0 + j2*N];
		const num gdi2i0 = Gdtt_t[i2 + i0*N];
		const num gdi0i2 = Gdtt_t[i0 + i2*N];
		const num gdi0j0 = Gdt0_t[i0 + j0*N];
		const num gdi2j0 = Gdt0_t[i2 + j0*N];
		const num gdi0j2 = Gdt0_t[i0 + j2*N];
		const num gdi2j2 = Gdt0_t[i2 + j2*N];
		const num gdj0i0 = Gd0t_t[j0 + i0*N];
		const num gdj2i0 = Gd0t_t[j2 + i0*N];
		const num gdj0i2 = Gd0t_t[j0 + i2*N];
		const num gdj2i2 = Gd0t_t[j2 + i2*N];
		const num gdj2j0 = Gd00[j2 + j0*N];
		const num gdj0j2 = Gd00[j0 + j2*N];

		const num x = ppui0i2*ppuj0j2*(delta_i0j2 - guj2i0)*gui2j0 +
					  ppui2i0*ppuj2j0*(delta_i2j0 - guj0i2)*gui0j2 +
					  ppdi0i2*ppdj0j2*(delta_i0j2 - gdj2i0)*gdi2j0 +
					  ppdi2i0*ppdj2j0*(delta_i2j0 - gdj0i2)*gdi0j2;
		const num y = ppui0i2*ppuj2j0*(delta_i0j0 - guj0i0)*gui2j2 +
		              ppui2i0*ppuj0j2*(delta_i2j2 - guj2i2)*gui0j0 +
		              ppdi0i2*ppdj2j0*(delta_i0j0 - gdj0i0)*gdi2j2 +
		              ppdi2i0*ppdj0j2*(delta_i2j2 - gdj2i2)*gdi0j0;
		m->j2j2[bb + num_b2b2*t] += 
			pre*((ppui2i0*gui0i2 - ppui0i2*gui2i0 + ppdi2i0*gdi0i2 - ppdi0i2*gdi2i0)
		        *(ppuj2j0*guj0j2 - ppuj0j2*guj2j0 + ppdj2j0*gdj0j2 - ppdj0j2*gdj2j0) 
		        + x - y);

	}
	}
	}

	// measurement of J2-J2: 4 fermion product, 4 phases, t > 0
// 	if (meas_hop2_corr)
// 	#pragma omp parallel for
// 	for (int t = 1; t < L; t++) {
// 		const num *const restrict Gu0t_t = Gu0t + N*N*t;
// 		const num *const restrict Gutt_t = Gutt + N*N*t;
// 		const num *const restrict Gut0_t = Gut0 + N*N*t;
// 		const num *const restrict Gd0t_t = Gd0t + N*N*t;
// 		const num *const restrict Gdtt_t = Gdtt + N*N*t;
// 		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
// 	for (int c = 0; c < num_hop2; c++) {
// 		const int j0 = p->hop2s[c];
// 		const int j1 = p->hop2s[c + num_hop2];
// 		const int j2 = p->hop2s[c + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num puj0j1 = p->peierlsu[j0 + N*j1];
// 		const num puj1j0 = p->peierlsu[j1 + N*j0];
// 		const num pdj0j1 = p->peierlsd[j0 + N*j1];
// 		const num pdj1j0 = p->peierlsd[j1 + N*j0];
// 		const num puj1j2 = p->peierlsu[j1 + N*j2];
// 		const num puj2j1 = p->peierlsu[j2 + N*j1];
// 		const num pdj1j2 = p->peierlsd[j1 + N*j2];
// 		const num pdj2j1 = p->peierlsd[j2 + N*j1];
// #endif
// 	for (int b = 0; b < num_hop2; b++) {
// 		const int i0 = p->hop2s[b];
// 		const int i1 = p->hop2s[b + num_hop2];
// 		const int i2 = p->hop2s[b + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num pui0i1 = p->peierlsu[i0 + N*i1];
// 		const num pui1i0 = p->peierlsu[i1 + N*i0];
// 		const num pdi0i1 = p->peierlsd[i0 + N*i1];
// 		const num pdi1i0 = p->peierlsd[i1 + N*i0];
// 		const num pui1i2 = p->peierlsu[i1 + N*i2];
// 		const num pui2i1 = p->peierlsu[i2 + N*i1];
// 		const num pdi1i2 = p->peierlsd[i1 + N*i2];
// 		const num pdi2i1 = p->peierlsd[i2 + N*i1];
// #endif
// 		const int bb = p->map_hop2_hop2[b + c*num_hop2];
// 		const num pre = phase / p->degen_hop2_hop2[bb];

// 		const num gui2i0 = Gutt_t[i2 + i0*N];
// 		const num gui0i2 = Gutt_t[i0 + i2*N];
// 		const num gui0j0 = Gut0_t[i0 + j0*N];
// 		const num gui2j0 = Gut0_t[i2 + j0*N];
// 		const num gui0j2 = Gut0_t[i0 + j2*N];
// 		const num gui2j2 = Gut0_t[i2 + j2*N];
// 		const num guj0i0 = Gu0t_t[j0 + i0*N];
// 		const num guj2i0 = Gu0t_t[j2 + i0*N];
// 		const num guj0i2 = Gu0t_t[j0 + i2*N];
// 		const num guj2i2 = Gu0t_t[j2 + i2*N];
// 		const num guj2j0 = Gu00[j2 + j0*N];
// 		const num guj0j2 = Gu00[j0 + j2*N];
// 		const num gdi2i0 = Gdtt_t[i2 + i0*N];
// 		const num gdi0i2 = Gdtt_t[i0 + i2*N];
// 		const num gdi0j0 = Gdt0_t[i0 + j0*N];
// 		const num gdi2j0 = Gdt0_t[i2 + j0*N];
// 		const num gdi0j2 = Gdt0_t[i0 + j2*N];
// 		const num gdi2j2 = Gdt0_t[i2 + j2*N];
// 		const num gdj0i0 = Gd0t_t[j0 + i0*N];
// 		const num gdj2i0 = Gd0t_t[j2 + i0*N];
// 		const num gdj0i2 = Gd0t_t[j0 + i2*N];
// 		const num gdj2i2 = Gd0t_t[j2 + i2*N];
// 		const num gdj2j0 = Gd00[j2 + j0*N];
// 		const num gdj0j2 = Gd00[j0 + j2*N];

// 		const num x = -pui0i1*pui1i2*puj0j1*puj1j2*guj2i0*gui2j0
// 					  -pui1i0*pui2i1*puj1j0*puj2j1*guj0i2*gui0j2
// 					  -pdi0i1*pdi1i2*pdj0j1*pdj1j2*gdj2i0*gdi2j0
// 					  -pdi1i0*pdi2i1*pdj1j0*pdj2j1*gdj0i2*gdi0j2;
// 		const num y = -pui0i1*pui1i2*puj1j0*puj2j1*guj0i0*gui2j2
// 		              -pui1i0*pui2i1*puj0j1*puj1j2*guj2i2*gui0j0
// 		              -pdi0i1*pdi1i2*pdj1j0*pdj2j1*gdj0i0*gdi2j2
// 		              -pdi1i0*pdi2i1*pdj0j1*pdj1j2*gdj2i2*gdi0j0;
// 		m->J2J2[bb + num_hop2_hop2*t] +=
// 		  pre*((pui1i0*pui2i1*gui0i2 - pui0i1*pui1i2*gui2i0 + pdi1i0*pdi2i1*gdi0i2 - pdi0i1*pdi1i2*gdi2i0)
// 		      *(puj1j0*puj2j1*guj0j2 - puj0j1*puj1j2*guj2j0 + pdj1j0*pdj2j1*gdj0j2 - pdj0j1*pdj1j2*gdj2j0) 
// 		      + x - y);
// 	}
// 	}
// 	}

    // measurement of jn(i0i1)-j2(j0j1j2): 6 fermion product, 3 phases, t > 0
	//                j(i0i1) -j2(j0j1j2): 4 fermion product, 3 phases, t > 0
	// i = i0 <-> i1
	// j = j0 <-> j1 <-> j2
	// Essentially matrix[j,i] = bond(i) x bond2(j)
	// This is "clever" way to do it
	if (meas_thermal || meas_2bond_corr) 
	#pragma omp parallel for
	for (int t = 1; t < L; t++) {
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int c = 0; c < num_b2; c++) {
		const int jtype = c / N;
		const int j = c % N;
#ifdef USE_PEIERLS
		const num ppuj0j2 = p->pp_u[ j + N*jtype];
		const num ppuj2j0 = p->ppr_u[j + N*jtype];
		const num ppdj0j2 = p->pp_d[ j + N*jtype];
		const num ppdj2j0 = p->ppr_d[j + N*jtype];
#endif
		const int j0 = p->bond2s[c];
		const int j2 = p->bond2s[c + num_b2];
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_b2b[b + c*num_b];
		const num pre = phase / p->degen_b2b[bb];
		const int delta_i0j0 = 0;
		const int delta_i1j0 = 0;
		const int delta_i0j2 = 0;
		const int delta_i1j2 = 0;
		const num gui1i0 = Gutt_t[i1 + i0*N];
		const num gui0i1 = Gutt_t[i0 + i1*N];
		const num gui0j0 = Gut0_t[i0 + j0*N];
		const num gui1j0 = Gut0_t[i1 + j0*N];
		const num gui0j2 = Gut0_t[i0 + j2*N];
		const num gui1j2 = Gut0_t[i1 + j2*N];
		const num guj0i0 = Gu0t_t[j0 + i0*N];
		const num guj2i0 = Gu0t_t[j2 + i0*N];
		const num guj0i1 = Gu0t_t[j0 + i1*N];
		const num guj2i1 = Gu0t_t[j2 + i1*N];
		const num guj2j0 = Gu00[j2 + j0*N];
		const num guj0j2 = Gu00[j0 + j2*N];
		const num gdi1i0 = Gdtt_t[i1 + i0*N];
		const num gdi0i1 = Gdtt_t[i0 + i1*N];
		const num gdi0j0 = Gdt0_t[i0 + j0*N];
		const num gdi1j0 = Gdt0_t[i1 + j0*N];
		const num gdi0j2 = Gdt0_t[i0 + j2*N];
		const num gdi1j2 = Gdt0_t[i1 + j2*N];
		const num gdj0i0 = Gd0t_t[j0 + i0*N];
		const num gdj2i0 = Gd0t_t[j2 + i0*N];
		const num gdj0i1 = Gd0t_t[j0 + i1*N];
		const num gdj2i1 = Gd0t_t[j2 + i1*N];
		const num gdj2j0 = Gd00[j2 + j0*N];
		const num gdj0j2 = Gd00[j0 + j2*N];
		if (meas_thermal) {

			const num gui0i0 = Gutt_t[i0 + i0*N];
			const num gdi0i0 = Gdtt_t[i0 + i0*N];
			const num gui1i1 = Gutt_t[i1 + i1*N];
			const num gdi1i1 = Gdtt_t[i1 + i1*N];

			//jn(i0i1)-j2(j0j1j2): 6 fermion product, 3 phases, t > 0
			//TODO: further group these expressions together?
			const num _wick_jn = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
			 			         (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			const num _wick_j = - ppuj2j0 * guj0j2 + ppuj0j2 * guj2j0 
						        - ppdj2j0 * gdj0j2 + ppdj0j2 * gdj2j0;

			const num t1 = ( (delta_i0j2 - guj2i0) * gui0j0 + (delta_i1j2 - guj2i1) * gui1j0 ) * 
				ppuj0j2 * (pdi1i0 * gdi0i1 - pdi0i1 * gdi1i0);
			const num t2 = ( (delta_i0j0 - guj0i0) * gui0j2 + (delta_i1j0 - guj0i1) * gui1j2 ) * 
				ppuj2j0 * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1);
			const num t3 = ( (delta_i0j2 - gdj2i0) * gdi0j0 + (delta_i1j2 - gdj2i1) * gdi1j0 ) * 
				ppdj0j2 * (pui1i0 * gui0i1 - pui0i1 * gui1i0);
			const num t4 = ( (delta_i0j0 - gdj0i0) * gdi0j2 + (delta_i1j0 - gdj0i1) * gdi1j2 ) * 
				ppdj2j0 * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
			const num t5 = (2 - gui0i0 - gui1i1) * 
				(+pdi0i1 * ppdj0j2 * (delta_i0j2 - gdj2i0) * gdi1j0 
				 -pdi0i1 * ppdj2j0 * (delta_i0j0 - gdj0i0) * gdi1j2
				 -pdi1i0 * ppdj0j2 * (delta_i1j2 - gdj2i1) * gdi0j0
				 +pdi1i0 * ppdj2j0 * (delta_i1j0 - gdj0i1) * gdi0j2);
			const num t6 = (2 - gdi0i0 - gdi1i1) *
				(+pui0i1 * ppuj0j2 * (delta_i0j2 - guj2i0) * gui1j0
				 -pui0i1 * ppuj2j0 * (delta_i0j0 - guj0i0) * gui1j2
				 -pui1i0 * ppuj0j2 * (delta_i1j2 - guj2i1) * gui0j0
				 +pui1i0 * ppuj2j0 * (delta_i1j0 - guj0i1) * gui0j2);

			m->jnj2[bb + num_b2b*t]   += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
		}
		if (meas_2bond_corr) {
			//j(i0i1) -j2(j0j1j2): 4 fermion product, 3 phases, t > 0
			const num x = pui0i1 * ppuj0j2 * (delta_i0j2 - guj2i0)*gui1j0 +
						  pui1i0 * ppuj2j0 * (delta_i1j0 - guj0i1)*gui0j2 +
						  pdi0i1 * ppdj0j2 * (delta_i0j2 - gdj2i0)*gdi1j0 +
						  pdi1i0 * ppdj2j0 * (delta_i1j0 - gdj0i1)*gdi0j2;
			const num y = pui0i1 * ppuj2j0 * (delta_i0j0 - guj0i0)*gui1j2 +
			              pui1i0 * ppuj0j2 * (delta_i1j2 - guj2i1)*gui0j0 +
			              pdi0i1 * ppdj2j0 * (delta_i0j0 - gdj0i0)*gdi1j2 +
			              pdi1i0 * ppdj0j2 * (delta_i1j2 - gdj2i1)*gdi0j0;
			m->jj2[bb + num_b2b*t]  += pre*((pui1i0*gui0i1        - pui0i1*gui1i0        + pdi1i0*gdi0i1        - pdi0i1*gdi1i0)
			                   *(ppuj2j0 * guj0j2 - ppuj0j2 * guj2j0 + ppdj2j0 * gdj0j2 - ppdj0j2 * gdj2j0) 
			                   + x - y);
		}
	}
	}
    }

	// measurement of jn(i0i1)-J2(j0j1j2): 6 fermion product, 3 phases, t > 0
	//                j(i0i1) -J2(j0j1j2): 4 fermion product, 3 phases, t > 0
	// i = i0 <-> i1
	// j = j0 <-> j1 <-> j2
	// Essentially matrix[j,i] = bond(i) x hop2(j)
// 	if (meas_hop2_corr)
// 	#pragma omp parallel for
// 	for (int t = 1; t < L; t++) {
// 		const num *const restrict Gu0t_t = Gu0t + N*N*t;
// 		const num *const restrict Gutt_t = Gutt + N*N*t;
// 		const num *const restrict Gut0_t = Gut0 + N*N*t;
// 		const num *const restrict Gd0t_t = Gd0t + N*N*t;
// 		const num *const restrict Gdtt_t = Gdtt + N*N*t;
// 		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
// 	for (int c = 0; c < num_hop2; c++) {
// 		const int j0 = p->hop2s[c];
// 		const int j1 = p->hop2s[c + num_hop2];
// 		const int j2 = p->hop2s[c + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num puj0j1 = p->peierlsu[j0 + N*j1];
// 		const num puj1j0 = p->peierlsu[j1 + N*j0];
// 		const num pdj0j1 = p->peierlsd[j0 + N*j1];
// 		const num pdj1j0 = p->peierlsd[j1 + N*j0];
// 		const num puj1j2 = p->peierlsu[j1 + N*j2];
// 		const num puj2j1 = p->peierlsu[j2 + N*j1];
// 		const num pdj1j2 = p->peierlsd[j1 + N*j2];
// 		const num pdj2j1 = p->peierlsd[j2 + N*j1];
// #endif
// 	for (int b = 0; b < num_b; b++) {
// 		const int i0 = p->bonds[b];
// 		const int i1 = p->bonds[b + num_b];
// #ifdef USE_PEIERLS
// 		const num pui0i1 = p->peierlsu[i0 + N*i1];
// 		const num pui1i0 = p->peierlsu[i1 + N*i0];
// 		const num pdi0i1 = p->peierlsd[i0 + N*i1];
// 		const num pdi1i0 = p->peierlsd[i1 + N*i0];
// #endif
// 		const int bb = p->map_b_hop2[b + c*num_b];
// 		const num pre = phase / p->degen_b_hop2[bb];

// 		const num gui1i0 = Gutt_t[i1 + i0*N];
// 		const num gui0i1 = Gutt_t[i0 + i1*N];
// 		const num gui0j0 = Gut0_t[i0 + j0*N];
// 		const num gui1j0 = Gut0_t[i1 + j0*N];
// 		const num gui0j2 = Gut0_t[i0 + j2*N];
// 		const num gui1j2 = Gut0_t[i1 + j2*N];
// 		const num guj0i0 = Gu0t_t[j0 + i0*N];
// 		const num guj2i0 = Gu0t_t[j2 + i0*N];
// 		const num guj0i1 = Gu0t_t[j0 + i1*N];
// 		const num guj2i1 = Gu0t_t[j2 + i1*N];
// 		const num guj2j0 = Gu00[j2 + j0*N];
// 		const num guj0j2 = Gu00[j0 + j2*N];
// 		const num gdi1i0 = Gdtt_t[i1 + i0*N];
// 		const num gdi0i1 = Gdtt_t[i0 + i1*N];
// 		const num gdi0j0 = Gdt0_t[i0 + j0*N];
// 		const num gdi1j0 = Gdt0_t[i1 + j0*N];
// 		const num gdi0j2 = Gdt0_t[i0 + j2*N];
// 		const num gdi1j2 = Gdt0_t[i1 + j2*N];
// 		const num gdj0i0 = Gd0t_t[j0 + i0*N];
// 		const num gdj2i0 = Gd0t_t[j2 + i0*N];
// 		const num gdj0i1 = Gd0t_t[j0 + i1*N];
// 		const num gdj2i1 = Gd0t_t[j2 + i1*N];
// 		const num gdj2j0 = Gd00[j2 + j0*N];
// 		const num gdj0j2 = Gd00[j0 + j2*N];

// 		const num gui0i0 = Gutt_t[i0 + i0*N];
// 		const num gdi0i0 = Gdtt_t[i0 + i0*N];
// 		const num gui1i1 = Gutt_t[i1 + i1*N];
// 		const num gdi1i1 = Gdtt_t[i1 + i1*N];

// 		//jn(i0i1)-J2(j0j1j2): 6 fermion product, 3 phases, t > 0
// 		//TODO: further group these expressions together?
// 		const num _wick_jn = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
// 		 			         (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
// 		const num _wick_j = - puj1j0*puj2j1*guj0j2 + puj0j1*puj1j2*guj2j0 
// 					        - pdj1j0*pdj2j1*gdj0j2 + pdj0j1*pdj1j2*gdj2j0;

// 		const num t1 = -( guj2i0 * gui0j0 + guj2i1 * gui1j0 ) * 
// 			puj0j1 * puj1j2 * (pdi1i0 * gdi0i1 - pdi0i1 * gdi1i0);
// 		const num t2 = -( guj0i0 * gui0j2 + guj0i1 * gui1j2 ) * 
// 			puj1j0 * puj2j1 * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1);
// 		const num t3 = -( gdj2i0 * gdi0j0 + gdj2i1 * gdi1j0 ) * 
// 			pdj0j1 * pdj1j2 * (pui1i0 * gui0i1 - pui0i1 * gui1i0);
// 		const num t4 = -( gdj0i0 * gdi0j2 + gdj0i1 * gdi1j2 ) * 
// 			pdj1j0 * pdj2j1 * (pui0i1 * gui1i0 - pui1i0 * gui0i1);
// 		const num t5 = (2 - gui0i0 - gui1i1) * 
// 			(+pdi0i1 * pdj0j1 * pdj1j2 * ( - gdj2i0) * gdi1j0 
// 			 -pdi0i1 * pdj1j0 * pdj2j1 * ( - gdj0i0) * gdi1j2
// 			 -pdi1i0 * pdj0j1 * pdj1j2 * ( - gdj2i1) * gdi0j0
// 			 +pdi1i0 * pdj1j0 * pdj2j1 * ( - gdj0i1) * gdi0j2);
// 		const num t6 = (2 - gdi0i0 - gdi1i1) *
// 			(+pui0i1 * puj0j1 * puj1j2 * ( - guj2i0) * gui1j0
// 			 -pui0i1 * puj1j0 * puj2j1 * ( - guj0i0) * gui1j2
// 			 -pui1i0 * puj0j1 * puj1j2 * ( - guj2i1) * gui0j0
// 			 +pui1i0 * puj1j0 * puj2j1 * ( - guj0i1) * gui0j2);

// 		m->jnJ2[bb + num_b_hop2*t] += 
// 			pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
// 		//j(i0i1) -J2(j0j1j2): 4 fermion product, 3 phases, t > 0
// 		const num x = pui0i1*puj0j1*puj1j2*(- guj2i0)*gui1j0 +
// 					  pui1i0*puj1j0*puj2j1*(- guj0i1)*gui0j2 +
// 					  pdi0i1*pdj0j1*pdj1j2*(- gdj2i0)*gdi1j0 +
// 					  pdi1i0*pdj1j0*pdj2j1*(- gdj0i1)*gdi0j2;
// 		const num y = pui0i1*puj1j0*puj2j1*(- guj0i0)*gui1j2 +
// 		              pui1i0*puj0j1*puj1j2*(- guj2i1)*gui0j0 +
// 		              pdi0i1*pdj1j0*pdj2j1*(- gdj0i0)*gdi1j2 +
// 		              pdi1i0*pdj0j1*pdj1j2*(- gdj2i1)*gdi0j0;
// 		m->jJ2[bb + num_b_hop2*t]  += 
// 			pre*((pui1i0*gui0i1        - pui0i1*gui1i0        + pdi1i0*gdi0i1        - pdi0i1*gdi1i0)
// 		        *(puj1j0*puj2j1*guj0j2 - puj0j1*puj1j2*guj2j0 + pdj1j0*pdj2j1*gdj0j2 - pdj0j1*pdj1j2*gdj2j0) 
// 		        + x - y);
// 	}
// 	}
// 	}


	// measurement of j2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t > 0
	//                j2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t > 0
	// i = i0 <-> i1 <-> i2
	// j = j0 <-> j1 Is this the correct indexing?
	// Essentially matrix[j,i] = bond2(i) x bond(j)
	if (meas_thermal || meas_2bond_corr) 
	#pragma omp parallel for
	for (int t = 1; t < L; t++) {
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b2; b++) {
		const int itype = b / N;
		const int i = b % N;
#ifdef USE_PEIERLS
		const num ppui0i2 = p->pp_u[ i + N*itype];
		const num ppui2i0 = p->ppr_u[i + N*itype];
		const num ppdi0i2 = p->pp_d[ i + N*itype];
		const num ppdi2i0 = p->ppr_d[i + N*itype];
#endif
		const int i0 = p->bond2s[b];
		const int i2 = p->bond2s[b + num_b2];

		const int bb = p->map_bb2[b + c*num_b2];
		const num pre = phase / p->degen_bb2[bb];

		const int delta_i0j0 = 0;
		const int delta_i2j0 = 0;
		const int delta_i0j1 = 0;
		const int delta_i2j1 = 0;

		const num gui2i0 = Gutt_t[i2 + i0*N];
		const num gui0i2 = Gutt_t[i0 + i2*N];
		const num gui0j0 = Gut0_t[i0 + j0*N];
		const num gui2j0 = Gut0_t[i2 + j0*N];
		const num gui0j1 = Gut0_t[i0 + j1*N];
		const num gui2j1 = Gut0_t[i2 + j1*N];
		const num guj0i0 = Gu0t_t[j0 + i0*N];
		const num guj1i0 = Gu0t_t[j1 + i0*N];
		const num guj0i2 = Gu0t_t[j0 + i2*N];
		const num guj1i2 = Gu0t_t[j1 + i2*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num gdi2i0 = Gdtt_t[i2 + i0*N];
		const num gdi0i2 = Gdtt_t[i0 + i2*N];
		const num gdi0j0 = Gdt0_t[i0 + j0*N];
		const num gdi2j0 = Gdt0_t[i2 + j0*N];
		const num gdi0j1 = Gdt0_t[i0 + j1*N];
		const num gdi2j1 = Gdt0_t[i2 + j1*N];
		const num gdj0i0 = Gd0t_t[j0 + i0*N];
		const num gdj1i0 = Gd0t_t[j1 + i0*N];
		const num gdj0i2 = Gd0t_t[j0 + i2*N];
		const num gdj1i2 = Gd0t_t[j1 + i2*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		if (meas_thermal) {

			const num guj0j0 = Gu00[j0 + j0*N];
			const num guj1j1 = Gu00[j1 + j1*N];
			const num gdj0j0 = Gd00[j0 + j0*N];
			const num gdj1j1 = Gd00[j1 + j1*N];

			//j2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t > 0
			const num _wick_j = - ppui2i0 * gui0i2 + ppui0i2 * gui2i0 
			                    - ppdi2i0 * gdi0i2 + ppdi0i2 * gdi2i0;
			const num _wick_jn = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
			 		             (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

			const num t5 = (2 - gdj0j0 - gdj1j1) * 
				(+ppui0i2 * puj0j1 * (delta_i0j1 - guj1i0) * gui2j0
				 -ppui0i2 * puj1j0 * (delta_i0j0 - guj0i0) * gui2j1
				 -ppui2i0 * puj0j1 * (delta_i2j1 - guj1i2) * gui0j0
				 +ppui2i0 * puj1j0 * (delta_i2j0 - guj0i2) * gui0j1);

			const num t6 = (2 - guj0j0 - guj1j1) * 
				(+ppdi0i2 * pdj0j1 * (delta_i0j1 - gdj1i0) * gdi2j0
			     -ppdi0i2 * pdj1j0 * (delta_i0j0 - gdj0i0) * gdi2j1
				 -ppdi2i0 * pdj0j1 * (delta_i2j1 - gdj1i2) * gdi0j0
				 +ppdi2i0 * pdj1j0 * (delta_i2j0 - gdj0i2) * gdi0j1);

			const num t1 = ( (delta_i0j0 - guj0i0) * gui2j0 + (delta_i0j1 - guj1i0) * gui2j1 ) * 
				ppui0i2 * (pdj1j0 * gdj0j1 - pdj0j1 * gdj1j0);
			const num t2 = ( (delta_i2j0 - guj0i2) * gui0j0 + (delta_i2j1 - guj1i2) * gui0j1 ) * 
				ppui2i0 * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1);
			const num t3 = ( (delta_i0j0 - gdj0i0) * gdi2j0 + (delta_i0j1 - gdj1i0) * gdi2j1 ) * 
				ppdi0i2 * (puj1j0 * guj0j1 - puj0j1 * guj1j0);
			const num t4 = ( (delta_i2j0 - gdj0i2) * gdi0j0 + (delta_i2j1 - gdj1i2) * gdi0j1 ) * 
				ppdi2i0 * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

			m->j2jn[bb + num_bb2*t] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
		}
		if (meas_2bond_corr) {
			//j2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t > 0
			const num x = ppui0i2 * puj0j1*(delta_i0j1 - guj1i0)*gui2j0 +
						  ppui2i0 * puj1j0*(delta_i2j0 - guj0i2)*gui0j1 +
						  ppdi0i2 * pdj0j1*(delta_i0j1 - gdj1i0)*gdi2j0 +
						  ppdi2i0 * pdj1j0*(delta_i2j0 - gdj0i2)*gdi0j1;
			const num y = ppui0i2 * puj1j0*(delta_i0j0 - guj0i0)*gui2j1 +
			              ppui2i0 * puj0j1*(delta_i2j1 - guj1i2)*gui0j0 +
			              ppdi0i2 * pdj1j0*(delta_i0j0 - gdj0i0)*gdi2j1 +
			              ppdi2i0 * pdj0j1*(delta_i2j1 - gdj1i2)*gdi0j0;
			m->j2j[bb + num_bb2*t]  += pre*((ppui2i0 * gui0i2 - ppui0i2 * gui2i0 + ppdi2i0 * gdi0i2 - ppdi0i2 * gdi2i0)
			                   *( puj1j0 * guj0j1 -  puj0j1 * guj1j0 +  pdj1j0 * gdj0j1 -  pdj0j1 * gdj1j0) 
			                   + x - y);
		}
	}
	}
	}


	// measurement of J2(i0i1i2)-jn(i0i1): 6 fermion product, 3 phases, t > 0
	//                J2(i0i1i2)- j(i0i1): 4 fermion product, 3 phases, t > 0
	// i = i0 <-> i1 <-> i2
	// j = j0 <-> j1 Is this the correct indexing?
	// Essentially matrix[j,i] = hop2(i) x bond(j)
// 	if (meas_hop2_corr) 
// 	#pragma omp parallel for
// 	for (int t = 1; t < L; t++) {
// 		const num *const restrict Gu0t_t = Gu0t + N*N*t;
// 		const num *const restrict Gutt_t = Gutt + N*N*t;
// 		const num *const restrict Gut0_t = Gut0 + N*N*t;
// 		const num *const restrict Gd0t_t = Gd0t + N*N*t;
// 		const num *const restrict Gdtt_t = Gdtt + N*N*t;
// 		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
// 	for (int c = 0; c < num_b; c++) {
// 		const int j0 = p->bonds[c];
// 		const int j1 = p->bonds[c + num_b];
// #ifdef USE_PEIERLS
// 		const num puj0j1 = p->peierlsu[j0 + N*j1];
// 		const num puj1j0 = p->peierlsu[j1 + N*j0];
// 		const num pdj0j1 = p->peierlsd[j0 + N*j1];
// 		const num pdj1j0 = p->peierlsd[j1 + N*j0];
// #endif
// 	for (int b = 0; b < num_hop2; b++) {
// 		const int i0 = p->hop2s[b];
// 		const int i1 = p->hop2s[b + num_hop2];
// 		const int i2 = p->hop2s[b + 2*num_hop2];
// #ifdef USE_PEIERLS
// 		const num pui0i1 = p->peierlsu[i0 + N*i1];
// 		const num pui1i0 = p->peierlsu[i1 + N*i0];
// 		const num pdi0i1 = p->peierlsd[i0 + N*i1];
// 		const num pdi1i0 = p->peierlsd[i1 + N*i0];
// 		const num pui1i2 = p->peierlsu[i1 + N*i2];
// 		const num pui2i1 = p->peierlsu[i2 + N*i1];
// 		const num pdi1i2 = p->peierlsd[i1 + N*i2];
// 		const num pdi2i1 = p->peierlsd[i2 + N*i1];
// #endif
// 		const int bb = p->map_hop2_b[b + c*num_hop2];
// 		const num pre = phase / p->degen_hop2_b[bb];

// 		const num gui2i0 = Gutt_t[i2 + i0*N];
// 		const num gui0i2 = Gutt_t[i0 + i2*N];
// 		const num gui0j0 = Gut0_t[i0 + j0*N];
// 		const num gui2j0 = Gut0_t[i2 + j0*N];
// 		const num gui0j1 = Gut0_t[i0 + j1*N];
// 		const num gui2j1 = Gut0_t[i2 + j1*N];
// 		const num guj0i0 = Gu0t_t[j0 + i0*N];
// 		const num guj1i0 = Gu0t_t[j1 + i0*N];
// 		const num guj0i2 = Gu0t_t[j0 + i2*N];
// 		const num guj1i2 = Gu0t_t[j1 + i2*N];
// 		const num guj1j0 = Gu00[j1 + j0*N];
// 		const num guj0j1 = Gu00[j0 + j1*N];
// 		const num gdi2i0 = Gdtt_t[i2 + i0*N];
// 		const num gdi0i2 = Gdtt_t[i0 + i2*N];
// 		const num gdi0j0 = Gdt0_t[i0 + j0*N];
// 		const num gdi2j0 = Gdt0_t[i2 + j0*N];
// 		const num gdi0j1 = Gdt0_t[i0 + j1*N];
// 		const num gdi2j1 = Gdt0_t[i2 + j1*N];
// 		const num gdj0i0 = Gd0t_t[j0 + i0*N];
// 		const num gdj1i0 = Gd0t_t[j1 + i0*N];
// 		const num gdj0i2 = Gd0t_t[j0 + i2*N];
// 		const num gdj1i2 = Gd0t_t[j1 + i2*N];
// 		const num gdj1j0 = Gd00[j1 + j0*N];
// 		const num gdj0j1 = Gd00[j0 + j1*N];

// 		const num guj0j0 = Gu00[j0 + j0*N];
// 		const num guj1j1 = Gu00[j1 + j1*N];
// 		const num gdj0j0 = Gd00[j0 + j0*N];
// 		const num gdj1j1 = Gd00[j1 + j1*N];

// 		//J2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t > 0
// 		const num _wick_j = - pui1i0*pui2i1*gui0i2 + pui0i1*pui1i2*gui2i0 
// 		                    - pdi1i0*pdi2i1*gdi0i2 + pdi0i1*pdi1i2*gdi2i0;
// 		const num _wick_jn = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
// 		 		             (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

// 		const num t5 = (2 - gdj0j0 - gdj1j1) * 
// 			(+pui0i1 * pui1i2 * puj0j1 * ( - guj1i0) * gui2j0
// 			 -pui0i1 * pui1i2 * puj1j0 * ( - guj0i0) * gui2j1
// 			 -pui1i0 * pui2i1 * puj0j1 * ( - guj1i2) * gui0j0
// 			 +pui1i0 * pui2i1 * puj1j0 * ( - guj0i2) * gui0j1);

// 		const num t6 = (2 - guj0j0 - guj1j1) * 
// 			(+pdi0i1 * pdi1i2 * pdj0j1 * ( - gdj1i0) * gdi2j0
// 		     -pdi0i1 * pdi1i2 * pdj1j0 * ( - gdj0i0) * gdi2j1
// 			 -pdi1i0 * pdi2i1 * pdj0j1 * ( - gdj1i2) * gdi0j0
// 			 +pdi1i0 * pdi2i1 * pdj1j0 * ( - gdj0i2) * gdi0j1);

// 		const num t1 = -( guj0i0 * gui2j0 + guj1i0 * gui2j1 ) * 
// 			pui0i1 * pui1i2 * (pdj1j0 * gdj0j1 - pdj0j1 * gdj1j0);
// 		const num t2 = -( guj0i2 * gui0j0 + guj1i2 * gui0j1 ) * 
// 			pui1i0 * pui2i1 * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1);
// 		const num t3 = -( gdj0i0 * gdi2j0 + gdj1i0 * gdi2j1 ) * 
// 			pdi0i1 * pdi1i2 * (puj1j0 * guj0j1 - puj0j1 * guj1j0);
// 		const num t4 = -( gdj0i2 * gdi0j0 + gdj1i2 * gdi0j1 ) * 
// 			pdi1i0 * pdi2i1 * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

// 		m->J2jn[bb + num_hop2_b*t] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
// 		//J2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t > 0
// 		const num x = pui0i1*pui1i2*puj0j1*(- guj1i0)*gui2j0 +
// 					  pui1i0*pui2i1*puj1j0*(- guj0i2)*gui0j1 +
// 					  pdi0i1*pdi1i2*pdj0j1*(- gdj1i0)*gdi2j0 +
// 					  pdi1i0*pdi2i1*pdj1j0*(- gdj0i2)*gdi0j1;
// 		const num y = pui0i1*pui1i2*puj1j0*(- guj0i0)*gui2j1 +
// 		              pui1i0*pui2i1*puj0j1*(- guj1i2)*gui0j0 +
// 		              pdi0i1*pdi1i2*pdj1j0*(- gdj0i0)*gdi2j1 +
// 		              pdi1i0*pdi2i1*pdj0j1*(- gdj1i2)*gdi0j0;
// 		m->J2j[bb + num_hop2_b*t]  += 
// 			pre*((pui1i0*pui2i1*gui0i2 - pui0i1*pui1i2*gui2i0 + pdi1i0*pdi2i1*gdi0i2 - pdi0i1*pdi1i2*gdi2i0)
// 		        *(puj1j0*guj0j1        - puj0j1*guj1j0        + pdj1j0*gdj0j1        - pdj0j1*gdj1j0) 
// 		        + x - y);
// 	}
// 	}
// 	}

	// nematic correlator measurements, t > 0
	if (meas_nematic_corr)
	#pragma omp parallel for
	for (int t = 1; t < L; t++) {
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int c = 0; c < NEM_BONDS*N; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
	for (int b = 0; b < NEM_BONDS*N; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const num gui0i0 = Gutt_t[i0 + i0*N];
		const num gui1i0 = Gutt_t[i1 + i0*N];
		const num gui0i1 = Gutt_t[i0 + i1*N];
		const num gui1i1 = Gutt_t[i1 + i1*N];
		const num gui0j0 = Gut0_t[i0 + j0*N];
		const num gui1j0 = Gut0_t[i1 + j0*N];
		const num gui0j1 = Gut0_t[i0 + j1*N];
		const num gui1j1 = Gut0_t[i1 + j1*N];
		const num guj0i0 = Gu0t_t[j0 + i0*N];
		const num guj1i0 = Gu0t_t[j1 + i0*N];
		const num guj0i1 = Gu0t_t[j0 + i1*N];
		const num guj1i1 = Gu0t_t[j1 + i1*N];
		const num guj0j0 = Gu00[j0 + j0*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num guj1j1 = Gu00[j1 + j1*N];
		const num gdi0i0 = Gdtt_t[i0 + i0*N];
		const num gdi1i0 = Gdtt_t[i1 + i0*N];
		const num gdi0i1 = Gdtt_t[i0 + i1*N];
		const num gdi1i1 = Gdtt_t[i1 + i1*N];
		const num gdi0j0 = Gdt0_t[i0 + j0*N];
		const num gdi1j0 = Gdt0_t[i1 + j0*N];
		const num gdi0j1 = Gdt0_t[i0 + j1*N];
		const num gdi1j1 = Gdt0_t[i1 + j1*N];
		const num gdj0i0 = Gd0t_t[j0 + i0*N];
		const num gdj1i0 = Gd0t_t[j1 + i0*N];
		const num gdj0i1 = Gd0t_t[j0 + i1*N];
		const num gdj1i1 = Gd0t_t[j1 + i1*N];
		const num gdj0j0 = Gd00[j0 + j0*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		const num gdj1j1 = Gd00[j1 + j1*N];
		const int delta_i0i1 = 0;
		const int delta_j0j1 = 0;
		const int delta_i0j0 = 0;
		const int delta_i1j0 = 0;
		const int delta_i0j1 = 0;
		const int delta_i1j1 = 0;
		const num uuuu = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-guj1j0)*guj0j1-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j1-guj1i1)*guj0j1+(delta_i0i1-gui1i0)*gui0j1*(delta_i1j0-guj0i1)*(delta_j0j1-guj1j0)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0i1*gui1j1*(delta_j0j1-guj1j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-guj1i1)*gui1j1-(delta_i0j0-guj0i0)*gui0j1*(1.-gui1i1)*(delta_j0j1-guj1j0)-(delta_i0j0-guj0i0)*gui0j1*(delta_i1j1-guj1i1)*gui1j0+(delta_i0j1-guj1i0)*gui0i1*gui1j0*guj0j1+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gui1i1)*guj0j1-(delta_i0j1-guj1i0)*gui0j0*(delta_i1j0-guj0i1)*gui1j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num uuud = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-gdj1j1);
		const num uudu = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num uudd = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-gdj1j0)*gdj0j1;
		const num uduu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0j1*(1.-gdi1i1)*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gdi1i1)*guj0j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num udud = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi1j1;
		const num uddu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		const num uddd = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0);
		const num duuu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0);
		const num duud = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num dudu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-guj1i1)*gui1j1;
		const num dudd = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gui1i1)*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gui1i1)*gdj0j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num dduu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-guj1j0)*guj0j1;
		const num ddud = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num dddu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-guj1j1);
		const num dddd = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-gdj1j0)*gdj0j1-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdj0j1+(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j0-gdj0i1)*(delta_j0j1-gdj1j0)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0i1*gdi1j1*(delta_j0j1-gdj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdi1j1-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gdi1i1)*(delta_j0j1-gdj1j0)-(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j1-gdj1i1)*gdi1j0+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j0*gdj0j1+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gdi1i1)*gdj0j1-(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j0-gdj0i1)*gdi1j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		m->nem_nnnn[bb + num_bb*t] += pre*(uuuu + uuud + uudu + uudd
		                                 + uduu + udud + uddu + uddd
		                                 + duuu + duud + dudu + dudd
		                                 + dduu + ddud + dddu + dddd);
		m->nem_ssss[bb + num_bb*t] += pre*(uuuu - uuud - uudu + uudd
		                                 - uduu + udud + uddu - uddd
		                                 - duuu + duud + dudu - dudd
		                                 + dduu - ddud - dddu + dddd);
	}
	}
	}
}

