#include "meas.h"
#include "data.h"
#include "util.h"
#include "prof.h"
#include <unistd.h>
// #include "omp.h"
#include <stdio.h>
// #include <math.h>
// #include <assert.h>

// number of types of bonds kept for 4-particle nematic correlators.
// 2 by default since these are slow measurerments
#define NEM_BONDS 2

// Number of OpenMP threads used for expensive unequal time measurements.
// This overrides the omp_set_num_threads() function called by main if 
// OMP_MEAS_NUM_THREADS is not specified during compilation.
#ifndef OMP_MEAS_NUM_THREADS
	#define OMP_MEAS_NUM_THREADS 2
#endif

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
/**
 * these should not be defined as one in the nflux=0 case, they 
 * can be 1,2,4, 1+tp^2, so they are not constant. Unless I introduce scaling
 * in the gen_1band_hub.py file
 * Must keep consistent between gen_1band_hub, simulation here, and
 * data analysis in thermal.py and jqjq.py
 */
// #define ppui0i2 1 
// #define ppui2i0 1
// #define ppdi0i2 1
// #define ppdi2i0 1
// #define ppuj0j2 1
// #define ppuj2j0 1
// #define ppdj0j2 1
// #define ppdj2j0 1
#endif

/**
 * Take equal time measurements
 * @param p     [description]
 * @param phase [description]
 * @param gu    [description]
 * @param gd    [description]
 * @param m     [description]
 */
void measure_eqlt(const struct params *const restrict p, 
	const num phase,
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
		// NOTE: density is *sum* of density_u and density_d
		m->density[r] += pre*(2. - guii - gdii);
		m->density_u[r] += pre*(1. - guii);
		m->density_d[r] += pre*(1. - gdii);
		m->double_occ[r] += pre*(1. - guii)*(1. - gdii);
	}

	// 2 site measurements
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			const int delta = (i == j);
			const int r = p->map_ij[i + j*N];
			const num pre = phase / p->degen_ij[r];
			const num guii = gu[i + i*N], gdii = gd[i + i*N];
			const num guij = gu[i + j*N], gdij = gd[i + j*N];
			const num guji = gu[j + i*N], gdji = gd[j + i*N];
			const num gujj = gu[j + j*N], gdjj = gd[j + j*N];
			// NOTE: g00 is *average* of g00_u and g00_d
			#ifdef USE_PEIERLS
			m->g00[r] += 0.5*pre*(guij*p->peierlsu[j + i*N] + gdij*p->peierlsd[j + i*N]);
			m->g00_u[r] += pre*(guij*p->peierlsu[j + i*N]);
			m->g00_d[r] += pre*(gdij*p->peierlsd[j + i*N]);
			#else
			m->g00[r] += 0.5*pre*(guij + gdij);
			m->g00_u[r] += pre*guij;
			m->g00_d[r] += pre*gdij;
			#endif
			const num x = delta*(guii + gdii) - (guji*guij + gdji*gdij);
			m->nn[r] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
			m->xx[r] += 0.25*pre*(delta*(guii + gdii) - (guji*gdij + gdji*guij));
			m->zz[r] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
			m->pair_sw[r] += pre*guij*gdij;
			if (meas_energy_corr) {
				const num nuinuj = (1. - guii)*(1. - gujj) + (delta - guji)*guij;
				const num ndindj = (1. - gdii)*(1. - gdjj) + (delta - gdji)*gdij;
				m->vv[r] += pre*nuinuj*ndindj;
				m->vn[r] += pre*(nuinuj*(1. - gdii) + (1. - guii)*ndindj);
			}
		}
	}

	const int num_plaq_accum = p->num_plaq_accum;
	const int num_plaq = p->num_plaq;
	const int meas_chiral = p->meas_chiral;

	//printf("num_plaq_accum = %d, num_plaq=%d, meas_chiral= %d\n",num_plaq_accum,num_plaq,meas_chiral);
	if (meas_chiral) {
		for (int a = 0; a < num_plaq; a++){
			//printf("a=%d\n",a);
			const int i0 = p->plaqs[a];
			const int i1 = p->plaqs[a + 1*num_plaq];
			const int i2 = p->plaqs[a + 2*num_plaq];

			const num gui0i0 = gu[i0 + i0*N];
			const num gui1i1 = gu[i1 + i1*N];
			const num gui2i2 = gu[i2 + i2*N];
			const num gdi0i0 = gd[i0 + i0*N];
			const num gdi1i1 = gd[i1 + i1*N];
			const num gdi2i2 = gd[i2 + i2*N];

			const num gui0i1 = gu[i0 + N*i1];
			const num gui1i0 = gu[i1 + N*i0];
			const num gui0i2 = gu[i0 + N*i2];
			const num gui2i0 = gu[i2 + N*i0];
			const num gui1i2 = gu[i1 + N*i2];
			const num gui2i1 = gu[i2 + N*i1];

			const num gdi0i1 = gd[i0 + N*i1];
			const num gdi1i0 = gd[i1 + N*i0];
			const num gdi0i2 = gd[i0 + N*i2];
			const num gdi2i0 = gd[i2 + N*i0];
			const num gdi1i2 = gd[i1 + N*i2];
			const num gdi2i1 = gd[i2 + N*i1];
			
			// printf(stderr,"a=%d\n",a);
			// sleep(1);
			const int r = p->map_plaq[a];
			const num pre = phase / p->degen_plaq[r];
			// this is obtained using edwin's wick script.
			const num x = 
				gdi1i1*gdi2i0*gui0i2 - gdi0i2*gdi2i1*gui1i0 + gdi0i1*gdi2i2*gui1i0 - gdi2i1*gui0i2*gui1i0 - 
				gdi2i0*gui0i2*gui1i1 + gdi0i1*gdi2i0*gui1i2 - gdi0i0*gdi2i1*gui1i2 + gdi2i1*gui0i0*gui1i2 + 
				gdi2i0*gui0i1*gui1i2 - gdi0i2*gdi1i1*gui2i0 + gdi0i2*gui1i1*gui2i0 + gdi0i1*gui1i2*gui2i0 - 
				gdi0i2*gui1i0*gui2i1 + gdi1i2*(gdi2i0*gui0i1 + (gdi0i1 + gui0i1)*gui2i0 + (gdi0i0 - gui0i0)*gui2i1) - 
				                       gdi1i0*(gdi2i1*gui0i2 + (gdi0i2 + gui0i2)*gui2i1 + gui0i1*(gdi2i2 - gui2i2)) - 
				gdi0i1*gui1i0*gui2i2;

			m->chi[r] += pre * x;

			// if (a == 8 || a == 8+N){
			// 	printf("degen=%d, r=%d, i0 = %d, i1=%d, i2= %d\n",p->degen_plaq[r], r, i0,i1,i2);
			// }
		}
		
	}
	
	// printf("total[0]: %f + i %f \n", creal(m->chi[0]), cimag(m->chi[0]));
	// printf("total[1]: %f + i %f \n", creal(m->chi[1]), cimag(m->chi[1]));
	// fflush(stdout);
	if (!meas_energy_corr)
		return;

	// 1 bond 1 site measurements
	for (int j = 0; j < N; j++) {
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



void meas_uneqlt_energy(const struct params *const restrict p, 
	const num phase,
	const num *const Gu0t,
	const num *const Gutt,
	const num *const Gut0,
	const num *const Gd0t,
	const num *const Gdtt,
	const num *const Gdt0,
	struct meas_uneqlt *const restrict m)
{
	const int N = p->N;
	const int L = p->L;
	const int num_b = p->num_b;
	const int num_bs = p->num_bs;

	const num *const restrict Gu00 = Gutt;
	const num *const restrict Gd00 = Gdtt;
	
	#pragma omp parallel for num_threads(OMP_MEAS_NUM_THREADS)
	for (int t = 0; t < L; t++) {
		const int delta_t = (t == 0);
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
		for (int j = 0; j < N; j++) {
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

				const num ku = pui1i0*(delta_i0i1 - gui0i1) + 
							   pui0i1*(delta_i0i1 - gui1i0);
				const num kd = pdi1i0*(delta_i0i1 - gdi0i1) + 
							   pdi0i1*(delta_i0i1 - gdi1i0);
				const num xu = pui0i1*(delta_i0j - guji0)*gui1j + 
							   pui1i0*(delta_i1j - guji1)*gui0j;
				const num xd = pdi0i1*(delta_i0j - gdji0)*gdi1j + 
							   pdi1i0*(delta_i1j - gdji1)*gdi0j;

				m->kv[bs + num_bs*t] += pre*((ku*(1. - gujj) + xu)*(1. - gdjj)
				                           + (kd*(1. - gdjj) + xd)*(1. - gujj));
				m->kn[bs + num_bs*t] += pre*((ku + kd)*(2. - gujj - gdjj) + xu + xd);
			}
		}
	}
}


void meas_uneqlt_nematic(const struct params *const restrict p, 
	const num phase,
	const num *const Gu0t,
	const num *const Gutt,
	const num *const Gut0,
	const num *const Gd0t,
	const num *const Gdtt,
	const num *const Gdt0,
	struct meas_uneqlt *const restrict m)
{
	const int N = p->N;
	const int L = p->L;
	const int num_b = p->num_b;
	const int num_bb = p->num_bb;

	const num *const restrict Gu00 = Gutt;
	const num *const restrict Gd00 = Gdtt;
	#pragma omp parallel for num_threads(OMP_MEAS_NUM_THREADS)
	for (int t = 0; t < L; t++) {
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



/**
 * Take unequal time measurements
 * @param p     [description]
 * @param phase [description]
 * @param Gu0t  [description]
 * @param Gutt  [description]
 * @param Gut0  [description]
 * @param Gd0t  [description]
 * @param Gdtt  [description]
 * @param Gdt0  [description]
 * @param m     [description]
 */
void measure_uneqlt(const struct params *const restrict p, 
	const num phase,
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
	const int num_b2 = p->num_b2, num_b2b2 = p->num_b2b2;
	const int num_b2b = p->num_b2b, num_bb2 = p->num_bb2;
	const int meas_bond_corr = p->meas_bond_corr;
	const int meas_2bond_corr = p->meas_2bond_corr;
	const int meas_energy_corr = p->meas_energy_corr;
	const int meas_nematic_corr = p->meas_nematic_corr;
	const int meas_thermal = p->meas_thermal;

	const num *const restrict Gu00 = Gutt;
	const num *const restrict Gd00 = Gdtt;

	// t <- [0,L): 2 site measurements
	// gt0, nn, xx, zz, pair_sw, vv, vn
	#pragma omp parallel for num_threads(OMP_MEAS_NUM_THREADS)
	for (int t = 0; t < L; t++) {
		const int delta_t = (t == 0);
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
		for (int j = 0; j < N; j++) {
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
				// NOTE: gt0 is *average* of gt0_u and gt0_d
					#ifdef USE_PEIERLS
				m->gt0[r + num_ij*t] += 0.5*pre*(guij*p->peierlsu[j + i*N] + 
												 gdij*p->peierlsd[j + i*N]);
				m->gt0_u[r + num_ij*t] += pre*(guij*p->peierlsu[j + i*N]);
				m->gt0_d[r + num_ij*t] += pre*(gdij*p->peierlsd[j + i*N]);
					#else
				m->gt0[r + num_ij*t] += 0.5*pre*(guij + gdij);
				m->gt0_u[r + num_ij*t] += pre*guij;
				m->gt0_d[r + num_ij*t] += pre*gdij;
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
	}

	// t <- [0,L): 1 bond 1 site measurements
	// kv, kn
	if (meas_energy_corr) {
		meas_uneqlt_energy(p,phase, Gu0t,Gutt,Gut0,Gd0t,Gdtt,Gdt0,m);
	}


	/**
	 * Below includes: 	
	 * * 1 bond <-> 1 bond correlators: pair_bb, jj, jsjs, kk, ksks (4 fermion, 2 phases)
	 * * (1 site)1 bond <-> (1 site)1 bond correlator: jnjn (8 fermion, 2 phases)
	 * * (1 site)1 bond <-> 1 bond correlators: jjn, jnj (6 fermion, 2 phases)
	 * * 2 hop bond <-> (1 site)1 bond correlators: j2jn, jnj2 (6 fermion, 3 phases) 
	 * * 2 hop bond <-> 1 bond correlators: j2j, jj2 (4 fermion, 3 phases) 
	 * * 2 hop bond <-> 2 hop bond correlators: j2j2 (4 fermion, 4 phases)
	 * * nematic correlators: nem_nnnn, nem_ssss (8 fermions, 2 phases) - not complexified
	 * TODO: implement pair_b2b2,js2js2,k2k2,ks2ks2
	 */

	/**
	 * minor optimization: handle t = 0 separately, since there are no delta
	 * functions for t > 0. not really needed in 2-site measurements above
	 * as those are fast anyway.
	 */
	
	//====================================================================
	// t = 0:
	// meas_bond_corr => pair_bb, jj, jsjs, kk, ksks (4 fermion, 2 phases)
	// meas_thermal   => jnjn                        (8 fermion, 2 phases)
	//                   jjn, jnj                    (6 fermion, 2 phases)
	if (meas_bond_corr || meas_thermal) {
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
				// t = 0: pair_bb, jj, jsjs, kk, ksks 
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

					m->jnj[bb] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);

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

					t1 = ( (delta_i0j0 - guj0i0) * gui1j0 + (delta_i0j1 - guj1i0) * gui1j1 ) * 
						pui0i1 * (pdj1j0 * gdj0j1 - pdj0j1 * gdj1j0);
					t2 = ( (delta_i1j0 - guj0i1) * gui0j0 + (delta_i1j1 - guj1i1) * gui0j1 ) * 
						pui1i0 * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1);
					t3 = ( (delta_i0j0 - gdj0i0) * gdi1j0 + (delta_i0j1 - gdj1i0) * gdi1j1 ) * 
						pdi0i1 * (puj1j0 * guj0j1 - puj0j1 * guj1j0);
					t4 = ( (delta_i1j0 - gdj0i1) * gdi0j0 + (delta_i1j1 - gdj1i1) * gdi0j1 ) * 
						pdi1i0 * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

					m->jjn[bb] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);
			
					//TODO further simplify this expression for faster measurements?
					//TODO: declare these constant earlier?
					const num _wick_jn_i = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
					 			   		   (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);

					const num _wick_jn_j = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
					 		   			   (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

					const num c1 = ( (delta_i0j0-guj0i0)*gui0j0 + (delta_i1j0-guj0i1)*gui1j0 
						           + (delta_i0j1-guj1i0)*gui0j1 + (delta_i1j1-guj1i1)*gui1j1 ) *
						( pdi0i1*pdj0j1 * ( gdi1i0*gdj1j0 + (delta_i0j1-gdj1i0)*gdi1j0 ) 
						 -pdi0i1*pdj1j0 * ( gdi1i0*gdj0j1 + (delta_i0j0-gdj0i0)*gdi1j1 ) 
						 -pdi1i0*pdj0j1 * ( gdi0i1*gdj1j0 + (delta_i1j1-gdj1i1)*gdi0j0 ) 
						 +pdi1i0*pdj1j0 * ( gdi0i1*gdj0j1 + (delta_i1j0-gdj0i1)*gdi0j1 ));

					const num c3 = ( (delta_i0j0-gdj0i0)*gdi0j0 + (delta_i1j0-gdj0i1)*gdi1j0 
						           + (delta_i0j1-gdj1i0)*gdi0j1 + (delta_i1j1-gdj1i1)*gdi1j1 ) * 
						(+pui0i1*puj0j1 * ( gui1i0*guj1j0 + (delta_i0j1-guj1i0)*gui1j0 )
						 -pui0i1*puj1j0 * ( gui1i0*guj0j1 + (delta_i0j0-guj0i0)*gui1j1 )
						 -pui1i0*puj0j1 * ( gui0i1*guj1j0 + (delta_i1j1-guj1i1)*gui0j0 )
						 +pui1i0*puj1j0 * ( gui0i1*guj0j1 + (delta_i1j0-guj0i1)*gui0j1 ));

					const num c2 = (2-gui0i0-gui1i1) * (2-guj0j0-guj1j1) *
						( pdi0i1*pdj0j1 * (delta_i0j1-gdj1i0)*gdi1j0 
						 -pdi0i1*pdj1j0 * (delta_i0j0-gdj0i0)*gdi1j1 
						 -pdi1i0*pdj0j1 * (delta_i1j1-gdj1i1)*gdi0j0 
						 +pdi1i0*pdj1j0 * (delta_i1j0-gdj0i1)*gdi0j1);

					const num c4 = (2-gdi0i0-gdi1i1) * (2-gdj0j0-gdj1j1) * 
						(+pui0i1*puj0j1 * (delta_i0j1-guj1i0)*gui1j0
						 -pui0i1*puj1j0 * (delta_i0j0-guj0i0)*gui1j1
						 -pui1i0*puj0j1 * (delta_i1j1-guj1i1)*gui0j0
						 +pui1i0*puj1j0 * (delta_i1j0-guj0i1)*gui0j1);

					const num b1 = 
						(+pdi0i1 * ( (-gdi1i0)*(2-gdj0j0-gdj1j1) + (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 ) 
						 -pdi1i0 * ( (-gdi0i1)*(2-gdj0j0-gdj1j1) + (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 )) * 
						(+puj0j1 * ( (delta_i0j1-guj1i0)*gui0j0 + (delta_i1j1-guj1i1)*gui1j0 )
						 -puj1j0 * ( (delta_i0j0-guj0i0)*gui0j1 + (delta_i1j0-guj0i1)*gui1j1 ));

					const num b3 = 
						(+pui0i1 * ( (-gui1i0)*(2-guj0j0-guj1j1) + (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 ) 
						 -pui1i0 * ( (-gui0i1)*(2-guj0j0-guj1j1) + (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 )) *
						(+pdj0j1 * ( (delta_i0j1-gdj1i0)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi1j0 ) 
						 -pdj1j0 * ( (delta_i0j0-gdj0i0)*gdi0j1 + (delta_i1j0-gdj0i1)*gdi1j1 ));

					const num b2 = (2-gui0i0-gui1i1) * (puj0j1*(-guj1j0)-puj1j0*(-guj0j1)) *
						(+pdi0i1 * ( (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 )
						 -pdi1i0 * ( (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 ));

					const num b4 = (2-gdi0i0-gdi1i1) * (pdj0j1*(-gdj1j0)-pdj1j0*(-gdj0j1)) *
						(+pui0i1 * ( (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 ) 
						 -pui1i0 * ( (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 ));

					m->jnjn[bb] += pre*(_wick_jn_i * _wick_jn_j + c1 + c2 + c3 + c4 + 
															      b1 + b2 + b3 + b4);
				}
			}
		}
	}

	// t = 0: meas_2bond_corr => j2j2 (4 fermion, 4 phases)
	if (meas_2bond_corr) {
		for (int c = 0; c < num_b2; c++) {
			const int jtype = c / N;
			const int j = c % N;

			const num ppuj0j2 = p->pp_u[ j + N*jtype];
			const num ppuj2j0 = p->ppr_u[j + N*jtype];
			const num ppdj0j2 = p->pp_d[ j + N*jtype];
			const num ppdj2j0 = p->ppr_d[j + N*jtype];

			const int j0 = p->bond2s[c];
			const int j2 = p->bond2s[c + num_b2];
			for (int b = 0; b < num_b2; b++) {
				const int itype = b / N;
				const int i = b % N;

				const num ppui0i2 = p->pp_u[ i + N*itype];
				const num ppui2i0 = p->ppr_u[i + N*itype];
				const num ppdi0i2 = p->pp_d[ i + N*itype];
				const num ppdi2i0 = p->ppr_d[i + N*itype];

				const int i0 = p->bond2s[b];
				const int i2 = p->bond2s[b + num_b2];

				const int bb = p->map_b2b2[b + c*num_b2];
				const num pre = phase / p->degen_b2b2[bb];

				const int delta_i0j0 = (i0 == j0);
				const int delta_i2j0 = (i2 == j0);
				const int delta_i0j2 = (i0 == j2);
				const int delta_i2j2 = (i2 == j2);
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
                                
                                m->pair_b2b2[bb] += 0.5*pre*(gui0j0*gdi2j2 + gui2j0*gdi0j2 + gui0j2*gdi2j0 + gui2j2*gdi0j0);
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
			      m->js2js2[bb] += pre*((ppui2i0*gui0i2 - ppui0i2*gui2i0 - ppdi2i0*gdi0i2 + ppdi0i2*gdi2i0)
					           *(ppuj2j0*guj0j2 - ppuj0j2*guj2j0 - ppdj2j0*gdj0j2 + ppdj0j2*gdj2j0)
                                      + x - y);
			      m->k2k2[bb]  += pre*((ppui2i0*gui0i2 + ppui0i2*gui2i0 + ppdi2i0*gdi0i2 + ppdi0i2*gdi2i0)
					           *(ppuj2j0*guj0j2 + ppuj0j2*guj2j0 + ppdj2j0*gdj0j2 + ppdj0j2*gdj2j0)
                                      + x + y);
			      m->ks2ks2[bb] += pre*((ppui2i0*gui0i2 + ppui0i2*gui2i0 - ppdi2i0*gdi0i2 - ppdi0i2*gdi2i0)
					          *(ppuj2j0*guj0j2 + ppuj0j2*guj2j0 - ppdj2j0*gdj0j2 - ppdj0j2*gdj2j0)
                                      + x + y);
			}
		}
	}

	// t = 0: 
	// meas_thermal    => jnj2 (6 fermion, 3 phases) 
	// meas_2bond_corr => jj2  (4 fermion, 3 phases) 
	// i = i0 <-> i1
	// j = j0 <-> j1 <-> j2
	// Essentially matrix[j,i] = bond(i) x bond2(j) TODO check this?
	if (meas_thermal || meas_2bond_corr) {
		for (int c = 0; c < num_b2; c++) {
			const int jtype = c / N;
			const int j = c % N;

			const num ppuj0j2 = p->pp_u[ j + N*jtype];
			const num ppuj2j0 = p->ppr_u[j + N*jtype];
			const num ppdj0j2 = p->pp_d[ j + N*jtype];
			const num ppdj2j0 = p->ppr_d[j + N*jtype];

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

				//jn(i0i1)-j2(j0j1j2): 6 fermion product, 3 phases, t = 0
				if (meas_thermal){
					const num gui0i0 = Gu00[i0 + i0*N];
					const num gui1i1 = Gu00[i1 + i1*N];
					const num gdi0i0 = Gd00[i0 + i0*N];
					const num gdi1i1 = Gd00[i1 + i1*N];
					
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
				//j(i0i1) -j2(j0j1j2): 4 fermion product, 3 phases, t = 0
				if (meas_2bond_corr) {
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
	}

	// t = 0: 
	// meas_thermal    => j2jn (6 fermion, 3 phases) 
	// meas_2bond_corr => j2j  (4 fermion, 3 phases) 
	// i = i0 <-> i1 <-> i2
	// j = j0 <-> j1 
	// Essentially matrix[j,i] = bond2(i) x bond(j) TODO check this?
	if (meas_thermal || meas_2bond_corr) {
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

				const num ppui0i2 = p->pp_u[ i + N*itype];
				const num ppui2i0 = p->ppr_u[i + N*itype];
				const num ppdi0i2 = p->pp_d[ i + N*itype];
				const num ppdi2i0 = p->ppr_d[i + N*itype];
				// printf("ppui0i2 = %f\n",ppui0i2);

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
				//j2(i0i1i2)-jn(j0j1): 6 fermion product, 3 phases, t = 0
				if (meas_thermal) {
					const num guj0j0 = Gu00[j0 + j0*N];
					const num guj1j1 = Gu00[j1 + j1*N];
					const num gdj0j0 = Gd00[j0 + j0*N];
					const num gdj1j1 = Gd00[j1 + j1*N];

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
				//j2(i0i1i2)- j(j0j1): 4 fermion product, 3 phases, t = 0
				if (meas_2bond_corr) {
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
	}

#ifdef USE_PEIERLS

#else
	// NOTE: Peierls phase not implemented!
	// t = 0: meas_nematic_corr => nem_nnnn, nem_ssss
	if (meas_nematic_corr) {
		for (int c = 0; c < NEM_BONDS*N; c++) {
			const int j0 = p->bonds[c];
			const int j1 = p->bonds[c + num_b];
			for (int b = 0; b < NEM_BONDS*N; b++) {
				const int i0 = p->bonds[b];
				const int i1 = p->bonds[b + num_b];
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
	}
#endif

	//=======================================================================
	// now handle t > 0 case: no delta functions here.
	profile_begin(meas_uneq_sub);

	if (meas_bond_corr || meas_thermal || meas_2bond_corr) {
		#pragma omp parallel for num_threads(OMP_MEAS_NUM_THREADS) 
		for (int t = 1; t < L; t++) {
			//int id = omp_get_thread_num();
			//printf("Hello from thread %d out of %d threads\n", id, OMP_MEAS_NUM_THREADS);
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
				// t <- [1, L):
				// meas_bond_corr => pair_bb, jj, jsjs, kk, ksks (4 fermion, 2 phases)
				// meas_thermal   => jnjn                        (8 fermion, 2 phases)
				//                   jjn, jnj                    (6 fermion, 2 phases)
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
					// jnj, jjn, jnjn
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

						//j(i0i1)jn(j0j1), 6 fermion product, 2 phases, t > 0
						m->jnj[bb + num_bb*t] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);

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


						m->jjn[bb + num_bb*t] += pre*(_wick_j * _wick_jn + t1 + t2 + t3 + t4 + t5 + t6);

						// thermal: jnjn, t > 0. TODO: simplify this expression for faster measurements
						const num _wick_jn_i = (2 - gui0i0 - gui1i1) * (pdi0i1 * gdi1i0 - pdi1i0 * gdi0i1) + 
						 			   		   (2 - gdi0i0 - gdi1i1) * (pui0i1 * gui1i0 - pui1i0 * gui0i1);

						const num _wick_jn_j = (2 - guj0j0 - guj1j1) * (pdj0j1 * gdj1j0 - pdj1j0 * gdj0j1) + 
						 		   			   (2 - gdj0j0 - gdj1j1) * (puj0j1 * guj1j0 - puj1j0 * guj0j1);

						const num c1 = ( (delta_i0j0-guj0i0)*gui0j0 + (delta_i1j0-guj0i1)*gui1j0 
							           + (delta_i0j1-guj1i0)*gui0j1 + (delta_i1j1-guj1i1)*gui1j1 ) *
							( pdi0i1*pdj0j1 * ( gdi1i0*gdj1j0 + (delta_i0j1-gdj1i0)*gdi1j0 ) 
							 -pdi0i1*pdj1j0 * ( gdi1i0*gdj0j1 + (delta_i0j0-gdj0i0)*gdi1j1 ) 
							 -pdi1i0*pdj0j1 * ( gdi0i1*gdj1j0 + (delta_i1j1-gdj1i1)*gdi0j0 ) 
							 +pdi1i0*pdj1j0 * ( gdi0i1*gdj0j1 + (delta_i1j0-gdj0i1)*gdi0j1 ));

						const num c3 = ( (delta_i0j0-gdj0i0)*gdi0j0 + (delta_i1j0-gdj0i1)*gdi1j0 
							           + (delta_i0j1-gdj1i0)*gdi0j1 + (delta_i1j1-gdj1i1)*gdi1j1 ) * 
							(+pui0i1*puj0j1 * ( gui1i0*guj1j0 + (delta_i0j1-guj1i0)*gui1j0 )
							 -pui0i1*puj1j0 * ( gui1i0*guj0j1 + (delta_i0j0-guj0i0)*gui1j1 )
							 -pui1i0*puj0j1 * ( gui0i1*guj1j0 + (delta_i1j1-guj1i1)*gui0j0 )
							 +pui1i0*puj1j0 * ( gui0i1*guj0j1 + (delta_i1j0-guj0i1)*gui0j1 ));

						const num c2 = (2-gui0i0-gui1i1) * (2-guj0j0-guj1j1) *
							( pdi0i1*pdj0j1 * (delta_i0j1-gdj1i0)*gdi1j0 
							 -pdi0i1*pdj1j0 * (delta_i0j0-gdj0i0)*gdi1j1 
							 -pdi1i0*pdj0j1 * (delta_i1j1-gdj1i1)*gdi0j0 
							 +pdi1i0*pdj1j0 * (delta_i1j0-gdj0i1)*gdi0j1);

						const num c4 = (2-gdi0i0-gdi1i1) * (2-gdj0j0-gdj1j1) * 
							(+pui0i1*puj0j1 * (delta_i0j1-guj1i0)*gui1j0
							 -pui0i1*puj1j0 * (delta_i0j0-guj0i0)*gui1j1
							 -pui1i0*puj0j1 * (delta_i1j1-guj1i1)*gui0j0
							 +pui1i0*puj1j0 * (delta_i1j0-guj0i1)*gui0j1);

						const num b1 = 
							(+pdi0i1 * ( (-gdi1i0)*(2-gdj0j0-gdj1j1) + (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 ) 
							 -pdi1i0 * ( (-gdi0i1)*(2-gdj0j0-gdj1j1) + (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 )) * 
							(+puj0j1 * ( (delta_i0j1-guj1i0)*gui0j0 + (delta_i1j1-guj1i1)*gui1j0 )
							 -puj1j0 * ( (delta_i0j0-guj0i0)*gui0j1 + (delta_i1j0-guj0i1)*gui1j1 ));

						const num b3 = 
							(+pui0i1 * ( (-gui1i0)*(2-guj0j0-guj1j1) + (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 ) 
							 -pui1i0 * ( (-gui0i1)*(2-guj0j0-guj1j1) + (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 )) *
							(+pdj0j1 * ( (delta_i0j1-gdj1i0)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi1j0 ) 
							 -pdj1j0 * ( (delta_i0j0-gdj0i0)*gdi0j1 + (delta_i1j0-gdj0i1)*gdi1j1 ));

						const num b2 = (2-gui0i0-gui1i1) * (puj0j1*(-guj1j0)-puj1j0*(-guj0j1)) *
							(+pdi0i1 * ( (delta_i0j0-gdj0i0)*gdi1j0 + (delta_i0j1-gdj1i0)*gdi1j1 )
							 -pdi1i0 * ( (delta_i1j0-gdj0i1)*gdi0j0 + (delta_i1j1-gdj1i1)*gdi0j1 ));

						const num b4 = (2-gdi0i0-gdi1i1) * (pdj0j1*(-gdj1j0)-pdj1j0*(-gdj0j1)) *
							(+pui0i1 * ( (delta_i0j0-guj0i0)*gui1j0 + (delta_i0j1-guj1i0)*gui1j1 ) 
							 -pui1i0 * ( (delta_i1j0-guj0i1)*gui0j0 + (delta_i1j1-guj1i1)*gui0j1 ));

						m->jnjn[bb + num_bb*t] += pre*(_wick_jn_i * _wick_jn_j 
							+ c1 + c2 + c3 + c4
							+ b1 + b2 + b3 + b4);

					}

				}
				// t <- [1, L):
				// meas_thermal    => j2jn (6 fermion, 3 phases) 
				// meas_2bond_corr => j2j  (4 fermion, 3 phases) 
				// i = i0 <-> i1 <-> i2
				// j = j0 <-> j1 
				// Essentially matrix[j,i] = bond2(i) x bond(j) TODO check this?
				for (int b = 0; b < num_b2; b++) {
					const int itype = b / N;
					const int i = b % N;

					const num ppui0i2 = p->pp_u[ i + N*itype];
					const num ppui2i0 = p->ppr_u[i + N*itype];
					const num ppdi0i2 = p->pp_d[ i + N*itype];
					const num ppdi2i0 = p->ppr_d[i + N*itype];

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
		
			
			for (int c = 0; c < num_b2; c++) {
				const int jtype = c / N;
				const int j = c % N;

				const num ppuj0j2 = p->pp_u[ j + N*jtype];
				const num ppuj2j0 = p->ppr_u[j + N*jtype];
				const num ppdj0j2 = p->pp_d[ j + N*jtype];
				const num ppdj2j0 = p->ppr_d[j + N*jtype];

				const int j0 = p->bond2s[c];
				const int j2 = p->bond2s[c + num_b2];
				// t <- [1, L): meas_2bond_corr => j2j2 (4 fermion, 4 phases)
				if (meas_2bond_corr) {
					for (int b = 0; b < num_b2; b++) {
						const int itype = b / N;
						const int i = b % N;

						const num ppui0i2 = p->pp_u[ i + N*itype];
						const num ppui2i0 = p->ppr_u[i + N*itype];
						const num ppdi0i2 = p->pp_d[ i + N*itype];
						const num ppdi2i0 = p->ppr_d[i + N*itype];

						const int i0 = p->bond2s[b];
						const int i2 = p->bond2s[b + num_b2];

						const int bb = p->map_b2b2[b + c*num_b2];
						const num pre = phase / p->degen_b2b2[bb];

						const int delta_i0j0 = 0;
						const int delta_i2j0 = 0;
						const int delta_i0j2 = 0;
						const int delta_i2j2 = 0;
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

                                                m->pair_b2b2[bb + num_b2b2*t] += 0.5*pre*(gui0j0*gdi2j2 + gui2j0*gdi0j2 + gui0j2*gdi2j0 + gui2j2*gdi0j0);
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
                                      m->js2js2[bb + num_b2b2*t] += 
                                            pre*((ppui2i0*gui0i2 - ppui0i2*gui2i0 - ppdi2i0*gdi0i2 + ppdi0i2*gdi2i0)
					              *(ppuj2j0*guj0j2 - ppuj0j2*guj2j0 - ppdj2j0*gdj0j2 + ppdj0j2*gdj2j0)
                                              + x - y);
			                   m->k2k2[bb + num_b2b2*t]  += 
                                            pre*((ppui2i0*gui0i2 + ppui0i2*gui2i0 + ppdi2i0*gdi0i2 + ppdi0i2*gdi2i0)
					             *(ppuj2j0*guj0j2 + ppuj0j2*guj2j0 + ppdj2j0*gdj0j2 + ppdj0j2*gdj2j0)
                                             + x + y);
			                   m->ks2ks2[bb + num_b2b2*t] += 
                                             pre*((ppui2i0*gui0i2 + ppui0i2*gui2i0 - ppdi2i0*gdi0i2 - ppdi0i2*gdi2i0)
					             *(ppuj2j0*guj0j2 + ppuj0j2*guj2j0 - ppdj2j0*gdj0j2 - ppdj0j2*gdj2j0)
                                             + x + y);

					}
				}
				// t <- [1, L): 
				// meas_thermal    => jnj2 (6 fermion, 3 phases) 
				// meas_2bond_corr => jj2  (4 fermion, 3 phases) 
				// i = i0 <-> i1
				// j = j0 <-> j1 <-> j2
				// Essentially matrix[j,i] = bond(i) x bond2(j) TODO check this?
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
	}
	
	profile_end(meas_uneq_sub);


#ifdef USE_PEIERLS
	// NOTE: Peierls phase not implemented for nematic measurements
	if (meas_nematic_corr) {
	printf("error: peierls phase not implemented for nematic measurements\n");
	fflush(stdout);
	exit(EXIT_FAILURE);
}
#else
	
	// t <- [0, L): meas_nematic_corr => nem_nnnn, nem_ssss
	if (meas_nematic_corr) {
		meas_uneqlt_nematic(p, phase, Gu0t, Gutt, Gut0, Gd0t, Gdtt,  Gdt0, m);
	}
#endif
}
