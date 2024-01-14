#pragma once

#include <stdint.h>
#include <stdio.h>
#include "util.h"

struct params {
	int N, L;
	int *map_i, *map_ij;
	int *bonds, *bond2s, *map_bs, *map_bb, *map_b2b, *map_bb2, *map_b2b2;
	int *plaqs;
	int *map_plaq;
	num *peierlsu, *peierlsd;
	num *pp_u, *pp_d, *ppr_u, *ppr_d;
	int *map_b, *map_b2;
//	double dt;

	int n_matmul, n_delay;
	int n_sweep_warm, n_sweep_meas;
	int period_eqlt, period_uneqlt;
	int meas_bond_corr, meas_thermal, meas_2bond_corr, meas_energy_corr, meas_nematic_corr;
	int meas_chiral;
	int meas_local_JQ;
	int checkpoint_every;

	int num_i, num_ij;
	int num_plaq_accum, num_plaq;
	int num_b, num_b2, num_bs, num_bb, num_bb2, num_b2b, num_b2b2;
	int *degen_plaq;
	int *degen_i, *degen_ij, *degen_bs, *degen_bb, *degen_b2b2, *degen_b2b, *degen_bb2;
	num *exp_Ku, *exp_Kd, *inv_exp_Ku, *inv_exp_Kd;
	num *exp_halfKu, *exp_halfKd, *inv_exp_halfKu, *inv_exp_halfKd;
	double *exp_lambda, *del;
	int F, n_sweep;
	int *degen_b, *degen_b2; 
};

struct state {
	uint64_t rng[17];
	int sweep;
	int *hs;
};

struct meas_eqlt {
	int n_sample;
	num sign;

	num *density;
	num *density_u;
	num *density_d;
	num *double_occ;

	num *chi;
	num *g00;
	num *g00_u;
	num *g00_d;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *kk, *kv, *kn, *vv, *vn;
	num *j, *j2, *jn;
};

struct meas_uneqlt {
	int n_sample;
	num sign;

	num *gt0;
	num *gt0_u;
	num *gt0_d;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *pair_bb;
	num *jj, *jsjs;
	num *kk, *ksks;
	num *pair_b2b2;
	num *j2jn, *jnj2, *jnjn;
	num *j2j,  *jj2;
	num *j2j2, *js2js2;
	num *k2k2, *ks2ks2;
	num *jnj, *jjn;
	num *kv, *kn, *vv, *vn;
	num *nem_nnnn, *nem_ssss;
};

struct sim_data {
	const char *file;
	struct params p;
	struct state s;
	struct meas_eqlt m_eq;
	struct meas_uneqlt m_ue;
};

int set_num_h5t(void);

int get_memory_req(const char *file);

int consistency_check(const char *file, FILE * log);

int sim_data_read_alloc(struct sim_data *sim);

int sim_data_save(const struct sim_data *sim);

void sim_data_free(const struct sim_data *sim);
