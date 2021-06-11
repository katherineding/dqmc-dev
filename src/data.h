#pragma once

#include <stdint.h>
#include "util.h"

struct params {
	int N, L;
	int *map_i, *map_ij;
	int *bonds, *bond2s, *map_bs, *map_bb, *map_b2b, *map_bb2, *map_b2b2;
	int *hop2s, *map_hop2_b, *map_b_hop2, *map_hop2_hop2;
	num *peierlsu, *peierlsd;
	num *pp_u, *pp_d, *ppr_u, *ppr_d;
//	double *K, *U;
//	double dt;

	int n_matmul, n_delay;
	int n_sweep_warm, n_sweep_meas;
	int period_eqlt, period_uneqlt;
	int meas_bond_corr, meas_thermal, meas_2bond_corr, meas_energy_corr, meas_nematic_corr;
	int meas_hop2_corr;

	int num_i, num_ij;
	int num_b, num_b2, num_bs, num_bb, num_bb2, num_b2b, num_b2b2;
	int num_hop2, num_b_hop2, num_hop2_b, num_hop2_hop2;
	int *degen_i, *degen_ij, *degen_bs, *degen_bb, *degen_b2b2, *degen_b2b, *degen_bb2;
	int *degen_hop2_hop2, *degen_hop2_b, *degen_b_hop2;
	num *exp_Ku, *exp_Kd, *inv_exp_Ku, *inv_exp_Kd;
	num *exp_halfKu, *exp_halfKd, *inv_exp_halfKu, *inv_exp_halfKd;
	double *exp_lambda, *del;
	int F, n_sweep;
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
	num *double_occ;

	num *g00;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *kk, *kv, *kn, *vv, *vn;
};

struct meas_uneqlt {
	int n_sample;
	num sign;

	num *gt0;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *pair_bb;
	num *jj, *jsjs;
	num *kk, *ksks;
	num *pair_b2b2;
	num *jjn, *jnj, *jnjn;
	num *j2j2, *js2js2;
	num *k2k2, *ks2ks2;
	num *J2J2, *J2jn, *J2j, *jnJ2, *jJ2, *new_jnj, *new_jjn;
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

int sim_data_read_alloc(struct sim_data *sim, const char *file);

int sim_data_save(const struct sim_data *sim);

void sim_data_free(const struct sim_data *sim);
