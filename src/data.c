#include <stdio.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <string.h>
#include "greens.h"
#include <assert.h>

#include "data.h"

#define OPEN_FAIL -1
#define CORRUPT_FAIL -2
#define RW_FAIL -3
#define CLOSE_FAIL -4

#define return_if(cond, val, ...) \
	do {if (cond) {fprintf(stderr, __VA_ARGS__); return (val);}} while (0)

#define my_read(_type, name, ...) do { \
	status = H5LTread_dataset##_type(file_id, (name), __VA_ARGS__); \
	return_if(status < 0, RW_FAIL, "H5LTread_dataset() failed for %s: %d\n", (name), status); \
} while (0);

#define my_write(name, type, data) do { \
	dset_id = H5Dopen2(file_id, (name), H5P_DEFAULT); \
	return_if(dset_id < 0, RW_FAIL, "H5Dopen2() failed for %s: %ld\n", name, dset_id); \
	status = H5Dwrite(dset_id, (type), H5S_ALL, H5S_ALL, H5P_DEFAULT, (data)); \
	return_if(status < 0, RW_FAIL, "H5Dwrite() failed for %s: %d\n", name, status); \
	status = H5Dclose(dset_id); \
	return_if(status < 0, RW_FAIL, "H5Dclose() failed for %s: %d\n", name, status); \
} while (0);

static hid_t num_h5t = 0;
/**
 * If USE_COMPLEX, set type num_h5t to be a H5T_COMPOUND of two doubles. 
 * otherwise, set to H5T_NATIVE_DOUBLE.
 * This should be called once per ./dqmc_stack or ./dqmc_1 invocation, in the 
 * main() loop, to avoid repeatedly creating new complex datatypes.
 * Note H5Tclose() is not called so we have a memory leak, but this is not 
 * serious. TODO fix it anyways.
 * @return -1/abort(?) on failure
 *          0 on success.
 */
int set_num_h5t(void){
	int status;
#ifdef USE_CPLX
	num_h5t = H5Tcreate(H5T_COMPOUND, sizeof(num));
	return_if(num_h5t < 0, -1, "H5Tcreate() failed\n");
	status = H5Tinsert(num_h5t, "r", 0, H5T_NATIVE_DOUBLE);
	return_if(status < 0, -1, "H5Tinsert() failed: %d\n", status);
	status = H5Tinsert(num_h5t, "i", 8, H5T_NATIVE_DOUBLE);
	return_if(status < 0, -1, "H5Tinsert() failed: %d\n", status);
#else
	num_h5t = H5T_NATIVE_DOUBLE;
#endif
	return_if(sizeof(num) != H5Tget_size(num_h5t), -1, 
		"num_h5t and num inconsistent\n");
	return 0;
}

/**
 * Check hdf5 against executable version
 * @param  char* file_name of form <name>.h5
 * @return 0 if consistent
 *         nonzero if inconsistent
 * 		   -1 if H5Fopen or type-setting failed
 *         -3 if my_read failed
 *         -4 if H5F/Tclose() failed
 */
int consistency_check(const char *file, FILE * log){
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, OPEN_FAIL, "H5Fopen() failed for %s: %ld\n", 
		file, file_id);

	// Variable for error checking
	int status;

	hid_t attr_type = H5Tcopy(H5T_C_S1);
	return_if(attr_type < 0, OPEN_FAIL, "H5Tcopy() failed\n");
  	status = H5Tset_size(attr_type, H5T_VARIABLE);
  	return_if(status < 0, OPEN_FAIL, "H5Tset_size() failed\n");
  	status = H5Tset_strpad(attr_type, H5T_STR_NULLTERM);
  	return_if(status < 0, OPEN_FAIL, "H5Tset_strpad() failed\n");
  	status = H5Tset_cset(attr_type, H5T_CSET_UTF8);
  	return_if(status < 0, OPEN_FAIL, "H5Tset_cset() failed\n");

  	// HDF5 C API for reading variable length strings is ... quirky
	char * rdata[1] = {NULL};
	my_read(, "/metadata/commit" , attr_type, rdata);
	fprintf(log,"hdf5 generation script commit id %s\n", rdata[0]);

	//prevent mem leaks
	status = H5Tclose(attr_type);
	return_if(status < 0, CLOSE_FAIL, "H5Tclose() failed");
	status = H5Fclose(file_id);
	return_if(status < 0, CLOSE_FAIL, "H5Fclose() failed for %s: %d\n", 
		file, status);

	return strcmp(rdata[0],GIT_ID);
}

/**
 * Calculate the amount of heap memory required, without actually mallocing, 
 * if we are to run DQMC against this file. This is an estimate and doesn't 
 * account for -DCHECK_G_WRP, -DCHECK_G_ACC -DCHECK_G_UE
 * TODO: separate the sim vs calculation components?
 * @param  char*  file_name of form <name>.h5
 * @return int heap memory requirement
 *         -1 if H5Fopen() failed
 *         -3 if my_read failed
 *         -4 if H5Fclose() failed
 */
int get_memory_req(const char *file) {
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, OPEN_FAIL, "H5Fopen() failed for %s: %ld\n", 
		file, file_id);

	// Variable for error checking
	int status;

	// These params are used in sim_data mem allocation
	int N, L, num_i, num_ij,
		num_b, num_b2, num_bs, num_bb, num_b2b, num_bb2, num_b2b2,
		num_plaq_accum, num_b_accum, num_b2_accum, num_plaq;
	int period_uneqlt, meas_bond_corr,meas_thermal,meas_2bond_corr,
		meas_energy_corr, meas_nematic_corr, meas_chiral, meas_local_JQ, meas_gen_suscept;
	int meas_pair_bb_only;

	my_read(_int, "/params/N",        &N);
	my_read(_int, "/params/L",        &L);
	my_read(_int, "/params/num_i",    &num_i);
	my_read(_int, "/params/num_ij",   &num_ij);
	my_read(_int, "/params/num_plaq_accum",   &num_plaq_accum);
	my_read(_int, "/params/num_b_accum",  &num_b_accum);
	my_read(_int, "/params/num_b2_accum",  &num_b2_accum);
	my_read(_int, "/params/num_plaq",   &num_plaq);
	my_read(_int, "/params/num_b",    &num_b);
	my_read(_int, "/params/num_b2",   &num_b2);
	my_read(_int, "/params/num_bs",   &num_bs);
	my_read(_int, "/params/num_bb",   &num_bb);
	my_read(_int, "/params/num_b2b",  &num_b2b);
	my_read(_int, "/params/num_bb2",  &num_bb2);
	my_read(_int, "/params/num_b2b2", &num_b2b2);
	my_read(_int, "/params/period_uneqlt",     &period_uneqlt);
	my_read(_int, "/params/meas_bond_corr",    &meas_bond_corr);
	my_read(_int, "/params/meas_thermal",      &meas_thermal);
	my_read(_int, "/params/meas_2bond_corr",   &meas_2bond_corr);
	my_read(_int, "/params/meas_energy_corr",  &meas_energy_corr);
	my_read(_int, "/params/meas_local_JQ",     &meas_local_JQ);
	my_read(_int, "/params/meas_gen_suscept",  &meas_gen_suscept);
	my_read(_int, "/params/meas_nematic_corr", &meas_nematic_corr);
	my_read(_int, "/params/meas_pair_bb_only", &meas_pair_bb_only);
	my_read(_int, "/params/meas_chiral",       &meas_chiral);

	if (meas_pair_bb_only) {
		assert(meas_thermal==0);
		assert(meas_bond_corr==0);
		assert(meas_energy_corr==0);
		assert(meas_2bond_corr==0);
		assert(meas_nematic_corr==0);
		assert(meas_gen_suscept==0);
	}

	size_t sim_alloc_in_bytes = 0;
	sim_alloc_in_bytes +=
		+ N        * sizeof(int)
		+ N*N      * sizeof(int)
		+ num_b*2  * sizeof(int)
		+ num_b2*2  * sizeof(int)
		+ num_plaq*3* sizeof(int)
		+ num_plaq  * sizeof(int)
		+ num_b  * sizeof(int)
		+ num_b2  * sizeof(int)
		+ num_b*N  * sizeof(int)
		+ num_b*num_b * sizeof(int)
		+ num_b2*num_b2 * sizeof(int)
		+ num_b2*num_b * sizeof(int)
		+ num_b*num_b2 * sizeof(int)
		+ N*N      * sizeof(num)
		+ N*N      * sizeof(num)
		+ num_b2 * sizeof(num)
		+ num_b2 * sizeof(num)
		+ num_b2 * sizeof(num)
		+ num_b2 * sizeof(num)
		+ N*N*8    * sizeof(num)
		+ N*2      * sizeof(double)
		+ N*2      * sizeof(double)
		+ N*L      * sizeof(int)
		+ num_i*4  * sizeof(num)
		+ num_ij*7 * sizeof(num);
	if (meas_chiral) {
		sim_alloc_in_bytes += num_plaq_accum * sizeof(num);
	}
	if (meas_energy_corr) {
		sim_alloc_in_bytes +=
			+ num_bb * sizeof(num)
			+ num_bs * sizeof(num)
		 	+ num_bs * sizeof(num)
			+ num_ij * sizeof(num)
		 	+ num_ij * sizeof(num);
	}
	if (meas_local_JQ) {
		sim_alloc_in_bytes +=
			+ num_b_accum  * sizeof(num)
			+ num_b_accum  * sizeof(num)
			+ num_b2_accum * sizeof(num);
	}
	if (meas_gen_suscept) {
		sim_alloc_in_bytes +=
			+ num_ij * num_ij * sizeof(num) * 4;
	}
	if (period_uneqlt > 0) {
		sim_alloc_in_bytes +=
			+ num_ij*L*7 * sizeof(num);
		if (meas_pair_bb_only) {
			sim_alloc_in_bytes +=
			+ num_bb*L*1* sizeof(num);
		}
		if (meas_bond_corr) {
			sim_alloc_in_bytes +=
			+ num_bb*L*5 * sizeof(num);
		}
		if (meas_gen_suscept) {
			sim_alloc_in_bytes +=
				+ num_ij * num_ij * sizeof(num) * L * 4;
		}
		if (meas_thermal) {
			sim_alloc_in_bytes +=
			+ num_b2b*L * sizeof(num)
			+ num_bb2*L * sizeof(num)
			+ num_bb*L * sizeof(num)
			+ num_bb*L * sizeof(num)
			+ num_bb*L * sizeof(num);
		}
		if (meas_2bond_corr) {
			sim_alloc_in_bytes +=
			+ num_b2b2*L * sizeof(num)
			+ num_b2b2*L * sizeof(num)
			+ num_b2b*L * sizeof(num)
			+ num_bb2*L * sizeof(num)
			+ num_b2b2*L * sizeof(num)
			+ num_b2b2*L * sizeof(num)
			+ num_b2b2*L * sizeof(num);
		}
		if (meas_energy_corr) {
			sim_alloc_in_bytes +=
			+ num_bs*L * sizeof(num)
			+ num_bs*L * sizeof(num)
			+ num_ij*L * sizeof(num)
			+ num_ij*L * sizeof(num);
		}
		if (meas_nematic_corr) {
			sim_alloc_in_bytes +=
			+ num_bb*L * sizeof(num)
			+ num_bb*L * sizeof(num);
		}
	}

	// These params are used in compute mem allocation
	int F, n_matmul, n_delay;
	my_read(_int, "/params/F",        &F);
	my_read(_int, "/params/n_matmul", &n_matmul);
	my_read(_int, "/params/n_delay",  &n_delay);

	status = H5Fclose(file_id);
	return_if(status < 0, CLOSE_FAIL, "H5Fclose() failed for %s: %d\n", 
		file, status);

	size_t compute_alloc_in_bytes = 0;
	compute_alloc_in_bytes +=
		+ N*N*L * sizeof(num)
		+ N*N*L * sizeof(num)
		+ N*N*L * sizeof(num)
		+ N*N*L * sizeof(num)
		+ N*N*F * sizeof(num)
		+ N*N*F * sizeof(num)
		+ N*N * sizeof(num)
		+ N*N * sizeof(num)
		+ N * sizeof(double)

		/* work arrays for calc_eq_g and stuff.
		 two sets for easy 2x parallelization */
		+ N*N * sizeof(num)
		+ N*N * sizeof(num)
		+ N * sizeof(num)
		+ N * sizeof(num)
		+ N * sizeof(num)
		+ N * sizeof(int)

		+ N*N * sizeof(num)
		+ N*N * sizeof(num)
		+ N * sizeof(num)
		+ N * sizeof(num)
		+ N * sizeof(num)
		+ N * sizeof(int);


	if (period_uneqlt > 0) {
		const int E = 1 + (F - 1) / N_MUL;

		compute_alloc_in_bytes +=
			+ N*E*N*E * sizeof(num)
			+ N*E * sizeof(num)
			+ 4*N*N * sizeof(num)

			+ N*E*N*E * sizeof(num)
			+ N*E * sizeof(num)
			+ 4*N*N * sizeof(num)

			+ N*N*L * sizeof(num)
			+ N*N*L * sizeof(num)
			+ N*N*L * sizeof(num)
			+ N*N*L * sizeof(num)
			+ N*N*L * sizeof(num)
			+ N*N*L * sizeof(num);
	}

	// lapack work arrays
	int lwork = get_lwork_eq_g(N);
	if (period_uneqlt > 0) {
		const int E = 1 + (F - 1) / N_MUL;
		const int lwork_ue = get_lwork_ue_g(N, E);
		if (lwork_ue > lwork) lwork = lwork_ue;
	}

	compute_alloc_in_bytes += 
		+ lwork * sizeof(num)
		+ lwork * sizeof(num);

	// printf("sim alloc: %zu bytes \n", sim_alloc_in_bytes);
	// printf("compute alloc: %zu bytes \n", compute_alloc_in_bytes);

	const size_t total_mem = sim_alloc_in_bytes + compute_alloc_in_bytes;

	return (int) total_mem;
}

/**
 * Read data from sim->file and place into sim->p, sim->s, sim->m_eq, sim->m_ue
 * @param  sim  [description]
 * @return -1 if H5Fopen() failed
 *         -2 if file is corrupted
 *         -3 if my_read failed
 *         -4 if H5Fclose() failed
 */
int sim_data_read_alloc(struct sim_data *sim) {
	const hid_t file_id = H5Fopen(sim->file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, OPEN_FAIL, "H5Fopen() failed for %s: %ld\n", 
		sim->file, file_id);

	int status; //see my_read macro
	int partial_write;

	// Check if we are starting from a valid file
	my_read(_int,"/state/partial_write", &partial_write);
	if (partial_write) {
		return_if(partial_write, CORRUPT_FAIL, \
			"Last checkpoint in %s contained partial write\n", sim->file);
	}
	my_read(_int, "/params/N",      &sim->p.N);
	my_read(_int, "/params/L",      &sim->p.L);
	my_read(_int, "/params/Nx",      &sim->p.Nx);
	my_read(_int, "/params/Ny",      &sim->p.Ny);
	my_read(_int, "/params/num_i",  &sim->p.num_i);
	my_read(_int, "/params/num_ij", &sim->p.num_ij);
	my_read(_int, "/params/num_plaq_accum",  &sim->p.num_plaq_accum);
	my_read(_int, "/params/num_b_accum",  &sim->p.num_b_accum);
	my_read(_int, "/params/num_b2_accum",  &sim->p.num_b2_accum);
	my_read(_int, "/params/num_plaq",&sim->p.num_plaq);
	my_read(_int, "/params/num_b", &sim->p.num_b);
	my_read(_int, "/params/num_b2", &sim->p.num_b2);
	my_read(_int, "/params/num_bs", &sim->p.num_bs);
	my_read(_int, "/params/num_bb", &sim->p.num_bb);
	my_read(_int, "/params/num_b2b", &sim->p.num_b2b);
	my_read(_int, "/params/num_bb2", &sim->p.num_bb2);
	my_read(_int, "/params/num_b2b2", &sim->p.num_b2b2);

	my_read(_int, "/params/degen_i",&sim->p.degen_i);
	my_read(_int, "/params/degen_ij",&sim->p.degen_ij);
	my_read(_int, "/params/degen_plaq",&sim->p.degen_plaq);
	my_read(_int, "/params/degen_b", &sim->p.degen_b);
	my_read(_int, "/params/degen_b2", &sim->p.degen_b2);
	my_read(_int, "/params/degen_bs", &sim->p.degen_bs);
	my_read(_int, "/params/degen_bb", &sim->p.degen_bb);
	my_read(_int, "/params/degen_b2b", &sim->p.degen_b2b);
	my_read(_int, "/params/degen_bb2", &sim->p.degen_bb2);
	my_read(_int, "/params/degen_b2b2", &sim->p.degen_b2b2);

	my_read(_int, "/params/period_uneqlt", &sim->p.period_uneqlt);
	my_read(_int, "/params/meas_bond_corr", &sim->p.meas_bond_corr);
	my_read(_int, "/params/meas_thermal", &sim->p.meas_thermal);
	my_read(_int, "/params/meas_2bond_corr", &sim->p.meas_2bond_corr);
	my_read(_int, "/params/meas_local_JQ", &sim->p.meas_local_JQ);
	my_read(_int, "/params/meas_gen_suscept", &sim->p.meas_gen_suscept);
	my_read(_int, "/params/meas_energy_corr", &sim->p.meas_energy_corr);
	my_read(_int, "/params/meas_nematic_corr", &sim->p.meas_nematic_corr);
	my_read(_int, "/params/meas_pair_bb_only", &sim->p.meas_pair_bb_only);
	my_read(_int, "/params/meas_chiral", &sim->p.meas_chiral);
	my_read(_int, "/params/checkpoint_every", &sim->p.checkpoint_every);

	const int N = sim->p.N, L = sim->p.L;
	const int Nx = sim->p.Nx, Ny = sim->p.Ny;
	const int Norb = N / (Nx * Ny);
	const int num_i = sim->p.num_i, num_ij = sim->p.num_ij;
	const int num_plaq_accum = sim->p.num_plaq_accum;
	const int num_b_accum    = sim->p.num_b_accum;
	const int num_b2_accum   = sim->p.num_b2_accum;
	const int num_plaq       = sim->p.num_plaq;
	const int num_b = sim->p.num_b, num_bs = sim->p.num_bs, num_bb = sim->p.num_bb;
	const int num_b2 = sim->p.num_b2, num_b2b2 = sim->p.num_b2b2, 
	          num_b2b = sim->p.num_b2b, num_bb2 = sim->p.num_bb2;

	//allocate memory here
	sim->p.map_i         = my_calloc(N        * sizeof(int));
	sim->p.map_ij        = my_calloc(N*N      * sizeof(int));
	sim->p.bonds         = my_calloc(num_b*2  * sizeof(int));
	sim->p.bond2s        = my_calloc(num_b2*2  * sizeof(int));
	sim->p.plaqs         = my_calloc(num_plaq*3 * sizeof(int));
	sim->p.map_plaq      = my_calloc(num_plaq * sizeof(int));
	sim->p.map_b         = my_calloc(num_b  * sizeof(int));
	sim->p.map_b2         = my_calloc(num_b2  * sizeof(int));
	sim->p.map_bs        = my_calloc(num_b*N  * sizeof(int));
	sim->p.map_bb        = my_calloc(num_b*num_b * sizeof(int));
	sim->p.map_b2b2      = my_calloc(num_b2*num_b2 * sizeof(int));
	sim->p.map_b2b      = my_calloc(num_b2*num_b * sizeof(int));
	sim->p.map_bb2      = my_calloc(num_b*num_b2 * sizeof(int));
	sim->p.peierlsu      = my_calloc(N*N      * sizeof(num));
	sim->p.peierlsd      = my_calloc(N*N      * sizeof(num));
	sim->p.pp_u      = my_calloc(num_b2 * sizeof(num));
	sim->p.pp_d      = my_calloc(num_b2 * sizeof(num));
	sim->p.ppr_u      = my_calloc(num_b2 * sizeof(num));
	sim->p.ppr_d      = my_calloc(num_b2 * sizeof(num));
	sim->p.exp_Ku        = my_calloc(N*N      * sizeof(num));
	sim->p.exp_Kd        = my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_Ku    = my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_Kd    = my_calloc(N*N      * sizeof(num));
	sim->p.exp_halfKu    = my_calloc(N*N      * sizeof(num));
	sim->p.exp_halfKd    = my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_halfKu= my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_halfKd= my_calloc(N*N      * sizeof(num));
	sim->p.exp_lambda    = my_calloc(N*2      * sizeof(double));
	sim->p.del           = my_calloc(N*2      * sizeof(double));
	sim->s.hs            = my_calloc(N*L      * sizeof(int));
	sim->m_eq.density    = my_calloc(num_i    * sizeof(num));
	sim->m_eq.density_u  = my_calloc(num_i    * sizeof(num));
	sim->m_eq.density_d  = my_calloc(num_i    * sizeof(num));
	sim->m_eq.double_occ = my_calloc(num_i    * sizeof(num));
	sim->m_eq.g00        = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.g00_u      = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.g00_d      = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.nn         = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.xx         = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.zz         = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.pair_sw    = my_calloc(num_ij   * sizeof(num));

	if (sim->p.meas_gen_suscept){
		sim->m_eq.uuuu        = my_calloc(num_ij*num_ij * sizeof(num));
		sim->m_eq.dddd        = my_calloc(num_ij*num_ij * sizeof(num));
		sim->m_eq.dduu        = my_calloc(num_ij*num_ij * sizeof(num));
		sim->m_eq.uudd        = my_calloc(num_ij*num_ij * sizeof(num));
	}
	if (sim->p.meas_chiral) {
		sim->m_eq.chi = my_calloc(num_plaq_accum * sizeof(num));
	}
	if (sim->p.meas_energy_corr) {
		sim->m_eq.kk = my_calloc(num_bb * sizeof(num));
		sim->m_eq.kv = my_calloc(num_bs * sizeof(num));
		sim->m_eq.kn = my_calloc(num_bs * sizeof(num));
		sim->m_eq.vv = my_calloc(num_ij * sizeof(num));
		sim->m_eq.vn = my_calloc(num_ij * sizeof(num));
	}
	if (sim->p.meas_local_JQ) {
		sim->m_eq.j   = my_calloc(num_b_accum  * sizeof(num));
		sim->m_eq.jn  = my_calloc(num_b_accum  * sizeof(num));
		sim->m_eq.j2  = my_calloc(num_b2_accum * sizeof(num));
	}
	if (sim->p.period_uneqlt > 0) {
		sim->m_ue.gt0     = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.gt0_u   = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.gt0_d   = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.nn      = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.xx      = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.zz      = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.pair_sw = my_calloc(num_ij*L * sizeof(num));
		if (sim->p.meas_gen_suscept){
			sim->m_ue.uuuu        = my_calloc(num_ij*num_ij*L * sizeof(num));
			sim->m_ue.dddd        = my_calloc(num_ij*num_ij*L * sizeof(num));
			sim->m_ue.dduu        = my_calloc(num_ij*num_ij*L * sizeof(num));
			sim->m_ue.uudd        = my_calloc(num_ij*num_ij*L * sizeof(num));
		}
		if (sim->p.meas_pair_bb_only) {
			sim->m_ue.pair_bb = my_calloc(num_bb*L * sizeof(num));
		}
		if (sim->p.meas_bond_corr) {
			sim->m_ue.pair_bb = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.jj      = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.jsjs    = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.kk      = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.ksks    = my_calloc(num_bb*L * sizeof(num));
		}
		if (sim->p.meas_thermal) {
			sim->m_ue.j2jn = my_calloc(num_b2b*L * sizeof(num));
			sim->m_ue.jnj2 = my_calloc(num_bb2*L * sizeof(num));
			sim->m_ue.jnjn= my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.jnj= my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.jjn= my_calloc(num_bb*L * sizeof(num));
		}
		if (sim->p.meas_2bond_corr) {
			sim->m_ue.pair_b2b2= my_calloc(num_b2b2*L * sizeof(num));
			sim->m_ue.j2j2    = my_calloc(num_b2b2*L * sizeof(num));
			sim->m_ue.j2j     = my_calloc(num_b2b*L * sizeof(num));
			sim->m_ue.jj2     = my_calloc(num_bb2*L * sizeof(num));
			sim->m_ue.js2js2  = my_calloc(num_b2b2*L * sizeof(num));
			sim->m_ue.k2k2    = my_calloc(num_b2b2*L * sizeof(num));
			sim->m_ue.ks2ks2  = my_calloc(num_b2b2*L * sizeof(num));
		}
		if (sim->p.meas_energy_corr) {
			sim->m_ue.kv      = my_calloc(num_bs*L * sizeof(num));
			sim->m_ue.kn      = my_calloc(num_bs*L * sizeof(num));
			sim->m_ue.vv      = my_calloc(num_ij*L * sizeof(num));
			sim->m_ue.vn      = my_calloc(num_ij*L * sizeof(num));
		}
		if (sim->p.meas_nematic_corr) {
			sim->m_ue.nem_nnnn = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.nem_ssss = my_calloc(num_bb*L * sizeof(num));
		}
	}
	// make sure anything appended here is free'd in sim_data_free()

	my_read(_int,    "/params/map_i",          sim->p.map_i);
	my_read(_int,    "/params/map_ij",         sim->p.map_ij);
	my_read(_int,    "/params/map_plaq",       sim->p.map_plaq);
	my_read(_int,    "/params/plaqs",          sim->p.plaqs);
	my_read(_int,    "/params/bonds",          sim->p.bonds);
	my_read(_int,    "/params/bond2s",          sim->p.bond2s);
	my_read(_int,    "/params/map_bs",         sim->p.map_bs);
	my_read(_int,    "/params/map_bb",         sim->p.map_bb);
	my_read(_int,    "/params/map_b",         sim->p.map_b);
	my_read(_int,    "/params/map_b2",         sim->p.map_b2);
	my_read(_int,    "/params/map_bb2",         sim->p.map_bb2);
	my_read(_int,    "/params/map_b2b",         sim->p.map_b2b);
	my_read(_int,    "/params/map_b2b2",         sim->p.map_b2b2);
	my_read( , "/params/peierlsu", num_h5t,    sim->p.peierlsu);
	my_read( , "/params/peierlsd", num_h5t,    sim->p.peierlsd);
	my_read( , "/params/pp_u", num_h5t,    sim->p.pp_u);
	my_read( , "/params/pp_d", num_h5t,    sim->p.pp_d);
	my_read( , "/params/ppr_u", num_h5t,    sim->p.ppr_u);
	my_read( , "/params/ppr_d", num_h5t,    sim->p.ppr_d);
	//	my_read(_double, "/params/dt",            &sim->p.dt);
	my_read(_int,    "/params/n_matmul",      &sim->p.n_matmul);
	my_read(_int,    "/params/n_delay",       &sim->p.n_delay);
	my_read(_int,    "/params/n_sweep_warm",  &sim->p.n_sweep_warm);
	my_read(_int,    "/params/n_sweep_meas",  &sim->p.n_sweep_meas);
	my_read(_int,    "/params/period_eqlt",   &sim->p.period_eqlt);
	my_read( , "/params/exp_Ku",     num_h5t,   sim->p.exp_Ku);
	my_read( , "/params/exp_Kd",     num_h5t,   sim->p.exp_Kd);
	my_read( , "/params/inv_exp_Ku", num_h5t,   sim->p.inv_exp_Ku);
	my_read( , "/params/inv_exp_Kd", num_h5t,   sim->p.inv_exp_Kd);
	my_read( , "/params/exp_halfKu",     num_h5t,   sim->p.exp_halfKu);
	my_read( , "/params/exp_halfKd",     num_h5t,   sim->p.exp_halfKd);
	my_read( , "/params/inv_exp_halfKu", num_h5t,   sim->p.inv_exp_halfKu);
	my_read( , "/params/inv_exp_halfKd", num_h5t,   sim->p.inv_exp_halfKd);
	my_read(_double, "/params/exp_lambda",     sim->p.exp_lambda);
	my_read(_double, "/params/del",            sim->p.del);
	my_read(_int,    "/params/F",             &sim->p.F);
	my_read(_int,    "/params/n_sweep",       &sim->p.n_sweep);
	my_read( ,       "/state/rng", H5T_NATIVE_UINT64, sim->s.rng);
	my_read(_int,    "/state/sweep",          &sim->s.sweep);
	my_read(_int,    "/state/hs",              sim->s.hs);
	my_read(_int,    "/meas_eqlt/n_sample",   &sim->m_eq.n_sample);
	my_read( , "/meas_eqlt/sign",        num_h5t, &sim->m_eq.sign);
	my_read( , "/meas_eqlt/density",     num_h5t, sim->m_eq.density);
	my_read( , "/meas_eqlt/density_u",   num_h5t, sim->m_eq.density_u);
	my_read( , "/meas_eqlt/density_d",   num_h5t, sim->m_eq.density_d);
	my_read( , "/meas_eqlt/double_occ",  num_h5t, sim->m_eq.double_occ);
	my_read( , "/meas_eqlt/g00",         num_h5t, sim->m_eq.g00);
	my_read( , "/meas_eqlt/g00_u",       num_h5t, sim->m_eq.g00_u);
	my_read( , "/meas_eqlt/g00_d",       num_h5t, sim->m_eq.g00_d);
	my_read( , "/meas_eqlt/nn",          num_h5t, sim->m_eq.nn);
	my_read( , "/meas_eqlt/xx",          num_h5t, sim->m_eq.xx);
	my_read( , "/meas_eqlt/zz",          num_h5t, sim->m_eq.zz);
	my_read( , "/meas_eqlt/pair_sw",     num_h5t, sim->m_eq.pair_sw);
	if (sim->p.meas_gen_suscept) {
		my_read( , "/meas_eqlt/uuuu",    num_h5t, sim->m_eq.uuuu);
		my_read( , "/meas_eqlt/dddd",    num_h5t, sim->m_eq.dddd);
		my_read( , "/meas_eqlt/uudd",    num_h5t, sim->m_eq.uudd);
		my_read( , "/meas_eqlt/dduu",    num_h5t, sim->m_eq.dduu);
	}
	if (sim->p.meas_chiral) {
		my_read( , "/meas_eqlt/chi", num_h5t, sim->m_eq.chi);
	}
	if (sim->p.meas_energy_corr) {
		my_read( , "/meas_eqlt/kk", num_h5t, sim->m_eq.kk);
		my_read( , "/meas_eqlt/kv", num_h5t, sim->m_eq.kv);
		my_read( , "/meas_eqlt/kn", num_h5t, sim->m_eq.kn);
		my_read( , "/meas_eqlt/vv", num_h5t, sim->m_eq.vv);
		my_read( , "/meas_eqlt/vn", num_h5t, sim->m_eq.vn);
	}
	if (sim->p.meas_local_JQ){
		my_read( , "/meas_eqlt/j2", num_h5t, sim->m_eq.j2);
		my_read( , "/meas_eqlt/j",  num_h5t, sim->m_eq.j );
		my_read( , "/meas_eqlt/jn", num_h5t, sim->m_eq.jn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_read(_int,    "/meas_uneqlt/n_sample", &sim->m_ue.n_sample);
		my_read( , "/meas_uneqlt/sign",      num_h5t, &sim->m_ue.sign);
		my_read( , "/meas_uneqlt/gt0",       num_h5t, sim->m_ue.gt0);
		my_read( , "/meas_uneqlt/gt0_u",     num_h5t, sim->m_ue.gt0_u);
		my_read( , "/meas_uneqlt/gt0_d",     num_h5t, sim->m_ue.gt0_d);
		my_read( , "/meas_uneqlt/nn",        num_h5t, sim->m_ue.nn);
		my_read( , "/meas_uneqlt/xx",        num_h5t, sim->m_ue.xx);
		my_read( , "/meas_uneqlt/zz",        num_h5t, sim->m_ue.zz);
		my_read( , "/meas_uneqlt/pair_sw",   num_h5t, sim->m_ue.pair_sw);
		if (sim->p.meas_gen_suscept) {
			my_read( , "/meas_uneqlt/uuuu", num_h5t, sim->m_ue.uuuu);
			my_read( , "/meas_uneqlt/dddd", num_h5t, sim->m_ue.dddd);
			my_read( , "/meas_uneqlt/uudd", num_h5t, sim->m_ue.uudd);
			my_read( , "/meas_uneqlt/dduu", num_h5t, sim->m_ue.dduu);
		}
		if (sim->p.meas_pair_bb_only) {
			my_read( , "/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
		}
		if (sim->p.meas_bond_corr) {
			my_read( , "/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
			my_read( , "/meas_uneqlt/jj",      num_h5t, sim->m_ue.jj);
			my_read( , "/meas_uneqlt/jsjs",    num_h5t, sim->m_ue.jsjs);
			my_read( , "/meas_uneqlt/kk",      num_h5t, sim->m_ue.kk);
			my_read( , "/meas_uneqlt/ksks",    num_h5t, sim->m_ue.ksks);
		}
		if (sim->p.meas_thermal) {
			my_read( , "/meas_uneqlt/j2jn",     num_h5t, sim->m_ue.j2jn);
			my_read( , "/meas_uneqlt/jnj2",     num_h5t, sim->m_ue.jnj2);
			my_read( , "/meas_uneqlt/jnjn",     num_h5t, sim->m_ue.jnjn);
			my_read( , "/meas_uneqlt/jnj",    num_h5t, sim->m_ue.jnj);
			my_read( , "/meas_uneqlt/jjn",    num_h5t, sim->m_ue.jjn);
		}
		if (sim->p.meas_2bond_corr) {
			my_read( , "/meas_uneqlt/pair_b2b2", num_h5t, sim->m_ue.pair_b2b2);
			my_read( , "/meas_uneqlt/j2j2",      num_h5t, sim->m_ue.j2j2);
			my_read( , "/meas_uneqlt/j2j",       num_h5t, sim->m_ue.j2j);
			my_read( , "/meas_uneqlt/jj2",       num_h5t, sim->m_ue.jj2);
			my_read( , "/meas_uneqlt/js2js2",    num_h5t, sim->m_ue.js2js2);
			my_read( , "/meas_uneqlt/k2k2",      num_h5t, sim->m_ue.k2k2);
			my_read( , "/meas_uneqlt/ks2ks2",    num_h5t, sim->m_ue.ks2ks2);
		}
		if (sim->p.meas_energy_corr) {
			my_read( , "/meas_uneqlt/kv", num_h5t, sim->m_ue.kv);
			my_read( , "/meas_uneqlt/kn", num_h5t, sim->m_ue.kn);
			my_read( , "/meas_uneqlt/vv", num_h5t, sim->m_ue.vv);
			my_read( , "/meas_uneqlt/vn", num_h5t, sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_read( , "/meas_uneqlt/nem_nnnn", num_h5t, sim->m_ue.nem_nnnn);
			my_read( , "/meas_uneqlt/nem_ssss", num_h5t, sim->m_ue.nem_ssss);
		}
	}

	status = H5Fclose(file_id);
	return_if(status < 0, CLOSE_FAIL, "H5Fclose() failed for %s: %d\n", sim->file, status);
	return 0;
}

/**
 * Save simulation state and measurements to disk
 * @param  sim [description]
 * @return -1 if H5Fopen() failed
 *         -2 if file is corrupted
 *         -3 if my_read or my_write failed
 *         -4 if H5Fclose() failed
 */
int sim_data_save(const struct sim_data *sim) {
	const hid_t file_id = H5Fopen(sim->file, H5F_ACC_RDWR, H5P_DEFAULT);
	return_if(file_id < 0, OPEN_FAIL, "H5Fopen() failed for %s: %ld\n", 
		sim->file, file_id);

	int status; // see my_write macro
	hid_t dset_id; //see my_write macro

	int partial_write;
	// Check if we are saving to a valid file
	my_read(_int,"/state/partial_write", &partial_write);
	if (partial_write) {
		return_if(partial_write, CORRUPT_FAIL, \
			"Last checkpoint in %s contained partial write\n", sim->file);
	}
	
	//mark this file as having a write in progress
	partial_write = 1;
	my_write("/state/partial_write", H5T_NATIVE_INT, &partial_write);

	//write state + measurement data
	my_write("/state/rng",            H5T_NATIVE_UINT64,  sim->s.rng);
	my_write("/state/sweep",          H5T_NATIVE_INT,    &sim->s.sweep);
	my_write("/state/hs",             H5T_NATIVE_INT,     sim->s.hs);
	my_write("/meas_eqlt/n_sample",   H5T_NATIVE_INT,    &sim->m_eq.n_sample);
	my_write("/meas_eqlt/sign",       num_h5t, &sim->m_eq.sign);
	my_write("/meas_eqlt/density",    num_h5t,  sim->m_eq.density);
	my_write("/meas_eqlt/density_u",  num_h5t,  sim->m_eq.density_u);
	my_write("/meas_eqlt/density_d",  num_h5t,  sim->m_eq.density_d);
	my_write("/meas_eqlt/double_occ", num_h5t,  sim->m_eq.double_occ);
	my_write("/meas_eqlt/g00",        num_h5t,  sim->m_eq.g00);
	my_write("/meas_eqlt/g00_u",      num_h5t,  sim->m_eq.g00_u);
	my_write("/meas_eqlt/g00_d",      num_h5t,  sim->m_eq.g00_d);
	my_write("/meas_eqlt/nn",         num_h5t,  sim->m_eq.nn);
	my_write("/meas_eqlt/xx",         num_h5t,  sim->m_eq.xx);
	my_write("/meas_eqlt/zz",         num_h5t,  sim->m_eq.zz);
	my_write("/meas_eqlt/pair_sw",    num_h5t,  sim->m_eq.pair_sw);

	if (sim->p.meas_gen_suscept){
		my_write("/meas_eqlt/uuuu",   num_h5t,  sim->m_eq.uuuu);
		my_write("/meas_eqlt/dddd",   num_h5t,  sim->m_eq.dddd);
		my_write("/meas_eqlt/uudd",   num_h5t,  sim->m_eq.uudd);
		my_write("/meas_eqlt/dduu",   num_h5t,  sim->m_eq.dduu);
	}
	if (sim->p.meas_chiral) {
		my_write("/meas_eqlt/chi", num_h5t, sim->m_eq.chi);
	}
	if (sim->p.meas_energy_corr) {
		my_write("/meas_eqlt/kk", num_h5t, sim->m_eq.kk);
		my_write("/meas_eqlt/kv", num_h5t, sim->m_eq.kv);
		my_write("/meas_eqlt/kn", num_h5t, sim->m_eq.kn);
		my_write("/meas_eqlt/vv", num_h5t, sim->m_eq.vv);
		my_write("/meas_eqlt/vn", num_h5t, sim->m_eq.vn);
	}
	if (sim->p.meas_local_JQ){
		my_write("/meas_eqlt/j2", num_h5t, sim->m_eq.j2);
		my_write("/meas_eqlt/j",  num_h5t, sim->m_eq.j );
		my_write("/meas_eqlt/jn", num_h5t, sim->m_eq.jn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_write("/meas_uneqlt/n_sample", H5T_NATIVE_INT,    &sim->m_ue.n_sample);
		my_write("/meas_uneqlt/sign",     num_h5t, &sim->m_ue.sign);
		my_write("/meas_uneqlt/gt0",      num_h5t,  sim->m_ue.gt0);
		my_write("/meas_uneqlt/gt0_u",    num_h5t,  sim->m_ue.gt0_u);
		my_write("/meas_uneqlt/gt0_d",    num_h5t,  sim->m_ue.gt0_d);
		my_write("/meas_uneqlt/nn",       num_h5t,  sim->m_ue.nn);
		my_write("/meas_uneqlt/xx",       num_h5t,  sim->m_ue.xx);
		my_write("/meas_uneqlt/zz",       num_h5t,  sim->m_ue.zz);
		my_write("/meas_uneqlt/pair_sw",  num_h5t,  sim->m_ue.pair_sw);
		if (sim->p.meas_gen_suscept) {
			my_write("/meas_uneqlt/uuuu", num_h5t, sim->m_ue.uuuu);
			my_write("/meas_uneqlt/dddd", num_h5t, sim->m_ue.dddd);
			my_write("/meas_uneqlt/uudd", num_h5t, sim->m_ue.uudd);
			my_write("/meas_uneqlt/dduu", num_h5t, sim->m_ue.dduu);
		}
		if (sim->p.meas_pair_bb_only) {
			my_write("/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
		}
		if (sim->p.meas_bond_corr) {
			my_write("/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
			my_write("/meas_uneqlt/jj",      num_h5t, sim->m_ue.jj);
			my_write("/meas_uneqlt/jsjs",    num_h5t, sim->m_ue.jsjs);
			my_write("/meas_uneqlt/kk",      num_h5t, sim->m_ue.kk);
			my_write("/meas_uneqlt/ksks",    num_h5t, sim->m_ue.ksks);
		}
		if (sim->p.meas_thermal) {
			my_write("/meas_uneqlt/j2jn",     num_h5t, sim->m_ue.j2jn);
			my_write("/meas_uneqlt/jnj2",     num_h5t, sim->m_ue.jnj2);
			my_write("/meas_uneqlt/jnjn",    num_h5t, sim->m_ue.jnjn);
			my_write("/meas_uneqlt/jnj",    num_h5t, sim->m_ue.jnj);
			my_write("/meas_uneqlt/jjn",    num_h5t, sim->m_ue.jjn);
		}
		if (sim->p.meas_2bond_corr) {
			my_write("/meas_uneqlt/pair_b2b2", num_h5t, sim->m_ue.pair_b2b2);
			my_write("/meas_uneqlt/j2j2",      num_h5t, sim->m_ue.j2j2);
			my_write("/meas_uneqlt/j2j",       num_h5t, sim->m_ue.j2j);
			my_write("/meas_uneqlt/jj2",       num_h5t, sim->m_ue.jj2);
			my_write("/meas_uneqlt/js2js2",    num_h5t, sim->m_ue.js2js2);
			my_write("/meas_uneqlt/k2k2",      num_h5t, sim->m_ue.k2k2);
			my_write("/meas_uneqlt/ks2ks2",    num_h5t, sim->m_ue.ks2ks2);
		}
		if (sim->p.meas_energy_corr) {
			my_write("/meas_uneqlt/kv", num_h5t, sim->m_ue.kv);
			my_write("/meas_uneqlt/kn", num_h5t, sim->m_ue.kn);
			my_write("/meas_uneqlt/vv", num_h5t, sim->m_ue.vv);
			my_write("/meas_uneqlt/vn", num_h5t, sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_write("/meas_uneqlt/nem_nnnn", num_h5t, sim->m_ue.nem_nnnn);
			my_write("/meas_uneqlt/nem_ssss", num_h5t, sim->m_ue.nem_ssss);
		}
	}

	//mark this file as having no partial writes, i.e. a clean checkpoint.
	partial_write = 0;
	my_write("/state/partial_write", H5T_NATIVE_INT, &partial_write);

	status = H5Fclose(file_id);
	return_if(status < 0, CLOSE_FAIL, "H5Fclose() failed for %s: %d\n", 
		sim->file, status);
	return 0;
}

void sim_data_free(const struct sim_data *sim) {
	if (sim->p.period_uneqlt > 0) {
		if (sim->p.meas_nematic_corr) {
			my_free(sim->m_ue.nem_ssss);
			my_free(sim->m_ue.nem_nnnn);
		}
		if (sim->p.meas_gen_suscept) {
			my_free(sim->m_ue.uuuu);
			my_free(sim->m_ue.dddd);
			my_free(sim->m_ue.uudd);
			my_free(sim->m_ue.dduu);
		}
		if (sim->p.meas_energy_corr) {
			my_free(sim->m_ue.vn);
			my_free(sim->m_ue.vv);
			my_free(sim->m_ue.kn);
			my_free(sim->m_ue.kv);
		}
		if (sim->p.meas_thermal) {
			my_free(sim->m_ue.jnjn);
			my_free(sim->m_ue.jnj);
			my_free(sim->m_ue.jjn);
			my_free(sim->m_ue.jnj2);
			my_free(sim->m_ue.j2jn);
		}
		if (sim->p.meas_pair_bb_only) {
			my_free(sim->m_ue.pair_bb);
		}
		if (sim->p.meas_bond_corr) {
			my_free(sim->m_ue.ksks);
			my_free(sim->m_ue.kk);
			my_free(sim->m_ue.jsjs);
			my_free(sim->m_ue.jj);
			my_free(sim->m_ue.pair_bb);
		}
		if (sim->p.meas_2bond_corr) {
			my_free(sim->m_ue.ks2ks2);
			my_free(sim->m_ue.k2k2);
			my_free(sim->m_ue.js2js2);
			my_free(sim->m_ue.j2j2);
			my_free(sim->m_ue.j2j);
			my_free(sim->m_ue.jj2);
			my_free(sim->m_ue.pair_b2b2);
		}
		my_free(sim->m_ue.pair_sw);
		my_free(sim->m_ue.zz);
		my_free(sim->m_ue.xx);
		my_free(sim->m_ue.nn);
		my_free(sim->m_ue.gt0);
		my_free(sim->m_ue.gt0_u);
		my_free(sim->m_ue.gt0_d);
	}
	if (sim->p.meas_energy_corr) {
		my_free(sim->m_eq.vn);
		my_free(sim->m_eq.vv);
		my_free(sim->m_eq.kn);
		my_free(sim->m_eq.kv);
		my_free(sim->m_eq.kk);
	}
	if (sim->p.meas_local_JQ){
		my_free(sim->m_eq.j2);
		my_free(sim->m_eq.j);
		my_free(sim->m_eq.jn);
	}
	if (sim->p.meas_chiral) {
		my_free(sim->m_eq.chi);
	}
	if (sim->p.meas_gen_suscept){
		my_free(sim->m_eq.uuuu);
		my_free(sim->m_eq.dddd);
		my_free(sim->m_eq.uudd);
		my_free(sim->m_eq.dduu);
	}
	my_free(sim->m_eq.pair_sw);
	my_free(sim->m_eq.zz);
	my_free(sim->m_eq.xx);
	my_free(sim->m_eq.nn);
	my_free(sim->m_eq.g00);
	my_free(sim->m_eq.g00_u);
	my_free(sim->m_eq.g00_d);
	my_free(sim->m_eq.double_occ);
	my_free(sim->m_eq.density);
	my_free(sim->m_eq.density_u);
	my_free(sim->m_eq.density_d);
	my_free(sim->s.hs);
	my_free(sim->p.del);
	my_free(sim->p.exp_lambda);
	my_free(sim->p.inv_exp_halfKd);
	my_free(sim->p.inv_exp_halfKu);
	my_free(sim->p.exp_halfKd);
	my_free(sim->p.exp_halfKu);
	my_free(sim->p.inv_exp_Kd);
	my_free(sim->p.inv_exp_Ku);
	my_free(sim->p.exp_Kd);
	my_free(sim->p.exp_Ku);
	my_free(sim->p.peierlsd);
	my_free(sim->p.peierlsu);
	my_free(sim->p.pp_u);
	my_free(sim->p.pp_d);
	my_free(sim->p.ppr_u);
	my_free(sim->p.ppr_d);
	my_free(sim->p.map_b2b2);
	my_free(sim->p.map_bb2);
	my_free(sim->p.map_b2b);
	my_free(sim->p.map_bb);
	my_free(sim->p.map_b);
	my_free(sim->p.map_b2);
	my_free(sim->p.map_bs);
	my_free(sim->p.bond2s);
	my_free(sim->p.bonds);
	my_free(sim->p.plaqs);
	my_free(sim->p.map_plaq);
	my_free(sim->p.map_ij);
	my_free(sim->p.map_i);
}
