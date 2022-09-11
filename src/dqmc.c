#include <tgmath.h>
#include <stdio.h>
#include <assert.h>
#include "greens.h"
#include "linalg.h"
#include "meas.h"
#include "prof.h"
#include "rand.h"
#include "sig.h"
#include "updates.h"

#include "dqmc.h"

// uncomment to check recalculated g against wrapped g
// #define CHECK_G_WRP

// uncomment to check recalculated g against g from using QR for every multiply
// #define CHECK_G_ACC

// uncomment to check 0,0 block of unequal-time G against recalculated g
// #define CHECK_G_UE

// who needs function calls :D
#define matmul(C, A, B) do { \
	xgemm("N", "N", N, N, N, 1.0, (A), N, (B), N, 0.0, (C), N); \
} while (0);

#define calcBu(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*hs[j + N*(l)]]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Ku[i + N*j] * el; \
	} \
} while (0);

#define calcBd(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*!hs[j + N*(l)]]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Kd[i + N*j] * el; \
	} \
} while (0);

#define calciBu(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*!hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Ku[i + N*j]; \
	} \
} while (0);

#define calciBd(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Kd[i + N*j]; \
	} \
} while (0);

#define matdiff(m, n, A, ldA, B, ldB) do { \
	double max = 0.0, avg = 0.0; \
	for (int j = 0; j < (n); j++) \
	for (int i = 0; i < (m); i++) { \
		const double diff = fabs((A)[i + (ldA)*j] - (B)[i + (ldB)*j]); \
		if (diff > max) max = diff; \
		avg += diff; \
	} \
	avg /= N*N; \
	printf(#A " - " #B ":\tmax %.3e\tavg %.3e\n", max, avg); \
} while (0);

#define INTERRUPTED 5
#define CHECKPOINT_FAIL 6

/**
 * Core dqmc logic
 * @param  sim [description]
 * @return 0 on success
 *         1 on interrupt
 *         2 on checkpoint failure
 *         ?? TODO any other possibility?
 */
static int dqmc(struct sim_data *sim) {
	int dqmc_return_code = 0;

	const int N = sim->p.N;
	const int L = sim->p.L;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	const int F = sim->p.F;
	const num *const restrict exp_Ku = sim->p.exp_Ku;
	const num *const restrict exp_Kd = sim->p.exp_Kd;
	const num *const restrict inv_exp_Ku = sim->p.inv_exp_Ku;
	const num *const restrict inv_exp_Kd = sim->p.inv_exp_Kd;
	const num *const restrict exp_halfKu = sim->p.exp_halfKu;
	const num *const restrict exp_halfKd = sim->p.exp_halfKd;
	const num *const restrict inv_exp_halfKu = sim->p.inv_exp_halfKu;
	const num *const restrict inv_exp_halfKd = sim->p.inv_exp_halfKd;
	const double *const restrict exp_lambda = sim->p.exp_lambda;
	const double *const restrict del = sim->p.del;
	uint64_t *const restrict rng = sim->s.rng;
	int *const restrict hs = sim->s.hs;

	num *const Bu = my_calloc(N*N*L * sizeof(num));
	num *const Bd = my_calloc(N*N*L * sizeof(num));
	num *const iBu = my_calloc(N*N*L * sizeof(num));
	num *const iBd = my_calloc(N*N*L * sizeof(num));
	num *const Cu = my_calloc(N*N*F * sizeof(num));
	num *const Cd = my_calloc(N*N*F * sizeof(num));
	num *const restrict gu = my_calloc(N*N * sizeof(num));
	num *const restrict gd = my_calloc(N*N * sizeof(num));
	#ifdef CHECK_G_WRP
	num *const restrict guwrp = my_calloc(N*N * sizeof(num));
	num *const restrict gdwrp = my_calloc(N*N * sizeof(num));
	#endif
	#ifdef CHECK_G_ACC
	num *const restrict guacc = my_calloc(N*N * sizeof(num));
	num *const restrict gdacc = my_calloc(N*N * sizeof(num));
	#endif
	num phase;
	int *const site_order = my_calloc(N * sizeof(double));

	// work arrays for calc_eq_g and stuff. two sets for easy 2x parallelization
	num *const restrict tmpNN1u = my_calloc(N*N * sizeof(num));
	num *const restrict tmpNN2u = my_calloc(N*N * sizeof(num));
	num *const restrict tmpN1u = my_calloc(N * sizeof(num));
	num *const restrict tmpN2u = my_calloc(N * sizeof(num));
	num *const restrict tmpN3u = my_calloc(N * sizeof(num));
	int *const restrict pvtu = my_calloc(N * sizeof(int));

	num *const restrict tmpNN1d = my_calloc(N*N * sizeof(num));
	num *const restrict tmpNN2d = my_calloc(N*N * sizeof(num));
	num *const restrict tmpN1d = my_calloc(N * sizeof(num));
	num *const restrict tmpN2d = my_calloc(N * sizeof(num));
	num *const restrict tmpN3d = my_calloc(N * sizeof(num));
	int *const restrict pvtd = my_calloc(N * sizeof(int));

	// arrays for calc_ue_g
	num *restrict Gu0t = NULL;
	num *restrict Gutt = NULL;
	num *restrict Gut0 = NULL;
	// num *restrict ueGu = NULL;
	num *restrict Gredu = NULL;
	num *restrict tauu = NULL;
	num *restrict Qu = NULL;

	num *restrict Gd0t = NULL;
	num *restrict Gdtt = NULL;
	num *restrict Gdt0 = NULL;
	// num *restrict ueGd = NULL;
	num *restrict Gredd = NULL;
	num *restrict taud = NULL;
	num *restrict Qd = NULL;

	if (sim->p.period_uneqlt > 0) {
		const int E = 1 + (F - 1) / N_MUL;

		Gredu = my_calloc(N*E*N*E * sizeof(num));
		tauu = my_calloc(N*E * sizeof(num));
		Qu = my_calloc(4*N*N * sizeof(num));

		Gredd = my_calloc(N*E*N*E * sizeof(num));
		taud = my_calloc(N*E * sizeof(num));
		Qd = my_calloc(4*N*N * sizeof(num));

		Gu0t = my_calloc(N*N*L * sizeof(num));
		Gutt = my_calloc(N*N*L * sizeof(num));
		Gut0 = my_calloc(N*N*L * sizeof(num));
		Gd0t = my_calloc(N*N*L * sizeof(num));
		Gdtt = my_calloc(N*N*L * sizeof(num));
		Gdt0 = my_calloc(N*N*L * sizeof(num));
		// ueGu = my_calloc(N*N*L*L * sizeof(num));
		// ueGd = my_calloc(N*N*L*L * sizeof(num));
		// if (ueGu == NULL || ueGd == NULL) return -1;
	}

	// lapack work arrays
	int lwork = get_lwork_eq_g(N);
	if (sim->p.period_uneqlt > 0) {
		const int E = 1 + (F - 1) / N_MUL;
		const int lwork_ue = get_lwork_ue_g(N, E);
		if (lwork_ue > lwork) lwork = lwork_ue;
	}
	num *const restrict worku = my_calloc(lwork * sizeof(num));
	num *const restrict workd = my_calloc(lwork * sizeof(num));

	{
	num phaseu, phased;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBu(iBu + N*N*l, l);
	for (int l = 0; l < L; l++)
		calcBu(Bu + N*N*l, l);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bu, Cu + N*N*f, N, tmpNN1u);
	phaseu = calc_eq_g(0, N, F, N_MUL, Cu, gu, tmpNN1u, tmpNN2u,
	                  tmpN1u, tmpN2u, tmpN3u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBd(iBd + N*N*l, l);
	for (int l = 0; l < L; l++)
		calcBd(Bd + N*N*l, l);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bd, Cd + N*N*f, N, tmpNN1d);
	phased = calc_eq_g(0, N, F, N_MUL, Cd, gd, tmpNN1d, tmpNN2d,
	                  tmpN1d, tmpN2d, tmpN3d, pvtd, workd, lwork);
	}
	}
	phase = phaseu*phased;
	}

	for (; sim->s.sweep < sim->p.n_sweep; sim->s.sweep++) {
		// check for signal at every full H-S sweep
		const int stop_flag = sig_check_state(sim->s.sweep, 
			sim->p.n_sweep_warm, sim->p.n_sweep);
		// Received timeout or interrupt
		if (stop_flag) { 
			dqmc_return_code = INTERRUPTED;
			break;
		}

		// No timeout or interrupt received this sweep
		if (sim->p.checkpoint_every > 0 && 
			sim->s.sweep % sim->p.checkpoint_every == 0) {
			// printf("sweep = %d, checkpoint_every = %d\n", 
			// 	sim->s.sweep,sim->p.checkpoint_every);
			const int save_status = sim_data_save(sim);
			if (save_status < 0) {
				dqmc_return_code = CHECKPOINT_FAIL;
				break;
			}
		}

		// no timeout, interrupt, and checkpoint OK
		for (int l = 0; l < L; l++) {
			profile_begin(updates);
			shuffle(rng, N, site_order);
			update_delayed(N, n_delay, del, site_order,
			               rng, hs + N*l, gu, gd, &phase,
			               tmpNN1u, tmpNN2u, tmpN1u,
			               tmpNN1d, tmpNN2d, tmpN1d);
			profile_end(updates);

			const int f = l / n_matmul;
			const int recalc = ((l + 1) % n_matmul == 0);
			num phaseu, phased;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			num *const restrict Bul = Bu + N*N*l;
			num *const restrict iBul = iBu + N*N*l;
			num *const restrict Cuf = Cu + N*N*f;
			profile_begin(calcb);
			calcBu(Bul, l);
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBu(iBul, l);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bu, Cuf, N, tmpNN1u);
				profile_end(multb);
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBu(iBu + N*N*l, l);
				matmul(tmpNN1u, gu, iBu + N*N*l);
				matmul(guwrp, Bu + N*N*l, tmpNN1u);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, L, 1, Bu, guacc,
				          tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				          tmpN3u, pvtu, worku, lwork);
				#endif
				phaseu = calc_eq_g((f + 1) % F, N, F, N_MUL, Cu, gu,
				                  tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				                  tmpN3u, pvtu, worku, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				matmul(tmpNN1u, gu, iBul);
				matmul(gu, Bul, tmpNN1u);
				profile_end(wrap);
			}
			}
			#pragma omp section
			{
			num *const restrict Bdl = Bd + N*N*l;
			num *const restrict iBdl = iBd + N*N*l;
			num *const restrict Cdf = Cd + N*N*f;
			profile_begin(calcb);
			calcBd(Bdl, l);
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBd(iBdl, l);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bd, Cdf, N, tmpNN1d);
				profile_end(multb);
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBd(iBd + N*N*l, l);
				matmul(tmpNN1d, gd, iBd + N*N*l);
				matmul(gdwrp, Bd + N*N*l, tmpNN1d);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, L, 1, Bd, gdacc,
				          tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				          tmpN3d, pvtd, workd, lwork);
				#endif
				phased = calc_eq_g((f + 1) % F, N, F, N_MUL, Cd, gd,
				                  tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				                  tmpN3d, pvtd, workd, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				matmul(tmpNN1d, gd, iBdl);
				matmul(gd, Bdl, tmpNN1d);
				profile_end(wrap);
			}
			}
			}

			#ifdef CHECK_G_WRP
			if (recalc) {
				matdiff(N, N, gu, N, guwrp, N);
				matdiff(N, N, gd, N, gdwrp, N);
			}
			#endif
			#ifdef CHECK_G_ACC
			if (recalc) {
				matdiff(N, N, gu, N, guacc, N);
				matdiff(N, N, gd, N, gdacc, N);
			}
			#endif
			#if defined(CHECK_G_WRP) && defined(CHECK_G_ACC)
			if (recalc) {
				matdiff(N, N, guwrp, N, guacc, N);
				matdiff(N, N, gdwrp, N, gdacc, N);
			}
			#endif

			if (recalc) phase = phaseu*phased;

			//eqlt measurement
			if ((sim->s.sweep >= sim->p.n_sweep_warm) &&
					(sim->p.period_eqlt > 0) &&
					(l + 1) % sim->p.period_eqlt == 0) {
				#pragma omp parallel sections
				{
				#pragma omp section
				{
				profile_begin(half_wrap);
				matmul(tmpNN1u, gu, exp_halfKu);
				matmul(tmpNN2u, inv_exp_halfKu, tmpNN1u);
				profile_end(half_wrap);
				}
				#pragma omp section
				{
				profile_begin(half_wrap);
				matmul(tmpNN1d, gd, exp_halfKd);
				matmul(tmpNN2d, inv_exp_halfKd, tmpNN1d);
				profile_end(half_wrap);
				}
				}

				profile_begin(meas_eq);
				measure_eqlt(&sim->p, phase, tmpNN2u, tmpNN2d, &sim->m_eq);
				profile_end(meas_eq);
			}
		}

		//uneqlt measurement
		if ((sim->s.sweep >= sim->p.n_sweep_warm) && 
			(sim->p.period_uneqlt > 0) &&
			(sim->s.sweep % sim->p.period_uneqlt == 0)) {
			// calculate all unequal time Green's functions 
			// G(0,t), G(t,t), G(t,0) for up/down electrons in parallel.
			// Each G object is size N x N x L
			#pragma omp parallel sections
			{
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bu, iBu, Cu, Gu0t, Gutt, Gut0,
			          Gredu, tauu, Qu, worku, lwork);
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bd, iBd, Cd, Gd0t, Gdtt, Gdt0,
			          Gredd, taud, Qd, workd, lwork);
			}

			#ifdef CHECK_G_UE
			matdiff(N, N, gu, N, Gutt, N);
			matdiff(N, N, gd, N, Gdtt, N);
			#endif
			#if defined(CHECK_G_UE) && defined(CHECK_G_ACC)
			matdiff(N, N, Gutt, N, guacc, N);
			matdiff(N, N, Gdtt, N, gdacc, N);
			#endif

			// TODO: why do we need to half wrap before measuring here?
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gu0t + N*N*l, exp_halfKu);
				matmul(Gu0t + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gutt + N*N*l, exp_halfKu);
				matmul(Gutt + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gut0 + N*N*l, exp_halfKu);
				matmul(Gut0 + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			profile_end(half_wrap);
			}
			#pragma omp section
			{
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gd0t + N*N*l, exp_halfKd);
				matmul(Gd0t + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdtt + N*N*l, exp_halfKd);
				matmul(Gdtt + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdt0 + N*N*l, exp_halfKd);
				matmul(Gdt0 + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			profile_end(half_wrap);
			}
			}

			profile_begin(meas_uneq);
			measure_uneqlt(&sim->p, phase,
			               Gu0t, Gutt, Gut0, Gd0t, Gdtt, Gdt0,
			               &sim->m_ue);
			profile_end(meas_uneq);
		}
	}

	// deallocate calculation memory
	my_free(workd);
	my_free(worku);
	if (sim->p.period_uneqlt > 0) {
		my_free(Qd);
		my_free(taud);
		my_free(Gredd);
		// my_free(ueGd);
		my_free(Gdt0);
		my_free(Gdtt);
		my_free(Gd0t);
		my_free(Qu);
		my_free(tauu);
		my_free(Gredu);
		// my_free(ueGu);
		my_free(Gut0);
		my_free(Gutt);
		my_free(Gu0t);
	}
	my_free(pvtd);
	my_free(tmpN3d);
	my_free(tmpN2d);
	my_free(tmpN1d);
	my_free(tmpNN2d);
	my_free(tmpNN1d);
	my_free(pvtu);
	my_free(tmpN3u);
	my_free(tmpN2u);
	my_free(tmpN1u);
	my_free(tmpNN2u);
	my_free(tmpNN1u);
	my_free(site_order);
	#ifdef CHECK_G_ACC
	my_free(gdacc);
	my_free(guacc);
	#endif
	#ifdef CHECK_G_WRP
	my_free(gdwrp);
	my_free(guwrp);
	#endif
	my_free(gd);
	my_free(gu);
	my_free(Cd);
	my_free(Cu);
	my_free(iBd);
	my_free(iBu);
	my_free(Bd);
	my_free(Bu);

	return dqmc_return_code;
}

#define WRAP_FAIL -1
#define WRAP_INCOMPLETE 1

/**
 * [dqmc_wrapper description]
 * @param  sim_file [description]
 * @param  log_file [description]
 * @param  max_time [description]
 * @param  dry      [description]
 * @param  bench    [description]
 * @return          [description]
 */
int dqmc_wrapper(const char *sim_file, const char *log_file,
		const int64_t max_time, const bool dry, const bool bench)
{
	const int64_t wall_start = time_wall();
	profile_clear();

	int wrap_return_code = 0;

	// if no log_file specified, then prints go to stdout
	FILE *log = (log_file != NULL) ? fopen(log_file, "a") : stdout;
	// if fail to open log file, quit. This error typically never occurs
	if (log == NULL) {
		fprintf(stderr, "fopen() failed to open: %s\n", log_file);
		return WRAP_FAIL;
	}

	// printf("C complex size: %zu\n",sizeof(num));
	// printf("dcomplex size: %zu\n",sizeof(dcomplex));
	// printf("libFLAME \"complex\" size: %zu\n",sizeof(complex));
	// fflush(stdout);

	// opened a log stream
	fprintf(log, "commit id %s\n", GIT_ID);
	fprintf(log, "compiled on %s %s\n", __DATE__, __TIME__);

	// exit path if dry run
	if (dry) {
		fprintf(log, "toggled dry run for mem and commit checks\n");
		fprintf(log, "hdf5 and executable consistent? %s\n", 
			consistency_check(sim_file) ? "No" : "Yes");
		fprintf(log, "minimum RAM requirement: %.2f MB\n", 
			(double) get_memory_req(sim_file)*1e-6);
		fprintf(log, "Not performing: read sim file, alloc mem, "
			"run core DQMC logic, or save data\n");
		fprintf(log, "Integer size on this system: %zu bytes\n",sizeof(int));
		if (log != stdout) fclose(log);
		else fflush(log);
		return wrap_return_code;
	}

	// Not a dry run
	// initialize signal handling
	// Since main_stack repeatedly calls dqmc_wrapper(), this sig_init()
	// function may be repeatedly called by the same process, 
	// repeated init is not necessary, but not harmful either(?)
	// The signal handler is individual to each process and there's no 
	// interaction between processes.
	sig_init(log, wall_start, max_time);

	// open and read simulation file
	struct sim_data *sim = my_calloc(sizeof(struct sim_data));
	sim->file=sim_file;
	fprintf(log, "opening %s\n", sim_file); 
	fflush(log);
	const int read_status = sim_data_read_alloc(sim);
	// Fail might be b/c sim_file doesnt exist:
	// errno = 2, error message = 'No such file or directory'
	// Or because not enough RAM?? TODO
	// errno = 12, error message = 'Cannot allocate memory'
	// Or mismatch between executable compilation flag -DUSE_CPLX
	//  and gen_1band_hub nflux=?? flag.
	if (read_status) {
		fprintf(log, "sim_data_read_alloc() failed: %d\n", read_status);
		wrap_return_code = WRAP_FAIL;
		goto cleanup;
	}

	// Read and alloc memory successful, check existing progress
	if (sim->p.checkpoint_every) {
		fprintf(log, "checkpointing every %d sweeps\n", sim->p.checkpoint_every);
	} else {
		fprintf(log, "warning: running with no checkpoints\n");
	}
	
	fprintf(log, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);
	assert(sim->s.sweep <= sim->p.n_sweep);
	if (sim->s.sweep == sim->p.n_sweep) {
		fprintf(log, "already finished so don't do anything\n");
		goto cleanup;
	}

	// Now we know sweep < n_sweep: run dqmc
	fprintf(log, "starting dqmc\n"); 
	fflush(log);
	// This might take a while. 2 OMP threads, blocking.
	const int dqmc_status = dqmc(sim); 
	
	if (dqmc_status == INTERRUPTED) {
		fprintf(log, "dqmc() interrupted\n");
		wrap_return_code = WRAP_INCOMPLETE;
		goto cleanup;
	} else if (dqmc_status == CHECKPOINT_FAIL) {
		fprintf(log, "dqmc() checkpoint failed\n");
		wrap_return_code = WRAP_FAIL;
		goto cleanup;
	}

	//dqmc() completed
	fprintf(log, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);

	// If benchmark mode, don't save data, directly cleanup
	if (bench) {
		fprintf(log, "benchmark mode enabled; not saving data to disk\n");
		goto cleanup;
	}

	// Save data
	fprintf(log, "saving data to disk\n");
	const int save_status = sim_data_save(sim);
	if (save_status < 0) {
		fprintf(log,    "sim_data_save() failed: %d\n", save_status);
		wrap_return_code = WRAP_FAIL;
		goto cleanup;
	}
	fprintf(log, "sim_data_save() succeeded\n");

	// wrap_return_code = (sim->s.sweep == sim->p.n_sweep) ? 0 : 1;

cleanup:
	//this section is executed at the end always if there's no goto
	sim_data_free(sim);
	my_free(sim);

	const int64_t wall_time = time_wall() - wall_start;
	fprintf(log, "wall time: %.3f\n", wall_time * SEC_PER_TICK);
	profile_print(log, wall_time);

	if (log != stdout)
		fclose(log);
	else
		fflush(log);

	return wrap_return_code;
}
