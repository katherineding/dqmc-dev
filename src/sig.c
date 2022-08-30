#include <signal.h>
#include <stdio.h>
#include "sig.h"

//status variables in signal handlers must be declared volatile
//static volatile sig_atomic_t progress_flag = 0;
static volatile sig_atomic_t stop_flag = 0;

//signal handler functions must have void sigfunc(int) signature
// static void progress(int signum) { progress_flag = signum; }
static void stop(int signum) { stop_flag = signum; }

// signal(int signum, sighandler_t action)
// Establishes action as the action for the signal signum.
// Returns the action that was previously in effect for the specified signum.
// If signal can’t honor the request, it returns SIG_ERR instead. 
// But we're advised not to use signal() and use sigaction() instead.
// See https://man7.org/linux/man-pages/man2/signal.2.html
// See also https://man7.org/linux/man-pages/man7/signal-safety.7.html
// static void print_interrupt(int signum) {
// 	if (signum == SIGINT) 
// 		printf("received SIGINT\n");
// 	if (signum == SIGUSR1) 
// 		printf("received SIGUSR1\n");
// 	fflush(stdout);
// }

//?? TODO
// Be careful of these variables, they are only visible for this file,
// but have lifetime equal to the runtime of main_1, main_stack
static FILE *log = NULL;
static int64_t wall_start = 0;
static int64_t max_time = 0;
static int first = 0;
static int64_t t_first = 0;

/**
 * Initialize signal handlers for this thread
 * @param _log        [description]
 * @param _wall_start [description]
 * @param _max_time   [description]
 */
void sig_init(FILE *_log, const int64_t _wall_start, const int64_t _max_time)
{
	static int called = 0; // could be called multiple times
	if (called == 0) {
		called = 1;
		sigaction(SIGUSR1, &(const struct sigaction){.sa_handler = stop}, NULL);
		sigaction(SIGINT, &(const struct sigaction){.sa_handler = stop}, NULL);
		sigaction(SIGTERM, &(const struct sigaction){.sa_handler = stop}, NULL);
		sigaction(SIGHUP, &(const struct sigaction){.sa_handler = stop}, NULL);
	}

	log = _log;
	wall_start = _wall_start;
	max_time = _max_time;
	first = 0;
	t_first = 0;
}

/**
 * [sig_check_state description]
 * @param  sweep        [description]
 * @param  n_sweep_warm [description]
 * @param  n_sweep      [description]
 * @return  0 if no signal received, time limit not reached, we should continue
 *          -1 if time limit reached, we should stop
 *          signum if signal received, we should stop
 */
int sig_check_state(const int sweep, const int n_sweep_warm, const int n_sweep)
{
	const int64_t t_now = time_wall();

	//reaching max time limit sets stop flag to -1
	if (max_time > 0 && t_now >= wall_start + max_time)
		stop_flag = -1;

	// First call to sig_check_state
	if (t_first == 0) {
		first = sweep;
		t_first = t_now;
	}

	// If either stop_flag or progress_flag has been set
	if (stop_flag) {
		const int warmed_up = (sweep >= n_sweep_warm);
		const double t_elapsed = (t_now - wall_start) * SEC_PER_TICK;
		const double t_done = (t_now - t_first) * SEC_PER_TICK;
		const int sweep_done = sweep - first;
		const int sweep_left = n_sweep - sweep;
		const double t_left = (t_done / sweep_done) * sweep_left;
		fprintf(log, "%d/%d sweeps completed (%s)\n",
			sweep,
			n_sweep,
			warmed_up ? "measuring" : "warming up");
		fprintf(log, "\telapsed: %.3f%c\n",
			t_elapsed < 3600 ? t_elapsed : t_elapsed/3600,
			t_elapsed < 3600 ? 's' : 'h');
		fprintf(log, "\tremaining%s: %.3f%c\n",
			(first < n_sweep_warm) ? " (ignoring measurement cost)" : "",
			t_left < 3600 ? t_left : t_left/3600,
			t_left < 3600 ? 's' : 'h');
		fprintf(log, "stop_flag = %d, stopping\n", stop_flag);
		fflush(log);
	}

	if (sweep == n_sweep_warm) {
		first = sweep;
		t_first = t_now;
	}


	// if (stop_flag < 0){
	// 	fprintf(stderr, "reached time limit, stopping\n");
	// 	fprintf(log, "reached time limit, stopping\n");
	// }
	// else if (stop_flag > 0) {
	// 	fprintf(stderr, "signal %d received, stopping\n", stop_flag);
	// 	fprintf(log, "signal %d received, stopping\n", stop_flag);
	// }
	// else if (progress_flag != 0) {
	// 	fprintf(stderr, "\tsignal %d received, checkpointing to disk\n", progress_flag);
	// 	fprintf(log, "\tsignal %d received, checkpointing to disk\n", progress_flag);
	// }

	// const int retval = (stop_flag != 0) ? 1 : (progress_flag != 0) ? 2 : 0;
	// progress_flag = 0; // reset progress flag
	// return retval;
	return stop_flag;
}
