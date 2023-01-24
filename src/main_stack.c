//#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include <argp.h>

#include "data.h"
#include "dqmc.h"
#include "time_.h"


// TODO: hard vs soft link?
#define USE_HARD_LINK
// max allowed number of chars in absolute path string to any *.h5 file
//   in stack_file
#define MAX_LEN 512 
// TODO: ??
#define BUF_SZ 128

static char hostname[65];
static int pid;

// print arg (...) to console with hostname and pid info
#define my_printf(...) do { \
	printf("%16s %6d: ", hostname, pid); \
	printf(__VA_ARGS__); \
	fflush(stdout); \
} while (0)

// sleep a number of seconds between min and max (assumes both > 0)
// so that processes don't repeatedly try to do something simultaneously
static void sleep_rand(double min, double max)
{
	const double t = min + (max - min)*(double)rand()/RAND_MAX;
	const struct timespec ts = {(time_t)t, (long)(1e9*(t - (int)t))};
	nanosleep(&ts, NULL);
}

/**
 * "Lock" file for RW
 * @param  file  [description]
 * @param  retry [description]
 * @return       0 on successful lock
 *               1 on failure
 * TODO: can we replace this with something better?
 */
static int lock_file(const char *file, const int retry)
{
	const size_t len_file = strlen(file);
	//char *lfile = my_calloc(len_file + 2);
	char *lfile = calloc(len_file + 2, sizeof(char));
	memcpy(lfile, file, len_file);
	lfile[len_file] = '~';

	struct timespec lock_mtime = {0};
	int cycles_same_mtime = 0;
	while (1) {
#ifdef USE_HARD_LINK
		if (link(file, lfile) == 0) {
#else
		if (symlink(file, lfile) == 0) { // successfully locked
#endif
			free(lfile);
			return 0;
		}

		if (!retry) return 1;
		// this method to automatically release zombie locks allows for
		// a possible race condition due to the gap between stat and
		// releasing the zombie lock

		// s = stat, z = zombie lock released, c = create,
		// r = regular lock release, e = editing file
		// process 1: ...s    s    s  z
		// process 2:   ...s    s    s  z
		// process 3:                  c eeeeeee r
		// process 4:                       c eeeeeee r

		// these incidents can be logged because r would fail, but
		// failure of r doesn't necessarily imply simultaneous editing

		// check mtime
		struct stat statbuf = {0};
		if (stat(lfile, &statbuf) != 0) // file gone
			cycles_same_mtime = 0;
		else if (statbuf.st_mtim.tv_sec == lock_mtime.tv_sec &&
				statbuf.st_mtim.tv_nsec == lock_mtime.tv_nsec)
			cycles_same_mtime++;
		else {
			cycles_same_mtime = 0;
			lock_mtime = statbuf.st_mtim;
		}

		// if mtime stays the same for 10 cycles, assume the locking
		// process died and force unlock the file.
		if (cycles_same_mtime >= 10) {
			remove(lfile);
			my_printf("warning: zombie lock released\n");
			sleep_rand(1.0, 2.0);
		}

		// wait 1-3s before looping again
		sleep_rand(1.0, 3.0);
	}
}

static void unlock_file(const char *file)
{
	const size_t len_file = strlen(file);
	//char *lfile = my_calloc(len_file + 2);
	char *lfile = calloc(len_file + 2, sizeof(char));
	memcpy(lfile, file, len_file);
	lfile[len_file] = '~';

	if (remove(lfile) != 0)
		my_printf("warning: lock release failed (already removed?)\n");

	free(lfile);
}

/**
 * LIFO remove last path from file and place it in line
 * @param  char* file string
 * @param  char* line string
 * @return 0 on success of both operations; file modified, line has content
 *         -1 if I/O error;    file not modified, line empty
 *         1  if no I/O error; file not modified, line empty
 */
static int pop_stack(const char *file, char *line)
{
	int ret = 0;
	int len_line = 0;
	memset(line, 0, MAX_LEN);

	lock_file(file, 1);

	const int fd = open(file, O_RDWR);
	if (fd == -1) {
		unlock_file(file);
		my_printf("error: open() failed in pop_stack\n");
		return -1;
	}

	off_t offset = lseek(fd, 0, SEEK_END);
	if (offset == -1) {
		my_printf("error: lseek() failed in pop_stack()\n");
		ret = -1;
		goto end;
	}

	char buf[BUF_SZ + 1];

	int start = BUF_SZ, end = BUF_SZ;

	while (offset > 0) {
		offset -= BUF_SZ;
		const int ofs = offset < 0 ? 0 : offset;
		const int actual_ofs = lseek(fd, ofs, SEEK_SET);
		if (actual_ofs == -1) {
			my_printf("error: lseek() failed in pop_stack()\n");
			ret = -1;
			goto end;
		} else if (actual_ofs != ofs) {
			my_printf("error: actual_ofs=%d, requested=%d\n",
					   actual_ofs, ofs);
			ret = -1;
			goto end;
		}

		memset(buf, 0, sizeof(buf));
		if (end != BUF_SZ)
			end = offset < 0 ? BUF_SZ + offset - 1 : BUF_SZ - 1;
		if (read(fd, buf, end == BUF_SZ ? BUF_SZ : end + 1) == -1) {
			my_printf("error: read() failed in pop_stack()\n");
			ret = -1;
			goto end;
		}

		if (end == BUF_SZ)
			for (; end >= 0; end--)
				if (buf[end] != 0 && buf[end] != '\n')
					break;

		if (end == -1) {
			end = BUF_SZ;
			continue;
		}

		for (start = end; start >= 0; start--)
			if (buf[start] == '\n') {
				start++;
				break;
			}
		if (start == -1) start = 0;

		const int new_chars = end - start + 1;
		if (len_line + new_chars > MAX_LEN) {
			my_printf("len_line = %d, end = %d, start = %d, new_chars = %d, "
				"error: last line length > %d\n", len_line, end, start, new_chars, MAX_LEN);
			ret = -1;
			goto end;
		}
		//last line length is not an accurate error message!

		if (new_chars > 0) {
			if (len_line > 0)
				memmove(line + new_chars, line, len_line);
			memcpy(line, buf + start, new_chars);
			len_line += new_chars;
		}

		if (start != 0) // start of last line found
			break;
	}

	if (len_line == 0) {
		ret = 1;
	} else if (ftruncate(fd, (offset < 0 ? 0 : offset) + start) == -1) {
		my_printf("error: ftruncate failed in pop_stack()\n");
		ret = -1;
	}

end:
	if (close(fd) == -1)
		my_printf("error: close() failed in pop_stack()\n");
	unlock_file(file);
	return ret;
}

static void push_stack(const char *file, const char *line)
{
	// add newline here instead of using fputc
	const size_t len_line = strlen(line);
	//char *line_nl = my_calloc(len_line + 2);
	char *line_nl = calloc(len_line + 2, sizeof(char));
	memcpy(line_nl, line, len_line);
	line_nl[len_line] = '\n';
	char *backup = NULL;

	int status = 0;

	if (lock_file(file, 0) != 0) { // if locking fails on first try
		// save line to backup in case process gets killed
		const size_t len_file = strlen(file);
		//backup = my_calloc(len_file + 32);
		backup = calloc(len_file + 32, sizeof(char));
		snprintf(backup, len_file + 32, "%s_%.20s_%d", file, hostname, pid);

		const int bfd = open(backup, O_CREAT | O_WRONLY | O_APPEND, 0644);
		if (bfd == -1)
			status |= 1;
		else {
			if (write(bfd, line_nl, len_line + 1) != (ssize_t)len_line + 1)
				status |= 2;
			if (close(bfd) == -1)
				status |= 4;
		}

		// now try locking with retrying enabled
		lock_file(file, 1);
	}

	const int fd = open(file, O_WRONLY | O_APPEND);
	if (fd == -1)
		status |= 1;
	else {
		if (write(fd, line_nl, len_line + 1) != (ssize_t)len_line + 1)
			status |= 2;
		if (close(fd) == -1)
			status |= 4;
	}

	unlock_file(file);

	if (backup != NULL) {
		if (status == 0)
			remove(backup);
		free(backup);
	}

	free(line_nl);

	if (status & 1)
		my_printf("error: open() failed in push_stack()\n");
	if (status & 2)
		my_printf("error: write() failed or incomplete in push_stack()\n");
	if (status & 4)
		my_printf("error: close() failed in push_stack()\n");
}



// boilerplate required by argp
const char *argp_program_version= "commit id " GIT_ID "\n"
	"compiled on " __DATE__ " " __TIME__;
const char *argp_program_bug_address = "<jxding@stanford.edu>";

// Options.  Field 1 in ARGP.
// Order of fields: {NAME, KEY, ARG, FLAGS, DOC, GROUP}.
static struct argp_option options[] = {
	{0, 't', "TIME", 0, "Max allowed wall time in integer seconds", 0},
	{"dry-run", 'n', 0, 0,"Run through stack, report mem requirement only", 0},
	{"benchmark", 'b', 0, 0,"Don't save sim_data to disk", 0},
	{0}
};

/* Used by main to communicate with parse_opt */
struct arguments{
	int time; //optional argument
	bool dry_run;
	bool bench;
	char *args[1]; // mandatory argument
};

/* Parser function. Field 2 in ARGP.*/
static error_t parse_opt(int key, char *arg, struct argp_state *state) 
{
	/* Get the input argument from argp_parse, which we
	know is a pointer to our arguments structure. */
	struct arguments *arguments = state->input;
	switch (key) {
		case 't': 
			arguments->time = atoi(arg); 
			break;
		case 'b': 
			arguments->bench = true; 
			break;
		case 'n': 
			arguments->dry_run = true; 
			break;
		case ARGP_KEY_ARG: // mandatory argument
			if (state->arg_num >= 1) {
				/* Too many arguments. */
				argp_usage(state);
			}
			arguments->args[state->arg_num] = arg;
			break;
		case ARGP_KEY_END:
			if (state->arg_num < 1) {
				/* Not enough arguments. */
				argp_usage(state);
			}
			break;
		default: 
			return ARGP_ERR_UNKNOWN;
	}
	return 0;
} 

/* Brief program documentation. Field 4 in ARGP.*/
static char doc[] = "Complex-capable DQMC, stack version";
/* names of mandatory arguments Field 3 in ARGP.*/
static char args_doc[] = "stack_file";

// This is very important
static struct argp argp = { options, parse_opt, args_doc, doc }; 

int init_setting(void){
	//static variables for printout
	gethostname(hostname, 64); //from <unistd.h>
	pid = getpid(); //from <unistd.h>
	
	// OPENMP
	// const int max_threads = omp_get_max_threads();
	// const int num_procs = omp_get_num_procs();
	// const int default_device = omp_get_default_device();
	// const int num_devices = omp_get_num_devices();
	// const int dn = omp_get_device_num();
	// const int is_init = omp_is_initial_device();

	// my_printf("number of processors available to device: %d\n",num_procs);
	// my_printf("max omp threads: %d\n",max_threads);
	// my_printf("default device: %d\n",default_device);
	// my_printf("num_devices: %d\n",num_devices);
	// my_printf("device number: %d\n",dn);
	// my_printf("is initial device? %d\n",is_init);

	omp_set_num_threads(DQMC_NUM_SECTIONS);

	//set hdf5 library data type
	set_num_h5t(); 

	//seed random number generator
	srand((unsigned int)pid); 

	return 0;

}

int main(int argc, char **argv)
{	
	// Get command line arguments arguments
	struct arguments arguments;
	// Default: If no -t max time supplied, then run indefinitely until
	// user interrupt, done with stack file, or ?? TODO
	arguments.time = 0; 
	arguments.dry_run = false; //default: really run everything
	arguments.bench = false;   //default: no benchmark, save data
	argp_parse(&argp, argc, argv, 0, 0, &arguments); 

	init_setting();


	sleep_rand(0.0, 4.0); //why sleep for some time first?

	// Start timing
	const int64_t t_start = time_wall();
	const int max_time = arguments.time;
	const int64_t t_stop = t_start + max_time * TICK_PER_SEC;

	const char *stack_file = arguments.args[0];
	char sim_file[MAX_LEN + 1] = {0};
	char log_file[MAX_LEN + 5] = {0};

	//check that stack exists, and we can both R and W to it.
	if (access(stack_file,R_OK | W_OK ) != 0) {
		my_printf("[ERROR] %s does not exist or you don't have access; "
			"idling\n", stack_file);
		return EXIT_FAILURE;
	}

	int64_t t_remain = 0;
	while (1) {

		if (max_time > 0) {
			t_remain = t_stop - time_wall();
			if (t_remain <= 0) {
				my_printf("dqmc_stack reached wall time limit; idling");
				return EXIT_SUCCESS;
			}
		}

		int pop_status = pop_stack(stack_file, sim_file);
		if (pop_status == -1) { // pop_stack() failed
			my_printf("[ERROR] pop_stack() failed; idling\n");
			return EXIT_SUCCESS;
		}
		else if (pop_status == 1) { // stack_file empty
			my_printf("pop_stack() returned %d, %s empty; idling\n", 
				pop_status, stack_file);
			return EXIT_SUCCESS;
		}

		// pop success, sim_file has content
		const size_t len_sim_file = strlen(sim_file);
		memcpy(log_file, sim_file, len_sim_file);
		memcpy(log_file + len_sim_file, ".log", 5);

		// my_printf("memory requirement: %d bytes\n", get_memory_req(sim_file));
		my_printf("starting:   %s\n", sim_file);
		my_printf("logging to: %s\n", log_file);
		// run dqmc here 
		const int wrap_status = dqmc_wrapper(sim_file, log_file, t_remain, 
			arguments.dry_run, arguments.bench);

		if (wrap_status > 0) {
			push_stack(stack_file, sim_file);
			my_printf("dqmc_wrapper() incomplete, pushed: %s\n", sim_file);
			// checkpoint would only happen if signal received or
			// time limit reached, so break here
			break;
		} 
		else if (wrap_status == 0)  
			my_printf("completed: %s\n", sim_file);
		else
			my_printf("dqmc_wrapper() failed: %s\n", sim_file);
	}

	return EXIT_SUCCESS;
}
