#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <stdbool.h>
#include <argp.h>

#include "data.h"
#include "dqmc.h"
#include "time_.h"


// boilerplate required by argp
const char *argp_program_version= "commit id " GIT_ID "\n"
	"compiled on " __DATE__ " " __TIME__;
const char *argp_program_bug_address = "<jxding@stanford.edu>";

// Options.  Field 1 in ARGP.
// Order of fields: {NAME, KEY, ARG, FLAGS, DOC, GROUP}.
static struct argp_option options[] = {
	{0, 't', "TIME", 0, "Max allowed wall time in integer seconds", 0},
	{"dry-run", 'n', 0, 0,"Run through stack, report mem requirement only", 0},
	{"log", 'l', "LOGFILE", 0,"Redirect stdout to LOGFILE", 0},
	{"benchmark", 'b', 0, 0,"Don't save sim_data to disk", 0},
	{0}
};

/* Used by main to communicate with parse_opt */
struct arguments{
	int time; //optional argument
	char *log_file;
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
		case 'l': 
			arguments->log_file = arg; 
			break;
		case 'n': 
			arguments->dry_run = true; 
			break;
		case 'b': 
			arguments->bench = true; 
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
static char doc[] = "Complex-capable DQMC, single file version";
/* names of mandatory arguments Field 3 in ARGP.*/
static char args_doc[] = "sim_file.h5";

// This is very important
static struct argp argp = { options, parse_opt, args_doc, doc }; 


int main(int argc, char **argv)
{


	// Get command line arguments
	struct arguments arguments;
	// Default: If no -t max time supplied, then run indefinitely until
	// user interrupt, done with stack file, or ?? TODO
	arguments.time = 0; 
	arguments.log_file = NULL; //default: no log file
	arguments.dry_run = false; //default: really run everything
	arguments.bench = false;   //default: no benchmark, save data
	argp_parse(&argp, argc, argv, 0, 0, &arguments); 

	set_num_h5t(); //set hdf5 library data type
	omp_set_num_threads(DQMC_NUM_SECTIONS);

	const char *sim_file = arguments.args[0];

	const int wrap_status = dqmc_wrapper(sim_file, arguments.log_file,
		arguments.time, arguments.dry_run, arguments.bench);

	// if (wrap_status > 0) {
	// 	printf("dqmc_wrapper() returned incomplete: %s\n", sim_file);
	// } 
	// else if (wrap_status == 0)  
	// 	printf("completed: %s\n", sim_file);
	// else
	// 	printf("dqmc_wrapper() failed: %s\n", sim_file);
	return wrap_status;
}
