# Complex-capable DQMC

This program relies on POSIX C APIs, like `clock_gettime()` and `sigaction()`. It works in various flavours of Linux. It probably works in MacOS. It probably doesn't work in Windows. 

## Prerequisites

- `git`
- `make`

### For compilation

- Intel compiler `icc`/`icx`
- `imkl` headers and library `>= 2019`
- `hdf5` headers and library `>= 1.10`

### For python scripts in `util/`
- `python3`
- `numpy`
- `h5py >= 2.5.0`
- `gitpython`
- `scipy`

You can get these via miniconda/anaconda.

## Compilation

Go to build/

Optionally, replace `-xHost` in Makefile or Makefile.icx with appropriate instruction set flag.

Mandatory: pick whether to compile with `-DUSE_CPLX`. Real DQMC can only be used with hdf5 files generated with `nflux=0` option, while Complex DQMC can only be used with hdf5 files generated with `nflux!=0` option. 

Run `make` if using `icc` or `make -f Makefile.icx` when using `icx`

## Usage

To (batch-)generate simulation files, run
`python3 gen_1band_hub.py <parameter arguments>`

To push some .h5 files to a stack, run something like
`python3 push.py <stackfile_name> <some .h5 files>`

Run dqmc in single file mode:
`./dqmc_1 [options] file.h5`

Run dqmc in stack mode:
`./dqmc_stack [options] stackfile`

Command line options are found by using the standard `--help` or `--usage` flags.

To check estimated memory usage and commit version, toggle `--dry-run` or `-n`. But note this option edits the stack file. 

## Best practice

In cluster environments, you should 
- *definitely* place your compiled DQMC executable in some permanent storage directory like `$HOME`.
- *definitely* place hdf5 simulation files in a fast I/O directory (like `$SCRATCH`), because reading and writing to hdf5 files take time, and we want to, when interruped, be able to save program state, log, and checkpoint to disk gracefully.
- *preferably* place any stack files in a fast I/O directory (like `$SCRATCH`), because that's your job board that many processes compete to R/W.
- *preferably* submit batch slurm scripts from a fast I/O directory (like `$SCRATCH`), because slurm directs `stdout` and `stderr` to `.out` (and maybe `.err` if you requested separation) files. 
- *definitely* backup completed simulations in `$SCRATCH` to permanent long term storage.

## Notes

When running in stack mode on a cluster, you might do something like
`srun -n 4 ./dqmc_stack stack` which launches 4 processes/steps, each individually running dqmc, but accessing the same stack file. The situation is something like:

> p0: `./dqmc_stack stack`
>
> p1: `./dqmc_stack stack`
>
> p2: `./dqmc_stack stack`
>
> p3: `./dqmc_stack stack`

Thus many parallel processes compete to access the same `stack` file which serves as a LIFO job board listing all the .h5 files that needs to be worked on. There are `pop()` and `push()` functions which implement a crude locking mechanism for preventing race conditions. 

A catastropic failure mode that is NOT safeguarded against is if the same file is listed in `stack` twice, and it is picked up by two different processes to each individually R/W to the same .h5 file. This causes all sorts of inconsistent state/race conditions in both the .h5 and the .h5.log file!!  

Unix system signals SIGINT, SIGTERM, SIGHUP, SIGUSR1 are caught by signal handlers and used to set a stop flag. The dqmc() loop checks for the stop flag every full H-S sweep. Upon reaching time limit or receiving an interrupt signal, simulation stops and throws away all unsaved data. We essentially regress to the last valid checkpoint.

Checkpointing (save current simulation state and measurements to disk) is by default performed every 1000 full H-S sweeps. This is user adjustable in simulation file generation. If `checkpoint_every=0`, no checkpointing is performed at all (so upon any interrupt, all working data is lost).

To have true benchmarking mode, genereate simulation file with `checkpoint_every=0` and run dqmc with `./<executable> -b`.

A crude mechanism for detecting hdf5 file corruption is implemented by setting a `partial_write` flag. Upon detecting any corruption, we just give up on working on this file entirely. Partial writes can occur if the simulation is killed in the middle of performing `sim_data_save()`. I'm not sure this method is watertight yet, needs more testing.

## Differences from Edwin's Code

This version of the DQMC code is based on edwnh/dqmc commit c91ba61. Divergence from Edwin's code at this point include:

- Remove `tick_t` alias

- Rewrote command line option parsing for `dqmc_1` and `dqmc_stack`to use `argp`. Added `dry-run` for checking memory consumption and checking git version consistency for hdf5 simulation file generation script and compiled dqmc executable. Inconsistent versions do not necessarily indicate a problem.

- Changed checkpoint behavior to be more conservative. Added file corruption check. This may change in future.

- More verbose log information.

- Different return flags for functions. Most notably nonzero return codes for main() functions. This may Trigger unexpected SLURM behavior. Needs testing.

- More comments and documentation.

- Removed unused python dqmc source code.

- Added thermal phases, thermal, 2bond measurements. 

- Added triangular lattice python generation script.

## TODOs

- Make `double` vs `complex double` a runtime choice, so we don't have to do separate compilations
- Make dry run completely side-effect-free
- Improve stack mechanism to reduce competition and wait times -- double ended queue? process private queues? But this is not the main bottleneck right now.
- Add safeguards for simultaneous hdf5 file RW failure mode
- BUGFIX for thermal phase `#define`s b/c of premature optimization.
- `aocc + aocl` make path.