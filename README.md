# Complex-capable DQMC

This DQMC code is for single-band Hubbard model, allowing kinetic hopping to be modified by Peierl's phase (which requires complexifying the entire simulation).

Simulation parameters (including geometry and potentially multiple orbitals) are controlled by python utility script `gen_1band_unified_hub.py` in `util/`. The default geometry is square. The goal was to make the C code as "model parameter agnostic" as possible. Whether this is achieved is debatable -- it is usually still necessary to manually modify the C source code when we want to study a slightly different model, or add new measurements.

The source code is in C and uses idioms like `goto`, `restrict`, and implicit casting of `void *`pointers into other pointer types, which are not consistent with C++ standards. So if you try to compile the code with a C++ compiler, you'll likely get compilation errors.

This program relies on POSIX C APIs, like `clock_gettime()` and `sigaction()`. It works in various flavours of Linux. It probably works in MacOS. It probably doesn't work in Windows. There's no plans to add support for Windows.

## Prerequisites

- `git`
- `make`

### For compilation

Unfortunately, as we are dealing with a variety of computing environments, both the source code and the Makefile must be adjusted, based on what compilers, BLAS + LAPACK libraries, and offload devices (if any) are available.

1. [Sherlock](https://www.sherlock.stanford.edu/docs/) or [Cori KNL](https://docs.nersc.gov/systems/cori/), CPU only: `master` branch
	- Intel compiler `icc`
	- `imkl` headers and library `>= 2019`
	- `hdf5` headers and library `>= 1.10`
	To get correct paths to these headers and libraries on Sherlock, add `module load hdf5/1.10.2 icc/2019 imkl/2019` to your .bashrc.
2. [Perlmutter](https://docs.nersc.gov/systems/perlmutter/), CPU only: `perlmt-cpu` branch
	*Note: this branch may be modified to use the AOCC compiler and AOCL math libraries when it becomes supported in the future*
	- GNU Compiler `gcc`
	- `cray-libsci` headers and library
	- `hdf5` headers and library
3. [Perlmutter](https://docs.nersc.gov/systems/perlmutter/), with GPU offloading: `perlmt-gpu` branch 
	*Note: this branch may be modified to use the AOCL math libraries when it becomes supported in the future*
	- Nvidia Compiler `ncc`
	- `cray-libsci` headers and library
	- `hdf5` headers and library
4. gcc + imkl, CPU only: `master` branch
	- GNU compiler `gcc`
	- `imkl` headers and library `>= 2019`
	- `hdf5` headers and library `>= 1.10`
	To get correct paths to these headers and libraries on Sherlock, add `module load hdf5/1.10.2 gcc/10.1.0 imkl/2019` to your .bashrc. The `icc/2019` module must be unloaded or it messes up the search paths.

### For python scripts in `util/`
- `python3`
- `numpy`
- `h5py >= 2.5.0`
- `gitpython`
- `scipy`

You can get these via miniconda/anaconda.

On sherlock, do `pip3 install --user gitpython h5py` once. Add `module load python/3.6.1 py-scipy/1.1.0_py36 py-numpy/1.14.3_py36 viz py-matplotlib/3.1.1_py36` to your .bashrc.


## Compilation (Master Branch)

Go to build/

Optionally, replace `-xHost` in Makefile or Makefile.icx (`-march` in gcc.imkl.Makefile) with appropriate optimized instruction set flags.

Mandatory: pick whether to compile with `-DUSE_CPLX`. Real DQMC can only be used with hdf5 files generated with `nflux=0` option, while Complex DQMC can only be used with hdf5 files generated with `nflux!=0` option. 

Optional: Set a sensible number of `OMP_MEAS_NUM_THREADS` to use for the slowest unequal time measurements. The default is 2.

Run `make -f <makefilename>`.

## Usage

To (batch-)generate simulation files, run
`python3 gen_1band_unified_hub.py <parameter arguments>`

To push some .h5 files to a stack, run something like
`python3 push.py <stackfile_name> <some .h5 files>`

Run dqmc in single file mode:
`./dqmc_1 <options> file.h5`

Run dqmc in stack mode:
`./dqmc_stack <options> stackfile`

Command line options for `dqmc_1`, `dqmc_stack`,`gen_1band_unified_hub.py` are found by using the standard `--help` or `--usage` flags.

To check estimated memory usage and exit for `dqmc_1`, `dqmc_stack`, toggle `--dry-run` or `-n`. But note this option for `dqmc_stack` edits the stack file, so you have to re-add the .h5 files back to the stack file after this.

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

Thus many parallel processes compete to access the same `stack` file which serves as a LIFO job board listing all the .h5 files that needs to be worked on. There are `pop()` and `push()` functions which implement a crude locking mechanism for preventing race conditions. These functions are not fool-proof however, so you might get random warnings and failures. These are usually not critical, but -- 

A catastropic failure mode that is NOT safeguarded against is if the same file is listed in `stack` twice, or otherwise somehow picked up by two different processes to each individually R/W to the same .h5 file. This causes all sorts of inconsistent state/race conditions in both the .h5 and the .h5.log file!! I want to say this never happens, but this needs more testing.

Unix system signals SIGINT, SIGTERM, SIGHUP, SIGUSR1 are caught by signal handlers and used to set a stop flag. The dqmc() loop checks for the stop flag every full H-S sweep. Upon reaching time limit or receiving an interrupt signal, simulation stops and throws away all unsaved data. We essentially regress to the last valid checkpoint.

Checkpointing (save current simulation state and measurements to disk) is by default performed every 10000 full H-S sweeps. This may still be too frequent, so it's user adjustable in simulation file generation. If `--checkpoint_every=0`, no checkpointing is performed (so upon any interrupt, all working data is lost). OTOH, upon successful completion of all H-S sweeps,  simulation state and measurements will be saved to disk.

To have true benchmarking mode, where you never save *any* data to disk, (and the .h5 files remain in their initial, untouched state), genereate simulation file with `--checkpoint_every=0` and run dqmc with `./<executable> -b`.

A crude mechanism for detecting hdf5 file corruption is implemented by setting a `partial_write` flag. Upon detecting any corruption, we just give up on working on this file entirely. Partial writes can occur if the simulation is killed in the middle of performing `sim_data_save()`. I'm not sure this method is watertight yet, needs more testing.

## Differences from Edwin's Code

This version of the DQMC code is based on [edwnh/dqmc](https://github.com/edwnh/dqmc) commit c91ba610cab2418e575a2008094499ea0e35754a. Divergence from Edwin's code as of 08/2023 at this point include:

- Removed `tick_t` alias

- Rewrote command line option parsing for `dqmc_1` and `dqmc_stack`to use `argp`. Added `--dry-run` or `-n` for checking memory consumption.

- Added consistency check between hdf5 simulation file generation script and compiled dqmc executable. Inconsistent versions do not necessarily indicate a problem.

- Changed checkpoint behavior to be more conservative, as described above.

- Added `partial_write` file corruption check. 

- More verbose log information.

- ~~Different return flags for functions. Most notably nonzero return codes for main() functions. This may trigger unexpected SLURM behavior. Needs testing.~~

- Removed unused python dqmc source code.

- Added thermal phases, thermal, 2bond measurements. 

- Added python thermal transport analysis scripts in `util/`

- Added scalar spin chirality measurement for some lattices.

- Unified all `gen_1band_xxx.py` scripts into one file `gen_1band_unified_hub.py`, but bond definitions required for e.g. transport measurements, and optional measurements are not implemented for all lattices. 

- `gen_1band_unified_hub.py` takes `argp` style arguments as opposed to the default python syntax. This means all arguments are in the form `--name=value` rather than Edwin's version, which is `name=value`

- Added option to apply twisted boundary conditions

- Added hoppings farther than next nearst neighbor for some lattices.


## TODOs

- ~~Add profiling for how much overhead regular checkpoints add.~~
- Make `double` vs `complex double` a runtime choice, so we don't have to do separate compilations
- Make dry run completely side-effect-free
- Improve stack mechanism to reduce competition and wait times -- double ended queue? process private queues? But this is not the main bottleneck right now.
- ~~Add safeguards for simultaneous hdf5 file RW failure mode~~
- ~~BUGFIX for thermal phase `#define`s b/c of premature optimization.~~
- Add check for consistency between C standard `double _Complex` and whatever idiosyncratic complex type (probably a struct) the math library (AOCL, cray-libsci, IMKL, cuBLAS) is using, to make sure they are both 16 bytes.
- Add a last_modified field to keep hdf5 files refreshed and always in $SCRATCH dir?
- my_calloc() might no longer be the most optimal thing to do on AMD CPUs. It also may be less important to worry about memory alignment if matrix operations are offloaded to GPUs. 
- Add example workflow