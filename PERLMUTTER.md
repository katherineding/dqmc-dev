# Instructions for getting GPU-offloaded code running on Perlmutter

## Setup

Login: `ssh <username>@perlmutter-p1.nersc.gov`

Clone the dqmc-dev repo `git clone git@github.com:katherineding/dqmc-dev.git perlmt-gpu` and switch the local branch to perlmt-gpu: `cd perlmt-gpu/ && git checkout perlmt-gpu`

To set up Nvidia programming environment for compilation, add the following lines to your .bashrc or .bashrc.ext file

`module load PrgEnv-nvidia nvidia cudatoolkit craype-accel-nvidia80 cray-libsci cray-hdf5`

As of April 2024, this should result in a module environment that looks something like 
```
Currently Loaded Modules:
  1) craype-x86-milan                        5) perftools-base/23.12.0       9) cray-dsmml/0.2.2           13) cudatoolkit/12.2      (g)
  2) libfabric/1.15.2.0                      6) cpe/23.12                   10) PrgEnv-nvidia/8.5.0 (cpe)  14) craype-accel-nvidia80
  3) craype-network-ofi                      7) gpu/1.0                     11) nvidia/23.9         (g,c)  15) cray-libsci/23.12.5   (math)
  4) xpmem/2.6.2-2.5_2.38__gd067c3f.shasta   8) craype/2.7.30          (c)  12) cray-mpich/8.1.28   (mpi)  16) cray-hdf5/1.12.2.3    (io)
```

Also set these environment variables:

```bash
export HDF5_USE_FILE_LOCKING=FALSE # NERSC quirk, see https://docs.nersc.gov/development/languages/python/parallel-python/
export OMP_DISPLAY_ENV=FALSE # Can set to TRUE to see openMP settings
export OMP_DISPLAY_AFFINITY=FALSE # Can set to TRUE to see openMP settings
export OMP_TARGET_OFFLOAD=MANDATORY #Make DQMC executable fail & stop if no GPUs are found
export OMP_PLACES=cores     # CPU multicore affinity
export OMP_PROC_BIND=spread # CPU multicore affinity
export NVCOMPILER_ACC_NOTIFY=0 #Can set to 1, 2, or 3 for more verbose printouts
export DEV="${HOME}/perlmt-gpu/" #pytest looks for $DEV, so set this to the folder that you cloned the repo into
```

Useful aliases for getting 1 interactive node for either CPU or gpu work:

```bash
alias sdev_cpus="salloc --nodes 1 --cpus-per-task=40  --ntasks=1                    --qos interactive --time 04:00:00 --constraint cpu --account=m2757"
alias sdev_gpus="salloc --nodes 1 --cpus-per-task=32  --ntasks=4  --gpus-per-task=1 --qos interactive --time 04:00:00 --constraint gpu --account=m2757_g"
```

Useful alias for checking queue/job status:

```bash
squeue -u $USER --sort='+t' -o "%.17i %.25j %.12P %.12q %.2t %.5D %.5C %.8Q %.20V %.10S %.10M %.10l %.10e"; sprio -u $USER
```

To set up Python environment for generating hdf5 files, testing, and data analysis, add the following lines to your .bashrc or .bashrc.ext file

`module load python/3.11`

Additionally, you need to install `gitpython` and `pytest`. It's probably easiest to do `pip3 install --user GitPython pytest` (this only needs to be done one time, and then they're always installed)

## Compile

With everything set up as above, go to `$DEV/build` and run `bash build_rc_pgpu.sh` to compile. Everything should go through and you end up with 4 DQMC executables for both real and complex numbers in single and stack form.

## Test

Put the folder `ref` inside `$DEV/test`. Request an interactive gpu job via `sdev_gpus`. This ensures that you actually have GPUs to offload to. Within the `$DEV/test` folder, run `pytest -v -s test_gen_1band_hub_unified.py`. For more information, check TESTING.md.

Currently the test set should have 2 failures, in test_ref[0-honeycomb] and test_ref[3-honeycomb] due to a known bugs in a reference commit b535e68.