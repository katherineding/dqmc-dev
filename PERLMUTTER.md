# Instructions for getting GPU-offloaded code running on Perlmutter

## Setup

Login: `ssh <username>@perlmutter-p1.nersc.gov`

Clone the dqmc-dev repo `git clone git@github.com:katherineding/dqmc-dev.git perlmt-gpu` and switch the local branch to perlmt-gpu: `cd perlmt-gpu/ && git checkout perlmt-gpu`

The default module environment without any user intervention looks like:
```
Currently Loaded Modules:
  1) craype-x86-milan
  2) libfabric/1.15.2.0
  3) craype-network-ofi
  4) xpmem/2.6.2-2.5_2.38__gd067c3f.shasta
  5) PrgEnv-gnu/8.5.0
  6) cray-dsmml/0.2.2
  7) cray-libsci/23.12.5
  8) cray-mpich/8.1.28
  9) craype/2.7.30
 10) gcc-native/12.3
 11) perftools-base/23.12.0
 12) cpe/23.12
 13) cudatoolkit/12.2
 14) craype-accel-nvidia80
 15) gpu/1.0
```

<strike> To set up Nvidia programming environment for compilation, add the following lines to your .bashrc or .bashrc.ext file 

module load PrgEnv-nvidia nvidia cudatoolkit craype-accel-nvidia80 cray-libsci cray-hdf5

As of April 2024, this results in a default module environment that looks something like 

Currently Loaded Modules:
  1) craype-x86-milan
  2) libfabric/1.15.2.0
  3) craype-network-ofi
  4) xpmem/2.6.2-2.5_2.38__gd067c3f.shasta
  5) perftools-base/23.12.0
  6) cpe/23.12
  7) gpu/1.0
  8) craype/2.7.30                         (c)
  9) cray-dsmml/0.2.2
 10) PrgEnv-nvidia/8.5.0                   (cpe)
 11) nvidia/23.9                           (g,c)
 12) cray-mpich/8.1.28                     (mpi)
 13) cudatoolkit/12.2                      (g)
 14) craype-accel-nvidia80
 15) cray-libsci/23.12.5                   (math)
 16) cray-hdf5/1.12.2.3                    (io)
</strike>


For unclear reasons the Perlmutter [Jan 17, 2024 upgrade](https://docs.nersc.gov/systems/perlmutter/timeline/#january-17-2024) severely degrades performance compared to the prevous environment, notably in the `meas_uneq`-`meas_uneq_sub`, `meas_eq`, `recalc`, `half_wrap` and `update` parts of the code, resulting in doubling wall time (5 sec -> 10 sec) in the `square_8x8_nflux3_tp-0.25_nmeas40_11111` test case.

Best workaround right now: restore the Dec 2023 environment as best as possible. This can be done by adding to your .bashrc or .bashrc.ext file

```bash
module load cpe/23.03 
module load PrgEnv-nvidia/8.4.0 nvidia/22.7 cudatoolkit/11.7 craype-accel-nvidia80 
module load cray-libsci/23.09.1.1 cray-hdf5/1.12.2.7
```

This results in a module environment that looks something like 

```
Currently Loaded Modules:
  1) craype-x86-milan
  2) libfabric/1.15.2.0
  3) craype-network-ofi
  4) xpmem/2.6.2-2.5_2.38__gd067c3f.shasta
  5) gpu/1.0
  6) perftools-base/23.03.0                (dev)
  7) cpe/23.03                             (cpe)
  8) craype/2.7.20                         (c)
  9) cray-dsmml/0.2.2
 10) PrgEnv-nvidia/8.4.0                   (cpe)
 11) nvidia/22.7                           (g,c)
 12) cray-mpich/8.1.25                     (mpi)
 13) cudatoolkit/11.7                      (g)
 14) craype-accel-nvidia80
 15) cray-libsci/23.09.1.1                 (math)
 16) cray-hdf5/1.12.2.7                    (io)
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

<strike>module load python/3.11</strike> [This module causes slowdown in crude profiling, but is it a real effect? If so, why?] `module load python/3.9-anaconda-2021.11`

Additionally, you need to install `gitpython` and `pytest`. It's probably easiest to do `pip3 install --user GitPython pytest` (this only needs to be done one time, and then they're always installed)

## Compile

With everything set up as above, go to `$DEV/build` and run `bash build_rc_pgpu.sh` to compile. Everything should go through and you end up with 4 DQMC executables for both real and complex numbers in single and stack form.

## Test

Put the folder `ref` inside `$DEV/test`. Request an interactive gpu job via `sdev_gpus`. This ensures that you actually have GPUs to offload to. Within the `$DEV/test` folder, run `pytest -v -s test_gen_1band_hub_unified.py`. For more information, check TESTING.md.

Currently the test set should have 2 failures, in `test_ref[0-honeycomb]` and `test_ref[3-honeycomb]` due to known bugs in a reference commit b535e68.
