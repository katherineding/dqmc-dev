#!/bin/bash
echo using hdf5 generation file in $DEV
rm stack* *.h5.log -v
python3 ${DEV}/util/gen_1band_hub.py Nfiles=3 Nx=8 Ny=8 overwrite=1 tp=-0.25 L=80 dt=0.05 nflux=1 period_uneqlt=2 meas_thermal=1 meas_2bond_corr=1 meas_energy_corr=1 meas_nematic_corr=0 n_sweep_warm=10 n_sweep_meas=10 checkpoint_every=5 &&
python3 ${DEV}/util/push.py stack *.h5
