#!/bin/bash
echo using hdf5 generation file in $DEV
rm stack* *.h5.log -v
python3 ${DEV}/util/gen_1band_hub.py Nfiles=4 Nx=6 overwrite=1 tp=-0.25 nflux=1 period_uneqlt=2 meas_thermal=1 meas_2bond_corr=1 meas_energy_corr=1 meas_nematic_corr=0 n_sweep_warm=100 n_sweep_meas=400 checkpoint_every=0 &&
python3 ${DEV}/util/push.py stack *.h5
