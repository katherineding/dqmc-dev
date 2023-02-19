#!/bin/bash
export PREFIX="batch"
rm ${PREFIX}_*.h5* ${PREFIX}.h5.params stack -v
python3 ../util/gen_1band_triangular_hub.py prefix=${PREFIX} trans_sym=1 Nfiles=3 Nx=4 Ny=4 overwrite=1 tp=0 L=40 dt=0.05 U=10 mu=0 nflux=4 period_eqlt=20 meas_thermal=0 meas_2bond_corr=0 meas_energy_corr=0 meas_nematic_corr=0 meas_chiral=1 n_sweep_warm=200 n_sweep_meas=5 checkpoint_every=0 
python3 ../util/push.py stack ${PREFIX}_*.h5
