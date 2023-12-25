#!/bin/bash
PREFIX="batch"
rm -f $PREFIX* stack
python3 ../util/gen_1band_unified_hub.py --geometry=triangular --prefix=${PREFIX} --trans_sym=1 --Nfiles=4 --Nx=4 --Ny=4 --tp=0 --L=40 --dt=0.1 --U=0 --mu=0 --h=0.0 --nflux=1 --period_eqlt=5 --n_matmul=5 --period_uneqlt=1 --meas_bond_corr=1 --meas_2bond_corr=1 --meas_thermal=1 --meas_energy_corr=1 --meas_nematic_corr=1 --meas_chiral=1 --n_sweep_warm=100 --n_sweep_meas=20 --checkpoint_every=5
python3 ../util/push.py stack ${PREFIX}_*.h5

PREFIX="single"
rm -f $PREFIX* 
python3 ../util/gen_1band_unified_hub.py --geometry=triangular --prefix=${PREFIX} --trans_sym=1 --Nfiles=1 --Nx=4 --Ny=4 --tp=0 --L=40 --dt=0.1 --U=0 --mu=0 --h=0.0 --nflux=1 --period_eqlt=5 --n_matmul=5 --period_uneqlt=1 --meas_bond_corr=1 --meas_2bond_corr=1 --meas_thermal=1 --meas_energy_corr=1 --meas_nematic_corr=1 --meas_chiral=1 --n_sweep_warm=100 --n_sweep_meas=20 --checkpoint_every=5
