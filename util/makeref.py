import gen_1band_unified_hub as ghub
import util
import numpy as np
import os


# import matplotlib.pyplot as plt

src = os.environ["DEV"]

geometry_list = ["square", "triangular", "honeycomb", "kagome"]
seed = 1234


def gen_ref():
    for geometry in geometry_list:

        if geometry == "square":
            tp=-0.25
        else:
            tp = 0
        ghub.create_batch(
            prefix=f"{ghub.hash_short}_{geometry}_{seed}_nflux0",
            geometry=geometry,
            Nx = 4, Ny = 4, dt = 0.1, L = 40, nflux=0, period_uneqlt=2,
            meas_bond_corr = 1, meas_energy_corr = 1,
            meas_chiral=1, meas_thermal=1, meas_2bond_corr=1,
            trans_sym =1, Nfiles=3, tp = tp,
            seed=seed, overwrite=1, n_sweep_warm=20, n_sweep_meas=20
        )
        ghub.create_batch(
            prefix=f"{ghub.hash_short}_{geometry}_{seed}_nflux3",
            geometry=geometry,
            Nx = 4, Ny = 4, dt = 0.1, L = 40, nflux=3, period_uneqlt=2,
            meas_bond_corr = 1, meas_energy_corr = 1,
            meas_chiral=1, meas_thermal=1, meas_2bond_corr=1,
            trans_sym =1, Nfiles=3, tp = tp,
            seed=seed, overwrite=1, n_sweep_warm=20, n_sweep_meas=20
        )



def main():
    gen_ref()

if __name__ == "__main__":
    main()
