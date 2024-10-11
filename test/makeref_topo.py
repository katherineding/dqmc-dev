import numpy as np
import os, sys

src = os.environ["DEV"]
if src + "util/" not in sys.path:
    sys.path.insert(0, src + "util/")
import gen_topo_3band_hub as ghub
import util


# import matplotlib.pyplot as plt


seed = 1234


def gen_ref():
    ghub.create_batch(
        prefix=f"{ghub.hash_short}_{seed}",
        Nx=4,
        Ny=4,
        dt=0.1,
        L=40,
        period_uneqlt=2,
        meas_bond_corr=1,
        meas_energy_corr=1,
        meas_chiral=1,
        meas_thermal=1,
        meas_2bond_corr=1,
        trans_sym=1,
        Nfiles=3,
        tsp=1,
        seed=seed,
        overwrite=1,
        n_sweep_warm=20,
        n_sweep_meas=20,
    )


def main():
    gen_ref()


if __name__ == "__main__":
    main()
