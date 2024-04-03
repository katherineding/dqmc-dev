import gen_1band_unified_hub as ghub
import os
import pytest
import subprocess
from glob import glob


src = os.environ["DEV"]

geometry_list = ["square"]  # only profile square lattice for now
nflux_list = [0, 3]  # real and complex case
size_list = [4, 6, 8]
n_sweep_meas_list = [20, 40]
tps_list = ["0", "-0.25"]

seed = 1234

# measurement toggles
meas_bond_corr = 1
meas_energy_corr = 1
meas_thermal = 1
meas_2bond_corr = 1
meas_chiral = 1


# valid options for folder names
branches_list = ["master", "perlmt-gpu"]
hosts_list = ["sherlock", "home", "laptop", "perlmt"]
compilers_list = ["icx", "icc", "gcc", "nvc"]
libs_list = ["imkl", "openblas", "libsci"]


@pytest.fixture(scope="function", autouse=True)
def clean():
    """Clean up .h5, .h5.params, stack files in CWD, but leave logs"""
    yield
    # print("cleaning up per function: ... ")
    files = glob("*.h5")
    for f in files:
        os.remove(f)

    files = glob("*.h5.params")
    for f in files:
        os.remove(f)

    stacks = glob("stack")
    for s in stacks:
        os.remove(s)


def show_filesize():
    pass


def run_dqmc(complex: bool = False):
    """Push .h5 files in CWD to file called stack,
    then run dqmc_stack_r or dqmc_stack_c. This of couse generates
    corresponding log files.

    Args:
        type (str): [description] (default: `"r"`)
    """

    os.system(f"python3 {src}util/push.py stack *.h5")

    print("Run DQMC against test files:")
    os.system("cat stack")

    dqmc_exec = "dqmc_stack_c" if complex else "dqmc_stack_r"
    process = subprocess.Popen(
        [
            src + f"build/{dqmc_exec}",
            "stack",
        ],
        stdout=subprocess.PIPE,
    )
    process.wait()
    # for line in process.stdout:
    #     print(line)


@pytest.mark.parametrize("geometry", geometry_list)
@pytest.mark.parametrize("nflux", nflux_list)
@pytest.mark.parametrize("tps", tps_list)
@pytest.mark.parametrize("size", size_list)
@pytest.mark.parametrize("n_sweep_meas", n_sweep_meas_list)
def test_get_logs(geometry, nflux, tps, size, n_sweep_meas):
    """Check against a known good state"""
    # TODO: other geometries
    meas_toggles = f"{meas_bond_corr}{meas_energy_corr}{meas_thermal}{meas_2bond_corr}{meas_chiral}"
    print("measurement toggles:", meas_toggles)
    prefix = f"{ghub.hash_short}_{geometry}_{size}x{size}_nflux{nflux}_tp{tps}_nmeas{n_sweep_meas}_{meas_toggles}"
    ghub.create_batch(
        geometry=geometry,
        Nx=size,
        Ny=size,
        seed=seed,
        prefix=prefix,
        overwrite=1,
        n_sweep_warm=20,
        n_sweep_meas=n_sweep_meas,
        period_uneqlt=2,
        meas_bond_corr=meas_bond_corr,
        meas_energy_corr=meas_energy_corr,
        meas_thermal=meas_thermal,
        meas_2bond_corr=meas_2bond_corr,
        meas_chiral=meas_chiral,
        nflux=nflux,
        Nfiles=3,
        tp=float(tps),
    )

    size = os.path.getsize(f"{prefix}_0.h5") * 1e-6
    sizestr = f"size of one .h5 file = {size:.3g} MB\n"

    run_dqmc(complex=nflux)

    # Add extra information to log
    logs = glob(f"{prefix}*.log")
    for l in logs:
        with open(l, "a") as openfile:
            openfile.write(sizestr)
