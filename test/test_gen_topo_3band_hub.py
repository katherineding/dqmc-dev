import gen_topo_3band_hub as ghub
import h5py
import util
import numpy as np
import os
import pytest
import subprocess
from glob import glob

# import matplotlib.pyplot as plt
import hs_ref

src = os.environ["DEV"]
seed = 1234

# TODO: update this
refhash = "eafbd47"

# @pytest.fixture(scope="session",autouse=True)
# def setup():
#     print("this is setup")
#     # compile DQMC with proper names
#     # Make sure reference DQMC data files are present
#     yield


@pytest.fixture(scope="function", autouse=True)
def clean():
    """Clean up .h5, .h5.params, .h5.files in CWD
    [description]
    """
    yield
    # print("cleaning up per function: ... ")
    files = glob("*.h5*")
    for f in files:
        os.remove(f)

    stacks = glob("stack")
    for s in stacks:
        os.remove(s)


def run_dqmc():
    """Push .h5 files in CWD to file called stack,
    then run dqmc_stack_r or dqmc_stack_c. This of couse generates
    corresponding log files.

    [description]

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


def test_1_default():
    ghub.create_1()
    assert os.path.isfile("sim.h5")
    Nx, Ny = util.load_file("sim.h5", "metadata/Nx", "metadata/Ny")

    assert Nx == 4
    assert Nx.dtype.type == np.int64
    assert Nx.shape == ()
    assert np.ndim(Nx) == 0
    assert Ny == Nx == 4


def test_batch_default():
    ghub.create_batch()
    assert os.path.isfile("sim_0.h5")
    assert os.path.isfile("sim.h5.params")
    Nx, Ny = util.load_file("sim_0.h5", "metadata/Nx", "metadata/Ny")

    assert Nx == 4
    assert Nx.dtype.type == np.int64
    assert Nx.shape == ()
    assert np.ndim(Nx) == 0
    assert Ny == Nx == 4


def test_batch_seed_basic_real():
    ghub.create_batch(seed=seed, overwrite=1, n_sweep_warm=20, n_sweep_meas=20)

    assert os.path.isfile("sim_0.h5")
    assert os.path.isfile("sim.h5.params")
    Nx, Ny, Norb, L = util.load_file(
        "sim_0.h5", "metadata/Nx", "metadata/Ny", "metadata/Norb", "params/L"
    )
    init_rng, rng, hs = util.load_file(
        "sim_0.h5", "state/init_rng", "state/rng", "state/hs"
    )

    # shape and dtype
    assert init_rng.shape == rng.shape == (17,)
    assert init_rng.dtype.type == rng.dtype.type == np.uint64
    assert hs.shape == (L, Nx * Ny * Norb)
    assert hs.dtype.type == np.int32

    # initial rng based on seed
    ref_rng = ghub.rand_seed_splitmix64(seed)
    assert np.allclose(init_rng, ref_rng)

    run_dqmc()

    hs_end = util.load_file("sim_0.h5", "state/hs")[0]


def compare_meta(path1, path2):
    meta_ref = util.load(
        path1,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/Norb",
        "metadata/bps",
        "metadata/b2ps",
        "metadata/plaq_per_cell",
        "metadata/U",
        "metadata/t",
        "metadata/tsp",
        "metadata/lam",
        "metadata/g",
        "metadata/mu",
        "metadata/beta",
        "metadata/trans_sym",
    )

    meta_c = util.load(
        path2,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/Norb",
        "metadata/bps",
        "metadata/b2ps",
        "metadata/plaq_per_cell",
        "metadata/U",
        "metadata/t",
        "metadata/tsp",
        "metadata/lam",
        "metadata/g",
        "metadata/mu",
        "metadata/beta",
        "metadata/trans_sym",
    )

    for i in range(len(meta_ref)):
        assert np.allclose(meta_ref[i], meta_c[i])


def compare_state(path1, path2):
    state_ref = util.load(
        path1, "state/sweep", "state/init_rng", "state/rng", "state/hs"
    )

    state_c = util.load(path2, "state/sweep", "state/init_rng", "state/rng", "state/hs")

    for i in range(len(state_ref)):
        assert np.allclose(state_ref[i], state_c[i])


def compare_params(path1, path2):
    with h5py.File(path1 + f"{refhash}_{seed}_1.h5", "r") as f:
        param_names = list(f["params"].keys())
        for name in param_names:
            params_c = util.load(path2, f"params/{name}")
            params_ref = util.load(path1, f"params/{name}")
            assert np.allclose(params_c[0], params_ref[0])


def compare_eqlt_meas(path1, path2):
    with h5py.File(path1 + f"{refhash}_{seed}_1.h5", "r") as f:
        param_names = list(f["meas_eqlt"].keys())
        for name in param_names:
            print(name)
            meas_c = util.load(path2, f"meas_eqlt/{name}")
            meas_ref = util.load(path1, f"meas_eqlt/{name}")

            print("max diff:", np.abs(np.max(meas_ref[0] - meas_c[0])))
            assert np.allclose(meas_c[0], meas_ref[0], equal_nan=True)


def compare_uneqlt_meas(path1, path2):
    with h5py.File(path1 + f"{refhash}_{seed}_1.h5", "r") as f:
        param_names = list(f["meas_uneqlt"].keys())
        for name in param_names:
            print(name)
            meas_c = util.load(path2, f"meas_uneqlt/{name}")
            meas_ref = util.load(path1, f"meas_uneqlt/{name}")

            print("max diff:", np.abs(np.max(meas_ref[0] - meas_c[0])))
            assert np.allclose(meas_c[0], meas_ref[0], equal_nan=True)


def test_ref():
    """Check against a known good state"""
    # TODO: other geometries
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

    run_dqmc()

    # ========== metadata, params, state ======================
    refpath = src + f"test/ref/{refhash}/"
    print("reference commit:", refhash)
    compare_state(refpath, "")
    compare_meta(refpath, "")
    compare_params(refpath, "")
    print("state, meta, params OK!")
    # ========== measurements ======================
    compare_eqlt_meas(refpath, "")
    compare_uneqlt_meas(refpath, "")


# @pytest.mark.skip(reason="need to find good reference for this")
def test_batch_chiral():
    ghub.create_batch(
        seed=seed,
        overwrite=1,
        n_sweep_warm=20,
        n_sweep_meas=20,
        meas_chiral=1,
    )

    run_dqmc()

    chi = util.load("", "meas_eqlt/chi")[0]

    assert np.allclose(chi, 0)
