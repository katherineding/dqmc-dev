import gen_1band_unified_hub as ghub
import gen_util_shared as gus
import util
import numpy as np
import os
import pytest
import subprocess
from glob import glob

# import matplotlib.pyplot as plt
import hs_ref

src = os.environ["DEV"]

geometry_list = ["square", "triangular", "honeycomb", "kagome"]
nflux_list = [0, 3]
seed = 1234

# This version contains bug in vv and vn equal time energy correlator.
# Bug is fixed in commit d0c3425
refhash = "c91ba61"

# This version has bond-dependent uneqlt measurements for square lattice
# and chiral eqlt measurements for square and triangular
# Known bugs: honeycomb and kagome bps, num_b, b2ps and num_b2 are just placeholders
refhash2 = "b535e68"

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


def run_dqmc(complex: bool = False):
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


@pytest.mark.parametrize("geometry", geometry_list)
def test_1_default(geometry):
    ghub.create_1(geometry=geometry)
    assert os.path.isfile("sim.h5")
    Nx, Ny = util.load_file("sim.h5", "metadata/Nx", "metadata/Ny")

    assert Nx == 4
    assert Nx.dtype.type == np.int64
    assert Nx.shape == ()
    assert np.ndim(Nx) == 0
    assert Ny == Nx == 4


@pytest.mark.parametrize("geometry", geometry_list)
def test_batch_default(geometry):
    ghub.create_batch(geometry=geometry)
    assert os.path.isfile("sim_0.h5")
    assert os.path.isfile("sim.h5.params")
    Nx, Ny = util.load_file("sim_0.h5", "metadata/Nx", "metadata/Ny")

    assert Nx == 4
    assert Nx.dtype.type == np.int64
    assert Nx.shape == ()
    assert np.ndim(Nx) == 0
    assert Ny == Nx == 4


# @pytest.mark.parametrize("geometry", geometry_list)
# def test_1_seed(geometry):
#     ref = ghub.rand_seed_splitmix64(1234)
#     ghub.create_1(geometry=geometry,init_rng=ref,overwrite=1)
#     assert os.path.isfile('sim.h5')
#     init_rng,rng = util.load_file('sim.h5','state/init_rng','state/rng')
#     print(init_rng)
#     print(rng)
#     os.remove("sim.h5")


@pytest.mark.parametrize("geometry", geometry_list)
def test_batch_seed_basic_real(geometry):
    ghub.create_batch(
        geometry=geometry, seed=seed, overwrite=1, n_sweep_warm=20, n_sweep_meas=20
    )

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
    ref_rng = gus.rand_seed_splitmix64(seed)
    assert np.allclose(init_rng, ref_rng)
    assert np.allclose(hs, hs_ref.hs_init[geometry])

    run_dqmc(complex=False)

    hs_end = util.load_file("sim_0.h5", "state/hs")[0]
    assert np.allclose(hs_end, hs_ref.hs_end[geometry])


def compare_meta(path1, path2):
    meta_ref = util.load(
        path1,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/bps",
        "metadata/U",
        "metadata/t'",
        "metadata/nflux",
        "metadata/beta",
    )

    meta_c = util.load(
        path2,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/bps",
        "metadata/U",
        "metadata/t'",
        "metadata/nflux",
        "metadata/beta",
    )

    for i in range(len(meta_ref)):
        assert np.allclose(meta_ref[i], meta_c[i])


def compare_meta_extra(path1, path2):
    meta_ref = util.load(
        path1,
        "metadata/Norb",
        "metadata/b2ps",
        "metadata/plaq_per_cell",
        "metadata/t''",
        "metadata/twistx",
        "metadata/twisty",
        "metadata/mu",
        "metadata/h",
        "metadata/trans_sym",
    )

    meta_c = util.load(
        path2,
        "metadata/Norb",
        "metadata/b2ps",
        "metadata/plaq_per_cell",
        "metadata/t''",
        "metadata/twistx",
        "metadata/twisty",
        "metadata/mu",
        "metadata/h",
        "metadata/trans_sym",
    )

    for i in range(len(meta_ref)):
        assert np.allclose(meta_ref[i], meta_c[i])

    # np.allclose can't be used to compare an array of (byte) strings
    g_ref = util.load_firstfile(
        path1,
        "metadata/geometry",
    )[0]

    g_c = util.load_firstfile(
        path2,
        "metadata/geometry",
    )[0]

    assert g_ref == g_c


def compare_state(path1, path2):
    state_ref = util.load(
        path1, "state/sweep", "state/init_rng", "state/rng", "state/hs"
    )

    state_c = util.load(path2, "state/sweep", "state/init_rng", "state/rng", "state/hs")

    for i in range(len(state_ref)):
        assert np.allclose(state_ref[i], state_c[i])


def compare_eqlt_meas(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_eqlt/n_sample",
        "meas_eqlt/sign",
        "meas_eqlt/density",
        "meas_eqlt/double_occ",
        "meas_eqlt/g00",
        "meas_eqlt/nn",
        "meas_eqlt/xx",
        "meas_eqlt/zz",
        "meas_eqlt/pair_sw",
    )

    meas_c = util.load(
        path2,
        "meas_eqlt/n_sample",
        "meas_eqlt/sign",
        "meas_eqlt/density",
        "meas_eqlt/double_occ",
        "meas_eqlt/g00",
        "meas_eqlt/nn",
        "meas_eqlt/xx",
        "meas_eqlt/zz",
        "meas_eqlt/pair_sw",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_eqlt_meas_extra(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_eqlt/density_u",
        "meas_eqlt/density_d",
        "meas_eqlt/g00_u",
        "meas_eqlt/g00_d",
    )

    meas_c = util.load(
        path2,
        "meas_eqlt/density_u",
        "meas_eqlt/density_d",
        "meas_eqlt/g00_u",
        "meas_eqlt/g00_d",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_eqlt_meas_chi(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_eqlt/chi",
    )

    meas_c = util.load(
        path2,
        "meas_eqlt/chi",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_eqlt_energy_meas(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_eqlt/kk",
        "meas_eqlt/kv",
        "meas_eqlt/kn",
        "meas_eqlt/vv",
        "meas_eqlt/vn",
    )

    meas_c = util.load(
        path2,
        "meas_eqlt/kk",
        "meas_eqlt/kv",
        "meas_eqlt/kn",
        "meas_eqlt/vv",
        "meas_eqlt/vn",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_uneqlt_meas(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_uneqlt/n_sample",
        "meas_uneqlt/sign",
        "meas_uneqlt/gt0",
        "meas_uneqlt/nn",
        "meas_uneqlt/xx",
        "meas_uneqlt/zz",
        "meas_uneqlt/pair_sw",
    )

    meas_c = util.load(
        path2,
        "meas_uneqlt/n_sample",
        "meas_uneqlt/sign",
        "meas_uneqlt/gt0",
        "meas_uneqlt/nn",
        "meas_uneqlt/xx",
        "meas_uneqlt/zz",
        "meas_uneqlt/pair_sw",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_uneqlt_meas_extra(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_uneqlt/gt0_u",
        "meas_uneqlt/gt0_d",
    )

    meas_c = util.load(
        path2,
        "meas_uneqlt/gt0_u",
        "meas_uneqlt/gt0_d",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_uneqlt_bond_meas(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_uneqlt/pair_bb",
        "meas_uneqlt/jj",
        "meas_uneqlt/jsjs",
        "meas_uneqlt/kk",
        "meas_uneqlt/ksks",
    )

    meas_c = util.load(
        path2,
        "meas_uneqlt/pair_bb",
        "meas_uneqlt/jj",
        "meas_uneqlt/jsjs",
        "meas_uneqlt/kk",
        "meas_uneqlt/ksks",
    )

    for i in range(len(meas_ref)):
        # print(i)
        # print("norm:",np.linalg.norm(meas_ref[i]))
        # print("current:",meas_c[i])
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_uneqlt_thermal_meas(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_uneqlt/j2jn",
        "meas_uneqlt/jnj2",
        "meas_uneqlt/jnjn",
        "meas_uneqlt/jjn",
        "meas_uneqlt/jnj",
    )

    meas_c = util.load(
        path2,
        "meas_uneqlt/j2jn",
        "meas_uneqlt/jnj2",
        "meas_uneqlt/jnjn",
        "meas_uneqlt/jjn",
        "meas_uneqlt/jnj",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_uneqlt_2bond_meas(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_uneqlt/j2j2",
        "meas_uneqlt/j2j",
        "meas_uneqlt/jj2",
        "meas_uneqlt/pair_b2b2",
        "meas_uneqlt/js2js2",
        "meas_uneqlt/k2k2",
        "meas_uneqlt/ks2ks2",
    )

    meas_c = util.load(
        path2,
        "meas_uneqlt/j2j2",
        "meas_uneqlt/j2j",
        "meas_uneqlt/jj2",
        "meas_uneqlt/pair_b2b2",
        "meas_uneqlt/js2js2",
        "meas_uneqlt/k2k2",
        "meas_uneqlt/ks2ks2",
    )

    for i in range(len(meas_ref)):
        # TODO: is it possible to make atol tighter?
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])  # ,atol=1e-7)


def compare_uneqlt_energy_meas(path1, path2):
    meas_ref = util.load(
        path1,
        "meas_uneqlt/kv",
        "meas_uneqlt/kn",
        "meas_uneqlt/vv",
        "meas_uneqlt/vn",
    )

    meas_c = util.load(
        path2,
        "meas_uneqlt/kv",
        "meas_uneqlt/kn",
        "meas_uneqlt/vv",
        "meas_uneqlt/vn",
    )

    for i in range(len(meas_ref)):
        print("max diff:", np.abs(np.max(meas_ref[i] - meas_c[i])))
        assert np.allclose(meas_ref[i], meas_c[i])


def compare_params(path1, path2):
    params_ref = util.load(
        path1,
        "params/N",
        "params/L",
        "params/map_i",
        "params/map_ij",
        "params/bonds",
        "params/map_bs",
        "params/map_bb",
        "params/peierlsu",
        "params/peierlsd",
        "params/Ku",
        "params/Kd",
        "params/U",
        "params/dt",
        "params/n_matmul",
        "params/n_delay",
        "params/n_sweep_warm",
        "params/n_sweep_meas",
        "params/period_eqlt",
        "params/period_uneqlt",
        "params/meas_bond_corr",
        "params/meas_energy_corr",
        "params/meas_nematic_corr",
        "params/num_i",
        "params/num_ij",
        "params/num_b",
        "params/num_bs",
        "params/num_bb",
        "params/exp_lambda",
        "params/del",
        "params/F",
    )

    params_c = util.load(
        path2,
        "params/N",
        "params/L",
        "params/map_i",
        "params/map_ij",
        "params/bonds",
        "params/map_bs",
        "params/map_bb",
        "params/peierlsu",
        "params/peierlsd",
        "params/Ku",
        "params/Kd",
        "params/U",
        "params/dt",
        "params/n_matmul",
        "params/n_delay",
        "params/n_sweep_warm",
        "params/n_sweep_meas",
        "params/period_eqlt",
        "params/period_uneqlt",
        "params/meas_bond_corr",
        "params/meas_energy_corr",
        "params/meas_nematic_corr",
        "params/num_i",
        "params/num_ij",
        "params/num_b",
        "params/num_bs",
        "params/num_bb",
        "params/exp_lambda",
        "params/del",
        "params/F",
    )

    for i in range(len(params_ref)):
        assert np.allclose(params_ref[i], params_c[i])


def compare_params_extra(path1, path2):
    params_ref = util.load(
        path1,
        "params/bond2s",
        "params/map_plaq",
        "params/plaqs",
        "params/map_b2b",
        "params/map_bb2",
        "params/map_b2b2",
        "params/pp_u",
        "params/pp_d",
        "params/ppr_u",
        "params/ppr_d",
        "params/meas_thermal",
        "params/meas_2bond_corr",
        "params/meas_chiral",
        "params/num_plaq",
        "params/num_plaq_accum",
        "params/num_b2",
        "params/num_b2b",
        "params/num_bb2",
        "params/num_b2b2",
    )

    params_c = util.load(
        path2,
        "params/bond2s",
        "params/map_plaq",
        "params/plaqs",
        "params/map_b2b",
        "params/map_bb2",
        "params/map_b2b2",
        "params/pp_u",
        "params/pp_d",
        "params/ppr_u",
        "params/ppr_d",
        "params/meas_thermal",
        "params/meas_2bond_corr",
        "params/meas_chiral",
        "params/num_plaq",
        "params/num_plaq_accum",
        "params/num_b2",
        "params/num_b2b",
        "params/num_bb2",
        "params/num_b2b2",
    )

    for i in range(len(params_ref)):
        assert np.allclose(params_ref[i], params_c[i])


def compare_params_degen(path1, path2):
    """Account for degen shape change introduced in ee61ee4"""
    params_ref = util.load(
        path1,
        "params/degen_i",
        "params/degen_ij",
        "params/degen_bs",
        "params/degen_bb",
    )

    params_c = util.load(
        path2,
        "params/degen_i",
        "params/degen_ij",
        "params/degen_bs",
        "params/degen_bb",
    )

    for i in range(len(params_ref)):
        assert np.allclose(params_ref[i][:, 0], params_c[i])


def compare_params_degen_extra(path1, path2):
    """Account for degen shape change introduced in ee61ee4"""
    params_ref = util.load(
        path1,
        "params/degen_plaq",
        "params/degen_bb2",
        "params/degen_b2b",
        "params/degen_b2b2",
    )

    params_c = util.load(
        path2,
        "params/degen_plaq",
        "params/degen_bb2",
        "params/degen_b2b",
        "params/degen_b2b2",
    )

    for i in range(len(params_ref)):
        assert np.allclose(params_ref[i][:, 0], params_c[i])


@pytest.mark.parametrize("nflux", nflux_list)
def test_square_ref(nflux):
    """Check against a known good state"""
    # TODO: other geometries
    geometry = "square"
    ghub.create_batch(
        geometry=geometry,
        seed=seed,
        prefix=f"{ghub.hash_short}_{geometry}_{seed}",
        overwrite=1,
        n_sweep_warm=20,
        n_sweep_meas=20,
        period_uneqlt=2,
        meas_bond_corr=1,
        meas_energy_corr=1,
        nflux=nflux,
        Nfiles=3,
    )

    run_dqmc(nflux)

    # ========== metadata, params, state ======================
    refpath = src + f"test/ref/{refhash}/{geometry}_nflux{nflux}/"
    print("reference commit:", refhash)
    compare_state(refpath, "")
    compare_meta(refpath, "")
    compare_params(refpath, "")
    compare_params_degen(refpath, "")

    # ========== measurements ======================
    compare_eqlt_meas(refpath, "")
    compare_uneqlt_meas(refpath, "")
    compare_uneqlt_bond_meas(refpath, "")
    compare_uneqlt_energy_meas(refpath, "")

    # I don't care about nematic correlators so I'm not checking them

    if nflux == 3:
        pytest.xfail("known bug in complex vv and vn equal time correlator")
        compare_eqlt_energy_meas(refpath, "")


@pytest.mark.parametrize("geometry", geometry_list)
@pytest.mark.parametrize("nflux", nflux_list)
def test_ref(geometry, nflux):
    """Check against a known good state"""
    if geometry == "square":
        tp = -0.25
    else:
        tp = 0
    ghub.create_batch(
        geometry=geometry,
        seed=seed,
        prefix=f"{ghub.hash_short}_{geometry}_{seed}_nflux{nflux}",
        overwrite=1,
        n_sweep_warm=20,
        n_sweep_meas=20,
        period_uneqlt=2,
        meas_bond_corr=1,
        meas_energy_corr=1,
        meas_thermal=1,
        meas_2bond_corr=1,
        meas_chiral=1,
        nflux=nflux,
        tp=tp,
        Nfiles=3,
    )

    run_dqmc(complex=nflux)

    # ========== metadata, params ======================
    refpath = src + f"test/ref/{refhash2}/{geometry}_nflux{nflux}/"
    print("reference commit:", refhash2)
    compare_state(refpath, "")
    compare_meta(refpath, "")
    compare_meta_extra(refpath, "")
    compare_params(refpath, "")
    compare_params_extra(refpath, "")
    compare_params_degen(refpath, "")
    compare_params_degen_extra(refpath, "")

    # ============ measurements ========================
    compare_eqlt_meas(refpath, "")
    compare_eqlt_meas_extra(refpath, "")
    compare_eqlt_meas_chi(refpath, "")
    compare_uneqlt_meas(refpath, "")
    compare_uneqlt_meas_extra(refpath, "")

    # Only square lattice has correct bond correlator definitions
    if geometry == "square":
        compare_uneqlt_bond_meas(refpath, "")
        compare_uneqlt_2bond_meas(refpath, "")
        compare_uneqlt_thermal_meas(refpath, "")
        compare_uneqlt_energy_meas(refpath, "")
        compare_eqlt_energy_meas(refpath, "")

    # I don't care about nematic correlators so I'm not checking them


@pytest.mark.skip(reason="need to find good reference for this")
@pytest.mark.parametrize("geometry", geometry_list)
def test_batch_chiral_B0(geometry):
    nflux = 0
    ghub.create_batch(
        geometry=geometry,
        seed=seed,
        overwrite=1,
        n_sweep_warm=20,
        n_sweep_meas=20,
        meas_chiral=1,
        nflux=nflux,
    )

    run_dqmc(complex=nflux)

    chi = util.load("", "meas_eqlt/chi")[0]
    assert np.allclose(chi, 0)


@pytest.mark.skip(reason="need to find good reference for this")
@pytest.mark.parametrize("geometry", geometry_list)
def test_batch_chiral(geometry):
    nflux = 3
    ghub.create_batch(
        geometry=geometry,
        seed=seed,
        overwrite=1,
        n_sweep_warm=20,
        n_sweep_meas=20,
        meas_chiral=1,
        nflux=nflux,
    )

    run_dqmc(complex=nflux)

    chi = util.load("", "meas_eqlt/chi")[0]

    assert np.allclose(chi, 0)
