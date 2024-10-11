import os
import shutil
import sys
import argparse
import warnings

import h5py
import numpy as np
from scipy.linalg import expm

import tight_binding

np.seterr(over="ignore")
np.set_printoptions(precision=3)

import git  # relies on gitpython module

path = os.path.dirname(os.path.abspath(__file__))
repo = git.Repo(path, search_parent_directories=True)
hash_short = repo.git.rev_parse(repo.head, short=True)


def rand_seed_urandom():
    rng = np.zeros(17, dtype=np.uint64)
    rng[:16] = np.frombuffer(os.urandom(16 * 8), dtype=np.uint64)
    return rng


# http://xoroshiro.di.unimi.it/splitmix64.c
def rand_seed_splitmix64(x):
    x = np.uint64(x)
    rng = np.zeros(17, dtype=np.uint64)
    for i in range(16):
        x += np.uint64(0x9E3779B97F4A7C15)
        z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        rng[i] = z ^ (z >> np.uint64(31))
    return rng


# http://xoroshiro.di.unimi.it/xorshift1024star.c
def rand_uint(rng):
    s0 = rng[rng[16]]
    p = (int(rng[16]) + 1) & 15
    rng[16] = p
    s1 = rng[p]
    s1 ^= s1 << np.uint64(31)
    rng[p] = s1 ^ s0 ^ (s1 >> np.uint64(11)) ^ (s0 >> np.uint64(30))
    return rng[p] * np.uint64(1181783497276652981)


def rand_jump(rng):
    JMP = np.array(
        (
            0x84242F96ECA9C41D,
            0xA3C65B8776F96855,
            0x5B34A39F070B5837,
            0x4489AFFCE4F31A1E,
            0x2FFEEB0A48316F40,
            0xDC2D9891FE68C022,
            0x3659132BB12FEA70,
            0xAAC17D8EFA43CAB8,
            0xC4CB815590989B13,
            0x5EE975283D71C93B,
            0x691548C86C1BD540,
            0x7910C41D10A1E6A5,
            0x0B5FC64563B3E2A8,
            0x047F7684E9FC949D,
            0xB99181F2D8F685CA,
            0x284600E3F30E38C3,
        ),
        dtype=np.uint64,
    )

    t = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        for b in range(64):
            if JMP[i] & (np.uint64(1) << np.uint64(b)):
                for j in range(16):
                    t[j] ^= rng[(np.uint64(j) + rng[16]) & np.uint64(15)]
            rand_uint(rng)

    for j in range(16):
        rng[(np.uint64(j) + rng[16]) & np.uint64(15)] = t[j]


def plaquette_params(
    Nx: int, Ny: int, trans_sym: int
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Geometry specific plaquette parameters. TODO: honeycomb
    Returns:
        (plaq_per_cell, num_plaq_total, num_plaq_accum), map_plaq, degen_plaq, plaqs
    """
    Ncell = Nx * Ny
    plaq_per_cell = 1
    num_plaq_total = plaq_per_cell * Ncell

    # per plaquette (per CELL) measurement mapping
    if trans_sym:
        map_plaq = np.zeros(num_plaq_total, dtype=np.int32)
        # translation average for each plaquette type
        for p in range(plaq_per_cell):
            map_plaq[Ncell * p : Ncell * (p + 1)] = p
        degen_plaq = np.full(plaq_per_cell, Ncell, dtype=np.int32)
    else:
        map_plaq = np.arange(num_plaq_total, dtype=np.int32)
        degen_plaq = np.ones(num_plaq_total, dtype=np.int32)

    num_plaq_accum = map_plaq.max() + 1
    assert num_plaq_accum == degen_plaq.size
    assert np.all(degen_plaq == degen_plaq[0])

    # plaquette definitions
    plaqs = np.zeros((3, num_plaq_total), dtype=np.int32)  # NOTE: placeholder

    return (plaq_per_cell, num_plaq_total, num_plaq_accum), map_plaq, degen_plaq, plaqs


def bond_params(
    Nx: int, Ny: int, t: float, trans_sym: int
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """1-bond parameters
    Returns:
        (bps, num_b, num_b_accum), map_b, degen_b, bonds
    """
    Norb = 3
    N = Nx * Ny * Norb

    # 1-bonds per site
    bps = 4 if t != 0.0 else 2  # OK
    num_b = bps * N  # total bonds in cluster

    # per 1-bond (per SITE) measurement mapping
    if trans_sym:
        map_b = np.tile(np.arange(bps, dtype=np.int32), (N, 1)).T.flatten()  # N*bps
        degen_b = np.full(bps, N, dtype=np.int32)
    else:
        map_b = np.arange(num_b, dtype=np.int32)  # N*bps
        degen_b = np.ones(num_b, dtype=np.int32)  # length N*bps
    num_b_accum = map_b.max() + 1
    assert num_b_accum == degen_b.size
    assert np.all(degen_b == degen_b[0])

    # bond definitions: defined by one hopping step
    bonds = np.zeros((2, num_b), dtype=np.int32)  # NOTE: placeholder

    return (bps, num_b, num_b_accum), map_b, degen_b, bonds


def bond2_params(
    Nx: int, Ny: int, t: float, trans_sym: int
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """2-bond parameters
    Returns:
        (b2ps, num_b2, num_b2_accum), map_b2, degen_b2, bond2s
    """
    Norb = 3
    N = Nx * Ny * Norb

    # 1-bonds per site
    b2ps = 4  # NOTE: Placeholder
    num_b2 = b2ps * N  # total 2-bonds in cluster

    # per 2-bond (per SITE) measurement mapping
    if trans_sym:
        map_b2 = np.tile(
            np.arange(b2ps, dtype=np.int32), (N, 1)
        ).T.flatten()  # length N*b2ps
        degen_b2 = np.full(b2ps, N, dtype=np.int32)
    else:
        map_b2 = np.arange(num_b2, dtype=np.int32)  # length N*b2ps
        degen_b2 = np.ones(num_b2, dtype=np.int32)  # length N*b2ps
    num_b2_accum = map_b2.max() + 1
    assert num_b2_accum == degen_b2.size
    assert np.all(degen_b2 == degen_b2[0])

    bond2s = np.zeros((2, num_b2), dtype=np.int32)  # NOTE: placeholder

    return (
        (b2ps, num_b2, num_b2_accum),
        map_b2,
        degen_b2,
        bond2s,
    )


def create_1(
    file_sim=None,
    file_params=None,
    init_rng=None,
    Nx: int = 4,
    Ny: int = 4,
    mu: float = 0.0,
    t: float = 0.0,
    tsp: float = 1.0,
    lam: float = 1.0,
    g: float = 0.5,
    U: float = 6.0,
    dt: float = 0.1,
    L: int = 40,
    overwrite=0,
    n_delay: int = 16,
    n_matmul: int = 8,
    n_sweep_warm: int = 200,
    n_sweep_meas: int = 2000,
    period_eqlt: int = 8,
    period_uneqlt: int = 0,
    trans_sym: int = 1,
    checkpoint_every: int = 10000,
    meas_bond_corr: int = 0,
    meas_energy_corr: int = 0,
    meas_nematic_corr: int = 0,
    meas_thermal: int = 0,
    meas_2bond_corr: int = 0,
    meas_chiral: int = 0,
    meas_local_JQ: int = 0,
):
    assert L % n_matmul == 0 and L % period_eqlt == 0
    dtype_num = np.complex128

    if file_sim is None:
        file_sim = "sim.h5"
    if file_params is None:
        file_params = file_sim

    one_file = os.path.abspath(file_sim) == os.path.abspath(file_params)

    if init_rng is None:
        init_rng = rand_seed_urandom()
    rng = init_rng.copy()

    Ncell = Nx * Ny
    Norb = 3
    N = Norb * Ncell

    init_hs = np.zeros((L, N), dtype=np.int32)
    for l in range(L):
        for i in range(N):
            init_hs[l, i] = rand_uint(rng) >> np.uint64(63)

    # ------------------measurement maps----------------------
    # per Norb (per site) measurement mapping
    if trans_sym:
        map_i = np.zeros(N, dtype=np.int32)
        # translation average for each orbital
        for orb in range(Norb):
            map_i[Ncell * orb : Ncell * (orb + 1)] = orb
        degen_i = np.full(Norb, Ncell, dtype=np.int32)
    else:
        map_i = np.arange(N, dtype=np.int32)
        degen_i = np.ones(N, dtype=np.int32)
    num_i = map_i.max() + 1
    assert num_i == degen_i.size
    assert np.all(degen_i == degen_i[0])

    # 2 site mapping
    map_ij = np.zeros((N, N), dtype=np.int32)
    num_ij = Norb * N if trans_sym else N * N
    degen_ij = np.zeros(num_ij, dtype=np.int32)
    for jo in range(Norb):
        for jy in range(Ny):
            for jx in range(Nx):
                for io in range(Norb):
                    for iy in range(Ny):
                        for ix in range(Nx):
                            if trans_sym:
                                ky = (iy - jy) % Ny
                                kx = (ix - jx) % Nx
                                # total column index of matrix index [kx,ky,io,jo]
                                k = kx + Nx * ky + Nx * Ny * io + N * jo
                            else:
                                # total column index of matrix index [ix,iy,io,jx,jy,jo]
                                k = (ix + Nx * iy + Nx * Ny * io) + N * (
                                    jx + Nx * jy + Nx * Ny * jo
                                )
                            map_ij[
                                jx + Nx * jy + Nx * Ny * jo,
                                ix + Nx * iy + Nx * Ny * io,
                            ] = k
                            degen_ij[k] += 1
    assert num_ij == map_ij.max() + 1 == degen_ij.size
    assert np.all(degen_ij == degen_ij[0])

    # ------------------------------------------------------
    # per plaquette (per site) measurement mapping
    (
        (plaq_per_cell, num_plaq_total, num_plaq_accum),
        map_plaq,
        degen_plaq,
        plaqs,
    ) = plaquette_params(Nx, Ny, trans_sym)

    # ------------------------------------------------------
    # per bond (per site) measurement mapping
    (
        (bps, num_b, num_b_accum),
        map_b,
        degen_b,
        bonds,
    ) = bond_params(Nx, Ny, t, trans_sym)

    # ------------------------------------------------------
    # per 2-bond (per site) measurement mapping
    (
        (b2ps, num_b2, num_b2_accum),
        map_b2,
        degen_b2,
        bond2s,
    ) = bond2_params(Nx, Ny, t, trans_sym)

    # 1 bond 1 site mapping NOTE: placeholder
    map_bs = np.zeros((N, num_b), dtype=np.int32)
    num_bs = bps * N if trans_sym else num_b * N
    degen_bs = np.zeros(num_bs, dtype=np.int32)

    # 1 bond - 1 bond mapping NOTE: placeholder
    map_bb = np.zeros((num_b, num_b), dtype=np.int32)
    num_bb = bps * bps * N if trans_sym else num_b * num_b
    degen_bb = np.zeros(num_bb, dtype=np.int32)

    # 2 2-bond mapping NOTE: placeholder
    num_b2b2 = b2ps * b2ps * N if trans_sym else num_b2 * num_b2
    map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
    degen_b2b2 = np.zeros(num_b2b2, dtype=np.int32)

    # bond 2-bond mapping NOTE: placeholder
    num_bb2 = bps * b2ps * N if trans_sym else num_b * num_b2
    map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
    degen_bb2 = np.zeros(num_bb2, dtype=np.int32)

    # 2-bond bond mapping NOTE: placeholder
    num_b2b = b2ps * bps * N if trans_sym else num_b2 * num_b
    map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
    degen_b2b = np.zeros(num_b2b, dtype=np.int32)

    Ku, peierls = tight_binding.H_periodic_3band(Nx, Ny, t, tsp, lam, g)

    # NOTE: placeholder
    thermal_phases = np.ones((b2ps, N), dtype=np.complex128)

    Kd = Ku.conj().transpose()
    for i in range(N):
        Ku[i, i] -= mu
        Kd[i, i] -= mu

    exp_Ku = expm(-dt * Ku)
    inv_exp_Ku = expm(dt * Ku)
    exp_halfKu = expm(-dt / 2 * Ku)
    inv_exp_halfKu = expm(dt / 2 * Ku)

    exp_Kd = expm(-dt * Kd)
    inv_exp_Kd = expm(dt * Kd)
    exp_halfKd = expm(-dt / 2 * Kd)
    inv_exp_halfKd = expm(dt / 2 * Kd)

    U_i = U * np.ones_like(degen_i, dtype=np.float64)
    assert U_i.shape[0] == num_i

    exp_lmbd = np.exp(0.5 * U_i * dt) + np.sqrt(np.expm1(U_i * dt))
    exp_lambda = np.array((exp_lmbd[map_i] ** -1, exp_lmbd[map_i]))
    delll = np.array((exp_lmbd[map_i] ** 2 - 1, exp_lmbd[map_i] ** -2 - 1))

    with h5py.File(file_params, "w" if overwrite else "x") as f:
        # parameters not used by dqmc code, but useful for analysis
        f.create_group("metadata")
        f["metadata"]["commit"] = hash_short
        f["metadata"]["version"] = 0.1
        f["metadata"]["model"] = "Topo 3-band (complex)"
        f["metadata"]["Nx"] = Nx
        f["metadata"]["Ny"] = Ny
        f["metadata"]["Norb"] = Norb
        f["metadata"]["bps"] = bps
        f["metadata"]["b2ps"] = b2ps
        f["metadata"]["plaq_per_cell"] = plaq_per_cell
        f["metadata"]["U"] = U
        f["metadata"]["t"] = t
        f["metadata"]["tsp"] = tsp
        f["metadata"]["lam"] = lam
        f["metadata"]["g"] = g
        f["metadata"]["mu"] = mu
        f["metadata"]["beta"] = L * dt
        f["metadata"]["trans_sym"] = trans_sym

        # parameters used by dqmc code
        f.create_group("params")
        # model parameters
        f["params"]["N"] = np.array(N, dtype=np.int32)
        f["params"]["L"] = np.array(L, dtype=np.int32)
        f["params"]["map_i"] = map_i
        f["params"]["map_ij"] = map_ij
        f["params"]["bonds"] = bonds
        f["params"]["bond2s"] = bond2s
        f["params"]["plaqs"] = plaqs
        f["params"]["map_plaq"] = map_plaq
        f["params"]["map_b"] = map_b
        f["params"]["map_bs"] = map_bs
        f["params"]["map_bb"] = map_bb
        f["params"]["map_b2"] = map_b2
        f["params"]["map_b2b"] = map_b2b
        f["params"]["map_bb2"] = map_bb2
        f["params"]["map_b2b2"] = map_b2b2
        f["params"]["peierlsu"] = peierls
        f["params"]["peierlsd"] = f["params"]["peierlsu"]
        f["params"]["pp_u"] = thermal_phases.conj()
        f["params"]["pp_d"] = thermal_phases.conj()
        f["params"]["ppr_u"] = thermal_phases
        f["params"]["ppr_d"] = thermal_phases
        f["params"]["Ku"] = Ku
        f["params"]["Kd"] = Kd
        f["params"]["U"] = U_i
        f["params"]["dt"] = np.array(dt, dtype=np.float64)  # not actually used

        # simulation parameters
        f["params"]["n_matmul"] = np.array(n_matmul, dtype=np.int32)
        f["params"]["n_delay"] = np.array(n_delay, dtype=np.int32)
        f["params"]["n_sweep_warm"] = np.array(n_sweep_warm, dtype=np.int32)
        f["params"]["n_sweep_meas"] = np.array(n_sweep_meas, dtype=np.int32)
        f["params"]["period_eqlt"] = np.array(period_eqlt, dtype=np.int32)
        f["params"]["period_uneqlt"] = np.array(period_uneqlt, dtype=np.int32)
        f["params"]["meas_bond_corr"] = meas_bond_corr
        f["params"]["meas_thermal"] = meas_thermal
        f["params"]["meas_2bond_corr"] = meas_2bond_corr
        f["params"]["meas_energy_corr"] = meas_energy_corr
        f["params"]["meas_nematic_corr"] = meas_nematic_corr
        f["params"]["meas_chiral"] = meas_chiral
        f["params"]["meas_local_JQ"] = meas_local_JQ
        f["params"]["checkpoint_every"] = checkpoint_every

        # precalculated stuff
        f["params"]["num_i"] = num_i
        f["params"]["num_ij"] = num_ij
        f["params"]["num_plaq_accum"] = num_plaq_accum
        f["params"]["num_b_accum"] = num_b_accum
        f["params"]["num_b2_accum"] = num_b2_accum
        f["params"]["num_plaq"] = num_plaq_total
        f["params"]["num_b"] = num_b
        f["params"]["num_b2"] = num_b2
        f["params"]["num_bs"] = num_bs
        f["params"]["num_bb"] = num_bb
        f["params"]["num_b2b"] = num_b2b
        f["params"]["num_bb2"] = num_bb2
        f["params"]["num_b2b2"] = num_b2b2
        f["params"]["degen_i"] = degen_i[0]
        f["params"]["degen_ij"] = degen_ij[0]
        f["params"]["degen_plaq"] = degen_plaq[0]
        f["params"]["degen_b"] = degen_b[0]
        f["params"]["degen_b2"] = degen_b2[0]
        f["params"]["degen_bs"] = degen_bs[0]
        f["params"]["degen_bb"] = degen_bb[0]
        f["params"]["degen_bb2"] = degen_bb2[0]
        f["params"]["degen_b2b"] = degen_b2b[0]
        f["params"]["degen_b2b2"] = degen_b2b2[0]
        f["params"]["exp_Ku"] = exp_Ku
        f["params"]["exp_Kd"] = exp_Kd
        f["params"]["inv_exp_Ku"] = inv_exp_Ku
        f["params"]["inv_exp_Kd"] = inv_exp_Kd
        f["params"]["exp_halfKu"] = exp_halfKu
        f["params"]["exp_halfKd"] = exp_halfKd
        f["params"]["inv_exp_halfKu"] = inv_exp_halfKu
        f["params"]["inv_exp_halfKd"] = inv_exp_halfKd
        f["params"]["exp_lambda"] = exp_lambda
        f["params"]["del"] = delll
        f["params"]["F"] = np.array(L // n_matmul, dtype=np.int32)
        f["params"]["n_sweep"] = np.array(n_sweep_warm + n_sweep_meas, dtype=np.int32)

    with h5py.File(file_sim, "a" if one_file else "w" if overwrite else "x") as f:
        # simulation state
        params_relpath = os.path.relpath(file_params, os.path.dirname(file_sim))
        f["params_file"] = params_relpath
        if not one_file:
            f["metadata"] = h5py.ExternalLink(params_relpath, "metadata")
            f["params"] = h5py.ExternalLink(params_relpath, "params")

        f.create_group("state")
        f["state"]["sweep"] = np.array(0, dtype=np.int32)
        f["state"]["init_rng"] = init_rng  # save if need to replicate data
        f["state"]["rng"] = rng
        f["state"]["hs"] = init_hs
        f["state"]["partial_write"] = 0

        # measurements
        f.create_group("meas_eqlt")
        f["meas_eqlt"]["n_sample"] = np.array(0, dtype=np.int32)
        f["meas_eqlt"]["sign"] = np.array(0.0, dtype=dtype_num)
        f["meas_eqlt"]["density"] = np.zeros(num_i, dtype=dtype_num)
        f["meas_eqlt"]["density_u"] = np.zeros(num_i, dtype=dtype_num)
        f["meas_eqlt"]["density_d"] = np.zeros(num_i, dtype=dtype_num)
        f["meas_eqlt"]["double_occ"] = np.zeros(num_i, dtype=dtype_num)
        f["meas_eqlt"]["g00"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["g00_u"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["g00_d"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["nn"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["xx"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["zz"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["pair_sw"] = np.zeros(num_ij, dtype=dtype_num)
        if meas_chiral:
            f["meas_eqlt"]["chi"] = np.zeros(num_plaq_accum, dtype=dtype_num)

        if meas_energy_corr:
            f["meas_eqlt"]["kk"] = np.zeros(num_bb, dtype=dtype_num)
            f["meas_eqlt"]["kv"] = np.zeros(num_bs, dtype=dtype_num)
            f["meas_eqlt"]["kn"] = np.zeros(num_bs, dtype=dtype_num)
            f["meas_eqlt"]["vv"] = np.zeros(num_ij, dtype=dtype_num)
            f["meas_eqlt"]["vn"] = np.zeros(num_ij, dtype=dtype_num)

        if period_uneqlt > 0:
            f.create_group("meas_uneqlt")
            f["meas_uneqlt"]["n_sample"] = np.array(0, dtype=np.int32)
            f["meas_uneqlt"]["sign"] = np.array(0.0, dtype=dtype_num)
            f["meas_uneqlt"]["gt0"] = np.zeros(num_ij * L, dtype=dtype_num)
            f["meas_uneqlt"]["gt0_u"] = np.zeros(num_ij * L, dtype=dtype_num)
            f["meas_uneqlt"]["gt0_d"] = np.zeros(num_ij * L, dtype=dtype_num)
            f["meas_uneqlt"]["nn"] = np.zeros(num_ij * L, dtype=dtype_num)
            f["meas_uneqlt"]["xx"] = np.zeros(num_ij * L, dtype=dtype_num)
            f["meas_uneqlt"]["zz"] = np.zeros(num_ij * L, dtype=dtype_num)
            f["meas_uneqlt"]["pair_sw"] = np.zeros(num_ij * L, dtype=dtype_num)
            if meas_bond_corr:
                f["meas_uneqlt"]["pair_bb"] = np.zeros(num_bb * L, dtype=dtype_num)
                f["meas_uneqlt"]["jj"] = np.zeros(num_bb * L, dtype=dtype_num)
                f["meas_uneqlt"]["jsjs"] = np.zeros(num_bb * L, dtype=dtype_num)
                f["meas_uneqlt"]["kk"] = np.zeros(num_bb * L, dtype=dtype_num)
                f["meas_uneqlt"]["ksks"] = np.zeros(num_bb * L, dtype=dtype_num)
            if meas_thermal:
                f["meas_uneqlt"]["j2jn"] = np.zeros(num_b2b * L, dtype=dtype_num)
                f["meas_uneqlt"]["jnj2"] = np.zeros(num_bb2 * L, dtype=dtype_num)
                f["meas_uneqlt"]["jnjn"] = np.zeros(num_bb * L, dtype=dtype_num)
                f["meas_uneqlt"]["jjn"] = np.zeros(num_bb * L, dtype=dtype_num)
                f["meas_uneqlt"]["jnj"] = np.zeros(num_bb * L, dtype=dtype_num)
            if meas_2bond_corr:
                # use j2j2 should correspond to J2J2 results after summation
                f["meas_uneqlt"]["j2j2"] = np.zeros(num_b2b2 * L, dtype=dtype_num)
                # use j2j should correspond to J2j results after summation
                f["meas_uneqlt"]["j2j"] = np.zeros(num_b2b * L, dtype=dtype_num)  # new
                # use jj2 should correspond to jJ2 results after summation
                f["meas_uneqlt"]["jj2"] = np.zeros(num_bb2 * L, dtype=dtype_num)  # new
                # these below are not implemented with phases currently
                f["meas_uneqlt"]["pair_b2b2"] = np.zeros(num_b2b2 * L, dtype=dtype_num)
                f["meas_uneqlt"]["js2js2"] = np.zeros(num_b2b2 * L, dtype=dtype_num)
                f["meas_uneqlt"]["k2k2"] = np.zeros(num_b2b2 * L, dtype=dtype_num)
                f["meas_uneqlt"]["ks2ks2"] = np.zeros(num_b2b2 * L, dtype=dtype_num)
            if meas_energy_corr:
                f["meas_uneqlt"]["kv"] = np.zeros(num_bs * L, dtype=dtype_num)
                f["meas_uneqlt"]["kn"] = np.zeros(num_bs * L, dtype=dtype_num)
                f["meas_uneqlt"]["vv"] = np.zeros(num_ij * L, dtype=dtype_num)
                f["meas_uneqlt"]["vn"] = np.zeros(num_ij * L, dtype=dtype_num)
            if meas_nematic_corr:
                f["meas_uneqlt"]["nem_nnnn"] = np.zeros(num_bb * L, dtype=dtype_num)
                f["meas_uneqlt"]["nem_ssss"] = np.zeros(num_bb * L, dtype=dtype_num)


def create_batch(Nfiles=1, prefix=None, seed=None, **kwargs):
    if seed is None:
        init_rng = rand_seed_urandom()
    else:
        init_rng = rand_seed_splitmix64(seed)

    if prefix is None:
        prefix = "sim"

    file_0 = "{}_{}.h5".format(prefix, 0)
    file_p = "{}.h5.params".format(prefix)

    create_1(file_sim=file_0, file_params=file_p, init_rng=init_rng, **kwargs)

    with h5py.File(file_p, "r") as f:
        N = f["params"]["N"][...]
        L = f["params"]["L"][...]

    for i in range(1, Nfiles):
        rand_jump(init_rng)
        rng = init_rng.copy()
        init_hs = np.zeros((L, N), dtype=np.int32)

        for l in range(L):
            for r in range(N):
                init_hs[l, r] = rand_uint(rng) >> np.uint64(63)

        file_i = "{}_{}.h5".format(prefix, i)
        shutil.copy2(file_0, file_i)
        with h5py.File(file_i, "r+") as f:
            f["state"]["init_rng"][...] = init_rng
            f["state"]["rng"][...] = rng
            f["state"]["hs"][...] = init_hs
    print(
        "created simulation files:",
        file_0 if Nfiles == 1 else "{} ... {}".format(file_0, file_i),
    )
    print("parameter file:", file_p)


# def main(argv):
#     kwargs = {}
#     for arg in argv[1:]:
#         eq = arg.find("=")
#         if eq == -1:
#             print("couldn't find \"=\" in argument " + arg)
#             return
#         key = arg[:eq]
#         val = arg[(eq + 1):]
#         try:
#             val = int(val)
#         except ValueError:
#             try:
#                 val = float(val)
#             except:
#                 pass
#         kwargs[key] = val
#     create_batch(**kwargs)

# if __name__ == "__main__":
#     main(sys.argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate .h5 files for dqmc simulation. "
        + "All parameters have defaults. Use arguments of form --name value or --name=value to override.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-V", "--version", action="version", version=hash_short)

    group1 = parser.add_argument_group("Physics parameters")
    group1.add_argument(
        "--geometry",
        choices=["square", "triangular", "honeycomb", "kagome"],
        type=str,
        default="square",
    )
    group1.add_argument(
        "--Nx",
        type=int,
        default=4,
        metavar="X",
        help="Number of lattice sites along x direction",
    )
    group1.add_argument(
        "--Ny",
        type=int,
        default=4,
        metavar="X",
        help="Number of lattice sites along y direction",
    )
    group1.add_argument(
        "--tp",
        type=float,
        default=0.0,
        metavar="X",
        help="Next nearest hopping integral",
    )
    group1.add_argument(
        "--tpp",
        type=float,
        default=0.0,
        metavar="X",
        help="Third nearest hopping integral",
    )
    group1.add_argument(
        "--nflux",
        type=int,
        default=0,
        metavar="X",
        help="Number of flux threading the cluster",
    )
    group1.add_argument(
        "--U",
        type=float,
        default=6.0,
        metavar="X",
        help="On-site Hubbard repulsion strength",
    )
    group1.add_argument(
        "--bc",
        type=int,
        default=1,
        metavar="X",
        help="Boundary conditions, 1 for periodic, 2 for open",
    )
    group1.add_argument(
        "--dt",
        type=float,
        default=0.1,
        metavar="X",
        help="Imaginary time discretization interval",
    )
    group1.add_argument(
        "--L", type=int, default=40, metavar="X", help="Number of imaginary time steps"
    )

    group1.add_argument(
        "--mu", type=float, default=0.0, metavar="X", help="Chemical potential"
    )
    group1.add_argument(
        "--h",
        type=float,
        default=0.0,
        metavar="X",
        help="Zeeman field strength. Down electrons feel net (mu+h) chemical potential",
    )
    group1.add_argument(
        "--twistx",
        type=float,
        default=0.0,
        metavar="X",
        help="Twist phase per bond along x. Equivalent to total twist Nx * twistx on boundary.",
    )
    group1.add_argument(
        "--twisty",
        type=float,
        default=0.0,
        metavar="X",
        help="Twist phase per bond along y. Equivalent to total twist Ny * twisty on boundary.",
    )

    group2 = parser.add_argument_group("Simulation file settings")
    group2.add_argument(
        "--prefix",
        type=str,
        default=None,
        metavar="X",
        help='Prefix for the name of each simulation file. If None, use "sim"',
    )
    group2.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="X",
        help="User-defined RNG seed. If None, use rand_seed_urandom()",
    )
    group2.add_argument(
        "--Nfiles",
        type=int,
        default=1,
        metavar="X",
        help="Number of simulation files to generate",
    )
    group2.add_argument(
        "--printout",
        type=int,
        default=1,
        metavar="X",
        help="Whether to print out parameter choices as .h5 files are created.",
    )
    group2.add_argument(
        "--overwrite",
        type=int,
        default=0,
        metavar="X",
        help="Whether to overwrite existing files",
    )
    group2.add_argument(
        "--n_delay",
        type=int,
        default=16,
        metavar="X",
        help="Number of updates to group together in the delayed update scheme",
    )
    group2.add_argument(
        "--n_matmul",
        type=int,
        default=8,
        metavar="X",
        help="Half the maximum number of direct matrix multiplications before applying a QR decomposition",
    )
    group2.add_argument(
        "--n_sweep_warm",
        type=int,
        default=200,
        metavar="X",
        help="Number of warmup sweeps",
    )
    group2.add_argument(
        "--n_sweep_meas",
        type=int,
        default=2000,
        metavar="X",
        help="Number of measurement sweeps",
    )
    group2.add_argument(
        "--period_eqlt",
        type=int,
        default=8,
        metavar="X",
        help="Period of equal-time measurements in units of single-site updates",
    )
    group2.add_argument(
        "--period_uneqlt",
        type=int,
        default=0,
        metavar="X",
        help="Period of unequal-time measurements in units of full H-S sweeps. 0 means disabled",
    )
    group2.add_argument(
        "--trans_sym",
        type=int,
        default=1,
        metavar="X",
        help="Whether to apply translational symmetry to compress measurement data",
    )
    group2.add_argument(
        "--checkpoint_every",
        type=int,
        default=10000,
        metavar="X",
        help="Number of full H-S sweeps between checkpoints. 0 means disabled",
    )

    group3 = parser.add_argument_group("Expensive measurement toggles")

    group3.add_argument(
        "--meas_bond_corr",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure bond-bond correlations (current, kinetic energy, bond singlets)",
    )
    group3.add_argument(
        "--meas_energy_corr",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure energy-energy correlations.",
    )
    group3.add_argument(
        "--meas_nematic_corr",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure spin and charge nematic correlations",
    )
    group3.add_argument(
        "--meas_thermal",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure extra jnj(2) type correlations for themal conductivity",
    )
    group3.add_argument(
        "--meas_2bond_corr",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure extra jj(2) type correlations for themal conductivity",
    )
    group3.add_argument(
        "--meas_chiral",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure scalar spin chirality",
    )
    group3.add_argument(
        "--meas_local_JQ",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure local JQ for energy magnetization contribution to thermal Hall",
    )

    # parser.add_argument
    args = parser.parse_args()

    argdict = vars(args)

    if args.printout == 1:
        for k, v in argdict.items():
            print(k, v)

    delattr(args, "printout")

    create_batch(**vars(args))
