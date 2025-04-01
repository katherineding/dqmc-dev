import os
import shutil
import sys
import argparse
import warnings

import h5py
import numpy as np
from scipy.linalg import expm

import tight_binding
import triangle_HH_tb
import gen_util_shared as gus

np.seterr(over="ignore")
np.set_printoptions(precision=3)

import git  # relies on gitpython module

path = os.path.dirname(os.path.abspath(__file__))
repo = git.Repo(path, search_parent_directories=True)
hash_short = repo.git.rev_parse(repo.head, short=True)

Norb_per_cell_dict = {}
Norb_per_cell_dict["square"] = 1
Norb_per_cell_dict["triangular"] = 1
Norb_per_cell_dict["honeycomb"] = 2
Norb_per_cell_dict["kagome"] = 3


# Set up geometry-specific parameters
# FIXME: are plaquettes constrained by requiring nonzero hopping element
# to connect sites, or can plaquettes be defined for arbitrary 3 sites?
plaq_per_cell_dict = {}
plaq_per_cell_dict["square"] = 2
plaq_per_cell_dict["triangular"] = 2
plaq_per_cell_dict["honeycomb"] = 1  # PLACEHOLDER
plaq_per_cell_dict["kagome"] = 2


def plaquette_params(
    geometry: str, Nx: int, Ny: int, trans_sym: int
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Geometry specific plaquette parameters. TODO: honeycomb
    Returns:
        (plaq_per_cell, num_plaq_total, num_plaq_accum), map_plaq, degen_plaq, plaqs
    """
    Ncell = Nx * Ny
    plaq_per_cell = plaq_per_cell_dict[geometry]
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
    if geometry == "square" or geometry == "triangular":
        # Use same plaquette definition as triangular TODO: check correctness
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx * iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                plaqs[0, i] = i  # i0 = i
                plaqs[1, i] = ix1 + Nx * iy  # i1 = i + x
                plaqs[2, i] = ix + Nx * iy1  # i2 = i + y // counterclockwise
                plaqs[0, i + Nx * Ny] = ix1 + Nx * iy  # i0 = i + x
                plaqs[1, i + Nx * Ny] = ix1 + Nx * iy1  # i1 = i + x + y
                plaqs[2, i + Nx * Ny] = ix + Nx * iy1  # i2 = i + y //counterclockwise
    elif geometry == "kagome":
        # plaquette definitions TODO: check correctness
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx * iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                plaqs[0, i] = i  # i0 = i(A)
                plaqs[1, i] = i + Nx * Ny * 2  # i1 = i(C)
                plaqs[2, i] = i + Nx * Ny * 1  # i2 = i(B) // counterclockwise
                plaqs[0, i + Nx * Ny] = i  # i0 = i(A)
                plaqs[1, i + Nx * Ny] = ix1 + Nx * iy + Nx * Ny * 2  # i1 = i+x(C)
                plaqs[2, i + Nx * Ny] = (
                    ix + Nx * iy1 + Nx * Ny * 1
                )  # i2 = i+y(B) //counterclockwise
    else:
        print(
            f"WARN: using placeholder {sys._getframe().f_code.co_name} for {geometry}"
        )

    return (plaq_per_cell, num_plaq_total, num_plaq_accum), map_plaq, degen_plaq, plaqs


def bond_params(
    geometry: str, Nx: int, Ny: int, tp: float, tpp: float, trans_sym: int
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Geometry specific 1-bond parameters TODO: triangular, honeycomb, kagome
    TODO: tpp != 0 for all
    Probably want to eventually get bond params correct for triangular, honeycomb, kagome
    Returns:
        (bps, num_b, num_b_accum), map_b, degen_b, bonds
    """
    N = Nx * Ny * Norb_per_cell_dict[geometry]
    if tpp != 0.0:
        raise NotImplementedError

    # 1-bonds per site
    if geometry == "square":
        bps = 4 if tp != 0.0 else 2  # OK
    elif geometry == "triangular":
        bps = 6 if tp != 0.0 else 3  # OK
    elif geometry == "honeycomb":
        bps = 6 if tp != 0.0 else 3  # CHECK
    elif geometry == "kagome":
        bps = 8 if tp != 0 else 4  # CHECK
    else:
        raise NotImplementedError
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
    if geometry == "square":
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx * iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                bonds[0, i] = i  # i0 = i
                bonds[1, i] = ix1 + Nx * iy  # i1 = i + x
                bonds[0, i + N] = i  # i0 = i
                bonds[1, i + N] = ix + Nx * iy1  # i1 = i + y
                if bps == 4:
                    bonds[0, i + 2 * N] = i  # i0 = i
                    bonds[1, i + 2 * N] = ix1 + Nx * iy1  # i1 = i + x + y
                    bonds[0, i + 3 * N] = ix1 + Nx * iy  # i0 = i + x
                    bonds[1, i + 3 * N] = ix + Nx * iy1  # i1 = i + y
    elif geometry == "triangular":
        # Three bonds of lattice are down_left / (0), down right \ (1), and right -- (2)    
        # Chosen such that bonds only go towards larger Nx and Ny
        # In the picture of square <--> triangular, which is easier to see the labeling in,
        # these are down, down right, and right                          
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx * iy
                i_down_left_y = (iy + 1) % Ny
                i_down_right_x = (ix + 1) % Nx
                i_down_right_y = (iy + 1) % Ny
                i_right_x = (ix + 1) % Nx
                bonds[0, i] = i
                bonds[1, i] = ix + Nx * i_down_left_y
                bonds[0, i + N] = i
                bonds[1, i + N] = i_down_right_x + Nx * i_down_right_y
                bonds[0, i + 2 * N] = i
                bonds[1, i + 2 * N] = i_right_x + Nx * iy
                if bps == 6:
                    raise NotImplementedError
    else:
        print(
            f"WARN: using placeholder {sys._getframe().f_code.co_name} for {geometry}"
        )

    return (bps, num_b, num_b_accum), map_b, degen_b, bonds


def bond2_params(
    geometry: str, Nx: int, Ny: int, tp: float, tpp: float, trans_sym: int
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Geometry specific 1-bond parameters TODO: triangular, honeycomb, kagome
    TODO: tpp != 0 for all
    Returns:
        (bps, num_b, num_b_accum), map_b, degen_b, bonds
    """
    N = Nx * Ny * Norb_per_cell_dict[geometry]
    if tpp != 0.0:
        raise NotImplementedError

    # 2-bonds per site
    if geometry == "square":
        b2ps = 12 if tp != 0.0 else 4  # OK
    elif geometry == "triangular":
        b2ps = 12 if tp != 0.0 else 4  # probably wrong
    elif geometry == "honeycomb":
        b2ps = 12 if tp != 0.0 else 4  # probably wrong
    elif geometry == "kagome":
        b2ps = 12 if tp != 0.0 else 4  # probably wrong
    else:
        raise NotImplementedError
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
    if geometry == "square":
        # 2-bond definition is modified -- NOT consistent with Wen's!
        # Now only bonds defined by two hopping steps.
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx * iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                iy2 = (iy + 2) % Ny
                ix2 = (ix + 2) % Nx
                # one t^2 path + two t'^2 paths [0,22,23]
                bond2s[0, i + 0 * N] = i  # i0 = i
                bond2s[1, i + 0 * N] = ix2 + Nx * iy  # i1 = i + 2x   -- /\ \/
                # one t^2 path + two t'^2 paths [1,24,25]
                bond2s[0, i + 1 * N] = i  # i0 = i       |  /  \
                bond2s[1, i + 1 * N] = ix + Nx * iy2  # i1 = i + 2y  |  \  /
                # two t^2 paths [2,3]
                bond2s[0, i + 2 * N] = i  # i0 = i          _|   _
                bond2s[1, i + 2 * N] = ix1 + Nx * iy1  # i1 = i + x + y      |
                # two t^2 paths [4,5]
                bond2s[0, i + 3 * N] = ix1 + Nx * iy  # i0 = i + x     _
                bond2s[1, i + 3 * N] = ix + Nx * iy1  # i1 = i + y      |  |_
                if b2ps == 12:
                    # two t't paths [6,7]
                    bond2s[0, i + 4 * N] = i  # i0 = i              _
                    bond2s[1, i + 4 * N] = ix2 + Nx * iy1  # i1 = i + 2x + y _/ /
                    # two t't paths [8,9]
                    bond2s[0, i + 5 * N] = i  # i0 = i           |   /
                    bond2s[1, i + 5 * N] = ix1 + Nx * iy2  # i1 = i + x + 2y /   |
                    # two t't paths [10,11]
                    bond2s[0, i + 6 * N] = ix2 + Nx * iy  # i0 = i + 2x _
                    bond2s[1, i + 6 * N] = ix + Nx * iy1  # i1 = i + y   \  \_
                    # two t't paths [12,13]
                    bond2s[0, i + 7 * N] = ix1 + Nx * iy  # i0 = i + x   |   \
                    bond2s[1, i + 7 * N] = ix + Nx * iy2  # i1 = i + 2y   \   |
                    # four t't paths [14,15,16,17]
                    bond2s[0, i + 8 * N] = i  # i0 = i      _  _   \  /
                    bond2s[1, i + 8 * N] = ix + Nx * iy1  # i1 = i + y  /  \   -  -
                    # four t't paths [18,19,20,21]
                    bond2s[0, i + 9 * N] = i  # i0 = i      |\ /| \| |/
                    bond2s[1, i + 9 * N] = ix1 + Nx * iy  # i1 = i + x
                    # one t'^2 path [26]
                    bond2s[0, i + 10 * N] = i  # i0 = i             /
                    bond2s[1, i + 10 * N] = ix2 + Nx * iy2  # i1 = i + 2x + 2y  /
                    # one t'^2 path [27]
                    bond2s[0, i + 11 * N] = ix2 + Nx * iy  # i0 = i + 2x   \
                    bond2s[1, i + 11 * N] = ix + Nx * iy2  # i1 = i + 2y    \
    else:
        print(
            f"WARN: using placeholder {sys._getframe().f_code.co_name} for {geometry}"
        )

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
    geometry: str = "square",
    bc: int = 1,
    Nx: int = 4,
    Ny: int = 4,
    mu: float = 0.0,
    tp: float = 0.0,
    tpp: float = 0.0,
    # Careful: tpp will add more bonds to transport or other measurements. These parts are not included in this code.
    # adding tpp, currently, the measurments are not enough to get whole bond-anything correlations.
    U: float = 6.0,
    dt: float = 0.1,
    L: int = 40,
    nflux: int = 0,
    h: float = 0.0,
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
    meas_pair_bb_only: int = 0,
    meas_chiral: int = 0,
    meas_local_JQ: int = 0,
    meas_gen_suscept: int = 0,
    twistx: float = 0,
    twisty: float = 0,
):
    assert L % n_matmul == 0 and L % period_eqlt == 0
    if nflux != 0 or twistx != 0 or twisty != 0:
        dtype_num = np.complex128
    else:
        dtype_num = np.float64

    if file_sim is None:
        file_sim = "sim.h5"
    if file_params is None:
        file_params = file_sim

    one_file = os.path.abspath(file_sim) == os.path.abspath(file_params)

    if init_rng is None:
        init_rng = gus.rand_seed_urandom()
    rng = init_rng.copy()

    Ncell = Nx * Ny
    Norb = Norb_per_cell_dict[geometry]
    N = Norb * Ncell

    init_hs = np.zeros((L, N), dtype=np.int32)
    for l in range(L):
        for i in range(N):
            init_hs[l, i] = gus.rand_uint(rng) >> np.uint64(63)

    if meas_gen_suscept and not trans_sym:
        raise ValueError("Can't measure generalized susceptibility without trans_sym")

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

    # ------------------------------------------------------
    # per plaquette (per site) measurement mapping
    (
        (plaq_per_cell, num_plaq_total, num_plaq_accum),
        map_plaq,
        degen_plaq,
        plaqs,
    ) = plaquette_params(geometry, Nx, Ny, trans_sym)

    # ------------------------------------------------------
    # per bond (per site) measurement mapping
    (
        (bps, num_b, num_b_accum),
        map_b,
        degen_b,
        bonds,
    ) = bond_params(geometry, Nx, Ny, tp, tpp, trans_sym)

    # ------------------------------------------------------
    # per 2-bond (per site) measurement mapping
    (
        (b2ps, num_b2, num_b2_accum),
        map_b2,
        degen_b2,
        bond2s,
    ) = bond2_params(geometry, Nx, Ny, tp, tpp, trans_sym)

    # placeholder until other boundaries implemented for non square lattices
    if (bc != 1) and (geometry != "square"):
        raise NotImplementedError(
            "Non-periodic boundaries only implemented for square lattice"
        )

    # if non-periodic and trans_sym on, warn user that trans_sym will turn off
    if (bc != 1) and trans_sym:
        warnings.warn(
            "Non-periodic boundaries not translationally symmetric: turning off trans_sym"
        )
        trans_sym = 0

    if geometry == "square":
        # 2 site mapping: site r = (x,y) has total (column order) index x + Nx * y
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = N if trans_sym else N * N
        degen_ij = np.zeros(num_ij, dtype=np.int32)
        for jy in range(Ny):
            for jx in range(Nx):
                for iy in range(Ny):
                    for ix in range(Nx):
                        if trans_sym:
                            ky = (iy - jy) % Ny
                            kx = (ix - jx) % Nx
                            k = kx + Nx * ky
                        else:
                            k = (ix + Nx * iy) + N * (jx + Nx * jy)
                        map_ij[jx + Nx * jy, ix + Nx * iy] = k
                        degen_ij[k] += 1
        assert num_ij == map_ij.max() + 1 == degen_ij.size
        assert np.all(degen_ij == degen_ij[0])

        # 1 bond 1 site mapping
        # Translated to fortran order: [j,istuff] -> [istuff + num_b * j] -> [istuff,j]
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps * N if trans_sym else num_b * N
        degen_bs = np.zeros(num_bs, dtype=np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for ib in range(bps):
                    kk = k + num_ij * ib
                    map_bs[j, i + N * ib] = kk
                    degen_bs[kk] += 1
        assert num_bs == map_bs.max() + 1 == degen_bs.size
        assert np.all(degen_bs == degen_bs[0])

        # 1 bond - 1 bond mapping
        # Translated to Fortran order: [jstuff ,istuff] -> [istuff + num_b * jstuff] -> [istuff,jstuff]
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps * bps * N if trans_sym else num_b * num_b
        degen_bb = np.zeros(num_bb, dtype=np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(bps):
                    for ib in range(bps):
                        kk = k + num_ij * (ib + bps * jb)
                        map_bb[j + N * jb, i + N * ib] = kk
                        degen_bb[kk] += 1
        assert num_bb == map_bb.max() + 1 == degen_bb.size
        assert np.all(degen_bb == degen_bb[0])

        # my definition: Bonds defined by two hopping steps
        # Keep track of intermediate point!
        # TODO see if bonds 14 and 16 etc are not equivalent b/c start/end points changed...
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps * N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx * iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                iy2 = (iy + 2) % Ny
                ix2 = (ix + 2) % Nx

                iym1 = (iy - 1) % Ny
                ixm1 = (ix - 1) % Nx

                # t^2 terms: NN + NN
                hop2s[0, i] = i  # i0 = i
                hop2s[1, i] = ix1 + Nx * iy  # i1 = i + x      --
                hop2s[2, i] = ix2 + Nx * iy  # i2 = i + 2x
                # -----------------
                hop2s[0, i + N] = i  # i0 = i
                hop2s[1, i + N] = ix + Nx * iy1  # i1 = i + y     |
                hop2s[2, i + N] = ix + Nx * iy2  # i2 = i + 2y    |
                # -----------------
                hop2s[0, i + 2 * N] = i  # i0 = i               _|
                hop2s[1, i + 2 * N] = ix1 + Nx * iy  # i1 = i + x
                hop2s[2, i + 2 * N] = ix1 + Nx * iy1  # i2 = i + x + y

                hop2s[0, i + 3 * N] = i  # i0 = i               _
                hop2s[1, i + 3 * N] = ix + Nx * iy1  # i1 = i + y          |
                hop2s[2, i + 3 * N] = ix1 + Nx * iy1  # i2 = i + x + y
                # -----------------
                hop2s[0, i + 4 * N] = ix1 + Nx * iy  # i0 = i + x            _
                hop2s[1, i + 4 * N] = ix1 + Nx * iy1  # i1 = i + x + y         |
                hop2s[2, i + 4 * N] = ix + Nx * iy1  # i2 = i + y

                hop2s[0, i + 5 * N] = ix1 + Nx * iy  # i0 = i + x           |_
                hop2s[1, i + 5 * N] = i  # i1 = i
                hop2s[2, i + 5 * N] = ix + Nx * iy1  # i2 = i + y

                if hop2ps == 28:
                    # t*t' terms: NN + NNN or NNN + NN
                    hop2s[0, i + 6 * N] = i  # i0 = i
                    hop2s[1, i + 6 * N] = ix1 + Nx * iy  # i1 = i + x         _/
                    hop2s[2, i + 6 * N] = ix2 + Nx * iy1  # i2 = i + 2x + y

                    hop2s[0, i + 7 * N] = i  # i0 = i
                    hop2s[1, i + 7 * N] = ix1 + Nx * iy1  # i1 = i + x + y     _
                    hop2s[2, i + 7 * N] = ix2 + Nx * iy1  # i2 = i + 2x + y   /
                    # ------------------
                    hop2s[0, i + 8 * N] = i  # i0 = i             |
                    hop2s[1, i + 8 * N] = ix1 + Nx * iy1  # i1 = i + x + y    /
                    hop2s[2, i + 8 * N] = ix1 + Nx * iy2  # i2 = i + x + 2y

                    hop2s[0, i + 9 * N] = i  # i0 = i             /
                    hop2s[1, i + 9 * N] = ix + Nx * iy1  # i1 = i + y        |
                    hop2s[2, i + 9 * N] = ix1 + Nx * iy2  # i2 = i + x + 2y
                    # ------------------
                    hop2s[0, i + 10 * N] = ix2 + Nx * iy  # i0 = i + 2x
                    hop2s[1, i + 10 * N] = ix1 + Nx * iy1  # i1 = i + x + y    _
                    hop2s[2, i + 10 * N] = ix + Nx * iy1  # i2 = i + y         \

                    hop2s[0, i + 11 * N] = ix2 + Nx * iy  # i0 = i + 2x
                    hop2s[1, i + 11 * N] = ix1 + Nx * iy  # i1 = i + x       \_
                    hop2s[2, i + 11 * N] = ix + Nx * iy1  # i2 = i + y
                    # ------------------
                    hop2s[0, i + 12 * N] = ix1 + Nx * iy  # i0 = i + x
                    hop2s[1, i + 12 * N] = ix + Nx * iy1  # i1 = i + y       |
                    hop2s[2, i + 12 * N] = ix + Nx * iy2  # i2 = i + 2y       \

                    hop2s[0, i + 13 * N] = ix1 + Nx * iy  # i0 = i + x
                    hop2s[1, i + 13 * N] = ix1 + Nx * iy1  # i1 = i + x + y     \
                    hop2s[2, i + 13 * N] = ix + Nx * iy2  # i2 = i + 2y         |
                    # ------------------
                    hop2s[0, i + 14 * N] = i  # i0 = i
                    hop2s[1, i + 14 * N] = ix1 + Nx * iy1  # i1 = i + x + y     _
                    hop2s[2, i + 14 * N] = ix + Nx * iy1  # i2 = i + y         /

                    hop2s[0, i + 15 * N] = i  # i0 = i
                    hop2s[1, i + 15 * N] = ix1 + Nx * iy  # i1 = i + x         \
                    hop2s[2, i + 15 * N] = ix + Nx * iy1  # i2 = i + y         -

                    hop2s[0, i + 16 * N] = i  # i0 = i           _
                    hop2s[1, i + 16 * N] = ixm1 + Nx * iy1  # i1 = i - x + y   \
                    hop2s[2, i + 16 * N] = ix + Nx * iy1  # i2 = i + y

                    hop2s[0, i + 17 * N] = i  # i0 = i
                    hop2s[1, i + 17 * N] = ixm1 + Nx * iy  # i1 = i - x         /
                    hop2s[2, i + 17 * N] = ix + Nx * iy1  # i2 = i + y         -
                    # ------------------
                    hop2s[0, i + 18 * N] = i  # i0 = i
                    hop2s[1, i + 18 * N] = ix1 + Nx * iy1  # i1 = i + x + y     /|
                    hop2s[2, i + 18 * N] = ix1 + Nx * iy  # i2 = i + x

                    hop2s[0, i + 19 * N] = i  # i0 = i
                    hop2s[1, i + 19 * N] = ix + Nx * iy1  # i1 = i + y     |\
                    hop2s[2, i + 19 * N] = ix1 + Nx * iy  # i2 = i + x

                    hop2s[0, i + 20 * N] = i  # i0 = i
                    hop2s[1, i + 20 * N] = ix1 + Nx * iym1  # i1 = i + x - y     \|
                    hop2s[2, i + 20 * N] = ix1 + Nx * iy  # i2 = i + x

                    hop2s[0, i + 21 * N] = i  # i0 = i
                    hop2s[1, i + 21 * N] = ix + Nx * iym1  # i1 = i - y      |/
                    hop2s[2, i + 21 * N] = ix1 + Nx * iy  # i2 = i + x
                    # (t'^2) terms: NNN + NNN
                    hop2s[0, i + 22 * N] = i  # i0 = i
                    hop2s[1, i + 22 * N] = ix1 + Nx * iy1  # i1 = i + x + y   /\
                    hop2s[2, i + 22 * N] = ix2 + Nx * iy  # i2 = i + 2x

                    hop2s[0, i + 23 * N] = i  # i0 = i
                    hop2s[1, i + 23 * N] = ix1 + Nx * iym1  # i1 = i + x - y   \/
                    hop2s[2, i + 23 * N] = ix2 + Nx * iy  # i2 = i + 2x
                    # ------------------
                    hop2s[0, i + 24 * N] = i  # i0 = i
                    hop2s[1, i + 24 * N] = ix1 + Nx * iy1  # i1 = i + x + y    \
                    hop2s[2, i + 24 * N] = ix + Nx * iy2  # i2 = i + 2y       /

                    hop2s[0, i + 25 * N] = i  # i0 = i            /
                    hop2s[1, i + 25 * N] = ixm1 + Nx * iy1  # i1 = i - x + y    \
                    hop2s[2, i + 25 * N] = ix + Nx * iy2  # i2 = i + 2y
                    # ------------------
                    hop2s[0, i + 26 * N] = i  # i0 = i              /
                    hop2s[1, i + 26 * N] = ix1 + Nx * iy1  # i1 = i + x + y     /
                    hop2s[2, i + 26 * N] = ix2 + Nx * iy2  # i2 = i + 2x + 2y
                    # ------------------
                    hop2s[0, i + 27 * N] = ix2 + Nx * iy  # i0 = i + 2x       \
                    hop2s[1, i + 27 * N] = ix1 + Nx * iy1  # i1 = i + x + y     \
                    hop2s[2, i + 27 * N] = ix + Nx * iy2  # i2 = i + 2y

        # how bond2s and hop2s are related
        bond_hop_dict = {}
        if b2ps == 4:
            bond_hop_dict[0] = [0]
            bond_hop_dict[1] = [1]
            bond_hop_dict[2] = [2, 3]
            bond_hop_dict[3] = [4, 5]
        else:
            bond_hop_dict[0] = [0, 22, 23]
            bond_hop_dict[1] = [1, 24, 25]
            for i in range(2, 8):
                bond_hop_dict[i] = [2 * i - 2, 2 * i - 1]
            bond_hop_dict[8] = [14, 15, 16, 17]
            bond_hop_dict[9] = [18, 19, 20, 21]
            bond_hop_dict[10] = [26]
            bond_hop_dict[11] = [27]

        # 2 2-bond mapping
        # Translated to Fortran order: [jstuff ,istuff] -> [istuff + num_b2 * jstuff] -> [istuff,jstuff]
        num_b2b2 = b2ps * b2ps * N if trans_sym else num_b2 * num_b2
        map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
        degen_b2b2 = np.zeros(num_b2b2, dtype=np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(b2ps):
                    for ib in range(b2ps):
                        kk = k + num_ij * (ib + b2ps * jb)
                        map_b2b2[j + N * jb, i + N * ib] = kk
                        degen_b2b2[kk] += 1
        assert num_b2b2 == map_b2b2.max() + 1 == degen_b2b2.size
        # print(map_b2b2.shape, degen_b2b2.shape)
        assert np.all(degen_b2b2 == degen_b2b2[0])

        # bond 2-bond mapping
        num_bb2 = bps * b2ps * N if trans_sym else num_b * num_b2
        map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
        degen_bb2 = np.zeros(num_bb2, dtype=np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(bps):
                    for ib in range(b2ps):
                        kk = k + num_ij * (ib + b2ps * jb)
                        map_bb2[j + N * jb, i + N * ib] = kk
                        degen_bb2[kk] += 1
        assert num_bb2 == map_bb2.max() + 1 == degen_bb2.size
        assert np.all(degen_bb2 == degen_bb2[0])

        # 2-bond bond mapping
        num_b2b = b2ps * bps * N if trans_sym else num_b2 * num_b
        map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
        degen_b2b = np.zeros(num_b2b, dtype=np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(b2ps):
                    for ib in range(bps):
                        kk = k + num_ij * (ib + bps * jb)
                        map_b2b[j + N * jb, i + N * ib] = kk
                        degen_b2b[kk] += 1
        assert num_b2b == map_b2b.max() + 1 == degen_b2b.size
        assert np.all(degen_b2b == degen_b2b[0])

        if bc == 1:
            kij, peierls = tight_binding.H_periodic_square(
                Nx,
                Ny,
                t=1,
                tp=tp,
                tpp=tpp,
                nflux=nflux,
                alpha=1 / 2,
                twistx=twistx,
                twisty=twisty,
            )
        elif bc == 2:
            kij, peierls = tight_binding.H_open_square(
                Nx,
                Ny,
                t=1,
                tp=tp,
                tpp=tpp,
                nflux=nflux,
                alpha=1 / 2,
                twistx=twistx,
                twisty=twisty,
            )
        else:
            raise ValueError("Invalid bc choice, must be 1 for periodic or 2 for open")

        # phases accumulated by two-hop processes
        # Here: types 0,1 include t' factors
        #   Types 2-7: sum of two paths
        #   Types 8,9: sum of four paths
        #   Types 10,11: one path each
        #   ZF case: 1+2*tp, 1+2*tp, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1
        # non-zero tpp case is not considered yet.
        thermal_phases = np.ones((b2ps, N), dtype=np.complex128)
        for i in range(N):
            for btype in range(b2ps):
                i0 = bond2s[0, i + btype * N]  # start
                i2 = bond2s[1, i + btype * N]  # end
                pp = 0
                # list of intermediate pointscorresponding to this bond
                i1_type_list = bond_hop_dict[btype]
                # print("btype = ",btype,"i1_type_list = ",i1_type_list)
                # two bonds need manual weighting when t' != 0:
                if b2ps == 12 and (btype == 0 or btype == 1):
                    i1 = hop2s[1, i + i1_type_list[0] * N]
                    # print(f"i = {i}, btype = {btype}, path ({i0},{i1},{i2})")
                    pp += peierls[i0, i1] * peierls[i1, i2]
                    for i1type in i1_type_list[1:]:
                        i1 = hop2s[1, i + i1type * N]
                        pp += tp * tp * peierls[i0, i1] * peierls[i1, i2]
                    # print(pp)
                # general case
                else:
                    for i1type in i1_type_list:
                        i1 = hop2s[1, i + i1type * N]
                        pp += peierls[i0, i1] * peierls[i1, i2]
                thermal_phases[btype, i] = pp

    elif geometry == "triangular":
        # 2 site mapping
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = N if trans_sym else N * N
        degen_ij = np.zeros(num_ij, dtype=np.int32)
        for jy in range(Ny):
            for jx in range(Nx):
                for iy in range(Ny):
                    for ix in range(Nx):
                        if trans_sym:
                            ky = (iy - jy) % Ny
                            kx = (ix - jx) % Nx
                            k = kx + Nx * ky
                        else:
                            k = (ix + Nx * iy) + N * (jx + Nx * jy)
                        map_ij[jx + Nx * jy, ix + Nx * iy] = k
                        degen_ij[k] += 1
        assert num_ij == map_ij.max() + 1 == degen_ij.size
        assert np.all(degen_ij == degen_ij[0])

        '''
        Explanation of above.
        map_ij makes a matrix that will be our site-site mapping. It ends up encoding which
        two sites are connected. In the translationally symmetric case, many different site
        pairs will have the same k, which is fine because of the translational symmetry

        num_ij tells you the number of site-site pairs in the mapping. When there isn't trans
        symmetry, it is clearly N*N. With trans symmetry, only need to consider all sites' relations
        to a single one and bin things

        degen_ij saves space by putting all "identical pairs" into the same bin of a long 1d array.
        The "identical pairs" are identified by their k

        the loop goes over all sites. Each site is identified by an x and y,
        which leads to there being ix, iy, jx, & jy.
        
        If the system is translationally symmetric, it is equivalent to use a k 
        that is the difference between the sites mod the lattice size
        If it isn't translationally symmetric, k is then just the 
        label of i plus N times the label of j

        Then map_ij[j,i] is set to be k
        this is all same as square, which makes sense
        '''

        # 1 bond 1 site mapping NOTE: placeholder
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps * N if trans_sym else num_b * N
        degen_bs = np.zeros(num_bs, dtype=np.int32)

        # 1 bond - 1 bond mapping NOTE: placeholder
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps * bps * N if trans_sym else num_b * num_b
        degen_bb = np.zeros(num_bb, dtype=np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j,i]
                for jb in range(bps):
                    for ib in range(bps):
                        kk = k + num_ij * (ib + bps * jb)
                        map_bb[j + N * jb, i + N * ib] = kk
                        degen_bb[kk] += 1
        assert num_bb == map_bb.max() + 1 == degen_bb.size
        assert np.all(degen_bb == degen_bb[0])

        '''
        map_bb makes a matrix that will be our bond-bond mapping

        num_bb tells you the number of bond-bond pairs. w/o trans_sym, the number
        of bond-bond pairs is clearly num_b * num_b. With it, one would only need to consider
        the bonds of a single site, bps, paired with all bonds N*bps. This is then the number
        of bins for the compressed degen case.

        We then loop over all sites and all bonds to make our mapping. In a triangular lattice,
        each site has 6 connections --> 3 bonds per site. We want to construct the kk that
        identifies the mapping like the k in the site-site case. For two given sites i and j,
        we start by taking their k, which identifies the connection between the two sites. Then,
        we loop over the bond types i and j can have. For each one, we construct kk by using
        k (the site-site identifier) as a "base" and then adding a specifier about the bond
        with the same convention as we made site identifiers previously.
        
        A point about whether the mapping actually works.
        k encodes the info about the two involved sites k = (ix + Nx * iy) + N * (jx + Nx * jy)
        To extract, one could find the part of the number smaller than N, which tells you i,
        and mod N to get j. 
        kk takes that number, which uniquely determines the site-site pair, and adds to it
        the unique identifier of the bond type-bond type combination (0-0, 0-1, etc.)
        There are 9 bond-bond possibilities, and those just need to be encoded in a way
        that doesn't interfere with the k, which takes up all numbers up to N-1 + N*(N-1) = N^2 - 1
        Katherine's takes num_ij, which is N^2 in the trans_sym = false case. This is higher than
        N^2 - 1, so it is encoded as we want.

        What about the trans_sym case?
        In the trans_sym case, we consider all pairs from a single site i.e. some number of 
        steps forward in the lattice. So, k only ranges from 0 to N - 1, which specifies
        the jump from a site i. Then, in the bond-bond case we consdier all bonds that
        can be paired with a single site's bonds. To use the same code as the
        non-translationally symmetric code, we just use the same kk bond type-bond type info
        encoding to add to the k. This will still do the degen properly, as for any sites that are
        considered the same, if the bonds are the same too they'll end up in the same bucket
        '''

        # my definition: Bonds defined by two hopping steps
        # NOTE: placeholder
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps * N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)

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

        if nflux == N / 2:
            kij, peierls = triangle_HH_tb.H_periodic_triangular(
                Nx, Ny, t=1, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )
        else:
            kij, peierls = tight_binding.H_periodic_triangular(
                Nx, Ny, t=1, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )

        # NOTE: placeholder
        thermal_phases = np.ones((b2ps, N), dtype=np.complex128)

    elif geometry == "honeycomb":
        # 2 site mapping
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = Norb * Norb * Ny * Nx if trans_sym else N * N
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

        # print("Trans sym = ",trans_sym)
        # print("map",map_ij,map_ij.shape,"degen",degen_ij,degen_ij.shape,"num",num_ij)

        # 1 bond 1 site mapping NOTE: placeholder
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps * N if trans_sym else num_b * N
        degen_bs = np.zeros(num_bs, dtype=np.int32)

        # 1 bond - 1 bond mapping NOTE: placeholder
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps * bps * N if trans_sym else num_b * num_b
        degen_bb = np.zeros(num_bb, dtype=np.int32)

        # my definition: Bonds defined by two hopping steps
        # NOTE: placeholder
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps * N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)

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

        kij, peierls = tight_binding.H_periodic_honeycomb(
            Nx, Ny, t=1, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

        # phases accumulated by two-hop processes NOTE: placeholder
        thermal_phases = np.ones((b2ps, N), dtype=np.complex128)

    elif geometry == "kagome":
        # 2 site mapping
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = Norb * Norb * Ny * Nx if trans_sym else N * N
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
                                    k = (
                                        kx
                                        + Nx * ky
                                        + Nx * Ny * io
                                        + Nx * Ny * Norb * jo
                                    )
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

        # 1 bond 1 site mapping NOTE: placeholder
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps * N if trans_sym else num_b * N
        degen_bs = np.zeros(num_bs, dtype=np.int32)

        # 1 bond - 1 bond mapping NOTE: placeholder
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps * bps * N if trans_sym else num_b * num_b
        degen_bb = np.zeros(num_bb, dtype=np.int32)

        # my definition: Bonds defined by two hopping steps
        # NOTE: placeholder
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps * N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)

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

        kij, peierls = tight_binding.H_periodic_kagome(
            Nx, Ny, t=1, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

        # phases accumulated by two-hop processes NOTE: placeholder
        thermal_phases = np.ones((b2ps, N), dtype=np.complex128)

    else:
        raise NotImplementedError("Invalid geometry")

    # account for different data type when nflux=0
    thermal_phases = (
        thermal_phases
        if (nflux != 0 or twistx != 0 or twisty != 0)
        else thermal_phases.real
    )
    Ku = kij if (nflux != 0 or twistx != 0 or twisty != 0) else kij.real
    peierls = peierls if (nflux != 0 or twistx != 0 or twisty != 0) else peierls.real

    # Zeeman interaction
    Kd = Ku.copy()
    for i in range(N):
        Ku[i, i] -= mu - h
        Kd[i, i] -= mu + h

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
        f["metadata"]["model"] = (
            "Hubbard (complex)" if dtype_num == np.complex128 else "Hubbard"
        )
        f["metadata"]["Nx"] = Nx
        f["metadata"]["Ny"] = Ny
        f["metadata"]["Norb"] = Norb
        f["metadata"]["bps"] = bps
        f["metadata"]["b2ps"] = b2ps
        f["metadata"]["plaq_per_cell"] = plaq_per_cell
        f["metadata"]["U"] = U
        f["metadata"]["t'"] = tp
        f["metadata"]["t''"] = tpp
        f["metadata"]["nflux"] = nflux
        f["metadata"]["twistx"] = twistx
        f["metadata"]["twisty"] = twisty
        f["metadata"]["mu"] = mu
        f["metadata"]["h"] = h
        f["metadata"]["beta"] = L * dt
        f["metadata"]["trans_sym"] = trans_sym
        f["metadata"]["geometry"] = geometry
        f["metadata"]["bc"] = bc

        # parameters used by dqmc code
        f.create_group("params")
        # model parameters
        f["params"]["N"] = np.array(N, dtype=np.int32)
        f["params"]["L"] = np.array(L, dtype=np.int32)
        f["params"]["Nx"] = np.array(Nx, dtype=np.int32)
        f["params"]["Ny"] = np.array(Ny, dtype=np.int32)
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
        f["params"]["meas_pair_bb_only"] = meas_pair_bb_only
        f["params"]["meas_gen_suscept"] = meas_gen_suscept
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
        if meas_gen_suscept:
            f["meas_eqlt"]["uuuu"] = np.zeros(num_ij * num_ij, dtype=dtype_num)
            f["meas_eqlt"]["dddd"] = np.zeros(num_ij * num_ij, dtype=dtype_num)
            f["meas_eqlt"]["dduu"] = np.zeros(num_ij * num_ij, dtype=dtype_num)
            f["meas_eqlt"]["uudd"] = np.zeros(num_ij * num_ij, dtype=dtype_num)
        if meas_chiral:
            f["meas_eqlt"]["chi"] = np.zeros(num_plaq_accum, dtype=dtype_num)

        if meas_energy_corr:
            f["meas_eqlt"]["kk"] = np.zeros(num_bb, dtype=dtype_num)
            f["meas_eqlt"]["kv"] = np.zeros(num_bs, dtype=dtype_num)
            f["meas_eqlt"]["kn"] = np.zeros(num_bs, dtype=dtype_num)
            f["meas_eqlt"]["vv"] = np.zeros(num_ij, dtype=dtype_num)
            f["meas_eqlt"]["vn"] = np.zeros(num_ij, dtype=dtype_num)

        if meas_local_JQ:
            assert trans_sym == 0
            f["meas_eqlt"]["j"] = np.zeros(num_b, dtype=dtype_num)
            f["meas_eqlt"]["jn"] = np.zeros(num_b, dtype=dtype_num)
            f["meas_eqlt"]["j2"] = np.zeros(num_b2, dtype=dtype_num)

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
            if meas_gen_suscept:
                f["meas_uneqlt"]["uuuu"] = np.zeros(
                    num_ij * num_ij * L, dtype=dtype_num
                )
                f["meas_uneqlt"]["dddd"] = np.zeros(
                    num_ij * num_ij * L, dtype=dtype_num
                )
                f["meas_uneqlt"]["dduu"] = np.zeros(
                    num_ij * num_ij * L, dtype=dtype_num
                )
                f["meas_uneqlt"]["uudd"] = np.zeros(
                    num_ij * num_ij * L, dtype=dtype_num
                )
            if meas_pair_bb_only:
                meas_toggle_list = [
                    meas_thermal,
                    meas_bond_corr,
                    meas_2bond_corr,
                    meas_energy_corr,
                    meas_nematic_corr,
                    meas_gen_suscept,
                ]
                assert not any(meas_toggle_list)
                assert not trans_sym
                f["meas_uneqlt"]["pair_bb"] = np.zeros(num_bb * L, dtype=dtype_num)
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
        init_rng = gus.rand_seed_urandom()
    else:
        init_rng = gus.rand_seed_splitmix64(seed)

    if prefix is None:
        prefix = "sim"

    file_0 = "{}_{}.h5".format(prefix, 0)
    file_p = "{}.h5.params".format(prefix)

    create_1(file_sim=file_0, file_params=file_p, init_rng=init_rng, **kwargs)

    with h5py.File(file_p, "r") as f:
        N = f["params"]["N"][...]
        L = f["params"]["L"][...]

    for i in range(1, Nfiles):
        gus.rand_jump(init_rng)
        rng = init_rng.copy()
        init_hs = np.zeros((L, N), dtype=np.int32)

        for l in range(L):
            for r in range(N):
                init_hs[l, r] = gus.rand_uint(rng) >> np.uint64(63)

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

    gus.add_simFile_opts(parser)
    gus.add_meas_opts(parser)

    # parser.add_argument
    args = parser.parse_args()

    argdict = vars(args)

    if args.printout == 1:
        for k, v in argdict.items():
            print(k, v)

    delattr(args, "printout")

    create_batch(**vars(args))
