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

import git #relies on gitpython module
path = os.path.dirname(os.path.abspath(__file__))
repo = git.Repo(path,search_parent_directories=True)
hash_short = repo.git.rev_parse(repo.head, short=True)

def rand_seed_urandom():
    rng = np.zeros(17, dtype=np.uint64)
    rng[:16] = np.frombuffer(os.urandom(16*8), dtype=np.uint64)
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
    JMP = np.array((0x84242f96eca9c41d,
                    0xa3c65b8776f96855, 0x5b34a39f070b5837, 0x4489affce4f31a1e,
                    0x2ffeeb0a48316f40, 0xdc2d9891fe68c022, 0x3659132bb12fea70,
                    0xaac17d8efa43cab8, 0xc4cb815590989b13, 0x5ee975283d71c93b,
                    0x691548c86c1bd540, 0x7910c41d10a1e6a5, 0x0b5fc64563b3e2a8,
                    0x047f7684e9fc949d, 0xb99181f2d8f685ca, 0x284600e3f30e38c3
                    ), dtype=np.uint64)

    t = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        for b in range(64):
            if JMP[i] & (np.uint64(1) << np.uint64(b)):
                for j in range(16):
                    t[j] ^= rng[(np.uint64(j) + rng[16]) & np.uint64(15)]
            rand_uint(rng)

    for j in range(16):
        rng[(np.uint64(j) + rng[16]) & np.uint64(15)] = t[j]


def create_1(file_sim=None, file_params=None, init_rng=None,
    geometry="square", bc=1, Nx=4, Ny=4, mu=0.0, tp=0.0, U=6.0, dt=0.1, L=40, nflux=0, h=0.0,
    overwrite=0, n_delay=16, n_matmul=8, n_sweep_warm=200, n_sweep_meas=2000,
    period_eqlt=8, period_uneqlt=0,trans_sym=1, checkpoint_every=10000,
    meas_bond_corr=0, meas_energy_corr=0, meas_nematic_corr=0,
    meas_thermal=0, meas_2bond_corr=0, meas_chiral=0,
    meas_local_JQ=0):

    assert L % n_matmul == 0 and L % period_eqlt == 0
    if nflux != 0: dtype_num = np.complex128
    else:          dtype_num = np.float64

    if file_sim is None:    file_sim = "sim.h5"
    if file_params is None: file_params = file_sim
    
    one_file = (os.path.abspath(file_sim) == os.path.abspath(file_params))

    if init_rng is None:
        init_rng = rand_seed_urandom()
    rng = init_rng.copy()

    Ncell = Nx*Ny
    
    # placeholder until other boundaries implemented for non square lattices 
    if (bc != 1) & (geometry != "square"):
        raise NotImplementedError("Non-periodic boundaries only implemented for square lattice")
    
    # if non-periodic and trans_sym on, warn user that trans_sym will turn off 
    if (bc != 1) & trans_sym:
        warnings.warn("Non-periodic boundaries not translationally symmetric: turning off trans_sym")
        trans_sym = 0

    if geometry == "square":
        
        Norb = 1
        N = Norb * Nx * Ny

        init_hs = np.zeros((L, N), dtype=np.int32)
        for l in range(L):
            for i in range(N):
                init_hs[l, i] = rand_uint(rng) >> np.uint64(63)

        # Norb site mapping
        if trans_sym:
            map_i = np.zeros(N, dtype=np.int32)
            degen_i = np.array((N,), dtype=np.int32)
        else:
            map_i = np.arange(N, dtype=np.int32)
            degen_i = np.ones(N, dtype=np.int32)
        num_i = map_i.max() + 1
        assert num_i == degen_i.size

        # plaquette definitions NOTE: placeholder
        plaq_per_cell = 1
        num_plaq = plaq_per_cell * Nx*Ny
        plaqs = np.zeros((3, num_plaq), dtype=np.int32)

        # plaq_per_cell slot mapping NOTE: placeholder
        if trans_sym:
            #first Nx*Ny goes to slot 0, second Nx*Ny goes to slot 1
            map_plaq = np.zeros(num_plaq, dtype=np.int32)
            degen_plaq = np.array((Nx*Ny,), dtype=np.int32)
        else:
            map_plaq = np.arange(num_plaq, dtype=np.int32)
            degen_plaq = np.ones(num_plaq, dtype=np.int32)
        
        num_plaq_accum = map_plaq.max() + 1
        
        assert num_plaq_accum == degen_plaq.size

        # 2 site mapping: site r = (x,y) has total (column order) index x + Nx * y
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = N if trans_sym else N*N
        degen_ij = np.zeros(num_ij, dtype=np.int32)
        for jy in range(Ny):
            for jx in range(Nx):
                for iy in range(Ny):
                    for ix in range(Nx):
                        if trans_sym:
                            ky = (iy - jy) % Ny
                            kx = (ix - jx) % Nx
                            k = kx + Nx*ky
                        else:
                            k = (ix + Nx*iy) + N*(jx + Nx*jy)
                        map_ij[jx + Nx*jy, ix + Nx*iy] = k
                        degen_ij[k] += 1
        assert num_ij == map_ij.max() + 1

        # bond definitions: defined by one hopping step
        bps = 4 if tp != 0.0 else 2  # bonds per site
        num_b = bps*N  # total bonds in cluster
        bonds = np.zeros((2, num_b), dtype=np.int32)
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx*iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                bonds[0, i] = i            # i0 = i
                bonds[1, i] = ix1 + Nx*iy  # i1 = i + x
                bonds[0, i + N] = i            # i0 = i
                bonds[1, i + N] = ix + Nx*iy1  # i1 = i + y
                if bps == 4:
                    bonds[0, i + 2*N] = i             # i0 = i
                    bonds[1, i + 2*N] = ix1 + Nx*iy1  # i1 = i + x + y
                    bonds[0, i + 3*N] = ix1 + Nx*iy   # i0 = i + x
                    bonds[1, i + 3*N] = ix + Nx*iy1   # i1 = i + y

        # 1 bond mapping
        if trans_sym:
            # maps bond to degenerate bond type 
            # note that map_b only maps to the first four elements of degen_b, but for 
            # the sake of not having to define a new num_b vairable and allocating different amounts of
            # memory for degen_b in data.c, I'm making degen_b the same size 
            map_b = np.tile(np.arange(bps, dtype=np.int32), (N,1)).T.flatten() # N*bps 
            degen_b = np.ones(num_b, dtype=np.int32) * N  # length N*bps
        else:
            map_b = np.arange(num_b, dtype=np.int32)    # N*bps 
            degen_b = np.ones(num_b, dtype=np.int32)    # length N*bps 

        # 1 bond 1 site mapping
        # Translated to fortran order: [j,istuff] -> [istuff + num_b * j] -> [istuff,j]
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps*N if trans_sym else num_b*N
        degen_bs = np.zeros(num_bs, dtype=np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for ib in range(bps):
                    kk = k + num_ij*ib
                    map_bs[j, i + N*ib] = kk
                    degen_bs[kk] += 1
        assert num_bs == map_bs.max() + 1

        # 1 bond - 1 bond mapping
        # Translated to Fortran order: [jstuff ,istuff] -> [istuff + num_b * jstuff] -> [istuff,jstuff]
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps*bps*N if trans_sym else num_b*num_b
        degen_bb = np.zeros(num_bb, dtype = np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(bps):
                    for ib in range(bps):
                        kk = k + num_ij*(ib + bps*jb)
                        map_bb[j + N*jb, i + N*ib] = kk
                        degen_bb[kk] += 1
        assert num_bb == map_bb.max() + 1

        # 2-bond definition is modified -- NOT consistent with Wen's!
        # Now only bonds defined by two hopping steps.
        b2ps = 12 if tp != 0.0 else 4  # 2-bonds per site
        num_b2 = b2ps*N  # total 2-bonds in cluster
        bond2s = np.zeros((2, num_b2), dtype=np.int32)
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx*iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                iy2 = (iy + 2) % Ny
                ix2 = (ix + 2) % Nx
                # one t^2 path + two t'^2 paths [0,22,23]
                bond2s[0, i + 0*N] = i             # i0 = i
                bond2s[1, i + 0*N] = ix2 + Nx*iy   # i1 = i + 2x   -- /\ \/
                # one t^2 path + two t'^2 paths [1,24,25]
                bond2s[0, i + 1*N] = i            # i0 = i       |  /  \
                bond2s[1, i + 1*N] = ix + Nx*iy2  # i1 = i + 2y  |  \  /
                #two t^2 paths [2,3]
                bond2s[0, i + 2*N] = i             # i0 = i          _|   _ 
                bond2s[1, i + 2*N] = ix1 + Nx*iy1  # i1 = i + x + y      | 
                # two t^2 paths [4,5]
                bond2s[0, i + 3*N] = ix1 + Nx*iy   # i0 = i + x     _
                bond2s[1, i + 3*N] = ix + Nx*iy1   # i1 = i + y      |  |_
                if b2ps == 12:
                    #two t't paths [6,7]
                    bond2s[0, i + 4*N] = i             # i0 = i              _
                    bond2s[1, i + 4*N] = ix2 + Nx*iy1  # i1 = i + 2x + y _/ /
                    #two t't paths [8,9]
                    bond2s[0, i + 5*N] = i              # i0 = i           |   /
                    bond2s[1, i + 5*N] = ix1 + Nx*iy2   # i1 = i + x + 2y /   |
                    #two t't paths [10,11]
                    bond2s[0, i + 6*N] = ix2 + Nx*iy   # i0 = i + 2x _
                    bond2s[1, i + 6*N] = ix + Nx*iy1   # i1 = i + y   \  \_
                    #two t't paths [12,13]
                    bond2s[0, i + 7*N] = ix1 + Nx*iy   # i0 = i + x   |   \
                    bond2s[1, i + 7*N] = ix + Nx*iy2   # i1 = i + 2y   \   |
                    # four t't paths [14,15,16,17]
                    bond2s[0, i + 8*N] = i            # i0 = i      _  _   \  /
                    bond2s[1, i + 8*N] = ix + Nx*iy1  # i1 = i + y  /  \   -  -
                    # four t't paths [18,19,20,21]
                    bond2s[0, i + 9*N] = i            # i0 = i      |\ /| \| |/
                    bond2s[1, i + 9*N] = ix1 + Nx*iy  # i1 = i + x
                    #one t'^2 path [26]
                    bond2s[0, i + 10*N] = i             # i0 = i             /
                    bond2s[1, i + 10*N] = ix2 + Nx*iy2  # i1 = i + 2x + 2y  /
                    #one t'^2 path [27]
                    bond2s[0, i + 11*N] = ix2 + Nx*iy   # i0 = i + 2x   \
                    bond2s[1, i + 11*N] = ix + Nx*iy2   # i1 = i + 2y    \

        # 2 bond mapping
        if trans_sym:
            # note that map_b only maps to the first four elements of degen_b, but for 
            # the sake of not having to define a new num_b vairable and allocating different amounts of
            # memory for degen_b in data.c, I'm making degen_b the same size 
            map_b2 = np.tile(np.arange(b2ps, dtype=np.int32), (N,1)).T.flatten() #length N*b2ps
            degen_b2 = np.ones(num_b2, dtype=np.int32) * N  # length N*b2ps
        else:
            map_b2 = np.arange(num_b2, dtype=np.int32)    # length N*b2ps
            degen_b2 = np.ones(num_b2, dtype=np.int32)    # length N*b2ps 

        # my definition: Bonds defined by two hopping steps
        # Keep track of intermediate point!
        # TODO see if bonds 14 and 16 etc are not equivalent b/c start/end points changed...
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps*N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx*iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                iy2 = (iy + 2) % Ny
                ix2 = (ix + 2) % Nx

                iym1 = (iy - 1) % Ny
                ixm1 = (ix - 1) % Nx

                #t^2 terms: NN + NN
                hop2s[0, i] = i             # i0 = i
                hop2s[1, i] = ix1 + Nx*iy   # i1 = i + x      --
                hop2s[2, i] = ix2 + Nx*iy   # i2 = i + 2x
                #-----------------
                hop2s[0, i + N] = i            # i0 = i         
                hop2s[1, i + N] = ix + Nx*iy1  # i1 = i + y     |
                hop2s[2, i + N] = ix + Nx*iy2  # i2 = i + 2y    |
                #-----------------
                hop2s[0, i + 2*N] = i             # i0 = i               _|
                hop2s[1, i + 2*N] = ix1 + Nx*iy   # i1 = i + x           
                hop2s[2, i + 2*N] = ix1 + Nx*iy1  # i2 = i + x + y

                hop2s[0, i + 3*N] = i            # i0 = i               _  
                hop2s[1, i + 3*N] = ix + Nx*iy1  # i1 = i + y          |
                hop2s[2, i + 3*N] = ix1 + Nx*iy1  # i2 = i + x + y
                #-----------------
                hop2s[0, i + 4*N] = ix1 + Nx*iy   # i0 = i + x            _ 
                hop2s[1, i + 4*N] = ix1 + Nx*iy1  # i1 = i + x + y         |
                hop2s[2, i + 4*N] = ix + Nx*iy1   # i2 = i + y

                hop2s[0, i + 5*N] = ix1 + Nx*iy   # i0 = i + x           |_
                hop2s[1, i + 5*N] = i             # i1 = i                
                hop2s[2, i + 5*N] = ix + Nx*iy1   # i2 = i + y

                if hop2ps == 28:
                    # t*t' terms: NN + NNN or NNN + NN
                    hop2s[0, i + 6*N] = i             # i0 = i            
                    hop2s[1, i + 6*N] = ix1 + Nx*iy   # i1 = i + x         _/
                    hop2s[2, i + 6*N] = ix2 + Nx*iy1  # i2 = i + 2x + y    

                    hop2s[0, i + 7*N] = i             # i0 = i
                    hop2s[1, i + 7*N] = ix1 + Nx*iy1  # i1 = i + x + y     _
                    hop2s[2, i + 7*N] = ix2 + Nx*iy1  # i2 = i + 2x + y   /
                    #------------------
                    hop2s[0, i + 8*N] = i              # i0 = i             |
                    hop2s[1, i + 8*N] = ix1 + Nx*iy1   # i1 = i + x + y    /
                    hop2s[2, i + 8*N] = ix1 + Nx*iy2   # i2 = i + x + 2y

                    hop2s[0, i + 9*N] = i              # i0 = i             / 
                    hop2s[1, i + 9*N] = ix  + Nx*iy1   # i1 = i + y        |
                    hop2s[2, i + 9*N] = ix1 + Nx*iy2   # i2 = i + x + 2y
                    #------------------
                    hop2s[0, i + 10*N] = ix2 + Nx*iy    # i0 = i + 2x
                    hop2s[1, i + 10*N] = ix1 + Nx*iy1   # i1 = i + x + y    _
                    hop2s[2, i + 10*N] = ix  + Nx*iy1   # i2 = i + y         \

                    hop2s[0, i + 11*N] = ix2 + Nx*iy    # i0 = i + 2x     
                    hop2s[1, i + 11*N] = ix1 + Nx*iy    # i1 = i + x       \_
                    hop2s[2, i + 11*N] = ix  + Nx*iy1   # i2 = i + y        
                    #------------------
                    hop2s[0, i + 12*N] = ix1 + Nx*iy    # i0 = i + x
                    hop2s[1, i + 12*N] = ix  + Nx*iy1   # i1 = i + y       |
                    hop2s[2, i + 12*N] = ix + Nx*iy2    # i2 = i + 2y       \

                    hop2s[0, i + 13*N] = ix1 + Nx*iy     # i0 = i + x
                    hop2s[1, i + 13*N] = ix1 + Nx*iy1    # i1 = i + x + y     \ 
                    hop2s[2, i + 13*N] = ix + Nx*iy2     # i2 = i + 2y         |
                    #------------------
                    hop2s[0, i + 14*N] = i             # i0 = i         
                    hop2s[1, i + 14*N] = ix1 + Nx*iy1  # i1 = i + x + y     _
                    hop2s[2, i + 14*N] = ix  + Nx*iy1  # i2 = i + y         /

                    hop2s[0, i + 15*N] = i             # i0 = i         
                    hop2s[1, i + 15*N] = ix1 + Nx*iy   # i1 = i + x         \
                    hop2s[2, i + 15*N] = ix  + Nx*iy1  # i2 = i + y         -

                    hop2s[0, i + 16*N] = i               # i0 = i           _
                    hop2s[1, i + 16*N] = ixm1 + Nx*iy1   # i1 = i - x + y   \
                    hop2s[2, i + 16*N] = ix  + Nx*iy1    # i2 = i + y    

                    hop2s[0, i + 17*N] = i             # i0 = i       
                    hop2s[1, i + 17*N] = ixm1 + Nx*iy  # i1 = i - x         /
                    hop2s[2, i + 17*N] = ix  + Nx*iy1  # i2 = i + y         -
                    #------------------
                    hop2s[0, i + 18*N] = i             # i0 = i         
                    hop2s[1, i + 18*N] = ix1 + Nx*iy1  # i1 = i + x + y     /|
                    hop2s[2, i + 18*N] = ix1 + Nx*iy   # i2 = i + x     

                    hop2s[0, i + 19*N] = i             # i0 = i         
                    hop2s[1, i + 19*N] = ix  + Nx*iy1  # i1 = i + y     |\
                    hop2s[2, i + 19*N] = ix1 + Nx*iy   # i2 = i + x   

                    hop2s[0, i + 20*N] = i             # i0 = i       
                    hop2s[1, i + 20*N] = ix1 + Nx*iym1 # i1 = i + x - y     \|
                    hop2s[2, i + 20*N] = ix1 + Nx*iy   # i2 = i + x   

                    hop2s[0, i + 21*N] = i             # i0 = i   
                    hop2s[1, i + 21*N] = ix  + Nx*iym1 # i1 = i - y      |/
                    hop2s[2, i + 21*N] = ix1 + Nx*iy   # i2 = i + x     
                    #(t'^2) terms: NNN + NNN
                    hop2s[0, i + 22*N] = i             # i0 = i         
                    hop2s[1, i + 22*N] = ix1 + Nx*iy1  # i1 = i + x + y   /\
                    hop2s[2, i + 22*N] = ix2 + Nx*iy   # i2 = i + 2x        

                    hop2s[0, i + 23*N] = i             # i0 = i            
                    hop2s[1, i + 23*N] = ix1 + Nx*iym1 # i1 = i + x - y   \/
                    hop2s[2, i + 23*N] = ix2 + Nx*iy   # i2 = i + 2x    
                    #------------------
                    hop2s[0, i + 24*N] = i              # i0 = i         
                    hop2s[1, i + 24*N] = ix1 + Nx*iy1   # i1 = i + x + y    \
                    hop2s[2, i + 24*N] = ix + Nx*iy2    # i2 = i + 2y       /

                    hop2s[0, i + 25*N] = i             # i0 = i            /
                    hop2s[1, i + 25*N] = ixm1 + Nx*iy1 # i1 = i - x + y    \
                    hop2s[2, i + 25*N] = ix + Nx*iy2   # i2 = i + 2y    
                    #------------------
                    hop2s[0, i + 26*N] = i             # i0 = i              /   
                    hop2s[1, i + 26*N] = ix1 + Nx*iy1  # i1 = i + x + y     / 
                    hop2s[2, i + 26*N] = ix2 + Nx*iy2  # i2 = i + 2x + 2y    
                    #------------------
                    hop2s[0, i + 27*N] = ix2 + Nx*iy    # i0 = i + 2x       \
                    hop2s[1, i + 27*N] = ix1 + Nx*iy1   # i1 = i + x + y     \ 
                    hop2s[2, i + 27*N] = ix + Nx*iy2    # i2 = i + 2y

        #how bond2s and hop2s are related
        bond_hop_dict = {}
        if b2ps == 4:
            bond_hop_dict[0] = [0]
            bond_hop_dict[1] = [1]
            bond_hop_dict[2] = [2,3]
            bond_hop_dict[3] = [4,5]
        else:
            bond_hop_dict[0] = [0,22,23];
            bond_hop_dict[1] = [1,24,25];
            for i in range(2,8):
                bond_hop_dict[i] = [2*i-2,2*i-1];
            bond_hop_dict[8] = [14,15,16,17]
            bond_hop_dict[9] = [18,19,20,21]
            bond_hop_dict[10] = [26]
            bond_hop_dict[11] = [27]

        # 2 2-bond mapping
        # Translated to Fortran order: [jstuff ,istuff] -> [istuff + num_b2 * jstuff] -> [istuff,jstuff]
        num_b2b2 = b2ps*b2ps*N if trans_sym else num_b2*num_b2
        map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
        degen_b2b2 = np.zeros(num_b2b2, dtype = np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(b2ps):
                    for ib in range(b2ps):
                        kk = k + num_ij*(ib + b2ps*jb)
                        map_b2b2[j + N*jb, i + N*ib] = kk
                        degen_b2b2[kk] += 1
        assert num_b2b2 == map_b2b2.max() + 1

        # bond 2-bond mapping
        num_bb2 = bps*b2ps*N if trans_sym else num_b*num_b2
        map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
        degen_bb2 = np.zeros(num_bb2, dtype = np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(bps):
                    for ib in range(b2ps):
                        kk = k + num_ij*(ib + b2ps*jb)
                        map_bb2[j + N*jb, i + N*ib] = kk
                        degen_bb2[kk] += 1
        assert num_bb2 == map_bb2.max() + 1

        # 2-bond bond mapping
        num_b2b = b2ps*bps*N if trans_sym else num_b2*num_b
        map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
        degen_b2b = np.zeros(num_b2b, dtype = np.int32)
        for j in range(N):
            for i in range(N):
                k = map_ij[j, i]
                for jb in range(b2ps):
                    for ib in range(bps):
                        kk = k + num_ij*(ib + bps*jb)
                        map_b2b[j + N*jb, i + N*ib] = kk
                        degen_b2b[kk] += 1
        assert num_b2b == map_b2b.max() + 1

        if bc == 1:
            kij,peierls = tight_binding.H_periodic_square(Nx,Ny,t=1,tp=tp,nflux=nflux,alpha=1/2)
        elif bc == 2:
            kij,peierls = tight_binding.H_open_square(Nx,Ny,t=1,tp=tp,nflux=nflux,alpha=1/2)
        else:
            raise ValueError("Invalid bc choice, must be 1 for periodic or 2 for open")
        #phases accumulated by two-hop processes
        #Here: types 0,1 include t' factors
        #   Types 2-7: sum of two paths
        #   Types 8,9: sum of four paths
        #   Types 10,11: one path each
        #   ZF case: 1+2*tp, 1+2*tp, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1
        thermal_phases = np.ones((b2ps, N),dtype=np.complex128)
        for i in range(N):
            for btype in range(b2ps):
                i0 = bond2s[0, i + btype*N] #start
                i2 = bond2s[1, i + btype*N] #end
                pp = 0; 
                #list of intermediate pointscorresponding to this bond
                i1_type_list = bond_hop_dict[btype]
                #print("btype = ",btype,"i1_type_list = ",i1_type_list)
                #two bonds need manual weighting when t' != 0: 
                if b2ps == 12 and (btype == 0 or btype == 1):
                    i1 = hop2s[1, i + i1_type_list[0]*N]
                    #print(f"i = {i}, btype = {btype}, path ({i0},{i1},{i2})")
                    pp += peierls[i0,i1] * peierls[i1,i2]
                    for i1type in i1_type_list[1:]:
                        i1 = hop2s[1, i + i1type*N]
                        pp += tp*tp*peierls[i0,i1] * peierls[i1,i2]
                    # print(pp)
                #general case
                else:
                    for i1type in i1_type_list:
                        i1 = hop2s[1, i + i1type*N]
                        pp += peierls[i0,i1] * peierls[i1,i2]
                thermal_phases[btype,i] = pp

    elif geometry == "triangular":
        Norb = 1
        N = Norb * Nx * Ny

        init_hs = np.zeros((L, N), dtype=np.int32)
        for l in range(L):
            for i in range(N):
                init_hs[l, i] = rand_uint(rng) >> np.uint64(63)

        # 1 site mapping
        if trans_sym:
            map_i = np.zeros(N, dtype=np.int32)
            degen_i = np.array((N,), dtype=np.int32)
        else:
            map_i = np.arange(N, dtype=np.int32)
            degen_i = np.ones(N, dtype=np.int32)
        num_i = map_i.max() + 1
        assert num_i == degen_i.size

        # print("Trans sym = ",trans_sym)
        # print("map",map_i,"degen",degen_i,"num",num_i)

        # plaquette definitions NOTE: only valid for t' = 0
        plaq_per_cell = 2
        num_plaq = plaq_per_cell * Nx*Ny
        plaqs = np.zeros((3, num_plaq), dtype=np.int32)
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx*iy
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                plaqs[0, i] = i             # i0 = i
                plaqs[1, i] = ix1 + Nx*iy   # i1 = i + x
                plaqs[2, i] = ix  + Nx*iy1  # i2 = i + y // counterclockwise
                plaqs[0, i + Nx*Ny] = ix1 + Nx*iy   # i0 = i + x
                plaqs[1, i + Nx*Ny] = ix1 + Nx*iy1  # i1 = i + x + y
                plaqs[2, i + Nx*Ny] = ix  + Nx*iy1  # i2 = i + x //counterclockwise


        #print("total number of triangular plaqs:",num_plaq)
        # 1 plaquette mapping
        if trans_sym:
            #first Nx*Ny goes to slot 0, second Nx*Ny goes to slot 1
            map_plaq = np.zeros(num_plaq, dtype=np.int32)
            map_plaq[Nx*Ny:] = 1
            degen_plaq = np.array((Nx*Ny,Nx*Ny), dtype=np.int32)
        else:
            map_plaq = np.arange(num_plaq, dtype=np.int32)
            degen_plaq = np.ones(num_plaq, dtype=np.int32)
        
        num_plaq_accum = map_plaq.max() + 1
        
        assert num_plaq_accum == degen_plaq.size
        # print("map:",map_plaq)
        # print("degen:",degen_plaq, degen_plaq.size)
        # print("save slots:",num_plaq_accum)

        # 2 site mapping 
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = N if trans_sym else N*N
        degen_ij = np.zeros(num_ij, dtype=np.int32)
        for jy in range(Ny):
            for jx in range(Nx):
                for iy in range(Ny):
                    for ix in range(Nx):
                        if trans_sym:
                            ky = (iy - jy) % Ny
                            kx = (ix - jx) % Nx
                            k = kx + Nx*ky
                        else:
                            k = (ix + Nx*iy) + N*(jx + Nx*jy)
                        map_ij[jx + Nx*jy, ix + Nx*iy] = k
                        degen_ij[k] += 1
        assert num_ij == map_ij.max() + 1

        # bond definitions: defined by one hopping step NOTE: placeholder
        bps = 6 if tp != 0.0 else 3  # bonds per site
        num_b = bps*N  # total bonds in cluster
        bonds = np.zeros((2, num_b), dtype=np.int32)

        # 1 bond mapping NOTE: placeholder
        map_b = np.zeros(num_b, dtype=np.int32)    # N*bps 
        degen_b = np.zeros(num_b, dtype=np.int32)    # length N*bps 

        # 1 bond 1 site mapping NOTE: placeholder
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps*N if trans_sym else num_b*N
        degen_bs = np.zeros(num_bs, dtype=np.int32)

        # 1 bond - 1 bond mapping NOTE: placeholder
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps*bps*N if trans_sym else num_b*num_b
        degen_bb = np.zeros(num_bb, dtype = np.int32)

        # 2-bond definition is modified -- NOT consistent with Wen's!
        # Now only bonds defined by two hopping steps.
        # NOTE: placeholder
        b2ps = 12 if tp != 0.0 else 4  # 2-bonds per site
        num_b2 = b2ps*N  # total 2-bonds in cluster
        bond2s = np.zeros((2, num_b2), dtype=np.int32)

        # 2 bond mapping NOTE: placeholder
        map_b2 = np.zeros(num_b2, dtype=np.int32)    # N*b2ps 
        degen_b2 = np.zeros(num_b2, dtype=np.int32)    # length N*b2ps 

        # my definition: Bonds defined by two hopping steps
        # NOTE: placeholder
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps*N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)

        # 2 2-bond mapping NOTE: placeholder
        num_b2b2 = b2ps*b2ps*N if trans_sym else num_b2*num_b2
        map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
        degen_b2b2 = np.zeros(num_b2b2, dtype = np.int32)

        # bond 2-bond mapping NOTE: placeholder
        num_bb2 = bps*b2ps*N if trans_sym else num_b*num_b2
        map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
        degen_bb2 = np.zeros(num_bb2, dtype = np.int32)

        # 2-bond bond mapping NOTE: placeholder
        num_b2b = b2ps*bps*N if trans_sym else num_b2*num_b
        map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
        degen_b2b = np.zeros(num_b2b, dtype = np.int32)

        kij,peierls = tight_binding.H_periodic_triangular(Nx,Ny,t=1,tp=tp,nflux=nflux,alpha=1/2)

        # NOTE: placeholder
        thermal_phases = np.ones((b2ps, N),dtype=np.complex128)

    elif geometry == "honeycomb":
        Norb = 2
        #NOTE: N = total number of orbitals, not total number of unit cells
        N = Norb * Nx * Ny 

        #location (ix, iy) orbital io is 3d matrix (ix,iy,io)
        # with total index ix + Nx * iy + (Nx * Ny) * i0
        init_hs = np.zeros((L, N), dtype=np.int32)
        for l in range(L):
            for i in range(N):
                init_hs[l, i] = rand_uint(rng) >> np.uint64(63)

        # 1 site mapping
        if trans_sym:
            map_i = np.zeros(N, dtype=np.int32)
            map_i[Ny*Nx:] = 1 #second orbital
            degen_i = np.array((Ny*Nx, Ny*Nx), dtype=np.int32)
        else:
            map_i = np.arange(N, dtype=np.int32)
            degen_i = np.ones(N, dtype=np.int32)
        num_i = map_i.max() + 1
        assert num_i == degen_i.size


        # plaquette definitions NOTE: placeholder
        plaq_per_cell = 1
        num_plaq = plaq_per_cell * Nx * Ny
        plaqs = np.zeros((3, num_plaq), dtype=np.int32)

        # 1 plaquette mapping NOTE: placeholder
        if trans_sym:
            #first Nx*Ny gioes to slot 0, second Nx*Ny goes to slot 1
            map_plaq = np.zeros(num_plaq, dtype=np.int32)
            degen_plaq = np.array((Nx*Ny,), dtype=np.int32)
        else:
            map_plaq = np.arange(num_plaq, dtype=np.int32)
            degen_plaq = np.ones(num_plaq, dtype=np.int32)
        
        num_plaq_accum = map_plaq.max() + 1
        
        assert num_plaq_accum == degen_plaq.size

        # print("Trans sym = ",trans_sym)
        # print("map",map_plaq,"degen",degen_plaq,"num",num_plaq_accum)

        # 2 site mapping
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = Norb*Norb*Ny*Nx if trans_sym else N*N
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
                                    #total column index of matrix index [kx,ky,io,jo]
                                    k = kx + Nx*ky + Nx*Ny*io + N*jo
                                else:
                                    #total column index of matrix index [ix,iy,io,jx,jy,jo]
                                    k = (ix + Nx*iy + Nx*Ny*io) + N*(jx + Nx*jy + Nx*Ny*jo)
                                map_ij[jx + Nx*jy + Nx*Ny*jo, ix + Nx*iy + Nx*Ny*io] = k
                                degen_ij[k] += 1
        assert num_ij == map_ij.max() + 1

        # print("Trans sym = ",trans_sym)
        # print("map",map_ij,map_ij.shape,"degen",degen_ij,degen_ij.shape,"num",num_ij)

        # bond definitions: defined by one hopping step NOTE: placeholder
        bps = 3 if tp != 0.0 else 9  # bonds per site (not per orbital!)
        num_b = bps*Nx*Ny  # total bonds in cluster
        bonds = np.zeros((2, num_b), dtype=np.int32)

        # 1 bond mapping NOTE: placeholder
        map_b = np.zeros(num_b, dtype=np.int32)    # N*bps 
        degen_b = np.zeros(num_b, dtype=np.int32)    # length N*bps 

        # 1 bond 1 site mapping NOTE: placeholder
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps*N if trans_sym else num_b*N
        degen_bs = np.zeros(num_bs, dtype=np.int32)

        # 1 bond - 1 bond mapping NOTE: placeholder
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps*bps*N if trans_sym else num_b*num_b
        degen_bb = np.zeros(num_bb, dtype = np.int32)

        # 2-bond definition is modified -- NOT consistent with Wen's!
        # Now only bonds defined by two hopping steps.
        # NOTE: placeholder
        b2ps = 12 if tp != 0.0 else 4  # 2-bonds per site
        num_b2 = b2ps*N  # total 2-bonds in cluster
        bond2s = np.zeros((2, num_b2), dtype=np.int32)

        # 2 bond mapping NOTE: placeholder
        map_b2 = np.zeros(num_b2, dtype=np.int32)    # N*b2ps 
        degen_b2 = np.zeros(num_b2, dtype=np.int32)    # length N*b2ps 

        # my definition: Bonds defined by two hopping steps
        # NOTE: placeholder
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps*N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)

        # 2 2-bond mapping NOTE: placeholder
        num_b2b2 = b2ps*b2ps*N if trans_sym else num_b2*num_b2
        map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
        degen_b2b2 = np.zeros(num_b2b2, dtype = np.int32)

        # bond 2-bond mapping NOTE: placeholder
        num_bb2 = bps*b2ps*N if trans_sym else num_b*num_b2
        map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
        degen_bb2 = np.zeros(num_bb2, dtype = np.int32)

        # 2-bond bond mapping NOTE: placeholder
        num_b2b = b2ps*bps*N if trans_sym else num_b2*num_b
        map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
        degen_b2b = np.zeros(num_b2b, dtype = np.int32)

        kij,peierls = tight_binding.H_periodic_honeycomb(Nx,Ny,t=1,tp=tp,nflux=nflux,alpha=1/2)
        
        #phases accumulated by two-hop processes NOTE: placeholder
        thermal_phases = np.ones((b2ps, N),dtype=np.complex128)

    elif geometry == "kagome":
        Norb=3
        #NOTE: N = total number of orbitals, not total number of unit cells
        N = Norb * Nx*Ny 
        #location (ix, iy) orbital io is 3d matrix (ix,iy,io)
        # with total index ix + Nx * iy + (Nx*Ny) * i0
        init_hs = np.zeros((L, N), dtype=np.int32)
        for l in range(L):
            for i in range(N):
                init_hs[l, i] = rand_uint(rng) >> np.uint64(63)

        # 1 site mapping
        if trans_sym:
            map_i = np.zeros(N, dtype=np.int32)
            map_i[Ny*Nx:2*Ny*Nx] = 1 #second orbital
            map_i[2*Ny*Nx:] = 2 #third orbital
            degen_i = np.array((Ny*Nx, Ny*Nx, Ny*Nx), dtype=np.int32)
        else:
            map_i = np.arange(N, dtype=np.int32)
            degen_i = np.ones(N, dtype=np.int32)
        num_i = map_i.max() + 1
        assert num_i == degen_i.size

        # plaquette definitions TODO: check correctness
        plaq_per_cell = 2
        num_plaq = plaq_per_cell * Nx*Ny
        plaqs = np.zeros((3, num_plaq), dtype=np.int32)
        for iy in range(Ny):
            for ix in range(Nx):
                i = ix + Nx*iy 
                iy1 = (iy + 1) % Ny
                ix1 = (ix + 1) % Nx
                plaqs[0, i] = i             # i0 = i(A)
                plaqs[1, i] = i + Nx*Ny * 2 # i1 = i(C)
                plaqs[2, i] = i + Nx*Ny * 1 # i2 = i(B) // counterclockwise
                plaqs[0, i + Nx*Ny] = i                        # i0 = i(A)
                plaqs[1, i + Nx*Ny] = ix1 + Nx*iy + Nx*Ny * 2  # i1 = i+x(C)
                plaqs[2, i + Nx*Ny] = ix  + Nx*iy1 + Nx*Ny * 1 # i2 = i+y(B) //counterclockwise

        # 1 plaquette mapping 
        if trans_sym:
            #first Nx*Ny gioes to slot 0, second Nx*Ny goes to slot 1
            map_plaq = np.zeros(num_plaq, dtype=np.int32)
            map_plaq[Nx*Ny:] = 1
            degen_plaq = np.array((Nx*Ny,Nx*Ny), dtype=np.int32)
        else:
            map_plaq = np.arange(num_plaq, dtype=np.int32)
            degen_plaq = np.ones(num_plaq, dtype=np.int32)
        
        num_plaq_accum = map_plaq.max() + 1
        
        assert num_plaq_accum == degen_plaq.size

        # print("Trans sym = ",trans_sym)
        # print("map",map_plaq,"degen",degen_plaq,"num",num_plaq_accum)

        # 2 site mapping
        map_ij = np.zeros((N, N), dtype=np.int32)
        num_ij = Norb*Norb*Ny*Nx if trans_sym else N*N
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
                                    #total column index of matrix index [kx,ky,io,jo]
                                    k = kx + Nx*ky + Nx*Ny*io + Nx*Ny*Norb*jo
                                else:
                                    #total column index of matrix index [ix,iy,io,jx,jy,jo]
                                    k = (ix + Nx*iy + Nx*Ny*io) + N*(jx + Nx*jy + Nx*Ny*jo)
                                map_ij[jx + Nx*jy + Nx*Ny*jo, ix + Nx*iy + Nx*Ny*io] = k
                                degen_ij[k] += 1
        assert num_ij == map_ij.max() + 1

        # bond definitions: defined by one hopping step NOTE: placeholder
        bps = 2 if tp != 0.0 else 4  # bonds per site
        num_b = bps*N  # total bonds in cluster
        bonds = np.zeros((2, num_b), dtype=np.int32)

        # 1 bond mapping NOTE: placeholder
        map_b = np.zeros(num_b, dtype=np.int32)    # N*bps 
        degen_b = np.zeros(num_b, dtype=np.int32)    # length N*bps 

        # 1 bond 1 site mapping NOTE: placeholder
        map_bs = np.zeros((N, num_b), dtype=np.int32)
        num_bs = bps*N if trans_sym else num_b*N
        degen_bs = np.zeros(num_bs, dtype=np.int32)

        # 1 bond - 1 bond mapping NOTE: placeholder
        map_bb = np.zeros((num_b, num_b), dtype=np.int32)
        num_bb = bps*bps*N if trans_sym else num_b*num_b
        degen_bb = np.zeros(num_bb, dtype = np.int32)

        # 2-bond definition is modified -- NOT consistent with Wen's!
        # Now only bonds defined by two hopping steps.
        # NOTE: placeholder
        b2ps = 12 if tp != 0.0 else 4  # 2-bonds per site
        num_b2 = b2ps*N  # total 2-bonds in cluster
        bond2s = np.zeros((2, num_b2), dtype=np.int32)

        # 2 bond mapping NOTE: placeholder
        map_b2 = np.zeros(num_b2, dtype=np.int32)    # N*b2ps 
        degen_b2 = np.zeros(num_b2, dtype=np.int32)    # length N*b2ps 

        # my definition: Bonds defined by two hopping steps
        # NOTE: placeholder
        hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
        num_hop2 = hop2ps*N  # total 2-hop-bonds in cluster
        hop2s = np.zeros((3, num_hop2), dtype=np.int32)

        # 2 2-bond mapping NOTE: placeholder
        num_b2b2 = b2ps*b2ps*N if trans_sym else num_b2*num_b2
        map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
        degen_b2b2 = np.zeros(num_b2b2, dtype = np.int32)

        # bond 2-bond mapping NOTE: placeholder
        num_bb2 = bps*b2ps*N if trans_sym else num_b*num_b2
        map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
        degen_bb2 = np.zeros(num_bb2, dtype = np.int32)

        # 2-bond bond mapping NOTE: placeholder
        num_b2b = b2ps*bps*N if trans_sym else num_b2*num_b
        map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
        degen_b2b = np.zeros(num_b2b, dtype = np.int32)

        kij,peierls = tight_binding.H_periodic_kagome(Nx,Ny,t=1,tp=tp,nflux=nflux,alpha=1/2)

        #phases accumulated by two-hop processes NOTE: placeholder
        thermal_phases = np.ones((b2ps, N),dtype=np.complex128)

        
    #account for different data type when nflux=0
    thermal_phases = thermal_phases if nflux !=0 else thermal_phases.real
    Ku = kij if nflux != 0 else kij.real
    peierls = peierls if nflux !=0 else peierls.real
    
    #Zeeman interaction
    Kd = Ku.copy()
    for i in range(Ny*Nx):
        Ku[i, i] -= (mu - h)
        Kd[i, i] -= (mu + h)

    exp_Ku = expm(-dt * Ku)
    inv_exp_Ku = expm(dt * Ku)
    exp_halfKu = expm(-dt/2 * Ku)
    inv_exp_halfKu = expm(dt/2 * Ku)

    exp_Kd = expm(-dt * Kd)
    inv_exp_Kd = expm(dt * Kd)
    exp_halfKd = expm(-dt/2 * Kd)
    inv_exp_halfKd = expm(dt/2 * Kd)

    U_i = U*np.ones_like(degen_i, dtype=np.float64)
    assert U_i.shape[0] == num_i

    exp_lmbd = np.exp(0.5*U_i*dt) + np.sqrt(np.expm1(U_i*dt))
    exp_lambda = np.array((exp_lmbd[map_i]**-1, exp_lmbd[map_i]))
    delll = np.array((exp_lmbd[map_i]**2 - 1, exp_lmbd[map_i]**-2 - 1))

    with h5py.File(file_params, "w" if overwrite else "x") as f:
        # parameters not used by dqmc code, but useful for analysis
        f.create_group("metadata")
        f["metadata"]["commit"] = hash_short
        f["metadata"]["version"] = 0.1
        f["metadata"]["model"] = \
            "Hubbard (complex)" if dtype_num == np.complex128 else "Hubbard"
        f["metadata"]["Nx"] = Nx
        f["metadata"]["Ny"] = Ny
        f["metadata"]["Norb"] = Norb
        f["metadata"]["bps"] = bps
        f["metadata"]["b2ps"] = b2ps
        f["metadata"]["plaq_per_cell"] = plaq_per_cell
        f["metadata"]["U"] = U
        f["metadata"]["t'"] = tp
        f["metadata"]["nflux"] = nflux
        f["metadata"]["mu"] = mu
        f["metadata"]["h"] = h
        f["metadata"]["beta"] = L*dt
        f["metadata"]["trans_sym"] = trans_sym
        f["metadata"]["geometry"] = geometry
        f["metadata"]["bc"] = bc

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
        f["params"]["dt"] = np.array(dt, dtype=np.float64) # not actually used

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
        f["params"]["num_plaq"] = num_plaq
        f["params"]["num_b"] = num_b
        f["params"]["num_b2"] = num_b2
        f["params"]["num_bs"] = num_bs
        f["params"]["num_bb"] = num_bb
        f["params"]["num_b2b"] = num_b2b
        f["params"]["num_bb2"] = num_bb2
        f["params"]["num_b2b2"] = num_b2b2
        f["params"]["degen_i"] = degen_i
        f["params"]["degen_ij"] = degen_ij
        f["params"]["degen_plaq"] = degen_plaq
        f["params"]["degen_b"] = degen_b
        f["params"]["degen_b2"] = degen_b2
        f["params"]["degen_bs"] = degen_bs
        f["params"]["degen_bb"] = degen_bb
        f["params"]["degen_bb2"] = degen_bb2
        f["params"]["degen_b2b"] = degen_b2b
        f["params"]["degen_b2b2"] = degen_b2b2
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
        f["params"]["F"] = np.array(L//n_matmul, dtype=np.int32)
        f["params"]["n_sweep"] = np.array(n_sweep_warm + n_sweep_meas,
                                          dtype=np.int32)

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
        
        if meas_local_JQ:
            f["meas_eqlt"]["j"]  = np.zeros(num_b,  dtype=dtype_num)
            f["meas_eqlt"]["jn"] = np.zeros(num_b,  dtype=dtype_num)
            f["meas_eqlt"]["j2"] = np.zeros(num_b2, dtype=dtype_num)

        if period_uneqlt > 0:
            f.create_group("meas_uneqlt")
            f["meas_uneqlt"]["n_sample"] = np.array(0, dtype=np.int32)
            f["meas_uneqlt"]["sign"] = np.array(0.0, dtype=dtype_num)
            f["meas_uneqlt"]["gt0"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["gt0_u"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["gt0_d"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["nn"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["xx"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["zz"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["pair_sw"] = np.zeros(num_ij*L, dtype=dtype_num)
            if meas_bond_corr:
                f["meas_uneqlt"]["pair_bb"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["jj"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["jsjs"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["kk"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["ksks"] = np.zeros(num_bb*L, dtype=dtype_num)
            if meas_thermal:
                f["meas_uneqlt"]["j2jn"] = np.zeros(num_b2b*L, dtype=dtype_num) 
                f["meas_uneqlt"]["jnj2"] = np.zeros(num_bb2*L, dtype=dtype_num) 
                f["meas_uneqlt"]["jnjn"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["jjn"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["jnj"] = np.zeros(num_bb*L, dtype=dtype_num)
            if meas_2bond_corr:
                #use j2j2 should correspond to J2J2 results after summation
                f["meas_uneqlt"]["j2j2"] = np.zeros(num_b2b2*L, dtype=dtype_num)
                #use j2j should correspond to J2j results after summation
                f["meas_uneqlt"]["j2j"] = np.zeros(num_b2b*L, dtype=dtype_num) #new
                #use jj2 should correspond to jJ2 results after summation
                f["meas_uneqlt"]["jj2"] = np.zeros(num_bb2*L, dtype=dtype_num) #new
                #these below are not implemented with phases currently
                # f["meas_uneqlt"]["pair_b2b2"] = np.zeros(num_b2b2*L, dtype=dtype_num)
                # f["meas_uneqlt"]["js2js2"] = np.zeros(num_b2b2*L, dtype=dtype_num)
                # f["meas_uneqlt"]["k2k2"] = np.zeros(num_b2b2*L, dtype=dtype_num)
                # f["meas_uneqlt"]["ks2ks2"] = np.zeros(num_b2b2*L, dtype=dtype_num)
            if meas_energy_corr:
                f["meas_uneqlt"]["kv"] = np.zeros(num_bs*L, dtype=dtype_num)
                f["meas_uneqlt"]["kn"] = np.zeros(num_bs*L, dtype=dtype_num)
                f["meas_uneqlt"]["vv"] = np.zeros(num_ij*L, dtype=dtype_num)
                f["meas_uneqlt"]["vn"] = np.zeros(num_ij*L, dtype=dtype_num)
            if meas_nematic_corr:
                f["meas_uneqlt"]["nem_nnnn"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["nem_ssss"] = np.zeros(num_bb*L, dtype=dtype_num)


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
    print("created simulation files:",
          file_0 if Nfiles == 1 else "{} ... {}".format(file_0, file_i))
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
    parser = argparse.ArgumentParser(description="Generate .h5 files for dqmc simulation",\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-V','--version', action='version', version=hash_short)

    group1 = parser.add_argument_group('Physics parameters')
    group1.add_argument('--geometry', choices=['square', 'triangular', 'honeycomb', 'kagome'], type=str, default="square");
    group1.add_argument('--Nx',    type=int,   default = 4,   metavar='X',help="Number of lattice sites along x direction");
    group1.add_argument('--Ny',    type=int,   default = 4,   metavar='X',help="Number of lattice sites along y direction");
    group1.add_argument('--tp',    type=float, default = 0.0, metavar='X',help="Next nearest hopping integral");
    group1.add_argument('--nflux', type=int,   default = 0,   metavar='X',help="Number of flux threading the cluster");
    group1.add_argument('--U',     type=float, default = 6.0, metavar='X',help="On-site Hubbard repulsion strength");
    group1.add_argument('--bc',    type=int,   default = 1,   metavar='X',help="Boundary conditions, 1 for periodic, 2 for open");

    group1.add_argument('--dt',    type=float, default = 0.1, metavar='X',help="Imaginary time discretization interval");
    group1.add_argument('--L',     type=int,   default = 40,  metavar='X',help="Number of imaginary time steps");
    
    group1.add_argument('--mu',    type=float, default = 0.0, metavar='X',help="chemical potential");
    group1.add_argument('--h',     type=float, default = 0.0, metavar='X',help="Zeeman field strength. Down electrons feel net (mu+h) chemical potential");

    group2 = parser.add_argument_group('Simulation file settings')
    group2.add_argument('--prefix',type=str, default=None,metavar='X',help="Prefix for the name of each simulation file");
    group2.add_argument('--seed',  type=int, default=None,metavar='X',help="User-defined RNG seed");
    group2.add_argument('--Nfiles',type=int, default=1,   metavar='X',help="Number of simulation files to generate");

    group2.add_argument('--overwrite',       type=int,  default=0,    metavar='X',help="Whether to overwrite existing files");
    group2.add_argument('--n_delay',         type=int,  default=16,   metavar='X',help="Number of updates to group together in the delayed update scheme");
    group2.add_argument('--n_matmul',        type=int,  default=8,    metavar='X',help="Half the maximum number of direct matrix multiplications before applying a QR decomposition"); 
    group2.add_argument('--n_sweep_warm',    type=int,  default=200,  metavar='X',help="Number of warmup sweeps"); 
    group2.add_argument('--n_sweep_meas',    type=int,  default=2000, metavar='X',help="Number of measurement sweeps"); 
    group2.add_argument('--period_eqlt',     type=int,  default=8,    metavar='X',help="Period of equal-time measurements in units of single-site updates"); 
    group2.add_argument('--period_uneqlt',   type=int,  default=0,    metavar='X',help="Period of unequal-time measurements in units of full H-S sweeps. 0 means disabled"); 
    group2.add_argument('--trans_sym',       type=int,  default=1,    metavar='X',help="Whether to apply translational symmetry to compress measurement data"); 
    group2.add_argument('--checkpoint_every',type=int,  default=10000,metavar='X',help="Number of full H-S sweeps between checkpoints. 0 means disabled"); 

    group3 = parser.add_argument_group('Expensive measurement toggles')

    group3.add_argument('--meas_bond_corr',   type=int,  default=0, metavar='X',help="Whether to measure bond-bond correlations (current, kinetic energy, bond singlets)"); 
    group3.add_argument('--meas_energy_corr', type=int,  default=0, metavar='X',help="Whether to measure energy-energy correlations."); 
    group3.add_argument('--meas_nematic_corr',type=int,  default=0, metavar='X',help="Whether to measure spin and charge nematic correlations"); 
    group3.add_argument('--meas_thermal',     type=int,  default=0, metavar='X',help="Whether to measure extra jnj(2) type correlations for themal conductivity"); 
    group3.add_argument('--meas_2bond_corr',  type=int,  default=0, metavar='X',help="Whether to measure extra jj(2) type correlations for themal conductivity"); 
    group3.add_argument('--meas_chiral',      type=int,  default=0, metavar='X',help="Whether to measure scalar spin chirality");
    group3.add_argument('--meas_local_JQ',    type=int,  default=0, metavar='X',help="Whether to measure local JQ for energy magnetization contribution to thermal Hall");

    # parser.add_argument
    args = parser.parse_args()

    argdict = vars(args)

    for (k,v) in argdict.items():
        print(k,v)

    create_batch(**vars(args))

