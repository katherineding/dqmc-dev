import os
import shutil
import sys
import matplotlib.pyplot as plt

import h5py
import numpy as np
from scipy.linalg import expm

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


def create_1(file_sim=None, file_params=None, overwrite=False, init_rng=None,
             Nx=16, Ny=4, mu=0.0, tp=0.0, U=6.0, dt=0.115, L=40,
             nflux=0,
             n_delay=16, n_matmul=8, n_sweep_warm=200, n_sweep_meas=2000,
             period_eqlt=8, period_uneqlt=0,
             meas_bond_corr=0, meas_energy_corr=0, meas_nematic_corr=0,
             meas_thermal=0, meas_2bond_corr=0, 
             trans_sym=1, checkpoint_every=10000):
    assert L % n_matmul == 0 and L % period_eqlt == 0
    N = Nx * Ny

    if nflux != 0:
        dtype_num = np.complex128
    else:
        dtype_num = np.float64

    if init_rng is None:
        init_rng = rand_seed_urandom()
    rng = init_rng.copy()
    init_hs = np.zeros((L, N), dtype=np.int32)

    if file_sim is None:
        file_sim = "sim.h5"
    if file_params is None:
        file_params = file_sim
    
    one_file = (os.path.abspath(file_sim) == os.path.abspath(file_params))

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

    # bond definitions: defined by one hopping step
    bps = 6 if tp != 0.0 else 3  # bonds per site
    num_b = bps*N  # total bonds in cluster
    bonds = np.zeros((2, num_b), dtype=np.int32)
    # for iy in range(Ny):
    #     for ix in range(Nx):
    #         i = ix + Nx*iy
    #         iy1 = (iy + 1) % Ny
    #         ix1 = (ix + 1) % Nx
    #         bonds[0, i] = i            # i0 = i
    #         bonds[1, i] = ix1 + Nx*iy  # i1 = i + x
    #         bonds[0, i + N] = i            # i0 = i
    #         bonds[1, i + N] = ix + Nx*iy1  # i1 = i + y
    #         if bps == 4:
    #             bonds[0, i + 2*N] = i             # i0 = i
    #             bonds[1, i + 2*N] = ix1 + Nx*iy1  # i1 = i + x + y
    #             bonds[0, i + 3*N] = ix1 + Nx*iy   # i0 = i + x
    #             bonds[1, i + 3*N] = ix + Nx*iy1   # i1 = i + y

    # 1 bond 1 site mapping
    map_bs = np.zeros((N, num_b), dtype=np.int32)
    num_bs = bps*N if trans_sym else num_b*N
    degen_bs = np.zeros(num_bs, dtype=np.int32)
    # for j in range(N):
    #     for i in range(N):
    #         k = map_ij[j, i]
    #         for ib in range(bps):
    #             kk = k + num_ij*ib
    #             map_bs[j, i + N*ib] = kk
    #             degen_bs[kk] += 1
    # assert num_bs == map_bs.max() + 1

    # 1 bond - 1 bond mapping
    map_bb = np.zeros((num_b, num_b), dtype=np.int32)
    num_bb = bps*bps*N if trans_sym else num_b*num_b
    degen_bb = np.zeros(num_bb, dtype = np.int32)
    # for j in range(N):
    #     for i in range(N):
    #         k = map_ij[j, i]
    #         for jb in range(bps):
    #             for ib in range(bps):
    #                 kk = k + num_ij*(ib + bps*jb)
    #                 map_bb[j + N*jb, i + N*ib] = kk
    #                 degen_bb[kk] += 1
    # assert num_bb == map_bb.max() + 1

    # 2-bond definition is modified -- NOT consistent with Wen's!
    # Now only bonds defined by two hopping steps.
    # TODO
    b2ps = 12 if tp != 0.0 else 4  # 2-bonds per site
    num_b2 = b2ps*N  # total 2-bonds in cluster
    bond2s = np.zeros((2, num_b2), dtype=np.int32)
    # for iy in range(Ny):
    #     for ix in range(Nx):
    #         i = ix + Nx*iy
    #         iy1 = (iy + 1) % Ny
    #         ix1 = (ix + 1) % Nx
    #         iy2 = (iy + 2) % Ny
    #         ix2 = (ix + 2) % Nx
    #         # one t^2 path + two t'^2 paths [0,22,23]
    #         bond2s[0, i + 0*N] = i             # i0 = i
    #         bond2s[1, i + 0*N] = ix2 + Nx*iy   # i1 = i + 2x   -- /\ \/
    #         # one t^2 path + two t'^2 paths [1,24,25]
    #         bond2s[0, i + 1*N] = i            # i0 = i       |  /  \
    #         bond2s[1, i + 1*N] = ix + Nx*iy2  # i1 = i + 2y  |  \  /
    #         #two t^2 paths [2,3]
    #         bond2s[0, i + 2*N] = i             # i0 = i          _|   _ 
    #         bond2s[1, i + 2*N] = ix1 + Nx*iy1  # i1 = i + x + y      | 
    #         # two t^2 paths [4,5]
    #         bond2s[0, i + 3*N] = ix1 + Nx*iy   # i0 = i + x     _
    #         bond2s[1, i + 3*N] = ix + Nx*iy1   # i1 = i + y      |  |_
    #         if b2ps == 12:
    #             #two t't paths [6,7]
    #             bond2s[0, i + 4*N] = i             # i0 = i              _
    #             bond2s[1, i + 4*N] = ix2 + Nx*iy1  # i1 = i + 2x + y _/ /
    #             #two t't paths [8,9]
    #             bond2s[0, i + 5*N] = i              # i0 = i           |   /
    #             bond2s[1, i + 5*N] = ix1 + Nx*iy2   # i1 = i + x + 2y /   |
    #             #two t't paths [10,11]
    #             bond2s[0, i + 6*N] = ix2 + Nx*iy   # i0 = i + 2x _
    #             bond2s[1, i + 6*N] = ix + Nx*iy1   # i1 = i + y   \  \_
    #             #two t't paths [12,13]
    #             bond2s[0, i + 7*N] = ix1 + Nx*iy   # i0 = i + x   |   \
    #             bond2s[1, i + 7*N] = ix + Nx*iy2   # i1 = i + 2y   \   |
    #             # four t't paths [14,15,16,17]
    #             bond2s[0, i + 8*N] = i            # i0 = i      _  _   \  /
    #             bond2s[1, i + 8*N] = ix + Nx*iy1  # i1 = i + y  /  \   -  -
    #             # four t't paths [18,19,20,21]
    #             bond2s[0, i + 9*N] = i            # i0 = i      |\ /| \| |/
    #             bond2s[1, i + 9*N] = ix1 + Nx*iy  # i1 = i + x
    #             #one t'^2 path [26]
    #             bond2s[0, i + 10*N] = i             # i0 = i             /
    #             bond2s[1, i + 10*N] = ix2 + Nx*iy2  # i1 = i + 2x + 2y  /
    #             #one t'^2 path [27]
    #             bond2s[0, i + 11*N] = ix2 + Nx*iy   # i0 = i + 2x   \
    #             bond2s[1, i + 11*N] = ix + Nx*iy2   # i1 = i + 2y    \

    # my definition: Bonds defined by two hopping steps
    # Keep track of intermediate point!
    # TODO see if bonds 14 and 16 etc are not equivalent b/c start/end points changed...
    hop2ps = 28 if tp != 0.0 else 6  # 2-hop-bonds per site
    num_hop2 = hop2ps*N  # total 2-hop-bonds in cluster
    hop2s = np.zeros((3, num_hop2), dtype=np.int32)
    # for iy in range(Ny):
    #     for ix in range(Nx):
    #         i = ix + Nx*iy
    #         iy1 = (iy + 1) % Ny
    #         ix1 = (ix + 1) % Nx
    #         iy2 = (iy + 2) % Ny
    #         ix2 = (ix + 2) % Nx

    #         iym1 = (iy - 1) % Ny
    #         ixm1 = (ix - 1) % Nx

    #         #t^2 terms: NN + NN
    #         hop2s[0, i] = i             # i0 = i
    #         hop2s[1, i] = ix1 + Nx*iy   # i1 = i + x      --
    #         hop2s[2, i] = ix2 + Nx*iy   # i2 = i + 2x
    #         #-----------------
    #         hop2s[0, i + N] = i            # i0 = i         
    #         hop2s[1, i + N] = ix + Nx*iy1  # i1 = i + y     |
    #         hop2s[2, i + N] = ix + Nx*iy2  # i2 = i + 2y    |
    #         #-----------------
    #         hop2s[0, i + 2*N] = i             # i0 = i               _|
    #         hop2s[1, i + 2*N] = ix1 + Nx*iy   # i1 = i + x           
    #         hop2s[2, i + 2*N] = ix1 + Nx*iy1  # i2 = i + x + y

    #         hop2s[0, i + 3*N] = i            # i0 = i               _  
    #         hop2s[1, i + 3*N] = ix + Nx*iy1  # i1 = i + y          |
    #         hop2s[2, i + 3*N] = ix1 + Nx*iy1  # i2 = i + x + y
    #         #-----------------
    #         hop2s[0, i + 4*N] = ix1 + Nx*iy   # i0 = i + x            _ 
    #         hop2s[1, i + 4*N] = ix1 + Nx*iy1  # i1 = i + x + y         |
    #         hop2s[2, i + 4*N] = ix + Nx*iy1   # i2 = i + y

    #         hop2s[0, i + 5*N] = ix1 + Nx*iy   # i0 = i + x           |_
    #         hop2s[1, i + 5*N] = i             # i1 = i                
    #         hop2s[2, i + 5*N] = ix + Nx*iy1   # i2 = i + y

    #         if hop2ps == 28:
    #             # t*t' terms: NN + NNN or NNN + NN
    #             hop2s[0, i + 6*N] = i             # i0 = i            
    #             hop2s[1, i + 6*N] = ix1 + Nx*iy   # i1 = i + x         _/
    #             hop2s[2, i + 6*N] = ix2 + Nx*iy1  # i2 = i + 2x + y    

    #             hop2s[0, i + 7*N] = i             # i0 = i
    #             hop2s[1, i + 7*N] = ix1 + Nx*iy1  # i1 = i + x + y     _
    #             hop2s[2, i + 7*N] = ix2 + Nx*iy1  # i2 = i + 2x + y   /
    #             #------------------
    #             hop2s[0, i + 8*N] = i              # i0 = i             |
    #             hop2s[1, i + 8*N] = ix1 + Nx*iy1   # i1 = i + x + y    /
    #             hop2s[2, i + 8*N] = ix1 + Nx*iy2   # i2 = i + x + 2y

    #             hop2s[0, i + 9*N] = i              # i0 = i             / 
    #             hop2s[1, i + 9*N] = ix  + Nx*iy1   # i1 = i + y        |
    #             hop2s[2, i + 9*N] = ix1 + Nx*iy2   # i2 = i + x + 2y
    #             #------------------
    #             hop2s[0, i + 10*N] = ix2 + Nx*iy    # i0 = i + 2x
    #             hop2s[1, i + 10*N] = ix1 + Nx*iy1   # i1 = i + x + y    _
    #             hop2s[2, i + 10*N] = ix  + Nx*iy1   # i2 = i + y         \

    #             hop2s[0, i + 11*N] = ix2 + Nx*iy    # i0 = i + 2x     
    #             hop2s[1, i + 11*N] = ix1 + Nx*iy    # i1 = i + x       \_
    #             hop2s[2, i + 11*N] = ix  + Nx*iy1   # i2 = i + y        
    #             #------------------
    #             hop2s[0, i + 12*N] = ix1 + Nx*iy    # i0 = i + x
    #             hop2s[1, i + 12*N] = ix  + Nx*iy1   # i1 = i + y       |
    #             hop2s[2, i + 12*N] = ix + Nx*iy2    # i2 = i + 2y       \

    #             hop2s[0, i + 13*N] = ix1 + Nx*iy     # i0 = i + x
    #             hop2s[1, i + 13*N] = ix1 + Nx*iy1    # i1 = i + x + y     \ 
    #             hop2s[2, i + 13*N] = ix + Nx*iy2     # i2 = i + 2y         |
    #             #------------------
    #             hop2s[0, i + 14*N] = i             # i0 = i         
    #             hop2s[1, i + 14*N] = ix1 + Nx*iy1  # i1 = i + x + y     _
    #             hop2s[2, i + 14*N] = ix  + Nx*iy1  # i2 = i + y         /

    #             hop2s[0, i + 15*N] = i             # i0 = i         
    #             hop2s[1, i + 15*N] = ix1 + Nx*iy   # i1 = i + x         \
    #             hop2s[2, i + 15*N] = ix  + Nx*iy1  # i2 = i + y         -

    #             hop2s[0, i + 16*N] = i               # i0 = i           _
    #             hop2s[1, i + 16*N] = ixm1 + Nx*iy1   # i1 = i - x + y   \
    #             hop2s[2, i + 16*N] = ix  + Nx*iy1    # i2 = i + y    

    #             hop2s[0, i + 17*N] = i             # i0 = i       
    #             hop2s[1, i + 17*N] = ixm1 + Nx*iy  # i1 = i - x         /
    #             hop2s[2, i + 17*N] = ix  + Nx*iy1  # i2 = i + y         -
    #             #------------------
    #             hop2s[0, i + 18*N] = i             # i0 = i         
    #             hop2s[1, i + 18*N] = ix1 + Nx*iy1  # i1 = i + x + y     /|
    #             hop2s[2, i + 18*N] = ix1 + Nx*iy   # i2 = i + x     

    #             hop2s[0, i + 19*N] = i             # i0 = i         
    #             hop2s[1, i + 19*N] = ix  + Nx*iy1  # i1 = i + y     |\
    #             hop2s[2, i + 19*N] = ix1 + Nx*iy   # i2 = i + x   

    #             hop2s[0, i + 20*N] = i             # i0 = i       
    #             hop2s[1, i + 20*N] = ix1 + Nx*iym1 # i1 = i + x - y     \|
    #             hop2s[2, i + 20*N] = ix1 + Nx*iy   # i2 = i + x   

    #             hop2s[0, i + 21*N] = i             # i0 = i   
    #             hop2s[1, i + 21*N] = ix  + Nx*iym1 # i1 = i - y      |/
    #             hop2s[2, i + 21*N] = ix1 + Nx*iy   # i2 = i + x     
    #             #(t'^2) terms: NNN + NNN
    #             hop2s[0, i + 22*N] = i             # i0 = i         
    #             hop2s[1, i + 22*N] = ix1 + Nx*iy1  # i1 = i + x + y   /\
    #             hop2s[2, i + 22*N] = ix2 + Nx*iy   # i2 = i + 2x        

    #             hop2s[0, i + 23*N] = i             # i0 = i            
    #             hop2s[1, i + 23*N] = ix1 + Nx*iym1 # i1 = i + x - y   \/
    #             hop2s[2, i + 23*N] = ix2 + Nx*iy   # i2 = i + 2x    
    #             #------------------
    #             hop2s[0, i + 24*N] = i              # i0 = i         
    #             hop2s[1, i + 24*N] = ix1 + Nx*iy1   # i1 = i + x + y    \
    #             hop2s[2, i + 24*N] = ix + Nx*iy2    # i2 = i + 2y       /

    #             hop2s[0, i + 25*N] = i             # i0 = i            /
    #             hop2s[1, i + 25*N] = ixm1 + Nx*iy1 # i1 = i - x + y    \
    #             hop2s[2, i + 25*N] = ix + Nx*iy2   # i2 = i + 2y    
    #             #------------------
    #             hop2s[0, i + 26*N] = i             # i0 = i              /   
    #             hop2s[1, i + 26*N] = ix1 + Nx*iy1  # i1 = i + x + y     / 
    #             hop2s[2, i + 26*N] = ix2 + Nx*iy2  # i2 = i + 2x + 2y    
    #             #------------------
    #             hop2s[0, i + 27*N] = ix2 + Nx*iy    # i0 = i + 2x       \
    #             hop2s[1, i + 27*N] = ix1 + Nx*iy1   # i1 = i + x + y     \ 
    #             hop2s[2, i + 27*N] = ix + Nx*iy2    # i2 = i + 2y

    #how bond2s and hop2s are related
    # bond_hop_dict = {}
    # if b2ps == 4:
    #     bond_hop_dict[0] = [0]
    #     bond_hop_dict[1] = [1]
    #     bond_hop_dict[2] = [2,3]
    #     bond_hop_dict[3] = [4,5]
    # else:
    #     bond_hop_dict[0] = [0,22,23];
    #     bond_hop_dict[1] = [1,24,25];
    #     for i in range(2,8):
    #         bond_hop_dict[i] = [2*i-2,2*i-1];
    #     bond_hop_dict[8] = [14,15,16,17]
    #     bond_hop_dict[9] = [18,19,20,21]
    #     bond_hop_dict[10] = [26]
    #     bond_hop_dict[11] = [27]


    # 2 2-bond mapping
    num_b2b2 = b2ps*b2ps*N if trans_sym else num_b2*num_b2
    map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
    degen_b2b2 = np.zeros(num_b2b2, dtype = np.int32)
    # for j in range(N):
    #     for i in range(N):
    #         k = map_ij[j, i]
    #         for jb in range(b2ps):
    #             for ib in range(b2ps):
    #                 kk = k + num_ij*(ib + b2ps*jb)
    #                 map_b2b2[j + N*jb, i + N*ib] = kk
    #                 degen_b2b2[kk] += 1
    # assert num_b2b2 == map_b2b2.max() + 1

    # bond 2-bond mapping
    num_bb2 = bps*b2ps*N if trans_sym else num_b*num_b2
    map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
    degen_bb2 = np.zeros(num_bb2, dtype = np.int32)
    # for j in range(N):
    #     for i in range(N):
    #         k = map_ij[j, i]
    #         for jb in range(bps):
    #             for ib in range(b2ps):
    #                 kk = k + num_ij*(ib + b2ps*jb)
    #                 map_bb2[j + N*jb, i + N*ib] = kk
    #                 degen_bb2[kk] += 1
    # assert num_bb2 == map_bb2.max() + 1

    # 2-bond bond mapping
    num_b2b = b2ps*bps*N if trans_sym else num_b2*num_b
    map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
    degen_b2b = np.zeros(num_b2b, dtype = np.int32)
    # for j in range(N):
    #     for i in range(N):
    #         k = map_ij[j, i]
    #         for jb in range(b2ps):
    #             for ib in range(bps):
    #                 kk = k + num_ij*(ib + bps*jb)
    #                 map_b2b[j + N*jb, i + N*ib] = kk
    #                 degen_b2b[kk] += 1
    # assert num_b2b == map_b2b.max() + 1

    # First: hopping (assuming periodic boundaries and no field)
    kij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) % Ny
            iyn = (iy - 1) % Ny
            ix1 = (ix + 1) % Nx
            ixn = (ix - 1) % Nx
                #jx      jy    jo     ix       iy    io
            kij[ix1+Nx*iy , ix +Nx*iy ] += -1
            kij[ix +Nx*iy , ix1+Nx*iy ] += -1
            kij[ix +Nx*iy1, ix +Nx*iy ] += -1
            kij[ix +Nx*iy , ix +Nx*iy1] += -1
            kij[ix1+Nx*iy , ix +Nx*iy1] += -1
            kij[ix +Nx*iy1, ix1+Nx*iy ] += -1

    # Next: phases accumulated by single-hop processes
    # "a1" primitive vector: (1/2, sqrt(3)/2)
    # "a2" primitive vector: (-1/2, sqrt(3)/2)
    # phi[i,j] = path integral det by (j-i) * A(mid) = 
    # hopping phase in j -> i hop
    alpha = 0.5  # gauge choice. 0.5 for symmetric gauge.
    beta = 1 - alpha
    phi = np.zeros((Ny*Nx, Ny*Nx))

    phi2 = np.zeros((Ny*Nx, Ny*Nx))
    # path is straight line
    # if Ny is even, prefer dy - -Ny/2 over Ny/2. likewise for even Nx
    const = np.sqrt(3)
    prefactor = 2*np.pi*2*nflux/(const*Nx*Ny)
    #displacement vector: d
    for dy in range((1-Ny)//2, (1+Ny)//2):
        for dx in range((1-Nx)//2, (1+Nx)//2):
            for iy in range(Ny):
                for ix in range(Nx):
                    #start site: j
                    jy = iy + dy
                    jjy = jy % Ny
                    jx = ix + dx
                    jjx = jx % Nx

                    #index offset = \pm Nx, \pm Ny
                    offset_a1 = jx - jjx
                    offset_a2 = jy - jjy

                    #true spatial location R_i
                    irx = (ix - iy)/2
                    iry = const * (ix + iy)/2

                    #true spatial location R_j
                    jrx = (jx - jy)/2
                    jry = const * (jx + jy)/2

                    # wrapped spatial location R_j
                    jjrx = (jjx - jjy)/2
                    jjry = const * (jjx + jjy)/2
                    
                    #true displacement distance R_d
                    drx = (dx - dy)/2
                    dry = const * (dx + dy)/2

                    # true displacement mid point    
                    mrx = (irx + jrx)/2
                    mry = (iry + jry)/2

                    #indices are opposite edwin's code, but 
                    # results should be equivalent, 
                    # since phi_ij sign also flipped?
                    # NOTE This is important: How to get this boundary phase?
                    # 
                    #True spatial offset: wrap along a1 direction
                    offset_a1_rx = offset_a1 / 2
                    offset_a1_ry = offset_a1 * const / 2

                    #True spatial offset: wrap along a2 direction
                    offset_a2_rx = - offset_a2 / 2
                    offset_a2_ry = offset_a2 * const / 2

                    xwrap_phase = (+alpha) * offset_a1_ry * jjrx \
                                    - beta * offset_a1_rx * jjry
                    ywrap_phase = (+alpha) * offset_a2_ry * jjrx \
                                    - beta * offset_a2_rx * jjry

                    #either choice of xywrap or yxwrap phase should lead to \
                    #the correct Hamiltonian
                    xywrap_phase = alpha * offset_a2_ry * offset_a1_rx \
                                   -beta * offset_a2_rx * offset_a1_ry


                    yxwrap_phase = alpha * offset_a1_ry * offset_a2_rx \
                                   -beta * offset_a1_rx * offset_a2_ry

                    # if not np.isclose(0,xywrap_phase):
                    #     print(prefactor*xywrap_phase, prefactor*yxwrap_phase)

                    # Field quantization condition due to finite lattice
                    corner_diff = prefactor*(xywrap_phase - yxwrap_phase)/(2*np.pi)
                    #assert np.isclose(corner_diff,0) or np.isclose(np.abs(corner_diff),nflux)
                    #print("nflux",nflux,corner_diff)
                    
                    phi2[jjx + Nx*jjy,ix + Nx*iy] = - alpha*mry*drx + beta*mrx*dry \
                        + xwrap_phase + ywrap_phase + xywrap_phase

                    phi[jjx + Nx*jjy,ix + Nx*iy] = \
                        - alpha*mry*drx + beta*mrx*dry + \
                        - ((-alpha*offset_a1*jrx*const/2+beta*offset_a1*jry/2) + \
                        (-alpha*offset_a2*jrx*const/2-beta*offset_a2*jry/2) - \
                        alpha*offset_a1*offset_a2*const/2 )

    #Lattice total area = Nx * Ny * sqrt(3) * a^2 / 2
    #prefactor = 2*np.pi*2*nflux/(const*Nx*Ny)
    peierls = np.exp(1j*prefactor*phi)

    peierls2 = np.exp(1j*prefactor*phi2)

    # print(np.allclose(peierls,peierls2))

    #phases accumulated by two-hop processes
    #Here: types 0,1 include t' factors
    #   Types 2-7: sum of two paths
    #   Types 8,9: sum of four paths
    #   Types 10,11: one path each
    #   ZF case: 1+2*tp, 1+2*tp, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1
    thermal_phases = np.ones((b2ps, N),dtype=np.complex128)
    # for i in range(N):
    #     for btype in range(b2ps):
    #         i0 = bond2s[0, i + btype*N] #start
    #         i2 = bond2s[1, i + btype*N] #end
    #         pp = 0; 
    #         #list of intermediate pointscorresponding to this bond
    #         i1_type_list = bond_hop_dict[btype]
    #         #print("btype = ",btype,"i1_type_list = ",i1_type_list)
    #         #two bonds need manual weighting when when t' != 0: 
    #         if b2ps == 12 and (btype == 0 or btype == 1):
    #             i1 = hop2s[1, i + i1_type_list[0]*N]
    #             #print(f"i = {i}, btype = {btype}, path ({i0},{i1},{i2})")
    #             pp += peierls[i0,i1] * peierls[i1,i2]
    #             for i1type in i1_type_list[1:]:
    #                 i1 = hop2s[1, i + i1type*N]
    #                 pp += tp*tp*peierls[i0,i1] * peierls[i1,i2]
    #             # print(pp)
    #         #general case
    #         else:
    #             for i1type in i1_type_list:
    #                 i1 = hop2s[1, i + i1type*N]
    #                 pp += peierls[i0,i1] * peierls[i1,i2]
    #         thermal_phases[btype,i] = pp
    #         
                
    if dtype_num == np.complex128:
        Ku = kij * peierls
        Ku2 = kij * peierls2


        # plt.figure()
        # plt.matshow(Ku.real)
        # plt.colorbar()

        # plt.figure()
        # plt.matshow(Ku2.real)
        # plt.colorbar()

        # print(np.linalg.norm(Ku-Ku2))

        assert np.linalg.norm(Ku - Ku.T.conj()) < 1e-10, f"max diff {np.linalg.norm(Ku - Ku.T.conj())}"
    else:
        Ku = kij.real
        assert np.max(np.abs(peierls.imag)) < 1e-10
        peierls = peierls.real
        assert np.max(np.abs(thermal_phases.imag)) < 1e-10
        thermal_phases = thermal_phases.real
    #print(thermal_phases.dtype,thermal_phases.shape)
    #
    

    for i in range(Ny*Nx):
        Ku[i, i] -= mu

    exp_Ku = expm(-dt * Ku)
    inv_exp_Ku = expm(dt * Ku)
    exp_halfKu = expm(-dt/2 * Ku)
    inv_exp_halfKu = expm(dt/2 * Ku)
#   exp_K = np.array(mpm.expm(mpm.matrix(-dt * K)).tolist(), dtype=np.float64)

    U_i = U*np.ones_like(degen_i, dtype=np.float64)
    assert U_i.shape[0] == num_i

    exp_lmbd = np.exp(0.5*U_i*dt) + np.sqrt(np.expm1(U_i*dt))
#    exp_lmbd = np.exp(np.arccosh(np.exp(0.5*U_i*dt)))
#    exp_lmbd = float(mpm.exp(mpm.acosh(mpm.exp(0.5*float(U*dt)))))
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
        f["metadata"]["bps"] = bps
        f["metadata"]["b2ps"] = b2ps
        f["metadata"]["U"] = U
        f["metadata"]["t'"] = tp
        f["metadata"]["nflux"] = nflux
        f["metadata"]["mu"] = mu
        f["metadata"]["beta"] = L*dt

        # parameters used by dqmc code
        f.create_group("params")
        # model parameters
        f["params"]["N"] = np.array(N, dtype=np.int32)
        f["params"]["L"] = np.array(L, dtype=np.int32)
        f["params"]["map_i"] = map_i
        f["params"]["map_ij"] = map_ij
        f["params"]["bonds"] = bonds
        f["params"]["bond2s"] = bond2s
        f["params"]["map_bs"] = map_bs
        f["params"]["map_bb"] = map_bb
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
        f["params"]["Kd"] = f["params"]["Ku"]
        f["params"]["U"] = U_i
        f["params"]["dt"] = np.array(dt, dtype=np.float64)

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
        f["params"]["checkpoint_every"] = checkpoint_every

        # precalculated stuff
        f["params"]["num_i"] = num_i
        f["params"]["num_ij"] = num_ij
        f["params"]["num_b"] = num_b
        f["params"]["num_b2"] = num_b2
        f["params"]["num_bs"] = num_bs
        f["params"]["num_bb"] = num_bb
        f["params"]["num_b2b"] = num_b2b
        f["params"]["num_bb2"] = num_bb2
        f["params"]["num_b2b2"] = num_b2b2
        f["params"]["degen_i"] = degen_i
        f["params"]["degen_ij"] = degen_ij
        f["params"]["degen_bs"] = degen_bs
        f["params"]["degen_bb"] = degen_bb
        f["params"]["degen_bb2"] = degen_bb2
        f["params"]["degen_b2b"] = degen_b2b
        f["params"]["degen_b2b2"] = degen_b2b2
        f["params"]["exp_Ku"] = exp_Ku
        f["params"]["exp_Kd"] = f["params"]["exp_Ku"]
        f["params"]["inv_exp_Ku"] = inv_exp_Ku
        f["params"]["inv_exp_Kd"] = f["params"]["inv_exp_Ku"]
        f["params"]["exp_halfKu"] = exp_halfKu
        f["params"]["exp_halfKd"] = f["params"]["exp_halfKu"]
        f["params"]["inv_exp_halfKu"] = inv_exp_halfKu
        f["params"]["inv_exp_halfKd"] = f["params"]["inv_exp_halfKu"]
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
        f["meas_eqlt"]["double_occ"] = np.zeros(num_i, dtype=dtype_num)
        f["meas_eqlt"]["g00"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["nn"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["xx"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["zz"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["pair_sw"] = np.zeros(num_ij, dtype=dtype_num)
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
            f["meas_uneqlt"]["gt0"] = np.zeros(num_ij*L, dtype=dtype_num)
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
            #thermal is subset of bond-bond type measurements
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


def main(argv):
    kwargs = {}
    for arg in argv[1:]:
        eq = arg.find("=")
        if eq == -1:
            print("couldn't find \"=\" in argument " + arg)
            return
        key = arg[:eq]
        val = arg[(eq + 1):]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except:
                pass
        kwargs[key] = val
    create_batch(**kwargs)

if __name__ == "__main__":
    main(sys.argv)
