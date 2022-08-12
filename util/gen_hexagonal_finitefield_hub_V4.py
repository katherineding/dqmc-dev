import shutil
import sys
import time

import h5py
import numpy as np
from scipy.linalg import expm

np.seterr(over="ignore")

# removed edwin's urandom default generation with time-seeded generation
# using rand_uint
# 
# which commit of edwin's is this based on?


# http://xoroshiro.di.unimi.it/splitmix64.c
def rand_seed(x):
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

# "x" primitive vector: (1/2, sqrt(3)/2)
# "y" primitive vector: (-1/2, sqrt(3)/2)
# no parameter file?
# whats with the nflux?
def create_1(filename=None, overwrite=False, seed=None,
             Nx=8, Ny=8, mu=0.0, t1=1.0, t2=0.0, t3=0.0, U=6.0, dt=0.1, L=40,
             nflux=np.sqrt(3)/2, ## NOTE: not quantized??
             n_delay=16, n_matmul=8, n_sweep_warm=200, n_sweep_meas=2000,
             period_eqlt=8, period_uneqlt=0,
             meas_bond_corr=0, meas_energy_corr=0, meas_nematic_corr=0,
             trans_sym=1,alpha=1/2):
    assert L % n_matmul == 0 and L % period_eqlt == 0
    #what's the point of Norb? Hexagon Norb = 2, Kagome Norb = 3?
    Norb = 1
    N = Norb * Ny * Nx

    #TODO: need to allow for dtype_num = np.float64
    dtype_num = np.complex128
    

    if seed is None:
        seed = int(time.time())
    init_rng = rand_seed(seed)
    init_hs = np.zeros((L, N), dtype=np.int32)

    for l in range(L):
        for i in range(N):
            init_hs[l, i] = rand_uint(init_rng) >> np.uint64(63)

    # 1 site mapping
    if trans_sym:
        map_i = np.zeros(N, dtype=np.int32)
        #map_i[Ny*Nx:] = 1 #????
        #degen_i = np.array((Ny*Nx, Ny*Nx), dtype=np.int32)
        degen_i = np.array((Ny*Nx, ), dtype=np.int32)
    else:
        map_i = np.arange(N, dtype=np.int32)
        degen_i = np.ones(N, dtype=np.int32)
    num_i = map_i.max() + 1
    assert num_i == degen_i.size

    # 2 site mapping
    map_ij = np.zeros((N, N), dtype=np.int32)
    num_ij = Norb*Norb*Ny*Nx if trans_sym else N*N #norb
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
                                k = kx + Nx*ky + Nx*Ny*io + Nx*Ny*Norb*jo #norb
                            else:
                                k = (ix + Nx*iy + Nx*Ny*io) + N*(jx + Nx*jy + Nx*Ny*jo)
                            map_ij[jx + Nx*jy + Nx*Ny*jo, ix + Nx*iy + Nx*Ny*io] = k
                            degen_ij[k] += 1
    assert num_ij == map_ij.max() + 1

    # bond definitions, used in bs and bb correlators
    bps = 3 # bonds per site TODO ##
    num_b = bps*N  # total bonds in cluster
    bonds = np.zeros((2, num_b), dtype=np.int32)
    #TODO
#     for iy in range(Ny):
#         for ix in range(Nx):
#             i = ix + Nx*iy
#             iy1 = (iy + 1) % Ny
#             ix1 = (ix + 1) % Nx
#             bonds[0, i] = i            # i0 = i
#             bonds[1, i] = ix1 + Nx*iy  # i1 = i + x
#             bonds[0, i + N] = i            # i0 = i
#             bonds[1, i + N] = ix + Nx*iy1  # i1 = i + y
#             if bps == 4:
#                 bonds[0, i + 2*N] = i             # i0 = i
#                 bonds[1, i + 2*N] = ix1 + Nx*iy1  # i1 = i + x + y
#                 bonds[0, i + 3*N] = ix1 + Nx*iy   # i0 = i + x
#                 bonds[1, i + 3*N] = ix + Nx*iy1   # i1 = i + y

    # 1 bond 1 site mapping
    map_bs = np.zeros((N, num_b), dtype=np.int32)
    num_bs = bps*N if trans_sym else num_b*N
    degen_bs = np.zeros(num_bs, dtype=np.int32)
    #TODO
#     for j in range(N):
#         for i in range(N):
#             k = map_ij[j, i]
#             for ib in range(bps):
#                 kk = k + num_ij*ib
#                 map_bs[j, i + N*ib] = kk
#                 degen_bs[kk] += 1
#     assert num_bs == map_bs.max() + 1

    # 2 bond mapping
    map_bb = np.zeros((num_b, num_b), dtype=np.int32)
    num_bb = bps*bps*N if trans_sym else num_b*num_b
    degen_bb = np.zeros(num_bb, dtype=np.int32)
    #TODO
#     for j in range(N):
#         for i in range(N):
#             k = map_ij[j, i]
#             for jb in range(bps):
#                 for ib in range(bps):
#                     kk = k + num_ij*(ib + bps*jb)
#                     map_bb[j + N*jb, i + N*ib] = kk
#                     degen_bb[kk] += 1
#     assert num_bb == map_bb.max() + 1

    # hopping (assuming periodic boundaries and no field)
    # NOTE only uniform NN hopping implemented here
    kij = np.zeros((Norb*Ny*Nx, Norb*Ny*Nx), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) % Ny
            iyn = (iy - 1) % Ny
            ix1 = (ix + 1) % Nx
            ixn = (ix - 1) % Nx
                #jx      jy    jo     ix       iy    io
            kij[ix1+Nx*iy , ix +Nx*iy ] += t1
            kij[ix +Nx*iy , ix1+Nx*iy ] += t1
            kij[ix +Nx*iy1, ix +Nx*iy ] += t1
            kij[ix +Nx*iy , ix +Nx*iy1] += t1
            kij[ix1+Nx*iy , ix +Nx*iy1] += t1
            kij[ix +Nx*iy1, ix1+Nx*iy ] += t1

    # this is the key bit
    #alpha = 0.5  # gauge choice. 0.5 for symmetric gauge.
    beta = 1 - alpha
    phi = np.zeros((N, N))
    # path is straight line
    # if Ny is even, prefer dy - -Ny/2 over Ny/2. likewise for even Nx
    const = np.sqrt(3)
    #start site: i
    for iy in range(Ny):
        for ix in range(Nx):
            #displacement vector: d
            for dy in range((1-Ny)//2, (1+Ny)//2):
                for dx in range((1-Nx)//2, (1+Nx)//2):
                    #end site: j
                    jy = iy + dy
                    jjy = jy % Ny
                    jx = ix + dx
                    jjx = jx % Nx
                    
                    #true spatial location R_i
                    irx = (ix - iy)/2
                    iry = const * (ix + iy)/2

                    #true spatial location R_j
                    jrx = (jx - jy)/2
                    jry = const * (jx + jy)/2

                    # TODO: check jx wrapped, jy wrapped
                    jjrx = (jjx - jjy)/2
                    jjry = const * (jjx + jjy)/2
                    
                    #true displacement distance R_d
                    drx = (dx - dy)/2
                    dry = const * (dx + dy)/2

                    #boundary phase offset
                    offset_x = jx - jjx
                    offset_y = jy - jjy

                    # true displacement mid point    
                    mx = (irx + jrx)/2
                    my = (iry + jry)/2

                    #indices are opposite edwin's code, but 
                    # results should be equivalent, 
                    # since phi_ij sign also flipped?
                    # NOTE This is important: How to get this boundary phase?
                    phi[ix + Nx*iy,jjx + Nx*jjy] = \
                        alpha*my*drx - beta*mx*dry + \
                        (-alpha*offset_x*jrx*const/2+beta*offset_x*jry/2) + \
                        (-alpha*offset_y*jrx*const/2-beta*offset_y*jry/2) - \
                        alpha*offset_x*offset_y*const/2


    peierls = np.exp(2j*np.pi*(nflux/(Nx*Ny*const/2))*phi)

    if dtype_num == np.complex128:
        Ku = kij * peierls
        Kd = kij * peierls
        assert np.max(np.abs(Ku - Ku.T.conj())) < 1e-10
    else:
        Ku = kij.copy().real
        Kd = kij.copy().real
        assert np.max(np.abs(peierls.imag)) < 1e-10
        peierls = peierls.real

    for i in range(N):
        Ku[i, i] -= mu
        Kd[i, i] -= mu

    exp_Ku = expm(-dt * Ku)
    exp_Kd = expm(-dt * Kd)
    inv_exp_Ku = expm(dt * Ku)
    inv_exp_Kd = expm(dt * Kd)
    exp_halfKu = expm(-dt/2 * Ku)
    exp_halfKd = expm(-dt/2 * Kd)
    inv_exp_halfKu = expm(dt/2 * Ku)
    inv_exp_halfKd = expm(dt/2 * Kd)

    return peierls, Ku
    
#   exp_K = np.array(mpm.expm(mpm.matrix(-dt * K)).tolist(), dtype=np.float64)

    U_i = U*np.ones_like(degen_i, dtype=np.float64)
    assert U_i.shape[0] == num_i

    exp_lmbd = np.exp(0.5*U_i*dt) + np.sqrt(np.expm1(U_i*dt))
#    exp_lmbd = np.exp(np.arccosh(np.exp(0.5*U_i*dt)))
#    exp_lmbd = float(mpm.exp(mpm.acosh(mpm.exp(0.5*float(U*dt)))))
    exp_lambda = np.array((exp_lmbd[map_i]**-1, exp_lmbd[map_i]))
    delll = np.array((exp_lmbd[map_i]**2 - 1, exp_lmbd[map_i]**-2 - 1))

    if filename is None:
        filename = "{}.h5".format(seed)
    with h5py.File(filename, "w" if overwrite else "x") as f:
        # parameters not used by dqmc code, but useful for analysis
        f.create_group("metadata")
        f["metadata"]["version"] = 0.0
        f["metadata"]["model"] = \
            "Hubbard (complex)" if dtype_num == np.complex128 else "Hubbard"
        f["metadata"]["Norb"] = Norb
        f["metadata"]["Nx"] = Nx
        f["metadata"]["Ny"] = Ny
        f["metadata"]["bps"] = bps
        f["metadata"]["U"] = U
        f["metadata"]["t1"] = t1 # potentially anisotropic NN triangular bonds?
        f["metadata"]["t2"] = t2 #
        f["metadata"]["t3"] = t3 #
        f["metadata"]["nflux"] = nflux # no longer integer?
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
        f["params"]["map_bs"] = map_bs
        f["params"]["map_bb"] = map_bb
        f["params"]["peierlsu"] = peierls
        f["params"]["peierlsd"] = peierls
        f["params"]["Ku"] = Ku
        f["params"]["Kd"] = Kd
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
        f["params"]["meas_energy_corr"] = meas_energy_corr
        f["params"]["meas_nematic_corr"] = meas_nematic_corr
        f["params"]["init_rng"] = init_rng  # save if need to replicate data

        # precalculated stuff
        f["params"]["num_i"] = num_i
        f["params"]["num_ij"] = num_ij
        f["params"]["num_b"] = num_b
        f["params"]["num_bs"] = num_bs
        f["params"]["num_bb"] = num_bb
        f["params"]["degen_i"] = degen_i
        f["params"]["degen_ij"] = degen_ij
        f["params"]["degen_bs"] = degen_bs
        f["params"]["degen_bb"] = degen_bb
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

        # simulation state
        f.create_group("state")
        f["state"]["sweep"] = np.array(0, dtype=np.int32)
        f["state"]["rng"] = init_rng
        f["state"]["hs"] = init_hs

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
            if meas_energy_corr:
                f["meas_uneqlt"]["kv"] = np.zeros(num_bs*L, dtype=dtype_num)
                f["meas_uneqlt"]["kn"] = np.zeros(num_bs*L, dtype=dtype_num)
                f["meas_uneqlt"]["vv"] = np.zeros(num_ij*L, dtype=dtype_num)
                f["meas_uneqlt"]["vn"] = np.zeros(num_ij*L, dtype=dtype_num)
            if meas_nematic_corr:
                f["meas_uneqlt"]["nem_nnnn"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["nem_ssss"] = np.zeros(num_bb*L, dtype=dtype_num)
    return filename


def create_batch(Nfiles=1, prefix=None, seed=None, **kwargs):
    if seed is None:
        seed = int(time.time())
    if prefix is None:
        prefix = str(seed)
    rng = rand_seed(seed)

    file_0 = "{}_{}.h5".format(prefix, 0)

    create_1(filename=file_0, seed=seed, **kwargs)
    with h5py.File(file_0, "r") as f:
        N = f["params"]["N"][...]
        L = f["params"]["L"][...]

    for i in range(1, Nfiles):
        rand_jump(rng)
        init_rng = rng.copy()
        init_hs = np.zeros((L, N), dtype=np.int32)

        for l in range(L):
            for r in range(N):
                init_hs[l, r] = rand_uint(init_rng) >> np.uint64(63)

        file_i = "{}_{}.h5".format(prefix, i)
        shutil.copy2(file_0, file_i)
        with h5py.File(file_i, "r+") as f:
            f["params"]["init_rng"][...] = init_rng
            f["state"]["rng"][...] = init_rng
            f["state"]["hs"][...] = init_hs
    return file_0 if Nfiles == 1 else "{} ... {}".format(file_0, file_i)


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
    print("created simulation files:", create_batch(**kwargs))

if __name__ == "__main__":
    main(sys.argv)