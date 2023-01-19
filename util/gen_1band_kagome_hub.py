import os
import shutil
import sys
import tight_binding

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
             meas_thermal=0, meas_2bond_corr=0, meas_chiral=0,
             trans_sym=1, checkpoint_every=10000):
    if (not np.isclose(tp,0.0)) or period_uneqlt or meas_energy_corr:
        raise NotImplementedError

    assert L % n_matmul == 0 and L % period_eqlt == 0
    Norb=3
    #NOTE: N = total number of orbitals, not total number of unit cells
    N = Norb * Nx*Ny 
    #location (ix, iy) orbital io is 3d matrix (ix,iy,io)
    # with total index ix + Nx * iy + (Nx*Ny) * i0

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

    #phases accumulated by two-hop processes NOTE: placeholder
    thermal_phases = np.ones((b2ps, N),dtype=np.complex128)
    thermal_phases = thermal_phases if nflux !=0 else thermal_phases.real
    
    kij,peierls = tight_binding.H_periodic_kagome(Nx,Ny,t=1,tp=tp,nflux=nflux,alpha=1/2)
    #account for different data type when nflux=0
    Ku = kij if nflux != 0 else kij.real
    peierls = peierls if nflux !=0 else peierls.real

    for i in range(N):
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
        f["metadata"]["Norb"] = Norb
        f["metadata"]["bps"] = bps
        f["metadata"]["b2ps"] = b2ps
        f["metadata"]["plaq_per_site"] = plaq_per_cell
        f["metadata"]["U"] = U
        f["metadata"]["t'"] = tp
        f["metadata"]["nflux"] = nflux
        f["metadata"]["mu"] = mu
        f["metadata"]["beta"] = L*dt
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
        f["params"]["meas_chiral"] = meas_chiral
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
