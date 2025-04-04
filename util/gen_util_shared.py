import os
import numpy as np


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


def set_U(U, dt, num_i, map_i, degen_i):
    U_i = U * np.ones_like(degen_i, dtype=np.float64)
    assert U_i.shape[0] == num_i
    if U >= 0:
        # cosh(lmbd) = exp[dt*U/2]
        # exp(lmbd) = exp[dt*U/2) + sqrt[-1 + exp[dt*U]]
        exp_lmbd = np.exp(0.5 * U_i * dt) + np.sqrt(np.expm1(U_i * dt))
        exp_lambda = np.array((exp_lmbd[map_i] ** -1, exp_lmbd[map_i]))
        delll = np.array((exp_lmbd[map_i] ** 2 - 1, exp_lmbd[map_i] ** -2 - 1))
    else:
        # cosh(lmbd) = exp[dt*|U|/2]
        # exp(lmbd) = exp[dt*|U|/2) + sqrt[-1 + exp[dt*|U|]]
        # TODO: check me
        exp_lmbd = np.exp(0.5 * np.abs(U_i) * dt) + np.sqrt(np.expm1(np.abs(U_i) * dt))
        exp_lambda = np.array((exp_lmbd[map_i] ** -1, exp_lmbd[map_i] ** -1))
        delll = np.array((exp_lmbd[map_i] ** 2 - 1, exp_lmbd[map_i] ** 2 - 1))

    return U_i, exp_lambda, delll


def add_simFile_opts(parser):
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


def add_meas_opts(parser):
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
    group3.add_argument(
        "--meas_gen_suscept",
        type=int,
        default=0,
        metavar="X",
        help="Whether to measure generalized susceptibility",
    )
    group3.add_argument(
        "--meas_pair_bb_only",
        type=int,
        default=0,
        metavar="X",
        help="Whether to, among expensive measurements, to only measure bond singlet pair correlators in order to save on storage",
    )
