import glob
from itertools import product
from scipy.signal import hilbert
import sys

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from scipy.interpolate import interp1d
import misc
import util  # edwins util file

np.set_printoptions(precision=4)

plaq_per_cell_dict = {}
plaq_per_cell_dict["square"] = 2
plaq_per_cell_dict["triangular"] = 2
plaq_per_cell_dict["honeycomb"] = 1
plaq_per_cell_dict["kagome"] = 2

Norb_dict = {}
Norb_dict["square"] = 1
Norb_dict["triangular"] = 1
Norb_dict["honeycomb"] = 2
Norb_dict["kagome"] = 3


def multiply_err(x, y, xerr, yerr) -> tuple[np.ndarray, np.ndarray]:
    res = x * y
    rerr = np.sqrt((xerr / x) ** 2 + (yerr / y) ** 2)

    return res, np.abs(res * rerr)


def ratio_err(x, y, xerr, yerr) -> tuple[np.ndarray, np.ndarray]:
    res = x / y
    rerr = np.sqrt((xerr / x) ** 2 + (yerr / y) ** 2)

    return res, np.abs(res * rerr)


def add_err(x, y, xerr, yerr) -> tuple[np.ndarray, np.ndarray]:
    res = x + y
    err = np.sqrt(xerr**2 + yerr**2)

    return res, err


def kk_hilbert(dat: np.ndarray) -> np.ndarray:
    """Assumes data is uniformly sampled, compute K-K using
    scipy.signal.hilbert. Note loss of DC offset

    Args:
        dat (np.ndarray): shape (nbins, nomega)

    Returns:
        np.ndarray: [description]
    """
    return np.imag(hilbert(-dat, axis=-1))


def uniform_resample(w, dat, ns) -> tuple[np.ndarray, np.ndarray]:
    assert w.shape == dat.shape

    f = interp1d(w, dat, kind="cubic", fill_value=0)

    w_new = np.linspace(w.min(), w.max(), ns)
    f_new = f(w_new)
    return w_new, f_new


def xy_DC_response_dumb(
    omega: np.ndarray,
    domega: np.ndarray,
    diff: np.ndarray,
    beta: Optional[float] = None,
) -> tuple[float, float]:
    """In: diff = i chi^2_xy(w) / w NO MINUS SIGN

    Out: i chi^1_xy / w (w->0) = int dw' i chi^2_xy(w')/w'^2 / pi

    Args:
        omega (np.ndarray): (nw,) array
        domega (np.ndarray): (nw,) array
        diff (np.ndarray): (bs, nw) array
        beta (float): [description] (default: `None`)
    """

    assert diff.ndim == 2
    n_bs = diff.shape[0]
    v = np.full(n_bs, np.nan, dtype=float)
    for i in range(n_bs):
        # just rectangular rule
        v[i] = np.sum(diff[i] / omega * domega / np.pi)
        if beta is not None:
            v[i] *= beta

    return np.mean(v), np.std(v)


# def xy_response(
#     omega: np.ndarray,
#     diff: np.ndarray,
#     ns: int,
#     name="sigma",
#     beta: Optional[float] = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     assert diff.ndim == 2
#     n_bs = diff.shape[0]
#     v = np.full((n_bs, ns), np.nan, dtype=float)
#     for i in range(n_bs):
#         # multiply by omega to get -i chi^2_xy
#         diffm = diff[i] * omega

#         w_new, d_new = uniform_resample(omega, diffm, ns)

#         v[i] = kk_hilbert(d_new) / w_new
#         if name == "kappa" or name == "L12":
#             assert beta is not None
#             v[i] *= beta

#     # NOTE: bootstrap error analysis.
#     return np.mean(v, axis=0), np.std(v, axis=0)


def cv(
    beta_arr: np.ndarray, E_arr: np.ndarray, E_err_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert beta_arr.shape == E_arr.shape
    # check: imaginary part is small, or no data
    if np.all(np.isnan(E_arr)):
        return (
            np.full(E_arr.shape[0] - 1, np.nan),
            np.full(E_arr.shape[0] - 1, np.nan),
            np.full(E_arr.shape[0] - 1, np.nan),
        )
    if np.nanmax(np.abs(E_arr.imag / E_arr.real)) > 1e-3:
        print("Imag part size:", np.abs(E_arr.imag / E_arr.real))

    # print(np.nanmax(np.abs(E_arr.imag / E_arr.real)) or np.all(np.isnan(E_arr)))

    temp_mid = 1 / 2 * (1 / beta_arr[1:] + 1 / beta_arr[:-1])
    temp_diff = 1 / beta_arr[1:] - 1 / beta_arr[:-1]

    dE = E_arr[1:].real - E_arr[:-1].real

    dE_err = np.sqrt(E_err_arr[1:].real ** 2 + E_err_arr[:-1].real ** 2)

    return temp_mid, dE / temp_diff, -dE_err / temp_diff


def jackknife(asgn, sgn):
    """this jackknife function keeps track of imaginary parts,
    for real measurements, gives same estimator as edwin's jackknife,
    except includes imaginary parts. deosn't allow arbitrary function input
    like edwin's jackknife function"""
    # print(asgn.shape, sgn.shape)
    assert asgn.shape[0] == sgn.shape[0], f"{asgn.shape},{sgn.shape}"
    # 0 axis is nbin
    nbin = sgn.shape[0]

    if nbin == 1:
        print("only one bin, thus zero error")
        return asgn[0] / sgn[0], np.zeros(asgn.shape[1:])

    jk_resample = np.full(asgn.shape, np.nan, dtype=complex)

    for j in range(nbin):
        jk_resample[j] = (np.sum(asgn, axis=0) - asgn[j]) / (np.sum(sgn) - sgn[j])
    jk_mean = np.mean(jk_resample, axis=0)
    jk_variance = np.mean(np.abs(jk_resample - jk_mean) ** 2, axis=0)
    jk_std = np.sqrt((nbin - 1) * jk_variance)

    return jk_mean, jk_std


def info(
    path: str, uneqlt: bool = False, show: bool = False, imagtol: float = 1e-2
) -> int:
    """return integer based on if the runs are complete

    If return 1, then directory doesn't have any completed MC runs
    If return 2, then imaginary parts of sign or density estimator too large
    If return -1, then not enough bins for maxent
    If return 0, then this directory has OK data, can proceed.
    """

    # if no *h5 files, likely pathname is wrong
    nfiles = len(glob.glob(path + "*.h5"))
    # print("nfiles = ", nfiles)
    if nfiles == 0:
        print(f"\033[4mNo *h5 files in {path}\033[0m")
        return 1

    (
        U,
        beta,
        mu,
        Nx,
        Ny,
        bps,
        N,
        L,
        dt,
        n_sweep_warm,
        n_sweep_meas,
        period_uneqlt,
        period_eqlt,
        meas_energy_corr,
        meas_bond_corr,
    ) = util.load_firstfile(
        path,
        "metadata/U",
        "metadata/beta",
        "metadata/mu",
        "metadata/Nx",
        "metadata/Ny",
        "metadata/bps",
        "params/N",
        "params/L",
        "params/dt",
        "params/n_sweep_warm",
        "params/n_sweep_meas",
        "params/period_uneqlt",
        "params/period_eqlt",
        "params/meas_energy_corr",
        "params/meas_bond_corr",
    )

    if show:
        print(path)
        print(
            f"{Nx}x{Ny}    U={U}     beta={beta:.3g}    dt={dt:.3g}\n"
            + f"mu={mu:.3f}\tn_sweep_warm={n_sweep_warm}\tn_sweep_meas={n_sweep_meas}\t"
            + f"period_eqlt={period_eqlt}\tnbins target={nfiles}\n"
            + f"Uneqlt period? {period_uneqlt}\tEnergy corr? {bool(meas_energy_corr)}\t"
            + f"Bond corr? {bool(meas_bond_corr)}"
        )
        try:
            meas_2bond, meas_thermal = util.load_firstfile(
                path, "params/meas_2bond_corr", "params/meas_thermal"
            )
            print(
                f"2 bond corr? {bool(meas_2bond)}\t"
                + f"thermal corr? {bool(meas_thermal)}"
            )
        except KeyError:
            print("At least one of 2 bond or thermal measurements toggled to False")

        try:
            meas_chiral = util.load_firstfile(path, "params/meas_chiral")[0]
            print(f"meas chiral? {bool(meas_chiral)}")
        except KeyError:
            print("No chiral measurements")

    # calculate expected n_sample
    if uneqlt:
        try:
            ns, s = util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/sign")
        except KeyError:
            print("data_analysis.info(): No uneqlt data saved")
            return 1
        ns_expect = n_sweep_meas // period_uneqlt
    else:
        ns, s, d = util.load(
            path, "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/density"
        )
        ns_expect = n_sweep_meas * L // period_eqlt

    # no data or incomplete
    if ns.max() < ns_expect:
        print(f"\033[4mshould have {ns_expect} data, actual: {ns.max()}\033[0m")
        return 1

    if ns.max() > ns_expect:
        print(
            f"\033[91mshould have {ns_expect} data, actual: {ns.max()}, need check\033[91m"
        )
        raise ValueError

    # at least one bin is full, print info
    nbin_all = ns.shape[0]
    mask = ns == ns.max()
    nbin = mask.sum()

    # use only filled bins by applying mask
    ns, s = ns[mask], s[mask]
    avgsgn, sgn_stderr = jackknife(s, ns)
    # print sign (and density) real part and its error
    if uneqlt:
        if show:
            colorize = "\033[93m " if nbin < nbin_all else "\033[0m "
            print(
                f"uneqlt complete: {colorize} {nbin}/{nbin_all} \033[0m\ts={avgsgn:.3g} SE(s)={sgn_stderr:.3g}"
            )
    else:
        d = d[mask]
        avgd, d_stderr = jackknife(d, s)
        avgd = np.mean(avgd)
        d_stderr = np.mean(d_stderr)

        if show:
            colorize = "\033[93m " if nbin < nbin_all else "\033[0m "
            print(
                f"eqlt complete: {colorize} {nbin}/{nbin_all}\033[0m",
                f"\t<sign>={avgsgn:.3g} SE(<sign>)={sgn_stderr:.3g}",
                f"\x1b[1m  n={avgd:.3g} SE(n)={d_stderr:.3g} \x1b[0m",
            )
        # imaginary part of density mean is too large
        if np.abs(avgd.imag) / np.abs(avgd.real) > imagtol:
            print(
                f"\033[91md imag/real norm = {np.abs(avgd.imag)/np.abs(avgd.real)} > {imagtol}\033[0m"
            )
            return 2
        # standard error of density mean estimator is too large
        if d_stderr / np.abs(avgd) > imagtol * 5:
            print(
                f"\033[91mSE(d)/abs(d) = {d_stderr/np.abs(avgd):.3g} > {imagtol*5}\033[0m"
            )
            return 2

    # imaginary part of sign estimator is too large
    if np.abs(avgsgn.imag) / np.abs(avgsgn.real) > imagtol:
        print(
            f"\033[91ms imag/real norm = {np.abs(avgsgn.imag)/np.abs(avgsgn.real)} > {imagtol}\033[0m"
        )
        return 2

    # standard error of sign mean estimator is too large
    if sgn_stderr / np.abs(avgsgn) > imagtol * 5:
        print(
            f"\033[91mSE(s)/abs(s) = {sgn_stderr/np.abs(avgsgn):.3g} > {imagtol*5}\033[0m"
        )
        return 2

    # Check for maxent
    if uneqlt and nbin <= 2 * L:
        print(f"\033[93m{nbin}/{nbin_all} bins not sufficient for maxent \033[0m")
        return -1

    return 0


def infer_metadata(path: str, warn: bool = False) -> tuple[int, int]:
    """For backwards compatibility with older DQMC versions.
    Infer Norb, plaq_per_cell, trans_sym, geometry. Issue warnings
    if no inference can be made

    [description]

    Args:
        path ([type]): [description]

    Returns:
        (int,int) : Norb, trans_sym
    """

    Nx, Ny, num_i = util.load_firstfile(
        path, "metadata/Nx", "metadata/Ny", "params/num_i"
    )

    # Issue warnings when processing old data
    if warn:
        try:
            commit = util.load_firstfile(path, "metadata/commit")[0]
        except KeyError:
            print(
                f"\033[93mWarning\033[0m: No commit info saved in metadata, inferring that"
                + " this .h5 file was generated before commit 7d986ee made on Apr 16, 2022"
            )

    if warn:
        try:
            twistx, twisty = util.load_firstfile(
                path, "metadata/twistx", "metadata/twisty"
            )
        except KeyError:
            print(
                "\033[93mWarning\033[0m: No twistx, twisty info saved in metadata, inferring they are both 0.0"
            )

    # Issue warnings when processing old data
    try:
        Norb = util.load_firstfile(path, "metadata/Norb")[0]
    except KeyError:
        Norb = num_i if num_i < Nx * Ny else num_i // (Nx * Ny)
        if warn:
            print(
                f"\033[93mWarning\033[0m: No Norb info saved in metadata, inferring from num_i: Norb={Norb}"
            )

    try:
        plaq_per_cell = util.load_firstfile(path, "metadata/plaq_per_cell")[0]
    except KeyError:
        plaq_per_cell = num_i if num_i < Nx * Ny else num_i // (Nx * Ny)
        if warn:
            print(
                "\033[93mWarning\033[0m: No plaq_per_cell info saved in metadata, plaq_per_cell unknown"
            )

    try:
        trans_sym = util.load_firstfile(path, "metadata/trans_sym")[0]
    except KeyError:
        trans_sym = not (num_i >= Nx * Ny)
        if warn:
            print(
                f"\033[93mWarning\033[0m: No trans_sym toggle saved in metadata, inferring from num_i: {trans_sym}"
            )

    return Norb, trans_sym


# return size: (Nx, Ny, Norb, Norb) or (Ncell, Norb, Ncell, Norb)
# TODO: check that this is actually the correct mapping order, and F order is OK?
def eqlt_meas_ij(
    path: str, meas_list: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if info(path, uneqlt=False, show=False) == 1:
        raise OSError("No completed MC bin")

    Norb, trans_sym = infer_metadata(path)

    Nx, Ny, U, beta = util.load_firstfile(
        path, "metadata/Nx", "metadata/Ny", "metadata/U", "metadata/beta"
    )
    ns, s = util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign")
    # at least one bin is full, print info
    # nbin_all = ns.shape[0]
    mask = ns == ns.max()
    nbin = mask.sum()
    dm_dict = {}
    de_dict = {}
    for meas_name in meas_list:
        #      y
        # Ny-1  |
        #     . |
        #     . |
        #     4 |________ (pi, pi)
        #     3 |        |
        #     2 |        |
        #     1 |        |
        #     0 |________|_______ x
        #       0 1 2 3 4 5 ... Nx-1
        #    (0, 0)

        s, dat = util.load(path, "meas_eqlt/sign", f"meas_eqlt/{meas_name}")
        # use only filled bins by applying mask
        s, dat = s[mask], dat[mask]
        datm, date = jackknife(dat, s)
        if trans_sym:
            if Norb == 1:
                datm = np.reshape(datm, (Nx, Ny), order="F")
                date = np.reshape(date, (Nx, Ny), order="F")
            else:
                datm = np.reshape(datm, (Nx, Ny, Norb, Norb), order="F")
                date = np.reshape(date, (Nx, Ny, Norb, Norb), order="F")
        else:
            if Norb == 1:
                datm = np.reshape(datm, (Nx * Ny, Nx * Ny), order="F")
                date = np.reshape(date, (Nx * Ny, Nx * Ny), order="F")
            else:
                datm = np.reshape(datm, (Nx * Ny, Norb, Nx * Ny, Norb), order="F")
                date = np.reshape(date, (Nx * Ny, Norb, Nx * Ny, Norb), order="F")

        dm_dict[meas_name] = datm
        de_dict[meas_name] = date

    return dm_dict, de_dict


def eqlt_meas_ijkl(
    path: str, meas_list: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if info(path, uneqlt=False, show=False) == 1:
        raise OSError("No completed MC bin")

    Norb, trans_sym = infer_metadata(path)
    assert trans_sym

    Nx, Ny = util.load_firstfile(path, "metadata/Nx", "metadata/Ny")
    ns, s = util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign")
    # at least one bin is full, print info
    # nbin_all = ns.shape[0]
    mask = ns == ns.max()
    dm_dict = {}
    de_dict = {}
    for meas_name in meas_list:
        s, dat = util.load(path, "meas_eqlt/sign", f"meas_eqlt/{meas_name}")
        # use only filled bins by applying mask
        s, dat = s[mask], dat[mask]
        datm, date = jackknife(dat, s)
        datm = np.reshape(datm, (Nx, Ny, Norb, Norb, Nx, Ny, Norb, Norb), order="F")
        date = np.reshape(date, (Nx, Ny, Norb, Norb, Nx, Ny, Norb, Norb), order="F")

        dm_dict[meas_name] = datm
        de_dict[meas_name] = date

    return dm_dict, de_dict


# return size: (plaq_per_cell,) or (plaq_per_cell, Ncell)
def eqlt_meas_plaq(
    path: str, meas_list: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if info(path, uneqlt=False, show=False) == 1:
        raise OSError("No completed MC bin")

    Norb, trans_sym = infer_metadata(path)

    Nx, Ny, U, beta = util.load_firstfile(
        path, "metadata/Nx", "metadata/Ny", "metadata/U", "metadata/beta"
    )

    plaq_per_cell = 1

    ns, s = util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign")

    # at least one bin is full, print info
    # nbin_all = ns.shape[0]
    mask = ns == ns.max()
    nbin = mask.sum()
    dm_dict = {}
    de_dict = {}

    ns, s = ns[mask], s[mask]
    for meas_name in meas_list:
        # if meas_name == "chi":
        # if not trans_sym: raise NotImplementedError(meas_name)
        dat = util.load(path, f"meas_eqlt/{meas_name}")[0]
        dat = dat[mask]
        # dat = dat if trans_sym else np.reshape(dat, (-1, Nx, Ny), order='F')
        # print(dat.shape)
        datm, date = jackknife(dat, s)
        # else:
        #     # print(meas_name)
        #     raise NotImplementedError(meas_name)
        dm_dict[meas_name] = datm
        de_dict[meas_name] = date

    return dm_dict, de_dict


def eqlt_meas_i(
    path: str, meas_list: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Given a list of measurement names and data path directory,
    returns:
        A tuple of dict[str, np.ndarray] (dm, de).
        dm contains the measurement means,
        de contains the measurement errors.
        dict.values() have shape (Norb,) or (Ncell*Norb,)"""

    if info(path, uneqlt=False, show=False) == 1:
        raise OSError("No completed MC bin")

    Norb, trans_sym = infer_metadata(path)

    Nx, Ny, U, beta = util.load_firstfile(
        path, "metadata/Nx", "metadata/Ny", "metadata/U", "metadata/beta"
    )

    ns, s = util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign")
    # at least one bin is full, print info
    # nbin_all = ns.shape[0]
    mask = ns == ns.max()
    nbin = mask.sum()
    dm_dict = {}
    de_dict = {}

    ns, s = ns[mask], s[mask]
    for meas_name in meas_list:
        if meas_name == "mz":
            d, double_occ = util.load(path, "meas_eqlt/density", "meas_eqlt/double_occ")
            # print(d.shape, double_occ.shape)
            # use only filled bins by applying mask
            d, double_occ = d[mask], double_occ[mask]
            datm, date = jackknife(d - 2 * double_occ, s)

        elif "density" in meas_name:
            dat = util.load(path, f"meas_eqlt/{meas_name}")[0]
            # use only filled bins by applying mask
            dat = dat[mask]
            datm, date = jackknife(dat, s)

        elif meas_name == "sign":
            datm, date = jackknife(s, ns)
        # TODO: IMPLEMENT ME
        elif meas_name == "energy":
            assert trans_sym
            d, g00, double_occ = util.load(
                path, "meas_eqlt/density", "meas_eqlt/g00", "meas_eqlt/double_occ"
            )
            g00 = np.reshape(g00, (-1, Nx, Ny, Norb, Norb), order="F")
            # use only filled bins by applying mask
            d, double_occ, g00 = (
                d[mask],
                double_occ[mask],
                g00[mask],
            )
            # kinetic terms = t1 + t2
            t1 = 0
            t2 = 0
            t3 = U * (double_occ)
            datm, date = jackknife(t1 + t2 + t3, s)
        # TODO: IMPLEMENT ME
        elif meas_name == "kinetic":
            assert trans_sym
            g00 = util.load(path, "meas_eqlt/g00")[0]
            g00 = np.reshape(g00, (-1, Nx, Ny, Norb, Norb), order="F")
            # use only filled bins by applying mask
            g00 = g00[mask]
            # kinetic terms = t1 + t2
            t1 = 0
            t2 = 0
            datm, date = jackknife(g00[:, 0, 0, 0], s)

        # TODO: IMPLEMENT ME
        elif meas_name == "potential":
            assert trans_sym
            d, double_occ = util.load(path, "meas_eqlt/density", "meas_eqlt/double_occ")
            # use only filled bins by applying mask
            d, double_occ = d[mask], double_occ[mask]

            # potential energy = t3
            t3 = U * (double_occ)
            datm, date = jackknife(t3, s)

        elif meas_name == "compress":
            assert trans_sym
            # special case: use Edwin's jackknife function, so return avg of type float
            nn, dens = util.load(path, "meas_eqlt/nn", "meas_eqlt/density")
            nn, dens = nn[mask], dens[mask]
            nn = np.reshape(nn, (-1, Nx, Ny, Norb, Norb), order="F")
            nn_q0 = np.sum(nn, axis=(1, 2))
            tryf = lambda s, sx, sy: beta * (
                (sx.T / s.T).T.real - (sy.T / s.T).T.real ** 2 * Nx * Ny
            )
            datm, date = util.jackknife(
                s, np.diagonal(nn_q0, axis1=1, axis2=2), dens, f=tryf
            )
        else:
            raise NotImplementedError(meas_name)
        dm_dict[meas_name] = datm
        de_dict[meas_name] = date

    return dm_dict, de_dict


def readmu(
    path: str, fname: str, show: bool = False, nflux_string: str = "?"
) -> tuple[list[str], list[str], dict[tuple[str, str], float]]:
    betas_list = []
    Us_list = []
    mu_dict = {}

    print(f"mu info source: {path}{fname}")
    with open(path + fname, "r") as f:
        c = f.readlines()
        for i in range(len(c)):
            if "target n = " in c[i]:
                nt = float(c[i][10:-1])
            if "beta" in c[i]:
                bi = c[i].find("beta")
                ui = c[i].find("U")
                bs = c[i][bi + 4 : ui - 1]
                Us = c[i][ui + 1 : -1]
                if bs not in betas_list:
                    betas_list.append(bs)
                if Us not in Us_list:
                    Us_list.append(Us)
                mu = float(c[i - 1][1:-2])
                mu_dict[(bs, Us)] = mu  # full precision float

    beta_arr = np.array(list(map(float, betas_list)))
    order = np.argsort(beta_arr)
    U_arr = np.array(list(map(float, Us_list)))
    Uorder = np.argsort(U_arr)
    nT = beta_arr.shape[0]
    nU = U_arr.shape[0]
    # if want to plot chemical potential as function of temperature for each U
    if show:
        print(f"readmu: target filling = {nt}")
        print(f"readmu: beta list = {betas_list}", len(betas_list))
        print(f"readmu: U list = {Us_list}", len(Us_list))
        # colors = plt.cm.viridis(np.linspace(1, 0, nU))
        for j in Uorder:
            Us = Us_list[j]
            mu_arr = np.empty(nT)
            for i in range(nT):
                try:
                    mu_arr[i] = mu_dict[(betas_list[i], Us)]
                except KeyError as e:
                    print("KeyError: (beta, U) = ", e)
                    mu_arr[i] = np.nan
            plt.plot(
                1 / beta_arr[order],
                mu_arr[order],
                f"{misc.marker_list[j]}-",
                color=plt.cm.tab10(int(nflux_string)),
                label=f"U={Us}, nflux={nflux_string}",
            )
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel(r"temperature $T/t$")
        plt.ylabel(r"chemical potential $\mu$")
        plt.grid(True)
        plt.xscale("log")
        plt.xlim(np.min(1 / beta_arr) / 2, np.max(1 / beta_arr) * 2)

    return betas_list, Us_list, mu_dict
