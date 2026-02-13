import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home1/wyndham/dqmc-dev/util/")
import my_maxent
from typing import Optional, Any

import maxent  # edwin's maxent implementation
import util  # Edwin's util file


def plot_imag_time(path: str, geometry: str):

    # try:
    #     Norb = util.load_firstfile(path, "metadata/Norb")[0]
    # except KeyError as e:
    #     print("KeyError:", e)
    #     Norb = 1

    Nx, Ny, U, tp, beta, L, dt = util.load_firstfile(
        path,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/U",
        "metadata/t'",
        "metadata/beta",
        "params/L",
        "params/dt",
    )
    ns, s, gt0 = util.load(
        path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/gt0"
    )
    # reshape jj into more intuitive shape
    gt0 = np.reshape(gt0, (-1, Nx, Ny, L), order="F")
    gt0 = np.transpose(gt0, (0, 3, 1, 2))

    # use only completed bins
    mask = ns == ns.max()
    nbin = mask.sum()
    ns, s, gt0 = ns[mask], s[mask], gt0[mask]

    print("gt0 real/imag = ", np.linalg.norm(gt0.imag) / np.linalg.norm(gt0.real))

    plt.figure()
    plt.ylabel("r = (0, 0)")
    test = (gt0[:, :, 0, 0] / np.mean(s)).real
    # note: errorbar is += 1 std error of mean
    plt.errorbar(
        np.arange(L) * dt,
        test.mean(0),
        yerr=np.std(test, axis=0, ddof=1) / np.sqrt(nbin),
        fmt="s",
        label="data",
    )
    # print(np.std(test, axis=0, ddof=1)/np.sqrt(nbin))
    # plt.axhline(y=0, color='k')
    discretesum = np.sum((dt * test), axis=-1)
    # print(.shape)
    print("disrete sum mean", np.mean(discretesum))
    print("discrete sum std", np.std(discretesum))
    plt.show()


def local_dos(
    path: str,
    geometry: str,
    omega: np.ndarray,
    domega: np.ndarray,
    bs: int,
    method: str = "BT",
    anneal_arr: Optional[np.ndarray] = None,
    checks: bool = False,
    phs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    nw = omega.shape[0]

    Nx, Ny, U, tp, beta, L, dt = util.load_firstfile(
        path,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/U",
        "metadata/t'",
        "metadata/beta",
        "params/L",
        "params/dt",
    )
    ns, s, gt0 = util.load(
        path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/gt0"
    )
    # reshape jj into more intuitive shape
    gt0 = np.reshape(gt0, (-1, Nx*Ny, Ny*Nx, L), order="F")
    gt0 = np.transpose(gt0, (0, 3, 2, 1))

    # use only completed bins
    mask = ns == ns.max()
    nbin = mask.sum()
    ns, s, gt0 = ns[mask], s[mask], gt0[mask]

    gt0_ii = np.zeros((nbin, L), dtype=np.complex128)
    for i in range(Nx*Ny):
        gt0_ii += gt0[:,:,i,i]
    gt0_ii /= (Nx * Ny)

    A_bs = np.full((bs, nw), np.nan, dtype=float)
    dos_bs = np.full((bs, nw), np.nan, dtype=float)

    for i in range(bs):
        print("bs rep #", i)
        resample = np.random.randint(nbin, size=nbin)  # sample with replacement
        gt0_ii_bs = (gt0_ii[resample] / np.mean(s[resample])).real  # divide by sign

        pre = my_maxent.Preprocess(
            gt0_ii_bs,
            dt,
            beta,
            grid_info=(omega, domega),
            op_type="fermion",
            sym=phs,
            model_arr=anneal_arr,
        )

        A = my_maxent.MaxEnt(pre, method=method, printout=checks, inspect=checks)
        A_bs[i, :] = A
        dos_bs[i, :] = (A / domega) * pre["norm"]

    if checks:
        my_maxent.plot_bs_results(
            omega, gt0_ii / np.mean(s), L, dt, "DOS", pre, A_bs, dos_bs
        )

    return np.nanmean(A_bs, axis=0), dos_bs


def pair_local_dos(
    path: str,
    geometry: str,
    omega: np.ndarray,
    domega: np.ndarray,
    bs: int,
    method: str = "BT",
    anneal_arr: Optional[np.ndarray] = None,
    checks: bool = False,
    phs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    nw = omega.shape[0]

    bps = 3
    Nx, Ny, U, tp, beta, L, dt = util.load_firstfile(
        path,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/U",
        "metadata/t'",
        "metadata/beta",
        "params/L",
        "params/dt",
    )
    ns, s, pair_bb = util.load(
        path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/pair_bb"
    )
    # reshape jj into more intuitive shape
    pair_bb = np.reshape(pair_bb, (-1, Nx*Ny, Ny*Nx, bps, bps, L), order="F")
    pair_bb = np.transpose(pair_bb, (0, 5, 4, 3, 2, 1))

    # use only completed bins
    mask = ns == ns.max()
    nbin = mask.sum()
    ns, s, pair_bb = ns[mask], s[mask], pair_bb[mask]

    gt0_ib_ib = np.zeros((nbin, L), dtype=np.complex128)
    for i in range(Nx * Ny):
        for i_btype in range(bps):
            gt0_ib_ib += pair_bb[:, :, i_btype, i_btype, i, i]

    gt0_ib_ib /= (Nx * Ny * bps)

    A_bs = np.full((bs, nw), np.nan, dtype=float)
    pair_dos_bs = np.full((bs, nw), np.nan, dtype=float)

    for i in range(bs):
        print("bs rep #", i)
        resample = np.random.randint(nbin, size=nbin)  # sample with replacement
        gt0_ib_ib_bs = (gt0_ib_ib[resample] / np.mean(s[resample])).real  # divide by sign

        pre = my_maxent.Preprocess(
            gt0_ib_ib_bs,
            dt,
            beta,
            grid_info=(omega, domega),
            op_type="boson",
            sym=phs,
            model_arr=anneal_arr,
        )
        A = my_maxent.MaxEnt(pre, method=method, printout=checks, inspect=checks)
        A_bs[i, :] = A
        pair_dos_bs[i, :] = (A / domega) * pre["norm"]

    if checks:
        my_maxent.plot_bs_results(
            omega, gt0_ib_ib / np.mean(s), L, dt, "Pair DOS", pre, A_bs, pair_dos_bs
        )

    return np.nanmean(A_bs, axis=0), pair_dos_bs

def get_site_bond(site, b_type, Nx, Ny):
    site_x = site % Nx
    site_y = site // Nx
    # if the bond is hosted by the original site just hand it back
    if b_type == 0 or b_type == 1 or b_type == 2:
        new_site = site
        new_b_type = b_type
        return site, b_type
    # the bond is hosted by the site up and to the right. It is bond 0 for site + Nx
    if b_type == 3:
        new_site_y = (site_y + 1) % Ny 
        new_site = site_x + new_site_y * Nx
        new_b_type = 0
        return new_site, new_b_type
    # the bond is hosted by the site up and to the left. It is bond 1 there.
    if b_type == 4:
        new_site_y = (site_y + 1) % Ny
        new_site_x = (site_x - 1 + Nx) % Nx
        new_site = new_site_x + new_site_y * Nx
        new_b_type = 1
        return new_site, new_b_type
    # the bond is hosted by the site to the left. It is bond 2 there.
    if b_type == 5:
        new_site_x = (site_x - 1 + Nx) % Nx
        new_site = new_site_x + site_y * Nx
        new_b_type = 2
        return new_site, new_b_type
    raise Exception("Something broken in bond mapping with get_site_bond")

def pairing_susceptibility(
    path: str,
    geometry: str,
    omega: np.ndarray,
    domega: np.ndarray,
    bs: int,
    qx: float,
    qy: float,
    rho_n: int,
    method: str = "BT",
    anneal_arr: Optional[np.ndarray] = None,
    checks: bool = False,
    phs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    nw = omega.shape[0]

    bps = 3
    v1 = np.array([1, 0])
    v2 = np.array([1/2, np.sqrt(3)/2])

    Nx, Ny, U, tp, beta, L, dt = util.load_firstfile(
        path,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/U",
        "metadata/t'",
        "metadata/beta",
        "params/L",
        "params/dt",
    )
    ns, s, pair_bb = util.load(
        path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/pair_bb"
    )
    # reshape jj into more intuitive shape
    pair_bb = np.reshape(pair_bb, (-1, Nx*Ny, Ny*Nx, bps, bps, L), order="F")
    pair_bb = np.transpose(pair_bb, (0, 5, 4, 3, 2, 1))

    # use only completed bins
    mask = ns == ns.max()
    nbin = mask.sum()
    ns, s, pair_bb = ns[mask], s[mask], pair_bb[mask]

    pair_bb_q = np.zeros((nbin, L), dtype=np.complex128)

    for i in range(Nx * Ny):
        i_x = i % Nx
        i_y = i // Nx
        r_i_x = (i_x-1) * v1[0] + (i_y-1) * v2[0]
        r_i_y = (i_x-1) * v1[1] + (i_y-1) * v2[1]
        for j in range(Nx * Ny):
            j_x = j % Nx
            j_y = j // Nx
            r_j_x = (j_x-1) * v1[0] + (j_y-1) * v2[0]
            r_j_y = (j_x-1) * v1[1] + (j_y-1) * v2[1]

            q_dot_r = qx * (r_j_x - r_i_x) + qy * (r_j_y - r_i_y)
            e_to_i_q_dot_R_prime_minus_R = np.exp(1j * q_dot_r)

            site_site_pair_total = np.zeros((nbin, L), dtype = np.complex128)

            for i_btype in range(6):
                a_i = (i_btype + 4) % 6
                actual_i, actual_i_btype = get_site_bond(site=i, b_type=i_btype, Nx=Nx, Ny=Ny)
                for j_btype in range(6):
                    a_j = (j_btype + 4) % 6
                    actual_j, actual_j_btype = get_site_bond(site=j, b_type=j_btype, Nx=Nx, Ny=Ny)

                    bond_bond_correlation = pair_bb[:, :, actual_j_btype, actual_i_btype, actual_j, actual_i]

                    M1_i = abs(i_x - 1)
                    M2_i = abs(i_y - 1)
                    M1_j = abs(j_x - 1)
                    M2_j = abs(j_y - 1)

                    if i_btype == 0 or i_btype == 3:
                        i_sign_prefactor = 1
                    elif i_btype == 1 or i_btype == 4:
                        i_sign_prefactor = pow(-1, M1_i)
                    elif i_btype == 2 or i_btype == 5:
                        i_sign_prefactor = pow(-1, M2_i)
                    else:
                        raise Exception("Something broken in sign assignment with M's")

                    if j_btype == 0 or j_btype == 3:
                        j_sign_prefactor = 1
                    elif j_btype == 1 or j_btype == 4:
                        j_sign_prefactor = pow(-1, M1_j)
                    elif j_btype == 2 or j_btype == 5:
                        j_sign_prefactor = pow(-1, M2_j)
                    else:
                        raise Exception("Something broken in sign assignment with M's")
                    
                    rho = np.exp(1j * rho_n * np.pi/3)
                    rho_contribution = np.power(rho, a_i - a_j)
                    pair_bb_q += rho_contribution * i_sign_prefactor * j_sign_prefactor * bond_bond_correlation * e_to_i_q_dot_R_prime_minus_R
    
    pair_bb_q /= (Nx * Ny)

    A_bs = np.full((bs, nw), np.nan, dtype=float)
    pair_dos_bs = np.full((bs, nw), np.nan, dtype=float)

    for i in range(bs):
        print("bs rep #", i)
        resample = np.random.randint(nbin, size=nbin)  # sample with replacement
        pair_bb_q_bs = (pair_bb_q[resample] / np.mean(s[resample])).real  # divide by sign

        pre = my_maxent.Preprocess(
            pair_bb_q_bs,
            dt,
            beta,
            grid_info=(omega, domega),
            op_type="boson",
            sym=phs,
            model_arr=anneal_arr,
        )
        A = my_maxent.MaxEnt(pre, method=method, printout=checks, inspect=checks)
        A_bs[i, :] = A
        pair_dos_bs[i, :] = (A / domega) * pre["norm"]

    if checks:
        my_maxent.plot_bs_results(
            omega, pair_bb_q / np.mean(s), L, dt, "Pair Susceptibility", pre, A_bs, pair_dos_bs
        )

    return np.nanmean(A_bs, axis=0), pair_dos_bs

def q_resolved_bond_bond(
    path: str,
    geometry: str,
    omega: np.ndarray,
    domega: np.ndarray,
    bs: int,
    qx: float,
    qy: float,
    method: str = "BT",
    anneal_arr: Optional[np.ndarray] = None,
    checks: bool = False,
    phs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    nw = omega.shape[0]

    bps = 3
    v1 = np.array([1, 0])
    v2 = np.array([1/2, np.sqrt(3)/2])
    # here we say that the location is the bond center

    Nx, Ny, U, tp, beta, L, dt = util.load_firstfile(
        path,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/U",
        "metadata/t'",
        "metadata/beta",
        "params/L",
        "params/dt",
    )
    ns, s, pair_bb = util.load(
        path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/pair_bb"
    )
    # reshape jj into more intuitive shape
    pair_bb = np.reshape(pair_bb, (-1, Nx*Ny, Ny*Nx, bps, bps, L), order="F")
    pair_bb = np.transpose(pair_bb, (0, 5, 4, 3, 2, 1))

    # use only completed bins
    mask = ns == ns.max()
    nbin = mask.sum()
    ns, s, pair_bb = ns[mask], s[mask], pair_bb[mask]

    pair_bb_q = np.zeros((nbin, L), dtype=np.complex128)

    # at each site get every bond
    # we use the position of the bond center, and all bond centers form a kagome
    # lattice that is dual to the original lattice. So, allowed q's will be defined
    # through this kagome 
    for i in range(Nx * Ny):
        i_x = i % Nx
        i_y = i // Nx
        r_i_x = (i_x-1) * v1[0] + (i_y-1) * v2[0]
        r_i_y = (i_y-1) * v1[1] + (i_y-1) * v2[1]
        for j in range(Nx * Ny):
            j_x = j % Nx
            j_y = j // Nx
            r_j_x = (j_x-1) * v1[0] + (j_y-1) * v2[0]
            r_j_y = (j_y-1) * v1[1] + (j_y-1) * v2[1]

            for i_btype in range(3):
                i_b_position_x = r_i_x + bond_type_to_displacement_vector(i_btype)[0]
                i_b_position_y = r_i_y + bond_type_to_displacement_vector(i_btype)[1]
                for j_btype in range(3):
                    j_b_position_x = r_j_x + bond_type_to_displacement_vector(j_btype)[0]
                    j_b_position_y = r_j_y + bond_type_to_displacement_vector(j_btype)[1]

                    q_dot_r = qx * (i_b_position_x + j_b_position_x) + qy * (i_b_position_y + j_b_position_y)
                    e_to_i_q_dot_R_minus_R_prime = np.exp(1j * q_dot_r)
                    # apply bonds to get positions 
                    
                    bond_bond_correlation = pair_bb[:, :, j_btype, i_btype, j, i]

                    M1_i = abs((i % Nx) - 1)
                    M2_i = abs((i // Nx) - 1)
                    M1_j = abs((j % Nx) - 1)
                    M2_j = abs((j // Nx) - 1)

                    if i_btype == 0 or i_btype == 3:
                        i_sign_prefactor = 1
                    elif i_btype == 1 or i_btype == 4:
                        i_sign_prefactor = pow(-1, M1_i)
                    elif i_btype == 2 or i_btype == 5:
                        i_sign_prefactor = pow(-1, M2_i)
                    else:
                        raise Exception("Something broken in sign assignment with M's")

                    if j_btype == 0 or j_btype == 3:
                        j_sign_prefactor = 1
                    elif j_btype == 1 or j_btype == 4:
                        j_sign_prefactor = pow(-1, M1_j)
                    elif j_btype == 2 or j_btype == 5:
                        j_sign_prefactor = pow(-1, M2_j)
                    else:
                        raise Exception("Something broken in sign assignment with M's")
                    
                    pair_bb_q += i_sign_prefactor * j_sign_prefactor * bond_bond_correlation * e_to_i_q_dot_R_minus_R_prime

    # pair_bb_q /= (Nx * Ny * bps)

    A_bs = np.full((bs, nw), np.nan, dtype=float)
    pair_dos_bs = np.full((bs, nw), np.nan, dtype=float)

    for i in range(bs):
        print("bs rep #", i)
        resample = np.random.randint(nbin, size=nbin)  # sample with replacement
        pair_bb_q_bs = (pair_bb_q[resample] / np.mean(s[resample])).real  # divide by sign

        pre = my_maxent.Preprocess(
            pair_bb_q_bs,
            dt,
            beta,
            grid_info=(omega, domega),
            op_type="boson",
            sym=phs,
            model_arr=anneal_arr,
        )
        A = my_maxent.MaxEnt(pre, method=method, printout=checks, inspect=checks)
        A_bs[i, :] = A
        pair_dos_bs[i, :] = (A / domega) * pre["norm"]

    if checks:
        my_maxent.plot_bs_results(
            omega, pair_bb_q / np.mean(s), L, dt, f"Bond-Bond qx={qx} qy={qy}", pre, A_bs, pair_dos_bs
        )

    return np.nanmean(A_bs, axis=0), pair_dos_bs