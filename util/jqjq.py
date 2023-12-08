import numpy as np
import util
import data_analysis as da

# one-hop bond dx, dy
dx_arr = [1, 0, 1, -1]
dy_arr = [0, 1, 1, 1]


def get_component(path: str, name: str = "j2j2"):
    """
    Address RAM overflow kill:
        By separating components out, python only keeps full
        shape (Nbin_completed, L, b[2]ps, b[2]ps, Nx, Ny) of one type of correlator
        in memory at a time. Performs summation over (Nx, Ny) spatial components to
        return correlator(Q=0) for maxent analysis. Not divided by sign
    """

    Nx, Ny, bps, b2ps, N, L = util.load_firstfile(
        path,
        "metadata/Nx",
        "metadata/Ny",
        "metadata/bps",
        "metadata/b2ps",
        "params/N",
        "params/L",
    )

    ns, correlator = util.load(path, "meas_uneqlt/n_sample", f"meas_uneqlt/{name}")

    if name == "j2j2":
        # four phases
        correlator.shape = -1, L, b2ps, b2ps, Ny, Nx
    elif name == "jj2" or name == "jnj2":
        # three phases
        correlator.shape = -1, L, b2ps, bps, Ny, Nx
    elif name == "j2j" or name == "j2jn":
        # three phases
        correlator.shape = -1, L, bps, b2ps, Ny, Nx
    elif (
        name == "jnj"
        or name == "jjn"
        or name == "jnjn"
        or name == "jj"
        or name == "new_jnj"
        or name == "new_jjn"
    ):
        # two phases
        correlator.shape = -1, L, bps, bps, Ny, Nx
    else:
        raise ValueError("invalid correlator name")

    # use only completed bins
    mask = ns == ns.max()
    correlator = correlator[mask]
    # take q == 0, don't divide by sign
    correlator_q0 = correlator.sum((-1, -2))

    return correlator_q0


def thermal_sum(path: str, q0_corrs) -> dict[str, np.ndarray]:
    """
    Take tuple of correlators, each element of shape (Nbin_completed, L, b[2]ps, b[2]ps),
    Perform appropriate summation over bond types to obtain JNJN, JQJN, JNJQ, JQJQ
    Return a dictionary with named elements.
    E.g. result["JQJN"].shape = (4, Nbin_complete, L)
    Input is not divided by sign.
    """

    U, mu, beta, Nx, Ny, bps, b2ps, nflux, tp, N, L, dt = util.load_firstfile(
        path,
        "metadata/U",
        "metadata/mu",
        "metadata/beta",
        "metadata/Nx",
        "metadata/Ny",
        "metadata/bps",
        "metadata/b2ps",
        "metadata/nflux",
        "metadata/t'",
        "params/N",
        "params/L",
        "params/dt",
    )

    j2j2_q0, jj2_q0, j2j_q0, jnj2_q0, j2jn_q0, jjn_q0, jnj_q0, jnjn_q0, jj_q0 = q0_corrs

    # bond type t factors
    t_arr = [1, 1, tp, tp]
    # 2bond type t factors
    if "half-fill-tp" in path and nflux == 0:
        print("using legacy t2_arr!")
        t2_arr = [
            1 + 2 * tp**2,
            1 + 2 * tp**2,
            2,
            2,
            2 * tp,
            2 * tp,
            2 * tp,
            2 * tp,
            4 * tp,
            4 * tp,
            tp**2,
            tp**2,
        ]
    else:
        t2_arr = [1, 1, 1, 1, tp, tp, tp, tp, tp, tp, tp**2, tp**2]
    # my two-hop bond dx,dy distances
    dx2_arr = [2, 0, 1, -1, 2, 1, -2, -1, 0, 1, 2, -2]
    dy2_arr = [0, 2, 1, 1, 1, 2, 1, 2, 1, 0, 2, 2]

    result_dict = {}

    # bond-bond types: jj, jnj, jjn, jnjn
    jj_xx = jj_yy = 0
    jnjn_xx = jnjn_yy = 0
    jnj_xx = jnj_yy = 0
    jjn_xx = jjn_yy = 0
    jj_xy = jj_yx = 0
    jnjn_xy = jnjn_yx = 0
    jnj_xy = jnj_yx = 0
    jjn_xy = jjn_yx = 0
    for itype in range(bps):
        for jtype in range(bps):
            jj_xx += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dx_arr[jtype]
                * jj_q0[:, :, itype, jtype]
            )
            jj_yy += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dy_arr[jtype]
                * jj_q0[:, :, itype, jtype]
            )
            jnjn_xx += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dx_arr[jtype]
                * jnjn_q0[:, :, itype, jtype]
            )
            jnjn_yy += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dy_arr[jtype]
                * jnjn_q0[:, :, itype, jtype]
            )
            jnj_xx += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dx_arr[jtype]
                * jnj_q0[:, :, itype, jtype]
            )
            jnj_yy += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dy_arr[jtype]
                * jnj_q0[:, :, itype, jtype]
            )
            jjn_xx += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dx_arr[jtype]
                * jjn_q0[:, :, itype, jtype]
            )
            jjn_yy += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dy_arr[jtype]
                * jjn_q0[:, :, itype, jtype]
            )

            jj_xy += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dy_arr[jtype]
                * jj_q0[:, :, itype, jtype]
            )
            jj_yx += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dx_arr[jtype]
                * jj_q0[:, :, itype, jtype]
            )
            jnjn_xy += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dy_arr[jtype]
                * jnjn_q0[:, :, itype, jtype]
            )
            jnjn_yx += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dx_arr[jtype]
                * jnjn_q0[:, :, itype, jtype]
            )
            jnj_xy += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dy_arr[jtype]
                * jnj_q0[:, :, itype, jtype]
            )
            jnj_yx += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dx_arr[jtype]
                * jnj_q0[:, :, itype, jtype]
            )
            jjn_xy += (
                t_arr[itype]
                * t_arr[jtype]
                * dx_arr[itype]
                * dy_arr[jtype]
                * jjn_q0[:, :, itype, jtype]
            )
            jjn_yx += (
                t_arr[itype]
                * t_arr[jtype]
                * dy_arr[itype]
                * dx_arr[jtype]
                * jjn_q0[:, :, itype, jtype]
            )

    # Note: this is the JNJN correlator with (-i)^2 factor!
    result_dict["JNJN"] = (-1) * np.stack((jj_xx, jj_yy, jj_xy, jj_yx), axis=0)

    # ==========================================================================
    # 2bond-2bond types: j2j2
    j2j2_xx = j2j2_yy = 0
    j2j2_xy = j2j2_yx = 0
    for itype in range(b2ps):
        for jtype in range(b2ps):
            j2j2_xx += (
                t2_arr[itype]
                * t2_arr[jtype]
                * dx2_arr[itype]
                * dx2_arr[jtype]
                * j2j2_q0[:, :, itype, jtype]
            )
            j2j2_yy += (
                t2_arr[itype]
                * t2_arr[jtype]
                * dy2_arr[itype]
                * dy2_arr[jtype]
                * j2j2_q0[:, :, itype, jtype]
            )
            j2j2_xy += (
                t2_arr[itype]
                * t2_arr[jtype]
                * dx2_arr[itype]
                * dy2_arr[jtype]
                * j2j2_q0[:, :, itype, jtype]
            )
            j2j2_yx += (
                t2_arr[itype]
                * t2_arr[jtype]
                * dy2_arr[itype]
                * dx2_arr[jtype]
                * j2j2_q0[:, :, itype, jtype]
            )

    # ==========================================================================
    # 2bond-1bond types
    j2jn_xx = j2jn_yy = 0
    j2j_xx = j2j_yy = 0
    j2jn_xy = j2jn_yx = 0
    j2j_xy = j2j_yx = 0
    for itype in range(bps):
        for jtype in range(b2ps):
            j2jn_xx += (
                t_arr[itype]
                * t2_arr[jtype]
                * dx_arr[itype]
                * dx2_arr[jtype]
                * j2jn_q0[:, :, itype, jtype]
            )
            j2jn_yy += (
                t_arr[itype]
                * t2_arr[jtype]
                * dy_arr[itype]
                * dy2_arr[jtype]
                * j2jn_q0[:, :, itype, jtype]
            )
            j2j_xx += (
                t_arr[itype]
                * t2_arr[jtype]
                * dx_arr[itype]
                * dx2_arr[jtype]
                * j2j_q0[:, :, itype, jtype]
            )
            j2j_yy += (
                t_arr[itype]
                * t2_arr[jtype]
                * dy_arr[itype]
                * dy2_arr[jtype]
                * j2j_q0[:, :, itype, jtype]
            )
            j2jn_xy += (
                t_arr[itype]
                * t2_arr[jtype]
                * dx_arr[itype]
                * dy2_arr[jtype]
                * j2jn_q0[:, :, itype, jtype]
            )
            j2jn_yx += (
                t_arr[itype]
                * t2_arr[jtype]
                * dy_arr[itype]
                * dx2_arr[jtype]
                * j2jn_q0[:, :, itype, jtype]
            )
            j2j_xy += (
                t_arr[itype]
                * t2_arr[jtype]
                * dx_arr[itype]
                * dy2_arr[jtype]
                * j2j_q0[:, :, itype, jtype]
            )
            j2j_yx += (
                t_arr[itype]
                * t2_arr[jtype]
                * dy_arr[itype]
                * dx2_arr[jtype]
                * j2j_q0[:, :, itype, jtype]
            )

    # ==========================================================================
    # 1bond-2bond types
    jnj2_xx = jnj2_yy = 0
    jj2_xx = jj2_yy = 0
    jnj2_xy = jnj2_yx = 0
    jj2_xy = jj2_yx = 0
    for itype in range(b2ps):
        for jtype in range(bps):
            jnj2_xx += (
                t2_arr[itype]
                * t_arr[jtype]
                * dx2_arr[itype]
                * dx_arr[jtype]
                * jnj2_q0[:, :, itype, jtype]
            )
            jnj2_yy += (
                t2_arr[itype]
                * t_arr[jtype]
                * dy2_arr[itype]
                * dy_arr[jtype]
                * jnj2_q0[:, :, itype, jtype]
            )
            jj2_xx += (
                t2_arr[itype]
                * t_arr[jtype]
                * dx2_arr[itype]
                * dx_arr[jtype]
                * jj2_q0[:, :, itype, jtype]
            )
            jj2_yy += (
                t2_arr[itype]
                * t_arr[jtype]
                * dy2_arr[itype]
                * dy_arr[jtype]
                * jj2_q0[:, :, itype, jtype]
            )
            jnj2_xy += (
                t2_arr[itype]
                * t_arr[jtype]
                * dx2_arr[itype]
                * dy_arr[jtype]
                * jnj2_q0[:, :, itype, jtype]
            )
            jnj2_yx += (
                t2_arr[itype]
                * t_arr[jtype]
                * dy2_arr[itype]
                * dx_arr[jtype]
                * jnj2_q0[:, :, itype, jtype]
            )
            jj2_xy += (
                t2_arr[itype]
                * t_arr[jtype]
                * dx2_arr[itype]
                * dy_arr[jtype]
                * jj2_q0[:, :, itype, jtype]
            )
            jj2_yx += (
                t2_arr[itype]
                * t_arr[jtype]
                * dy2_arr[itype]
                * dx_arr[jtype]
                * jj2_q0[:, :, itype, jtype]
            )

    # ==========================================================================
    # prefactors required to actually form JQ(tau)JQ(0) operator
    # see transport notes for detailed derivation
    # NOTE: the prefactor is 1/4, not 1/16, because in DQMC measurements,
    # each bond is only counted once. In notes, each bond is counted twice.
    # Extra (-1) factor due to (i)^2
    pre_arr = 1 / 4 * np.outer([1, -U, U + 2 * mu], [1, -U, U + 2 * mu]) * (-1)
    xx = (
        pre_arr[0, 0] * j2j2_xx
        + pre_arr[0, 1] * j2jn_xx
        + pre_arr[0, 2] * j2j_xx
        + pre_arr[1, 0] * jnj2_xx
        + pre_arr[1, 1] * jnjn_xx
        + pre_arr[1, 2] * jnj_xx
        + pre_arr[2, 0] * jj2_xx
        + pre_arr[2, 1] * jjn_xx
        + pre_arr[2, 2] * jj_xx
    )

    yy = (
        pre_arr[0, 0] * j2j2_yy
        + pre_arr[0, 1] * j2jn_yy
        + pre_arr[0, 2] * j2j_yy
        + pre_arr[1, 0] * jnj2_yy
        + pre_arr[1, 1] * jnjn_yy
        + pre_arr[1, 2] * jnj_yy
        + pre_arr[2, 0] * jj2_yy
        + pre_arr[2, 1] * jjn_yy
        + pre_arr[2, 2] * jj_yy
    )

    xy = (
        pre_arr[0, 0] * j2j2_xy
        + pre_arr[0, 1] * j2jn_xy
        + pre_arr[0, 2] * j2j_xy
        + pre_arr[1, 0] * jnj2_xy
        + pre_arr[1, 1] * jnjn_xy
        + pre_arr[1, 2] * jnj_xy
        + pre_arr[2, 0] * jj2_xy
        + pre_arr[2, 1] * jjn_xy
        + pre_arr[2, 2] * jj_xy
    )

    yx = (
        pre_arr[0, 0] * j2j2_yx
        + pre_arr[0, 1] * j2jn_yx
        + pre_arr[0, 2] * j2j_yx
        + pre_arr[1, 0] * jnj2_yx
        + pre_arr[1, 1] * jnjn_yx
        + pre_arr[1, 2] * jnj_yx
        + pre_arr[2, 0] * jj2_yx
        + pre_arr[2, 1] * jjn_yx
        + pre_arr[2, 2] * jj_yx
    )

    result_dict["j2j2"] = np.stack((j2j2_xx, j2j2_yy, j2j2_xy, j2j2_yx), axis=0)
    result_dict["j2jn"] = np.stack((j2jn_xx, j2jn_yy, j2jn_xy, j2jn_yx), axis=0)
    result_dict["j2j"] = np.stack((j2j_xx, j2j_yy, j2j_xy, j2j_yx), axis=0)
    result_dict["jnj2"] = np.stack((jnj2_xx, jnj2_yy, jnj2_xy, jnj2_yx), axis=0)
    result_dict["jnjn"] = np.stack((jnjn_xx, jnjn_yy, jnjn_xy, jnjn_yx), axis=0)
    result_dict["jnj"] = np.stack((jnj_xx, jnj_yy, jnj_xy, jnj_yx), axis=0)
    result_dict["jj2"] = np.stack((jj2_xx, jj2_yy, jj2_xy, jj2_yx), axis=0)
    result_dict["jjn"] = np.stack((jjn_xx, jjn_yy, jjn_xy, jjn_yx), axis=0)
    result_dict["jj"] = np.stack((jj_xx, jj_yy, jj_xy, jj_yx), axis=0)
    result_dict["JQJQ"] = np.stack((xx, yy, xy, yx), axis=0)
    result_dict["metadata"] = np.array((L, dt))

    # ==========================================================================
    # prefactors same for JQ(tau)JN(0),JN(tau)JQ(0)
    # TODO: check if this is actually consistent with bond, map definitions
    # JQJN and JNJQ might be flipped around
    # NOTE: No (-e) factor here, because using the particle-current J_N
    pre_arr = np.array([1, -U, U + 2 * mu]) * 1 / 2
    xx = pre_arr[0] * j2j_xx + pre_arr[1] * jnj_xx + pre_arr[2] * jj_xx
    yy = pre_arr[0] * j2j_yy + pre_arr[1] * jnj_yy + pre_arr[2] * jj_yy
    xy = pre_arr[0] * j2j_xy + pre_arr[1] * jnj_xy + pre_arr[2] * jj_xy
    yx = pre_arr[0] * j2j_yx + pre_arr[1] * jnj_yx + pre_arr[2] * jj_yx

    result_dict["JQJN"] = np.stack((xx, yy, xy, yx), axis=0)

    # ==========================================================================
    xx = pre_arr[0] * jj2_xx + pre_arr[1] * jjn_xx + pre_arr[2] * jj_xx
    yy = pre_arr[0] * jj2_yy + pre_arr[1] * jjn_yy + pre_arr[2] * jj_yy
    xy = pre_arr[0] * jj2_xy + pre_arr[1] * jjn_xy + pre_arr[2] * jj_xy
    yx = pre_arr[0] * jj2_yx + pre_arr[1] * jjn_yx + pre_arr[2] * jj_yx

    result_dict["JNJQ"] = np.stack((xx, yy, xy, yx), axis=0)
    # jjq_dict['metadata'] = (L,dt)
    return result_dict


def symmetry_checks(result_dict):
    atol = 1e-3
    # ============Electrical============================
    xx = result_dict["JNJN"][0].mean(0)
    yy = result_dict["JNJN"][1].mean(0)
    xy = result_dict["JNJN"][2].mean(0)
    yx = result_dict["JNJN"][3].mean(0)

    # JXJX is mostly real
    assert np.linalg.norm(xx.imag / xx.real) < atol
    assert np.linalg.norm(yy.imag / yy.real) < atol

    # JXJY is mostly imag, but only do this check when not all zero
    if not (np.allclose(xy, 0, atol=atol) and np.allclose(yx, 0, atol=atol)):
        assert (
            np.linalg.norm(xy.real / xy.imag) < atol
        ), f"{np.linalg.norm(xy.real/xy.imag)}"
        assert (
            np.linalg.norm(yx.real / yx.imag) < atol
        ), f"{np.linalg.norm(xy.real/xy.imag)}"
    else:
        assert np.allclose(xy, 0, atol=atol)
        assert np.allclose(yx, 0, atol=atol)

    # C4 symmetry
    assert np.allclose(xx, yy, atol=atol)
    assert np.allclose(xy, -yx, atol=atol)

    # =============Thermal===========================
    xx = result_dict["JQJQ"][0].mean(0)
    yy = result_dict["JQJQ"][1].mean(0)
    xy = result_dict["JQJQ"][2].mean(0)
    yx = result_dict["JQJQ"][3].mean(0)

    # JQXJQX is mostly real
    assert (
        np.linalg.norm(xx.imag / xx.real) < atol
    ), f"{np.linalg.norm(xx.imag/xx.real)}"
    assert (
        np.linalg.norm(yy.imag / yy.real) < atol
    ), f"{np.linalg.norm(yy.imag/yy.real)}"

    # JQXJQY is mostly imag, but only do this check when not all zero
    if not (np.allclose(xy, 0, atol=atol) and np.allclose(yx, 0, atol=atol)):
        assert (
            np.linalg.norm(xy.real / xy.imag) < atol
        ), f"{np.linalg.norm(xy.real/xy.imag)}"
        assert (
            np.linalg.norm(yx.real / yx.imag) < atol
        ), f"{np.linalg.norm(xy.real/xy.imag)}"
    else:
        assert np.allclose(xy, 0, atol=atol)
        assert np.allclose(yx, 0, atol=atol)

    # C4 symmetry
    assert np.allclose(xx, yy, atol=atol), f"{np.linalg.norm(xx-yy)}"
    assert np.allclose(xy, -yx, atol=atol)

    # =============Mixed===========================
    xx = result_dict["JNJQ"][0].mean(0)
    yy = result_dict["JNJQ"][1].mean(0)
    xy = result_dict["JNJQ"][2].mean(0)
    yx = result_dict["JNJQ"][3].mean(0)

    xx2 = result_dict["JQJN"][0].mean(0)
    yy2 = result_dict["JQJN"][1].mean(0)
    xy2 = result_dict["JQJN"][2].mean(0)
    yx2 = result_dict["JQJN"][3].mean(0)

    # JXJQX is mostly real, but only do this check when not all zero
    if not (np.allclose(xx, 0, atol=atol) and np.allclose(yy, 0, atol=atol)):
        assert (
            np.linalg.norm(xx.imag / xx.real) < atol
        ), f"{np.linalg.norm(xx.imag/xx.real)}"
        assert (
            np.linalg.norm(yy.imag / yy.real) < atol
        ), f"{np.linalg.norm(yy.imag/yy.real)}"
    else:
        assert np.allclose(xx, 0, atol=atol)
        assert np.allclose(yy, 0, atol=atol)

    # JQXJX is mostly real, but only do this check when not all zero
    if not (np.allclose(xx2, 0, atol=atol) and np.allclose(yy2, 0, atol=atol)):
        assert (
            np.linalg.norm(xx2.imag / xx2.real) < atol
        ), f"{np.linalg.norm(xx2.imag/xx2.real)}"
        assert (
            np.linalg.norm(yy2.imag / yy2.real) < atol
        ), f"{np.linalg.norm(yy2.imag/yy2.real)}"
    else:
        assert np.allclose(xx2, 0, atol=atol)
        assert np.allclose(yy2, 0, atol=atol)

    # JXJQY is mostly imag, but only do this check when not all zero
    if not (np.allclose(xy, 0, atol=atol) and np.allclose(yx, 0, atol=atol)):
        assert (
            np.linalg.norm(xy.real / xy.imag) < atol
        ), f"{np.linalg.norm(xy.real/xy.imag)}"
        assert (
            np.linalg.norm(yx.real / yx.imag) < atol
        ), f"{np.linalg.norm(xy.real/xy.imag)}"
    else:
        assert np.allclose(xy, 0, atol=atol)
        assert np.allclose(yx, 0, atol=atol)

    # JQXJY is mostly imag, but only do this check when not all zero
    if not (np.allclose(xy2, 0, atol=atol) and np.allclose(yx2, 0, atol=atol)):
        assert (
            np.linalg.norm(xy2.real / xy2.imag) < atol
        ), f"{np.linalg.norm(xy2.real/xy2.imag)}"
        assert (
            np.linalg.norm(yx2.real / yx2.imag) < atol
        ), f"{np.linalg.norm(yx2.real/yx2.imag)}"
    else:
        assert np.allclose(xy2, 0, atol=atol)
        assert np.allclose(yx2, 0, atol=atol)

    # C4 symmetry within one block
    assert np.allclose(xx, yy, atol=atol)
    assert np.allclose(xy, -yx, atol=atol)

    # C4 symmetry within one block
    assert np.allclose(xx2, yy2, atol=atol)
    assert np.allclose(xy2, -yx2, atol=atol)

    # Onsager symmetry across blocks
    assert np.allclose(xx, xx2, atol=atol)
    assert np.allclose(yy, yy2, atol=atol)

    # Onsager symmetry across blocks
    assert np.allclose(xy, xy2, atol=atol)
    assert np.allclose(yx, yx2, atol=atol)


def my_correlators(path: str) -> dict[str, np.ndarray]:
    """
    Given path with trailing backslash, run get_component() and thermal_sum()
    to get a dictionary with JNJN, JQJN, JNJQ, JQJQ
    Return a dictionary with named elements.
    E.g. result["JQJN"].shape = (4, Nbin_complete, L)
    NOTE: elements divided by sign for prettier plotting
    """

    if da.info(path, uneqlt=True, show=True, imagtol=1e-2) == 1:
        raise ValueError("failed to obtain correlator")

    U, mu, beta, Nx, Ny, bps, b2ps, nflux, tp, N, L, dt = util.load_firstfile(
        path,
        "metadata/U",
        "metadata/mu",
        "metadata/beta",
        "metadata/Nx",
        "metadata/Ny",
        "metadata/bps",
        "metadata/b2ps",
        "metadata/nflux",
        "metadata/t'",
        "params/N",
        "params/L",
        "params/dt",
    )

    try:
        jjn_q0 = get_component(path, "jjn")
        jnj_q0 = get_component(path, "jnj")
    except KeyError as e:
        print(f"caught KeyError: {e}, using legacy names new_jjn and new_jnj")
        jjn_q0 = get_component(path, "new_jjn")
        jnj_q0 = get_component(path, "new_jnj")

    j2j2_q0 = get_component(path, "j2j2")
    j2j_q0 = get_component(path, "j2j")
    jj2_q0 = get_component(path, "jj2")
    j2jn_q0 = get_component(path, "j2jn")
    jnj2_q0 = get_component(path, "jnj2")
    jnjn_q0 = get_component(path, "jnjn")
    jj_q0 = get_component(path, "jj")

    # TODO: generate grid range based on U?
    dm_dict, de_dict = da.eqlt_meas_i(path, ["density"])
    dm = dm_dict["density"].real
    de = de_dict["density"]
    # tt = fr"{Nx}x{Ny} nflux={nflux} n={dm.real:.7g} t'={tp} U={U} $\beta$={beta:.3g}"
    print(f"Achieved n-1={(dm-1):.3g} , SE(n) =  {de:.3g}")

    ns, s = util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/sign")
    mask = ns == ns.max()
    nbin = mask.sum()
    ns, s = ns[mask], s[mask]

    # NOTE: no error analysis, just divided by sign
    j2j2_q0 /= np.mean(s)

    jj2_q0 /= np.mean(s)
    j2j_q0 /= np.mean(s)
    jnj2_q0 /= np.mean(s)
    j2jn_q0 /= np.mean(s)

    jjn_q0 /= np.mean(s)
    jnj_q0 /= np.mean(s)

    jnjn_q0 /= np.mean(s)
    jj_q0 /= np.mean(s)

    q0_corrs = (
        j2j2_q0,
        jj2_q0,
        j2j_q0,
        jnj2_q0,
        j2jn_q0,
        jjn_q0,
        jnj_q0,
        jnjn_q0,
        jj_q0,
    )

    return thermal_sum(path, q0_corrs)


def compare(my_dict: dict[str, np.ndarray], wen_dict: dict[str, np.ndarray]):
    print(np.max(np.abs(my_dict["JQJQ"] - wen_dict["JQJQ"])))
    print(np.max(np.abs(my_dict["JQJN"] - wen_dict["JQJN"])))
    print(np.max(np.abs(my_dict["JNJQ"] - wen_dict["JNJQ"])))
    print(np.max(np.abs(my_dict["JNJN"] - wen_dict["JNJN"])))
