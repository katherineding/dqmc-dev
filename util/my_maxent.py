from typing import Any, Optional, Callable
import numpy.typing as npt

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy import linalg as la
from scipy.interpolate import InterpolatedUnivariateSpline

# from scipy.optimize import root_scalar
from scipy.special import xlogy

import misc

alpha_arr_default = np.logspace(1, 9, 1 + 20 * (9 - 1))

# aliases for type checking
dNum = np.dtype[np.float64]
cNum = np.dtype[np.complex128]


def tau_from_omega(
    op_type: str,
    beta: float,
    tau: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64],
    f_w: npt.NDArray[np.float64],
):
    if op_type == "boson":
        return Kernel_B(beta, tau, omega, sym=False) / omega @ f_w
    elif op_type == "fermion":
        return Kernel_F(beta, tau, omega, sym=False) @ f_w
    else:
        raise NotImplementedError


# ---Checked kernels, same as edwin, except fermion kernel sign----
def Kernel_B(
    beta: float,
    tau: np.ndarray[tuple[int], dNum],
    omega: np.ndarray[tuple[int], dNum],
    sym: bool = False,
) -> np.ndarray[tuple[int, int], dNum]:
    """bosonic kernel: K(tau, w) = w*exp(-tau*w)/(1-exp(-beta*w))

    Returns:
        np.ndarray: (ntau,nomega) float matrix
    """
    assert tau.max() <= beta and tau.min() >= 0
    # avoid pure zero by adding machine eps to zero
    omega[omega == 0] += np.finfo(float).eps
    if sym:
        top = omega * (
            np.exp(-np.outer(tau, omega)) + np.exp(-np.outer(beta - tau, omega))
        )
    else:
        top = omega * np.exp(-np.outer(tau, omega))

    bot = 1 - np.exp(-beta * omega)
    return top / bot


def Kernel_F(
    beta: float,
    tau: np.ndarray[tuple[int], dNum],
    omega: np.ndarray[tuple[int], dNum],
    sym: bool = False,
) -> np.ndarray[tuple[int, int], dNum]:
    """fermionic kernel: K(tau, w) = exp(-tau*w)/(1+exp(-beta*w))

    Returns:
        np.ndarray: (ntau,nomega) float matrix
    """
    assert tau.max() <= beta and tau.min() >= 0
    if sym:
        top = np.exp(-np.outer(tau, omega)) + np.exp(-np.outer(beta - tau, omega))
    else:
        top = np.exp(-np.outer(tau, omega))
    bot = 1 + np.exp(-beta * omega)

    return top / bot


# def Kernel_T(beta, tau, omega):
#     assert tau.max() <= beta and tau.min() == 0
#     omega[omega == 0] += np.finfo(float).eps
#     top = np.exp(np.outer(tau, omega)) - np.exp(np.outer((beta - tau), omega))
#     bot = 1 - np.exp(beta * omega)
#     return top / bot


# ------------------------------------------------------------------
def Entropy(A, m):
    """Nonpositive entropy term, sum(-A*log(A/m))"""
    # USING XLOGY to handle A = 0 case
    # return np.sum((-xlogy(A,A/m)))
    return np.sum((A - m - xlogy(A, A / m)))


def Chi_Sq(A, Kp, Gp, W):
    """Chi^2 error generalized least squares problem"""
    KA_G = Kp @ A - Gp
    return np.vdot(KA_G * W, KA_G)


def Qp(A, *args):
    """Objective function to maximize, alpha*S[A,m] - Chi^2[A]/2"""
    m, alpha, Kp, Gp, W = args
    return Entropy(A, m) * alpha - Chi_Sq(A, Kp, Gp, W) / 2


def Qm(A, *args):
    """Objective function to minimize, -alpha*S[A,m] + Chi^2[A]/2"""
    m, alpha, Kp, Gp, W = args
    return -Entropy(A, m) * alpha + Chi_Sq(A, Kp, Gp, W) / 2


# -----------------------------------------------------------------------


def MaxEnt(
    pre: dict[str, np.ndarray],
    alpha_arr: np.ndarray = alpha_arr_default,
    method: str = "BT",
    printout: bool = False,
    inspect: bool = False,
):
    """Perform MaxEnt, pick best alpha based on method spec.
    Args:
        pre: dictionary with preprocessed data
        alpha_arr : array of alpha values to chose from
        method = 'historic' or "classic" or "bryan" or "BT" (default)
        printout : bool, whether to print norm info
        inspect : bool, whether to plot intermediate checks

    Returns:
        A tuple.
        [0]: (N_omega,) array = best estimate of A(omega_i) * domega_i
        [1]: corresponding best alpha
    """
    eigv_threshold = 1e-8  # clip covar matrix eigenval up if < max eigen_val * this
    svd_threshold = 1e-12  # drop kernel singular vals if < max singular_val * this
    # -------------------------------------------------------------------------------------
    G = pre["lhs"]  # data
    K = pre["K"]  # kernel
    m = pre["m"]  # model
    tau = pre["tau"]  # tau grid

    # calc_A(G, K, m, alpha_arr, plot=True, useBT=False)
    # -------------------------------------------------------------------------------------

    # covariance matrix processing
    Nbin, _ = G.shape
    C = np.cov(G, rowvar=False) / Nbin
    s, Q = la.eigh(C)
    # s = sigma^2 array
    # clip small eigenvalues upwards to threshold
    mask = np.abs(s) <= np.max(s) * eigv_threshold
    # print(C.shape)
    # s[mask] = np.max(s)*eigv_threshold;
    dof = tau.shape[0] - np.count_nonzero(mask)

    if printout:
        print(f"Choose optimal alpha using {method} method")
        print(f"covariance matrix condition number: {np.linalg.cond(C):.3g}")
        print(f"Using {dof}/{tau.shape[0]} eigenvals in C")

    if inspect:
        plt.figure()
        plt.title("Covariance matrix C eigenvals")
        plt.plot(s, ".")
        plt.yscale("log")
        plt.ylabel(r"$\sigma^2_{\ell}$")
        plt.xlabel(r"eigval index $\ell$")
        plt.grid(True)
        plt.show()

    # rotate kernel and data to make covariance matrix diagonal
    Kp = Q.conj().T @ K
    Gp = Q.conj().T @ np.mean(G, axis=0)  # avg data value
    W = 1 / s  # vector of 1/sigma^2

    # precalculate SVD of Kp
    V, sigma, Uh = np.linalg.svd(Kp, full_matrices=False)
    # drop singular values less than threshold
    mask = sigma >= svd_threshold * np.max(sigma)
    if printout:
        print(f"Using {np.count_nonzero(mask)}/{Kp.shape[0]} singular values of Kp")
    if inspect:
        plt.figure()
        plt.title("kernel mat K singular values")
        plt.plot(sigma, ".")
        plt.yscale("log")
        plt.xlabel("singular val index")
        plt.grid(True)
        plt.show()

    # reduce matrix dimensions
    V = V[:, mask]
    sigma = sigma[mask]
    Uh = Uh[mask, :]
    precalc_svd = V, sigma, Uh

    # -------------------------------Tune alpha value----------------------------------
    # if given a scalar alpha_arr, do not tune alpha
    if np.isscalar(alpha_arr):
        print("input alpha array is scalar, no tuning")
        A_out, _, _ = MaxEnt_Fixed_Alpha(Gp, W, Kp, m, alpha_arr, precalc_svd)
        return A_out

    # get optimized A, chi as function of alpha
    N_alpha = alpha_arr.shape[0]
    A_arr = np.full((N_alpha, m.shape[0]), np.nan, dtype=float)
    Q_arr = np.full(N_alpha, np.nan, dtype=float)
    lnP_arr = np.full(N_alpha, np.nan, dtype=float)
    Ngood_arr = np.full(N_alpha, np.nan, dtype=float)
    chi2_arr = np.full(N_alpha, np.nan, dtype=float)
    for i in range(N_alpha):
        A_arr[i, :], lnP_arr[i], Ngood_arr[i] = MaxEnt_Fixed_Alpha(
            Gp, W, Kp, m, alpha_arr[i], precalc_svd
        )
        Q_arr[i] = Qp(A_arr[i], m, alpha_arr[i], Kp, Gp, W)
        chi2_arr[i] = Chi_Sq(A_arr[i], Kp, Gp, W)

    if method == "historic":
        # goal: find alpha that produces A giving Chi^2 = dof
        # root finding does not need to be very precise
        # removed this method since MaxEnt_Fixed_Alpha now returns tuples
        # this method is kinda bad anyways
        """
        obj_f = lambda al : Chi_Sq(MaxEnt_Fixed_Alpha(Gp,W,Kp,m,al,precalc_svd)[0], Kp,   Gp,  W) - dof
        sol = root_scalar(obj_f,x0 = alpha_arr[0],x1 = alpha_arr[-1],rtol=1e-2)
        alpha_out = sol.root
        A_out,_ = MaxEnt_Fixed_Alpha(Gp,W,Kp,m,alpha_out,precalc_svd)"""
        raise NotImplementedError(f"{method} not implemented")
    elif method == "classic":
        pos = np.argmax(lnP_arr)
        alpha_out = alpha_arr[pos]
        A_out = A_arr[pos, :]
        if inspect:
            plt.figure()
            plt.title(f"dof = {dof}, Ngood = {Ngood_arr[pos]}")
            plt.plot(alpha_arr, np.exp(lnP_arr), ".", ms=2, color="g")
            plt.axvline(x=alpha_out, lw=1, color="g")
            plt.xscale("log")
            # plt.yscale("log")
            plt.grid(True)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\log P(\alpha|G)$")
            plt.show()
    elif method == "bryan":
        # TODO: better integration than rectangle rule
        dalpha = np.diff(alpha_arr)
        dalpha = np.insert(dalpha, 0, dalpha[0])
        Z = np.sum(np.exp(lnP_arr) * dalpha)
        print("probability normalization factor = ", Z)
        # spl = InterpolatedUnivariateSpline(alpha_arr,np.exp(lnP_arr))
        # print(spl.integral(alpha_arr[0],alpha_arr[-1]))
        # assert dalpha.shape == alpha_arr.shape
        A_out = np.sum((np.exp(lnP_arr) * dalpha)[:, None] * A_arr, axis=0) / Z
        # not sure this alpha_out in the average sense is accurate at all
        alpha_out = np.sum(np.exp(lnP_arr) * dalpha * alpha_arr) / Z
        if inspect:
            plt.figure()
            plt.title(f"dof = {dof}")
            plt.plot(alpha_arr, np.exp(lnP_arr), ".", ms=2, color="m")
            plt.xscale("log")
            # plt.yscale("log")
            plt.grid(True)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$P(\alpha|G)$")
            plt.show()
    elif method == "BT":
        # from Bergeron Tremblay 2016 paper
        log_a = np.log(alpha_arr)
        log_c = np.log(chi2_arr)
        # cubic spline fit, get curvature
        spl = InterpolatedUnivariateSpline(log_a, log_c, ext=2, check_finite=True)
        top = spl(log_a, nu=2)
        bot = np.power((1 + np.power(spl(log_a, nu=1), 2)), 1.5)
        # BT soln
        pos = np.argmax(top / bot)  # find max of signed curvature
        alpha_out = alpha_arr[pos]
        A_out = A_arr[pos, :]
        if inspect:
            plt.figure()
            plt.title(f"curvature, dof={dof}, Ngood={Ngood_arr[pos]}")
            plt.plot(alpha_arr, top / bot, ".", ms=2, color="b")
            plt.axvline(x=alpha_out, lw=1, color="b")
            plt.axhline(y=0, lw=1, color="k")
            plt.grid(True)
            plt.xscale("log")
            plt.show()
    else:
        raise ValueError("invalid method spec")

    # ----------------------------End tune alpha value----------------------------------------
    c2lo, c2hi = scipy.stats.chi2.interval(0.95, dof)
    if inspect:
        plt.figure()
        plt.plot(alpha_arr, chi2_arr, lw=1, label=r"likelihood $\chi^2$")
        # plt.plot(alpha_arr,lnP_arr,lw=1)
        plt.title(f"dof = {dof}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$\alpha$")
        plt.axvline(x=alpha_out, color="b", lw=1, label=rf"{method} optimal $\alpha$")
        plt.axhline(
            y=c2lo, color="k", ls="-.", lw=1, label=r"$\chi^2$ 0.95 lower bound"
        )
        plt.axhline(y=dof, color="k", ls="-", lw=1)
        plt.axhline(
            y=c2hi, color="k", ls="--", lw=1, label=r"$\chi^2$ 0.95 upper bound"
        )
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

    cf = Chi_Sq(A_out, Kp, Gp, W)
    if printout:
        print(
            f"alpha = {alpha_out:.3f}, chi^2/dof = {cf:.3f}/{dof} = {cf/dof:.3f}",
            f" sum(A*domega) = {A_out.sum():.6f}",
            f" entropy = {np.sum((A_out-m)):.3g} + {np.sum(-xlogy(A_out,A_out/m)):.3g},",
            f" -2*alpha*S = {-2*alpha_out*Entropy(A_out,m):.3g}\n",
        )

    if cf > c2hi:
        print(f"\033[93mChi-squared error = {cf:.3f} > {c2hi:.3f}\033[0m")
        # return np.full(A_out.shape,np.nan)
    if np.abs(A_out.sum() - 1) > 5e-2:
        print(f"\033[91msum(A*domega) = {A_out.sum():.6f}, return A nan\033[0m")
        return np.full(A_out.shape, np.nan)

    return A_out


# beta, tau, omega, L, Nbin, dt?
def MaxEnt_Fixed_Alpha(Gp, W, Kp, m, alpha, precalc_svd=None):
    """Perform Bryans Optiization Algorithm for fixed alpha value
    Args:
        Gp: symmetry appropriate (N_tau,) data, rotated and normalized, divided by sign
        W: (N_tau,) data errors
        Kp: (N_tau, N_omega) kernel
        m: m(omega_i) * domega_i, shape (N_omega,)
        alpha: user specified alpha value
        precalc_svd:
    Returns:
        best estimate of A(omega_i) * domega_i, shape (N_omega,), given fixed alpha
    """
    svd_threshold = 1e-12  # drop kernel singular vals if < max singular_val * this
    maxiter = 1234  # max number of root finding iterations in Bryans algorithm
    max_step_size = m.sum()
    small_step_threshold = 0.125
    mu_multiplier, mu_min, mu_max = 2.0, alpha / 4, alpha * 1e100
    dQ_threshold = 1e-10
    conseq_threshold = 7

    # m[np.abs(m) < 1e-16] = np.nan

    # -----------------------Bryan's optimization algorithm--------------------------------------
    # svd of kernel: K = V Sigma U.H
    if precalc_svd is None:
        print("Getting svd of Kp")
        V, sigma, Uh = np.linalg.svd(Kp, full_matrices=False)
        # drop singular values less than threshold
        mask = sigma >= svd_threshold * np.max(sigma)
        print(f"Using {np.count_nonzero(mask)}/{Kp.shape[0]} singular values of Kp")

        # reduce matrix dimensions
        V = V[:, mask]
        sigma = sigma[mask]
        Uh = Uh[mask, :]
    else:
        V, sigma, Uh = precalc_svd

    # precalculated stuff, doesnt depend on state u
    U = Uh.conj().T
    sigVT = sigma.conj()[:, None] * V.conj().T
    M = np.dot(sigVT * W, V * sigma)
    I = np.identity(sigma.shape[0])

    # initialize u state
    u = np.zeros(sigma.shape)  # variable we optimize over
    # quantities that depend on u state
    A = m * np.exp(np.dot(U, u))
    T = (Uh * A) @ U
    f = alpha * u + (sigVT * W) @ (Kp @ A - Gp)
    q_old = Qm(A, m, alpha, Kp, Gp, W)
    # assert np.all(np.isfinite(A)) and np.all(np.isfinite(T)) and \
    #    np.all(np.isfinite(f)) and np.all(np.isfinite(q_old))

    nit = 0
    n_conseq = 0
    mu = alpha  # LM param
    # Newton root finding in u space, with extra LM param
    while nit < maxiter:
        # assert np.all(np.isfinite(T))
        # propose a du step
        # jac = (alpha + mu)*I + M @ T
        Xi, P = np.linalg.eigh(T)
        assert np.all(np.abs(Xi[Xi < 0]) < 2e-14), f"error {np.abs(Xi[Xi<0])}"
        Xi[Xi < 0] = 0
        # print("P is U?", np.allclose(P,U.T))
        AA = np.sqrt(Xi)[:, None] * P.T @ M @ P * np.sqrt(Xi)
        # print("A matrix", AA)
        Lam, R = np.linalg.eigh(AA)
        # print(Lam)
        Yinv = (R.T * np.sqrt(Xi)) @ P.T
        Yinvu = -R.T @ (np.sqrt(Xi) * (P.T @ f)) / (alpha + mu + Lam)
        du = -(f + M @ (P @ (np.sqrt(Xi) * (R @ Yinvu)))) / (alpha + mu)
        step_size = np.dot(Yinvu, Yinvu)
        A = m * np.exp(np.dot(U, u + du))
        A[A > m.max() * 1e3] = m.max() * 1e3
        # if m too small, then exp term can overflow
        # if np.max(np.dot(U,u+du)) > 100:
        # print(f"alpha = {alpha}, iter = {nit}, max U@u",np.max(np.dot(U,u+du)), \
        #     f"U norm {np.linalg.norm(U):.3g} u norm: {np.linalg.norm(u+du)} qnew = { Qm(A,m,alpha,Kp,Gp,W)}\
        #     qold = {q_old}")
        # plt.figure()
        # plt.plot(m,label="m")
        # plt.plot(A,label="A")
        # plt.ylim(0,0.1)
        # plt.legend(loc="best")
        # plt.show()
        q_new = Qm(A, m, alpha, Kp, Gp, W)
        if (
            step_size > max_step_size
            or np.any(np.logical_not(np.isfinite(A)))
            or q_new / q_old > 1e3
        ):
            # print(f"reject_step {nit}, increase mu")
            A = m * np.exp(np.dot(U, u))
            mu = np.clip(mu * mu_multiplier, mu_min, mu_max)
            nit += 1
            continue

        # turns out, Q ratio is too big, have to reject this step
        # if q_new/q_old > 1e3 :
        #    u -= du
        #    A = m * np.exp(np.dot(U,u))
        #    T = (Uh * A) @ U
        #   f = alpha * u + (sigVT * W) @ (Kp @ A -Gp)
        #    print(f"WARNING: Q ratio = {q_new/q_old}, step_size = {step_size}")

        # print(f"accept step {nit},update u,T,f,Q")
        u += du
        T = (Uh * A) @ U
        f = alpha * u + (sigVT * W) @ (Kp @ A - Gp)
        # this step size small, decrease mu for next iter
        if step_size < small_step_threshold:
            mu = mu / mu_multiplier if mu > mu_min else 0

        # count consequtive dQ/Q < dQ_threshold
        if np.abs(q_old / q_new - 1) < dQ_threshold:
            n_conseq += 1
        else:
            n_conseq = 0
        # if converged, then break
        if n_conseq >= conseq_threshold:
            break
        # refresh q value
        q_old = q_new
        nit += 1

    # if nit == maxiter:
    # print(f"\033[4miIter {nit} reached, alpha={alpha:.3g}, probably no converge\033[0m")

    lam = np.linalg.eigvalsh(np.sqrt(A)[:, None] * U @ M @ Uh * np.sqrt(A))
    Ngood = np.sum(lam / (lam + alpha))
    # print("logP(alpha)",-np.log(alpha))
    # print("Q",Qp(A,m,alpha,Kp,Gp,W))
    # print("extra",0.5*np.log(alpha/(lam+alpha)).sum())
    # print(f"alpha = {alpha}, iter = {nit}, max U@u",np.max(np.dot(U,u+du)), f"U norm {np.linalg.norm(U):.3g} u norm: {np.linalg.norm(u+du)} m min {np.min(m)} qnew = { Qm(A,m,alpha,Kp,Gp,W)}")
    # plt.figure()
    # plt.plot(m)
    # plt.plot(A)
    # plt.ylim(0,0.1)
    # plt.show()
    # not using jeffrys prior
    lnP = +Qp(A, m, alpha, Kp, Gp, W) + 0.5 * np.sum(np.log(alpha / (lam + alpha)))

    return A, lnP, Ngood


# -----------------------------------------------------------------
def Preprocess(
    G: npt.NDArray[np.float64],
    dt: float,
    beta: float,
    grid_info: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    op_type: str,
    sym: bool,
    append: Optional[npt.NDArray[np.float64]] = None,
    model_arr: Optional[npt.NDArray[np.float64]] = None,
) -> dict[str, np.ndarray]:
    """
    Args:
        G: (Nbin, L) float array, divided by sign, no end, no rotate, no norm
        dt: float
        beta: float = dt*L
        grid_info: (w,dw) typle
        op_type: "boson" or "fermion"
        sym: True or False
        append: (Nbin,1) array, only used when op_type = 'boson' and sym=False
        model_arr: None or model array

    Returns:
        a dictionary with fields
            "tau": tau grid, (Ntau,) float array
            "m": model, (Nw,) float array
            "lhs" : G, LHS of AC, div by sign, sym shape ok, normalized, before rotation
            "norm" : normalization factor so that A integrates to 1, float
            "K": K, (Ntau,Nomega) kernel
    """
    print(f"Preprocessing...{op_type} kernel, sym = {sym}")
    # grid info is user specified, check consistent with symmetry
    omega, domega = grid_info
    assert np.all(np.diff(omega) > 0), "w not strictly increasing"
    assert np.all(domega > 0), "dw has nonpositive values"
    if sym:
        assert omega.min() > 0, f"sym = {sym}, only positive w needed"
    else:
        assert omega.min() < 0, f"sym = {sym}, need negative w"

    # Process based on if particle-hole symmetry
    Nbin, L = G.shape

    # Create tau grid, kernel, Normalize G
    if op_type == "boson":
        if sym:
            assert append is None
            G0 = np.reshape(G[:, 0], (Nbin, 1))
            G = np.concatenate((G, G0), axis=1)
            # process based on symmetry, keep only fist half of interval
            Grev = np.fliplr(G)
            G = 0.5 * (G + Grev)[:, : (L // 2 + 1)]
            # normalization
            spl = InterpolatedUnivariateSpline(
                np.arange(L // 2 + 1) * dt, G.mean(0), ext=2, check_finite=True
            )
            norm_factor = spl.integral(0, beta / 2)
            tau = np.arange(L // 2 + 1) * dt
        else:
            # append extra data at tau = beta if provided
            if append is not None:
                assert append.shape == (G.shape[0], 1)
                G = np.concatenate((G, append), axis=1)
                # normalization
                spl = InterpolatedUnivariateSpline(
                    np.arange(L + 1) * dt, G.mean(0), ext=2, check_finite=True
                )
                norm_factor = spl.integral(0, beta)
                # keep extra bin
                tau = np.arange(L + 1) * dt
            else:
                # otherwise, only use G(tau) data at tau = 0...L-1 for MaxEnt fit,
                # but spline extrapolate to tau=beta and use extrapolated function
                # to estimate norm
                spl = InterpolatedUnivariateSpline(
                    np.arange(L) * dt, G.mean(0), ext=0, check_finite=True
                )
                # Don't use spl.integral(), it doesn't properly extrapolate
                tfine = np.linspace(0, beta, 201)
                norm_factor = np.trapz(spl(tfine), tfine)
                # only use first L bins to fit
                tau = np.arange(L) * dt
        K = Kernel_B(beta, tau, omega, sym=sym)
    # WARNING =========== This is totally bodged  ==================
    # Placeholder for easy plotting of diff output, Don't ever feed this into maxent
    elif op_type == "xy":
        # assert append is None
        G0 = np.reshape(G[:, 0], (Nbin, 1))
        G = np.concatenate((G, -G0), axis=1)
        tau = np.arange(L + 1) * dt
        norm_factor = append
        K = Kernel_B(beta, tau, omega)
    # =========================================================================
    elif op_type == "fermion":
        assert append is None
        if sym:
            G0 = np.reshape(G[:, 0], (Nbin, 1))
            G = np.concatenate((G, G0), axis=1)
            # process based on symmetry, keep only fist half of interval
            Grev = np.fliplr(G)
            G = 0.5 * (G + Grev)[:, : (L // 2 + 1)]
            tau = np.arange(0, L // 2 + 1) * dt
            # Norm factor should be exactly 0.5 , but use mean of G(tau=0) to allow for
            # 1) numerical error and 2) scaling by constant factor
            norm_factor = np.nanmean(G[:, 0])
        else:
            # TODO: relax norm factor to allow for
            # 1) numerical error and 2) scaling by constant factor??
            norm_factor = 1.0
            G0 = np.reshape(G[:, 0], (Nbin, 1))
            res = np.random.randint(Nbin, size=Nbin)
            # TODO: is this way OK? Maybe I should just remove the first element
            G = np.concatenate((G, norm_factor - G0[res]), axis=1)
            tau = np.arange(0, L + 1) * dt
        K = Kernel_F(beta, tau, omega, sym=sym)
    else:
        raise ValueError(f"{op_type} operator, symmetry = {sym} invalid")

    assert np.abs(norm_factor) > 1e-10, "zero norm will result in NaN lhs"

    # anneal model
    if model_arr is not None:
        assert (
            model_arr.shape[0] == omega.shape[0]
        ), f"{model_arr.shape[0]} != {omega.shape[0]}"
        # clip small values to avoid exp(U@u) overflow in bryan's algorithm
        model_arr[model_arr < model_arr.max() * 1e-4] = model_arr.max() * 1e-4
        print("model norm before normalization:", model_arr.sum())
        m = model_arr / model_arr.sum()
        print("processed model min", m.min())
    else:
        # default to flat model
        print("using default model function: flat")
        m = domega / np.sum(domega)

    # return structure
    d = {
        "tau": tau,
        "m": m,
        "lhs": G / norm_factor,  # div by sign, sym shape ok, normalized
        "norm": norm_factor,  # normalization factor
        "K": K,
    }

    return d


def report_norms(JJ):
    realnorm = np.linalg.norm(JJ.real, ord=np.inf)
    imagnorm = np.linalg.norm(JJ.imag, ord=np.inf)
    errnorm = np.linalg.norm(
        np.std(JJ, axis=0, ddof=1) / np.sqrt(JJ.shape[0]), ord=np.inf
    )
    print(
        f"shape: {JJ.shape}, dtype: {JJ.dtype}, "
        + f" real: {realnorm:.3g}"
        + f" imag: {imagnorm:.3g}"
        + f" error: {errnorm:3g}"
        + misc.bcolors.BOLD
        + f" imag/real ratio: {imagnorm/realnorm:.3g}"
        + f" err/real ratio: {errnorm/realnorm:.3g}"
        + misc.bcolors.ENDC
    )


def plot_bs_results(
    omega: np.ndarray[tuple[int], dNum],
    data: np.ndarray[tuple[int, int], dNum],
    L: int,
    dt: float,
    tt: str,
    pre: dict[str, np.ndarray],
    A_bs: np.ndarray,
    spectra_bs: np.ndarray,
):
    assert A_bs.shape == spectra_bs.shape

    nbin = data.shape[0]
    bs = A_bs.shape[0]

    # Frequency domain output bootstraps
    plt.figure()
    plt.title(tt)
    plt.ylabel(f"spectra bs # = {bs}")
    plt.xlabel(r"$\omega/t$")
    plt.plot(omega, spectra_bs.T, lw=1, color="k")
    plt.grid(True)

    plt.figure()
    plt.title(tt)
    plt.ylabel(f"raw maxent bs # = {bs}")
    plt.xlabel(r"$\omega/t$")
    plt.plot(omega, A_bs.T, lw=1, color="k")
    plt.grid(True)

    # ========= reproducing G(tau) ============
    plt.figure()
    plt.title(tt)
    plt.ylabel(rf"$G(\tau)$ bs # = {bs}")
    plt.xlabel(r"$\tau$")
    # note: errorbar is += 1 std error of mean of DQMC data
    print(pre["tau"].shape)
    plt.errorbar(
        np.arange(L) * dt,
        np.mean(data.real, axis=0),
        yerr=np.std(data, axis=0, ddof=1) / np.sqrt(nbin),
        fmt=".",
        label="original data",
    )
    for i in range(bs):
        plt.plot(
            pre["tau"],
            pre["K"] @ A_bs[i, :] * pre["norm"],
            lw=1,
            color="k",
            label="reproduction" if i == 0 else "_",
        )
    plt.legend(loc="best")
    plt.grid(True)

    # ========== Residues G(tau) - K @ A ===============
    plt.figure()
    plt.title(tt)
    plt.ylabel(rf"$G(\tau) - K(\tau,\omega) \cdot A(\omega)$ bs # = {bs}")
    plt.xlabel(r"$\tau$")
    for i in range(bs):
        plt.plot(
            pre["tau"],
            np.mean(pre["lhs"], axis=0) * pre["norm"]
            - pre["K"] @ A_bs[i, :] * pre["norm"],
            "s",
            label="residues" if i == 0 else "_",
        )
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()


# == copied from Edwin's maxent.py ==
def gen_grid(
    nw: int, x_min: float, x_max: float, w_x: Callable
) -> tuple[np.ndarray, np.ndarray]:
    """
    generate grid with nw points scaled by the function w_x.

    w[i] = w_x((i+0.5)/nw * (x_max-x_min) + x_min)
    dw[i] = w_x((i+1)/nw * (x_max-x_min) + x_min) -
            w_x(i/nw * (x_max-x_min) + x_min)

    returns w, dw
    """
    x_all = np.linspace(x_min, x_max, 2 * nw + 1)
    w_all = np.apply_along_axis(w_x, 0, x_all)
    return w_all[1::2], np.abs(np.diff(w_all[::2]))


# ===================================


def uniform_grid(nw: int, w_min: float, w_max: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.linspace(w_min, w_max, nw)
    dw = np.ones(nw) * (w[1] - w[0])
    return w, dw


def semicircle(w: np.ndarray, r: float = 1, a: float = 0) -> np.ndarray:
    """Make a semicircle centered at a with radius r,
    then normalize it downwards so that \int A(w) dw = 1

    Args:
        w (np.ndarray): [description]
        r (float): [description] (default: `1`)
        a (float): [description] (default: `0`)

    Returns:
        np.ndarray: [description]
    """
    inrange = np.logical_and(w > a - r, w < a + r)
    out = np.full(w.shape, np.nan, dtype=np.float64)
    out[inrange] = np.sqrt(r**2 - (w[inrange] - a) ** 2)
    out[~inrange] = 0
    out = out / (np.pi * r**2 / 2)
    return out


def drude_xx(w: np.ndarray, wc: float) -> np.ndarray:
    """Drude xx conductivity with cyclotron frequency wc

    Args:
        w (np.ndarray): [description]
        wc (float): [description]

    Returns:
        np.ndarray: [description]
    """
    out = (1 - 1j * w) / ((1 - 1j * w) ** 2 + wc**2)
    return out


def drude_xy(w: np.ndarray, wc: float) -> np.ndarray:
    """Drude xy conductivity with cyclotron frequency wc

    Args:
        w (np.ndarray): [description]
        wc (float): [description]

    Returns:
        np.ndarray: [description]
    """
    out = -wc / ((1 - 1j * w) ** 2 + wc**2)
    return out


def model_flat(dw: np.ndarray) -> np.ndarray:
    return dw / dw.sum()


def lorentzian(x, x0=0, gamma=1):
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma**2)
