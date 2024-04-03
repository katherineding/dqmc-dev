import numpy as np
from numpy import array as npa 
import matplotlib.pyplot as plt
import data_analysis as da
import my_maxent
from cached_property import cached_property

import util #Edwin's util file
import maxent #Edwin's maxent file
import jqjq

# #one-hop bond dx, dy
dx_arr = [1,0,1,-1]
dy_arr = [0,1,1, 1]

#katherine hopping arrays
dx2_arr = [2,0,1,-1,2,1,-2,-1,0,1,2,-2]
dy2_arr = [0,2,1, 1,1,2, 1, 2,1,0,2, 2]

class Transport:
    def __init__(self, path):
        self.path = path
        
        # load metadata 
        meta_keys = ["U", "mu", "beta", "Nx", "Ny", "bps", "b2ps", "nflux",
                     "tp", "N", "L", "dt", "bonds", "bond2s"]
        meta_args = util.load_firstfile(path,"metadata/U","metadata/mu","metadata/beta",
                                            "metadata/Nx","metadata/Ny","metadata/bps",
                                            "metadata/b2ps", "metadata/nflux",
                                            "metadata/t'","params/N","params/L","params/dt",
                                            "params/bonds", "params/bond2s")    
        self.metadata = dict(zip(meta_keys, meta_args))
        
        # get number of samples and sign 
        ns, s = util.load(path,"meas_uneqlt/n_sample", "meas_uneqlt/sign")
        mask = ns == ns.max(); nbin = mask.sum()
        ns, s = ns[mask], s[mask]
        self.mask = mask
        self.nbin = nbin
        self.sign = np.mean(s).real
        self.n_sample = ns.max()

        # get density info 
        dm_dict, de_dict = da.eqlt_meas_1(path,['density'])
        self.dm = dm_dict["density"].real
        self.de = de_dict["density"]
        
        self.omega_grid = None 
        

    @cached_property
    def corr(self):
        return get_corr(self.path)

    @cached_property
    def plot_title_str(self):
        Nx = self.metadata["Nx"]
        Ny = self.metadata["Ny"]
        nflux = self.metadata["nflux"]
        beta = self.metadata["beta"]
        tp = self.metadata["tp"]
        U = self.metadata["U"]
        tt = fr"{Nx}x{Ny} nflux={nflux} n={self.dm:.3g} t'={tp} U={U} $\beta$={beta:.3g}"
        return tt

    def lhs(self, corr, check=False):
        """Construct lhs based on components of the correlator O(tau)O^{dagger}
        Args: 
            corr: array containing components of the correlator xx, yy, xy, yx 
            (i.e. output defined in Katherine's thermal sum output)
        Returns: 
            A tuple of tuples 
                [0]: (longitudinal correlator, tau=0 element to be appended for MaxEnt), i.e. O=Jx
                [1]: (hermitian correlelator, composite tau=0 object for appending), i.e. O=Jx+iJy
        """
        # components of correlators divided by sign 
        # (Nbin, L) matrices 
        xx = corr[0].real / self.sign
        yy = corr[1].real / self.sign 
        xy = corr[2].imag / self.sign 
        yx = corr[3].imag / self.sign 

        # constructing operators (lhs) to invert with maxent 
        chi_xx =  0.5 * (xx + yy) # do an average to reduce noise 
        chi_hermit = (xx + yy + xy - yx) 

        # for appending tau=L component for maxEnt. needs to have the correct symmetry 
        append = xx[:,0] + yy[:,0] - xy[:,0] + yx[:,0]   # (Nbin, 1) array 
        append.shape = (self.nbin, 1)
        append_xx = chi_xx[:,0]
        append_xx.shape = (self.nbin, 1)

        if check:
            tt = self.plot_title_str
            dt = self.metadata["dt"]
            L = self.metadata["L"]
            tau = np.arange(L) * dt 
            plt.figure()
            plt.title(tt)
            plt.ylabel(r"chi_xx")
            plt.plot(tau,chi_xx.T,lw=1,color='k')

            plt.figure()
            plt.title(tt)
            plt.ylabel(r"chi_xy")
            plt.plot(tau,xy.T,lw=1,color='k')

            plt.figure()
            plt.title(tt)
            plt.ylabel(r"chi_hermitian")
            plt.plot(tau, chi_hermit.T,lw=1,color='k')
            plt.show()
        return ( (chi_xx, append_xx), (chi_hermit, append) )

    def perform_maxent(self, chi_, bs=1, method = "BT", anneal_arr = None, checks=False,
                       alpha_arr = np.logspace(1, 9, 1+20*(9-1)), inspect=False, sym=True, drop=False):
        """Performs MaxEnt on correlations of the form O(tau)O^{dagger}
        Args: 
            chi: a tuple containing lhs of the integral to invert (real) 
            and an (Nbin,1) array to append as the tau=beta component of G
        Returns: 
            A dictionary with components  
                "A": numpy array with raw MaxEnt data 
                "s": numpy array with Re[L_{OO^{\dagger}}](\omega) with all bootstrap elements 
        """
        
        dt = self.metadata["dt"]
        beta = self.metadata["beta"]
        chi, append = chi_ # if sym=True, append doesn't get used
        
        # omega_grid is by default set to None; if not user-specified, print a warning 
        # and set default values based on symemtric or non-symmetric 
        try: 
            omega, domega = self.omega_grid
        except TypeError as e:
            print("Omega grid not previously specified, generating grid for sym={}".format(sym))
            if sym:
                omega, domega = maxent.gen_grid(200//2, 0, 2.1, lambda x: 0.4*np.sinh(2.5*x))
            else:
                omega, domega = maxent.gen_grid(200, -2.1, 2.1, lambda x: 0.4*np.sinh(2.5*x))
        nw = omega.shape[0] 

        s_bs = np.full((bs,nw),np.nan,dtype=float)
        A_bs = np.full((bs,nw),np.nan,dtype=float)
        for i in range(bs):
            resample = np.random.randint(self.nbin,size=self.nbin) #sample with replacement
            pre = my_maxent.Preprocess(chi[resample], dt, beta, grid_info = (omega,domega),
                                    op_type = 'boson', sym=sym, model_arr = anneal_arr, append=append[resample])
            if drop:
                pre["tau"] = pre["tau"][:-1]
                pre["lhs"] = pre["lhs"][:,:-1]
                pre["K"] = pre["K"][:-1,:]
            A = my_maxent.MaxEnt(pre, alpha_arr = alpha_arr, method=method,
                                printout=checks,inspect=inspect)
            s = (A/domega)*pre["norm"]*np.pi 
            A_bs[i,:] = A
            s_bs[i,:] = s

        if checks: #and relerr <= 1:
            L = self.metadata["L"]
            tt = self.plot_title_str

            plt.figure()
            plt.title(tt)
            plt.ylabel(r"$L(\omega)$ bootstrap")
            plt.plot(omega,s_bs.T,lw=1,color='k')

            plt.figure()
            plt.title(tt)
            plt.ylabel("raw maxent output bootstrap")
            plt.plot(omega, A_bs.T,lw=1,color='k')

            plt.figure()
            plt.title(tt)
            plt.ylabel("imaginary time data reproduction bootstrap")
   
            #note: errorbar is += 1 std error of mean
            plt.errorbar(np.arange(L)*dt,chi.mean(0),\
                yerr = np.std(chi, axis=0,ddof=1)/np.sqrt(self.nbin),fmt='s',label="data")
            for i in range(bs):
                plt.plot(pre["tau"], pre["K"] @ A_bs[i,:] * pre["norm"],lw=1,color='k')
            plt.legend(loc='best')
            plt.show()

        return { "A": np.nanmean(A_bs,axis=0), "s": s_bs}
    
    def kappa_0(self, **kwargs):
        """Computes Kubo contribution to thermal conductivity coefficient.
        Returns:
            Tuples of size (bootstraps, nomega)
            [0]: kappa_xx(omega)
            [1]: kappa_xy(omega)
        """
        lhs = self.lhs(self.corr["JQJQ"]) # construct LHS 
        maxent_results = [self.perform_maxent(chi, **kwargs) for chi in lhs] # perform MaxEnt on all lhs 

        # return kappa components 
        kappa_xx = (maxent_results[0]["s"] * self.metadata["beta"])
        kappa_xy = (maxent_results[0]["s"] * self.metadata["beta"] - 
                    maxent_results[1]["s"]/ 2 * self.metadata["beta"])  # (2*Lxx-L_hermitian)/2
        return kappa_xx, kappa_xy
    
    def sigma(self, **kwargs):
        """Computes optical conductivity.
        Returns:
            Tuples of size (bootstraps, nomega)
            [0]: sigma_xx(omega)
            [1]: sigma_xy(omega)
        """
        lhs = self.lhs(self.corr["JNJN"]) # construct LHS 
        maxent_results = [self.perform_maxent(chi, **kwargs) for chi in lhs] # perform MaxEnt on all lhs 

        # return kappa components 
        sigma_xx = maxent_results[0]["s"] 
        sigma_xy = (maxent_results[0]["s"] - maxent_results[1]["s"]/2)  # (2*Lxx-L_hermitian)/2

        return sigma_xx, sigma_xy

    @cached_property
    def site_currents(self):
        """Computes site currents. 
        Note: 
            DQMC code outputs currents with shifted indices, thus certain components are rolled 
        Returns:
            Dictionary
            "JQ": Heat current 
            "JN": Particle current
        """
        # load site current components 
        j, jn, j2 = util.load(self.path, "meas_eqlt/j", "meas_eqlt/jn", "meas_eqlt/j2"
                                            )
        j.shape = -1, self.metadata["bps"], self.metadata["Nx"], self.metadata["Ny"]
        j2.shape = -1, self.metadata["b2ps"], self.metadata["Nx"], self.metadata["Ny"]
        jn.shape = -1, self.metadata["bps"], self.metadata["Nx"], self.metadata["Ny"]
        j = j[self.mask,:,:,:] 
        jn = jn[self.mask,:,:,:] 
        j2 = j2[self.mask,:,:,:] 
        nbin = self.nbin
        tp, N, bps, b2ps = self.metadata["tp"], self.metadata["N"], self.metadata["bps"], self.metadata["b2ps"]

        # store a copy so the original arrays don't get rolled in-place
        j_ = j.copy()
        jn_ = jn.copy()
        j2_ = j2.copy()

        # roll the x axis 
        j_[:,3,:,:] = np.roll(j_.copy()[:,3,:,:], 1, axis=-1)
        jn_[:,3,:,:] = np.roll(jn_.copy()[:,3,:,:], 1, axis=-1)
        j2_[:,3,:,:] = np.roll(j2_.copy()[:,3,:,:], 1, axis=-1)
        if tp != 0:
            j2_[:,6,:,:] = np.roll(j2_.copy()[:,6,:,:], 2, axis=-1)
            j2_[:,7,:,:] = np.roll(j2_.copy()[:,7,:,:], 1, axis=-1)
            j2_[:,11,:,:] = np.roll(j2_.copy()[:,11,:,:], 2, axis=-1)

        j2 = j2_.reshape( (nbin,b2ps,N))
        j = j_.reshape( (nbin,bps,N))
        jn = jn_.reshape( (nbin,bps,N))

        # NOT DIVIDED BY SIGN 
        JQ = npa([compute_heat_site_current(j[bin], jn[bin], j2[bin], **self.metadata) 
                  for bin in range(nbin)])
        JN = npa([compute_particle_site_current(jn[bin], **self.metadata)
                  for bin in range(nbin)])
        return {"JQ": JQ, "JN": JN}


    def J_bootstrapped(self, bs=1):
        """Bootstraps heat current measurements.
        Kwargs:
            bs: number of bootstraps
        Returns:
            Dictionary
            "JQ": Heat current (bs, N)
            "JN": Particle current (bs, N)
        """
        Nx, Ny = self.metadata["Nx"], self.metadata["Ny"]
        JQ = self.site_currents["JQ"] / self.sign 
        JN = self.site_currents["JN"] / self.sign 
        if bs > 0:
            JQ_bs = np.full((bs, Nx*Ny, 2),np.nan,dtype=complex)
            JN_bs = np.full((bs, Nx*Ny, 2),np.nan,dtype=complex)
            for i in range(bs):
                resample = np.random.randint(self.nbin,size=self.nbin) #sample with replacement
                JQ_bs[i,:,:] = JQ[resample].mean(0)
                JN_bs[i,:,:] = JN[resample].mean(0)

            return {"JQ": JQ_bs, "JN": JN_bs}
        else:
            return {"JQ": JQ, "JN": JN}

    def kappa_1(self, JQ):
        """Computes energy magnetization contribution to thermal conductivity

        Args:
            JQ: heat site current either size (nbin, 2) including JQx and JQy
        Returns:
            Tuple
            [0]: kappa_xy_1 (nbin, 2) 
            [1]: kappa_yx_1 (nbin, 2) 
        """
        Nx, Ny = self.metadata["Nx"], self.metadata["Ny"]
        beta = self.metadata["beta"]
        Rx, Ry = np.meshgrid(np.arange(Nx), np.arange(Ny))
        Rx = Rx.flatten()
        Ry = Ry.flatten()
        return - 2*(Rx * JQ[:,:,1]).sum(axis=1)*beta, - 2*(Ry * JQ[:,:,0]).sum(axis=1)*beta


def get_corr(path):
    '''Get JQ(tau)JQ q=0 correlators from path.'''
    try:
        jjn_q0  = jqjq.get_component(path,'jjn')
        jnj_q0  = jqjq.get_component(path,'jnj')
    except KeyError as e:
        print(f"KeyError: {e}, using legacy names")
        jjn_q0  = jqjq.get_component(path,'new_jjn')
        jnj_q0  = jqjq.get_component(path,'new_jnj')

    j2j2_q0 = jqjq.get_component(path,'j2j2')
    j2j_q0  = jqjq.get_component(path,'j2j')
    jj2_q0  = jqjq.get_component(path,'jj2')
    j2jn_q0 = jqjq.get_component(path,'j2jn')
    jnj2_q0 = jqjq.get_component(path,'jnj2')
    jnjn_q0 = jqjq.get_component(path,'jnjn')
    jj_q0   = jqjq.get_component(path,'jj')

    q0_corrs = (j2j2_q0,\
        jj2_q0,j2j_q0,\
        jnj2_q0,j2jn_q0,\
        jjn_q0, jnj_q0,\
        jnjn_q0,jj_q0)
    return jqjq.thermal_sum(path,q0_corrs)


def compute_heat_site_current(j, jn, j2, **kwargs):
    """
    Compute heat site currents. j, jn are shapes (bps, N) and j2 is of shape (b2ps, N)
    Returns:
        Array (N, 2)
    """
    N = kwargs["N"]
    bps = kwargs["bps"]
    b2ps = kwargs["b2ps"]
    tp = kwargs["tp"]
    U = kwargs["U"]
    mu = kwargs["mu"]
    bonds = kwargs["bonds"]
    bonds2 = kwargs["bond2s"]

    t_arr = np.ones((N, bps)) * npa([1,1,tp,tp]) if tp != 0 else np.ones((N, bps)) 
    t2_arr = np.ones((N, b2ps)) * npa([1,1,1,1,tp,tp,tp,tp,tp,tp,tp**2,tp**2]) if tp != 0 else np.ones((N, b2ps)) 

    jqx = 0.0 + 0.0j
    jqy = 0.0 + 0.0j

    pairs = npa([bonds[0,np.where(bonds == i)[1]][bps:] for i in range(N)])
    pairsb2 = npa([bonds2[0,np.where(bonds2 == i)[1]][b2ps:] for i in range(N)])

    for itype in range(bps):
        # j contribution
        jqx += 1j/4 * U * t_arr[:,itype]*dx_arr[itype]*j[itype,:]  
        jqy += 1j/4 * U * t_arr[:,itype]*dy_arr[itype]*j[itype,:]
        jqx += 1j/4 * U * t_arr[:,itype]*dx_arr[itype]*j[itype,pairs[:,itype]]     
        jqy += 1j/4 * U * t_arr[:,itype]*dy_arr[itype]*j[itype,pairs[:,itype]]

        # jn contribution 
        jqx += -1j/4 * U*t_arr[:,itype]*dx_arr[itype]*jn[itype,:]
        jqy += -1j/4 * U*t_arr[:,itype]*dy_arr[itype]*jn[itype,:]
        jqx += -1j/4 * U*t_arr[:,itype]*dx_arr[itype]*jn[itype,pairs[:,itype]]
        jqy += -1j/4 * U*t_arr[:,itype]*dy_arr[itype]*jn[itype,pairs[:,itype]]

    for itype in range(b2ps):
         # j2 contribution 
        jqx += 1j/4 * t2_arr[:,itype]*dx2_arr[itype]*j2[itype,:]
        jqy += 1j/4 * t2_arr[:,itype]*dy2_arr[itype]*j2[itype,:]
        jqx += 1j/4 * t2_arr[:,itype]*dx2_arr[itype]*j2[itype,pairsb2[:,itype]]
        jqy += 1j/4 * t2_arr[:,itype]*dy2_arr[itype]*j2[itype,pairsb2[:,itype]]

    jq = npa(list(zip(jqx, jqy)))
    jN = compute_particle_site_current(jn, **kwargs)
    return jq - mu*jN

def compute_particle_site_current(jn, **kwargs):
    """
    Compute particle site currents. jn is of shape (bps, N) 
    Returns:
        Array (N, 2)
    """
    N = kwargs["N"]
    bps = kwargs["bps"]
    tp = kwargs["tp"]
    bonds = kwargs["bonds"]
    t_arr = np.ones((N, bps)) * npa([1,1,tp,tp]) if tp != 0 else np.ones((N, bps)) 

    jx = 0.0 + 0.0j
    jy = 0.0 + 0.0j
    pairs = npa([bonds[0,np.where(bonds == i)[1]][bps:] for i in range(N)])
    for itype in range(bps):
        jx += 1j/2 * t_arr[:,itype]*dx_arr[itype]*jn[itype,:]
        jy += 1j/2 * t_arr[:,itype]*dy_arr[itype]*jn[itype,:]
        jx += 1j/2 * t_arr[:,itype]*dx_arr[itype]*jn[itype,pairs[:,itype]]
        jy += 1j/2 * t_arr[:,itype]*dy_arr[itype]*jn[itype,pairs[:,itype]]

    return npa(list(zip(jx, jy)))