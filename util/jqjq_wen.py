import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

import sys
import os
src = os.environ['SRC']
if not src+"edwin/util" in sys.path:
    sys.path.insert(0,src+"edwin/util")
import util #Edwin's util file
import data_analysis as da

#one-hop bond dx, dy
dx_arr = [1,0,1,-1]
dy_arr = [0,1,1, 1]

def get_component(path,name="j2j2"): 
    """
    Address RAM overflow kill:
        By separating components out, python only keeps full
        shape (Nbin_completed, L, b[2]ps, b[2]ps, Nx, Ny) of one type of correlator
        in memory at a time. Performs summation over (Nx, Ny) spatial components to
        return correlator(Q=0) for maxent analysis. Not divided by sign
    """

    Nx, Ny, bps, b2ps, N, L= util.load_firstfile(path,\
        "metadata/Nx","metadata/Ny","metadata/bps",\
        "metadata/b2ps", "params/N","params/L")  

    ns, correlator = util.load(path, "meas_uneqlt/n_sample", \
        f"meas_uneqlt/{name}")

    if name == "j2j2":
        # four phases
        correlator.shape = -1, L, b2ps, b2ps, Ny, Nx
    elif name == "jnj":
        # three phases
        correlator.shape = -1, L, b2ps, bps, Ny, Nx
    elif name == "jjn":
        # three phases
        correlator.shape = -1, L, bps, b2ps, Ny, Nx
    elif name == "jnjn":
        # two phases
        correlator.shape = -1, L, bps, bps, Ny, Nx
    else:
        raise ValueError("invalid correlator name")

    #use only completed bins
    mask = ns == ns.max();
    correlator = correlator[mask] 
    #take q == 0, don't divide by sign
    correlator_q0 = correlator.sum((-1,-2))

    return correlator_q0


def thermal_sum(path,q0_corrs):
    
    U, mu, beta, Nx, Ny, bps, b2ps, tp, N, L, dt= \
        util.load_firstfile(path,"metadata/U","metadata/mu","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps","metadata/b2ps", \
            "metadata/t'","params/N","params/L","params/dt")

    j2j2_q0,\
    jjn_q0, jnj_q0,\
    jnjn_q0 = q0_corrs

    t_arr = [1,1,tp,tp]
    # 2bond type t factors
    t2_arr = [4*tp,4*tp,2,2,1+2*tp**2,1+2*tp**2,2*tp,2*tp,tp**2,2*tp,2*tp,tp**2]
    # wen's two-hop bond dx,dy distances
    dx2_arr = [1,0,1,-1,2,0,2,1,2,-2,-1,-2]
    dy2_arr = [0,1,1, 1,0,2,1,2,2, 1, 2, 2]
    result_dict = {}

    #bond-bond types: jj, jnj, jjn, jnjn
    jj_xx   =   jj_yy = 0;
    jnjn_xx = jnjn_yy = 0;
    jnj_xx  =  jnj_yy = 0;
    jjn_xx  =  jjn_yy = 0;
    jj_xy   =   jj_yx = 0;
    jnjn_xy = jnjn_yx = 0;
    jnj_xy  =  jnj_yx = 0;
    jjn_xy  =  jjn_yx = 0;
    for itype in range(bps):
        for jtype in range(bps):
            jj_xx += t_arr[itype]*t_arr[jtype]*dx_arr[itype]*dx_arr[jtype]*\
                j2j2_q0[:,:,itype,jtype]
            jj_yy += t_arr[itype]*t_arr[jtype]*dy_arr[itype]*dy_arr[jtype]*\
                j2j2_q0[:,:,itype,jtype]
            jnjn_xx += t_arr[itype]*t_arr[jtype]*dx_arr[itype]*dx_arr[jtype]*\
                jnjn_q0[:,:,itype,jtype]
            jnjn_yy += t_arr[itype]*t_arr[jtype]*dy_arr[itype]*dy_arr[jtype]*\
                jnjn_q0[:,:,itype,jtype]
            jnj_xx += t_arr[itype]*t_arr[jtype]*dx_arr[itype]*dx_arr[jtype]*\
                jnj_q0[:,:,itype,jtype]
            jnj_yy += t_arr[itype]*t_arr[jtype]*dy_arr[itype]*dy_arr[jtype]*\
                jnj_q0[:,:,itype,jtype]
            jjn_xx += t_arr[itype]*t_arr[jtype]*dx_arr[itype]*dx_arr[jtype]*\
                jjn_q0[:,:,itype,jtype]
            jjn_yy += t_arr[itype]*t_arr[jtype]*dy_arr[itype]*dy_arr[jtype]*\
                jjn_q0[:,:,itype,jtype]

            jj_xy += t_arr[itype]  *t_arr[jtype]*dx_arr[itype]*dy_arr[jtype]*\
                j2j2_q0[:,:,itype,jtype]
            jj_yx += t_arr[itype]  *t_arr[jtype]*dy_arr[itype]*dx_arr[jtype]*\
                j2j2_q0[:,:,itype,jtype]
            jnjn_xy += t_arr[itype]*t_arr[jtype]*dx_arr[itype]*dy_arr[jtype]*\
                jnjn_q0[:,:,itype,jtype]
            jnjn_yx += t_arr[itype]*t_arr[jtype]*dy_arr[itype]*dx_arr[jtype]*\
                jnjn_q0[:,:,itype,jtype]
            jnj_xy += t_arr[itype] *t_arr[jtype]*dx_arr[itype]*dy_arr[jtype]*\
                jnj_q0[:,:,itype,jtype]
            jnj_yx += t_arr[itype] *t_arr[jtype]*dy_arr[itype]*dx_arr[jtype]*\
                jnj_q0[:,:,itype,jtype]
            jjn_xy += t_arr[itype] *t_arr[jtype]*dx_arr[itype]*dy_arr[jtype]*\
                jjn_q0[:,:,itype,jtype]
            jjn_yx += t_arr[itype] *t_arr[jtype]*dy_arr[itype]*dx_arr[jtype]*\
                jjn_q0[:,:,itype,jtype]

    #Note: this is the JNJN correlator with (-i)^2 factor!
    result_dict["JNJN"] = (-1) * np.stack((jj_xx, jj_yy, jj_xy, jj_yx),axis=0)

    #==========================================================================
    # 2bond-2bond types: j2j2
    j2j2_xx = j2j2_yy = 0;
    j2j2_xy = j2j2_yx = 0;
    for itype in range(b2ps):
        for jtype in range(b2ps):
            j2j2_xx += t2_arr[itype]*t2_arr[jtype]*\
                dx2_arr[itype]*dx2_arr[jtype]*j2j2_q0[:,:,itype,jtype]
            j2j2_yy += t2_arr[itype]*t2_arr[jtype]*\
                dy2_arr[itype]*dy2_arr[jtype]*j2j2_q0[:,:,itype,jtype]
            j2j2_xy += t2_arr[itype]*t2_arr[jtype]*\
                dx2_arr[itype]*dy2_arr[jtype]*j2j2_q0[:,:,itype,jtype]
            j2j2_yx += t2_arr[itype]*t2_arr[jtype]*\
                dy2_arr[itype]*dx2_arr[jtype]*j2j2_q0[:,:,itype,jtype]

    #==========================================================================
    # 2bond-1bond types
    j2jn_xx = j2jn_yy = 0;
    j2j_xx  =  j2j_yy = 0
    j2jn_xy = j2jn_yx = 0;
    j2j_xy  =  j2j_yx = 0
    for itype in range(bps):
        for jtype in range(b2ps):
            j2jn_xx += t_arr[itype]*t2_arr[jtype]*dx_arr[itype]*dx2_arr[jtype]*\
                       jjn_q0[:,:,itype,jtype]
            j2jn_yy += t_arr[itype]*t2_arr[jtype]*dy_arr[itype]*dy2_arr[jtype]*\
                       jjn_q0[:,:,itype,jtype]
            j2j_xx +=  t_arr[itype]*t2_arr[jtype]*dx_arr[itype]*dx2_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]
            j2j_yy +=  t_arr[itype]*t2_arr[jtype]*dy_arr[itype]*dy2_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]
            j2jn_xy += t_arr[itype]*t2_arr[jtype]*dx_arr[itype]*dy2_arr[jtype]*\
                       jjn_q0[:,:,itype,jtype]
            j2jn_yx += t_arr[itype]*t2_arr[jtype]*dy_arr[itype]*dx2_arr[jtype]*\
                       jjn_q0[:,:,itype,jtype]
            j2j_xy +=  t_arr[itype]*t2_arr[jtype]*dx_arr[itype]*dy2_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]
            j2j_yx +=  t_arr[itype]*t2_arr[jtype]*dy_arr[itype]*dx2_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]

    #==========================================================================
    # 1bond-2bond types
    jnj2_xx = jnj2_yy = 0;
    jj2_xx  =  jj2_yy = 0;
    jnj2_xy = jnj2_yx = 0;
    jj2_xy  =  jj2_yx = 0;
    for itype in range(b2ps):
        for jtype in range(bps):
            jnj2_xx += t2_arr[itype]*t_arr[jtype]*dx2_arr[itype]*dx_arr[jtype]*\
                       jnj_q0[:,:,itype,jtype]
            jnj2_yy += t2_arr[itype]*t_arr[jtype]*dy2_arr[itype]*dy_arr[jtype]*\
                       jnj_q0[:,:,itype,jtype]
            jj2_xx +=  t2_arr[itype]*t_arr[jtype]*dx2_arr[itype]*dx_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]
            jj2_yy +=  t2_arr[itype]*t_arr[jtype]*dy2_arr[itype]*dy_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]
            jnj2_xy += t2_arr[itype]*t_arr[jtype]*dx2_arr[itype]*dy_arr[jtype]*\
                       jnj_q0[:,:,itype,jtype]
            jnj2_yx += t2_arr[itype]*t_arr[jtype]*dy2_arr[itype]*dx_arr[jtype]*\
                       jnj_q0[:,:,itype,jtype]
            jj2_xy +=  t2_arr[itype]*t_arr[jtype]*dx2_arr[itype]*dy_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]
            jj2_yx +=  t2_arr[itype]*t_arr[jtype]*dy2_arr[itype]*dx_arr[jtype]*\
                       j2j2_q0[:,:,itype,jtype]

    #==========================================================================
    # prefactors required to actually form JQ(tau)JQ(0) operator
    # see transport notes
    pre_arr = 1/4*np.outer([1,-U,U+2*mu],[1,-U,U+2*mu]) * (-1)
    xx = pre_arr[0,0]*j2j2_xx + pre_arr[0,1]*j2jn_xx + pre_arr[0,2]*j2j_xx + \
         pre_arr[1,0]*jnj2_xx + pre_arr[1,1]*jnjn_xx + pre_arr[1,2]*jnj_xx + \
         pre_arr[2,0]*jj2_xx  + pre_arr[2,1]*jjn_xx  + pre_arr[2,2]*jj_xx 

    yy = pre_arr[0,0]*j2j2_yy + pre_arr[0,1]*j2jn_yy + pre_arr[0,2]*j2j_yy + \
         pre_arr[1,0]*jnj2_yy + pre_arr[1,1]*jnjn_yy + pre_arr[1,2]*jnj_yy + \
         pre_arr[2,0]*jj2_yy  + pre_arr[2,1]*jjn_yy  + pre_arr[2,2]*jj_yy 

    xy = pre_arr[0,0]*j2j2_xy + pre_arr[0,1]*j2jn_xy + pre_arr[0,2]*j2j_xy + \
         pre_arr[1,0]*jnj2_xy + pre_arr[1,1]*jnjn_xy + pre_arr[1,2]*jnj_xy + \
         pre_arr[2,0]*jj2_xy  + pre_arr[2,1]*jjn_xy  + pre_arr[2,2]*jj_xy 

    yx = pre_arr[0,0]*j2j2_yx + pre_arr[0,1]*j2jn_yx + pre_arr[0,2]*j2j_yx + \
         pre_arr[1,0]*jnj2_yx + pre_arr[1,1]*jnjn_yx + pre_arr[1,2]*jnj_yx + \
         pre_arr[2,0]*jj2_yx  + pre_arr[2,1]*jjn_yx  + pre_arr[2,2]*jj_yx 

    result_dict["j2j2"] = np.stack((j2j2_xx, j2j2_yy, j2j2_xy, j2j2_yx),axis=0)
    result_dict["j2jn"] = np.stack((j2jn_xx, j2jn_yy, j2jn_xy, j2jn_yx),axis=0)
    result_dict["j2j"] =  np.stack((j2j_xx,   j2j_yy,  j2j_xy,  j2j_yx),axis=0)
    result_dict["jnj2"] = np.stack((jnj2_xx, jnj2_yy, jnj2_xy, jnj2_yx),axis=0)
    result_dict["jnjn"] = np.stack((jnjn_xx, jnjn_yy, jnjn_xy, jnjn_yx),axis=0)
    result_dict["jnj"] =  np.stack((jnj_xx,  jnj_yy,   jnj_xy,  jnj_yx),axis=0)
    result_dict["jj2"] =  np.stack((jj2_xx,  jj2_yy,   jj2_xy,  jj2_yx),axis=0)
    result_dict["jjn"] =  np.stack((jjn_xx,  jjn_yy,   jjn_xy,  jjn_yx),axis=0)
    result_dict["jj"] =   np.stack((jj_xx,    jj_yy,    jj_xy,   jj_yx),axis=0)
    result_dict["JQJQ"] = np.stack((   xx,       yy,       xy,      yx),axis=0)
    result_dict['metadata'] = (L,dt)

    #==========================================================================
    #prefactors same for all three methods for JQ(tau)JN(0),JN(tau)JQ(0)
    #TODO: check if this is actually consistent with bond, map definitions
    #JQJN and JNJQ might be flipped around
    pre_arr = np.array([1,-U,U+2*mu]) * 1/2
    xx = pre_arr[0]*j2j_xx + pre_arr[1]*jnj_xx + pre_arr[2]*jj_xx
    yy = pre_arr[0]*j2j_yy + pre_arr[1]*jnj_yy + pre_arr[2]*jj_yy
    xy = pre_arr[0]*j2j_xy + pre_arr[1]*jnj_xy + pre_arr[2]*jj_xy
    yx = pre_arr[0]*j2j_yx + pre_arr[1]*jnj_yx + pre_arr[2]*jj_yx

    result_dict["JQJN"] =  np.stack((   xx,     yy,      xy,     yx),axis=0)

    #==========================================================================
    xx = pre_arr[0]*jj2_xx + pre_arr[1]*jjn_xx + pre_arr[2]*jj_xx
    yy = pre_arr[0]*jj2_yy + pre_arr[1]*jjn_yy + pre_arr[2]*jj_yy
    xy = pre_arr[0]*jj2_xy + pre_arr[1]*jjn_xy + pre_arr[2]*jj_xy
    yx = pre_arr[0]*jj2_yx + pre_arr[1]*jjn_yx + pre_arr[2]*jj_yx

    result_dict["JNJQ"] =  np.stack((   xx,     yy,      xy,     yx), axis=0)
    return result_dict

def plot_components(result_dict):
    L, dt = result_dict["metadata"] 
    taus = range(L)*dt
    for k,v in result_dict.items():
        if k == "metadata": continue
        else:
            xx,yy = v[0],v[1];
            #print(np.linalg.norm(xx.real),np.linalg.norm(xx.imag))
            plt.figure()
            plt.title(k)
            plt.errorbar(taus,(xx.real).mean(0),yerr=xx.std(0), 
                fmt='s-',label = "xx real")
            plt.errorbar(taus,(xx.imag).mean(0),yerr=xx.std(0), 
                fmt='.', label = "xx imag")
            plt.errorbar(taus,(yy.real).mean(0),yerr=yy.std(0), 
                fmt='s-',label = "yy real")
            plt.errorbar(taus,(yy.imag).mean(0),yerr=yy.std(0), 
                fmt='.', label = "yy imag")
            plt.xlabel(r"$\tau$")
            plt.legend()


def wen_correlators(path):

    # if da.info(path,uneqlt=True,show=True,imagtol=1e-2) == 1: 
    #     return None

    U, mu, beta, Nx, Ny, bps, b2ps, tp, N, L, dt= \
        util.load_firstfile(path,"metadata/U","metadata/mu","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps","metadata/b2ps", \
            "metadata/t'","params/N","params/L","params/dt")    
    
    j2j2_q0 = get_component(path,'j2j2')
    jjn_q0  = get_component(path,'jjn')
    jnj_q0  = get_component(path,'jnj')
    jnjn_q0 = get_component(path,'jnjn')


    ns, s = util.load(path,"meas_uneqlt/n_sample", "meas_uneqlt/sign")
    mask = ns == ns.max(); nbin = mask.sum()
    ns, s = ns[mask], s[mask]


    #NOTE: no error analysis, just divided by sign
    j2j2_q0 /= np.mean(s)

    jjn_q0 /= np.mean(s)
    jnj_q0 /= np.mean(s)

    jnjn_q0 /= np.mean(s)

    q0_corrs = (j2j2_q0,\
        jjn_q0, jnj_q0,\
        jnjn_q0)

    return thermal_sum(path,q0_corrs)

