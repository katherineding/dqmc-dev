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

time_slice = 0;


def j2j2_jxding(path):
    U, beta, Nx, Ny, bps, b2ps, nflux, tp, N, L, dt= \
        util.load_firstfile(path,"metadata/U","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps",\
            "metadata/b2ps", "metadata/nflux",\
            "metadata/t'","params/N","params/L","params/dt")    
    ns, s, j2j2, init_rng = util.load(path, 
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", \
        "meas_uneqlt/j2j2","state/init_rng")

    j2j2.shape = -1, L, b2ps, b2ps, Ny, Nx
    v_q0 = j2j2.sum((-1,-2))
    accum = 0;
    pre_array = [1,1,1,1,tp,tp,tp,tp,tp,tp,tp**2,tp**2]
    for itype in range(b2ps):
        for jtype in range(b2ps):
            accum += pre_array[itype]*pre_array[jtype]*v_q0[:,time_slice,itype,jtype]

    print("j2j2 jxding",accum); 

    return accum

def j2j2_wen(path):
    U, beta, Nx, Ny, bps, b2ps, tp, N, L, dt= \
        util.load_firstfile(path,"metadata/U","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps",\
            "metadata/b2ps", \
            "metadata/t'","params/N","params/L","params/dt")    
    ns, s, j2j2, init_rng = util.load(path, 
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", \
        "meas_uneqlt/j2j2","state/init_rng")

    j2j2.shape = -1, L, b2ps, b2ps, Ny, Nx
    v_q0 = j2j2.sum((-1,-2))

    pre_array = [4*tp, 4*tp,  2,  2,  1+2*tp**2, 1+2*tp**2, 2*tp, 2*tp, tp**2, 2*tp, 2*tp, tp**2]
    accum = 0;
    for itype in range(b2ps):
        for jtype in range(b2ps):
            accum += pre_array[itype]*pre_array[jtype]*v_q0[:,time_slice,itype,jtype]

    print("j2j2 wen   ",accum); 

    return accum


def jnj2_jxding(path):
    U, beta, Nx, Ny, bps, b2ps, nflux, tp, N, L, dt, init_rng= \
        util.load_firstfile(path,"metadata/U","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps","metadata/b2ps",\
            "metadata/nflux",\
            "metadata/t'","params/N","params/L","params/dt","state/init_rng")    
    ns, s, jnj2= util.load(path, 
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", \
        "meas_uneqlt/jnj2")

    jnj2.shape = -1, L, b2ps, bps,Ny, Nx

    #sum over all locations i.e. take q=0
    v_q0 = jnj2.sum((-1,-2))  #shape = -1, L, bps, hop2ps

    #print(v_q0.shape)
    accum = 0;
    pre_array = [1,1,1,1,tp,tp,tp,tp,tp,tp,tp**2,tp**2]
    pre_array_bond = [1,1,tp,tp]
        
    for itype in range(b2ps):
        for jtype in range(bps):
            accum += pre_array[itype]*pre_array_bond[jtype]*v_q0[:,time_slice,itype,jtype]

    print("jnj2 jxding",accum); 

    return accum

def jnj2_wen(path):

    U, beta, Nx, Ny, bps, b2ps, tp, N, L, dt, init_rng= \
        util.load_firstfile(path,"metadata/U","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps",\
            "metadata/b2ps", \
            "metadata/t'","params/N","params/L","params/dt","state/init_rng")    
    ns, s, jnj2 = util.load(path, 
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", \
        "meas_uneqlt/jnj")

    jnj2.shape = -1, L, b2ps, bps, Ny, Nx
    v_q0 = jnj2.sum((-1,-2))
    accum = 0;
    pre_array = [4*tp, 4*tp,  2,  2,  1+2*tp**2, 1+2*tp**2, 2*tp, 2*tp, tp**2, 2*tp, 2*tp, tp**2]
    pre_array_bond = [1,1,tp,tp]
    for itype in range(b2ps):
        for jtype in range(bps):
            accum += pre_array[itype]*pre_array_bond[jtype]*v_q0[:,time_slice,itype,jtype]

    print("jnj2 wen   ",accum); 

    return accum


def jj2_jxding(path):
    U, beta, Nx, Ny, bps, b2ps, nflux, tp, N, L, dt, init_rng= \
        util.load_firstfile(path,"metadata/U","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps","metadata/b2ps",\
            "metadata/nflux",\
            "metadata/t'","params/N","params/L","params/dt","state/init_rng")    
    ns, s, jj2= util.load(path, 
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", \
        "meas_uneqlt/jj2")

    jj2.shape = -1, L,  b2ps, bps,Ny, Nx

    #sum over all locations i.e. take q=0
    v_q0 = jj2.sum((-1,-2))  #shape = -1, L, b2ps, bps

    accum = 0;
    pre_array = [1,1,1,1,tp,tp,tp,tp,tp,tp,tp**2,tp**2]
    pre_array_bond = [1,1,tp,tp]
    for itype in range(b2ps):
        for jtype in range(bps):
            accum += pre_array[itype]*pre_array_bond[jtype]*v_q0[:,time_slice,itype,jtype]

    print("jj2 jxding",accum); 

    return accum


def jj2_wen(path):

    U, beta, Nx, Ny, bps, b2ps, tp, N, L, dt, init_rng= \
        util.load_firstfile(path,"metadata/U","metadata/beta",\
            "metadata/Nx","metadata/Ny","metadata/bps",\
            "metadata/b2ps", \
            "metadata/t'","params/N","params/L","params/dt","state/init_rng")    
    ns, s, j2j2 = util.load(path, 
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", \
        "meas_uneqlt/j2j2")


    j2j2.shape = -1, L, b2ps, b2ps, Ny, Nx
    v_q0 = j2j2.sum((-1,-2))
    accum = 0;
    pre_array_bond = [1,1,tp,tp]
    pre_array = [4*tp, 4*tp,  2,  2,  1+2*tp**2, 1+2*tp**2, 2*tp, 2*tp, tp**2, 2*tp, 2*tp, tp**2]
    for itype in range(b2ps):
        for jtype in range(0,4):
            accum += pre_array[itype]*pre_array_bond[jtype]*v_q0[:,time_slice,itype,jtype]

    print("jj2 wen   ",accum); 

    return accum