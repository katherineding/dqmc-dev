import numpy as np

def make_peierls_mat(Nx, Ny, shape, nflux, alpha = 1/2, jr_orbit = np.array((0,0)),ir_orbit = np.array((0,0))):
    beta = 1 - alpha
    if shape == "triangular":
        #triangular lattice unit vectors
        #Lattice total area = Nx * Ny * unit_area * a^2
        a1 = np.array((0.5,np.sqrt(3)/2))
        a2 = np.array((-0.5,np.sqrt(3)/2))
        unit_area = np.sqrt(3)/2
    elif shape == 'square':
        #triangular lattice unit vectors
        #Lattice total area = Nx * Ny * unit_area * a^2
        a1 = np.array((1,0))
        a2 = np.array((0,1))
        unit_area = 1
        
    # p = 2\pi B/ \Phi_0
    prefactor = 2*np.pi*nflux/(unit_area*Nx*Ny)
    
    phi = np.zeros((Ny*Nx, Ny*Nx))
    #displacement index vector: d
    for dy in range((1-Ny)//2, (1+Ny)//2):
        for dx in range((1-Nx)//2, (1+Nx)//2):
            for iy in range(Ny):
                for ix in range(Nx):
                    #start site: j
                    jy = iy + dy
                    jjy = jy % Ny
                    jx = ix + dx
                    jjx = jx % Nx

                    #boundary phase offset = \pm Nx, \pm Ny
                    offset_1 = jx - jjx
                    offset_2 = jy - jjy
                    #offset displacement
                    offsetr = offset_1 * a1 + offset_2 * a2
                    #true spatial location R_i
                    ir = ix * a1 + iy * a2 + ir_orbit
                    #true spatial location R_j
                    jr = jx * a1 + jy * a2 + jr_orbit
                    # wrapped spatial location R_j
                    jjr = jjx * a1 + jjy * a2 + jr_orbit
                    #true displacement distance R_d
                    dr = jr - ir
                    # true displacement mid point    
                    mr = (ir + jr)/2

                    #used in wrapping term
                    A_offset = np.array((-alpha*(offset_1*a1[1]+offset_2*a2[1]),
                                           beta*(offset_1*a1[0]+offset_2*a2[0])))

                    interior = - alpha*mr[1]*dr[0] + beta*mr[0]*dr[1]
                    wrap = np.dot(A_offset,0.5*(jr+jjr))
                    extra = offset_1*offset_2 * \
                        (beta * a2[0] * a1[1] - alpha * a1[0] * a2[1]) 

                    phi[jjx + Nx*jjy,ix + Nx*iy] = interior + (-1) * (wrap + extra)

    peierls = np.exp(1j*prefactor*phi)

    return peierls

def H_periodic_square(Nx, Ny, t=1, tp = 0, nflux = 0, alpha = 1/2):
    # This function is now consistent with existing DQMC simulations
    # First: hopping (assuming periodic boundaries and no field)
    tij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) % Ny
            ix1 = (ix + 1) % Nx
                #jx    jy    ix    iy 
            tij[ix1+Nx*iy , ix +Nx*iy ] += -t
            tij[ix +Nx*iy , ix1+Nx*iy ] += -t
            tij[ix +Nx*iy1, ix +Nx*iy ] += -t
            tij[ix +Nx*iy , ix +Nx*iy1] += -t

            tij[ix +Nx*iy , ix1+Nx*iy1] += -tp
            tij[ix1+Nx*iy1, ix +Nx*iy ] += -tp

            tij[ix1+Nx*iy , ix +Nx*iy1] += -tp
            tij[ix +Nx*iy1, ix1+Nx*iy ] += -tp

    peierls = make_peierls_mat(Nx,Ny,'square',nflux,alpha,np.array((0,0)))

    K = tij * peierls
    # complex hopping matrix
    if nflux > 0:
        #assert np.linalg.norm(peierls - peierls.T.conj()) < 1e-10, "???"
        assert np.linalg.norm(K - K.T.conj()) < 1e-10
    else: #peierls = 1, hopping real
        assert np.max(np.abs(peierls.imag)) < 1e-10
        assert np.max(np.abs(K.imag)) < 1e-10
    return K, peierls

def H_open_square(Nx, Ny, t=1, tp = 0, nflux = 0, alpha = 1/2):
    """Square lattice with open BC in both directions
    NOTE: Peierl's matrix is not modified for open bc but since it always appears with tij, the term goes to zero"""
    tij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) 
            ix1 = (ix + 1) 
             
            if (ix1 > Nx-1) & (iy1 > Ny-1):
                continue    
            elif (ix1 > Nx-1) & (iy1 <= Ny-1):
                tij[ix +Nx*iy1, ix +Nx*iy ] += -t
                tij[ix +Nx*iy , ix +Nx*iy1] += -t
                continue 
            elif (iy1 > Ny-1) & (ix1 <= Nx-1):
                tij[ix1+Nx*iy , ix +Nx*iy ] += -t
                tij[ix +Nx*iy , ix1+Nx*iy ] += -t
                continue

            tij[ix1+Nx*iy , ix +Nx*iy ] += -t
            tij[ix +Nx*iy , ix1+Nx*iy ] += -t
            tij[ix +Nx*iy1, ix +Nx*iy ] += -t
            tij[ix +Nx*iy , ix +Nx*iy1] += -t
            tij[ix +Nx*iy , ix1+Nx*iy1] += -tp
            tij[ix1+Nx*iy1, ix +Nx*iy ] += -tp
            tij[ix1+Nx*iy , ix +Nx*iy1] += -tp
            tij[ix +Nx*iy1, ix1+Nx*iy ] += -tp

    peierls = make_peierls_mat(Nx,Ny,'square',nflux,alpha,np.array((0,0)))

    K = tij * peierls
    # complex hopping matrix
    if nflux > 0:
        #assert np.linalg.norm(peierls - peierls.T.conj()) < 1e-10, "???"
        assert np.linalg.norm(K - K.T.conj()) < 1e-10
    else: #peierls = 1, hopping real
        assert np.max(np.abs(peierls.imag)) < 1e-10
        assert np.max(np.abs(K.imag)) < 1e-10
    return K, peierls

def H_periodic_triangular(Nx, Ny, t=1, tp = 0, nflux = 0, alpha = 1/2):
    """ TB model with Peierls phase:
    H = sum -t_ij exp[i phi_ij] c_i^dag c_j
    phi_ij is phase from j -> i hopping

    """
    if tp != 0:
        raise NotImplementedError
    # First: hopping (assuming periodic boundaries and no field)
    kij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) % Ny
            iyn = (iy - 1) % Ny
            ix1 = (ix + 1) % Nx
            ixn = (ix - 1) % Nx
                #jx    jy    ix    iy 
            kij[ix1+Nx*iy , ix +Nx*iy ] += -t
            kij[ix +Nx*iy , ix1+Nx*iy ] += -t
            kij[ix +Nx*iy1, ix +Nx*iy ] += -t
            kij[ix +Nx*iy , ix +Nx*iy1] += -t
            kij[ix1+Nx*iy , ix +Nx*iy1] += -t
            kij[ix +Nx*iy1, ix1+Nx*iy ] += -t

    peierls = make_peierls_mat(Nx,Ny,'triangular',nflux,alpha,np.array((0,0)))

    K = kij * peierls
    # complex hopping matrix
    if nflux != 0:
        #assert np.linalg.norm(peierls - peierls.T.conj()) < 1e-10, "???"
        assert np.linalg.norm(K - K.T.conj()) < 1e-10
    else: #peierls = 1, hopping real
        assert np.max(np.abs(peierls.imag)) < 1e-10
        assert np.max(np.abs(K.imag)) < 1e-10
    return K, peierls

def H_periodic_honeycomb(Nx, Ny, t=1, tp = 0, nflux = 0, alpha = 1/2):
    # NOTE displacement vectors need to be consistent with primitive vectors

    delta_1 = np.array((1/2, 1/(2*np.sqrt(3))))
    delta_2 = np.array((-1/2, 1/(2*np.sqrt(3)))) 
    delta_3 = np.array((0,-1/np.sqrt(3))) #offset of B relative to A
    
    tij = np.zeros((2*Ny*Nx, 2*Ny*Nx), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) % Ny
            ix1 = (ix + 1) % Nx
            iyn1 = (iy - 1) % Ny
            ixn1 = (ix - 1) % Nx
                #B: Orbital = 1       #A:Orbital = 0
                #jx      jy    jo     ix       iy    io
                #These are a_i^\dagger b_j terms
            tij[ix +Nx*iy1+ Nx*Ny*1, ix +Nx*iy  + Nx*Ny*0] += -t #rB(ix,iy+1) = rA(ix,iy) + delta_1
            tij[ix1+Nx*iy + Nx*Ny*1, ix +Nx*iy  + Nx*Ny*0] += -t #rB(ix+1,iy) = rA(ix,iy) + delta_2
            tij[ix +Nx*iy + Nx*Ny*1, ix +Nx*iy  + Nx*Ny*0] += -t #rB(ix,iy  ) = rA(ix,iy) + delta_3
                #A: Orbital = 0       #B: Orbital = 1 
                #jx      jy    jo     ix       iy    io
                #these are b_j^\dagger a_i terms
            tij[ix +Nx*iy  + Nx*Ny*0, ix +Nx*iy1+ Nx*Ny*1] += -t
            tij[ix +Nx*iy  + Nx*Ny*0, ix1+Nx*iy + Nx*Ny*1] += -t
            tij[ix +Nx*iy  + Nx*Ny*0, ix +Nx*iy + Nx*Ny*1] += -t

                #A: Orbital = 0       #A:Orbital = 0
                #jx      jy    jo     ix       iy    io
                #These are a^\dagger_{i+\delta'} a_i + a^\dagger_i a_{i+\delta'}  terms
            tij[ix1+Nx*iy  + Nx*Ny*0, ix +Nx*iy  + Nx*Ny*0] += -tp
            tij[ix +Nx*iy  + Nx*Ny*0, ix1+Nx*iy  + Nx*Ny*0] += -tp

            tij[ix +Nx*iy1 + Nx*Ny*0, ix +Nx*iy  + Nx*Ny*0] += -tp
            tij[ix +Nx*iy  + Nx*Ny*0, ix +Nx*iy1 + Nx*Ny*0] += -tp

            tij[ix1+Nx*iy  + Nx*Ny*0, ix +Nx*iy1 + Nx*Ny*0] += -tp
            tij[ix +Nx*iy1 + Nx*Ny*0, ix1+Nx*iy  + Nx*Ny*0] += -tp
                
                #B: Orbital = 1       #B:Orbital = 1
                #jx      jy    jo     ix       iy    io
                #These are a^\dagger_{i+\delta'} a_i + a^\dagger_i a_{i+\delta'}  terms
            tij[ix1+Nx*iy  + Nx*Ny*1, ix +Nx*iy  + Nx*Ny*1] += -tp
            tij[ix +Nx*iy  + Nx*Ny*1, ix1+Nx*iy  + Nx*Ny*1] += -tp

            tij[ix +Nx*iy1 + Nx*Ny*1, ix +Nx*iy  + Nx*Ny*1] += -tp
            tij[ix +Nx*iy  + Nx*Ny*1, ix +Nx*iy1 + Nx*Ny*1] += -tp

            tij[ix1+Nx*iy  + Nx*Ny*1, ix +Nx*iy1 + Nx*Ny*1] += -tp
            tij[ix +Nx*iy1 + Nx*Ny*1, ix1+Nx*iy  + Nx*Ny*1] += -tp


    peierls_diag = make_peierls_mat(Nx,Ny,"triangular",nflux,alpha=alpha,jr_orbit = np.array((0,0)))

    peierls_offdiag = make_peierls_mat(Nx,Ny,"triangular",nflux,alpha=alpha,jr_orbit = delta_3)

    K = tij.copy()
    P = np.full((2*Ny*Nx, 2*Ny*Nx), np.nan, dtype=np.complex128)

    #(0,0) and (1,1) block hop according to triangular lattice
    # TODO: check off-diagonal has correct direction/sign for phase
    P[:Nx*Ny,:Nx*Ny] =  peierls_diag
    P[Nx*Ny:,Nx*Ny:] =  peierls_diag
    P[:Nx*Ny,Nx*Ny:] =  peierls_offdiag.T.conj()
    P[Nx*Ny:,:Nx*Ny] =  peierls_offdiag


    #(0,0) and (1,1) block hop according to triangular lattice
    K[:Nx*Ny,:Nx*Ny] *=  peierls_diag
    K[Nx*Ny:,Nx*Ny:] *=  peierls_diag

    #(0,1) and (1,0) block hop according to ??
    K[:Nx*Ny,Nx*Ny:] *=  peierls_offdiag.T.conj()
    K[Nx*Ny:,:Nx*Ny] *=  peierls_offdiag

    if nflux != 0:
        #assert np.linalg.norm(peierls_diag - peierls_diag.T.conj()) < 1e-10, "???"
        assert np.linalg.norm(K - K.T.conj()) < 1e-10
    else: #peierls = 1, hopping real
        assert np.max(np. abs(peierls_diag.imag)) < 1e-10
        assert np.max(np. abs(peierls_offdiag.imag)) < 1e-10
        assert np.max(np.abs(K.imag)) < 1e-10

    return K, P

def H_periodic_kagome(Nx, Ny, t=1, tp = 0, nflux = 0, alpha = 1/2):
    if tp != 0:
        raise NotImplementedError
    
    tij = np.zeros((3*Ny*Nx, 3*Ny*Nx), dtype=np.complex128)
    N = Nx*Ny
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) % Ny
            ix1 = (ix + 1) % Nx
            iyn1 = (iy - 1) % Ny
            ixn1 = (ix - 1) % Nx
                #B: Orbital = 1       #A:Orbital = 0
                #jx      jy    jo     ix       iy    io
                #These are a_i^\dagger b_j terms
            tij[ix +Nx*iy1  +N*1, ix +Nx*iy +N*0] += -t #rB(ix,iy+1) = rA(ix,iy) + delta_2
            tij[ix +Nx*iy   +N*1, ix +Nx*iy +N*0] += -t #rB(ix,iy  ) = rA(ix,iy) + delta_4
                #A: Orbital = 0       #B: Orbital = 1 
                #jx      jy    jo     ix       iy    io
                #these are b_j^\dagger a_i terms
            tij[ix +Nx*iy +N*0, ix +Nx*iy1  +N*1] += -t
            tij[ix +Nx*iy +N*0, ix +Nx*iy   +N*1] += -t

                #A:Orbital = 0        #C: Orbital = 2
                #jx      jy    jo     ix       iy    io
                #These are c_i^\dagger a_j terms
            tij[ixn1 +Nx*iy  +N*0, ix +Nx*iy +N*2] += -t #rA(ix-1,iy) = rC(ix,iy) + delta_3
            tij[ix   +Nx*iy  +N*0, ix +Nx*iy +N*2] += -t #rA(ix,iy) = rC(ix,iy) + delta_1
            
                #C: Orbital = 2       #A: Orbital = 0
                #jx      jy    jo     ix       iy    io
                #these are a_j^\dagger c_i terms
            tij[ix +Nx*iy +N*2, ixn1 +Nx*iy +N*0] += -t 
            tij[ix +Nx*iy +N*2, ix   +Nx*iy +N*0] += -t 

                #C: Orbital = 2       #B:Orbital = 1
                #jx      jy    jo     ix       iy    io
                #These are b_i^\dagger c_j terms
            tij[ix1 +Nx*iyn1 +N*2, ix +Nx*iy +N*1] += -t #rC(ix+1,iy-1) = rB(ix,iy) + delta_6
            tij[ix  +Nx*iy   +N*2, ix +Nx*iy +N*1] += -t #rC(ix,iy) = rB(ix,iy) + delta_5
                #B: Orbital = 1       #C: Orbital = 2 
                #jx      jy    jo     ix       iy    io
                #these are c_j^\dagger b_i terms
            tij[ix +Nx*iy +N*1, ix1 +Nx*iyn1 +N*2] += -t
            tij[ix +Nx*iy +N*1, ix  +Nx*iy   +N*2] += -t

    #peierls phase with offsets
    delta_4 = np.array((1/4,-np.sqrt(3)/4)) #offset of B relative to A
    peierls_01 = make_peierls_mat(Nx,Ny,"triangular",nflux,alpha=alpha,jr_orbit = delta_4,ir_orbit = np.array((0,0)))
    delta_3 = np.array((-1/4,-np.sqrt(3)/4)) #offset of C relative to A
    peierls_20 = make_peierls_mat(Nx,Ny,"triangular",nflux,alpha=alpha,jr_orbit = 0, ir_orbit = delta_3)
    #delta_5 = np.array((-1/2, 0)) #offset of C relative to B
    peierls_12 = make_peierls_mat(Nx,Ny,"triangular",nflux,alpha=alpha,jr_orbit = delta_3, ir_orbit = delta_4)
    #print(delta_3 - delta_4)
    K = tij.copy()

    P = np.full((3*Ny*Nx, 3*Ny*Nx), np.nan, dtype=np.complex128)
    peierls_diag = make_peierls_mat(Nx,Ny,"triangular",nflux,alpha=alpha,jr_orbit = np.array((0,0)))

    #(0,0) and (1,1) block hop according to triangular lattice
    # TODO: check off-diagonal has correct direction/sign for phase
    P[0  :1*N, 0  :1*N] =  peierls_diag
    P[N  :2*N, N  :2*N] =  peierls_diag
    P[2*N:3*N, 2*N:3*N] =  peierls_diag
    
    P[0  :1*N, 1*N:2*N] =  peierls_01.T.conj()
    P[1*N:2*N, 0  :1*N] =  peierls_01
    P[2*N:3*N, 0  :1*N] =  peierls_20.T.conj()
    P[0  :1*N, 2*N:3*N] =  peierls_20
    P[2*N:3*N, 1*N:2*N] =  peierls_12
    P[1*N:2*N, 2*N:3*N] =  peierls_12.T.conj()
    
    K[1*N : 2*N, 0 : 1*N] *=  peierls_01
    K[0 : 1*N, 1*N : 2*N] *=  peierls_01.T.conj()

    K[2*N : 3*N, 0 : 1*N] *=  peierls_20.T.conj()
    K[0 : 1*N, 2*N : 3*N] *=  peierls_20

    K[2*N : 3*N, 1*N : 2*N] *=  peierls_12
    K[1*N : 2*N, 2*N : 3*N] *=  peierls_12.T.conj()

    if nflux != 0:
        #assert np.linalg.norm(peierls - peierls.T.conj()) < 1e-10, "???"
        assert np.linalg.norm(K - K.T.conj()) < 1e-10
    else: #peierls = 1, hopping real
        assert np.max(np. abs(peierls_01.imag)) < 1e-10
        assert np.max(np. abs(peierls_20.imag)) < 1e-10
        assert np.max(np. abs(peierls_12.imag)) < 1e-10
        assert np.max(np.abs(K.imag)) < 1e-10

    return K, P
