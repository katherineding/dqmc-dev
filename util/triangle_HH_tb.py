import numpy as np



# given a label within the triangular lattice, returns the vector of neighbors
# from the square lattice with extra \ perspective, the order is:
# right, up, up left, left, down, down right
def neighbor_list(Nx: int, Ny: int, N:int):
    
    def x_y_to_N(x: int, y: int):
        return (x + y * Nx)
    
    # decompose N into the x and y
    x, y = (N % Nx), (N // Nx)

    # find all the neighbors
    right       = x_y_to_N((x+1) % Nx, y)
    up          = x_y_to_N(x, (y+1) % Ny)
    up_left     = x_y_to_N((x + Nx - 1) % Nx, (y+1) % Ny)
    left        = x_y_to_N((x + Nx - 1) % Nx, y)
    down        = x_y_to_N(x, (y + Ny - 1) % Ny)
    down_right  = x_y_to_N((x + 1) % Nx, (y + Ny - 1) % Ny)

    neighbors = np.array([right, up, up_left, left, down, down_right])

    return neighbors

# NOTE: flux for this is set by hand to be pi/2 per triangle
# there is no real reason to split the two apart for me right now, but I am doing it
# as practice for the future in general
def H_periodic_triangular(
    Nx: int,
    Ny: int,
    t: float = 1.0,
    tp: float = 0.0,
    tpp: float = 0.0,
    nflux: float = 0.0,
    alpha: float = 1 / 2
) -> tuple[np.ndarray, np.ndarray]:
    
    if tp != 0 or tpp != 0:
        raise NotImplementedError
    
    # as of now, my construction only works for an even Nx and Ny, otherwise
    # the hopping scheme breaks. 
    if (Nx % 2) != 0 or (Ny % 2) != 0:
        raise NotImplementedError  
      
    # define function for going x,y -> label
    def x_y_to_N(x: int, y: int):
        return (x + y * Nx)

    # begin by initializing the kinetic matrix
    kij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)
    
    for site in range(Ny*Nx):
        neighbors = neighbor_list(Nx, Ny, site)
        for neighbor in neighbors:
            kij[neighbor][site] = 1     # form is matrix[row][column]

    # this step finalizes the part of the kinetic matrix with no phase
    kij *= -t

    hopping_phases = np.ones((Ny*Nx, Ny*Nx), dtype=np.complex128)

    # go from middle of big magnetic cell to middle big magnetic cell.
    # first, make a list of all the centers.
    central_sites_list = []
    for y in range(Ny // 2):
        for x in range(Nx // 2):
            central_sites_list.append(x_y_to_N(2*x + 1, 2*y + 1))

    # define the vector of signs for each spot when they count. This goes R, U, UL, L, D, DR on the square
    # lattice version of our triangular lattice
    # NOTE it *is* different from the ordering of the sites in local site

    signs_up_left = [-1, 1, -1, -1, 1, -1]
    signs_up = [1, -1, -1, 1, -1, -1]
    signs_up_right = [-1, 1, -1, -1, 1, -1]
    signs_left = [-1, -1, 1, -1, -1, 1]
    signs_middle = [1, 1, 1, 1, 1, 1]
    signs_right = [-1, -1, 1, -1, -1, 1]
    signs_down_left = [-1, 1, -1, -1, 1, -1]
    signs_down = [1, -1, -1, 1, -1, -1]
    signs_down_right = [-1, 1, -1, -1, 1, -1]

    all_signs = [signs_up_left, signs_up, signs_up_right, signs_left,
                 signs_middle, signs_right, signs_down_left, signs_down,
                 signs_down_right]
    
    for central_site in central_sites_list:
        # decompose the middle site into its x and y
        x, y = (central_site % Nx), (central_site // Nx)

        # identify the positions of the region
        up_left = x_y_to_N((x-1+Nx) % Nx, (y+1) % Ny)
        up = x_y_to_N(x, (y+1) % Ny)
        up_right = x_y_to_N((x+1) % Nx, (y+1) % Ny)
        left = x_y_to_N((x-1+Nx) % Nx, y)
        middle = central_site
        right = x_y_to_N((x+1) % Nx, y)
        down_left = x_y_to_N((x-1+Nx) % Nx, (y-1+Ny) % Ny)
        down = x_y_to_N(x, (y-1+Ny) % Ny)
        down_right = x_y_to_N((x+1) % Nx, (y-1+Ny) % Ny)

        local_sites = [up_left, up, up_right, left, middle,
                       right, down_left, down, down_right]

        # the sign is defined for hopping from column to row, and matrices are mat[row][column], so site goes second
        for site, signs in zip(local_sites, all_signs):
            neighbors = neighbor_list(Nx=Nx, Ny=Ny, N=site)
            for neighbor, sign in zip(neighbors, signs):
                hopping_phases[neighbor][site] = sign * 1j

    return (kij * hopping_phases), hopping_phases # element wise multiplication with phase

def peierls_triangular(
    Nx: int,
    Ny: int,
    t: float = 1.0,
    tp: float = 0.0,
    tpp: float = 0.0
) -> np.ndarray:
    
    peierls_mat = np.ones((Ny*Nx, Ny*Nx), dtype=np.complex128)

    # # define function for going x,y -> label
    # def x_y_to_N(x: int, y: int):
    #     return (x + y * Nx)

    # # as of now, my construction only works for an even Nx and Ny, otherwise
    # # the hopping scheme breaks. 
    # if (Nx % 2) != 0 or (Ny % 2) != 0:
    #     return NotImplementedError
    
    # # go from middle of big magnetic cell to middle big magnetic cell.
    # # first, make a list of all the centers.
    # central_sites_list = []
    # for y in range(Ny // 2):
    #     for x in range(Nx // 2):
    #         central_sites_list.append(x_y_to_N(2*x + 1, 2*y + 1))

    # # this isn't efficient, but it is very readable and the time scale
    # # of this is nothing compared to the program using it overall, so
    # # it is preferable to have readability > efficiency. Could use
    # # a hashmap or something similar to check what has already been added

    # # define the vector of signs for each spot when they count. This goes R, U, UL, L, D, DR on the square
    # # lattice version of our triangular lattice
    # # NOTE it *is* different from the ordering of the sites in local site

    # signs_up_left = [-1, 1, -1, -1, 1, -1]
    # signs_up = [1, -1, -1, 1, -1, -1]
    # signs_up_right = [-1, 1, -1, -1, 1, -1]
    # signs_left = [-1, -1, 1, -1, -1, 1]
    # signs_middle = [1, 1, 1, 1, 1, 1]
    # signs_right = [-1, -1, 1, -1, -1, 1]
    # signs_down_left = [-1, 1, -1, -1, 1, -1]
    # signs_down = [1, -1, -1, 1, -1, -1]
    # signs_down_right = [-1, 1, -1, -1, 1, -1]

    # all_signs = [signs_up_left, signs_up, signs_up_right, signs_left,
    #              signs_middle, signs_right, signs_down_left, signs_down,
    #              signs_down_right]

    # for central_site in central_sites_list:
    #     # decompose the middle site into its x and y
    #     x, y = (central_site % Nx), (central_site // Nx)

    #     # identify the positions of the region
    #     up_left = x_y_to_N((x-1+Nx) % Nx, (y+1) % Ny)
    #     up = x_y_to_N(x, (y+1) % Ny)
    #     up_right = x_y_to_N((x+1) % Nx, (y+1) % Ny)
    #     left = x_y_to_N((x-1+Nx) % Nx, y)
    #     middle = central_site
    #     right = x_y_to_N((x+1) % Nx, y)
    #     down_left = x_y_to_N((x-1+Nx) % Nx, (y-1+Ny) % Ny)
    #     down = x_y_to_N(x, (y-1+Ny) % Ny)
    #     down_right = x_y_to_N((x+1) % Nx, (y-1+Ny) % Ny)

    #     local_sites = [up_left, up, up_right, left, middle,
    #                    right, down_left, down, down_right]
    #     # print(f"middle: {middle}  right: {right}  up_right: {up_right}  up: {up}  up_left: {up_left}  left: {left}  down_left: {down_left}  down: {down}  down_right: {down_right}")

    #     # for each site, make the relevant hoppings

    #     # the sign is defined for hopping from column to row, and matrices are mat[row][column], so site goes second
    #     for site, signs in zip(local_sites, all_signs):
    #         neighbors = neighbor_list(Nx=Nx, Ny=Ny, N=site)
    #         for neighbor, sign in zip(neighbors, signs):
    #             peierls_mat[neighbor][site] = sign

    # # tack on the i's
    # return peierls_mat * 1j

    # previously was actualy making a peierls matrix only for NN,
    # now just returning a matrix of all 1s because of a measurement workaround
    return peierls_mat


# returns the neighbors and signs
def xperiodic_neighbor_hopping_list(Nx:int, Ny:int, N:int):
    def x_y_to_N(x: int, y: int):
        return (x + y * Nx)

    x, y = (N % Nx), (N // Nx)
    right = x_y_to_N((x+1) % Nx, y)
    up_right = None
    up_left = None
    if y != (Ny-1):
        up_right = x_y_to_N(x, y+1)
        up_left = x_y_to_N((x + Nx - 1) % Nx, y+1)
    left = x_y_to_N((x + Nx - 1) % Nx, y)
    down_left = None
    down_right = None
    if y != 0:
        down_left = x_y_to_N(x, y-1)
        down_right = x_y_to_N((x+1) % Nx, y-1)

    if y % 2 == 0: # if an A site:
        hoppings = [-1, 1, -1j, -1, 1, -1j]
    else:
        hoppings = [1, 1, 1j, 1, 1, 1j]

    neighbors = [right, up_right, up_left, left, down_left, down_right]
    return neighbors, hoppings

def yperiodic_neighbor_hopping_list(Nx:int, Ny:int, N:int):
    def x_y_to_N(x: int, y: int):
        return (x + y * Nx)
    x, y = (N % Nx), (N // Nx)
    down_right = None
    right = None
    if x != (Nx-1):
        down_right = x_y_to_N(x+1, (y + Ny - 1) % Ny)
        right = x_y_to_N(x+1, y)
    up_right = x_y_to_N(x, (y+1) % Ny)
    up_left = None
    left = None
    if x != 0:
        up_left = x_y_to_N(x-1, (y+1) % Ny)
        left = x_y_to_N(x-1, y)
    down_left = x_y_to_N(x, (y-1+Ny) % Ny)

    if x % 2 == 0:
        hoppings = [1, -1, 1j, 1, -1, 1j]
    else:
        hoppings = [1, 1, -1j, 1, 1, -1j]

    neighbors = [right, up_right, up_left, left, down_left, down_right]
    return neighbors, hoppings

def H_xperiodic_triangular(
    Nx: int,
    Ny: int,
    t: float = 1.0,
    tp: float = 0.0,
    tpp: float = 0.0,
    nflux: float = 0.0,
    alpha: float = 1 / 2
) -> tuple[np.ndarray, np.ndarray]:
    
    if tp != 0 or tpp != 0:
        raise NotImplementedError
    
    # this construction only works for an even Ny and any Nx
    if (Ny % 2) != 0:
        raise NotImplementedError  

    # begin by initializing the kinetic matrix
    kij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)

    for site in range(Nx*Ny):
        neighbors, hoppings = xperiodic_neighbor_hopping_list(Nx=Nx, Ny=Ny, N=site)

        for neighbor, hopping in zip(neighbors, hoppings):
            if neighbor == None:
                continue
            kij[neighbor][site] = hopping
    kij *= -t
    ones_matrix = np.ones((Ny*Nx, Ny*Nx), dtype=np.complex128)
    return kij, ones_matrix # element wise multiplication with phase

def H_yperiodic_triangular(
    Nx: int,
    Ny: int,
    t: float = 1.0,
    tp: float = 0.0,
    tpp: float = 0.0,
    nflux: float = 0.0,
    alpha: float = 1 / 2
) -> tuple[np.ndarray, np.ndarray]:
    
    if tp != 0 or tpp != 0:
        raise NotImplementedError
    
    # this construction only works for an even Nx and any Ny
    if (Nx % 2) != 0:
        raise NotImplementedError  

    # begin by initializing the kinetic matrix
    kij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)

    for site in range(Nx*Ny):
        neighbors, hoppings = yperiodic_neighbor_hopping_list(Nx=Nx, Ny=Ny, N=site)

        for neighbor, hopping in zip(neighbors, hoppings):
            if neighbor == None:
                continue
            kij[neighbor][site] = hopping
    kij *= -t
    ones_matrix = np.ones((Ny*Nx, Ny*Nx), dtype=np.complex128)
    return kij, ones_matrix # element wise multiplication with phase