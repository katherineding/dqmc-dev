# Takes in a file with different parameters you specify and spits out the fermion sign for each
import h5py
import numpy as np
import sys
sys.path.insert(0, "../dqmc-dev/util/")
import util
import data_analysis

target_file = 'params.txt'

Nx = 6
Ny = 6
N = Nx * Ny
bps = 3
num_ij = N * N
num_b = N * bps
num_bb = num_b * num_b

manual_list = True
U_list = []
mu_list = []
L_list = []

if manual_list:
    U_list = [5, 10, 15]
    mu_list = [-0.4855, -1.3767, -3.5585]
    L_list = [20, 20, 20]
else: 
    with open(target_file, 'r') as f:
        for line in f:
            if line.strip():  # skip empty lines
                U_str, mu_str, L_str = line.strip().split()
                U_list.append(U_str)   
                mu_list.append(mu_str)
                L_list.append(L_str)

for U, mu, L in zip(U_list, mu_list, L_list):
    path = f'/pscratch/sd/w/wyndham/dqmc-data/triangleHH/n0.95/U{U}/mu{mu}/L{L}'
    n_sample, sign = util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/sign")
    if n_sample.max() == 0:
        print("no data")
    mask = (n_sample == n_sample.max())
    sign = sign[mask]
    sign_dataset, sign_uncertainty = data_analysis.jackknife(n_sample[mask], sign)
    print(f'{U} {mu} {L} {sign_dataset} {sign_uncertainty}')


        
                                                                

