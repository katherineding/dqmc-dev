import h5py
import numpy as np
import sys
sys.path.insert(0, "../dqmc-dev/util/")
import util
import data_analysis

Nx = 8
Ny = 8
N = Nx * Ny
bps = 3
num_ij = N * N
num_b = N * bps
num_bb = num_b * num_b

path = './bond_bond_testing/U0/'
n_sample, sign, pair_bb = util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/pair_bb")
if n_sample.max() == 0:
    print("no data")
mask = (n_sample == n_sample.max())
sign, pair_bb = sign[mask], pair_bb[mask] 
pair_bb_dataset, pair_bb_uncertainties = data_analysis.jackknife(pair_bb, sign)

print(f"Shape of pair_bb: {pair_bb_dataset.shape}")
print(f"Dtype of pair_bb: {pair_bb_dataset.dtype}")

i = 45
i_btype = 2

with open('45_2_to_all_type_bonds_mag_jk.txt', 'w') as file:
    sites = np.zeros((N,bps), dtype=np.complex128)
    magnitudes = np.zeros((N,bps), dtype=float)
    for j_btype in range(bps):
        for tau in range(40):
            for j in range(N):
                ix = i % Nx
                iy = i // Nx
                jx = j % Nx
                jy = j // Nx
                k = (ix + Nx * iy) + N * (jx + Nx * jy) 
                kk = k + num_ij * (i_btype + bps * j_btype)
                sites[j, j_btype] += pair_bb_dataset[kk + num_bb * tau]
    
    for j in range(N):
        for bond_type in range(bps):
            magnitudes[j, bond_type] = np.abs(sites[j,bond_type])

    for j in range(N):
        for bond_type in range(bps):
            file.write(f"{j} {bond_type} {magnitudes[j, bond_type]}\n")





