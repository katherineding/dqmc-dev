import gen_1band_unified_hub as ghub
import util
import numpy as np
import os
import pytest


geometry_list = ["square", "triangular", 'honeycomb', 'kagome']


@pytest.mark.parametrize("geometry", geometry_list)
def test_1_default(geometry):
    ghub.create_1(geometry=geometry)

    assert os.path.isfile('sim.h5') 
    Nx,Ny = util.load_file('sim.h5','metadata/Nx','metadata/Ny')

    assert Nx == 4
    assert Nx.dtype.type == np.int64
    assert Nx.shape == ()
    assert np.ndim(Nx) == 0
    assert Ny == Nx == 4
    os.remove("sim.h5")


@pytest.mark.parametrize("geometry", geometry_list)
def test_batch_default(geometry):
    ghub.create_batch(geometry=geometry)
    assert os.path.isfile('sim_0.h5') 
    assert os.path.isfile("sim.h5.params") 
    Nx,Ny = util.load_file('sim_0.h5','metadata/Nx','metadata/Ny')

    assert Nx == 4
    assert Nx.dtype.type == np.int64
    assert Nx.shape == ()
    assert np.ndim(Nx) == 0
    assert Ny == Nx == 4
    os.remove("sim_0.h5") 
    os.remove("sim.h5.params") 

