import tight_binding as tb
import numpy as np
import sys
import pytest


class TestKagome:
    Nx = 4
    Ny = 4
    t = 1

    def test_0(self):
        tp = 0
        tpp = 0
        nflux = 0
        Hmat, Peierls = tb.H_periodic_kagome(self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2)

        assert Hmat.shape == (3*self.Nx*self.Ny,3*self.Nx*self.Ny)
        assert Hmat.dtype==np.complex128
        assert np.max(Peierls.imag) < 1e-10
        assert np.max(np.abs(Hmat.imag)) < 1e-10

    def test_tp(self):
        tp = 1
        tpp = 0
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_kagome(self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2)
        

    def test_tpp(self):
        tp = 1
        tpp = 0.3
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_kagome(self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2)


class TestOther:
    Nx = 10
    Ny = 10
    t = 1

    def test1(self):
        nflux = 1
        tp = 0
        Hmat, _ = tb.H_periodic_triangular(self.Nx, self.Ny, t=self.t, tp=tp, nflux=nflux, alpha=1 / 2)
        evals, V = np.linalg.eigh(Hmat)

    def test2(self):
        tp = 1
        tpp = 0
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_kagome(self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2)
        
        #evals, V = np.linalg.eigh(Hmat)

    def test3(self):
        Ncell = self.Nx * self.Ny
        tp = 0
        tpp = 0
        alpha = 1 / 2
        geometry = "triangular"
        nf_arr = np.arange(0, Ncell + 1)
        nF = nf_arr.shape[0]

        beta = 5
        mu = -3
        nflux = 2
        print(nflux / Ncell)
        Hmat, _ = tb.H_periodic_square(self.Nx, self.Ny, t=self.t, tp=0, nflux=nflux, alpha=1 / 2)
        evals, V = np.linalg.eigh(Hmat)

