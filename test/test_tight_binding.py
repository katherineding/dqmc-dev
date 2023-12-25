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
        Hmat, Peierls = tb.H_periodic_kagome(
            self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

        assert Hmat.shape == (3 * self.Nx * self.Ny, 3 * self.Nx * self.Ny)
        assert Hmat.dtype == np.complex128
        assert np.max(Peierls.imag) < 1e-10
        assert np.max(np.abs(Hmat.imag)) < 1e-10

    def test_tp(self):
        tp = 1
        tpp = 0
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_kagome(
                self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )

    def test_tpp(self):
        tp = 1
        tpp = 0.3
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_kagome(
                self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )


class TestHoneycomb:
    Nx = 4
    Ny = 4
    t = 1

    def test_0(self):
        tp = 0
        tpp = 0
        nflux = 0
        Hmat, Peierls = tb.H_periodic_honeycomb(
            self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

        assert Hmat.shape == (2 * self.Nx * self.Ny, 2 * self.Nx * self.Ny)
        assert Hmat.dtype == np.complex128
        assert np.max(Peierls.imag) < 1e-10
        assert np.max(np.abs(Hmat.imag)) < 1e-10

    def test_tp(self):
        tp = 1
        tpp = 0
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_honeycomb(
                self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )

    def test_tpp(self):
        tp = 1
        tpp = 0.3
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_honeycomb(
                self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )


class TestSquare:
    Nx = 6
    Ny = 6
    t = 1

    def test_0(self):
        tp = 0
        tpp = 0
        nflux = 0
        Hmat, Peierls = tb.H_periodic_square(
            self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

        assert Hmat.shape == (1 * self.Nx * self.Ny, 1 * self.Nx * self.Ny)
        assert Hmat.dtype == np.complex128
        assert np.max(Peierls.imag) < 1e-10
        assert np.max(np.abs(Hmat.imag)) < 1e-10

    def test_tp(self):
        tp = 1
        tpp = 0
        nflux = 1
        Hmat, Peierls = tb.H_periodic_square(
            self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

    def test_tpp(self):
        tp = 1
        tpp = 0.3
        nflux = 1
        Hmat, Peierls = tb.H_periodic_square(
            self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

    def test_size(self):
        Nx = 4
        Ny = 4
        tp = 1
        tpp = 0.3
        nflux = 1
        with pytest.raises(Exception):
            # This lattice is too small to accomodate tpp
            Hmat, Peierls = tb.H_periodic_square(
                Nx, Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )


class TestTriangular:
    Nx = 6
    Ny = 6
    t = 1

    def test_0(self):
        tp = 0
        tpp = 0
        nflux = 0
        Hmat, Peierls = tb.H_periodic_triangular(
            self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
        )

        assert Hmat.shape == (1 * self.Nx * self.Ny, 1 * self.Nx * self.Ny)
        assert Hmat.dtype == np.complex128
        assert np.max(Peierls.imag) < 1e-10
        assert np.max(np.abs(Hmat.imag)) < 1e-10

    def test_tp(self):
        tp = 1
        tpp = 0
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_triangular(
                self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )

    def test_tpp(self):
        tp = 1
        tpp = 0.3
        nflux = 1
        with pytest.raises(Exception):
            Hmat, Peierls = tb.H_periodic_triangular(
                self.Nx, self.Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )

    def test_size(self):
        Nx = 4
        Ny = 4
        tp = 1
        tpp = 0.3
        nflux = 1
        with pytest.raises(Exception):
            # This lattice is too small to accomodate tpp
            Hmat, Peierls = tb.H_periodic_triangular(
                Nx, Ny, t=self.t, tp=tp, tpp=tpp, nflux=nflux, alpha=1 / 2
            )
