#!/usr/bin/env python3

import numpy as np
import h5py as h5

from grid.periodicgrid import PeriodicGrid

from denspart.mbis import partition
from denspart.properties import compute_rcubed


np.seterr(invalid="raise", divide="raise", over="raise")


def main():
    print("Loading density and grid from HDF5 file")
    with h5.File("quartz.h5") as f:
        atnums = f["denspart/numbers"][:]
        atcoords = f["denspart/coordinates"][:]
        points = f["denspart/grid/points"][:]
        weights = f["denspart/grid/weights"][:]
        rho = f["denspart/grid/rho"][:]
        realvecs = f["/denspart/cell_vecs"][:]
    grid = PeriodicGrid(points, weights, realvecs)
    print("Sanity checks")
    print(grid.integrate(rho))
    print("MBIS partitioning")
    pro_model = partition(atnums, atcoords, grid, rho)
    print("Charges:")
    print(pro_model.charges)
    print("Total charge:", pro_model.charges.sum())
    print("R^3 moments:")
    print(compute_rcubed(pro_model, grid, rho))


if __name__ == "__main__":
    main()
