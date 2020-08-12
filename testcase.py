#!/usr/bin/env python3

import numpy as np

from pickle import dump

from iodata import load_one
from denspart.adapters.horton3 import prepare_input
from denspart.mbis import partition
from denspart.properties import compute_rcubed


def main():
    # fn_wfn = "h2o_sto3g.fchk"
    fn_wfn = "orca.molden.input"
    iodata = load_one(fn_wfn)
    print("Computing density on the grid")
    grid, rho = prepare_input(iodata)
    print("Sanity checks")
    print("Integral of rho:", grid.integrate(rho))
    print("MBIS partitioning")
    pro_model = partition(iodata.atnums, iodata.atcoords, grid, rho)
    print("Properties")
    iodata.atffparams["charges"] = pro_model.charges
    iodata.atffparams["rcubed"] = compute_rcubed(pro_model, grid, rho)
    print("Charges:")
    print(pro_model.charges)
    print("Total charge:", pro_model.charges.sum())
    print("R^3 moments:")
    print(iodata.atffparams["rcubed"])
    with open(fn_wfn + ".pp", "wb") as f:
        dump(iodata, f)


if __name__ == "__main__":
    main()
