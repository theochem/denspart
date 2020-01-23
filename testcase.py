#!/usr/bin/env python3

import numpy as np

from pickle import dump

from iodata import load_one
from denspart.adapters.horton3 import prepare_input
from denspart.mbis import partition
from denspart.properties import compute_rcubed


np.seterr(invalid="raise", divide="raise", over="raise")


def main():
    iodata = load_one("h2o_sto3g.fchk")
    grid, rho = prepare_input(iodata)
    pro_model = partition(iodata.atnums, iodata.atcoords, grid, rho)
    iodata.atffparams["charges"] = pro_model.charges
    iodata.atffparams["rcubed"] = compute_rcubed(pro_model, grid, rho)
    print(pro_model.charges)
    print(iodata.atffparams["rcubed"])
    with open("h2o_sto3.pp", "wb") as f:
        dump(iodata, f)


if __name__ == "__main__":
    main()
