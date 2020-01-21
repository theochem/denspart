#!/usr/bin/env python3

import numpy as np

from pickle import dump

from iodata import load_one
from denspart.api import partition


np.seterr(invalid="raise", divide="raise", over="raise")


def main():
    iodata = load_one("h2o_sto3g.fchk")
    partition(iodata, "MBIS")
    with open("h2o_sto3.pp", "wb") as f:
        dump(iodata, f)


if __name__ == "__main__":
    main()
