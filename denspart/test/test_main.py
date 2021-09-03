# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
#
# DensPart is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# DensPart is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Unit tests for the module denspart.__main__."""


from importlib import resources
import os
import tempfile

import numpy as np

from ..vh import ProModel
from ..__main__ import main


def test_cli(ndarrays_regression):
    with resources.path("denspart.test", "density-water.npz") as fn_npz:
        with tempfile.TemporaryDirectory("denspart", "test_cli") as dn:
            fn_results = os.path.join(dn, "results.npz")
            main([str(fn_npz), fn_results])
            assert os.path.isfile(fn_results)
            results = np.load(fn_results)
            pro_model = ProModel.from_dict(results)
    assert len(pro_model.fns) == 4
    ndarrays_regression.check(dict(results))
