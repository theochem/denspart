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
"""Unit tests for the module denspart.mbis."""


from pathlib import Path

import numpy as np
import pytest
from denspart.gisa import GaussianFunction, GISAProModel
from denspart.vh import optimize_reduce_pro_model
from grid.basegrid import Grid
from numpy.testing import assert_allclose


@pytest.mark.filterwarnings("ignore:delta_grad:UserWarning")
@pytest.mark.filterwarnings("ignore:exponent:UserWarning")
def test_example():
    data = np.load(Path("tests", "density-water.npz"))
    grid = Grid(data["points"], data["weights"])
    pro_model0 = GISAProModel.from_geometry(data["atnums"], data["atcoords"])
    pro_model1 = optimize_reduce_pro_model(pro_model0, grid, data["density"])[0]
    assert len(pro_model1.fns) == 14
    # data = pro_model1.to_dict()
    # pro_model2 = ProModel.from_dict(data)
    # assert isinstance(pro_model2, pro_model1.__class__)
    # for fn1, fn2 in zip(pro_model1.fns, pro_model2.fns, strict=True):
    #     assert isinstance(fn2, fn1.__class__)
    #     assert fn1.iatom == fn2.iatom
    #     assert_allclose(fn1.center, fn2.center)
    #     assert_allclose(fn1.pars, fn2.pars)


def test_reduce_pop():
    atnums = np.array([1, 1])
    atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pro_model0 = GISAProModel.from_geometry(atnums, atcoords)
    pro_model0.fns.append(GaussianFunction(1, atcoords[1], [1e-5], 2.0))
    pro_model1 = pro_model0.reduce()
    assert len(pro_model0.fns) == 9
    assert len(pro_model1.fns) == 8
    for i in range(8):
        assert pro_model1.fns[i].iatom == 0 if i < 4 else 1
        assert_allclose(pro_model1.fns[i].pars, pro_model0.fns[i].pars)


def test_reduce_exp():
    atnums = np.array([1, 1])
    atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pro_model0 = GISAProModel.from_geometry(atnums, atcoords)
    pro_model0.fns.append(
        GaussianFunction(
            1, atcoords[1], pro_model0.fns[-1].pars.copy(), pro_model0.fns[-1].exponent
        )
    )
    population = pro_model0.fns[-1].pars[0]
    pro_model0.fns[-1].exponent *= 1.000001
    pro_model1 = pro_model0.reduce()
    # the order of function will be changed after reduce.
    assert len(pro_model1.fns) == 8
    for i in range(8):
        assert pro_model1.fns[i].iatom == 0 if i < 4 else 1

        if np.isclose(pro_model1.fns[i].pars[0], population * 2):
            np.allclose(pro_model0.fns[i].exponent, pro_model0.fns[i].exponent * 1.0000005)
