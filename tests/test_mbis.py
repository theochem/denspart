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
from denspart.mbis import ExponentialFunction, MBISProModel, connected_vertices
from denspart.vh import ProModel, optimize_reduce_pro_model
from grid.basegrid import Grid
from numpy.testing import assert_allclose


def test_connected_vertices_simple():
    assert connected_vertices([[0, 1], [1, 2]], [0, 1, 2, 3]) == set(
        [frozenset([0, 1, 2]), frozenset([3])]
    )
    assert connected_vertices([[0, 1], [2, 3]], [0, 1, 2, 3]) == set(
        [frozenset([0, 1]), frozenset([2, 3])]
    )


def test_connected_vertices_random():
    rng = np.random.default_rng()
    for _ in range(100):
        # Randomly generate pairs
        vertices = list(range(10))
        pairs = rng.integers(0, 10, (20, 2))
        clusters = connected_vertices(pairs, vertices)
        for pair in pairs:
            for cluster in clusters:
                assert not (pair[0] in cluster) ^ (pair[1] in cluster)
        for vertex in vertices:
            assert sum(vertex in cluster for cluster in clusters) == 1


@pytest.mark.filterwarnings("ignore:delta_grad:UserWarning")
def test_example():
    data = np.load(Path("tests", "density-water.npz"))
    grid = Grid(data["points"], data["weights"])
    pro_model0 = MBISProModel.from_geometry(data["atnums"], data["atcoords"])
    pro_model1 = optimize_reduce_pro_model(pro_model0, grid, data["density"])[0]
    assert len(pro_model1.fns) == 4
    data = pro_model1.to_dict()
    pro_model2 = ProModel.from_dict(data)
    assert isinstance(pro_model2, pro_model1.__class__)
    for fn1, fn2 in zip(pro_model1.fns, pro_model2.fns, strict=True):
        assert isinstance(fn2, fn1.__class__)
        assert fn1.iatom == fn2.iatom
        assert_allclose(fn1.center, fn2.center)
        assert_allclose(fn1.pars, fn2.pars)


def test_reduce_pop():
    atnums = np.array([1, 1])
    atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pro_model0 = MBISProModel.from_geometry(atnums, atcoords)
    pro_model0.fns.append(ExponentialFunction(1, atcoords[1], [1e-5, 1.0]))
    pro_model1 = pro_model0.reduce()
    assert len(pro_model1.fns) == 2
    for i in 0, 1:
        assert pro_model1.fns[i].iatom == i
        assert_allclose(pro_model1.fns[i].pars, pro_model0.fns[i].pars)


def test_reduce_exp():
    atnums = np.array([1, 1])
    atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pro_model0 = MBISProModel.from_geometry(atnums, atcoords)
    pro_model0.fns.append(ExponentialFunction(1, atcoords[1], pro_model0.fns[-1].pars.copy()))
    pro_model0.fns[-1].pars[1] *= 1.000001
    pro_model1 = pro_model0.reduce()
    assert len(pro_model1.fns) == 2
    assert pro_model1.fns[0].iatom == 0
    assert_allclose(pro_model1.fns[0].pars, pro_model0.fns[0].pars)
    assert pro_model1.fns[1].iatom == 1
    assert_allclose(pro_model1.fns[1].pars[0], 2 * pro_model0.fns[1].pars[0])
    assert_allclose(pro_model1.fns[1].pars[1], pro_model0.fns[1].pars[1] * 1.0000005)
