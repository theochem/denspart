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
"""Test the input preparation with GAPW."""


import os
import shutil
from importlib import resources

import numpy as np
from ase import Atoms
from gpaw import GPAW, setup_paths
from numpy.testing import assert_allclose

from ..gpaw import main


def install_setups(tmpdir, names):
    """Install setups into a temporary directory."""
    for name in names:
        fn_setup2 = os.path.join(tmpdir, name)
        with resources.path("denspart.adapters.test", name) as fn_setup:
            shutil.copyfile(fn_setup, fn_setup2)
    setup_paths.insert(0, tmpdir)


def test_h2(tmpdir):
    install_setups(tmpdir, ["H.LDA.gz"])

    # Run H2 calculation
    system = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
    )
    system.set_cell((6.0, 6.0, 6.0))
    system.center()
    fn_txt = os.path.join(tmpdir, "h2.txt")
    system.calc = GPAW(txt=fn_txt)
    fn_gpw = os.path.join(tmpdir, "h2.gpw")
    system.get_potential_energy()
    system.calc.write(fn_gpw)

    # Convert output
    fn_density = os.path.join(tmpdir, "density.npz")
    main([fn_gpw, fn_density])
    data = np.load(fn_density)
    assert_allclose(np.dot(data["density"], data["weights"]), 2.0)


def test_mgo(tmpdir):
    install_setups(tmpdir, ["O.LDA.gz", "Mg.LDA.gz"])

    # Run H2 calculation
    system = Atoms(
        "Mg4O4",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        cell=np.identity(3) * 4.212,
        pbc=(1, 1, 1),
    )
    fn_txt = os.path.join(tmpdir, "mgo.txt")
    system.calc = GPAW(txt=fn_txt)
    fn_gpw = os.path.join(tmpdir, "mgo.gpw")
    system.get_potential_energy()
    system.calc.write(fn_gpw)

    # Convert output
    fn_density = os.path.join(tmpdir, "density.npz")
    main([fn_gpw, fn_density])
    data = np.load(fn_density)
    assert_allclose(np.dot(data["density"], data["weights"]), 80.0)
