#!/usr/bin/env python3
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


import psi4
from denspart.adapters.psi4 import write_density_npz

psi4.core.set_output_file('output.txt', False)

with open("water.xyz") as f:
    mol = psi4.core.Molecule.from_string(f.read(), dtype="xyz")
mol.set_molecular_charge(0)
mol.set_multiplicity(1)
mol.reset_point_group("c1")  # This is required!

energy, wfn = psi4.optimize("b3lyp/6-31g", molecule=mol, return_wfn=True)
write_density_npz(wfn)
