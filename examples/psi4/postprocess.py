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
"""Denspart post-processing example."""

import json
import numpy as np

results = np.load("results.npz")
print(results["charges"])

np.savetxt("charges.csv", results["charges"], delimiter=",")
with open("charges.json", "w") as f:
    json.dump(results["charges"].tolist(), f)

print(np.dot(results["atcoords"].T, results["charges"]))
print(results["multipole_moments"][:, [1, 2, 0]].sum(axis=0))
