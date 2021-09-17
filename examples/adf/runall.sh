#!/usr/bin/env bash
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

# Tested with AMS 2021.202
# You may already have the following in your .bashrc. If not, uncomment.
# source ${ADFHOME}/amsbashrc.sh
# Avoid setting ADF and AMS environment variables manually, because these may change
# with different versions of AMS.

# Let AMS know where the license and the scratch space are. Uncomment and modify
# to your situation.
# export SCMLICENSE=your_license.txt
# export SCM_TMPDIR="$TMPDIR"

# Installation is commented out.
# amspython -m pip install git+https://github.com/theochem/grid.git
# amspython -m pip install git+https://github.com/theochem/iodata.git
# amspython -m pip install git+https://github.com/theochem/denspart.git

rm adf-water.out
rm -r ams.results
ams -n 1 < adf-water.in > adf-water.out
amspython -m denspart.adapters.adf ams.results density.npz
amspython -m denspart density.npz results.npz
amspython -m denspart.utils.write_extxyz results.npz results.xyz
