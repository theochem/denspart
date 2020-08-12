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
"""Collection of utilities to prepare densities on grids for denspart.

The modules in this subpackage are possibly ugly and may require dependencies
beyond the ones mentioned in setup.py. Other modules in denspart should never
directly import from these modules. They are solely provided to help end users
with preparing and importing densities on grids from various sources.
"""
