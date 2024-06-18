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
"""Computation cache with selective discard of obsolete intermediates."""

from collections.abc import Callable

__all__ = ("ComputeCache",)


class ComputeCache:
    def __init__(self):
        self._cache = {}

    def keep(self, stage, key, result):
        self._cache.setdefault(stage, {})[key] = result

    def fetch(self, stage, key):
        return self._cache.get(stage, {}).get(key)

    def discard(self, stage):
        self._cache.pop(stage)


def compute_cached(cache: ComputeCache, until: str, key: tuple, func: Callable):
    """Lookup a result or compute it.

    Parameters
    ----------
    cache
        An instance of ComputeCache or None.
    until
        The stage of the calculation where the cached result becomes invalid
        or should be discarded to save memory.
    key
        A dictionary key for the result.
        Note that the result is uniquely identified by the tuple (until, key).
    func
        A callable taking no arguments to compute the result when it
        is not available in the cache.

    Returns
    -------
    result
        The cached or computed result.
    """
    if cache is None:
        return func()
    result = cache.fetch(until, key)
    if result is None:
        result = func()
        cache.keep(until, key, result)
    return result
