"""Compute atom-in-molecules properties."""


import numpy as np


def compute_rcubed(pro_model, grid, rho):
    pro = pro_model.compute_density(grid)
    result = np.zeros(pro_model.natom)
    for iatom, atcoord in enumerate(pro_model.atcoords):
        localgrid = grid.get_localgrid(atcoord, 8.0)
        dists = np.linalg.norm(localgrid.points - atcoord, axis=1)
        pro_atom = pro_model.compute_proatom(iatom, localgrid)
        ratio = pro_atom / pro[localgrid.indices]
        result[iatom] = localgrid.integrate(rho[localgrid.indices], dists ** 3, ratio)
    return result
