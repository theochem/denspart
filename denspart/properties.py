"""Compute atom-in-molecules properties."""


import numpy as np


def compute_rcubed(pro_model, grid, rho):
    pro = pro_model.compute_density(grid)
    result = np.zeros(pro_model.natom)
    for iatom, atcoord in enumerate(pro_model.atcoords):
        subgrid = grid.get_subgrid(atcoord, 8.0)
        dists = np.linalg.norm(subgrid.points - atcoord, axis=1)
        pro_atom = pro_model.compute_proatom(iatom, subgrid)
        ratio = pro_atom / pro[subgrid.indices]
        result[iatom] = subgrid.integrate(rho[subgrid.indices], dists ** 3, ratio)
    return result
