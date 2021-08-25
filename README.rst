DensPart
########


Atoms-in-molecules density partitioning schemes based on stockholder recipe

This is a prototype code. It is incomplete, yet it works, but use with care.


Minimal setup
=============

Required dependencies:

- QC-Grid: https://github.com/theochem/grid
- NumPy: https://numpy.org

Install (with dependencies):

.. code-block:: bash

    pip install git+https://github.com/theochem/grid.git
    pip install git+https://github.com/theochem/denspart.git

(There are no releases yet.)

With this basic setup, one needs to construct a ``density.npz`` file, which is used as input
for the ``denspart`` script. (The optional dependencies below provide convenient tools
to make such files.)

The file ``density.npz`` is a NumPy ZIP file as created with the function ``numpy.savez``.
NPZ is a simple container format for multiple arrays and more details can be found here:
https://numpy.org/doc/stable/reference/routines.io.html

The file ``density.npz`` must contain the following arrays:

- ``points``: Array with quadrature grid points, shape ``(npoint, 3)``.
- ``weights``: Array with quadrature grid weights, shape ``(npoint,)``.
- ``density``: Array with electron density at the grid points, shape ``(npoint,)``.
- ``atnums``: Integer array with atomic numbers, shape ``(natom,)``.
- ``atcoords``: Array with nuclear coordinates, shape ``(natom, 3)``.

It may also contain the following:

- ``cellvecs``: One, two or three cell vectors (rows) defining periodic boundary
  conditions, shape ``(nvec, 3)``.

With a ``density.npz`` file, one can run the following:

.. code-block:: bash

    # Perform the partitioning
    denspart density.npz results.npz


Optional dependencies
=====================


IOData
------

See https://github.com/theochem/iodata

When IOData is installed, the npz output of the partitioning can be converted into an
extended XYZ file as follows:

.. code-block:: bash

    denspart-write-extxyz results.npz results.xyz


IOData and GBasis
-----------------

In order to derive a ``density.npz`` from several wavefunction file formats
(wfn, wfx, molden, fchk, ...), one needs install a few additional packages (whose API
changes regularly):

- https://github.com/theochem/iodata
- https://github.com/theochem/gbasis

Once these are installed, one can run:

.. code-block:: bash

    denspart-from-horton3 some-file.fchk density.npz

A minimal working example can be found in ``examples/horton3``.


GPAW
----

Another option is to derive a ``density.npz`` file from a GPAW calculation, for which
one needs to install GPAW:

- https://wiki.fysik.dtu.dk/gpaw/

Once these are installed, one can run:

.. code-block:: bash

    denspart-from-horton3 some-file.fchk density.npz

A minimal working example can be found in ``examples/gpaw``.


PLAMS
-----

One can create ``density.npz`` from an ADF AMSJob when PLAMS is installed, see

- https://github.com/SCM-NV/PLAMS

In this case, you can run:

.. code-block:: bash

    denspart-from-adf ams.results density.npz

where ``ams.results`` is the directory with output files. You need to disable symmetry
and write out the TAPE10 file. More details can be found the the denspart.adapters.adf
module. When you have ADF installed, you may need to use ``amspython``, which is a bit
awkward. A minimal working example can be found in ``examples/adf``.


Psi4 Interface
==============

By adding a few lines to the Psi4 input script, it will write an NPZ file with Psi4's
built-in molecular quadrature grids:

.. code-block:: python

    energy, wfn = psi4.energy(return_wfn=True)
    from denspart.adapters.psi4 import write_density_npz
    write_density_npz(wfn)

Symmetry is not supported, so you need to set the point group to ``c1`` when specifying
the geometry. A minimal working example can be found in ``examples/psi4``.


Development setup
=================

To set up the development environment, do the following:

.. code-block:: bash

    # Install the CI driver
    pip install theochem::roberto
    # Clone git repo, assuming you have ssh access to github
    # If not, use git clone https://github.com/theochem/denspart.git instead
    git clone git@github.com:theochem/denspart.git
    cd denspart
    git checkout prototype0
    # Run first part of the CI, includes making a new test env with all dependencies.
    rob lint-static
    # Activates the development env
    source activate-denspart-dev-python-3.7.sh
    # Fix missing dependency
    pip install git+https://github.com/theochem/grid.git
