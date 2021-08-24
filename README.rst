Denspart
########


Atoms-in-molecules density partitioning schemes based on stockholder recipe

This is prototype code. It works, but use with care.

To set up the development environment, do the following:

.. code-block:: bash

    # Activate conda, may be different on your machine
    source ~/miniconda3/bin/activate  
    # Install the CI driver
    conda install theochem::roberto
    # Clone git repo, assuming you have ssh access to github
    # If not, use git clone https://github.com/theochem/denspart.git instead
    git clone git@github.com:theochem/denspart.git  
    cd denspart
    git checkout prototype0
    # Run first part of the CI, includes making a new conda env with all dependencies.
    rob lint-static
    # Activates the development env
    source activate-denspart-dev-python-3.7.sh
    # Install development link
    python setup.py develop
    
Some day, this will become as easy as "pip install denspart", but we're not
there yet.

Usage:

.. code-block:: bash

    # Compute electronic density on a default grid (not optimal yet)
    denspart-from-horton3 wfn_file density.npz
    # Run the partitioning
    denspart density.npz results.npz
    # Convert results to human-readable format
    denspart-write-extxyz results.npz results.xyz
