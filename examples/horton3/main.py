from pathlib import Path

import numpy as np
from denspart.__main__ import main

np.random.seed(42)

main([str(Path("density.npz")), "output.npz", "-t", "GISA"])
# main([str(Path("density.npz")), 'output.npz', '-t', 'MBIS'])
data = np.load("output.npz")
print(data["charges"])
