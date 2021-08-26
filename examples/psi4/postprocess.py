#!/usr/bin/env python

import numpy as np
import json

results = np.load("results.npz")
print(results["charges"])

np.savetxt("charges.csv", results["charges"], delimiter=",")
json.dump(results["charges"].tolist(), open("charges.json", "w"))

print(np.dot(results["atcoords"].T, results["charges"]))
print(results["multipole_moments"][:, [1, 2, 0]].sum(axis=0))

